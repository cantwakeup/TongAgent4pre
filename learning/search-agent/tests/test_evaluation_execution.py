"""Regression tests for crash-safe, process-isolated evaluation orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation.aggregate import aggregate_experiment
from evaluation.execution import (
    EvaluationStateError,
    FairnessMismatchError,
    WorkerJob,
    _experiment_lock,
    atomic_write_json,
    load_jsonl_dataset,
    persist_terminal_result,
    resolve_system_config,
    run_dataset,
    sanitized_subprocess_env,
    validate_persisted_result,
)
from evaluation.schema import (
    CompletionStatus,
    EvalTask,
    FailureDetail,
    FailureType,
    RunResult,
    json_ready,
)
from evaluation.worker import run_worker


def _write_dataset(path: Path, task_ids: tuple[str, ...] = ("task-a",)) -> Path:
    records = [
        {
            "id": task_id,
            "question": f"Question for {task_id}?",
            "reference_answer": "fixture answer",
            "metadata": {"fixture": True},
        }
        for task_id in task_ids
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _write_fake_worker(
    directory: Path,
    *,
    sleep: bool = False,
    partial_telemetry: bool = False,
    exit_code: int | None = None,
    corrupt_result: bool = False,
) -> str:
    if sleep:
        module_name = "sleep_worker"
        if partial_telemetry:
            body = """
import argparse
import json
import pathlib
import time

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
args = parser.parse_args()
attempt = pathlib.Path(args.job).parent
native = attempt / "native"
native.mkdir(exist_ok=True)
(native / "partial_telemetry.json").write_text(json.dumps({
    "schema_version": 1,
    "request_start": "2026-07-23T00:00:00+00:00",
    "first_model_response": "2026-07-23T00:00:01+00:00",
    "first_tool_call": "2026-07-23T00:00:02+00:00",
    "last_progress_timestamp": "2026-07-23T00:00:03+00:00",
    "last_event_type": "tool_call_started",
    "started_tool_counts": {"web_search": 2, "fetch_url": 1},
    "completed_tool_counts": {"web_search": 1},
    "budget_snapshot": {
        "search_calls": 2,
        "fetch_calls": 1,
        "external_retrieval_calls": 3,
        "internal_tool_calls": 0
    },
    "token_usage_status": "usage_unavailable",
    "token_usage": None
}))
time.sleep(30)
"""
        else:
            body = "import time\ntime.sleep(30)\n"
    elif exit_code is not None:
        module_name = "crash_worker"
        body = f"raise SystemExit({exit_code})\n"
    elif corrupt_result:
        module_name = "corrupt_worker"
        body = """
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
args = parser.parse_args()
(pathlib.Path(args.job).parent / "result.json").write_text("{corrupt")
"""
    else:
        module_name = "fake_worker"
        body = """
import argparse
import datetime
import json
import os
import pathlib
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
args = parser.parse_args()
job_path = pathlib.Path(args.job)
attempt = job_path.parent
job = json.loads(job_path.read_text())
task = json.loads((attempt / job["task_file"]).read_text())
config = json.loads((attempt / job["config_file"]).read_text())
(attempt / "native").mkdir(exist_ok=True)
(attempt / "native" / "pid.txt").write_text(str(os.getpid()))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = {
    "run_id": job["run_id"],
    "task_id": task["id"],
    "system_id": config["system_id"],
    "git_sha": job["git_sha"],
    "resolved_config": config,
    "config_fingerprint": config["config_fingerprint"],
    "fairness_fingerprint": config["fairness_fingerprint"],
    "started_at": now,
    "finished_at": now,
    "wall_time_seconds": 0.001,
    "final_answer": "fixture answer",
    "citations": [],
    "tool_calls": [],
    "search_calls": 0,
    "fetch_calls": 0,
    "relevant_searches": 0,
    "evidence_count": None,
    "structural_subquestion_coverage": None,
    "token_usage": None,
    "estimated_cost": None,
    "completion_status": "completed",
    "failure_type": None,
    "failure": None,
    "artifact_directory": str(attempt.resolve()),
    "fixture_smoke": True,
    "normalized_exact_match": True,
    "judge_score": None,
}
(attempt / "answer.md").write_text("fixture answer\\n")
(attempt / "trace.json").write_text(json.dumps({
    "schema_version": 1,
    "run_id": result["run_id"],
    "task_id": result["task_id"],
    "system_id": result["system_id"],
    "trace_scope": "canonical_tool_calls",
    "native_trace_jsonl": None,
    "tool_calls": [],
}))
(attempt / "metrics.json").write_text(json.dumps({
    "schema_version": 1,
    "run_id": result["run_id"],
    "completion_status": "completed",
    "runtime_mode": config.get("runtime_mode", "strict"),
    "wall_time_seconds": result["wall_time_seconds"],
    "search_calls": 0,
    "fetch_calls": 0,
    "relevant_searches": 0,
    "evidence_count": None,
    "structural_subquestion_coverage": None,
    "token_usage": None,
    "estimated_cost": None,
    "normalized_exact_match": True,
    "judge_score": None,
    "judge_result": None,
    "workflow_metrics": None,
}))
(attempt / "failure.json").write_text(json.dumps({
    "failure_type": None,
    "failure": None,
}))
fd, temporary = tempfile.mkstemp(dir=attempt, prefix=".result.", suffix=".tmp")
with os.fdopen(fd, "w") as handle:
    json.dump(result, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, attempt / "result.json")
"""
    (directory / f"{module_name}.py").write_text(body, encoding="utf-8")
    return module_name


def _make_result(
    *,
    task: EvalTask,
    config,
    answer: str | None = "fixture answer",
    completion_status: CompletionStatus = CompletionStatus.COMPLETED,
    failure_type: FailureType | None = None,
    evidence_count: int | None = None,
    coverage: float | None = None,
    exact_match: bool | None = True,
    wall_time: float = 0.25,
    git_sha: str = "deadbeef",
    search_calls: int | None = 1,
    fetch_calls: int | None = 1,
    relevant_searches: int | None = 1,
) -> RunResult:
    started = datetime(2026, 7, 19, tzinfo=UTC)
    failure = (
        FailureDetail(
            failure_type=failure_type,
            message="fixture failure",
            stage="test",
            retryable=False,
            details={},
        )
        if failure_type is not None
        else None
    )
    return RunResult(
        run_id=f"run-{task.id}-{config.system_id}",
        task_id=task.id,
        system_id=config.system_id,
        git_sha=git_sha,
        resolved_config=config,
        config_fingerprint=config.config_fingerprint,
        fairness_fingerprint=config.fairness_fingerprint,
        started_at=started,
        finished_at=started + timedelta(seconds=wall_time),
        wall_time_seconds=wall_time,
        final_answer=answer,
        citations=[],
        tool_calls=[],
        search_calls=search_calls,
        fetch_calls=fetch_calls,
        relevant_searches=relevant_searches,
        evidence_count=evidence_count,
        structural_subquestion_coverage=coverage,
        token_usage=None,
        estimated_cost=None,
        completion_status=completion_status,
        failure_type=failure_type,
        failure=failure,
        artifact_directory=config.artifact_directory,
        fixture_smoke=True,
        normalized_exact_match=exact_match,
        judge_score=None,
    )


def test_jsonl_digest_seed_limit_and_duplicate_rejection(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path / "tasks.jsonl",
        ("task-a", "task-b", "task-c", "task-d"),
    )
    raw = dataset.read_bytes()
    first = load_jsonl_dataset(dataset, seed=17, limit=2)
    again = load_jsonl_dataset(dataset, seed=17, limit=2)
    expected_ids = ["task-a", "task-b", "task-c", "task-d"]
    random.Random(17).shuffle(expected_ids)

    assert first.digest == f"sha256:{hashlib.sha256(raw).hexdigest()}"
    assert [task.id for task in first.selected_tasks] == expected_ids[:2]
    assert first.selected_tasks == again.selected_tasks
    assert first.total_tasks == 4

    duplicate = tmp_path / "duplicates.jsonl"
    _write_dataset(duplicate, ("same", "same"))
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl_dataset(duplicate)


def test_subprocess_attempt_resume_rerun_and_incomplete_preservation(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    worker_directory = tmp_path / "worker"
    worker_directory.mkdir()
    worker_module = _write_fake_worker(worker_directory)
    output = tmp_path / "output"

    first = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=output,
        experiment_id="resume-demo",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        git_sha="deadbeef",
    )
    attempt_1 = output / "resume-demo" / "simple_react" / "task-a" / "attempt-0001"
    assert first.executed == 1
    assert (attempt_1 / "result.json").is_file()
    assert {
        "job.json",
        "task.json",
        "config.json",
        "result.json",
        "worker.stdout.log",
        "worker.stderr.log",
    }.issubset({item.name for item in attempt_1.iterdir()})

    resumed = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=output,
        experiment_id="resume-demo",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        git_sha="deadbeef",
    )
    assert resumed.skipped == 1
    assert not (attempt_1.parent / "attempt-0002").exists()

    rerun = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=output,
        experiment_id="resume-demo",
        rerun=True,
        worker_module=worker_module,
        worker_cwd=worker_directory,
        git_sha="deadbeef",
    )
    assert rerun.executed == 1
    assert (attempt_1.parent / "attempt-0002" / "result.json").is_file()

    # A newer interrupted rerun is not hidden by an older successful marker.
    (attempt_1.parent / "attempt-0003").mkdir()
    resumed_after_interruption = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=output,
        experiment_id="resume-demo",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        git_sha="deadbeef",
    )
    assert resumed_after_interruption.executed == 1
    assert (attempt_1.parent / "attempt-0003").is_dir()
    assert (attempt_1.parent / "attempt-0004" / "result.json").is_file()

    incomplete_root = output / "incomplete-demo" / "simple_react" / "task-a"
    (incomplete_root / "attempt-0001").mkdir(parents=True)
    after_incomplete = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=output,
        experiment_id="incomplete-demo",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        git_sha="deadbeef",
    )
    assert after_incomplete.executed == 1
    assert (incomplete_root / "attempt-0001").is_dir()
    assert (incomplete_root / "attempt-0002" / "result.json").is_file()


def test_resume_reexecutes_when_git_sha_changes(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    worker_directory = tmp_path / "worker"
    worker_directory.mkdir()
    worker_module = _write_fake_worker(worker_directory)
    output = tmp_path / "output"
    common = {
        "systems": ["simple_react"],
        "output_directory": output,
        "experiment_id": "git-aware-resume",
        "worker_module": worker_module,
        "worker_cwd": worker_directory,
    }

    first = run_dataset(dataset, git_sha="deadbeef", **common)
    same = run_dataset(dataset, git_sha="deadbeef", **common)
    changed = run_dataset(dataset, git_sha="cafebabe", **common)

    assert first.executed == 1
    assert same.skipped == 1
    assert changed.executed == 1
    second_attempt = changed.outcomes[0].attempt_directory
    assert second_attempt.name == "attempt-0002"
    assert changed.outcomes[0].result is not None
    assert changed.outcomes[0].result.git_sha == "cafebabe"


def test_result_marker_requires_untampered_companion_artifacts(tmp_path: Path) -> None:
    task = EvalTask(id="atomic-result", question="Question?")
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        attempt,
    )
    persist_terminal_result(
        attempt,
        _make_result(task=task, config=config),
    )
    validate_persisted_result(attempt / "result.json", task=task)

    (attempt / "metrics.json").unlink()
    with pytest.raises(EvaluationStateError, match="missing regular metrics.json"):
        validate_persisted_result(attempt / "result.json", task=task)


def test_resume_fails_closed_on_corrupt_or_mismatched_result(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    loaded = load_jsonl_dataset(dataset)
    task = loaded.selected_tasks[0]
    output = tmp_path / "output"

    corrupt_attempt = output / "corrupt" / "simple_react" / task.id / "attempt-0001"
    corrupt_attempt.mkdir(parents=True)
    (corrupt_attempt / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(EvaluationStateError, match="invalid result"):
        run_dataset(
            dataset,
            systems=["simple_react"],
            output_directory=output,
            experiment_id="corrupt",
            dry_run=True,
            git_sha="deadbeef",
        )
    assert not (corrupt_attempt.parent / "attempt-0002").exists()

    mismatch_attempt = output / "mismatch" / "simple_react" / task.id / "attempt-0001"
    mismatch_attempt.mkdir(parents=True)
    other_config = resolve_system_config(
        "simple_react",
        "sha256:different-dataset",
        0,
        mismatch_attempt,
    )
    persist_terminal_result(
        mismatch_attempt,
        _make_result(task=task, config=other_config),
    )
    with pytest.raises(EvaluationStateError, match="fingerprint mismatch"):
        run_dataset(
            dataset,
            systems=["simple_react"],
            output_directory=output,
            experiment_id="mismatch",
            dry_run=True,
            git_sha="deadbeef",
        )


def test_new_worker_corrupt_result_fails_closed_without_replacement(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    worker_directory = tmp_path / "worker"
    worker_directory.mkdir()
    worker_module = _write_fake_worker(worker_directory, corrupt_result=True)
    output = tmp_path / "output"

    with pytest.raises(EvaluationStateError, match="invalid result marker"):
        run_dataset(
            dataset,
            systems=["simple_react"],
            output_directory=output,
            experiment_id="corrupt-worker",
            worker_module=worker_module,
            worker_cwd=worker_directory,
            git_sha="deadbeef",
        )
    result_path = (
        output
        / "corrupt-worker"
        / "simple_react"
        / "task-a"
        / "attempt-0001"
        / "result.json"
    )
    assert result_path.read_text(encoding="utf-8") == "{corrupt"
    assert not (result_path.parent / "failure.json").exists()


def test_subprocess_timeout_writes_structured_terminal_failure(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    worker_directory = tmp_path / "worker"
    worker_directory.mkdir()
    worker_module = _write_fake_worker(worker_directory, sleep=True)
    report = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=tmp_path / "output",
        experiment_id="timeout",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        subprocess_timeout_seconds=0.05,
        git_sha="deadbeef",
    )

    result = report.outcomes[0].result
    assert result is not None
    assert result.completion_status == CompletionStatus.TIMED_OUT
    assert result.failure_type == FailureType.DEADLINE_EXCEEDED
    assert result.final_answer == "FINAL_ANSWER: ABSTAIN"
    assert result.answer_status.value == "abstain"
    assert result.extracted_answer is None
    assert result.search_calls is None
    assert result.fetch_calls is None
    assert result.relevant_searches is None
    attempt = report.outcomes[0].attempt_directory
    assert (attempt / "failure.json").is_file()
    assert (attempt / "metrics.json").is_file()
    assert (attempt / "trace.json").is_file()
    assert (attempt / "result.json").is_file()


def test_subprocess_timeout_recovers_atomic_partial_telemetry_before_kill(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    worker_directory = tmp_path / "worker"
    worker_directory.mkdir()
    worker_module = _write_fake_worker(
        worker_directory,
        sleep=True,
        partial_telemetry=True,
    )
    report = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=tmp_path / "output",
        experiment_id="timeout-with-telemetry",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        subprocess_timeout_seconds=0.2,
        git_sha="deadbeef",
    )

    result = report.outcomes[0].result
    assert result is not None
    assert result.completion_status == CompletionStatus.TIMED_OUT
    assert result.search_calls == 2
    assert result.fetch_calls == 1
    assert result.external_retrieval_calls == 3
    assert result.internal_tool_calls == 0
    assert result.token_usage is None
    assert result.failure is not None
    assert result.failure.details["token_usage_status"] == "usage_unavailable"
    attempt = report.outcomes[0].attempt_directory
    watchdog = json.loads(
        (attempt / "native" / "watchdog_telemetry.json").read_text(encoding="utf-8")
    )
    assert watchdog["partial_telemetry_available"] is True
    assert watchdog["partial_telemetry"]["started_tool_counts"] == {
        "fetch_url": 1,
        "web_search": 2,
    }


def test_parent_converts_worker_crash_before_result_to_terminal_failure(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    worker_directory = tmp_path / "worker"
    worker_directory.mkdir()
    worker_module = _write_fake_worker(worker_directory, exit_code=7)
    report = run_dataset(
        dataset,
        systems=["simple_react"],
        output_directory=tmp_path / "output",
        experiment_id="worker-crash",
        worker_module=worker_module,
        worker_cwd=worker_directory,
        git_sha="deadbeef",
    )

    result = report.outcomes[0].result
    assert result is not None
    assert result.completion_status == CompletionStatus.FAILED
    assert result.failure_type == FailureType.RUNNER_ERROR
    assert result.failure is not None
    assert result.failure.details == {"return_code": 7}
    assert result.final_answer == "FINAL_ANSWER: ABSTAIN"
    assert result.answer_status.value == "abstain"
    assert result.extracted_answer is None
    assert result.search_calls is None
    assert result.fetch_calls is None
    assert result.relevant_searches is None
    attempt = report.outcomes[0].attempt_directory
    persisted = RunResult.model_validate_json(
        (attempt / "result.json").read_text(encoding="utf-8")
    )
    assert persisted == result
    trace = json.loads((attempt / "trace.json").read_text(encoding="utf-8"))
    assert trace["trace_scope"] == "canonical_tool_calls"
    assert trace["native_trace_jsonl"] is None
    assert trace["tool_calls"] == []


def test_worker_owns_exact_match_and_publishes_result_last(tmp_path: Path) -> None:
    task = EvalTask(
        id="exact-task",
        question="What is the answer?",
        reference_answer="Paris",
        metadata={},
    )
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        attempt,
    )
    job = WorkerJob(run_id="run-canonical", git_sha="deadbeef")
    atomic_write_json(attempt / "task.json", json_ready(task))
    atomic_write_json(
        attempt / "config.json",
        config.model_dump(mode="json", exclude_none=False),
    )
    atomic_write_json(
        attempt / "job.json",
        job.model_dump(mode="json", exclude_none=False),
    )
    runner_result = _make_result(
        task=task,
        config=config,
        answer="The answer is Paris.",
        exact_match=True,
    )

    class Runner:
        def run(self, received_task, received_config):
            assert received_task == task
            assert received_config == config
            return runner_result

    import evaluation.systems as systems

    with patch.object(systems, "get_runner", return_value=Runner(), create=True):
        result = run_worker(attempt / "job.json")

    assert result.run_id == "run-canonical"
    assert result.normalized_exact_match is False
    canonical_trace = json.loads((attempt / "trace.json").read_text(encoding="utf-8"))
    assert canonical_trace["trace_scope"] == "canonical_tool_calls"
    assert canonical_trace["native_trace_jsonl"] is None
    assert {
        "answer.md",
        "trace.json",
        "metrics.json",
        "failure.json",
        "native",
        "result.json",
    }.issubset({item.name for item in attempt.iterdir()})
    with pytest.raises(EvaluationStateError, match="already exists"):
        run_worker(attempt / "job.json")


def test_judge_failure_preserves_system_answer_and_metrics(tmp_path: Path) -> None:
    task = EvalTask(id="judge-task", question="What is the answer?")
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        attempt,
        overrides={
            "judge": {
                "id": "exploding-judge",
                "version": "1",
                "config": {},
            }
        },
    )
    job = WorkerJob(run_id="run-judge-failure", git_sha="deadbeef")
    atomic_write_json(attempt / "task.json", json_ready(task))
    atomic_write_json(
        attempt / "config.json",
        config.model_dump(mode="json", exclude_none=False),
    )
    atomic_write_json(
        attempt / "job.json",
        job.model_dump(mode="json", exclude_none=False),
    )
    runner_result = _make_result(
        task=task,
        config=config,
        answer="system answer survives",
    )

    class Runner:
        def run(self, received_task, received_config):
            assert received_task == task
            assert received_config == config
            return runner_result

    class ExplodingJudge:
        id = "exploding-judge"
        version = "1"

        def evaluate(self, received_task, received_result, judge_config):
            assert received_task == task
            assert received_result.final_answer == "system answer survives"
            assert judge_config.id == self.id
            raise RuntimeError("judge provider failed")

    import evaluation.systems as systems

    with patch.object(systems, "get_runner", return_value=Runner(), create=True):
        result = run_worker(
            attempt / "job.json",
            judge=ExplodingJudge(),
        )

    assert result.completion_status == CompletionStatus.FAILED
    assert result.failure_type == FailureType.JUDGE_ERROR
    assert result.failure is not None
    assert result.failure.stage == "judge"
    assert result.final_answer == "system answer survives"
    assert result.search_calls == runner_result.search_calls
    assert result.fetch_calls == runner_result.fetch_calls
    assert result.judge_score is None
    assert result.judge_result is None


def test_judge_is_not_run_over_an_existing_system_failure(tmp_path: Path) -> None:
    task = EvalTask(id="judge-system-failure", question="What failed?")
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        attempt,
        overrides={
            "judge": {
                "id": "must-not-run-judge",
                "version": "1",
                "config": {},
            }
        },
    )
    job = WorkerJob(run_id="run-system-failure", git_sha="deadbeef")
    atomic_write_json(attempt / "task.json", json_ready(task))
    atomic_write_json(
        attempt / "config.json",
        config.model_dump(mode="json", exclude_none=False),
    )
    atomic_write_json(
        attempt / "job.json",
        job.model_dump(mode="json", exclude_none=False),
    )
    runner_result = _make_result(
        task=task,
        config=config,
        answer=None,
        completion_status=CompletionStatus.FAILED,
        failure_type=FailureType.SEARCH_ERROR,
        exact_match=None,
    )

    class Runner:
        def run(self, received_task, received_config):
            assert received_task == task
            assert received_config == config
            return runner_result

    class MustNotRunJudge:
        id = "must-not-run-judge"
        version = "1"

        def evaluate(self, *_args, **_kwargs):
            raise AssertionError("judge must not run over a system failure")

    import evaluation.systems as systems

    with patch.object(systems, "get_runner", return_value=Runner(), create=True):
        result = run_worker(
            attempt / "job.json",
            judge=MustNotRunJudge(),
        )

    assert result.completion_status == CompletionStatus.FAILED
    assert result.failure_type == FailureType.SEARCH_ERROR
    assert result.failure is not None
    assert result.failure.failure_type == FailureType.SEARCH_ERROR
    assert result.judge_score is None
    assert result.judge_result is None


def test_aggregate_latest_terminal_nulls_failures_and_fairness(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "evaluations" / "aggregate-demo"
    task_a = EvalTask(
        id="task-a",
        question="A?",
        reference_answer="fixture answer",
        metadata={},
    )
    task_b = EvalTask(
        id="task-b",
        question="B?",
        reference_answer=None,
        metadata={},
    )
    simple_a = experiment / "simple_react" / task_a.id / "attempt-0001"
    simple_a.mkdir(parents=True)
    simple_config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        simple_a,
    )
    persist_terminal_result(
        simple_a,
        _make_result(
            task=task_a,
            config=simple_config,
            evidence_count=None,
            coverage=None,
        ),
    )
    (simple_a.parent / "attempt-0002").mkdir()

    simple_b = experiment / "simple_react" / task_b.id / "attempt-0001"
    simple_b.mkdir(parents=True)
    simple_b_config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        simple_b,
    )
    persist_terminal_result(
        simple_b,
        _make_result(
            task=task_b,
            config=simple_b_config,
            answer=None,
            completion_status=CompletionStatus.FAILED,
            failure_type=FailureType.RUNNER_ERROR,
            exact_match=None,
            search_calls=None,
            fetch_calls=None,
            relevant_searches=None,
        ),
    )

    tong = experiment / "tongagent" / task_a.id / "attempt-0001"
    tong.mkdir(parents=True)
    tong_config = resolve_system_config(
        "tongagent",
        "sha256:dataset",
        0,
        tong,
    )
    persist_terminal_result(
        tong,
        _make_result(
            task=task_a,
            config=tong_config,
            evidence_count=2,
            coverage=1.0,
        ),
    )

    summary = aggregate_experiment(experiment)
    assert summary["selected_result_count"] == 3
    assert summary["incomplete_attempt_count"] == 1
    simple_row = next(
        row
        for row in summary["results"]
        if row["system_id"] == "simple_react" and row["task_id"] == "task-a"
    )
    assert simple_row["attempt"] == "attempt-0001"
    assert simple_row["evidence_count"] is None
    assert simple_row["structural_subquestion_coverage"] is None
    simple_summary = next(
        item for item in summary["systems"] if item["system_id"] == "simple_react"
    )
    assert simple_summary["failure_distribution"] == {"runner_error": 1}
    assert simple_summary["known_search_call_runs"] == 1
    assert simple_summary["known_fetch_call_runs"] == 1
    assert simple_summary["known_relevant_search_runs"] == 1
    assert simple_summary["total_search_calls"] is None
    assert simple_summary["total_fetch_calls"] is None
    assert simple_summary["total_relevant_searches"] is None
    assert simple_summary["total_tokens"] is None

    with (experiment / "summary.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_simple = next(
        row
        for row in csv_rows
        if row["system_id"] == "simple_react" and row["task_id"] == "task-a"
    )
    assert csv_simple["evidence_count"] == ""
    assert csv_simple["total_tokens"] == ""
    csv_failed = next(
        row
        for row in csv_rows
        if row["system_id"] == "simple_react" and row["task_id"] == "task-b"
    )
    assert csv_failed["search_calls"] == ""
    assert csv_failed["fetch_calls"] == ""
    assert csv_failed["relevant_searches"] == ""
    markdown = (experiment / "summary.md").read_text(encoding="utf-8")
    assert "—" in markdown
    assert "| Not completed |" in markdown
    assert "| Failed |" not in markdown

    unfair = tmp_path / "evaluations" / "unfair"
    unfair_a = unfair / "simple_react" / task_a.id / "attempt-0001"
    unfair_a.mkdir(parents=True)
    fair_config = resolve_system_config(
        "simple_react",
        "sha256:dataset",
        0,
        unfair_a,
    )
    persist_terminal_result(
        unfair_a,
        _make_result(task=task_a, config=fair_config),
    )
    unfair_b = unfair / "tongagent" / task_a.id / "attempt-0001"
    unfair_b.mkdir(parents=True)
    changed_budget = resolve_system_config(
        "tongagent",
        "sha256:dataset",
        0,
        unfair_b,
        overrides={"budget": {"max_search_calls": 3}},
    )
    persist_terminal_result(
        unfair_b,
        _make_result(
            task=task_a,
            config=changed_budget,
            evidence_count=1,
            coverage=1.0,
        ),
    )
    with pytest.raises(FairnessMismatchError, match="mixed fairness"):
        aggregate_experiment(unfair)

    mixed_sha = tmp_path / "evaluations" / "mixed-sha"
    for task, git_sha in ((task_a, "deadbeef"), (task_b, "cafebabe")):
        attempt = mixed_sha / "simple_react" / task.id / "attempt-0001"
        attempt.mkdir(parents=True)
        config = resolve_system_config(
            "simple_react",
            "sha256:dataset",
            0,
            attempt,
        )
        persist_terminal_result(
            attempt,
            _make_result(
                task=task,
                config=config,
                git_sha=git_sha,
                exact_match=None if task.reference_answer is None else True,
            ),
        )
    with pytest.raises(EvaluationStateError, match="mixed git SHAs"):
        aggregate_experiment(mixed_sha)


def test_dry_run_has_no_side_effect_and_env_is_sanitized(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "tasks.jsonl")
    output = tmp_path / "output"
    report = run_dataset(
        dataset,
        systems=["simple_react", "tongagent"],
        output_directory=output,
        experiment_id="dry",
        dry_run=True,
        git_sha="deadbeef",
    )
    assert report.dry_run == 2
    assert not output.exists()

    env = sanitized_subprocess_env(
        seed=11,
        backend_kind="fixture",
        source={
            "PATH": os.environ.get("PATH", ""),
            "OPENAI_API_KEY": "secret",
            "OPENAI_API_BASE": "https://leaked.invalid",
            "HF_TOKEN": "secret",
            "TAVILY_API_KEY": "secret",
            "LANGSMITH_TRACING": "true",
            "MODEL_NAME": "leaked-model",
            "SAFE_FLAG": "kept",
        },
    )
    assert env["SAFE_FLAG"] == "kept"
    assert env["PYTHONHASHSEED"] == "11"
    assert env["TONGAGENT_OFFLINE"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_API_BASE" not in env
    assert "HF_TOKEN" not in env
    assert "TAVILY_API_KEY" not in env
    assert "LANGSMITH_TRACING" not in env
    assert "MODEL_NAME" not in env


def test_active_experiment_lock_rejects_second_launcher_without_attempts(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "evaluations" / "locked"
    manifest = {"experiment_id": "locked", "dataset_digest": "sha256:test"}
    with _experiment_lock(experiment, manifest=manifest):
        with pytest.raises(EvaluationStateError, match="active primary launcher"):
            with _experiment_lock(experiment, manifest=manifest):
                pass
        assert not list(experiment.rglob("attempt-0002"))
    assert not (experiment / ".experiment.lock").exists()
