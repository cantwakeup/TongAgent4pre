"""Run the frozen exploratory BrowseComp resource-envelope matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.aggregate import aggregate_experiment
from evaluation.execution import (
    atomic_write_json,
    load_jsonl_dataset,
    resolve_git_sha,
    resolve_system_config,
    run_evaluation,
    validate_persisted_result,
)
from evaluation.schema import EvalTask, RunResult
from evaluation.systems.transparent_react import TRANSPARENT_REACT_SYSTEM_PROMPT


EXPERIMENT_PREFIX = "browsecomp-high-budget-resource-envelope-v1"
BENCHMARK_STATUS = "exploratory BrowseComp high-budget study"
STUDY_TYPE = "exploratory resource-envelope study"
SYSTEMS = ("bare_simple_react", "tongagent_standard")
COST_CAP_USD = 11.60
UNCACHED_INPUT_PER_TOKEN = 5.0 / 1_000_000
CACHED_INPUT_PER_TOKEN = 0.5 / 1_000_000
OUTPUT_PER_TOKEN = 30.0 / 1_000_000
WORST_CASE_RUN_COST = 0.725
EXPECTED_BUDGET = {
    "max_search_calls": 8,
    "max_fetch_calls": 12,
    "max_total_tool_calls": 20,
    "max_model_calls": 20,
    "max_total_tokens": 120_000,
    "wall_time_seconds": 1_200.0,
    "max_results_per_search": 5,
    "max_page_chars": 12_000,
    "recursion_limit": 125,
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordered_tasks(
    dataset_path: Path,
    manifest: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[str, tuple[EvalTask, ...]]:
    expected_dataset_hash = manifest.get("parent_dataset_sha256")
    if _sha256_file(dataset_path) != expected_dataset_hash:
        raise ValueError("parent formal dataset hash differs from frozen manifest")
    loaded = load_jsonl_dataset(dataset_path, seed=seed)
    selected_ids = manifest.get("selected_task_ids")
    selected_indices = manifest.get("selected_source_indices")
    question_hashes = manifest.get("question_sha256")
    if (
        not isinstance(selected_ids, list)
        or not isinstance(selected_indices, list)
        or not isinstance(question_hashes, Mapping)
    ):
        raise ValueError("manifest lacks frozen BrowseComp identities")
    by_id = {task.id: task for task in loaded.selected_tasks}
    if not set(selected_ids).issubset(by_id):
        raise ValueError("frozen BrowseComp task is missing from parent dataset")
    ordered = tuple(by_id[str(task_id)] for task_id in selected_ids)
    if [task.source_index for task in ordered] != selected_indices:
        raise ValueError("BrowseComp source-index order differs from manifest")
    for task in ordered:
        observed = hashlib.sha256(task.question.encode("utf-8")).hexdigest()
        if observed != question_hashes.get(task.id):
            raise ValueError(f"BrowseComp question hash differs: {task.id}")
        if task.source_dataset != "smolagents/browse_comp":
            raise ValueError(f"non-BrowseComp task in frozen manifest: {task.id}")
    return loaded.digest, ordered


def _job_plan(tasks: tuple[EvalTask, ...]) -> list[dict[str, str]]:
    return [
        {"task_id": task.id, "system_id": system_id}
        for task in tasks
        for system_id in SYSTEMS
    ]


def _known_cost(result: RunResult) -> float | None:
    usage = result.token_usage
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = min(usage.input_tokens, usage.cached_input_tokens or 0)
    uncached = usage.input_tokens - cached
    return (
        uncached * UNCACHED_INPUT_PER_TOKEN
        + cached * CACHED_INPUT_PER_TOKEN
        + usage.output_tokens * OUTPUT_PER_TOKEN
    )


def _validate_fairness(
    *,
    dataset_digest: str,
    seed: int,
    config_overrides: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    configs = [
        resolve_system_config(
            system_id,
            dataset_digest,
            seed,
            output_root / "preflight" / system_id / "attempt-0001",
            overrides=config_overrides,
        )
        for system_id in SYSTEMS
    ]
    if len({item.fairness_fingerprint for item in configs}) != 1:
        raise ValueError("Bare and Standard resource configs are not fair")
    first = configs[0]
    if first.model.name != "gpt-5.5" or first.model.temperature != 1.0:
        raise ValueError("unexpected model identity or temperature")
    if first.model.max_output_tokens != 5_000:
        raise ValueError("max_output_tokens must be 5000")
    if first.budget.model_dump(mode="json") != EXPECTED_BUDGET:
        raise ValueError("resource-envelope budget differs from preregistration")
    if first.system_options.get("high_budget_finalization") is not None:
        raise ValueError("BrowseComp resource study must not add finalization logic")
    return {
        "schema_version": 1,
        "bare_standard_prompt_parity": True,
        "policy_prompt_sha256": hashlib.sha256(
            TRANSPARENT_REACT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "fairness_fingerprint": first.fairness_fingerprint,
        "model": first.model.model_dump(mode="json"),
        "tools": first.tools.model_dump(mode="json"),
        "budget": first.budget.model_dump(mode="json"),
        "seed": seed,
    }


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _launcher_lock(experiment_directory: Path, experiment_id: str) -> Iterator[None]:
    experiment_directory.mkdir(parents=True, exist_ok=True)
    path = experiment_directory / ".browsecomp-high-budget-launcher.lock"
    if path.exists():
        try:
            existing = _load_json(path)
            existing_pid = int(existing.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_pid = 0
        if _pid_is_alive(existing_pid):
            raise RuntimeError(f"launcher already active with pid {existing_pid}")
        path.unlink()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "experiment_id": experiment_id}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def _assert_frozen_checkout(package_root: Path, expected_head: str) -> str:
    observed = resolve_git_sha(package_root)
    if observed != expected_head:
        raise RuntimeError(
            f"HEAD mismatch: expected {expected_head}, observed {observed}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if status.strip():
        raise RuntimeError("worktree is not clean")
    return observed


def _completed_prefix(
    *,
    experiment_directory: Path,
    plan: list[dict[str, str]],
    task_by_id: Mapping[str, EvalTask],
    dataset_digest: str,
    seed: int,
    git_sha: str,
    config_overrides: Mapping[str, Any],
) -> list[RunResult]:
    completed: list[RunResult] = []
    gap_seen = False
    for job in plan:
        task = task_by_id[job["task_id"]]
        task_root = experiment_directory / job["system_id"] / task.id
        attempts = sorted(task_root.glob("attempt-*")) if task_root.exists() else []
        if attempts:
            expected_attempt = task_root / "attempt-0001"
            if attempts != [expected_attempt]:
                raise RuntimeError(f"unexpected attempt layout: {task_root}")
            result_path = expected_attempt / "result.json"
            if not result_path.is_file():
                raise RuntimeError(
                    "incomplete attempt-0001 requires an explicit infrastructure "
                    f"decision; refusing automatic rerun: {expected_attempt}"
                )
            if gap_seen:
                raise RuntimeError("terminal jobs do not form a fixed-order prefix")
            config = resolve_system_config(
                job["system_id"],
                dataset_digest,
                seed,
                expected_attempt,
                overrides=config_overrides,
            )
            completed.append(
                validate_persisted_result(
                    result_path,
                    task=task,
                    system_id=job["system_id"],
                    expected_config=config,
                    expected_git_sha=git_sha,
                )
            )
        else:
            gap_seen = True
    return completed


def _state_entry(index: int, result: RunResult, known: float | None) -> dict[str, Any]:
    return {
        "job_index": index,
        "task_id": result.task_id,
        "system_id": result.system_id,
        "completion_status": result.completion_status.value,
        "failure_type": result.failure_type.value if result.failure_type else None,
        "known_cost": known,
        "artifact_directory": result.artifact_directory,
        "raw_final_equal": result.raw_model_answer == result.final_answer,
    }


def run(
    *,
    dataset_path: Path,
    manifest_path: Path,
    config_path: Path,
    output_root: Path,
    experiment_id: str,
    seed: int,
    resume: bool,
    dry_run: bool,
    expected_head: str | None,
) -> dict[str, Any]:
    if not experiment_id.startswith(EXPERIMENT_PREFIX + "-"):
        raise ValueError(f"experiment id must start with {EXPERIMENT_PREFIX}-")
    manifest = _load_json(manifest_path)
    if manifest.get("manifest_id") != "browsecomp_high_budget_8":
        raise ValueError("unexpected BrowseComp resource manifest")
    package_root = Path(__file__).resolve().parents[1]
    parent_manifest = package_root / str(manifest.get("parent_manifest", ""))
    if _sha256_file(parent_manifest) != manifest.get("parent_manifest_sha256"):
        raise ValueError("parent formal manifest hash differs from preregistration")
    digest, tasks = _ordered_tasks(dataset_path, manifest, seed=seed)
    config_overrides = _load_json(config_path)
    fairness = _validate_fairness(
        dataset_digest=digest,
        seed=seed,
        config_overrides=config_overrides,
        output_root=output_root,
    )
    plan = _job_plan(tasks)
    if len(plan) != 16:
        raise ValueError("BrowseComp resource study must contain 16 jobs")
    if dry_run:
        return {
            "experiment_id": experiment_id,
            "benchmark_status": BENCHMARK_STATUS,
            "study_type": STUDY_TYPE,
            "dry_run_jobs": len(plan),
            "job_plan": plan,
            "fairness": fairness,
            "cost_cap": COST_CAP_USD,
            "worst_case_run_cost": WORST_CASE_RUN_COST,
            "worst_case_total_cost": WORST_CASE_RUN_COST * len(plan),
        }
    if expected_head is None:
        raise ValueError("--expected-head is required for a live launch")
    git_sha = _assert_frozen_checkout(package_root, expected_head)
    experiment_directory = output_root.resolve() / experiment_id
    if experiment_directory.exists() and not resume:
        raise RuntimeError("experiment exists; use --resume after inspecting artifacts")
    task_by_id = {task.id: task for task in tasks}

    def config_factory(
        system_id: str,
        dataset_digest: str,
        selected_seed: int,
        attempt_directory: Path,
    ):
        return resolve_system_config(
            system_id,
            dataset_digest,
            selected_seed,
            attempt_directory,
            overrides=config_overrides,
        )

    with _launcher_lock(experiment_directory, experiment_id):
        manifest_artifact = experiment_directory / "exploratory_manifest.json"
        manifest_payload = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "benchmark_status": BENCHMARK_STATUS,
            "study_type": STUDY_TYPE,
            "formal_result_replacement": False,
            "git_sha": git_sha,
            "dataset_digest": digest,
            "source_manifest": str(manifest_path.resolve()),
            "config": str(config_path.resolve()),
            "job_plan": plan,
            "fairness": fairness,
            "cost_cap": COST_CAP_USD,
            "worst_case_run_cost": WORST_CASE_RUN_COST,
            "worst_case_total_cost": WORST_CASE_RUN_COST * len(plan),
            "created_at": datetime.now(UTC).isoformat(),
        }
        if manifest_artifact.exists():
            existing = _load_json(manifest_artifact)
            for key in ("experiment_id", "git_sha", "dataset_digest", "job_plan"):
                if existing.get(key) != manifest_payload.get(key):
                    raise RuntimeError(f"resume manifest mismatch: {key}")
        else:
            atomic_write_json(manifest_artifact, manifest_payload)

        prefix = _completed_prefix(
            experiment_directory=experiment_directory,
            plan=plan,
            task_by_id=task_by_id,
            dataset_digest=digest,
            seed=seed,
            git_sha=git_sha,
            config_overrides=config_overrides,
        )
        completed_entries: list[dict[str, Any]] = []
        known_cost = 0.0
        conservative_cost = 0.0
        for index, result in enumerate(prefix, start=1):
            cost = _known_cost(result)
            known_cost += cost or 0.0
            conservative_cost += cost if cost is not None else WORST_CASE_RUN_COST
            completed_entries.append(_state_entry(index, result, cost))

        stop_reason: str | None = None
        for job_index, job in enumerate(plan[len(prefix) :], start=len(prefix) + 1):
            if conservative_cost + WORST_CASE_RUN_COST > COST_CAP_USD + 1e-9:
                stop_reason = "pre_dispatch_worst_case_cost_guard"
                break
            task = task_by_id[job["task_id"]]
            report = run_evaluation(
                tasks=(task,),
                dataset_digest=digest,
                systems=(job["system_id"],),
                seed=seed,
                output_directory=output_root,
                experiment_id=experiment_id,
                resume=True,
                rerun=False,
                dry_run=False,
                config_factory=config_factory,
                git_sha=git_sha,
            )
            result = report.outcomes[0].result
            if result is None:
                raise RuntimeError("executed BrowseComp job returned no result")
            cost = _known_cost(result)
            known_cost += cost or 0.0
            conservative_cost += cost if cost is not None else WORST_CASE_RUN_COST
            entry = _state_entry(job_index, result, cost)
            completed_entries.append(entry)
            state = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "completed_jobs": completed_entries,
                "completed_count": len(completed_entries),
                "planned_count": len(plan),
                "known_cost": known_cost,
                "conservative_cost": conservative_cost,
                "cost_cap": COST_CAP_USD,
                "stop_reason": None,
                "core_complete": False,
            }
            atomic_write_json(
                experiment_directory / "browsecomp_high_budget_run_state.json",
                state,
                overwrite=True,
            )
            aggregate_experiment(experiment_directory)
            print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
            if (
                job["system_id"] == "tongagent_standard"
                and not entry["raw_final_equal"]
            ):
                stop_reason = "standard_raw_final_mismatch"
                break
            if known_cost >= COST_CAP_USD:
                stop_reason = "known_cost_cap_reached"
                break

        final_state = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "completed_jobs": completed_entries,
            "completed_count": len(completed_entries),
            "planned_count": len(plan),
            "known_cost": known_cost,
            "conservative_cost": conservative_cost,
            "cost_cap": COST_CAP_USD,
            "stop_reason": stop_reason,
            "core_complete": len(completed_entries) == len(plan),
        }
        atomic_write_json(
            experiment_directory / "browsecomp_high_budget_run_state.json",
            final_state,
            overwrite=True,
        )
        if completed_entries:
            aggregate_experiment(experiment_directory)
    return final_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/evaluations"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--expected-head")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = run(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        config_path=args.config,
        output_root=args.output,
        experiment_id=args.experiment_id,
        seed=args.seed,
        resume=args.resume,
        dry_run=args.dry_run,
        expected_head=args.expected_head,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
