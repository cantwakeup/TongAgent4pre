"""Offline end-to-end tests for the B1 and B2 system adapters."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import deepagents
import pytest

from evaluation import (
    BudgetLimits,
    CompletionStatus,
    EvalTask,
    EvaluationModelConfig,
    ResolvedConfig,
    SharedToolConfig,
    ToolCallStatus,
)
from evaluation.execution import WorkerJob
from evaluation.offline import FixtureBackend
from evaluation.systems import (
    SimpleReactRunner,
    VanillaDeepAgentsRunner,
    get_runner,
)
from evaluation.worker import run_worker


pytestmark = pytest.mark.usefixtures("socket_disabled")


def _write_fixture(
    root: Path,
    *,
    searches: dict[str, Any],
    pages: dict[str, Any],
) -> FixtureBackend:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixture_id": "baseline-test",
                "search_file": "search.json",
                "page_file": "pages.json",
            }
        ),
        encoding="utf-8",
    )
    (root / "search.json").write_text(json.dumps(searches), encoding="utf-8")
    (root / "pages.json").write_text(json.dumps(pages), encoding="utf-8")
    return FixtureBackend.from_directory(root)


def _config(
    *,
    system_id: str,
    fixture: FixtureBackend,
    fixture_dir: Path,
    artifact_directory: Path,
    max_search_calls: int = 2,
    max_fetch_calls: int = 2,
    max_total_tool_calls: int = 4,
) -> ResolvedConfig:
    return ResolvedConfig(
        system_id=system_id,
        dataset_digest="sha256:baseline-test-dataset",
        backend_kind="fixture",
        fixture_revision=fixture.revision,
        model=EvaluationModelConfig(
            provider="fixture",
            name="fixture-chat-model",
            temperature=0.0,
            max_output_tokens=128,
        ),
        tools=SharedToolConfig(
            search_backend="fixture",
            fetch_backend="fixture",
        ),
        budget=BudgetLimits(
            max_search_calls=max_search_calls,
            max_fetch_calls=max_fetch_calls,
            max_total_tool_calls=max_total_tool_calls,
            max_model_calls=8,
            max_total_tokens=100_000,
            wall_time_seconds=30.0,
            max_results_per_search=3,
            max_page_chars=2_000,
        ),
        seed=7,
        system_options={"fixture_dir": str(fixture_dir)},
        artifact_directory=str(artifact_directory),
    )


def _research_task() -> EvalTask:
    answer = "Alpha is verified. [S1] https://fixture.test/alpha"
    return EvalTask(
        id="baseline-loop",
        question="Is Alpha verified?",
        reference_answer=answer,
        metadata={
            "answer": {
                "simple_react": answer,
                "vanilla_deepagents": answer,
            },
            "strict_fixture_tools": True,
            "research_script": {
                "simple_react": [
                    {
                        "tool": "web_search",
                        "args": {"query": "fixture alpha", "max_results": 3},
                    },
                    {
                        "tool": "fetch_url",
                        "args": {
                            "url": "https://fixture.test/alpha",
                            "max_chars": 2_000,
                        },
                    },
                ],
                "vanilla_deepagents": [
                    {
                        "tool": "web_search",
                        "args": {"query": "fixture alpha", "max_results": 3},
                    },
                    {
                        "tool": "fetch_url",
                        "args": {
                            "url": "https://fixture.test/alpha",
                            "max_chars": 2_000,
                        },
                    },
                ],
            },
        },
    )


@pytest.fixture
def research_fixture(tmp_path: Path) -> tuple[Path, FixtureBackend]:
    fixture_dir = tmp_path / "fixture"
    content = (
        "Alpha is verified by this deterministic primary fixture. "
        + "Supporting context remains local and deterministic. " * 12
    )
    backend = _write_fixture(
        fixture_dir,
        searches={
            "fixture alpha": {
                "status": "success",
                "results": [
                    {
                        "title": "Alpha primary fixture",
                        "url": "https://fixture.test/alpha",
                        "snippet": "Alpha is verified.",
                        "relevance_score": 100,
                    }
                ],
            }
        },
        pages={
            "https://fixture.test/alpha": {
                "status": "success",
                "title": "Alpha primary fixture",
                "content": content,
            }
        },
    )
    return fixture_dir, backend


@pytest.mark.parametrize(
    ("system_id", "runner_type"),
    [
        ("simple_react", SimpleReactRunner),
        ("vanilla_deepagents", VanillaDeepAgentsRunner),
    ],
)
def test_real_baseline_loops_share_tools_and_keep_tongagent_metrics_null(
    tmp_path: Path,
    research_fixture: tuple[Path, FixtureBackend],
    system_id: str,
    runner_type: type[SimpleReactRunner] | type[VanillaDeepAgentsRunner],
) -> None:
    fixture_dir, backend = research_fixture
    artifact_directory = tmp_path / "runs" / system_id / "attempt-1"
    config = _config(
        system_id=system_id,
        fixture=backend,
        fixture_dir=fixture_dir,
        artifact_directory=artifact_directory,
    )

    result = runner_type(fixture_backend=backend).run(_research_task(), config)

    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.final_answer == _research_task().reference_answer
    assert result.normalized_exact_match is True
    assert result.search_calls == 1
    assert result.fetch_calls == 1
    assert result.relevant_searches == 1
    assert result.evidence_count is None
    assert result.structural_subquestion_coverage is None
    assert result.estimated_cost is None
    assert result.judge_score is None
    assert result.token_usage is not None
    assert result.token_usage.total_tokens is not None
    assert [item.tool_name for item in result.tool_calls] == [
        "web_search",
        "fetch_url",
    ]
    assert all(item.status == ToolCallStatus.SUCCESS for item in result.tool_calls)
    assert result.citations[0].source_id == "S1"
    assert result.citations[0].url == "https://fixture.test/alpha"

    native_directory = artifact_directory / "native"
    assert (native_directory / "resolved_config.json").is_file()
    assert (native_directory / "native.json").is_file()
    assert (native_directory / "budget.json").is_file()
    assert (native_directory / "tool_calls.json").is_file()
    assert (native_directory / "trace.jsonl").is_file()
    assert (native_directory / "answer.md").is_file()
    assert not (artifact_directory / "result.json").exists()
    trace = (native_directory / "trace.jsonl").read_text(encoding="utf-8")
    assert "model_call_finished" in trace
    assert "tool_call_finished" in trace
    assert "relevant_search" in trace
    if system_id == "vanilla_deepagents":
        assert (native_directory / "workspace").is_dir()


def test_shared_execution_budget_denies_second_search_before_provider_call(
    tmp_path: Path,
) -> None:
    fixture_dir = tmp_path / "fixture"
    backend = _write_fixture(
        fixture_dir,
        searches={
            "fixture alpha": {
                "status": "success",
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://fixture.test/alpha",
                        "snippet": "Alpha",
                        "relevance_score": 100,
                    }
                ],
            }
        },
        pages={},
    )
    task = EvalTask(
        id="budget-denial",
        question="Search twice.",
        reference_answer="Stopped at the shared budget.",
        metadata={
            "answer": "Stopped at the shared budget.",
            "strict_fixture_tools": True,
            "research_script": [
                {
                    "tool": "web_search",
                    "args": {"query": "fixture alpha"},
                },
                {
                    "tool": "web_search",
                    "args": {"query": "fixture alpha"},
                },
            ],
        },
    )
    artifact_directory = tmp_path / "runs" / "attempt-1"
    config = _config(
        system_id="simple_react",
        fixture=backend,
        fixture_dir=fixture_dir,
        artifact_directory=artifact_directory,
        max_search_calls=1,
        max_fetch_calls=0,
        max_total_tool_calls=2,
    )

    result = SimpleReactRunner(fixture_backend=backend).run(task, config)

    assert result.completion_status == CompletionStatus.BUDGET_EXHAUSTED
    assert result.search_calls == 1
    assert result.relevant_searches == 1
    assert [item.status for item in result.tool_calls] == [
        ToolCallStatus.SUCCESS,
        ToolCallStatus.BUDGET_EXCEEDED,
    ]
    assert len(backend.calls) == 1
    assert result.failure_type is not None
    assert result.failure_type.value == "budget_exhausted"


@pytest.mark.parametrize("system_id", ["simple_react", "vanilla_deepagents"])
def test_worker_registry_runs_real_baselines_and_publishes_result_last(
    tmp_path: Path,
    research_fixture: tuple[Path, FixtureBackend],
    system_id: str,
) -> None:
    fixture_dir, backend = research_fixture
    attempt = tmp_path / "worker" / system_id / "attempt-0001"
    attempt.mkdir(parents=True)
    task = _research_task()
    config = _config(
        system_id=system_id,
        fixture=backend,
        fixture_dir=fixture_dir,
        artifact_directory=attempt,
    )
    job = WorkerJob(
        run_id=f"canonical-{system_id}",
        git_sha="deadbeef",
    )
    (attempt / "task.json").write_text(task.model_dump_json(), encoding="utf-8")
    (attempt / "config.json").write_text(
        config.model_dump_json(),
        encoding="utf-8",
    )
    (attempt / "job.json").write_text(job.model_dump_json(), encoding="utf-8")

    result = run_worker(attempt / "job.json")

    assert result.run_id == f"canonical-{system_id}"
    assert result.git_sha == "deadbeef"
    assert result.completion_status == CompletionStatus.COMPLETED
    assert (attempt / "result.json").is_file()
    assert (attempt / "answer.md").is_file()
    assert (attempt / "trace.json").is_file()
    assert (attempt / "metrics.json").is_file()
    assert (attempt / "failure.json").is_file()
    assert (attempt / "native" / "native.json").is_file()
    assert (attempt / "native" / "trace.jsonl").is_file()


def test_nonempty_irrelevant_fixture_does_not_increment_relevant_searches(
    tmp_path: Path,
) -> None:
    fixture_dir = tmp_path / "fixture"
    backend = _write_fixture(
        fixture_dir,
        searches={
            "target fact": {
                "status": "success",
                "results": [
                    {
                        "title": "Unrelated",
                        "url": "https://fixture.test/noise",
                        "snippet": "No target terms are present.",
                        "relevance_score": 0,
                    }
                ],
                "relevant_search": True,
                "relevant_results": 1,
            }
        },
        pages={},
    )
    task = EvalTask(
        id="irrelevant",
        question="Find the target fact.",
        metadata={
            "answer": "The fixture did not provide relevant evidence.",
            "research_script": [
                {
                    "tool": "web_search",
                    "args": {"query": "target fact"},
                }
            ],
        },
    )
    config = _config(
        system_id="simple_react",
        fixture=backend,
        fixture_dir=fixture_dir,
        artifact_directory=tmp_path / "runs" / "irrelevant",
    )

    result = SimpleReactRunner(fixture_backend=backend).run(task, config)

    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.search_calls == 1
    assert result.relevant_searches == 0
    search = result.tool_calls[0]
    assert search.metadata["provider_status"] == "success"
    assert search.metadata["nonempty_search"] is True
    assert search.metadata["relevant_search"] is False


def test_baseline_modules_do_not_import_tongagent_graphs_or_profiles() -> None:
    systems_dir = Path(__file__).resolve().parents[1] / "evaluation" / "systems"
    forbidden_modules = {
        "adaptive_control",
        "evidence_graph",
        "research_graph",
        "search_agent",
    }
    forbidden_names = {"AgentBundle", "ResearchPlan", "build_agent"}

    for filename in ("simple_react.py", "vanilla_deepagents.py"):
        tree = ast.parse((systems_dir / filename).read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
                imported_names.update(alias.name for alias in node.names)
        assert forbidden_modules.isdisjoint(imported_modules)
        assert forbidden_names.isdisjoint(imported_names)

    assert deepagents.__version__ == "0.6.12"
    assert isinstance(get_runner("simple_react"), SimpleReactRunner)
    assert isinstance(get_runner("vanilla_deepagents"), VanillaDeepAgentsRunner)
