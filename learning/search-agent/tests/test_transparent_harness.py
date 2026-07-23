"""Offline parity and recovery tests for TongAgent Standard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import (
    BudgetLimits,
    CompletionStatus,
    EvalTask,
    EvaluationModelConfig,
    ResolvedConfig,
    SharedToolConfig,
)
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.systems import BareSimpleReactRunner, TongAgentStandardRunner
from evaluation.systems import transparent_react


pytestmark = pytest.mark.usefixtures("socket_disabled")


def _backend() -> FixtureBackend:
    return FixtureBackend(
        searches={
            "fixture capital": {
                "status": "success",
                "results": [
                    {
                        "title": "Capital fact",
                        "url": "https://fixture.test/capital",
                        "snippet": "Paris is the capital.",
                        "relevance_score": 100,
                    }
                ],
            }
        },
        pages={
            "https://fixture.test/capital": {
                "status": "success",
                "title": "Capital fact",
                "content": "Paris is the capital of France. " * 24,
            }
        },
    )


def _task() -> EvalTask:
    script = [
        {"tool": "web_search", "args": {"query": "fixture capital"}},
        {
            "tool": "fetch_url",
            "args": {"url": "https://fixture.test/capital"},
        },
    ]
    return EvalTask(
        id="transparent-capital",
        question="What is the capital of France?",
        reference_answer="Paris",
        metadata={
            "answer": "FINAL_ANSWER: Paris",
            "research_script": script,
            "strict_fixture_tools": True,
        },
    )


def _config(
    tmp_path: Path,
    backend: FixtureBackend,
    *,
    system_id: str,
    artifact_name: str,
    high_budget_finalization: bool = False,
    high_budget_token_trigger: int = 0,
) -> ResolvedConfig:
    system_options: dict[str, Any] = {"fixture_dir": "evaluation/fixtures"}
    if high_budget_finalization:
        system_options["high_budget_finalization"] = {
            "enabled": True,
            "token_trigger": high_budget_token_trigger,
            "wall_time_trigger_seconds": 20.0,
            "remaining_model_calls_trigger": 1,
        }
    return ResolvedConfig(
        system_id=system_id,
        dataset_digest="sha256:transparent-harness-fixture",
        backend_kind="fixture",
        fixture_revision=backend.revision,
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
            max_search_calls=2,
            max_fetch_calls=2,
            max_total_tool_calls=4,
            max_model_calls=8,
            max_total_tokens=20_000,
            wall_time_seconds=30.0,
            max_results_per_search=3,
            max_page_chars=2_000,
        ),
        runtime_mode="tongagent_standard",
        seed=17,
        system_options=system_options,
        artifact_directory=str(tmp_path / artifact_name),
    )


def test_bare_and_standard_share_one_real_policy_and_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    real_create_agent = transparent_react.create_agent

    def recording_create_agent(*args: Any, **kwargs: Any) -> Any:
        prompts.append(str(kwargs.get("system_prompt")))
        return real_create_agent(*args, **kwargs)

    monkeypatch.setattr(transparent_react, "create_agent", recording_create_agent)
    task = _task()
    bare_backend = _backend()
    standard_backend = _backend()
    bare = BareSimpleReactRunner(fixture_backend=bare_backend).run(
        task,
        _config(
            tmp_path,
            bare_backend,
            system_id="bare_simple_react",
            artifact_name="bare",
        ),
    )
    standard = TongAgentStandardRunner(fixture_backend=standard_backend).run(
        task,
        _config(
            tmp_path,
            standard_backend,
            system_id="tongagent_standard",
            artifact_name="standard",
        ),
    )

    assert prompts == [
        transparent_react.TRANSPARENT_REACT_SYSTEM_PROMPT,
        transparent_react.TRANSPARENT_REACT_SYSTEM_PROMPT,
    ]
    assert bare.raw_model_answer == standard.raw_model_answer
    assert bare.final_answer == standard.final_answer == "FINAL_ANSWER: Paris"
    assert bare.raw_model_answer == bare.final_answer
    assert standard.raw_model_answer == standard.final_answer
    assert bare.standard_normalized_em is True
    assert standard.standard_normalized_em is True
    assert bare.answer_rate is True
    assert standard.answer_rate is True
    assert [(call.tool_name, call.arguments) for call in bare.tool_calls] == [
        (call.tool_name, call.arguments) for call in standard.tool_calls
    ]
    assert not (tmp_path / "bare" / "native" / "checkpoint.sqlite").exists()
    assert (tmp_path / "standard" / "native" / "checkpoint.sqlite").is_file()
    audit = json.loads(
        (tmp_path / "standard" / "native" / "posthoc_audit.json").read_text()
    )
    assert audit["answer_unchanged"] is True
    assert audit["mode"] == "post_hoc_non_blocking"


def test_standard_retries_one_temporary_model_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    task = _task()
    model = FixtureChatModel.from_task(task, system_id="tongagent_standard")
    real_generate = FixtureChatModel._generate
    attempts = 0

    def flaky_generate(self: FixtureChatModel, *args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary injected model connection failure")
        return real_generate(self, *args, **kwargs)

    monkeypatch.setattr(FixtureChatModel, "_generate", flaky_generate)
    result = TongAgentStandardRunner(
        fixture_backend=backend,
        model=model,
    ).run(
        task,
        _config(
            tmp_path,
            backend,
            system_id="tongagent_standard",
            artifact_name="retry",
        ),
    )

    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.final_answer == "FINAL_ANSWER: Paris"
    assert result.budget_snapshot is None
    trace = (tmp_path / "retry" / "native" / "trace.jsonl").read_text()
    assert "model_connection_retry" in trace
    assert attempts >= 2


def test_standard_retries_one_retryable_fetch_without_changing_tool_loop(
    tmp_path: Path,
) -> None:
    def flaky_backend() -> FixtureBackend:
        return FixtureBackend(
            searches={"fixture capital": _backend().search("fixture capital")},
            pages={
                "https://fixture.test/capital": {
                    "responses": [
                        {
                            "status": "error",
                            "url": "https://fixture.test/capital",
                            "error": "temporary injected fetch failure",
                            "retryable": True,
                        },
                        {
                            "status": "success",
                            "title": "Capital fact",
                            "content": "Paris is the capital of France. " * 24,
                        },
                    ]
                }
            },
        )

    task = _task()
    bare_backend = flaky_backend()
    standard_backend = flaky_backend()
    bare = BareSimpleReactRunner(fixture_backend=bare_backend).run(
        task,
        _config(
            tmp_path,
            bare_backend,
            system_id="bare_simple_react",
            artifact_name="bare-fetch-failure",
        ),
    )
    standard = TongAgentStandardRunner(fixture_backend=standard_backend).run(
        task,
        _config(
            tmp_path,
            standard_backend,
            system_id="tongagent_standard",
            artifact_name="standard-fetch-recovery",
        ),
    )

    assert bare.fetch_calls == 1
    assert bare.tool_calls[-1].status.value == "error"
    assert standard.fetch_calls == 2
    assert standard.tool_calls[-1].status.value == "success"
    assert standard.final_answer == bare.final_answer
    trace = (
        tmp_path / "standard-fetch-recovery" / "native" / "trace.jsonl"
    ).read_text()
    assert "retrieval_connection_retry" in trace


def test_standard_resumes_after_tool_checkpoint_without_refetch(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _task()
    config = _config(
        tmp_path,
        backend,
        system_id="tongagent_standard",
        artifact_name="resume",
    )

    interrupted = TongAgentStandardRunner(
        fixture_backend=backend,
        interrupt_after_tools=True,
    ).run(task, config)
    assert interrupted.final_answer is None
    assert interrupted.fetch_calls == 0
    assert interrupted.search_calls == 1

    fetched = TongAgentStandardRunner(
        fixture_backend=backend,
        interrupt_after_tools=True,
    ).run(task, config)
    assert fetched.final_answer is None
    assert fetched.search_calls == 1
    assert fetched.fetch_calls == 1

    resumed = TongAgentStandardRunner(fixture_backend=backend).run(task, config)

    assert resumed.completion_status == CompletionStatus.COMPLETED
    assert resumed.final_answer == "FINAL_ANSWER: Paris"
    assert resumed.search_calls == 1
    assert resumed.fetch_calls == 1
    assert [call.tool_name for call in resumed.tool_calls] == [
        "web_search",
        "fetch_url",
    ]
    manifest = json.loads(
        (tmp_path / "resume" / "native" / "checkpoint_manifest.json").read_text()
    )
    assert manifest["checkpoint_restored"] is True


def test_shared_high_budget_boundary_forces_one_tool_free_answer_for_both_systems(
    tmp_path: Path,
) -> None:
    task = EvalTask(
        id="transparent-forced-finalization",
        question="What is the capital of France?",
        reference_answer="Paris",
        metadata={"answer": "FINAL_ANSWER: Paris"},
    )
    bare_backend = _backend()
    standard_backend = _backend()
    bare_model = FixtureChatModel.from_task(task, system_id="bare_simple_react")
    standard_model = FixtureChatModel.from_task(
        task,
        system_id="tongagent_standard",
    )
    bare_config = _config(
        tmp_path,
        bare_backend,
        system_id="bare_simple_react",
        artifact_name="bare-forced",
        high_budget_finalization=True,
    )
    standard_config = _config(
        tmp_path,
        standard_backend,
        system_id="tongagent_standard",
        artifact_name="standard-forced",
        high_budget_finalization=True,
    )

    bare = BareSimpleReactRunner(
        fixture_backend=bare_backend,
        model=bare_model,
    ).run(task, bare_config)
    standard = TongAgentStandardRunner(
        fixture_backend=standard_backend,
        model=standard_model,
    ).run(task, standard_config)

    assert bare_config.fairness_fingerprint == standard_config.fairness_fingerprint
    assert bare.final_answer == standard.final_answer == "FINAL_ANSWER: Paris"
    assert bare.raw_model_answer == bare.final_answer
    assert standard.raw_model_answer == standard.final_answer
    assert bare.search_calls == standard.search_calls == 0
    assert bare.fetch_calls == standard.fetch_calls == 0
    assert bare_model.call_history[0]["bound_tools"] == []
    assert standard_model.call_history[0]["bound_tools"] == []
    for name in ("bare-forced", "standard-forced"):
        artifact = json.loads(
            (tmp_path / name / "native" / "finalization.json").read_text()
        )
        assert artifact["finalization_triggered"] is True
        assert artifact["finalization_reason"] == "total_tokens"
        assert artifact["natural_answer"] is None
        assert artifact["forced_final_answer"] == "FINAL_ANSWER: Paris"
        assert artifact["finalization_call_started"] is True
        assert artifact["finalization_call_finished"] is True
        assert artifact["blocked_tools_after_finalization"] == []
        assert not (tmp_path / name / "native" / "natural_answer.md").exists()
        assert (tmp_path / name / "native" / "forced_final_answer.md").is_file()


def test_high_budget_boundary_preserves_a_natural_answer_separately(
    tmp_path: Path,
) -> None:
    task = _task()
    backend = _backend()
    result = BareSimpleReactRunner(fixture_backend=backend).run(
        task,
        _config(
            tmp_path,
            backend,
            system_id="bare_simple_react",
            artifact_name="bare-natural",
            high_budget_finalization=True,
            high_budget_token_trigger=20_000,
        ),
    )

    artifact = json.loads(
        (tmp_path / "bare-natural" / "native" / "finalization.json").read_text()
    )
    assert result.final_answer == "FINAL_ANSWER: Paris"
    assert artifact["finalization_triggered"] is False
    assert artifact["natural_answer"] == "FINAL_ANSWER: Paris"
    assert artifact["forced_final_answer"] is None
    assert (tmp_path / "bare-natural" / "native" / "natural_answer.md").is_file()
    assert not (
        tmp_path / "bare-natural" / "native" / "forced_final_answer.md"
    ).exists()
