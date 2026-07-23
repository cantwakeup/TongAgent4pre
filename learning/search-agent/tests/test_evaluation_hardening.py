"""Adversarial contracts for evaluation credentials, fairness, and judging."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import search_agent
from deepagents.backends import FilesystemBackend
from deepagents._models import get_model_identifier, get_model_provider
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from evaluation import (
    BudgetExceeded,
    BudgetLimits,
    CompletionStatus,
    EvalTask,
    EvaluationJudgeConfig,
    ExecutionBudget,
    JudgeResult,
    ResolvedConfig,
    RunResult,
    apply_evaluation_judge,
    register_evaluation_judge,
)
from evaluation.execution import (
    resolve_system_config,
    sanitized_subprocess_env,
)
from evaluation.offline import FixtureChatModel
from evaluation.judging import JudgeUnavailableError, ensure_judge_available
from evaluation.systems.common import (
    EvaluationMiddleware,
    _HighBudgetFinalizationBoundary,
    build_evaluation_summarization_middleware,
)
from evaluation.tracing import TraceCollector


def _limits(*, recursion_limit: int = 25) -> BudgetLimits:
    return BudgetLimits(
        max_search_calls=2,
        max_fetch_calls=2,
        max_total_tool_calls=6,
        max_model_calls=4,
        max_total_tokens=1_000,
        wall_time_seconds=10.0,
        max_results_per_search=3,
        max_page_chars=1_000,
        recursion_limit=recursion_limit,
    )


def _config(
    *,
    backend_kind: str = "fixture",
    credential_env: list[str] | None = None,
    judge: EvaluationJudgeConfig | None = None,
    recursion_limit: int = 25,
    system_options: dict[str, object] | None = None,
    model_parameters: dict[str, object] | None = None,
) -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "system_id": "simple_react",
            "dataset_digest": "sha256:dataset",
            "backend_kind": backend_kind,
            "fixture_revision": "fixture-v1" if backend_kind == "fixture" else None,
            "model": {
                "provider": "fixture" if backend_kind == "fixture" else "openai",
                "name": "fixture-chat" if backend_kind == "fixture" else "gpt-test",
                "temperature": 0.0,
                "max_output_tokens": 32,
                "credential_env": credential_env or [],
                "parameters": model_parameters or {},
            },
            "tools": {
                "search_backend": "shared-search",
                "fetch_backend": "shared-fetch",
                "parameters": {},
            },
            "budget": _limits(recursion_limit=recursion_limit).model_dump(),
            "judge": judge.model_dump() if judge is not None else None,
            "seed": 7,
            "system_options": system_options or {},
            "artifact_directory": "/tmp/evaluation-hardening/attempt-0001",
        }
    )


def _result(config: ResolvedConfig) -> RunResult:
    now = datetime(2026, 7, 19, tzinfo=UTC)
    return RunResult(
        run_id="run-hardening",
        task_id="task-hardening",
        system_id=config.system_id,
        git_sha="deadbeef",
        resolved_config=config,
        config_fingerprint=config.config_fingerprint,
        fairness_fingerprint=config.fairness_fingerprint,
        started_at=now,
        finished_at=now,
        wall_time_seconds=0.0,
        final_answer="answer",
        citations=[],
        tool_calls=[],
        search_calls=0,
        fetch_calls=0,
        relevant_searches=0,
        evidence_count=None,
        structural_subquestion_coverage=None,
        token_usage=None,
        estimated_cost=None,
        completion_status=CompletionStatus.COMPLETED,
        failure_type=None,
        failure=None,
        artifact_directory=config.artifact_directory,
        fixture_smoke=config.backend_kind == "fixture",
        normalized_exact_match=None,
        judge_score=None,
        judge_result=None,
    )


def test_live_credentials_are_allowlisted_by_name_without_persisting_values() -> None:
    secret = "sk-test-value-that-must-never-be-persisted"
    config = _config(
        backend_kind="live",
        credential_env=["TAVILY_API_KEY", "OPENAI_API_KEY"],
    )
    env = sanitized_subprocess_env(
        seed=config.seed,
        backend_kind=config.backend_kind,
        credential_env=config.model.credential_env,
        source={
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": secret,
            "TAVILY_API_KEY": "tvly-test",
            "HF_TOKEN": "unlisted-secret",
            "OPENAI_BASE_URL": "https://implicit.invalid",
            "MODEL_NAME": "implicit-model",
            "PYTHONPATH": "/implicit/imports",
            "LANGSMITH_TRACING": "true",
            "SAFE_FLAG": "kept",
        },
    )

    assert config.model.credential_env == ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    assert env["OPENAI_API_KEY"] == secret
    assert env["TAVILY_API_KEY"] == "tvly-test"
    assert env["SAFE_FLAG"] == "kept"
    for name in {
        "HF_TOKEN",
        "OPENAI_BASE_URL",
        "MODEL_NAME",
        "PYTHONPATH",
        "LANGSMITH_TRACING",
    }:
        assert name not in env
    persisted = json.dumps(
        {
            "config": config.model_dump(mode="json"),
            "result": _result(config).model_dump(mode="json"),
        },
        sort_keys=True,
    )
    assert secret not in persisted
    assert "tvly-test" not in persisted


def test_fixture_credentials_and_implicit_allowlist_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="credential_env"):
        _config(credential_env=["OPENAI_API_KEY"])
    with pytest.raises(ValidationError, match="implicit runtime state"):
        _config(backend_kind="live", credential_env=["OPENAI_BASE_URL"])
    with pytest.raises(ValueError, match="cannot receive credential_env"):
        sanitized_subprocess_env(
            seed=1,
            backend_kind="fixture",
            credential_env=["OPENAI_API_KEY"],
            source={"OPENAI_API_KEY": "secret"},
        )
    fixture_env = sanitized_subprocess_env(
        seed=1,
        backend_kind="fixture",
        source={
            "AUTHORIZATION": "Bearer abcdef",
            "COOKIE": "session=secret",
            "X_API_KEY": "secret",
            "SAFE_FLAG": "Bearer hidden",
            "PATH": "/usr/bin",
        },
    )
    assert set(fixture_env).isdisjoint(
        {"AUTHORIZATION", "COOKIE", "X_API_KEY", "SAFE_FLAG"}
    )
    assert fixture_env["TONGAGENT_OFFLINE"] == "1"


@pytest.mark.parametrize(
    "parameters",
    [
        {"headers": {"Authorization": "Bearer abc"}},
        {"headers": {"Cookie": "session=abc"}},
        {"headers": {"X-API-Key": "abc"}},
        {"header_value": "Bearer abc"},
        {"refresh_token": "REFRESH-SECRET-123"},
        {"private_key": "PRIVATE-SECRET-456"},
        {"endpoint": ("https://api.test/v1?access_token=QUERY-SECRET-789&safe=true")},
        {"endpoint": "https://user:password@api.test/v1"},
        {"opaque": "sk-secret-value-12345678"},
    ],
)
def test_nested_config_secrets_are_rejected(
    tmp_path: Path,
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="forbidden in persisted config"):
        resolve_system_config(
            "simple_react",
            "sha256:dataset",
            1,
            tmp_path / "attempt",
            overrides={"model": {"parameters": parameters}},
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"headers": {"Authorization": "Bearer abc"}},
        {"nested": [{"Cookie": "session=abc"}]},
        {"header_value": "Bearer abc"},
        {"refresh_token": "REFRESH-SECRET-123"},
        {"private_key": "PRIVATE-SECRET-456"},
        {"endpoint": "https://api.test/v1?token=QUERY-SECRET-789"},
    ],
)
def test_programmatic_resolved_config_cannot_bypass_secret_rejection(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="forbidden in persisted config"):
        _config(model_parameters=parameters)


def test_provider_output_ceiling_cannot_hide_in_model_parameters() -> None:
    with pytest.raises(ValidationError, match="output-token ceilings"):
        _config(model_parameters={"max_tokens": 10_000})


def test_single_topology_profile_uses_resolved_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FixtureChatModel.from_task(
        EvalTask(id="profile-key", question="question"),
        system_id="tongagent",
    )
    provider = get_model_provider(model)
    identifier = get_model_identifier(model)
    assert provider is not None
    assert identifier is not None
    expected_key = f"{provider}:{identifier}"
    captured: list[tuple[str, object]] = []
    search_agent._REGISTERED_HARNESS_KEYS.discard(expected_key)
    monkeypatch.setattr(
        search_agent,
        "register_harness_profile",
        lambda key, profile: captured.append((key, profile)),
    )
    try:
        search_agent._disable_general_purpose_subagent(model)
    finally:
        search_agent._REGISTERED_HARNESS_KEYS.discard(expected_key)

    assert len(captured) == 1
    key, profile = captured[0]
    assert key == expected_key
    assert profile.general_purpose_subagent is not None
    assert profile.general_purpose_subagent.enabled is False


def test_recursion_limit_is_shared_fairness_state_not_a_system_option() -> None:
    first = _config(recursion_limit=25)
    changed = _config(recursion_limit=26)
    assert first.fairness_fingerprint != changed.fairness_fingerprint
    with pytest.raises(ValidationError, match="budget.recursion_limit"):
        _config(system_options={"recursion_limit": 25})


def test_judge_identity_is_fairness_state_and_unknown_is_not_silent() -> None:
    no_judge = _config()
    judge_config = EvaluationJudgeConfig(
        id="fixture-judge",
        version="1",
        config={"rubric": "exact"},
    )
    judged_config = _config(judge=judge_config)
    assert no_judge.fairness_fingerprint != judged_config.fairness_fingerprint
    assert (
        apply_evaluation_judge(
            EvalTask(id="task-hardening", question="question"),
            _result(no_judge),
            no_judge,
        ).judge_score
        is None
    )
    with pytest.raises(JudgeUnavailableError, match="no evaluation judge registered"):
        ensure_judge_available(judged_config)


def test_process_local_judge_protocol_persists_validated_result() -> None:
    class FixtureJudge:
        id = "injected-fixture-judge"
        version = "1"

        def evaluate(
            self,
            task: EvalTask,
            result: RunResult,
            config: EvaluationJudgeConfig,
        ) -> JudgeResult:
            assert task.id == result.task_id
            assert config.config == {"rubric": "fixture"}
            return JudgeResult(
                score=0.75,
                rationale="deterministic fixture score",
                metadata={"mode": "offline"},
            )

    judge = FixtureJudge()
    register_evaluation_judge(judge)
    judge_config = EvaluationJudgeConfig(
        id=judge.id,
        version=judge.version,
        config={"rubric": "fixture"},
    )
    config = _config(judge=judge_config)
    task = EvalTask(id="task-hardening", question="question")
    judged = apply_evaluation_judge(task, _result(config), config, judge=judge)

    assert judged.judge_score == 0.75
    assert judged.judge_result is not None
    assert judged.judge_result.rationale == "deterministic fixture score"
    with pytest.raises(JudgeUnavailableError, match="process-local"):
        ensure_judge_available(config)


def test_model_reservation_refunds_and_unknown_usage_is_conservatively_charged() -> (
    None
):
    budget = ExecutionBudget(_limits())
    reservation = budget.require_model_call(token_reservation=20)
    before = budget.snapshot()
    assert before.total_tokens == 0
    assert before.reserved_tokens == 20
    settled = budget.settle_model_call(reservation, actual_tokens=7)
    assert settled.refunded_tokens == 13
    assert settled.reservation_overrun_tokens == 0
    after = budget.snapshot()
    assert after.total_tokens == 7
    assert after.estimated_token_charges == 0
    assert after.reserved_tokens == 0
    assert after.accounted_tokens == 7

    unknown = budget.require_model_call(token_reservation=11)
    unknown_settlement = budget.settle_model_call(
        unknown,
        actual_tokens=None,
        charge_reservation_if_unknown=True,
    )
    assert unknown_settlement.estimated_tokens_charged == 11
    snapshot = budget.snapshot()
    assert snapshot.total_tokens == 7
    assert snapshot.estimated_token_charges == 11
    assert snapshot.accounted_tokens == 18


def test_zero_token_budget_never_calls_model_handler() -> None:
    limits = _limits().model_copy(update={"max_total_tokens": 0})
    budget = ExecutionBudget(limits)
    middleware = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=1,
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="hello")],
    )
    calls = 0

    def handler(_: ModelRequest[object]) -> ModelResponse[object]:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="must not run")])

    with pytest.raises(BudgetExceeded):
        middleware.wrap_model_call(request, handler)
    assert calls == 0
    snapshot = budget.snapshot()
    assert snapshot.model_calls == 0
    assert snapshot.total_tokens == 0
    assert snapshot.token_budget_exhausted is True


def test_model_usage_overrun_is_recorded_before_budget_exhaustion() -> None:
    limits = _limits().model_copy(update={"max_total_tokens": 10})
    budget = ExecutionBudget(limits)
    middleware = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=1,
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="x")],
    )
    calls = 0

    def handler(_: ModelRequest[object]) -> ModelResponse[object]:
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="overrun",
                    usage_metadata={
                        "input_tokens": 15,
                        "output_tokens": 5,
                        "total_tokens": 20,
                    },
                )
            ]
        )

    with pytest.raises(BudgetExceeded, match="tokens"):
        middleware.wrap_model_call(request, handler)
    assert calls == 1
    assert middleware.token_usage is not None
    assert middleware.token_usage.total_tokens == 20
    snapshot = budget.snapshot()
    assert snapshot.total_tokens == 20
    assert snapshot.accounted_tokens == 20
    assert snapshot.reserved_tokens == 0
    assert snapshot.token_budget_exhausted is True


def test_provider_failure_cancels_without_charging_unknown_usage() -> None:
    budget = ExecutionBudget(_limits())
    middleware = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=2,
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="x")],
    )

    def handler(_: ModelRequest[object]) -> ModelResponse[object]:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        middleware.wrap_model_call(request, handler)
    snapshot = budget.snapshot()
    assert snapshot.model_calls == 1
    assert snapshot.total_tokens == 0
    assert snapshot.estimated_token_charges == 0
    assert snapshot.reserved_tokens == 0


def test_deepagents_summarization_uses_shared_model_and_token_budget(
    tmp_path: Path,
) -> None:
    limits = _limits().model_copy(update={"max_total_tokens": 10_000})
    budget = ExecutionBudget(limits)
    trace = TraceCollector()
    accounting = EvaluationMiddleware(
        budget,
        trace,
        max_output_tokens=8,
    )
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="compact summary",
                usage_metadata={
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            )
        ]
    )
    summarization = build_evaluation_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        accounting,
    )

    summary = summarization._create_summary([HumanMessage(content="history")])

    assert summary == "compact summary"
    snapshot = budget.snapshot()
    assert snapshot.model_calls == 1
    assert snapshot.total_tokens == 10
    assert snapshot.reserved_tokens == 0
    assert accounting.token_usage is not None
    assert accounting.token_usage.total_tokens == 10
    assert any(
        event.event_type == "model_call_started"
        and event.payload["label"] == "deepagents.summarization"
        for event in trace.snapshot()
    )


def test_deepagents_summarization_budget_denial_never_calls_provider(
    tmp_path: Path,
) -> None:
    limits = _limits().model_copy(update={"max_model_calls": 0})
    budget = ExecutionBudget(limits)
    accounting = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=1,
    )
    model = FakeMessagesListChatModel(responses=[AIMessage(content="must not be used")])
    summarization = build_evaluation_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        accounting,
    )

    with pytest.raises(BudgetExceeded):
        summarization._create_summary([HumanMessage(content="history")])

    assert budget.snapshot().model_calls == 0
    assert model.i == 0


def test_fetch_body_is_never_persisted_in_canonical_or_native_trace() -> None:
    body = "short fixture body that must remain model-visible"
    budget = ExecutionBudget(_limits())
    trace = TraceCollector()
    middleware = EvaluationMiddleware(
        budget,
        trace,
        max_output_tokens=1,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "fetch_url",
            "args": {"url": "https://fixture.test/page"},
            "id": "fetch-1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )

    def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "status": "success",
                    "url": "https://fixture.test/page",
                    "source_id": "S1",
                    "content": body,
                }
            ),
            tool_call_id="fetch-1",
            name="fetch_url",
        )

    response = middleware.wrap_tool_call(request, handler)

    assert isinstance(response, ToolMessage)
    assert body in str(response.content)
    persisted_result = middleware.tool_calls[0].result
    assert isinstance(persisted_result, dict)
    body_summary = persisted_result["content"]
    assert isinstance(body_summary, dict)
    assert body_summary["_trace_value"] == "omitted_body_text"
    assert body_summary["chars"] == len(body)
    assert body not in trace.jsonl()


def test_high_budget_boundary_blocks_tool_before_budget_or_handler(
    tmp_path: Path,
) -> None:
    budget = ExecutionBudget(_limits())
    trace = TraceCollector()
    middleware = EvaluationMiddleware(
        budget,
        trace,
        max_output_tokens=8,
        question="fixture question",
        high_budget_finalization=_HighBudgetFinalizationBoundary(
            token_trigger=0,
            wall_time_trigger_seconds=9.0,
            remaining_model_calls_trigger=1,
            artifact_path=tmp_path / "finalization.json",
        ),
    )
    request = ToolCallRequest(
        tool_call={
            "name": "web_search",
            "args": {"query": "must not execute"},
            "id": "search-blocked",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    handler_called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        raise AssertionError("blocked tool handler must not execute")

    response = middleware.wrap_tool_call(request, handler)

    assert isinstance(response, ToolMessage)
    assert response.status == "error"
    assert handler_called is False
    assert budget.snapshot().search_calls == 0
    artifact = json.loads((tmp_path / "finalization.json").read_text())
    assert artifact["finalization_reason"] == "total_tokens"
    assert artifact["blocked_tools_after_finalization"] == ["web_search"]
    assert any(
        event.event_type == "tool_call_blocked_by_high_budget_finalization"
        for event in trace.snapshot()
    )


def test_high_budget_boundary_never_executes_a_second_finalization_model_call(
    tmp_path: Path,
) -> None:
    budget = ExecutionBudget(_limits())
    middleware = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=8,
        question="fixture question",
        high_budget_finalization=_HighBudgetFinalizationBoundary(
            token_trigger=0,
            wall_time_trigger_seconds=9.0,
            remaining_model_calls_trigger=1,
            artifact_path=tmp_path / "finalization-once.json",
        ),
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="fixture question")],
    )
    calls = 0

    def handler(_: ModelRequest[object]) -> ModelResponse[object]:
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="FINAL_ANSWER: fixture",
                    usage_metadata={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                )
            ]
        )

    first = middleware.wrap_model_call(request, handler)
    second = middleware.wrap_model_call(request, handler)

    assert calls == 1
    assert budget.snapshot().model_calls == 1
    assert first.result[0].content == second.result[0].content
