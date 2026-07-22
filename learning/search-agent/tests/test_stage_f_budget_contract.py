"""Offline regressions for Stage F budget and final-answer contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent_policy import EffortPolicy
from evaluation import (
    AnswerStatus,
    BudgetExceeded,
    BudgetLimits,
    BudgetResource,
    CompletionStatus,
    ExecutionBudget,
    FailureType,
    RunResult,
    TraceCollector,
)
from evaluation.schema import extract_answer_contract, normalized_exact_match
from evaluation.systems.common import (
    EvaluationMiddleware,
    _completion_and_failure,
    _semantic_tool_failure,
    _terminal_budget_observation,
)
from search_agent import ResearchBudget, build_budgeted_tools


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def _limits(**overrides: int | float) -> BudgetLimits:
    values: dict[str, int | float] = {
        "max_search_calls": 2,
        "max_fetch_calls": 2,
        "max_total_tool_calls": 3,
        "max_model_calls": 3,
        "max_total_tokens": 10_000,
        "wall_time_seconds": 10.0,
        "max_results_per_search": 3,
        "max_page_chars": 1_000,
        "recursion_limit": 3,
    }
    values.update(overrides)
    return BudgetLimits.model_validate(values)


def _representative_payload() -> dict[str, object]:
    project = Path(__file__).resolve().parents[1]
    return json.loads(
        (
            project / "evaluation/results/stage_d_offline_smoke/representative_run.json"
        ).read_text(encoding="utf-8")
    )["run_result"]


def _v2_completed_payload() -> dict[str, object]:
    payload = _representative_payload()
    payload.update(
        {
            "final_answer": "FINAL_ANSWER: Paris",
            "answer_status": "answer",
            "extracted_answer": "Paris",
            "external_retrieval_calls": 1,
            "internal_tool_calls": 0,
            "budget_accounting_version": 2,
        }
    )
    return payload


def test_budget_resources_are_stable_and_complete() -> None:
    assert {item.value for item in BudgetResource} == {
        "search",
        "fetch",
        "total_tool",
        "model_call",
        "token",
        "deadline",
        "recursion",
        "subquestion_slice",
    }


def test_internal_tools_do_not_consume_external_retrieval_quota() -> None:
    budget = ExecutionBudget(
        _limits(
            max_search_calls=1,
            max_fetch_calls=1,
            max_total_tool_calls=2,
            recursion_limit=3,
        )
    )

    budget.require_tool("web_search")
    budget.require_tool("get_research_plan")
    budget.require_tool("fetch_url")
    budget.require_tool("get_evidence_graph")
    snapshot = budget.snapshot()

    assert snapshot.search_calls == 1
    assert snapshot.fetch_calls == 1
    assert snapshot.external_retrieval_calls == 2
    assert snapshot.internal_tool_calls == 2
    assert snapshot.total_tool_calls == 4
    assert snapshot.remaining_external_retrieval_calls == 0
    with pytest.raises(BudgetExceeded) as search_denial:
        budget.require_tool("web_search")
    assert search_denial.value.resource == BudgetResource.SEARCH
    assert search_denial.value.snapshot is not None
    assert search_denial.value.snapshot.external_retrieval_calls == 2

    budget.require_tool("update_subquestion")
    with pytest.raises(BudgetExceeded) as loop_denial:
        budget.require_tool("get_source_ledger")
    assert loop_denial.value.resource == BudgetResource.RECURSION
    assert loop_denial.value.attempted["scope"] == "internal_tool_loop"


def test_subquestion_slice_denial_never_reaches_external_guard() -> None:
    policy = EffortPolicy(
        name="low",
        max_searches=2,
        max_fetches=2,
        min_successful_sources=0,
        max_results_per_search=3,
        max_chars_per_page=1_000,
        max_output_tokens=64,
        max_subquestions=2,
        require_reviewer=False,
    )
    research = ResearchBudget(policy)
    research.configure_subquestions(["SQ1", "SQ2"])
    research.activate_subquestion("SQ1")
    execution = ExecutionBudget(
        _limits(max_search_calls=2, max_fetch_calls=2, max_total_tool_calls=3)
    )
    provider_calls = 0
    guard_calls = 0

    @tool("web_search")
    def raw_search(query: str, max_results: int = 3) -> str:
        """Return one deterministic search result."""

        nonlocal provider_calls
        provider_calls += 1
        return json.dumps(
            {
                "status": "success",
                "results": [
                    {
                        "title": query,
                        "url": "https://fixture.test/result",
                        "snippet": f"{query} verified",
                        "relevance_score": 100,
                    }
                ][:max_results],
            }
        )

    @tool("fetch_url")
    def raw_fetch(url: str, max_chars: int = 1_000) -> str:
        """Unused deterministic fetch provider."""

        del url, max_chars
        return json.dumps({"status": "error", "error": "unused"})

    def guard(tool_name: str) -> None:
        nonlocal guard_calls
        guard_calls += 1
        execution.require_tool(tool_name)
        return None

    tools, _ = build_budgeted_tools(
        policy,
        raw_search_tool=raw_search,
        raw_fetch_tool=raw_fetch,
        budget=research,
        external_guard=guard,
    )
    search = next(item for item in tools if item.name == "web_search")

    first = json.loads(search.invoke({"query": "alpha entity 2026", "max_results": 3}))
    second = json.loads(
        search.invoke({"query": "alpha entity retry", "max_results": 3})
    )

    assert first["provider_success"] is True
    assert second["reason"] == "subquestion_budget_exceeded"
    assert provider_calls == 1
    assert guard_calls == 1
    assert execution.snapshot().external_retrieval_calls == 1


def test_model_token_deadline_and_total_tool_denials_are_exact() -> None:
    total = ExecutionBudget(
        _limits(
            max_search_calls=2,
            max_fetch_calls=2,
            max_total_tool_calls=1,
        )
    )
    total.require_tool("web_search")
    with pytest.raises(BudgetExceeded) as total_denial:
        total.require_tool("fetch_url")
    assert total_denial.value.resource == BudgetResource.TOTAL_TOOL

    token = ExecutionBudget(_limits(max_total_tokens=10))
    with pytest.raises(BudgetExceeded) as token_denial:
        token.require_model_call(token_reservation=11)
    assert token_denial.value.resource == BudgetResource.TOKEN
    assert token_denial.value.attempted == {
        "token_reservation": 11,
        "available_tokens": 10,
    }

    model = ExecutionBudget(_limits(max_model_calls=1))
    reservation = model.require_model_call(token_reservation=2)
    model.settle_model_call(reservation, actual_tokens=1)
    with pytest.raises(BudgetExceeded) as model_denial:
        model.require_model_call(token_reservation=2)
    assert model_denial.value.resource == BudgetResource.MODEL_CALL

    clock = FakeClock()
    deadline = ExecutionBudget(_limits(wall_time_seconds=1.0), clock=clock)
    clock.now = 11.0
    with pytest.raises(BudgetExceeded) as deadline_denial:
        deadline.require_tool("web_search")
    assert deadline_denial.value.resource == BudgetResource.DEADLINE
    assert deadline_denial.value.snapshot is not None
    assert deadline_denial.value.snapshot.deadline_exceeded is True


@pytest.mark.parametrize(
    ("taxonomy", "expected"),
    [
        ("access_blocked", "access_blocked"),
        ("rate_limited", "rate_limited"),
        ("network_timeout", "network_timeout"),
        ("dns_rejected", "dns_rejected"),
        ("dns_rebinding", "security_rejected"),
        ("redirect_rejected", "security_rejected"),
        ("security_rejected", "security_rejected"),
    ],
)
def test_fetch_failure_taxonomy_is_not_collapsed(
    taxonomy: str,
    expected: str,
) -> None:
    failure = _semantic_tool_failure(
        "fetch_url",
        {
            "status": "error",
            "error": taxonomy,
            "failure_taxonomy": taxonomy,
            "retryable": taxonomy in {"rate_limited", "network_timeout"},
        },
    )

    assert failure is not None
    assert failure.failure_type.value == expected
    assert failure.details["failure_taxonomy"] == taxonomy


def test_subquestion_slice_is_a_structured_nonterminal_tool_denial() -> None:
    failure = _semantic_tool_failure(
        "web_search",
        {
            "status": "budget_exceeded",
            "reason": "subquestion_budget_exceeded",
            "active_subquestion_id": "SQ1",
            "subquestion_limits": {"max_searches": 1},
            "subquestion_usage": {"search_calls": 1},
        },
    )

    assert failure is not None
    assert failure.failure_type.value == "budget_exhausted"
    assert failure.details["budget_resource"] == "subquestion_slice"
    assert failure.details["active_subquestion_id"] == "SQ1"


def test_final_synthesis_reservation_survives_general_call_denial() -> None:
    budget = ExecutionBudget(_limits(max_model_calls=1, max_total_tokens=10_000))
    middleware = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=64,
        reserve_final_synthesis=True,
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="research step")],
    )
    handler_calls = 0

    def handler(_: ModelRequest[object]) -> ModelResponse[object]:
        nonlocal handler_calls
        handler_calls += 1
        return ModelResponse(result=[AIMessage(content="unused")])

    with pytest.raises(BudgetExceeded) as denied:
        middleware.wrap_model_call(request, handler)
    assert denied.value.resource == BudgetResource.MODEL_CALL
    assert handler_calls == 0

    final_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="FINAL_ANSWER: 28",
                usage_metadata={
                    "input_tokens": 40,
                    "output_tokens": 8,
                    "total_tokens": 48,
                },
            )
        ]
    )
    final = middleware.finalize_answer(
        final_model,
        question="How many?",
        draft="Partial evidence supports a numerical calculation.",
    )
    status, answer = extract_answer_contract(final)

    assert status == AnswerStatus.ANSWER
    assert answer == "28"
    snapshot = budget.snapshot()
    assert snapshot.model_calls == 1
    assert snapshot.outstanding_model_reservations == 0
    assert snapshot.reserved_tokens == 0


def test_final_answer_contract_separates_answer_abstain_and_empty() -> None:
    assert extract_answer_contract("FINAL_ANSWER: Halifax") == (
        AnswerStatus.ANSWER,
        "Halifax",
    )
    assert extract_answer_contract("I could not browse.\n\nFINAL_ANSWER: ABSTAIN") == (
        AnswerStatus.ABSTAIN,
        None,
    )
    assert extract_answer_contract("I could not browse.") == (
        AnswerStatus.EMPTY,
        None,
    )
    assert normalized_exact_match(
        "Long explanation.\nFINAL_ANSWER: Halifax",
        "Halifax",
    )
    assert not normalized_exact_match(
        "I cannot answer because browsing failed.",
        "Halifax",
    )


def test_persisted_answer_derivatives_must_match_the_contract() -> None:
    payload = _v2_completed_payload()
    payload["answer_status"] = "abstain"
    payload["extracted_answer"] = "WRONG"

    with pytest.raises(ValueError, match="answer_status"):
        RunResult.model_validate_json(json.dumps(payload))


def test_v2_result_rejects_nonterminal_budget_fields_and_snapshot_drift() -> None:
    payload = _v2_completed_payload()
    execution = ExecutionBudget(
        BudgetLimits.model_validate(payload["resolved_config"]["budget"])
    )
    execution.require_tool("web_search")
    snapshot = execution.snapshot().model_dump(mode="json")
    payload["budget_resource"] = "token"
    payload["budget_snapshot"] = snapshot

    with pytest.raises(ValueError, match="only valid"):
        RunResult.model_validate_json(json.dumps(payload))

    payload["completion_status"] = "budget_exhausted"
    payload["failure_type"] = "budget_exhausted"
    payload["failure"] = {
        "failure_type": "budget_exhausted",
        "message": "token reservation denied",
        "stage": "model",
        "retryable": False,
        "details": {"budget_resource": "token"},
    }
    snapshot["search_calls"] = 0
    snapshot["external_retrieval_calls"] = 0
    snapshot["total_tool_calls"] = 0
    snapshot["remaining_search_calls"] += 1
    snapshot["remaining_external_retrieval_calls"] += 1
    snapshot["remaining_total_tool_calls"] += 1
    payload["budget_snapshot"] = snapshot

    with pytest.raises(ValueError, match="denial-time snapshot"):
        RunResult.model_validate_json(json.dumps(payload))


def test_budget_completion_failure_and_resource_taxonomy_is_closed() -> None:
    payload = _v2_completed_payload()
    execution = ExecutionBudget(
        BudgetLimits.model_validate(payload["resolved_config"]["budget"])
    )
    execution.require_tool("web_search")
    payload.update(
        {
            "completion_status": "budget_exhausted",
            "failure_type": "model_error",
            "failure": {
                "failure_type": "model_error",
                "message": "wrong terminal taxonomy",
                "stage": "model",
                "retryable": False,
                "details": {},
            },
            "budget_resource": "deadline",
            "budget_snapshot": execution.snapshot().model_dump(mode="json"),
        }
    )

    with pytest.raises(ValueError, match="budget_exhausted failure_type"):
        RunResult.model_validate_json(json.dumps(payload))


def test_nonbudget_failure_ignores_historical_middleware_denial() -> None:
    execution = ExecutionBudget(_limits())
    snapshot = execution.snapshot()
    historical = BudgetExceeded(
        BudgetResource.SEARCH,
        snapshot=snapshot,
        attempted={"tool_name": "web_search"},
    )
    runtime = SimpleNamespace(middleware=SimpleNamespace(budget_failure=historical))

    resource, denied_at = _terminal_budget_observation(
        completion_status=CompletionStatus.FAILED,
        failure=None,
        caught=RuntimeError("later model failure"),
        runtime=runtime,
        execution_snapshot=snapshot,
    )

    assert resource is None
    assert denied_at is None


def test_provider_timeout_is_not_mislabeled_as_wall_deadline() -> None:
    execution = ExecutionBudget(_limits())
    snapshot = execution.snapshot()
    status, failure = _completion_and_failure(
        caught=TimeoutError("provider timed out"),
        final_answer="FINAL_ANSWER: ABSTAIN",
        runtime=None,
        deadline_exceeded=False,
    )

    assert status == CompletionStatus.FAILED
    assert failure is not None
    assert failure.failure_type == FailureType.RUNNER_ERROR
    resource, denied_at = _terminal_budget_observation(
        completion_status=status,
        failure=failure,
        caught=TimeoutError("provider timed out"),
        runtime=None,
        execution_snapshot=snapshot,
    )
    assert resource is None
    assert denied_at is None


def test_v2_timed_out_requires_a_deadline_exceeded_snapshot() -> None:
    payload = _v2_completed_payload()
    execution = ExecutionBudget(
        BudgetLimits.model_validate(payload["resolved_config"]["budget"])
    )
    execution.require_tool("web_search")
    payload.update(
        {
            "completion_status": "timed_out",
            "failure_type": "deadline_exceeded",
            "failure": {
                "failure_type": "deadline_exceeded",
                "message": "claimed deadline",
                "stage": "run",
                "retryable": True,
                "details": {},
            },
            "budget_resource": "deadline",
            "budget_snapshot": execution.snapshot().model_dump(mode="json"),
        }
    )

    with pytest.raises(ValueError, match="deadline-exceeded snapshot"):
        RunResult.model_validate_json(json.dumps(payload))


def test_schema_v1_representative_result_remains_loadable() -> None:
    payload = _representative_payload()

    result = RunResult.model_validate_json(json.dumps(payload))

    assert result.budget_accounting_version is None
    assert result.external_retrieval_calls is None
    assert result.internal_tool_calls is None


def test_legacy_tool_denial_without_failure_is_versioned_compatibility_only() -> None:
    payload = _representative_payload()
    payload["tool_calls"][0]["status"] = "budget_exceeded"
    payload["tool_calls"][0]["failure"] = None

    legacy = RunResult.model_validate_json(json.dumps(payload))
    assert legacy.budget_accounting_version is None

    payload = _v2_completed_payload()
    payload["tool_calls"][0]["status"] = "budget_exceeded"
    payload["tool_calls"][0]["failure"] = None
    with pytest.raises(ValueError, match="v2 budget-exceeded"):
        RunResult.model_validate_json(json.dumps(payload))
