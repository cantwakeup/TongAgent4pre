"""Offline regressions for TongAgent stage caps and context rescue."""

from __future__ import annotations

import json

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

from evaluation import BudgetLimits, ExecutionBudget, TraceCollector
from evaluation.systems.common import (
    EvaluationMiddleware,
    _compact_tongagent_model_request,
    _compact_tool_message_content,
)
from evaluation.token_control import (
    DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS,
    PartitionDenial,
    StageTokenController,
)
from research_graph import _compact_research_history


def _limits() -> BudgetLimits:
    return BudgetLimits(
        max_search_calls=4,
        max_fetch_calls=6,
        max_total_tool_calls=12,
        max_model_calls=20,
        max_total_tokens=50_000,
        wall_time_seconds=60.0,
        max_results_per_search=5,
        max_page_chars=12_000,
        recursion_limit=125,
    )


def test_stage_caps_are_distinct_and_auditable() -> None:
    assert DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS == {
        "planner": 1_000,
        "control_status": 500,
        "research_step": 1_800,
        "evidence_selection": 800,
        "final_synthesis": 2_200,
        "final_extractor": 500,
    }


def test_stage_caps_never_exceed_the_runner_global_output_limit() -> None:
    middleware = EvaluationMiddleware(
        ExecutionBudget(_limits()),
        TraceCollector(),
        max_output_tokens=1_024,
        stage_output_caps=DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS,
        enable_token_partitions=True,
    )

    assert middleware.stage_output_cap("research_step") == 1_024
    assert middleware.stage_output_cap("final_synthesis") == 1_024
    assert middleware.stage_output_cap("control_status") == 500


def test_subquestion_partition_cannot_consume_sibling_or_final_reserve() -> None:
    controller = StageTokenController(total_token_limit=10_000)
    controller.configure(["SQ1", "SQ2"])
    controller.activate("SQ1")

    first = controller.reserve(
        stage="research_step",
        token_reservation=3_000,
    )
    assert not isinstance(first, PartitionDenial)
    controller.settle(
        first,
        actual_tokens=3_000,
        charge_reservation_if_unknown=False,
    )
    denied = controller.reserve(
        stage="research_step",
        token_reservation=200,
    )
    assert isinstance(denied, PartitionDenial)
    assert denied.bucket == "sq:SQ1"

    snapshot = controller.snapshot()
    assert snapshot["buckets"]["sq:SQ2"]["remaining_tokens"] == 3_150
    assert snapshot["buckets"]["final"]["remaining_tokens"] == 2_700
    assert snapshot["buckets"]["buffer"]["remaining_tokens"] == 1_000


def test_two_hop_partition_matches_the_live_budget_contract() -> None:
    controller = StageTokenController(total_token_limit=100_000)
    controller.configure(["SQ1", "SQ2"])

    snapshot = controller.snapshot()

    assert snapshot["buckets"]["sq:SQ1"]["limit_tokens"] == 32_500
    assert snapshot["buckets"]["sq:SQ2"]["limit_tokens"] == 32_500
    assert snapshot["buckets"]["final"]["limit_tokens"] == 25_000
    assert snapshot["buckets"]["buffer"]["limit_tokens"] == 10_000


def test_middleware_reserves_with_the_active_stage_cap() -> None:
    budget = ExecutionBudget(_limits())
    trace = TraceCollector()
    middleware = EvaluationMiddleware(
        budget,
        trace,
        max_output_tokens=5_000,
        stage_output_caps=DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS,
        enable_context_compaction=True,
        enable_token_partitions=True,
    )
    middleware.configure_token_partitions(["SQ1", "SQ2"])
    middleware.activate_token_subquestion("SQ1")
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[
            HumanMessage(
                content=(
                    "[RESEARCH STEP]\n"
                    "ACTIVE RESEARCH CONTEXT:\n"
                    '{"plan_id":"p","canonical_claims":[{"claim_id":"C1"}],'
                    '"relevant_sources":[{"source_id":"S1"}]}\n\n'
                    "Token partition for this SQ: {}"
                ),
                id="research-step-plan-SQ1-0",
            )
        ],
        state={
            "active_subquestion_id": "SQ1",
            "research_plan": {
                "subquestions": [{"id": "SQ1", "question": "Find Alpha."}]
            },
        },
    )
    observed: dict[str, object] = {}

    def handler(effective: ModelRequest[object]) -> ModelResponse[object]:
        observed.update(effective.model_settings)
        return ModelResponse(
            result=[
                AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "total_tokens": 60,
                    },
                )
            ]
        )

    middleware.wrap_model_call(request, handler)

    assert observed["max_tokens"] == 1_800
    events = trace.snapshot()
    started = next(item for item in events if item.event_type == "model_call_started")
    assert started.payload["stage"] == "research_step"
    assert started.payload["max_output_tokens"] == 1_800
    assert started.payload["active_subquestion_id"] == "SQ1"
    assert started.payload["context"]["active_context_chars"] > 0
    assert started.payload["context"]["active_context_claim_count"] == 1
    assert started.payload["context"]["active_context_source_count"] == 1
    assert middleware.orchestration_snapshot()["model_calls_per_subquestion"] == {
        "SQ1": 1
    }
    partition = middleware.token_partition_snapshot()
    assert partition["stage_usage"]["research_step"]["actual_tokens"] == 60


def test_context_compaction_keeps_checkpoint_state_and_only_current_sq_messages() -> (
    None
):
    old_page = "old page text " * 800
    current_page = "current fact sentence. " * 600
    messages = [
        HumanMessage(
            content="[RESEARCH STEP]\nActive subquestion: SQ1",
            id="research-step-plan-SQ1-0",
        ),
        ToolMessage(
            content=json.dumps({"status": "success", "content": old_page}),
            tool_call_id="old-fetch",
            name="fetch_url",
        ),
        HumanMessage(
            content="[RESEARCH STEP]\nActive subquestion: SQ2",
            id="research-step-plan-SQ2-0",
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "success",
                    "source_id": "S2",
                    "content": current_page,
                    "content_chars": len(current_page),
                }
            ),
            tool_call_id="current-fetch",
            name="fetch_url",
        ),
    ]
    state = {
        "active_subquestion_id": "SQ2",
        "research_plan": {
            "subquestions": [
                {"id": "SQ1", "question": "Find the first fact."},
                {"id": "SQ2", "question": "Find the current fact."},
            ]
        },
    }
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=messages,
        state=state,
    )

    compact = _compact_tongagent_model_request(request)

    assert len(request.messages) == 4
    assert len(compact.messages) == 2
    assert compact.messages[0].id == "research-step-plan-SQ2-0"
    payload = json.loads(str(compact.messages[1].content))
    assert payload["content_compacted_for_model"] is True
    assert len(payload["content"]) <= 3_500
    assert old_page not in json.dumps([item.content for item in compact.messages])


def test_research_history_summary_is_fixed_length_and_delta_only() -> None:
    state = {
        "research_events": [
            {
                "event": f"event-{index}",
                "subquestion_id": "SQ1",
                "details": {"large": "x" * 2_000},
            }
            for index in range(20)
        ],
        "adaptive_control": {
            "decision_history": [
                {
                    "action": "continue",
                    "reason_codes": ["claim_gap", "search_retry_needed"],
                    "large": "y" * 2_000,
                }
            ]
        },
        "compact_checkpoints": [{"cycle": index} for index in range(3)],
    }

    summary = _compact_research_history(state)  # type: ignore[arg-type]

    assert len(summary) <= 480
    assert '"event_count":20' in summary
    assert '"event":"event-19"' in summary
    assert '"action":"continue"' in summary
    assert "x" * 100 not in summary
    assert "y" * 100 not in summary


def test_orchestration_metrics_count_read_roundtrips_and_duplicate_retries() -> None:
    middleware = EvaluationMiddleware(
        ExecutionBudget(_limits()),
        TraceCollector(),
        max_output_tokens=1_024,
    )

    def call_tool(name: str, payload: dict[str, object]) -> None:
        request = ToolCallRequest(
            tool_call={
                "name": name,
                "args": {},
                "id": f"{name}-call",
                "type": "tool_call",
            },
            tool=None,
            state={},
            runtime=None,  # type: ignore[arg-type]
        )
        middleware.wrap_tool_call(
            request,
            lambda _: ToolMessage(
                content=json.dumps(payload),
                tool_call_id=f"{name}-call",
                name=name,
                status=(
                    "error"
                    if payload.get("status") == "duplicate_retry_blocked"
                    else "success"
                ),
            ),
        )

    call_tool("get_source_ledger", {"status": "success"})
    model_request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="continue")],
        state={"active_subquestion_id": "SQ1"},
    )
    middleware.wrap_model_call(
        model_request,
        lambda _: ModelResponse(
            result=[
                AIMessage(
                    content="continue",
                    usage_metadata={
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "total_tokens": 7,
                    },
                )
            ]
        ),
    )
    call_tool(
        "record_evidence",
        {"status": "duplicate_retry_blocked", "retryable": False},
    )

    metrics = middleware.orchestration_snapshot()
    assert metrics["read_only_state_tool_calls"] == 1
    assert metrics["avoidable_model_calls"] == 1
    assert metrics["duplicate_tool_retries"] == 1


def test_two_sq_normal_orchestration_budget_is_four_calls_each() -> None:
    middleware = EvaluationMiddleware(
        ExecutionBudget(_limits()),
        TraceCollector(),
        max_output_tokens=1_024,
    )
    sequence = 0

    def model_call(subquestion_id: str) -> None:
        nonlocal sequence
        sequence += 1
        request = ModelRequest(
            model=FakeListChatModel(responses=["unused"]),
            messages=[
                HumanMessage(
                    content=f"[RESEARCH STEP]\n{subquestion_id}",
                    id=f"research-step-plan-{subquestion_id}-1",
                )
            ],
            state={"active_subquestion_id": subquestion_id},
        )
        middleware.wrap_model_call(
            request,
            lambda _: ModelResponse(
                result=[
                    AIMessage(
                        content=f"step {sequence}",
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 2,
                            "total_tokens": 7,
                        },
                    )
                ]
            ),
        )

    def tool_call(name: str, subquestion_id: str) -> None:
        call_id = f"{subquestion_id}-{name}"
        request = ToolCallRequest(
            tool_call={
                "name": name,
                "args": {},
                "id": call_id,
                "type": "tool_call",
            },
            tool=None,
            state={"active_subquestion_id": subquestion_id},
            runtime=None,  # type: ignore[arg-type]
        )
        middleware.wrap_tool_call(
            request,
            lambda _: ToolMessage(
                content=json.dumps({"status": "success"}),
                tool_call_id=call_id,
                name=name,
            ),
        )

    for subquestion_id in ("SQ1", "SQ2"):
        model_call(subquestion_id)
        for tool_name in ("web_search", "fetch_url", "record_evidence"):
            tool_call(tool_name, subquestion_id)
            model_call(subquestion_id)

    metrics = middleware.orchestration_snapshot()
    assert metrics["model_calls_per_subquestion"] == {"SQ1": 4, "SQ2": 4}
    assert metrics["read_only_state_tool_calls"] == 0
    assert metrics["avoidable_model_calls"] == 0


def test_ledger_compaction_uses_fresh_tool_delta_not_stale_outer_state() -> None:
    content = json.dumps(
        {
            "active_subquestion_id": "SQ2",
            "successful_sources": [
                {
                    "source_id": "S2",
                    "title": "Current source",
                    "url": "https://current.invalid/source",
                    "evidence_quality": "full",
                    "acquisition_method": "direct_http",
                }
            ],
            "subquestion_limits": {"max_searches": 2, "max_fetches": 3},
            "subquestion_usage": {"search_calls": 1, "fetch_calls": 1},
            "evidence_graph_counts": {
                "claims": 1,
                "evidence_units": 1,
                "conflicts": 0,
            },
        }
    )

    compact = _compact_tool_message_content(
        "get_source_ledger",
        content,
        state={
            "active_subquestion_id": "SQ2",
            "budget_state": {"successful_sources": []},
        },
        active_question="Find the current fact.",
    )
    payload = json.loads(str(compact))

    assert payload["successful_sources"][0]["source_id"] == "S2"
    assert payload["subquestion_usage"]["fetch_calls"] == 1
    assert payload["evidence_graph_counts"]["claims"] == 1


def test_incomplete_required_subquestion_abstains_without_model_guess() -> None:
    budget = ExecutionBudget(_limits())
    middleware = EvaluationMiddleware(
        budget,
        TraceCollector(),
        max_output_tokens=5_000,
        reserve_final_synthesis=True,
        stage_output_caps=DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS,
        enable_token_partitions=True,
    )
    middleware.configure_token_partitions(["SQ1", "SQ2"])
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="FINAL_ANSWER: 117",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )
        ]
    )

    answer = middleware.finalize_answer(
        model,
        question="What is the derived number?",
        draft="One source supports only SQ1. The unsupported final guess is 117.",
        required_evidence_complete=False,
        missing_subquestions=["SQ2: establish the second required fact"],
    )

    assert "INSUFFICIENT_EVIDENCE" in answer
    assert "SQ2" in answer
    assert answer.endswith("FINAL_ANSWER: ABSTAIN")
    assert "117" not in answer
    assert model.i == 0
    assert budget.snapshot().total_tokens == 0
    assert budget.snapshot().outstanding_model_reservations == 0
    assert (
        middleware.token_partition_snapshot()["buckets"]["final"]["reserved_tokens"]
        == 0
    )
