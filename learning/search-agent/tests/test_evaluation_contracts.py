"""Contract tests for the shared benchmark evaluation layer."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from evaluation import (
    BudgetExceeded,
    BudgetLimits,
    Citation,
    CompletionStatus,
    EvalTask,
    ExecutionBudget,
    FailureDetail,
    FailureType,
    ResolvedConfig,
    RunResult,
    SystemRunner,
    TokenUsage,
    ToolCall,
    ToolCallStatus,
    TraceCollector,
    normalized_exact_match,
    parse_eval_task_jsonl_line,
    raw_whole_string_exact_match,
    standard_normalized_exact_match,
    strict_answer_rate,
)


class FakeClock:
    """Small mutable monotonic clock used to test exact deadline boundaries."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_limits(**overrides: int | float) -> BudgetLimits:
    """Return compact shared limits with optional test overrides."""
    values: dict[str, int | float] = {
        "max_search_calls": 2,
        "max_fetch_calls": 2,
        "max_total_tool_calls": 4,
        "max_model_calls": 3,
        "max_total_tokens": 100,
        "wall_time_seconds": 10.0,
        "max_results_per_search": 3,
        "max_page_chars": 20,
    }
    values.update(overrides)
    return BudgetLimits.model_validate(values)


def make_config(
    *,
    system_id: str = "simple_react",
    artifact_directory: str = "/tmp/eval/simple/task-1/attempt-1",
    backend_kind: str = "fixture",
    fixture_revision: str | None = "fixtures-v1",
    dataset_digest: str = "sha256:dataset",
    system_options: dict[str, object] | None = None,
    limits: BudgetLimits | None = None,
) -> ResolvedConfig:
    """Return a fully resolved deterministic fixture configuration."""
    return ResolvedConfig.model_validate(
        {
            "system_id": system_id,
            "dataset_digest": dataset_digest,
            "backend_kind": backend_kind,
            "fixture_revision": fixture_revision,
            "model": {
                "provider": "fixture",
                "name": "deterministic-chat-v1",
                "temperature": 0.0,
                "max_output_tokens": 64,
                "parameters": {"stop": ["DONE"], "alpha": 1},
            },
            "tools": {
                "search_backend": "shared-fixture-search",
                "fetch_backend": "shared-fixture-fetch",
                "parameters": {"revision": 1},
            },
            "budget": (limits or make_limits()).model_dump(),
            "seed": 7,
            "system_options": system_options or {},
            "artifact_directory": artifact_directory,
        }
    )


def test_eval_task_strict_jsonl_roundtrip_and_rejects_unsafe_records() -> None:
    task = parse_eval_task_jsonl_line(
        json.dumps(
            {
                "id": "two-source_01",
                "question": "Which statement is supported?",
                "reference_answer": None,
                "metadata": {"tags": ["fixture", "two-source"], "weight": 1},
            }
        ),
        line_number=4,
    )

    assert EvalTask.model_validate_json(task.model_dump_json()) == task
    assert task.metadata["weight"] == 1
    with pytest.raises(ValidationError):
        EvalTask.model_validate_json(
            '{"id":1,"question":"q","reference_answer":null,"metadata":{}}'
        )
    with pytest.raises(ValidationError):
        EvalTask.model_validate(
            {
                "id": "../escape",
                "question": "q",
                "reference_answer": None,
                "metadata": {},
            }
        )
    with pytest.raises(ValidationError):
        EvalTask.model_validate(
            {
                "id": "task",
                "question": "q",
                "reference_answer": None,
                "metadata": {},
                "unexpected": True,
            }
        )
    with pytest.raises(ValueError, match="line 9"):
        parse_eval_task_jsonl_line("   ", line_number=9)


def test_normalized_exact_match_is_unicode_whitespace_whole_string_only() -> None:
    assert normalized_exact_match("  ＣＡＦÉ\t答案\n", "café 答案") is True
    assert normalized_exact_match("The answer is Paris.", "Paris") is False
    assert normalized_exact_match(None, "Paris") is False
    assert normalized_exact_match("anything", None) is None


def test_frozen_standard_scoring_is_gold_independent_and_whole_string_only() -> None:
    prediction = "FINAL_ANSWER: The ＰＡＲＩＳ!"
    assert raw_whole_string_exact_match(prediction, "Paris") is False
    assert standard_normalized_exact_match(prediction, "Paris") is True
    assert strict_answer_rate(prediction) is True
    assert (
        standard_normalized_exact_match(
            "FINAL_ANSWER: The answer is Paris",
            "Paris",
        )
        is False
    )
    assert standard_normalized_exact_match("FINAL_ANSWER: Paris", None) is None
    assert strict_answer_rate("Paris") is False
    assert strict_answer_rate("FINAL_ANSWER: ABSTAIN") is False


def test_config_fingerprints_are_stable_complete_and_fair() -> None:
    first = make_config(
        system_id="simple_react",
        system_options={"react": {"max_steps": 4}, "order": [2, 1]},
    )
    reordered = ResolvedConfig.model_validate(
        {
            **first.model_dump(
                exclude={
                    "config_fingerprint",
                    "fairness_fingerprint",
                    "system_options",
                    "model",
                }
            ),
            "model": {
                **first.model.model_dump(exclude={"parameters"}),
                "parameters": {"alpha": 1, "stop": ["DONE"]},
            },
            "system_options": {"order": [2, 1], "react": {"max_steps": 4}},
        }
    )
    other_system = make_config(
        system_id="tongagent",
        artifact_directory="/another/location/attempt-9",
        system_options={"strategy": "adaptive", "topology": "single"},
    )

    assert first.config_fingerprint == reordered.config_fingerprint
    assert first.fairness_fingerprint == reordered.fairness_fingerprint
    assert first.config_fingerprint != other_system.config_fingerprint
    assert first.fairness_fingerprint == other_system.fairness_fingerprint

    relocated = first.model_copy(
        update={"artifact_directory": "/new/output/root/task/attempt-2"}
    )
    relocated = ResolvedConfig.model_validate(
        relocated.model_dump(exclude={"config_fingerprint", "fairness_fingerprint"})
    )
    assert relocated.config_fingerprint == first.config_fingerprint
    assert relocated.fairness_fingerprint == first.fairness_fingerprint

    changed_dataset = make_config(dataset_digest="sha256:different")
    changed_backend = make_config(
        backend_kind="live",
        fixture_revision=None,
    )
    changed_budget = make_config(
        limits=make_limits(max_search_calls=1),
    )
    assert changed_dataset.fairness_fingerprint != first.fairness_fingerprint
    assert changed_backend.fairness_fingerprint != first.fairness_fingerprint
    assert changed_budget.fairness_fingerprint != first.fairness_fingerprint

    with pytest.raises(ValidationError):
        ResolvedConfig.model_validate(
            {
                **first.model_dump(),
                "output_directory": "/must/not/enter/config",
            }
        )


def test_persisted_fingerprint_tampering_is_rejected() -> None:
    config = make_config()
    payload = config.model_dump()
    payload["config_fingerprint"] = "sha256:not-the-config"

    with pytest.raises(ValidationError, match="config_fingerprint"):
        ResolvedConfig.model_validate(payload)


def test_run_result_json_roundtrip_retains_all_explicit_null_metrics() -> None:
    config = make_config()
    started = datetime(2026, 7, 19, 0, 0, tzinfo=UTC)
    result = RunResult(
        run_id="run-001",
        task_id="task-001",
        system_id=config.system_id,
        git_sha="deadbeef",
        resolved_config=config,
        config_fingerprint=config.config_fingerprint,
        fairness_fingerprint=config.fairness_fingerprint,
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        wall_time_seconds=1.0,
        final_answer="Fixture answer.",
        citations=[
            Citation(
                citation_id="C1",
                url="https://fixture.test/source",
                source_id=None,
                title=None,
                quote=None,
                claim=None,
            )
        ],
        tool_calls=[
            ToolCall(
                call_id="TC1",
                tool_name="web_search",
                arguments={"query": "fixture"},
                started_at=started,
                finished_at=started + timedelta(milliseconds=10),
                duration_seconds=0.01,
                status=ToolCallStatus.SUCCESS,
                result={"result_count": 1},
            )
        ],
        search_calls=1,
        fetch_calls=0,
        relevant_searches=1,
        evidence_count=None,
        structural_subquestion_coverage=None,
        token_usage=None,
        estimated_cost=None,
        completion_status=CompletionStatus.COMPLETED,
        failure_type=None,
        failure=None,
        artifact_directory=config.artifact_directory,
        fixture_smoke=True,
        normalized_exact_match=None,
        judge_score=None,
    )

    payload = json.loads(result.model_dump_json())
    for name in {
        "evidence_count",
        "structural_subquestion_coverage",
        "token_usage",
        "estimated_cost",
        "failure_type",
        "failure",
        "normalized_exact_match",
        "judge_score",
    }:
        assert name in payload
        assert payload[name] is None
    assert RunResult.model_validate_json(result.model_dump_json()) == result

    unavailable = result.model_copy(
        update={
            "search_calls": None,
            "fetch_calls": None,
            "relevant_searches": None,
        }
    )
    unavailable_payload = json.loads(unavailable.model_dump_json())
    assert unavailable_payload["search_calls"] is None
    assert unavailable_payload["fetch_calls"] is None
    assert unavailable_payload["relevant_searches"] is None
    assert RunResult.model_validate_json(unavailable.model_dump_json()) == unavailable

    with pytest.raises(ValueError, match="relevant_searches"):
        RunResult.model_validate_json(
            json.dumps(
                {
                    **json.loads(result.model_dump_json()),
                    "search_calls": 0,
                    "relevant_searches": 1,
                }
            )
        )


def test_failure_and_token_schemas_reject_ambiguous_states() -> None:
    usage = TokenUsage(
        input_tokens=4,
        output_tokens=3,
        total_tokens=7,
        cached_input_tokens=0,
        reasoning_tokens=None,
    )
    assert TokenUsage.model_validate_json(usage.model_dump_json()) == usage
    with pytest.raises(ValidationError, match="unknown token usage"):
        TokenUsage()
    with pytest.raises(ValidationError, match="total_tokens"):
        TokenUsage(input_tokens=4, output_tokens=3, total_tokens=6)

    failure = FailureDetail(
        failure_type=FailureType.SEARCH_ERROR,
        message="fixture search failed",
        stage="search",
        retryable=True,
    )
    with pytest.raises(ValidationError, match="non-error"):
        ToolCall(
            call_id="TC1",
            tool_name="web_search",
            started_at=datetime.now(UTC),
            status=ToolCallStatus.SUCCESS,
            failure=failure,
        )


def test_execution_budget_enforces_exact_boundaries_monotonically() -> None:
    clock = FakeClock()
    budget = ExecutionBudget(
        make_limits(
            max_search_calls=1,
            max_fetch_calls=1,
            max_total_tool_calls=2,
            max_model_calls=1,
            max_total_tokens=5,
            wall_time_seconds=5.0,
            max_results_per_search=2,
            max_page_chars=4,
        ),
        clock=clock,
    )

    assert budget.try_reserve_tool("web_search") is True
    assert budget.try_reserve_tool("search") is False
    assert budget.try_reserve_tool("fetch_url") is True
    assert budget.try_reserve_tool("other") is False
    reservation = budget.try_reserve_model_call(token_reservation=3)
    assert reservation is not None
    assert budget.try_reserve_model_call(token_reservation=1) is None
    settlement = budget.settle_model_call(reservation, actual_tokens=3)
    assert settlement.refunded_tokens == 0
    assert budget.try_reserve_tokens(2) is True
    assert budget.try_reserve_tokens(1) is False
    assert budget.limit_search_results([1, 2, 3]) == [1, 2]
    assert budget.limit_page_content("abcdef") == ("abcd", True)
    assert budget.limit_page_content("abc") == ("abc", False)

    before_deadline = budget.snapshot()
    assert before_deadline.total_tool_calls == 2
    assert before_deadline.total_tokens == 5
    clock.now = 104.999
    assert budget.deadline_exceeded() is False
    clock.now = 105.0
    assert budget.deadline_exceeded() is True
    assert budget.try_reserve_tokens(0) is False
    after_deadline = budget.snapshot()
    assert after_deadline.total_tokens == before_deadline.total_tokens

    with pytest.raises(BudgetExceeded):
        budget.require_tool("web_search")
    with pytest.raises(ValueError):
        budget.try_reserve_tokens(-1)


def test_execution_budget_reservations_are_thread_safe() -> None:
    budget = ExecutionBudget(
        make_limits(
            max_search_calls=10,
            max_total_tool_calls=10,
            wall_time_seconds=100.0,
        ),
        clock=lambda: 10.0,
    )

    with ThreadPoolExecutor(max_workers=16) as pool:
        reservations = list(
            pool.map(lambda _: budget.try_reserve_tool("web_search"), range(100))
        )

    assert sum(reservations) == 10
    snapshot = budget.snapshot()
    assert snapshot.search_calls == 10
    assert snapshot.total_tool_calls == 10


def test_trace_collector_redacts_secrets_and_summarizes_large_text() -> None:
    collector = TraceCollector(
        max_text_chars=12,
        max_collection_items=3,
        clock=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )
    event = collector.record(
        "tool_finished",
        {
            "api_key": "sk-super-secret-value",
            "headers": {"Authorization": "Bearer abcdefghijklmnop"},
            "url": "https://fixture.test/?token=topsecret&safe=1",
            "content": "This fixture body is intentionally much too long.",
            "max_tokens": 64,
        },
    )

    assert event.sequence == 1
    assert event.payload["api_key"] == "[REDACTED]"
    assert event.payload["headers"] == {"Authorization": "[REDACTED]"}
    assert event.payload["_trace_omitted_items"] == 2
    assert event.payload["url"] == ("https://fixture.test/?token=[REDACTED]&safe=1")
    serialized = collector.jsonl()
    assert "super-secret" not in serialized
    assert "topsecret" not in serialized
    assert "intentionally much too long" not in serialized

    body_only = TraceCollector(max_text_chars=8).record(
        "fetch",
        {"content": "0123456789"},
    )
    summary = body_only.payload["content"]
    assert isinstance(summary, dict)
    assert summary["_trace_value"] == "omitted_body_text"
    assert summary["chars"] == 10


def test_trace_redaction_covers_non_openai_tokens_and_camel_case_keys() -> None:
    collector = TraceCollector()
    event = collector.record(
        "provider_failure",
        {
            "xApiKey": "custom-provider-secret",
            "message": (
                "credential=plain-secret "
                "hf_abcdefghijklmnopqrstuvwxyz123456 "
                "github_pat_abcdefghijklmnopqrstuvwxyz123456"
            ),
            "token_usage": {"total_tokens": 7},
        },
    )

    assert event.payload["xApiKey"] == "[REDACTED]"
    assert event.payload["token_usage"] == {"total_tokens": 7}
    serialized = collector.jsonl()
    assert "custom-provider-secret" not in serialized
    assert "plain-secret" not in serialized
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in serialized
    assert "github_pat_abcdefghijklmnopqrstuvwxyz123456" not in serialized


def test_system_runner_protocol_has_the_required_two_argument_interface() -> None:
    class StructurallyValidRunner:
        def run(
            self,
            task: EvalTask,
            resolved_config: ResolvedConfig,
        ) -> RunResult:
            raise NotImplementedError

    assert isinstance(StructurallyValidRunner(), SystemRunner)
