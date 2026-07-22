"""Socket-disabled tests for the performance-first Long-ReAct workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from evaluation.execution import resolve_system_config
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.schema import AnswerStatus, CompletionStatus, EvalTask, ToolCallStatus
from evaluation.systems.long_react import (
    _answer_from_successful_python,
    _format_short_answer,
    StructuredResearchState,
    build_answer_contract,
    build_python_tool,
    compact_long_react_messages,
)
from evaluation.systems.tongagent import TongAgentRunner


pytestmark = pytest.mark.usefixtures("socket_disabled")

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "permissive_controlled"
)


def _backend() -> FixtureBackend:
    return FixtureBackend.from_directory(_FIXTURES)


def _config(tmp_path: Path, backend: FixtureBackend, *, system: str = "tongagent"):
    return resolve_system_config(
        system,
        "sha256:" + "9" * 64,
        17,
        tmp_path,
        fixture_directory="evaluation/fixtures/permissive_controlled",
        overrides={"fixture_revision": backend.revision, "runtime_mode": "long_react"},
    )


def _task() -> EvalTask:
    return EvalTask(
        id="long-react-controlled",
        question="Where is Lunarite Mine located?",
        reference_answer="Alton Valley",
        metadata={
            "source_urls": ["https://must-not-reach-the-model.invalid/secret"],
            "strict_fixture_tools": True,
            "answer": {"tongagent": "FINAL_ANSWER: Alton Valley"},
            "research_script": {
                "tongagent": [
                    {
                        "tool": "web_search",
                        "args": {
                            "query": '"lunarite mine" location',
                            "max_results": 5,
                        },
                    },
                    {
                        "tool": "fetch_url",
                        "args": {
                            "url": "https://archive.fixture.test/lunarite-mine",
                            "max_chars": 12_000,
                        },
                    },
                ]
            },
        },
    )


@pytest.mark.parametrize(
    ("question", "answer_type", "unit"),
    [
        ("Who wrote the novel?", "entity", None),
        ("Where was the author born?", "location", None),
        ("What is the birthplace and hometown of the scorer?", "location", None),
        ("How many albums were released?", "count", None),
        ("When did the bridge open?", "date", None),
        ("How many years older was A than B?", "duration", "years"),
        ("What was the population difference?", "number", None),
    ],
)
def test_answer_contract_is_derived_only_from_question(
    question: str, answer_type: str, unit: str | None
) -> None:
    contract = build_answer_contract(question)

    assert contract.answer_type == answer_type
    assert contract.output_unit == unit
    assert contract.output_format == "short_answer"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {"operation": "arithmetic", "expression": "2 + 3 * 4"},
            14,
        ),
        (
            {
                "operation": "date_difference",
                "values": ["1787", "1872"],
                "output_unit": "years",
            },
            85,
        ),
        (
            {"operation": "sort", "values": ["10", "2", "5"]},
            [2.0, 5.0, 10.0],
        ),
        (
            {"operation": "count", "values": ["A", "A", "B"], "unique": True},
            2,
        ),
    ],
)
def test_python_tool_executes_only_bounded_operations(
    arguments: dict[str, object], expected: object
) -> None:
    payload = json.loads(build_python_tool().invoke(arguments))

    assert payload["status"] == "success"
    assert payload["result"] == expected


def test_python_tool_rejects_arbitrary_code() -> None:
    payload = json.loads(
        build_python_tool().invoke(
            {"operation": "arithmetic", "expression": "__import__('os').system('id')"}
        )
    )

    assert payload["status"] == "error"
    assert payload["error"] == "ValueError"


def test_synthesis_recovers_rounded_python_result_after_budget_stop() -> None:
    class Call:
        tool_name = "python"
        status = ToolCallStatus.SUCCESS
        result = {
            "status": "success",
            "operation": "arithmetic",
            "result": 28.0282,
        }

    answer = _answer_from_successful_python(
        "How many times would it fit? Give a rounded whole number.",
        [Call()],
    )

    assert answer == "FINAL_ANSWER: 28"


def test_synthesis_uses_completed_years_from_date_difference() -> None:
    class Call:
        tool_name = "python"
        status = ToolCallStatus.SUCCESS
        result = {
            "status": "success",
            "operation": "date_difference",
            "result": 85.55138,
            "unit": "years",
        }

    answer = _answer_from_successful_python(
        "How many years had passed between admission and the representative's birth?",
        [Call()],
    )

    assert answer == "FINAL_ANSWER: 85"


def test_synthesis_does_not_use_incompatible_python_intermediate() -> None:
    class Call:
        tool_name = "python"
        status = ToolCallStatus.SUCCESS
        result = {
            "status": "success",
            "operation": "count",
            "result": 3,
        }

    answer = _answer_from_successful_python("Which house is described?", [Call()])

    assert answer is None


def test_synthesis_strips_location_labels_and_entity_explanation() -> None:
    assert (
        _format_short_answer(
            "What is the birthplace and hometown of the scorer?",
            "FINAL_ANSWER: Sidney Crosby — birthplace: Halifax; hometown: Cole Harbour",
        )
        == "FINAL_ANSWER: Halifax; Cole Harbour"
    )
    assert (
        _format_short_answer(
            "Which famous house is described?",
            "FINAL_ANSWER: Chatsworth House, associated with the historical clues",
        )
        == "FINAL_ANSWER: Chatsworth House"
    )


def test_context_manager_keeps_five_recent_tools_and_summarizes_older() -> None:
    messages = [HumanMessage(content="question")]
    for index in range(7):
        messages.append(
            AIMessage(
                content=f"old reasoning that must be removed {index}",
                tool_calls=[
                    {
                        "id": f"call-{index}",
                        "name": "fetch_url",
                        "args": {"url": f"https://fixture.test/{index}"},
                    }
                ],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps(
                    {
                        "status": "success",
                        "source_id": f"S{index + 1}",
                        "url": f"https://fixture.test/{index}",
                        "content": f"confirmed fact {index} " + "x" * 2_000,
                    }
                ),
                tool_call_id=f"call-{index}",
                name="fetch_url",
            )
        )

    state = StructuredResearchState.from_question(
        "question", build_answer_contract("question")
    )
    compacted, summary, old_count = compact_long_react_messages(
        messages,
        research_state=state,
        budget_snapshot={"remaining_total_tokens": 42_000},
    )

    assert old_count == 2
    assert isinstance(compacted[0], SystemMessage)
    assert isinstance(compacted[1], SystemMessage)
    assert isinstance(compacted[2], HumanMessage)
    assert compacted[2].content == "question"
    assert json.loads(summary)["question_target"]["question"] == "question"
    assert "remaining_total_tokens" in str(compacted[1].content)
    tool_messages = [item for item in compacted if isinstance(item, ToolMessage)]
    assert len(tool_messages) == 5
    assert "confirmed fact 2" in str(tool_messages[0].content)
    assert "confirmed fact 0" not in json.dumps(
        [item.model_dump(mode="json") for item in tool_messages]
    )
    assert "confirmed fact 0" in summary
    assert all(item.content == "" for item in compacted if isinstance(item, AIMessage))
    assert len(summary) <= 12_000


def test_structured_state_tracks_fact_schema_operands_and_candidate_roles() -> None:
    question = "How many years after Alpha began did Beta end?"
    messages = [HumanMessage(content=question)]
    scripted = [
        (
            "web_search",
            {"query": "Alpha began year"},
            {
                "status": "success",
                "query": "Alpha began year",
                "results": [
                    {
                        "title": "Alpha",
                        "snippet": "Alpha began in 1899.",
                        "url": "https://fixture.test/alpha",
                    }
                ],
            },
        ),
        (
            "web_search",
            {"query": "Beta ended year"},
            {
                "status": "success",
                "query": "Beta ended year",
                "results": [
                    {
                        "title": "Beta",
                        "snippet": "Beta ended in 1906.",
                        "url": "https://fixture.test/beta",
                    }
                ],
            },
        ),
        (
            "python",
            {"operation": "arithmetic", "expression": "1906-1899"},
            {"status": "success", "operation": "arithmetic", "result": 7},
        ),
    ]
    for index, (name, arguments, payload) in enumerate(scripted, start=1):
        call_id = f"call-{index}"
        messages.append(
            AIMessage(
                content="discard me",
                tool_calls=[{"id": call_id, "name": name, "args": arguments}],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps(payload),
                tool_call_id=call_id,
                name=name,
            )
        )

    state = StructuredResearchState.from_question(
        question, build_answer_contract(question)
    )
    state.observe_messages(messages)
    snapshot = state.snapshot()

    assert snapshot["question_target"]["answer_type"] == "duration"
    assert {item["semantic_role"] for item in snapshot["intermediate_entities"]} == {
        "intermediate",
        "possible_final",
    }
    assert {item["semantic_role"] for item in snapshot["final_candidates"]} == {
        "possible_final",
        "final",
    }
    assert {item["value"] for item in snapshot["operands"]} >= {"1899", "1906"}
    assert all(
        set(item)
        == {
            "subject",
            "relation",
            "object",
            "semantic_role",
            "source_id",
            "introduced_turn",
        }
        for item in snapshot["confirmed_facts"]
    )


def test_long_react_preflight_and_controlled_loop_are_nonblocking(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _task()
    model = FixtureChatModel.from_task(task, system_id="tongagent")
    runner = TongAgentRunner(fixture_backend=backend, model=model)

    preflight = runner.preflight(task, _config(tmp_path / "preflight", backend))
    result = runner.run(task, _config(tmp_path, backend))

    assert preflight["runtime_mode"] == "long_react"
    assert preflight["model_invocations"] == 0
    assert set(preflight["runtime_tools"]) >= {"web_search", "fetch_url", "python"}
    assert result.runtime_mode == "long_react"
    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.answer_status == AnswerStatus.ANSWER
    assert result.final_answer == "FINAL_ANSWER: Alton Valley"
    assert result.normalized_exact_match is True
    assert result.search_calls == 1
    assert result.fetch_calls == 1
    assert result.evidence_count == 0
    assert result.workflow_metrics is not None
    assert result.workflow_metrics["posthoc_audit_blocked_answer"] is False
    native = tmp_path / "native" / "tongagent"
    contract = json.loads((native / "answer_contract.json").read_text())
    context = json.loads((native / "context_manager.json").read_text())
    structured = json.loads((native / "structured_research_state.json").read_text())
    audit = json.loads((native / "long_react_audit.json").read_text())
    assert contract["answer_contract"]["answer_type"] == "location"
    assert context["keep_last_k_tool_results"] == 5
    assert structured["question_target"]["question"] == task.question
    assert structured["confirmed_facts"]
    assert all(
        set(item)
        == {
            "subject",
            "relation",
            "object",
            "semantic_role",
            "source_id",
            "introduced_turn",
        }
        for item in structured["confirmed_facts"]
    )
    assert audit["posthoc_only"] is True
    assert audit["answer_unchanged_by_audit"] is True
    assert len(audit["source_mapping"]) == 1


def test_long_react_fairness_fingerprint_matches_all_three_systems(
    tmp_path: Path,
) -> None:
    backend = _backend()
    configs = [
        _config(tmp_path / system, backend, system=system)
        for system in ("simple_react", "vanilla_deepagents", "tongagent")
    ]

    assert {config.runtime_mode for config in configs} == {"long_react"}
    assert len({config.fairness_fingerprint for config in configs}) == 1
