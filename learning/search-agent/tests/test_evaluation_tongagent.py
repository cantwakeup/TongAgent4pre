"""Offline integration tests for the production B3 evaluation adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation.execution import resolve_system_config
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.schema import CompletionStatus, EvalTask
from evaluation.systems.tongagent import TongAgentRunner


pytestmark = pytest.mark.usefixtures("socket_disabled")


def _resolved_config(
    artifact_directory: Path,
    backend: FixtureBackend,
):
    return resolve_system_config(
        "tongagent",
        "sha256:" + ("a" * 64),
        7,
        artifact_directory,
        overrides={"fixture_revision": backend.revision},
    )


def _two_source_case() -> tuple[
    EvalTask,
    FixtureBackend,
    FixtureChatModel,
    dict[str, str],
]:
    claim = "The deterministic Alpha fact is corroborated by two fixture publishers."
    quote_one = "Publisher One states that the deterministic Alpha fact is confirmed."
    quote_two = "Publisher Two independently confirms the deterministic Alpha fact."
    url_one = "https://one.fixture.test/alpha"
    url_two = "https://two.fixture.test/alpha"
    report = f"""## Short Answer
- {claim} [C1][S1][S2]
## Key Findings

## Conflicts and Caveats

## Sources
- [S1] Publisher One — {url_one}
- [S2] Publisher Two — {url_two}
"""
    task = EvalTask(
        id="b3-two-source",
        question="Is the deterministic Alpha fact corroborated?",
        metadata={
            "answer": "Inner research cycle finished.",
            "planner": {
                "objective": "Corroborate the deterministic Alpha fact.",
                "subquestions": [
                    {
                        "question": (
                            "Do two fixture publishers corroborate the Alpha fact?"
                        ),
                        "rationale": "Two independent fixture hosts are required.",
                        "depends_on": [],
                    }
                ],
                "completion_criteria": [
                    "Register exact quotes from two distinct fixture publishers."
                ],
            },
            "strict_fixture_tools": True,
            "research_script": [
                {
                    "tool": "web_search",
                    "phase": "research",
                    "research_cycle": 1,
                    "args": {"query": "alpha publisher one"},
                },
                {
                    "tool": "fetch_url",
                    "phase": "research",
                    "research_cycle": 1,
                    "args": {"url": url_one},
                },
                {
                    "tool": "record_evidence",
                    "phase": "research",
                    "research_cycle": 1,
                    "stop_cycle": True,
                    "args": {
                        "source_id": "S1",
                        "claim": claim,
                        "quote": quote_one,
                        "stance": "supports",
                    },
                },
                # The Stage 03D controller observes one progress cycle, then one
                # no-progress cycle before releasing the reserve.  Scheduling
                # the second source in cycle 3 exercises that real controller.
                {
                    "tool": "web_search",
                    "phase": "research",
                    "research_cycle": 3,
                    "args": {"query": "alpha publisher two"},
                },
                {
                    "tool": "fetch_url",
                    "phase": "research",
                    "research_cycle": 3,
                    "args": {"url": url_two},
                },
                {
                    "tool": "record_evidence",
                    "phase": "research",
                    "research_cycle": 3,
                    "args": {
                        "source_id": "S2",
                        "claim_id": "C1",
                        "claim": claim,
                        "quote": quote_two,
                        "stance": "supports",
                    },
                },
                {
                    "tool": "update_subquestion",
                    "phase": "research",
                    "research_cycle": 3,
                    "args": {
                        "subquestion_id": "SQ1",
                        "status": "covered",
                        "evidence_source_ids": ["S1", "S2"],
                        "note": (
                            "Two distinct fixture publishers corroborate the claim."
                        ),
                    },
                },
                {
                    "tool": "write_file",
                    "phase": "report",
                    "args": {
                        "file_path": "/report.md",
                        "content": report,
                    },
                },
            ],
        },
    )
    page_content = {
        url_one: quote_one + " " + ("First independent context. " * 30),
        url_two: quote_two + " " + ("Second distinct context. " * 30),
    }
    backend = FixtureBackend(
        searches={
            "alpha publisher one": {
                "results": [
                    {
                        "title": "Publisher One",
                        "url": url_one,
                        "snippet": quote_one,
                        "relevance_score": 100,
                    }
                ]
            },
            "alpha publisher two": {
                "results": [
                    {
                        "title": "Publisher Two",
                        "url": url_two,
                        "snippet": quote_two,
                        "relevance_score": 100,
                    }
                ]
            },
        },
        pages={
            url_one: {
                "title": "Publisher One",
                "content": page_content[url_one],
            },
            url_two: {
                "title": "Publisher Two",
                "content": page_content[url_two],
            },
        },
    )
    model = FixtureChatModel.from_task(task, system_id="tongagent")
    return task, backend, model, page_content


def test_real_tongagent_graph_completes_with_native_exact_quote_provenance(
    tmp_path: Path,
) -> None:
    task, backend, model, page_content = _two_source_case()
    config = _resolved_config(tmp_path, backend)
    runner = TongAgentRunner(fixture_backend=backend, model=model)

    with (
        patch("search_agent._load_local_env") as load_env,
        patch("search_agent.ChatOpenAI") as chat_openai,
    ):
        result = runner.run(task, config)

    load_env.assert_not_called()
    chat_openai.assert_not_called()
    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.failure_type is None
    assert result.search_calls == 2
    assert result.fetch_calls == 2
    assert result.relevant_searches == 2
    assert result.evidence_count == 2
    assert result.structural_subquestion_coverage == 1.0
    assert result.final_answer is not None
    assert "[C1][S1][S2]" in result.final_answer
    assert {item.citation_id for item in result.citations} == {"E1", "E2"}
    for citation in result.citations:
        assert citation.quote is not None
        assert citation.quote in page_content[citation.url]
        assert citation.claim == (
            "The deterministic Alpha fact is corroborated by two fixture publishers."
        )

    native = tmp_path / "native"
    tongagent = native / "tongagent"
    expected = {
        native / "answer.md",
        native / "budget.json",
        native / "native.json",
        native / "resolved_config.json",
        native / "tool_calls.json",
        native / "trace.jsonl",
        tongagent / "checkpoint.sqlite",
        tongagent / "compact_checkpoints.json",
        tongagent / "control.json",
        tongagent / "evidence.json",
        tongagent / "events.jsonl",
        tongagent / "plan.json",
        tongagent / "report.md",
        tongagent / "sources.json",
        tongagent / "token_control.json",
        tongagent / "validation.json",
    }
    assert all(path.is_file() for path in expected)
    assert (tongagent / "checkpoint.sqlite").stat().st_size > 0

    plan_payload = json.loads((tongagent / "plan.json").read_text())
    assert plan_payload["plan"]["status"] == "completed"
    assert plan_payload["plan"]["structural_subquestion_coverage"] == 1.0
    evidence_payload = json.loads((tongagent / "evidence.json").read_text())
    assert len(evidence_payload["evidence_units"]) == 2
    assert all(
        item["quote"] in page_content[item["url"]]
        for item in evidence_payload["evidence_units"]
    )
    control = json.loads((tongagent / "control.json").read_text())
    actions = [item["action"] for item in control["decision_history"]]
    assert "expand_budget" in actions
    assert actions[-1] == "finish_success"
    token_control = json.loads((tongagent / "token_control.json").read_text())
    assert token_control["configured"] is True
    assert token_control["stage_output_caps"]["planner"] <= (
        config.model.max_output_tokens
    )
    assert token_control["buckets"]["sq:SQ1"]["actual_tokens"] > 0
    compact_checkpoints = json.loads(
        (tongagent / "compact_checkpoints.json").read_text()
    )
    assert compact_checkpoints
    assert compact_checkpoints[-1]["subquestion_id"] == "SQ1"
    assert json.loads((tongagent / "validation.json").read_text()) == {
        "completeness_errors": [],
        "fatal_errors": [],
        "source_section_canonicalized": False,
        "status": "passed",
    }

    execution = json.loads((native / "budget.json").read_text())["execution"]
    assert execution["model_calls"] == len(model.call_history) + 1
    main_tokens = sum(
        item["input_tokens"] + item["output_tokens"] for item in model.call_history
    )
    assert result.token_usage is not None
    assert result.token_usage.total_tokens is not None
    assert result.token_usage.total_tokens > main_tokens
    assert execution["total_tokens"] == result.token_usage.total_tokens
    assert execution["estimated_token_charges"] == 0
    assert execution["accounted_tokens"] == result.token_usage.total_tokens
    assert execution["reserved_tokens"] == 0
    assert execution["outstanding_model_reservations"] == 0
    trace = [
        json.loads(line) for line in (native / "trace.jsonl").read_text().splitlines()
    ]
    model_labels = {
        event["payload"].get("label")
        for event in trace
        if event["event_type"] == "model_call_started"
    }
    assert "tongagent.planner" in model_labels
    assert "fixture-chat-model" in model_labels
    assert any(event["event_type"] == "tool_call_started" for event in trace)
    assert all(call.tool_name != "task" for call in result.tool_calls)


def test_irrelevant_nonempty_search_stays_partial_without_evidence(
    tmp_path: Path,
) -> None:
    report = """## Short Answer

## Key Findings

## Conflicts and Caveats
- Structural subquestion coverage is partial; unsupported subquestions: SQ1.
- No canonical claim passed the evidence gate.
## Sources
"""
    task = EvalTask(
        id="b3-irrelevant",
        question="What does the missing Alpha evidence establish?",
        metadata={
            "answer": "No fixture evidence was established.",
            "planner": {
                "objective": "Test an irrelevant non-empty search.",
                "subquestions": [
                    {
                        "question": "Does the fixture establish Alpha?",
                        "rationale": "Reject irrelevant non-empty retrieval.",
                        "depends_on": [],
                    }
                ],
            },
            "strict_fixture_tools": True,
            "research_script": [
                {
                    "tool": "web_search",
                    "phase": "research",
                    "research_cycle": 1,
                    "stop_cycle": True,
                    "args": {"query": "irrelevant alpha"},
                },
                {
                    "tool": "write_file",
                    "phase": "report",
                    "args": {
                        "file_path": "/report.md",
                        "content": report,
                    },
                },
            ],
        },
    )
    backend = FixtureBackend(
        searches={
            "irrelevant alpha": {
                "results": [
                    {
                        "title": "Unrelated beta page",
                        "url": "https://noise.fixture.test/beta",
                        "snippet": "This result discusses only beta.",
                        "relevance_score": 0,
                    }
                ]
            }
        },
        pages={},
    )
    model = FixtureChatModel.from_task(task, system_id="tongagent")
    config = _resolved_config(tmp_path, backend)
    result = TongAgentRunner(fixture_backend=backend, model=model).run(task, config)

    assert result.completion_status == CompletionStatus.PARTIAL
    assert result.failure_type is None
    assert result.search_calls == 1
    assert result.fetch_calls == 0
    assert result.relevant_searches == 0
    assert result.evidence_count == 0
    assert result.structural_subquestion_coverage == 0.0
    assert result.citations == []
    assert result.final_answer is not None
    assert "No canonical claim passed" in result.final_answer

    tongagent = tmp_path / "native" / "tongagent"
    plan_payload = json.loads((tongagent / "plan.json").read_text())
    assert plan_payload["plan"]["status"] == "partial"
    assert plan_payload["plan"]["subquestions"][0]["status"] == "blocked"
    assert plan_payload["budget"]["nonempty_searches"] == 1
    assert plan_payload["budget"]["relevant_searches"] == 0
    validation = json.loads((tongagent / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["fatal_errors"] == []
    assert validation["completeness_errors"]


def test_multi_is_rejected_until_subagent_accounting_is_part_of_the_ablation(
    tmp_path: Path,
) -> None:
    task, backend, model, _ = _two_source_case()
    config = resolve_system_config(
        "tongagent",
        "sha256:" + ("b" * 64),
        7,
        tmp_path,
        overrides={
            "fixture_revision": backend.revision,
            "system_options": {"mode": "multi"},
        },
    )

    with pytest.raises(ValueError, match="requires mode=single"):
        TongAgentRunner(fixture_backend=backend, model=model).run(task, config)

    assert backend.calls == []
    assert model.call_history == []


def test_preparation_failure_keeps_native_metrics_unavailable_not_zero(
    tmp_path: Path,
) -> None:
    task, backend, model, _ = _two_source_case()
    config = resolve_system_config(
        "tongagent",
        "sha256:" + ("c" * 64),
        7,
        tmp_path,
        overrides={"fixture_revision": "mismatched-fixture-revision"},
    )

    result = TongAgentRunner(fixture_backend=backend, model=model).run(task, config)

    assert result.completion_status == CompletionStatus.FAILED
    assert result.evidence_count is None
    assert result.structural_subquestion_coverage is None
    assert result.relevant_searches == 0
    assert backend.calls == []
    assert model.call_history == []
