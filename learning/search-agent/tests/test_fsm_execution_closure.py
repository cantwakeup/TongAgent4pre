"""Regression tests for schema-bound FSM decisions and final-state integrity."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from agent_policy import EFFORT_POLICIES
from evidence_graph import allowed_report_caveat_lines, report_claim_mapping_errors
from evaluation.systems.tongagent import _canonical_phase_checkpoint
from research_graph import (
    DraftSubquestion,
    build_deterministic_abstain_report,
    build_phase_research_tools,
    build_research_graph,
    create_research_plan,
)
from research_state import ResearchPlan, TongAgentState
from search_agent import ResearchBudget


class _UnusedAgent:
    """The structured FSM path does not invoke the free-form agent."""

    def invoke(self, state: TongAgentState, **_kwargs: Any) -> dict[str, Any]:
        return dict(state)


def _two_sq_plan(topic: str, maximum: int) -> ResearchPlan:
    del maximum
    return create_research_plan(
        topic,
        [
            DraftSubquestion(question="Establish the first structured fact"),
            DraftSubquestion(question="Establish the second structured fact"),
        ],
        max_subquestions=2,
        plan_id_factory=lambda: "repair-plan",
    )


def test_invalid_sq2_query_is_repaired_once_and_never_blocks_first() -> None:
    budget = ResearchBudget(EFFORT_POLICIES["medium"], strategy="fixed")

    @tool("web_search")
    def search(query: str, max_results: int = 5) -> str:
        """Return one deterministic public search result."""
        del max_results
        assert budget.reserve_search()
        active = budget.active_subquestion_id
        url = f"https://{active.casefold()}.example/fact"
        payload = {
            "status": "success",
            "search_quality": "relevant",
            "results": [
                {
                    "title": f"{active} primary source",
                    "url": url,
                    "snippet": f"{active} confirms the structured fact.",
                    "relevance_tier": "relevant",
                    "final_rank": 1,
                }
            ],
        }
        budget.record_tool_attempt(
            tool_name="web_search", target=query, payload=payload
        )
        return json.dumps(payload)

    @tool("fetch_url")
    def fetch(url: str, max_chars: int = 12_000) -> str:
        """Return deterministic page content for the selected result."""
        del max_chars
        assert budget.reserve_fetch()
        active = budget.active_subquestion_id
        quote = f"{active} primary source confirms the structured fact."
        content = (quote + " Additional independently readable context.") * 20
        payload = {
            "status": "success",
            "url": url,
            "title": f"{active} primary source",
            "content": content,
            "content_chars": len(content),
            "evidence_quality": "full",
        }
        source_id = budget.record_fetch(payload)
        assert source_id is not None
        payload["source_id"] = source_id
        budget.record_tool_attempt(tool_name="fetch_url", target=url, payload=payload)
        return json.dumps(payload)

    phase_tools = build_phase_research_tools(
        search_tool=search,
        fetch_tool=fetch,
        evidence_record=budget.record_evidence,
        budget_snapshot=budget.snapshot,
    )
    drain = phase_tools[0].metadata["phase_action_drain"]
    apply = phase_tools[0].metadata["phase_action_apply"]
    decisions: list[tuple[str, str, bool]] = []

    def decider(
        state: TongAgentState,
        phase: str,
        repair: bool,
        _validation_error: str | None,
    ) -> dict[str, Any]:
        active = str(state["active_subquestion_id"])
        decisions.append((active, phase, repair))
        if active == "SQ2" and phase == "NEED_QUERY" and not repair:
            return {
                "status": "invalid_model_action",
                "validation_error": "empty_structured_response",
                "raw_output_summary": "<empty>",
            }
        if phase == "NEED_QUERY":
            return {
                "status": "success",
                "decision": {
                    "query": f"{active} structured fact",
                    "task_type": "single_fact_lookup",
                },
            }
        if phase == "NEED_RESULT_SELECTION":
            return {
                "status": "success",
                "decision": {"result_id": "R1", "reason": "top candidate"},
            }
        assert phase == "NEED_EVIDENCE"
        return {
            "status": "success",
            "decision": {
                "claim": f"{active} primary source confirms the structured fact.",
                "quote_id": "Q1",
                "stance": "support",
            },
        }

    graph = build_research_graph(
        research_agent=_UnusedAgent(),
        report_agent=_UnusedAgent(),
        planner=_two_sq_plan,
        budget_snapshot=budget.snapshot,
        budget_configure=budget.configure_subquestions,
        budget_activate=budget.activate_subquestion,
        checkpointer=None,
        max_subquestions=2,
        max_research_cycles=2,
        phase_action_drain=drain,
        phase_action_apply=apply,
        phase_decider=decider,
        strategy="fixed",
    )

    result = graph.invoke({"messages": [], "research_topic": "topic"})

    assert ("SQ2", "NEED_QUERY", False) in decisions
    assert ("SQ2", "NEED_QUERY", True) in decisions
    action_events = [
        item
        for item in result["model_action_events"]
        if item["subquestion_id"] == "SQ2"
    ]
    assert action_events[:2] == [
        {
            "phase": "NEED_QUERY",
            "subquestion_id": "SQ2",
            "event": "invalid_model_action",
            "validation_error": "empty_structured_response",
            "raw_output_summary": '"<empty>"',
        },
        {
            "phase": "NEED_QUERY",
            "subquestion_id": "SQ2",
            "event": "model_action_repair_attempted",
            "validation_error": "empty_structured_response",
        },
    ]
    assert any(
        item["event"] == "model_action_repair_succeeded" for item in action_events
    )
    sq2_transitions = [
        item
        for item in result["phase_transition_log"]
        if item["subquestion_id"] == "SQ2"
    ]
    assert sq2_transitions[0]["to"] == "NEED_QUERY"
    assert sq2_transitions[0]["action"] == "initialize"
    assert sq2_transitions[1] == {
        "subquestion_id": "SQ2",
        "from": "NEED_QUERY",
        "to": "NEED_RESULT_SELECTION",
        "action": "search",
    }


def test_canonical_phase_checkpoint_detects_silent_done_rewrite() -> None:
    plan = _two_sq_plan("topic", 2)
    state: dict[str, Any] = {
        "active_research_phase": "SQ_DONE",
        "phase_subquestion_id": "SQ1",
        "active_subquestion_id": None,
        "last_phase_action": {},
        "phase_transition_log": [
            {
                "subquestion_id": "SQ1",
                "from": "NEED_QUERY",
                "to": "SQ_BLOCKED",
                "action": "search",
            }
        ],
        "phase_timings": [],
    }
    checkpoint = _canonical_phase_checkpoint(state, plan, {})

    assert checkpoint["current_phase"] == "SQ_DONE"
    assert (
        "final_phase_mismatches_last_transition: SQ_DONE != SQ_BLOCKED"
        in (checkpoint["state_integrity_errors"])
    )
    assert "sq_done_without_covered_requirement" in checkpoint["state_integrity_errors"]


def test_zero_evidence_deterministic_abstain_is_validator_safe() -> None:
    plan = _two_sq_plan("topic", 2)
    plan["subquestions"][0]["status"] = "blocked"
    plan["subquestions"][0]["note"] = "No eligible source was fetched."
    plan["subquestions"][1]["status"] = "blocked"
    plan["subquestions"][1]["note"] = "Dependency could not be established."
    plan["status"] = "partial"
    plan["structural_subquestion_coverage"] = 0.0
    report = build_deterministic_abstain_report(plan, {}, integrity_failure=False)

    errors = report_claim_mapping_errors(
        report,
        plan_claim_ids=set(),
        claims=[],
        evidence_units=[],
        allowed_caveat_lines=allowed_report_caveat_lines(plan),
        allowed_non_factual_lines={"INSUFFICIENT_EVIDENCE", "ABSTAIN"},
    )
    assert "INSUFFICIENT_EVIDENCE" in report
    assert "ABSTAIN" in report
    assert all(not values for values in errors.values())
