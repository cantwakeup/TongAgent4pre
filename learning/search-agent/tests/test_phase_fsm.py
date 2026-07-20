"""Offline regression tests for code-enforced TongAgent research phases."""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from evidence_graph import text_sha256
from research_graph import (
    DraftSubquestion,
    build_phase_research_tools,
    create_research_plan,
)


def _runtime(state: dict[str, Any], call_id: str) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _value: None,
        tool_call_id=call_id,
        store=None,
    )


def _state() -> dict[str, Any]:
    plan = create_research_plan(
        "Find the fact.",
        [DraftSubquestion(question="Find the primary fact")],
        plan_id_factory=lambda: "phase-plan",
    )
    plan["subquestions"][0]["status"] = "researching"
    return {
        "research_plan": plan,
        "research_events": [],
        "active_subquestion_id": "SQ1",
        "active_research_phase": "NEED_QUERY",
        "active_search_scope": {},
        "active_fetch_scope": {},
        "active_evidence_attempts": {},
        "phase_timings": [],
    }


def _apply(state: dict[str, Any], command: Any) -> dict[str, Any]:
    state.update(
        {key: value for key, value in command.update.items() if key != "messages"}
    )
    return state


def _tools(*, fail_first_fetch: bool = False) -> tuple[list[Any], list[str]]:
    fetched: list[str] = []

    @tool("web_search")
    def search(query: str, max_results: int = 5) -> str:
        """Return one deterministic public candidate."""
        del query, max_results
        return json.dumps(
            {
                "status": "success",
                "search_quality": "relevant",
                "results": [
                    {
                        "title": "Primary source",
                        "url": "https://public.example/primary",
                        "snippet": "The primary fact.",
                        "provider": "fixture",
                        "final_rank": 1,
                        "relevance_tier": "relevant",
                    }
                ]
                + (
                    [
                        {
                            "title": "Fallback source",
                            "url": "https://fallback.example/primary",
                            "snippet": "The primary fact independently.",
                            "provider": "fixture",
                            "final_rank": 2,
                            "relevance_tier": "uncertain",
                        }
                    ]
                    if fail_first_fetch
                    else []
                ),
            }
        )

    @tool("fetch_url")
    def fetch(url: str, max_chars: int = 12_000) -> str:
        """Return deterministic page text for the selected candidate."""
        del max_chars
        fetched.append(url)
        if fail_first_fetch and url == "https://public.example/primary":
            return json.dumps(
                {
                    "status": "error",
                    "url": url,
                    "failure_taxonomy": "access_blocked",
                }
            )
        return json.dumps(
            {
                "status": "success",
                "url": url,
                "source_id": "S1",
                "content": "The primary fact is exactly forty two. This is a second sentence.",
            }
        )

    snapshot = lambda: {  # noqa: E731 - compact deterministic fixture.
        "subquestion_usage": {"SQ1": {"relevant_searches": 1}},
        "successful_sources": [
            {
                "source_id": "S1",
                "url": "https://public.example/primary",
                "title": "Primary source",
                "content_sha256": "1" * 64,
                "latest_content_sha256": "1" * 64,
                "content_chars": 120,
                "evidence_quality": "full",
                "content_revisions": [
                    {
                        "content_sha256": "1" * 64,
                        "content_chars": 120,
                        "title": "Primary source",
                        "evidence_quality": "full",
                        "quality_reason": "",
                    }
                ],
            }
        ],
        "min_successful_sources": 1,
        "max_searches": 1,
        "relevant_searches": 1,
        "claims": [
            {
                "claim_id": "C1",
                "subquestion_id": "SQ1",
                "text": "The primary fact is exactly forty two.",
                "status": "supported",
                "supporting_evidence_ids": ["E1"],
                "contradicting_evidence_ids": [],
                "source_ids": ["S1"],
            }
        ],
        "evidence_units": [
            {
                "evidence_id": "E1",
                "claim_id": "C1",
                "subquestion_id": "SQ1",
                "source_id": "S1",
                "stance": "supports",
                "quote": "The primary fact is exactly forty two.",
                "quote_sha256": text_sha256("The primary fact is exactly forty two."),
                "source_content_sha256": "1" * 64,
                "url": "https://public.example/primary",
                "title": "Primary source",
                "evidence_quality": "full",
            }
        ],
        "conflicts": [],
    }
    return build_phase_research_tools(
        search_tool=search,
        fetch_tool=fetch,
        evidence_record=lambda **_kwargs: {
            "claim": snapshot()["claims"][0],
            "evidence": snapshot()["evidence_units"][0],
        },
        budget_snapshot=snapshot,
    ), fetched


def test_parallel_dependent_actions_are_rejected_without_fetch_budget() -> None:
    tools, fetched = _tools()
    choose, select, _ = tools
    state = _state()

    search = choose.func(
        query="Primary fact attribute",
        task_type="single_fact_lookup",
        runtime=_runtime(state, "search"),
    )
    # This mirrors the old one-message search -> fetch request.  The fetch
    # sees the immutable pre-search NEED_QUERY state and cannot run.
    rejected = select.func(
        result_id="R1", reason="guess", runtime=_runtime(_state(), "parallel-fetch")
    )

    assert state["active_research_phase"] == "NEED_QUERY"
    _apply(state, search)
    assert state["active_research_phase"] == "NEED_RESULT_SELECTION"
    assert fetched == []
    payload = json.loads(rejected.update["messages"][0].content)
    assert payload["failure_type"] == "invalid_phase_action"
    assert fetched == []


def test_scoped_result_id_and_real_quote_complete_the_only_legal_path() -> None:
    tools, fetched = _tools()
    choose, select, evidence = tools
    state = _state()
    _apply(
        state,
        choose.func(
            query="Primary fact attribute",
            task_type="single_fact_lookup",
            runtime=_runtime(state, "search"),
        ),
    )
    guessed = select.func(
        result_id="R999", reason="not in scope", runtime=_runtime(state, "guessed")
    )
    assert (
        json.loads(guessed.update["messages"][0].content)["failure_type"]
        == "invalid_phase_action"
    )
    assert fetched == []
    _apply(
        state,
        select.func(
            result_id="R1", reason="top candidate", runtime=_runtime(state, "fetch")
        ),
    )
    assert fetched == ["https://public.example/primary"]
    assert state["active_research_phase"] == "NEED_EVIDENCE"
    _apply(
        state,
        evidence.func(
            claim="The primary fact is exactly forty two.",
            quote_id="Q1",
            stance="support",
            runtime=_runtime(state, "evidence"),
        ),
    )
    assert state["active_research_phase"] == "SQ_DONE"
    transitions = [
        item
        for item in state["research_events"]
        if item["event"] == "research_phase_transition"
    ]
    assert [item["details"]["to"] for item in transitions] == [
        "NEED_RESULT_SELECTION",
        "NEED_EVIDENCE",
        "SQ_DONE",
    ]


def test_selected_access_blocked_candidate_uses_one_scoped_fallback() -> None:
    tools, fetched = _tools(fail_first_fetch=True)
    choose, select, _ = tools
    state = _state()
    _apply(
        state,
        choose.func(
            query="Primary fact attribute",
            task_type="single_fact_lookup",
            runtime=_runtime(state, "search"),
        ),
    )
    fetch = select.func(
        result_id="R1", reason="top candidate", runtime=_runtime(state, "fetch")
    )
    _apply(state, fetch)
    assert fetched == [
        "https://public.example/primary",
        "https://fallback.example/primary",
    ]
    assert state["active_research_phase"] == "NEED_EVIDENCE"
    transition = state["research_events"][-1]["details"]
    assert transition["fallback_count"] == 1
