"""Offline tests for TongAgent's explicit Stage 03A research workflow."""

from __future__ import annotations

import tempfile
import unittest
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph import MessagesState

from research_graph import (
    DraftSubquestion,
    build_model_planner,
    build_research_graph,
    build_research_state_tools,
    build_source_ledger_tool,
    calculate_plan_coverage,
    create_research_plan,
    refresh_plan_status,
    select_next_subquestion,
    transition_subquestion,
)
from research_state import ResearchPlan, TongAgentState


def _fixed_plan(topic: str, max_subquestions: int) -> ResearchPlan:
    """Return a deterministic two-step dependency plan."""
    del max_subquestions
    return create_research_plan(
        topic,
        [
            DraftSubquestion(question="Establish the primary facts"),
            DraftSubquestion(
                question="Check limitations and disagreement", depends_on=[1]
            ),
        ],
        plan_id_factory=lambda: "plan-fixed",
    )


class _FakeLedger:
    """Small serializable ledger used by the fake research subgraph."""

    def __init__(self) -> None:
        self.sources: list[dict[str, Any]] = []
        self.claims: list[dict[str, Any]] = []
        self.evidence_units: list[dict[str, Any]] = []
        self.conflicts: list[dict[str, Any]] = []

    def add_source(
        self,
        subquestion_id: str = "SQ1",
        *,
        with_graph: bool = True,
        content_hash: str | None = None,
    ) -> str:
        """Add one unique successful source and return its ID."""
        source_id = f"S{len(self.sources) + 1}"
        claim_id = f"C{len(self.claims) + 1}"
        evidence_id = f"E{len(self.evidence_units) + 1}"
        canonical_hash = content_hash or f"{len(self.sources) + 1:064x}"
        self.sources.append(
            {
                "source_id": source_id,
                "url": f"https://example.com/{source_id}",
                "title": source_id,
                "content_chars": 800,
                "content_sha256": canonical_hash,
                "latest_content_sha256": canonical_hash,
                "content_revisions": [
                    {
                        "content_sha256": canonical_hash,
                        "title": source_id,
                        "content_chars": 800,
                        "evidence_quality": "full",
                        "quality_reason": "",
                    }
                ],
                "evidence_quality": "full",
            }
        )
        if not with_graph:
            return source_id
        self.claims.append(
            {
                "claim_id": claim_id,
                "subquestion_id": subquestion_id,
                "text": f"Claim for {subquestion_id}",
                "status": "supported",
                "supporting_evidence_ids": [evidence_id],
                "contradicting_evidence_ids": [],
                "source_ids": [source_id],
            }
        )
        self.evidence_units.append(
            {
                "evidence_id": evidence_id,
                "claim_id": claim_id,
                "subquestion_id": subquestion_id,
                "source_id": source_id,
                "stance": "supports",
                "source_content_sha256": canonical_hash,
            }
        )
        return source_id

    def snapshot(self) -> dict[str, Any]:
        """Return the budget shape consumed by the outer graph."""
        return {
            "effort": "test",
            "search_calls": len(self.sources),
            "successful_searches": len(self.sources),
            "max_searches": 10,
            "fetch_calls": len(self.sources),
            "max_fetches": 10,
            "min_successful_sources": 1,
            "successful_sources": list(self.sources),
            "failed_sources": [],
            "evidence_graph_version": 1,
            "claims": list(self.claims),
            "evidence_units": list(self.evidence_units),
            "conflicts": list(self.conflicts),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore successful sources as a fresh process would."""
        self.sources = [dict(item) for item in snapshot["successful_sources"]]
        self.claims = [dict(item) for item in snapshot.get("claims", [])]
        self.evidence_units = [
            dict(item) for item in snapshot.get("evidence_units", [])
        ]
        self.conflicts = [dict(item) for item in snapshot.get("conflicts", [])]


def _fake_research_agent(
    ledger: _FakeLedger,
    *,
    produce_evidence: bool = True,
    claim_evidence: bool = True,
) -> Any:
    """Compile a model-free subgraph compatible with the production workflow."""

    def reply(state: TongAgentState) -> dict[str, Any]:
        last = str(state["messages"][-1].content)
        if "[FINAL SYNTHESIS]" in last:
            return {
                "messages": [
                    AIMessage(content="final report prepared", id="fake-final")
                ]
            }
        active = state["active_subquestion_id"]
        if not produce_evidence:
            return {
                "messages": [
                    AIMessage(
                        content=f"no evidence for {active}",
                        id=f"fake-empty-{active}-{state['research_cycles']}",
                    )
                ]
            }
        source_id = ledger.add_source(str(active), with_graph=claim_evidence)
        claim_ids = [str(ledger.claims[-1]["claim_id"])] if claim_evidence else []
        if claim_evidence:
            plan = transition_subquestion(
                state["research_plan"],
                str(active),
                "covered",
                evidence_source_ids=[source_id],
                claim_ids=claim_ids,
                note="covered by fake evidence",
            )
        else:
            plan = deepcopy(state["research_plan"])
            target = next(item for item in plan["subquestions"] if item["id"] == active)
            target["status"] = "covered"
            target["evidence_source_ids"] = [source_id]
            target["note"] = "source-only state injected without transition helper"
            plan = refresh_plan_status(plan)
        return {
            "messages": [AIMessage(content=f"covered {active}", id=f"fake-{active}")],
            "research_plan": plan,
        }

    builder = StateGraph(TongAgentState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile()


class ResearchPlanTests(unittest.TestCase):
    """Verify normalized plans and deterministic state transitions."""

    def test_planner_output_becomes_pending_runtime_owned_state(self) -> None:
        plan = _fixed_plan("Research the topic", 2)

        self.assertEqual([item["id"] for item in plan["subquestions"]], ["SQ1", "SQ2"])
        self.assertEqual(
            [item["status"] for item in plan["subquestions"]],
            ["pending", "pending"],
        )
        self.assertEqual(plan["subquestions"][1]["depends_on"], ["SQ1"])
        self.assertEqual(plan["coverage"], 0.0)

    def test_model_planner_failure_uses_bounded_deterministic_fallback(self) -> None:
        class _FailingModel:
            def with_structured_output(self, schema: Any) -> Any:
                del schema
                raise RuntimeError("provider unavailable")

        planner = build_model_planner(_FailingModel())
        plan = planner("研究一个问题", 2)

        self.assertEqual(len(plan["subquestions"]), 2)
        self.assertEqual(plan["planner"], "deterministic-fallback:RuntimeError")
        self.assertEqual([item["id"] for item in plan["subquestions"]], ["SQ1", "SQ2"])

    def test_invalid_transition_and_evidence_free_coverage_are_rejected(self) -> None:
        plan, active = select_next_subquestion(_fixed_plan("topic", 2))
        self.assertEqual(active, "SQ1")

        with self.assertRaisesRegex(ValueError, "evidence source"):
            transition_subquestion(plan, "SQ1", "covered")
        with self.assertRaisesRegex(ValueError, "canonical claim"):
            transition_subquestion(plan, "SQ1", "covered", evidence_source_ids=["S1"])

        covered = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=["S1"],
            claim_ids=["C1"],
        )
        with self.assertRaisesRegex(ValueError, "covered -> researching"):
            transition_subquestion(covered, "SQ1", "researching")
        with self.assertRaisesRegex(ValueError, "S# format"):
            transition_subquestion(
                plan,
                "SQ1",
                "covered",
                evidence_source_ids=["not-a-source"],
            )

    def test_coverage_and_selection_skip_completed_work(self) -> None:
        plan, _ = select_next_subquestion(_fixed_plan("topic", 2))
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=["S1"],
            claim_ids=["C1"],
        )

        selected, active = select_next_subquestion(plan)

        self.assertEqual(active, "SQ2")
        self.assertEqual(selected["subquestions"][0]["attempts"], 1)
        self.assertEqual(calculate_plan_coverage(selected), 0.5)

    def test_blocked_dependency_is_cascaded_instead_of_researched(self) -> None:
        plan, _ = select_next_subquestion(_fixed_plan("topic", 2))
        plan = transition_subquestion(
            plan,
            "SQ1",
            "blocked",
            note="Primary facts were unavailable.",
        )

        selected, active = select_next_subquestion(plan)

        self.assertIsNone(active)
        self.assertEqual(
            [item["status"] for item in selected["subquestions"]],
            ["blocked", "blocked"],
        )
        self.assertIn("SQ1", selected["subquestions"][1]["note"])

    def test_stage03b_plan_migrates_without_inventing_claims(self) -> None:
        legacy = _fixed_plan("legacy topic", 2)
        del legacy["evidence_schema_version"]
        for item in legacy["subquestions"]:
            del item["claim_ids"]
            del item["conflict_ids"]

        migrated = refresh_plan_status(legacy)

        self.assertEqual(migrated["evidence_schema_version"], 0)
        self.assertEqual(migrated["subquestions"][0]["claim_ids"], [])
        self.assertEqual(migrated["subquestions"][0]["conflict_ids"], [])

    def test_state_tool_requires_active_ledger_claim_graph(self) -> None:
        ledger = _FakeLedger()
        plan, active = select_next_subquestion(_fixed_plan("topic", 2))
        runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [],
                "active_evidence_ids_before": [],
            },
            context=None,
            config={},
            stream_writer=lambda _value: None,
            tool_call_id="tool-call-1",
            store=None,
        )
        update_tool = build_research_state_tools(ledger.snapshot)[1]

        unknown = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=["S999"],
            note="invented",
            runtime=runtime,
        )
        wrong_item = update_tool.func(
            subquestion_id="SQ2",
            status="blocked",
            evidence_source_ids=[],
            note="wrong item",
            runtime=runtime,
        )

        self.assertEqual(json.loads(unknown)["status"], "error")
        self.assertIn("not present", json.loads(unknown)["error"])
        self.assertIn("Only the active", json.loads(wrong_item)["error"])

        source_id = ledger.add_source("SQ1")
        accepted = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="ledger-backed",
            runtime=runtime,
        )
        self.assertEqual(accepted.update["research_plan"]["coverage"], 0.5)
        accepted_payload = json.loads(accepted.update["messages"][0].content)
        self.assertEqual(
            accepted_payload["canonical_sources"][0]["url"],
            "https://example.com/S1",
        )

        ledger_payload = json.loads(
            build_source_ledger_tool(ledger.snapshot).invoke({})
        )
        self.assertEqual(ledger_payload["successful_sources"][0]["source_id"], "S1")

        stale_runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [source_id],
                "active_evidence_ids_before": ["E1"],
            },
            context=None,
            config={},
            stream_writer=lambda _value: None,
            tool_call_id="tool-call-2",
            store=None,
        )
        stale = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="stale source",
            runtime=stale_runtime,
        )
        self.assertEqual(stale.update["research_plan"]["coverage"], 0.5)
        stale_blocked = update_tool.func(
            subquestion_id="SQ1",
            status="blocked",
            evidence_source_ids=[source_id],
            note="blocked with stale evidence",
            runtime=stale_runtime,
        )
        self.assertIn("Blocked status is premature", json.loads(stale_blocked)["error"])

    def test_final_covered_update_requires_policy_source_minimum(self) -> None:
        ledger = _FakeLedger()
        plan = create_research_plan(
            "topic",
            [DraftSubquestion(question="One atomic question")],
            plan_id_factory=lambda: "plan-one",
        )
        plan, active = select_next_subquestion(plan)
        search_state = {"calls": 1, "successful": 0}

        def snapshot() -> dict[str, Any]:
            value = ledger.snapshot()
            value["min_successful_sources"] = 2
            value["search_calls"] = search_state["calls"]
            value["successful_searches"] = search_state["successful"]
            return value

        update_tool = build_research_state_tools(snapshot)[1]
        runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [],
                "active_evidence_ids_before": [],
            },
            context=None,
            config={},
            stream_writer=lambda _value: None,
            tool_call_id="source-minimum",
            store=None,
        )
        first = ledger.add_source("SQ1")

        insufficient = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[first],
            note="only one source",
            runtime=runtime,
        )

        self.assertIn("requires at least 2", json.loads(insufficient)["error"])
        premature_block = update_tool.func(
            subquestion_id="SQ1",
            status="blocked",
            evidence_source_ids=[first],
            note="give up despite remaining budget",
            runtime=runtime,
        )
        self.assertIn(
            "Blocked status is premature", json.loads(premature_block)["error"]
        )
        second = ledger.add_source("SQ1")
        missing_search = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[first, second],
            note="two independent sources",
            runtime=runtime,
        )
        self.assertIn(
            "requires at least 1 successful web searches",
            json.loads(missing_search)["error"],
        )
        search_state["successful"] = 1
        accepted = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[first, second],
            note="two independent sources and one search",
            runtime=runtime,
        )
        self.assertEqual(accepted.update["research_plan"]["coverage"], 1.0)

    def test_final_source_minimum_uses_evidence_revision_not_latest_alias(
        self,
    ) -> None:
        ledger = _FakeLedger()
        plan = create_research_plan(
            "topic",
            [DraftSubquestion(question="One atomic question")],
            plan_id_factory=lambda: "plan-revision-independence",
        )
        plan, active = select_next_subquestion(plan)
        shared_hash = "a" * 64
        first = ledger.add_source("SQ1", content_hash=shared_hash)
        second = ledger.add_source("SQ1", content_hash=shared_hash)
        ledger.sources[1]["latest_content_sha256"] = "b" * 64
        ledger.sources[1]["duplicate_of_source_id"] = None

        def snapshot() -> dict[str, Any]:
            value = ledger.snapshot()
            value["min_successful_sources"] = 2
            value["successful_searches"] = 1
            return value

        update_tool = build_research_state_tools(snapshot)[1]
        runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [],
                "active_evidence_ids_before": [],
            },
            context=None,
            config={},
            stream_writer=lambda _value: None,
            tool_call_id="revision-source-minimum",
            store=None,
        )

        result = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[first, second],
            note="latest aliases differ but cited revisions are exact copies",
            runtime=runtime,
        )

        self.assertIn("requires at least 2", json.loads(result)["error"])

    def test_nonfinal_blocked_update_rejects_recoverable_policy_gaps(self) -> None:
        ledger = _FakeLedger()
        plan, active = select_next_subquestion(_fixed_plan("topic", 2))
        update_tool = build_research_state_tools(ledger.snapshot)[1]
        runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [],
                "active_evidence_ids_before": [],
                "messages": [],
            },
            context=None,
            config={},
            stream_writer=lambda _value: None,
            tool_call_id="premature-nonfinal-block",
            store=None,
        )

        result = update_tool.func(
            subquestion_id="SQ1",
            status="blocked",
            evidence_source_ids=[],
            note="gave up before using the reserved budget",
            runtime=runtime,
        )

        self.assertIn("Blocked status is premature", json.loads(result)["error"])

    def test_multi_state_tool_requires_current_researcher_task(self) -> None:
        ledger = _FakeLedger()
        plan, active = select_next_subquestion(_fixed_plan("topic", 2))
        source_id = ledger.add_source("SQ1")
        update_tool = build_research_state_tools(
            ledger.snapshot, require_researcher=True
        )[1]
        base_messages = [
            HumanMessage(content="research", id="research-step-plan-fixed-SQ1-1")
        ]

        def runtime(messages: list[Any]) -> ToolRuntime:
            return ToolRuntime(
                state={
                    "messages": messages,
                    "research_plan": plan,
                    "research_events": [],
                    "active_subquestion_id": active,
                    "active_source_ids_before": [],
                    "active_evidence_ids_before": [],
                },
                context=None,
                config={},
                stream_writer=lambda _value: None,
                tool_call_id="delegation-gate",
                store=None,
            )

        rejected = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="no delegation",
            runtime=runtime(base_messages),
        )
        self.assertIn("requires task", json.loads(rejected)["error"])
        delegated = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "[SQ:SQ1] research the active subquestion",
                    },
                    "id": "task-call",
                    "type": "tool_call",
                }
            ],
        )
        unfinished = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="task has not returned",
            runtime=runtime([*base_messages, delegated]),
        )
        self.assertIn("requires task", json.loads(unfinished)["error"])
        failed = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="task failed",
            runtime=runtime(
                [
                    *base_messages,
                    delegated,
                    ToolMessage(
                        content="researcher failed",
                        tool_call_id="task-call",
                        status="error",
                    ),
                ]
            ),
        )
        self.assertIn("requires task", json.loads(failed)["error"])
        wrong_subquestion = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "[SQ:SQ2] research another subquestion",
                    },
                    "id": "wrong-sq-task",
                    "type": "tool_call",
                }
            ],
        )
        wrong_sq = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="delegated the wrong SQ",
            runtime=runtime(
                [
                    *base_messages,
                    wrong_subquestion,
                    ToolMessage(content="done", tool_call_id="wrong-sq-task"),
                ]
            ),
        )
        self.assertIn("requires task", json.loads(wrong_sq)["error"])
        no_current_step = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="delegation is not inside a research step",
            runtime=runtime(
                [
                    delegated,
                    ToolMessage(content="research complete", tool_call_id="task-call"),
                ]
            ),
        )
        self.assertIn("requires task", json.loads(no_current_step)["error"])
        accepted = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="delegated",
            runtime=runtime(
                [
                    *base_messages,
                    delegated,
                    ToolMessage(content="research complete", tool_call_id="task-call"),
                ]
            ),
        )
        self.assertEqual(accepted.update["research_plan"]["coverage"], 0.5)


class ResearchGraphTests(unittest.TestCase):
    """Verify graph ordering, bounded retries, and cross-process recovery."""

    def test_graph_completes_each_subquestion_before_reporting(self) -> None:
        ledger = _FakeLedger()
        planner_calls: list[str] = []

        def planner(topic: str, maximum: int) -> ResearchPlan:
            planner_calls.append(topic)
            return _fixed_plan(topic, maximum)

        graph = build_research_graph(
            research_agent=_fake_research_agent(ledger),
            planner=planner,
            budget_snapshot=ledger.snapshot,
            checkpointer=None,
            max_subquestions=2,
            max_research_cycles=4,
        )

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="topic", id="user-topic")],
                "research_topic": "topic",
            }
        )

        self.assertEqual(planner_calls, ["topic"])
        self.assertEqual(result["research_plan"]["status"], "completed")
        self.assertEqual(result["research_plan"]["coverage"], 1.0)
        self.assertEqual(len(ledger.sources), 2)
        event_types = [item["event"] for item in result["research_events"]]
        self.assertEqual(event_types[0], "plan_created")
        self.assertEqual(event_types[-1], "run_finished")
        self.assertEqual(event_types.count("subquestion_selected"), 2)

    def test_graph_configures_and_activates_subquestion_budget_scopes(self) -> None:
        ledger = _FakeLedger()
        configured: list[list[str]] = []
        activated: list[str | None] = []
        graph = build_research_graph(
            research_agent=_fake_research_agent(ledger),
            planner=_fixed_plan,
            budget_snapshot=ledger.snapshot,
            budget_configure=lambda ids: configured.append(list(ids)),
            budget_activate=activated.append,
            checkpointer=None,
            max_subquestions=2,
            max_research_cycles=4,
        )

        graph.invoke(
            {
                "messages": [HumanMessage(content="topic", id="budget-scope-user")],
                "research_topic": "topic",
            }
        )

        self.assertEqual(configured, [["SQ1", "SQ2"]])
        self.assertEqual(activated, [None, "SQ1", "SQ2", None])

    def test_attempt_cap_blocks_stalled_work_without_infinite_loop(self) -> None:
        ledger = _FakeLedger()
        graph = build_research_graph(
            research_agent=_fake_research_agent(ledger, produce_evidence=False),
            planner=_fixed_plan,
            budget_snapshot=ledger.snapshot,
            checkpointer=None,
            max_subquestions=2,
            max_research_cycles=4,
        )

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="topic", id="user-topic")],
                "research_topic": "topic",
            }
        )

        self.assertEqual(result["research_plan"]["status"], "partial")
        self.assertEqual(
            [item["status"] for item in result["research_plan"]["subquestions"]],
            ["blocked", "blocked"],
        )
        self.assertEqual(result["research_cycles"], 2)
        self.assertIn(
            "subquestion_dependency_blocked",
            [item["event"] for item in result["research_events"]],
        )

    def test_evaluate_rejects_source_only_covered_state(self) -> None:
        ledger = _FakeLedger()
        graph = build_research_graph(
            research_agent=_fake_research_agent(ledger, claim_evidence=False),
            planner=_fixed_plan,
            budget_snapshot=ledger.snapshot,
            checkpointer=None,
            max_subquestions=2,
            max_research_cycles=4,
        )

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="topic", id="source-only-user")],
                "research_topic": "topic",
            }
        )

        self.assertEqual(result["research_plan"]["status"], "partial")
        self.assertEqual(
            [item["status"] for item in result["research_plan"]["subquestions"]],
            ["blocked", "blocked"],
        )
        self.assertIn(
            "Covered state rejected", result["research_plan"]["subquestions"][0]["note"]
        )

    def test_resume_rejects_source_only_covered_before_unlocking_dependency(
        self,
    ) -> None:
        ledger = _FakeLedger()
        source_id = ledger.add_source("SQ1", with_graph=False)
        plan, _ = select_next_subquestion(_fixed_plan("topic", 2))
        injected = deepcopy(plan)
        injected["subquestions"][0]["status"] = "covered"
        injected["subquestions"][0]["evidence_source_ids"] = [source_id]
        injected = refresh_plan_status(injected)
        graph = build_research_graph(
            research_agent=_fake_research_agent(ledger),
            planner=_fixed_plan,
            budget_snapshot=ledger.snapshot,
            checkpointer=None,
            max_subquestions=2,
            max_research_cycles=4,
        )

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="topic", id="resume-source-only")],
                "research_topic": "topic",
                "research_plan": injected,
            }
        )

        self.assertEqual(len(ledger.sources), 1)
        self.assertEqual(
            [item["status"] for item in result["research_plan"]["subquestions"]],
            ["blocked", "blocked"],
        )
        self.assertIn(
            "covered_state_rejected",
            [item["event"] for item in result["research_events"]],
        )

    def test_sqlite_reopen_resumes_at_research_node_without_replanning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "research.sqlite"
            config = {"configurable": {"thread_id": "stage03a-resume"}}
            ledger = _FakeLedger()
            planner_calls: list[str] = []

            def planner(topic: str, maximum: int) -> ResearchPlan:
                planner_calls.append(topic)
                return _fixed_plan(topic, maximum)

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                interrupted = build_research_graph(
                    research_agent=_fake_research_agent(ledger),
                    planner=planner,
                    budget_snapshot=ledger.snapshot,
                    checkpointer=checkpointer,
                    max_subquestions=2,
                    max_research_cycles=4,
                    interrupt_before=["research_agent"],
                )
                interrupted.invoke(
                    {
                        "messages": [
                            HumanMessage(content="topic", id="resume-user-topic")
                        ],
                        "research_topic": "topic",
                    },
                    config=config,
                )
                saved = interrupted.get_state(config)
                self.assertEqual(saved.next, ("research_agent",))
                self.assertEqual(saved.values["active_subquestion_id"], "SQ1")

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                resumed = build_research_graph(
                    research_agent=_fake_research_agent(ledger),
                    planner=planner,
                    budget_snapshot=ledger.snapshot,
                    checkpointer=checkpointer,
                    max_subquestions=2,
                    max_research_cycles=4,
                )
                result = resumed.invoke(None, config=config)

            self.assertEqual(planner_calls, ["topic"])
            self.assertEqual(result["research_plan"]["status"], "completed")
            self.assertEqual(result["research_plan"]["coverage"], 1.0)
            sequences = [item["sequence"] for item in result["research_events"]]
            self.assertEqual(sequences, list(range(1, len(sequences) + 1)))

    def test_budget_snapshot_survives_reopen_after_research_before_evaluate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "budget-resume.sqlite"
            config = {"configurable": {"thread_id": "budget-resume"}}
            first_ledger = _FakeLedger()

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                interrupted = build_research_graph(
                    research_agent=_fake_research_agent(first_ledger),
                    planner=_fixed_plan,
                    budget_snapshot=first_ledger.snapshot,
                    checkpointer=checkpointer,
                    max_subquestions=2,
                    max_research_cycles=4,
                    interrupt_before=["evaluate"],
                )
                interrupted.invoke(
                    {
                        "messages": [HumanMessage(content="topic", id="budget-user")],
                        "research_topic": "topic",
                    },
                    config=config,
                )
                saved = interrupted.get_state(config)

            self.assertEqual(saved.next, ("evaluate",))
            self.assertEqual(
                saved.values["budget_state"]["successful_sources"][0]["source_id"],
                "S1",
            )

            fresh_ledger = _FakeLedger()
            fresh_ledger.restore(saved.values["budget_state"])
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                resumed = build_research_graph(
                    research_agent=_fake_research_agent(fresh_ledger),
                    planner=_fixed_plan,
                    budget_snapshot=fresh_ledger.snapshot,
                    checkpointer=checkpointer,
                    max_subquestions=2,
                    max_research_cycles=4,
                )
                result = resumed.invoke(None, config=config)

            evidence = [
                item["evidence_source_ids"][0]
                for item in result["research_plan"]["subquestions"]
            ]
            self.assertEqual(evidence, ["S1", "S2"])

    def test_stage02_message_only_checkpoint_starts_stage03a_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "legacy.sqlite"
            config = {"configurable": {"thread_id": "legacy-thread"}}

            def legacy_reply(state: MessagesState) -> dict[str, Any]:
                del state
                return {
                    "messages": [AIMessage(content="legacy reply", id="legacy-reply")]
                }

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                builder = StateGraph(MessagesState)
                builder.add_node("reply", legacy_reply)
                builder.add_edge(START, "reply")
                builder.add_edge("reply", END)
                legacy = builder.compile(checkpointer=checkpointer)
                legacy.invoke(
                    {
                        "messages": [
                            HumanMessage(content="legacy question", id="legacy-user")
                        ]
                    },
                    config=config,
                )

            ledger = _FakeLedger()
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                graph = build_research_graph(
                    research_agent=_fake_research_agent(ledger),
                    planner=_fixed_plan,
                    budget_snapshot=ledger.snapshot,
                    checkpointer=checkpointer,
                    max_subquestions=2,
                    max_research_cycles=4,
                )
                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content="new question", id="stage03a-user")
                        ],
                        "research_topic": "new question",
                    },
                    config=config,
                )

            message_ids = [message.id for message in result["messages"]]
            self.assertIn("legacy-user", message_ids)
            self.assertIn("legacy-reply", message_ids)
            self.assertEqual(result["research_plan"]["status"], "completed")

    def test_completed_plan_moves_to_history_when_a_new_topic_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "history.sqlite"
            config = {"configurable": {"thread_id": "history-thread"}}
            ledger = _FakeLedger()
            planner_calls: list[str] = []

            def planner(topic: str, maximum: int) -> ResearchPlan:
                del maximum
                planner_calls.append(topic)
                return create_research_plan(
                    topic,
                    ["Establish the facts"],
                    plan_id_factory=lambda: f"plan-{len(planner_calls)}",
                )

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                graph = build_research_graph(
                    research_agent=_fake_research_agent(ledger),
                    planner=planner,
                    budget_snapshot=ledger.snapshot,
                    checkpointer=checkpointer,
                    max_subquestions=1,
                    max_research_cycles=2,
                )
                graph.invoke(
                    {
                        "messages": [HumanMessage(content="first", id="first-user")],
                        "research_topic": "first",
                    },
                    config=config,
                )
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content="second", id="second-user")],
                        "research_topic": "second",
                    },
                    config=config,
                )

            self.assertEqual(planner_calls, ["first", "second"])
            self.assertEqual(result["research_plan"]["plan_id"], "plan-2")
            self.assertEqual(
                [item["plan_id"] for item in result["research_plan_history"]],
                ["plan-1"],
            )


if __name__ == "__main__":
    unittest.main()
