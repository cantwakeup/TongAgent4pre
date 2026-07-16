"""Offline tests for TongAgent's explicit Stage 03A research workflow."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph import MessagesState

from research_graph import (
    DraftSubquestion,
    build_model_planner,
    build_research_graph,
    build_research_state_tools,
    calculate_plan_coverage,
    create_research_plan,
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

    def add_source(self) -> str:
        """Add one unique successful source and return its ID."""
        source_id = f"S{len(self.sources) + 1}"
        self.sources.append(
            {
                "source_id": source_id,
                "url": f"https://example.com/{source_id}",
                "title": source_id,
                "content_chars": 800,
            }
        )
        return source_id

    def snapshot(self) -> dict[str, Any]:
        """Return the budget shape consumed by the outer graph."""
        return {
            "effort": "test",
            "search_calls": len(self.sources),
            "max_searches": 10,
            "fetch_calls": len(self.sources),
            "max_fetches": 10,
            "min_successful_sources": 1,
            "successful_sources": list(self.sources),
            "failed_sources": [],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore successful sources as a fresh process would."""
        self.sources = [dict(item) for item in snapshot["successful_sources"]]


def _fake_research_agent(ledger: _FakeLedger, *, produce_evidence: bool = True) -> Any:
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
        source_id = ledger.add_source()
        plan = transition_subquestion(
            state["research_plan"],
            str(active),
            "covered",
            evidence_source_ids=[source_id],
            note="covered by fake evidence",
        )
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

        covered = transition_subquestion(
            plan, "SQ1", "covered", evidence_source_ids=["S1"]
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
            plan, "SQ1", "covered", evidence_source_ids=["S1"]
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

    def test_state_tool_requires_active_current_step_ledger_evidence(self) -> None:
        ledger = _FakeLedger()
        plan, active = select_next_subquestion(_fixed_plan("topic", 2))
        runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [],
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

        source_id = ledger.add_source()
        accepted = update_tool.func(
            subquestion_id="SQ1",
            status="covered",
            evidence_source_ids=[source_id],
            note="ledger-backed",
            runtime=runtime,
        )
        self.assertEqual(accepted.update["research_plan"]["coverage"], 0.5)

        stale_runtime = ToolRuntime(
            state={
                "research_plan": plan,
                "research_events": [],
                "active_subquestion_id": active,
                "active_source_ids_before": [source_id],
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
        self.assertIn("active research step", json.loads(stale)["error"])


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
