"""Integration tests for the Stage 03D adaptive outer-graph loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent_policy import EFFORT_POLICIES
from research_graph import (
    build_research_graph,
    create_research_plan,
    transition_subquestion,
)
from research_state import ResearchPlan, TongAgentState
from search_agent import (
    ResearchBudget,
    _adaptive_audit_errors,
    _restore_checkpointed_report,
)


def _one_step_plan(topic: str, maximum: int) -> ResearchPlan:
    del maximum
    return create_research_plan(
        topic,
        ["Establish the adaptive fact from two independent sources"],
        max_subquestions=1,
        plan_id_factory=lambda: "plan-adaptive",
    )


def _adaptive_fake_agent(
    budget: ResearchBudget, *, report_path: Path | None = None
) -> Any:
    """Require two sources even though the initial adaptive slice exposes one."""

    def reply(state: TongAgentState) -> dict[str, Any]:
        if "[FINAL SYNTHESIS]" in str(state["messages"][-1].content):
            if report_path is not None:
                report_path.write_text("# Durable adaptive report\n")
            return {"messages": [AIMessage(content="report ready", id="fake-report")]}
        active = str(state["active_subquestion_id"])
        if budget.reserve_search():
            result_url = f"https://source{budget.search_calls}.example/fact"
            budget.record_tool_attempt(
                tool_name="web_search",
                target=f"adaptive query {budget.search_calls}",
                payload={
                    "status": "success",
                    "results": [
                        {
                            "url": result_url,
                            "relevance_score": 100,
                        }
                    ],
                    "relevant_results": 1,
                    "search_quality": "relevant",
                },
            )
        if budget.reserve_fetch():
            sequence = budget.fetch_calls
            quote = f"Official source {sequence} confirms the adaptive research fact."
            content = (quote + " Supporting context.") * 20
            payload = {
                "status": "success",
                "url": f"https://source{sequence}.example/fact",
                "title": f"Source {sequence}",
                "content": content,
                "content_chars": len(content),
                "evidence_quality": "full",
            }
            source_id = budget.record_fetch(payload)
            assert source_id is not None
            budget.record_tool_attempt(
                tool_name="fetch_url", target=payload["url"], payload=payload
            )
            budget.record_evidence(
                source_id=source_id,
                claim=f"Source {sequence} independently confirms the adaptive fact.",
                quote=quote,
                stance="supports",
            )
        snapshot = budget.snapshot()
        active_claims = [
            item for item in snapshot["claims"] if item["subquestion_id"] == active
        ]
        if len(active_claims) < 2:
            return {
                "messages": [
                    AIMessage(
                        content="more evidence required",
                        id=f"fake-progress-{state['research_cycles']}",
                    )
                ]
            }
        claim_ids = [str(item["claim_id"]) for item in active_claims]
        evidence_source_ids = sorted(
            {
                str(unit["source_id"])
                for unit in snapshot["evidence_units"]
                if unit["claim_id"] in set(claim_ids)
            }
        )
        plan = transition_subquestion(
            state["research_plan"],
            active,
            "covered",
            evidence_source_ids=evidence_source_ids,
            claim_ids=claim_ids,
            note="covered after an adaptive reserve release",
        )
        return {
            "messages": [AIMessage(content="covered", id="fake-covered")],
            "research_plan": plan,
        }

    builder = StateGraph(TongAgentState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile()


def _build_graph(
    budget: ResearchBudget,
    *,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    report_path: Path | None = None,
    write_report: bool = True,
    budget_grant: Any | None = None,
) -> Any:
    report_target = report_path if write_report else None
    return build_research_graph(
        research_agent=_adaptive_fake_agent(
            budget,
            report_path=report_target,
        ),
        report_agent=_adaptive_fake_agent(budget, report_path=report_target),
        planner=_one_step_plan,
        budget_snapshot=budget.snapshot,
        budget_configure=budget.configure_subquestions,
        budget_activate=budget.activate_subquestion,
        budget_grant=budget_grant or budget.grant_subquestion,
        report_read=(
            (lambda: report_path.read_text() if report_path.is_file() else "")
            if report_path is not None
            else None
        ),
        report_clear=(
            (lambda: report_path.unlink(missing_ok=True))
            if report_path is not None
            else None
        ),
        checkpointer=checkpointer,
        max_subquestions=1,
        max_research_cycles=5,
        interrupt_before=interrupt_before,
        strategy="adaptive",
        config_fingerprint="fingerprint",
        hard_effort="low",
        pinned_model="test-model",
        pinned_topology="single",
        max_escalations=2,
    )


class AdaptiveGraphTests(unittest.TestCase):
    """Verify that controller decisions change hard grants and survive resume."""

    def test_evidence_gap_releases_reserve_without_resetting_usage(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        result = _build_graph(budget).invoke(
            {
                "messages": [HumanMessage(content="topic", id="adaptive-user")],
                "research_topic": "topic",
            }
        )

        snapshot = budget.snapshot()
        control = result["adaptive_control"]
        actions = [item["action"] for item in control["decision_history"]]
        self.assertEqual(result["research_plan"]["status"], "completed")
        self.assertEqual(control["escalation_count"], 1)
        self.assertIn("expand_budget", actions)
        self.assertEqual(actions[-1], "finish_success")
        self.assertEqual(snapshot["search_calls"], 2)
        self.assertEqual(snapshot["fetch_calls"], 2)
        self.assertEqual(snapshot["subquestion_usage"]["SQ1"]["fetch_calls"], 2)
        self.assertEqual(snapshot["subquestion_limits"]["SQ1"]["max_fetches"], 2)
        self.assertEqual(len(snapshot["applied_grant_ids"]), 1)
        self.assertEqual(snapshot["evidence_graph_version"], 1)

    def test_resume_before_controller_applies_exactly_one_grant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "adaptive.sqlite"
            config = {"configurable": {"thread_id": "adaptive-resume"}}
            first_budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                interrupted = _build_graph(
                    first_budget,
                    checkpointer=checkpointer,
                    interrupt_before=["adaptive_control"],
                )
                interrupted.invoke(
                    {
                        "messages": [HumanMessage(content="topic", id="resume-user")],
                        "research_topic": "topic",
                    },
                    config=config,
                )
                saved = interrupted.get_state(config)

            self.assertEqual(saved.next, ("adaptive_control",))
            fresh_budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
            fresh_budget.restore(saved.values["budget_state"], strict_policy=True)
            fresh_budget.configure_subquestions(["SQ1"])
            fresh_budget.activate_subquestion(saved.values.get("active_subquestion_id"))
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                resumed = _build_graph(fresh_budget, checkpointer=checkpointer)
                result = resumed.invoke(None, config=config)

            self.assertEqual(result["research_plan"]["status"], "completed")
            self.assertEqual(result["adaptive_control"]["escalation_count"], 1)
            self.assertEqual(len(fresh_budget.snapshot()["applied_grant_ids"]), 1)

    def test_grant_replay_preserves_original_transition_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "grant-crash.sqlite"
            config = {"configurable": {"thread_id": "grant-crash"}}
            budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
            crashed = False

            def throwing_budget_grant(
                decision_id: str,
                subquestion_id: str,
                **deltas: Any,
            ) -> dict[str, Any]:
                nonlocal crashed
                result = budget.grant_subquestion(
                    decision_id,
                    subquestion_id,
                    **deltas,
                )
                if result["applied"] and not crashed:
                    crashed = True
                    raise RuntimeError("crash after grant side effect")
                return result

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                graph = _build_graph(
                    budget,
                    checkpointer=checkpointer,
                    budget_grant=throwing_budget_grant,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "crash after grant side effect"
                ):
                    graph.invoke(
                        {
                            "messages": [
                                HumanMessage(content="topic", id="crash-user")
                            ],
                            "research_topic": "topic",
                        },
                        config=config,
                    )

                saved = graph.get_state(config)
                self.assertEqual(saved.next, ("adaptive_control",))
                result = graph.invoke(None, config=config)

            control = result["adaptive_control"]
            ledger = budget.snapshot()
            expansions = [
                item
                for item in control["decision_history"]
                if item["action"] == "expand_budget"
            ]
            self.assertEqual(result["research_plan"]["status"], "completed")
            self.assertEqual(len(expansions), 1)
            self.assertEqual(len(ledger["applied_grant_ids"]), 1)
            self.assertEqual(len(ledger["applied_grants"]), 1)
            self.assertEqual(expansions[0]["budget_before"]["granted_searches"], 1)
            self.assertEqual(expansions[0]["budget_before"]["granted_fetches"], 1)
            self.assertEqual(expansions[0]["budget_after"]["granted_searches"], 2)
            self.assertEqual(expansions[0]["budget_after"]["granted_fetches"], 2)
            self.assertEqual(
                _adaptive_audit_errors(
                    adaptive_control=control,
                    ledger=ledger,
                    config_fingerprint="fingerprint",
                    model_name="test-model",
                    topology="single",
                    max_escalations=2,
                ),
                [],
            )

    def test_report_survives_finish_resume_into_a_fresh_backend_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "report-resume.sqlite"
            first_report = Path(temp_dir) / "first-run" / "report.md"
            second_report = Path(temp_dir) / "second-run" / "report.md"
            first_report.parent.mkdir()
            second_report.parent.mkdir()
            config = {"configurable": {"thread_id": "report-resume"}}

            first_budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                interrupted = _build_graph(
                    first_budget,
                    checkpointer=checkpointer,
                    interrupt_before=["finish"],
                    report_path=first_report,
                )
                interrupted.invoke(
                    {
                        "messages": [HumanMessage(content="topic", id="report-user")],
                        "research_topic": "topic",
                    },
                    config=config,
                )
                saved = interrupted.get_state(config)

            self.assertEqual(saved.next, ("finish",))
            self.assertEqual(
                saved.values["report_markdown"], "# Durable adaptive report\n"
            )
            fresh_budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
            fresh_budget.restore(saved.values["budget_state"], strict_policy=True)
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                resumed = _build_graph(
                    fresh_budget,
                    checkpointer=checkpointer,
                    report_path=second_report,
                )
                result = resumed.invoke(None, config=config)

            self.assertFalse(second_report.exists())
            self.assertTrue(_restore_checkpointed_report(second_report, result))
            self.assertEqual(second_report.read_text(), "# Durable adaptive report\n")

    def test_report_phase_removes_early_artifact_before_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.md"
            report_path.write_text("UNSAFE EARLY REPORT")
            budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")

            result = _build_graph(
                budget,
                report_path=report_path,
                write_report=False,
            ).invoke(
                {
                    "messages": [HumanMessage(content="topic")],
                    "research_topic": "topic",
                }
            )

            self.assertFalse(report_path.exists())
            self.assertEqual(result["report_markdown"], "")

    def test_fail_closed_blocks_plan_and_hides_research_messages_from_report(
        self,
    ) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["medium"], strategy="adaptive")
        budget.configure_subquestions(["SQ1", "SQ2"])
        budget.activate_subquestion("SQ1")
        quote = "The official source confirms the integrity sentinel fact."
        content = (quote + " Supporting context.") * 20
        source_id = budget.record_fetch(
            {
                "status": "success",
                "url": "https://integrity.example/fact",
                "title": "Integrity Source",
                "content": content,
                "content_chars": len(content),
                "evidence_quality": "full",
            }
        )
        assert source_id == "S1"
        recorded = budget.record_evidence(
            source_id=source_id,
            claim="The official source confirms the integrity sentinel fact.",
            quote=quote,
            stance="supports",
        )
        claim_id = str(recorded["claim"]["claim_id"])
        plan = create_research_plan(
            "integrity topic",
            ["Establish the first fact", "Establish the second fact"],
            max_subquestions=2,
            plan_id_factory=lambda: "plan-integrity",
        )
        plan = transition_subquestion(plan, "SQ1", "researching")
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=[source_id],
            claim_ids=[claim_id],
            note="covered before corruption",
        )
        report_inputs: list[list[str]] = []
        research_sentinel = "RESEARCH_HISTORY_MUST_NOT_REACH_REPORT"

        def research_reply(state: TongAgentState) -> dict[str, Any]:
            self.assertNotIn("[FINAL SYNTHESIS]", str(state["messages"][-1].content))
            budget.evidence_graph.evidence_units[0]["quote_sha256"] = "0" * 64
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"{research_sentinel} {claim_id} {source_id} "
                            "https://integrity.example/fact"
                        )
                    )
                ]
            }

        def report_reply(state: TongAgentState) -> dict[str, Any]:
            report_inputs.append(
                [str(message.content) for message in state["messages"]]
            )
            return {"messages": [AIMessage(content="safe partial report")]}

        research_builder = StateGraph(TongAgentState)
        research_builder.add_node("reply", research_reply)
        research_builder.add_edge(START, "reply")
        research_builder.add_edge("reply", END)
        report_builder = StateGraph(TongAgentState)
        report_builder.add_node("reply", report_reply)
        report_builder.add_edge(START, "reply")
        report_builder.add_edge("reply", END)
        graph = build_research_graph(
            research_agent=research_builder.compile(),
            report_agent=report_builder.compile(),
            planner=lambda topic, maximum: plan,
            budget_snapshot=budget.snapshot,
            budget_configure=budget.configure_subquestions,
            budget_activate=budget.activate_subquestion,
            budget_grant=budget.grant_subquestion,
            checkpointer=None,
            max_subquestions=2,
            max_research_cycles=4,
            strategy="adaptive",
            config_fingerprint="integrity-fingerprint",
            hard_effort="medium",
            pinned_model="test-model",
            pinned_topology="single",
            max_escalations=2,
        )

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="integrity topic")],
                "research_topic": "integrity topic",
                "research_plan": plan,
            }
        )

        self.assertEqual(result["research_plan"]["status"], "partial")
        self.assertEqual(result["research_plan"]["coverage"], 0.0)
        for item in result["research_plan"]["subquestions"]:
            self.assertEqual(item["status"], "blocked")
            self.assertEqual(item["evidence_source_ids"], [])
            self.assertEqual(item["claim_ids"], [])
            self.assertEqual(item["conflict_ids"], [])
        self.assertEqual(
            result["adaptive_control"]["decision_history"][-1]["action"],
            "fail_closed",
        )
        self.assertEqual(len(report_inputs), 1)
        self.assertEqual(len(report_inputs[0]), 1)
        report_input = report_inputs[0][0]
        self.assertNotIn(research_sentinel, report_input)
        self.assertNotIn(claim_id, report_input)
        self.assertNotIn(source_id, report_input)
        self.assertNotIn("https://integrity.example/fact", report_input)
        self.assertIn("CANONICAL SOURCE LEDGER:\n[]", report_input)
        self.assertIn('"claims": []', report_input)


if __name__ == "__main__":
    unittest.main()
