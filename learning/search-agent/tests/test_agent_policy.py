"""Tests for deterministic topology and effort policies."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_policy import EFFORT_POLICIES, policy_prompt, resolve_topology
from search_agent import _build_subagents, build_agent


class AgentPolicyTests(unittest.TestCase):
    """Verify user overrides and the initial auto-routing heuristic."""

    def test_explicit_mode_wins(self) -> None:
        self.assertEqual(resolve_topology("single", "xhigh", "复杂综述"), "single")
        self.assertEqual(resolve_topology("multi", "low", "简单问题"), "multi")

    def test_auto_routes_by_effort_and_complexity(self) -> None:
        self.assertEqual(resolve_topology("auto", "low", "Aqours 是什么"), "single")
        self.assertEqual(
            resolve_topology("auto", "medium", "比较 LangGraph 与 Deep Agents"), "multi"
        )
        self.assertEqual(resolve_topology("auto", "high", "Aqours 是什么"), "multi")

    def test_policy_prompt_exposes_enforced_budget(self) -> None:
        prompt = policy_prompt(EFFORT_POLICIES["xhigh"], "multi")

        self.assertIn("at most 12 searches", prompt)
        self.assertIn("at most 5 explicit research subquestions", prompt)
        self.assertIn("at least 4 successfully fetched", prompt)
        self.assertIn("reviewer subagent", prompt)

    def test_subagent_roles_follow_topology_and_effort(self) -> None:
        model = MagicMock()

        single = _build_subagents(
            topology="single",
            policy=EFFORT_POLICIES["xhigh"],
            model=model,
            reviewer_model=model,
            tools=[],
        )
        medium = _build_subagents(
            topology="multi",
            policy=EFFORT_POLICIES["medium"],
            model=model,
            reviewer_model=model,
            tools=[],
        )
        with patch("search_agent.create_agent", return_value=MagicMock()):
            high = _build_subagents(
                topology="multi",
                policy=EFFORT_POLICIES["high"],
                model=model,
                reviewer_model=model,
                tools=[],
            )

        self.assertEqual(single, [])
        self.assertEqual([agent["name"] for agent in medium], ["researcher"])
        self.assertEqual([agent["name"] for agent in high], ["researcher", "reviewer"])

    def test_report_agent_has_no_research_or_evidence_tools(self) -> None:
        research_inner = MagicMock(name="research-inner")
        report_inner = MagicMock(name="report-inner")
        outer = MagicMock(name="outer")
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.dict(
                "os.environ",
                {
                    "SEARCH_AGENT_API_KEY": "test-key",
                    "SEARCH_AGENT_BASE_URL": "https://api.example/v1",
                },
            ),
            patch("search_agent.ChatOpenAI", return_value=MagicMock()),
            patch("search_agent._disable_general_purpose_subagent"),
            patch(
                "search_agent.create_deep_agent",
                side_effect=[research_inner, report_inner],
            ) as create_deep_agent,
            patch("search_agent.build_model_planner", return_value=MagicMock()),
            patch(
                "search_agent.build_research_graph", return_value=outer
            ) as build_graph,
        ):
            bundle = build_agent(
                output_dir=Path(temp_dir),
                model_name="test-model",
                effort="low",
                mode="single",
                strategy="adaptive",
                topic="topic",
            )

        self.assertIs(bundle.agent, outer)
        self.assertEqual(create_deep_agent.call_count, 2)
        research_tool_names = {
            tool.name for tool in create_deep_agent.call_args_list[0].kwargs["tools"]
        }
        self.assertIn("get_source_ledger", research_tool_names)
        self.assertIn("get_evidence_graph", research_tool_names)
        self.assertEqual(create_deep_agent.call_args_list[1].kwargs["tools"], [])
        self.assertIs(
            build_graph.call_args.kwargs["report_agent"],
            report_inner,
        )
        self.assertEqual(build_graph.call_args.kwargs["max_research_cycles"], 6)


if __name__ == "__main__":
    unittest.main()
