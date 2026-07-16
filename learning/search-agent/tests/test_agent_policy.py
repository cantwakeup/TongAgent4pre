"""Tests for deterministic topology and effort policies."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agent_policy import EFFORT_POLICIES, policy_prompt, resolve_topology
from search_agent import _build_subagents


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


if __name__ == "__main__":
    unittest.main()
