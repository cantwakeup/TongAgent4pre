"""Regression tests for TongAgent's offline evaluation dependency seams."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from agent_policy import EFFORT_POLICIES
from search_agent import (
    AgentRuntimeDependencies,
    ResearchBudget,
    build_agent,
    build_budgeted_tools,
)


class EvaluationInjectionTests(unittest.TestCase):
    """Keep offline providers behind the production semantic wrappers."""

    def test_injected_raw_tools_share_budget_and_metric_semantics(self) -> None:
        answer_url = "https://fixture.test/answer"
        raw_search_calls: list[tuple[str, int]] = []
        raw_fetch_calls: list[tuple[str, int]] = []

        @tool("fixture_search")
        def fixture_search(query: str, max_results: int = 5) -> str:
            """Return deterministic search fixtures without network access."""
            raw_search_calls.append((query, max_results))
            if query == "noise":
                results = [
                    {
                        "title": "Unrelated result",
                        "url": "https://fixture.test/noise",
                        "snippet": "Nothing about the requested fact.",
                        "relevance_score": 0,
                    }
                ]
                reported_relevant = 1
            else:
                results = [
                    {
                        "title": "Fixture answer",
                        "url": answer_url,
                        "snippet": "The fixture fact is documented here.",
                        "relevance_score": 100,
                    }
                ]
                reported_relevant = 0
            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "results": results,
                    "relevant_results": reported_relevant,
                }
            )

        quote = "The fixture fact is supported by the canonical fixture page."
        content = f"{quote} " + ("Deterministic supporting context. " * 20)

        @tool("fixture_fetch")
        def fixture_fetch(url: str, max_chars: int = 12_000) -> str:
            """Return a deterministic page fixture without network access."""
            raw_fetch_calls.append((url, max_chars))
            return json.dumps(
                {
                    "status": "success",
                    "url": url,
                    "title": "Fixture answer",
                    "content": content,
                    "content_chars": len(content),
                    "content_length": len(content),
                    "observed_content_length": len(content),
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "truncated": False,
                    "truncation_reasons": [],
                }
            )

        policy = EFFORT_POLICIES["low"]
        injected_budget = ResearchBudget(policy)
        tools, reused_budget = build_budgeted_tools(
            policy,
            raw_search_tool=fixture_search,
            raw_fetch_tool=fixture_fetch,
            budget=injected_budget,
        )
        search = next(item for item in tools if item.name == "web_search")
        fetch = next(item for item in tools if item.name == "fetch_url")
        injected_budget.configure_subquestions(["SQ1"])
        injected_budget.activate_subquestion("SQ1")

        noise = json.loads(search.invoke({"query": "noise", "max_results": 99}))
        relevant = json.loads(search.invoke({"query": "answer", "max_results": 99}))
        fetched = json.loads(fetch.invoke({"url": answer_url, "max_chars": 99_999}))
        injected_budget.record_evidence(
            source_id=fetched["source_id"],
            claim="The canonical fixture page supports the requested fixture fact.",
            quote=quote,
            stance="supports",
        )
        snapshot = injected_budget.snapshot()

        self.assertIs(reused_budget, injected_budget)
        self.assertEqual(raw_search_calls, [("noise", 4), ("answer", 4)])
        self.assertEqual(raw_fetch_calls, [(answer_url, 8_000)])
        self.assertEqual(noise["outcome"], "low_relevance")
        self.assertFalse(noise["relevant_search"])
        self.assertEqual(relevant["outcome"], "success")
        self.assertTrue(relevant["relevant_search"])
        self.assertEqual(snapshot["provider_successes"], 2)
        self.assertEqual(snapshot["nonempty_searches"], 2)
        self.assertEqual(snapshot["relevant_searches"], 1)
        self.assertEqual(snapshot["evidence_producing_searches"], 1)
        self.assertTrue(snapshot["tool_attempts"][1]["evidence_producing_search"])

    def test_injected_agent_build_skips_env_and_online_model_construction(self) -> None:
        policy = EFFORT_POLICIES["low"]
        budget = ResearchBudget(policy, strategy="adaptive")
        model = FakeListChatModel(responses=["research"])
        reviewer_model = FakeListChatModel(responses=["review"])
        planner = MagicMock(name="fixture-planner")
        middleware = AgentMiddleware()
        dependencies = AgentRuntimeDependencies(
            model=model,
            reviewer_model=reviewer_model,
            network_tools=[],
            budget=budget,
            planner=planner,
            middleware=(middleware,),
        )
        research_inner = MagicMock(name="research-inner")
        report_inner = MagicMock(name="report-inner")
        outer = MagicMock(name="outer")

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.dict("os.environ", {}, clear=True),
            patch("search_agent._load_local_env") as load_env,
            patch("search_agent.ChatOpenAI") as chat_openai,
            patch("search_agent.build_budgeted_tools") as build_tools,
            patch("search_agent.build_model_planner") as build_planner,
            patch(
                "search_agent._disable_general_purpose_subagent"
            ) as disable_general_purpose,
            patch(
                "search_agent.create_deep_agent",
                side_effect=[research_inner, report_inner],
            ) as create_deep_agent,
            patch(
                "search_agent.build_research_graph", return_value=outer
            ) as build_graph,
        ):
            bundle = build_agent(
                output_dir=Path(temp_dir),
                model_name="fixture-model",
                worker_model_name="fixture-reviewer",
                effort="low",
                mode="single",
                strategy="adaptive",
                topic="fixture topic",
                runtime_dependencies=dependencies,
            )

        load_env.assert_not_called()
        chat_openai.assert_not_called()
        build_tools.assert_not_called()
        build_planner.assert_not_called()
        disable_general_purpose.assert_called_once_with(model)
        self.assertIs(bundle.agent, outer)
        self.assertIs(bundle.budget, budget)
        self.assertEqual(create_deep_agent.call_count, 2)
        for call in create_deep_agent.call_args_list:
            self.assertIs(call.kwargs["model"], model)
            self.assertEqual(call.kwargs["middleware"], (middleware,))
        self.assertIs(build_graph.call_args.kwargs["planner"], planner)
        self.assertIs(build_graph.call_args.kwargs["budget_snapshot"].__self__, budget)


if __name__ == "__main__":
    unittest.main()
