"""Tests for shared tool budgets and the structured source ledger."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent_policy import EFFORT_POLICIES
from search_agent import build_budgeted_tools


class ResearchBudgetTests(unittest.TestCase):
    """Verify hard limits, source IDs, and URL deduplication."""

    def test_search_budget_is_enforced(self) -> None:
        tools, budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        search = next(item for item in tools if item.name == "web_search")
        raw_result = json.dumps({"query": "test", "results": []})

        with patch("search_agent.web_search") as raw_search:
            raw_search.invoke.return_value = raw_result
            first = json.loads(search.invoke({"query": "one"}))
            second = json.loads(search.invoke({"query": "two"}))
            third = json.loads(search.invoke({"query": "three"}))

        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        self.assertEqual(third["status"], "budget_exceeded")
        self.assertEqual(budget.snapshot()["search_calls"], 2)

    def test_plan_budget_is_partitioned_between_subquestions(self) -> None:
        _, budget = build_budgeted_tools(EFFORT_POLICIES["medium"])
        budget.configure_subquestions(["SQ1", "SQ2"])

        self.assertFalse(budget.reserve_search())
        self.assertEqual(
            budget.budget_denial("search")["reason"], "no_active_subquestion"
        )
        budget.activate_subquestion("SQ1")
        self.assertTrue(budget.reserve_search())
        self.assertTrue(budget.reserve_search())
        self.assertFalse(budget.reserve_search())
        self.assertEqual(
            budget.budget_denial("search")["reason"],
            "subquestion_budget_exceeded",
        )

        budget.activate_subquestion("SQ2")
        self.assertTrue(budget.reserve_search())
        self.assertTrue(budget.reserve_search())
        self.assertFalse(budget.reserve_search())
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["search_calls"], 4)
        self.assertEqual(
            snapshot["subquestion_limits"],
            {
                "SQ1": {"max_searches": 2, "max_fetches": 3},
                "SQ2": {"max_searches": 2, "max_fetches": 3},
            },
        )
        self.assertEqual(snapshot["subquestion_usage"]["SQ1"]["search_calls"], 2)
        self.assertEqual(snapshot["subquestion_usage"]["SQ2"]["search_calls"], 2)

    def test_successful_fetches_receive_stable_source_ids(self) -> None:
        tools, budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        fetch = next(item for item in tools if item.name == "fetch_url")

        def result_for(call: dict[str, object]) -> str:
            url = str(call["url"])
            return json.dumps(
                {
                    "status": "success",
                    "url": url,
                    "title": f"Title for {url}",
                    "content": "evidence",
                    "content_chars": 800,
                }
            )

        with patch("search_agent.fetch_url") as raw_fetch:
            raw_fetch.invoke.side_effect = result_for
            first = json.loads(fetch.invoke({"url": "https://example.com/a"}))
            duplicate = json.loads(
                fetch.invoke({"url": "https://example.com/a#section"})
            )
            second = json.loads(fetch.invoke({"url": "https://example.com/b"}))

        self.assertEqual(first["source_id"], "S1")
        self.assertEqual(duplicate["source_id"], "S1")
        self.assertEqual(second["source_id"], "S2")
        self.assertEqual(len(budget.snapshot()["successful_sources"]), 2)

    def test_short_pages_do_not_count_as_evidence(self) -> None:
        tools, budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        fetch = next(item for item in tools if item.name == "fetch_url")
        short_page = json.dumps(
            {
                "status": "success",
                "url": "https://example.com/redirect",
                "title": "Redirecting",
                "content": "Go to the new documentation.",
                "content_chars": 28,
            }
        )

        with patch("search_agent.fetch_url") as raw_fetch:
            raw_fetch.invoke.return_value = short_page
            result = json.loads(fetch.invoke({"url": "https://example.com/redirect"}))

        self.assertEqual(result["status"], "insufficient_content")
        self.assertNotIn("source_id", result)
        self.assertEqual(budget.snapshot()["successful_sources"], [])

    def test_short_page_counts_only_after_full_same_host_anchor(self) -> None:
        tools, budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        fetch = next(item for item in tools if item.name == "fetch_url")
        responses = [
            {
                "status": "success",
                "url": "https://official.example/about",
                "title": "Official about page",
                "content": "x" * 800,
                "content_chars": 800,
            },
            {
                "status": "success",
                "url": "https://official.example/program",
                "title": "Official program contact",
                "content": "x" * 430,
                "content_chars": 430,
            },
        ]

        with patch("search_agent.fetch_url") as raw_fetch:
            raw_fetch.invoke.side_effect = [json.dumps(item) for item in responses]
            anchor = json.loads(fetch.invoke({"url": responses[0]["url"]}))
            limited = json.loads(fetch.invoke({"url": responses[1]["url"]}))

        self.assertEqual(anchor["evidence_quality"], "full")
        self.assertEqual(limited["status"], "success")
        self.assertEqual(limited["source_id"], "S2")
        self.assertEqual(limited["evidence_quality"], "limited")
        self.assertIn("same host", limited["quality_reason"])
        self.assertEqual(
            budget.snapshot()["successful_sources"][1]["evidence_quality"],
            "limited",
        )

    def test_unanchored_short_page_does_not_count(self) -> None:
        tools, budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        fetch = next(item for item in tools if item.name == "fetch_url")
        short_page = json.dumps(
            {
                "status": "success",
                "url": "https://unknown.example/program",
                "title": "Unanchored program page",
                "content": "x" * 430,
                "content_chars": 430,
            }
        )

        with patch("search_agent.fetch_url") as raw_fetch:
            raw_fetch.invoke.return_value = short_page
            result = json.loads(
                fetch.invoke({"url": "https://unknown.example/program"})
            )

        self.assertEqual(result["status"], "insufficient_content")
        self.assertIn("no full-length source", result["error"])
        self.assertEqual(budget.snapshot()["successful_sources"], [])

    def test_checkpoint_snapshot_restores_budget_and_source_ids(self) -> None:
        _, budget = build_budgeted_tools(EFFORT_POLICIES["low"])

        budget.restore(
            {
                "search_calls": 99,
                "fetch_calls": 2,
                "successful_sources": [
                    {
                        "source_id": "S1",
                        "url": "https://example.com/source",
                        "title": "Source",
                        "content_chars": 800,
                    }
                ],
                "failed_sources": [
                    {
                        "url": "https://example.com/failed",
                        "status": "error",
                        "error": "HTTP 503",
                    }
                ],
            }
        )

        snapshot = budget.snapshot()
        self.assertEqual(snapshot["search_calls"], EFFORT_POLICIES["low"].max_searches)
        self.assertEqual(snapshot["fetch_calls"], 2)
        self.assertEqual(snapshot["successful_sources"][0]["source_id"], "S1")
        self.assertEqual(snapshot["failed_sources"][0]["error"], "HTTP 503")

        _, next_plan_budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        next_plan_budget.restore(snapshot, reset_usage=True)
        next_snapshot = next_plan_budget.snapshot()
        self.assertEqual(next_snapshot["search_calls"], 0)
        self.assertEqual(next_snapshot["fetch_calls"], 0)
        self.assertEqual(next_snapshot["failed_sources"], [])
        self.assertEqual(next_snapshot["subquestion_limits"], {})
        self.assertEqual(next_snapshot["successful_sources"][0]["source_id"], "S1")
        self.assertTrue(next_plan_budget.reserve_search())
        self.assertTrue(next_plan_budget.reserve_fetch())
        next_source = next_plan_budget.record_fetch(
            {
                "status": "success",
                "url": "https://example.com/next",
                "title": "Next",
                "content_chars": 900,
            }
        )
        self.assertEqual(next_source, "S2")

    def test_checkpoint_restores_active_subquestion_slice(self) -> None:
        _, budget = build_budgeted_tools(EFFORT_POLICIES["medium"])
        budget.configure_subquestions(["SQ1", "SQ2"])
        budget.activate_subquestion("SQ1")
        self.assertTrue(budget.reserve_search())
        snapshot = budget.snapshot()

        _, restored = build_budgeted_tools(EFFORT_POLICIES["medium"])
        restored.restore(snapshot)

        self.assertEqual(restored.snapshot()["active_subquestion_id"], "SQ1")
        self.assertTrue(restored.reserve_search())
        self.assertFalse(restored.reserve_search())
        restored.activate_subquestion("SQ2")
        self.assertTrue(restored.reserve_search())


if __name__ == "__main__":
    unittest.main()
