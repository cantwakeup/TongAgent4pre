"""Offline regressions distilled from the failed Stage F canary topics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent_policy import EFFORT_POLICIES
from retrieval_backend import RetrievalSession, SearchBroker
from retrieval_providers import ProviderSearchResponse, SearchResult
from retrieval_quality import (
    MIN_SEARCH_RELEVANCE_SCORE,
    assess_search_relevance,
    deterministic_query_rewrite,
    search_relevance_score,
)
from search_agent import (
    _search_relevance_score,
    build_budgeted_tools,
    create_retrieval_tools,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "stage_f_search_regressions.json"
)


def _load_fixture() -> dict[str, object]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Stage F search fixture must be a JSON object")
    return payload


class StageFSearchRegressionTests(unittest.TestCase):
    """Keep the real pilot's one-word false positives from recurring."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cases = cls.fixture.get("cases")
        if not isinstance(cases, list):
            raise TypeError("Stage F search fixture cases must be a list")
        cls.cases = cases

    def test_fixture_is_sanitized_and_has_no_dataset_answer_hints(self) -> None:
        raw = FIXTURE_PATH.read_text(encoding="utf-8")

        self.assertNotIn('"source_urls"', raw)
        self.assertNotIn('"reference_answer"', raw)
        self.assertNotIn("frames-pilot-seed17.jsonl", raw)
        self.assertEqual(self.fixture["schema_version"], 1)
        self.assertEqual(len(self.cases), 4)
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                for result in [
                    *case["drift_results"],
                    *case["relevant_results"],
                ]:
                    host = str(result["url"]).split("/", 3)[2]
                    self.assertTrue(host.endswith((".invalid", ".example")))

    def test_generic_red_new_san_and_nelson_hits_stay_below_threshold(self) -> None:
        for case in self.cases:
            for result in case["drift_results"]:
                with self.subTest(case=case["case_id"], title=result["title"]):
                    self.assertLess(
                        _search_relevance_score(case["query"], result),
                        MIN_SEARCH_RELEVANCE_SCORE,
                    )

    def test_substrings_and_single_common_terms_never_clear_the_gate(self) -> None:
        probes = (
            (
                '"SAN" storage',
                {
                    "title": "Sandy beaches",
                    "url": "https://noise.invalid/sandy",
                    "snippet": "A storage-free travel story.",
                },
            ),
            (
                '"AI" research',
                {
                    "title": "Said another witness",
                    "url": "https://noise.invalid/said",
                    "snippet": "A generic research note.",
                },
            ),
            (
                "red",
                {
                    "title": "Red paint",
                    "url": "https://noise.invalid/red",
                    "snippet": "One ordinary word.",
                },
            ),
        )

        for query, result in probes:
            with self.subTest(query=query):
                self.assertLess(
                    search_relevance_score(query, result),
                    MIN_SEARCH_RELEVANCE_SCORE,
                )

    def test_entity_and_phrase_coverage_accepts_on_topic_results(self) -> None:
        for case in self.cases:
            drift_scores = [
                search_relevance_score(case["query"], result)
                for result in case["drift_results"]
            ]
            for result in case["relevant_results"]:
                with self.subTest(case=case["case_id"], title=result["title"]):
                    relevant_score = search_relevance_score(case["query"], result)
                    self.assertGreaterEqual(
                        relevant_score,
                        MIN_SEARCH_RELEVANCE_SCORE,
                    )
                    self.assertGreater(relevant_score, max(drift_scores))

    def test_matching_years_and_numbers_raise_relevance(self) -> None:
        numeric_cases = [case for case in self.cases if "numeric_weight_probe" in case]

        self.assertEqual(len(numeric_cases), 2)
        for case in numeric_cases:
            matching = assess_search_relevance(
                case["query"],
                case["numeric_weight_probe"]["matching"],
            )
            mismatch = assess_search_relevance(
                case["query"],
                case["numeric_weight_probe"]["mismatching"],
            )
            with self.subTest(case=case["case_id"]):
                self.assertGreater(matching["score"], mismatch["score"])
                self.assertTrue(matching["matched_years"])
                self.assertEqual(mismatch["matched_years"], [])

    def test_deterministic_rewrite_preserves_entities_and_numeric_constraints(
        self,
    ) -> None:
        for case in self.cases:
            first = deterministic_query_rewrite(case["query"])
            second = deterministic_query_rewrite(case["query"])
            with self.subTest(case=case["case_id"]):
                self.assertIsInstance(first, str)
                self.assertEqual(first, second)
                self.assertTrue(first.strip())
                self.assertNotEqual(first.casefold(), case["query"].casefold())
                rewritten = first.casefold()
                for alternatives in case["rewrite_required_term_groups"]:
                    self.assertTrue(
                        any(term.casefold() in rewritten for term in alternatives),
                        f"rewrite lost required terms {alternatives!r}: {first!r}",
                    )

    def test_search_agent_uses_the_entity_aware_scorer(self) -> None:
        for case in self.cases:
            for result in [
                *case["drift_results"],
                *case["relevant_results"],
            ]:
                with self.subTest(case=case["case_id"], title=result["title"]):
                    self.assertEqual(
                        _search_relevance_score(case["query"], result),
                        search_relevance_score(case["query"], result),
                    )

    def test_low_entity_coverage_uses_rewritten_backup_query(self) -> None:
        case = next(
            item
            for item in self.cases
            if item["case_id"] == "frames-0664-mandela-imprisonment"
        )

        class Provider:
            def __init__(self, name: str, rows: list[dict[str, str]]) -> None:
                self.name = name
                self.rows = rows
                self.queries: list[str] = []

            def search(
                self,
                query: str,
                max_results: int,
            ) -> ProviderSearchResponse:
                self.queries.append(query)
                return ProviderSearchResponse(
                    provider=self.name,
                    query=query,
                    status="success",
                    results=[
                        SearchResult(
                            title=row["title"],
                            url=row["url"],
                            snippet=row["snippet"],
                            provider=self.name,
                            provider_rank=rank,
                        )
                        for rank, row in enumerate(self.rows[:max_results], 1)
                    ],
                )

        primary = Provider("duckduckgo_html", case["drift_results"])
        backup = Provider("bing_rss", case["relevant_results"])
        session = RetrievalSession(SearchBroker([primary, backup]))
        search, _ = create_retrieval_tools(session)
        result = json.loads(search.invoke({"query": case["query"], "max_results": 5}))

        rewritten = backup.queries[0]
        self.assertIn('"Nelson Mandela"', rewritten)
        self.assertNotEqual(rewritten, case["query"])
        self.assertGreaterEqual(result["relevant_results"], 1)

    def test_access_blocked_url_and_host_are_suppressed_without_spending_fetch(
        self,
    ) -> None:
        guard = Mock(return_value=None)
        tools, budget = build_budgeted_tools(
            EFFORT_POLICIES["medium"],
            external_guard=guard,
        )
        fetch = next(tool for tool in tools if tool.name == "fetch_url")
        blocked = {
            "status": "error",
            "url": "https://blocked.example/first",
            "failure_type": "access_blocked",
            "error": "HTTP 403",
            "retry_with_another_source": True,
        }
        alternate = {
            "status": "success",
            "url": "https://alternate.example/page",
            "title": "Alternate evidence",
            "content": "evidence " * 120,
            "content_chars": 1_080,
        }

        with patch("search_agent.fetch_url") as provider:
            provider.invoke.side_effect = [
                json.dumps(blocked),
                json.dumps(alternate),
            ]
            first = json.loads(fetch.invoke({"url": "https://blocked.example/first"}))
            repeated_url = json.loads(
                fetch.invoke({"url": "https://blocked.example/first#again"})
            )
            repeated_host = json.loads(
                fetch.invoke({"url": "https://blocked.example/second"})
            )
            other_host = json.loads(
                fetch.invoke({"url": "https://alternate.example/page"})
            )

        self.assertEqual(first["failure_type"], "access_blocked")
        self.assertEqual(provider.invoke.call_count, 2)
        self.assertEqual(guard.call_count, 2)
        self.assertEqual(budget.snapshot()["fetch_calls"], 2)
        for suppressed in (repeated_url, repeated_host):
            self.assertEqual(suppressed["status"], "suppressed")
            self.assertEqual(suppressed["failure_type"], "access_blocked")
            self.assertTrue(suppressed["retry_with_another_source"])
            self.assertIn(
                "source",
                json.dumps(suppressed, ensure_ascii=False).casefold(),
            )
        self.assertEqual(other_host["status"], "success")

    def test_external_guard_denial_rolls_back_slice_and_skips_provider(self) -> None:
        denial = {
            "status": "budget_exceeded",
            "budget_resource": "search",
            "budget_snapshot": {"remaining_search_calls": 0},
            "attempted": {"tool_name": "web_search"},
        }
        guard = Mock(return_value=denial)
        provider = Mock()
        tools, budget = build_budgeted_tools(
            EFFORT_POLICIES["low"],
            raw_search_tool=provider,
            external_guard=guard,
        )
        search = next(tool for tool in tools if tool.name == "web_search")

        result = json.loads(search.invoke({"query": "entity research query"}))

        provider.invoke.assert_not_called()
        guard.assert_called_once_with("web_search")
        self.assertEqual(result["status"], "budget_exceeded")
        self.assertEqual(result["budget_resource"], "search")
        self.assertEqual(result["provider_outcome"], "not_called")
        self.assertEqual(budget.snapshot()["search_calls"], 0)
        self.assertEqual(
            budget.snapshot()["tool_attempts"][0]["provider_outcome"],
            "not_called",
        )


if __name__ == "__main__":
    unittest.main()
