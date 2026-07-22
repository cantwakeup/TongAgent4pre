"""Tests for deterministic search relevance and backup-engine routing."""

from __future__ import annotations

import json
import unittest

from retrieval_backend import RetrievalSession, SearchBroker
from retrieval_providers import ProviderSearchResponse, SearchResult
from search_agent import (
    MIN_SEARCH_RELEVANCE_SCORE,
    _search_relevance_score,
    create_retrieval_tools,
)


class _Provider:
    def __init__(
        self,
        name: str,
        results: list[dict[str, str]],
        *,
        status: str = "success",
    ) -> None:
        self.name = name
        self.results = results
        self.status = status
        self.calls: list[str] = []

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        self.calls.append(query)
        return ProviderSearchResponse(
            provider=self.name,
            query=query,
            status=self.status,
            failure_category=("provider_error" if self.status == "error" else None),
            results=[
                SearchResult(
                    title=item["title"],
                    url=item["url"],
                    snippet=item["snippet"],
                    provider=self.name,
                    provider_rank=rank,
                )
                for rank, item in enumerate(self.results[:max_results], 1)
            ],
        )


def _search_tool(providers: list[_Provider]) -> object:
    session = RetrievalSession(SearchBroker(providers))
    search, _ = create_retrieval_tools(session)
    return search


class SearchQualityTests(unittest.TestCase):
    """Verify that entity queries do not silently accept unrelated results."""

    def test_entity_relevance_rejects_generic_location_noise(self) -> None:
        query = '"北京通用人工智能研究院" "通计划"'
        irrelevant = {
            "title": "北京市人民政府门户网站",
            "url": "https://www.beijing.gov.cn/",
            "snippet": "北京市政府公开信息。",
        }
        relevant = {
            "title": "北京通用人工智能研究院BIGAI",
            "url": "https://www.bigai.ai/",
            "snippet": "北京通用人工智能研究院官方网站。",
        }

        self.assertLess(
            _search_relevance_score(query, irrelevant), MIN_SEARCH_RELEVANCE_SCORE
        )
        self.assertGreaterEqual(
            _search_relevance_score(query, relevant), MIN_SEARCH_RELEVANCE_SCORE
        )

    def test_site_operator_with_path_matches_the_result_host(self) -> None:
        score = _search_relevance_score(
            "site:bigai.ai/about/ info.bigai@bigai.ai",
            {
                "title": "关于通院",
                "url": "https://www.bigai.ai/about/",
                "snippet": "北京通用人工智能研究院官网信息。",
            },
        )

        self.assertGreaterEqual(score, MIN_SEARCH_RELEVANCE_SCORE)

    def test_site_operator_does_not_match_domain_suffix_without_label_boundary(
        self,
    ) -> None:
        score = _search_relevance_score(
            "site:bigai.ai",
            {
                "title": "lookalike",
                "url": "https://evilbigai.ai/page",
                "snippet": "unrelated",
            },
        )

        self.assertEqual(score, 0)

    def test_empty_site_domain_never_matches_an_empty_result_host(self) -> None:
        score = _search_relevance_score(
            "site:/",
            {"title": "unrelated", "url": "", "snippet": "unrelated"},
        )

        self.assertEqual(score, 0)

    def test_low_relevance_duckduckgo_results_trigger_bing(self) -> None:
        query = '"北京通用人工智能研究院" "通计划"'
        duckduckgo = [
            {
                "title": "北京旅游攻略",
                "url": "https://example.com/beijing-travel",
                "snippet": "北京景点与酒店。",
            }
        ]
        bing = [
            {
                "title": "TongProgram-2026 - 北京通用人工智能研究院BIGAI",
                "url": "https://www.bigai.ai/tongprogram-2026/",
                "snippet": "通计划联系邮箱。",
            }
        ]

        primary = _Provider("duckduckgo_html", duckduckgo)
        backup = _Provider("bing_rss", bing)
        search = _search_tool([primary, backup])

        result = json.loads(search.invoke({"query": query, "max_results": 5}))

        self.assertEqual(len(backup.calls), 1)
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [item["status"] for item in result["provider_statuses"]],
            ["success", "success"],
        )
        self.assertEqual(
            result["fallback_reason"],
            "duckduckgo_html_low_relevance",
        )
        self.assertEqual(result["results"][0]["engine"], "bing_rss")
        self.assertEqual(
            result["results"][0]["url"],
            "https://www.bigai.ai/tongprogram-2026/",
        )
        self.assertGreaterEqual(result["relevant_results"], 1)

    def test_two_relevant_primary_results_skip_backup(self) -> None:
        query = '"BIGAI" "TongProgram-2026"'
        duckduckgo = [
            {
                "title": "BIGAI official site",
                "url": "https://www.bigai.ai/",
                "snippet": "BIGAI official information.",
            },
            {
                "title": "TongProgram-2026",
                "url": "https://www.bigai.ai/tongprogram-2026/",
                "snippet": "TongProgram-2026 contact details.",
            },
        ]

        primary = _Provider("duckduckgo_html", duckduckgo)
        backup = _Provider("bing_rss", [])
        search = _search_tool([primary, backup])

        result = json.loads(search.invoke({"query": query, "max_results": 5}))

        self.assertEqual(backup.calls, [])
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["engines"], ["duckduckgo_html"])
        self.assertEqual(result["relevant_results"], 2)

    def test_both_engine_errors_return_auditable_empty_result(self) -> None:
        search = _search_tool(
            [
                _Provider("duckduckgo_html", [], status="error"),
                _Provider("bing_rss", [], status="error"),
            ]
        )

        result = json.loads(search.invoke({"query": "site:bigai.ai TongProgram-2026"}))

        self.assertEqual(
            result["fallback_reason"],
            "duckduckgo_html_provider_error",
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "all_search_providers_failed")
        self.assertEqual(
            [item["status"] for item in result["provider_statuses"]],
            ["error", "error"],
        )
        self.assertEqual(result["engines"], [])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["search_quality"], "no_candidates")


if __name__ == "__main__":
    unittest.main()
