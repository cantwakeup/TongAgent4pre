"""Tests for deterministic search relevance and backup-engine routing."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from search_agent import MIN_SEARCH_RELEVANCE_SCORE, _search_relevance_score, web_search


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

        with (
            patch("search_agent._duckduckgo_results", return_value=duckduckgo),
            patch("search_agent._bing_results", return_value=bing) as backup,
        ):
            result = json.loads(web_search.invoke({"query": query, "max_results": 5}))

        backup.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["engine_status"]["duckduckgo"], "success")
        self.assertEqual(result["engine_status"]["bing"], "success")
        self.assertEqual(result["fallback_reason"], "duckduckgo_low_relevance")
        self.assertEqual(result["results"][0]["engine"], "bing")
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

        with (
            patch("search_agent._duckduckgo_results", return_value=duckduckgo),
            patch("search_agent._bing_results") as backup,
        ):
            result = json.loads(web_search.invoke({"query": query, "max_results": 5}))

        backup.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["engines"], ["duckduckgo"])
        self.assertEqual(result["relevant_results"], 2)

    def test_both_engine_errors_return_auditable_empty_result(self) -> None:
        with (
            patch("search_agent._duckduckgo_results", side_effect=ValueError("ddg")),
            patch("search_agent._bing_results", side_effect=ValueError("bing")),
        ):
            result = json.loads(
                web_search.invoke({"query": "site:bigai.ai TongProgram-2026"})
            )

        self.assertEqual(result["fallback_reason"], "duckduckgo_error")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "all_search_engines_failed")
        self.assertEqual(
            result["engine_status"], {"duckduckgo": "error", "bing": "error"}
        )
        self.assertEqual(result["engines"], [])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["search_quality"], "low_relevance")


if __name__ == "__main__":
    unittest.main()
