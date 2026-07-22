"""Regression tests for benchmark-facing metric semantics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import ToolMessage

from adaptive_control import build_control_assessment, initialize_control_state
from agent_policy import EFFORT_POLICIES
from evidence_graph import source_diversity_metrics
from research_graph import (
    audit_structural_subquestion_closures,
    create_research_plan,
    invalid_covered_subquestions,
    refresh_plan_status,
    transition_subquestion,
)
from search_agent import (
    ResearchBudget,
    _build_tool_trace,
    build_budgeted_tools,
)
from telemetry import write_plan_snapshot


def _search_payload(
    urls: list[str],
    *,
    relevant_results: int,
) -> dict[str, object]:
    """Return one deterministic provider-successful search payload."""
    return {
        "status": "success",
        "provider_success": True,
        "results": [
            {
                "url": url,
                "relevance_score": 100 if index < relevant_results else 0,
            }
            for index, url in enumerate(urls)
        ],
        "relevant_results": relevant_results,
    }


def _page(url: str, content: str) -> dict[str, object]:
    """Return a deterministic successful page payload."""
    return {
        "status": "success",
        "url": url,
        "title": url,
        "content": content,
        "content_chars": len(content),
        "fetched_at": "2026-07-18T00:00:00+00:00",
        "content_length": len(content),
        "observed_content_length": len(content),
        "downloaded_bytes": len(content.encode()),
        "http_content_length": len(content.encode()),
        "truncated": False,
        "truncation_reasons": [],
    }


def _record_search(
    budget: ResearchBudget,
    *,
    query: str,
    urls: list[str],
    relevant_results: int,
) -> dict[str, object]:
    """Reserve and record one deterministic search."""
    if not budget.reserve_search():
        msg = "test search unexpectedly exceeded its budget"
        raise AssertionError(msg)
    return budget.record_tool_attempt(
        tool_name="web_search",
        target=query,
        payload=_search_payload(urls, relevant_results=relevant_results),
    )


def _evidence_unit(
    source_id: str,
    claim_id: str,
    content_hash: str,
    *,
    stance: str = "supports",
) -> dict[str, str]:
    """Return the fields source-diversity grouping consumes."""
    return {
        "evidence_id": f"E{source_id.removeprefix('S')}",
        "source_id": source_id,
        "claim_id": claim_id,
        "source_content_sha256": content_hash,
        "stance": stance,
    }


class SearchMetricSemanticsTests(unittest.TestCase):
    """Verify each search outcome has one non-overloaded counter."""

    def test_provider_success_with_empty_results_is_not_nonempty_or_relevant(
        self,
    ) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        attempt = _record_search(
            budget,
            query="empty",
            urls=[],
            relevant_results=0,
        )
        snapshot = budget.snapshot()

        self.assertEqual(snapshot["provider_successes"], 1)
        self.assertEqual(snapshot["nonempty_searches"], 0)
        self.assertEqual(snapshot["successful_searches"], 0)
        self.assertEqual(snapshot["relevant_searches"], 0)
        self.assertEqual(attempt["provider_outcome"], "success")
        self.assertEqual(attempt["outcome"], "empty_results")

    def test_nonempty_irrelevant_search_does_not_increment_relevant(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        attempt = _record_search(
            budget,
            query="noise",
            urls=["https://noise.test/page"],
            relevant_results=0,
        )
        snapshot = budget.snapshot()

        self.assertEqual(snapshot["provider_successes"], 1)
        self.assertEqual(snapshot["nonempty_searches"], 1)
        self.assertEqual(snapshot["successful_searches"], 1)
        self.assertEqual(snapshot["relevant_searches"], 0)
        self.assertFalse(attempt["relevant_search"])
        self.assertEqual(attempt["outcome"], "low_relevance")

    def test_multiple_relevant_results_increment_relevant_once(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        attempt = _record_search(
            budget,
            query="two results",
            urls=["https://one.test/page", "https://two.test/page"],
            relevant_results=2,
        )
        snapshot = budget.snapshot()

        self.assertEqual(snapshot["relevant_searches"], 1)
        self.assertEqual(attempt["relevant_results"], 2)
        self.assertEqual(len(attempt["relevant_result_urls"]), 2)

    def test_inconsistent_backend_flags_cannot_inflate_canonical_metrics(
        self,
    ) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        self.assertTrue(budget.reserve_search())
        attempt = budget.record_tool_attempt(
            tool_name="web_search",
            target="tampered",
            payload={
                "status": "success",
                "provider_success": True,
                "nonempty_search": True,
                "relevant_search": True,
                "relevant_results": 99,
                "results": [],
            },
        )
        self.assertTrue(budget.reserve_search())
        failed_attempt = budget.record_tool_attempt(
            tool_name="web_search",
            target="failed but self-reported",
            payload={
                "status": "error",
                "provider_success": True,
                "nonempty_search": True,
                "relevant_search": True,
                "relevant_results": 1,
                "results": [
                    {
                        "url": "https://noise.test/page",
                        "relevance_score": 100,
                    }
                ],
            },
        )
        snapshot = budget.snapshot()

        self.assertEqual(snapshot["provider_successes"], 1)
        self.assertEqual(snapshot["nonempty_searches"], 0)
        self.assertEqual(snapshot["relevant_searches"], 0)
        self.assertEqual(attempt["outcome"], "empty_results")
        self.assertIn("relevant_search", attempt["semantic_mismatches"])
        self.assertEqual(failed_attempt["provider_outcome"], "failure")
        self.assertFalse(failed_attempt["relevant_search"])

    def test_budget_denial_records_not_called_instead_of_provider_failure(
        self,
    ) -> None:
        tools, budget = build_budgeted_tools(EFFORT_POLICIES["low"])
        search = next(item for item in tools if item.name == "web_search")
        raw = json.dumps(
            _search_payload(["https://answer.test/page"], relevant_results=1)
        )

        with patch("search_agent.web_search") as provider:
            provider.invoke.return_value = raw
            search.invoke({"query": "one"})
            search.invoke({"query": "two"})
            denied = json.loads(search.invoke({"query": "three"}))

        self.assertEqual(denied["status"], "budget_exceeded")
        self.assertEqual(denied["provider_outcome"], "not_called")
        self.assertFalse(denied["provider_failure"])
        self.assertEqual(budget.snapshot()["provider_successes"], 2)

    def test_provider_runtime_errors_are_recorded_after_budget_reservation(
        self,
    ) -> None:
        for tool_name, argument, provider_kwarg in (
            ("web_search", {"query": "fixture"}, "raw_search_tool"),
            (
                "fetch_url",
                {"url": "https://fixture.test/page"},
                "raw_fetch_tool",
            ),
        ):
            with self.subTest(tool=tool_name):
                provider = Mock()
                provider.invoke.side_effect = RuntimeError("sensitive provider detail")
                tools, budget = build_budgeted_tools(
                    EFFORT_POLICIES["low"],
                    **{provider_kwarg: provider},
                )
                wrapped = next(item for item in tools if item.name == tool_name)

                result = json.loads(wrapped.invoke(argument))
                snapshot = budget.snapshot()

                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"], "provider_error")
                self.assertEqual(result["provider_error_type"], "RuntimeError")
                self.assertNotIn("sensitive provider detail", json.dumps(result))
                self.assertEqual(len(snapshot["tool_attempts"]), 1)
                self.assertEqual(
                    snapshot["tool_attempts"][0]["failure_class"],
                    "provider",
                )
                if tool_name == "web_search":
                    self.assertEqual(snapshot["search_calls"], 1)
                    self.assertEqual(
                        snapshot["tool_attempts"][0]["provider_outcome"],
                        "failure",
                    )
                else:
                    self.assertEqual(snapshot["fetch_calls"], 1)
                    self.assertEqual(len(snapshot["failed_sources"]), 1)
                    self.assertEqual(
                        snapshot["failed_sources"][0]["url"],
                        argument["url"],
                    )

    def test_malformed_or_non_object_provider_json_is_recorded(self) -> None:
        for tool_name, argument, provider_kwarg in (
            ("web_search", {"query": "fixture"}, "raw_search_tool"),
            (
                "fetch_url",
                {"url": "https://fixture.test/page"},
                "raw_fetch_tool",
            ),
        ):
            for raw_result, expected_type in (
                ("{not-json", "JSONDecodeError"),
                ("[]", "TypeError"),
            ):
                with self.subTest(
                    tool=tool_name,
                    raw_result=raw_result,
                ):
                    provider = Mock()
                    provider.invoke.return_value = raw_result
                    tools, budget = build_budgeted_tools(
                        EFFORT_POLICIES["low"],
                        **{provider_kwarg: provider},
                    )
                    wrapped = next(item for item in tools if item.name == tool_name)

                    result = json.loads(wrapped.invoke(argument))
                    snapshot = budget.snapshot()

                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error"], "provider_error")
                    self.assertEqual(result["provider_error_type"], expected_type)
                    self.assertEqual(len(snapshot["tool_attempts"]), 1)
                    self.assertEqual(
                        snapshot["tool_attempts"][0]["failure_class"],
                        "provider",
                    )
                    if tool_name == "fetch_url":
                        self.assertEqual(len(snapshot["failed_sources"]), 1)

    def test_provider_timeout_type_preserves_network_failure_class(self) -> None:
        for tool_name, argument, provider_kwarg in (
            ("web_search", {"query": "fixture"}, "raw_search_tool"),
            (
                "fetch_url",
                {"url": "https://fixture.test/page"},
                "raw_fetch_tool",
            ),
        ):
            with self.subTest(tool=tool_name):
                provider = Mock()
                provider.invoke.side_effect = httpx.ReadTimeout(
                    "sensitive timeout detail",
                    request=httpx.Request("GET", "https://fixture.test"),
                )
                tools, budget = build_budgeted_tools(
                    EFFORT_POLICIES["low"],
                    **{provider_kwarg: provider},
                )
                wrapped = next(item for item in tools if item.name == tool_name)

                result = json.loads(wrapped.invoke(argument))
                attempt = budget.snapshot()["tool_attempts"][0]

                self.assertEqual(result["error"], "provider_error")
                self.assertEqual(result["provider_error_type"], "ReadTimeout")
                self.assertNotIn("sensitive timeout detail", json.dumps(result))
                self.assertEqual(attempt["failure_class"], "network")

    def test_low_relevance_search_can_produce_evidence_but_not_coverage(
        self,
    ) -> None:
        url = "https://noise.test/page"
        content = "The fixture page contains a verifiable statement for the claim."
        claim = "The fixture page contains a verifiable statement."
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        _record_search(
            budget,
            query="noise",
            urls=[url],
            relevant_results=0,
        )
        source_id = budget.record_fetch(_page(url, content))
        evidence = budget.record_evidence(
            source_id=str(source_id),
            claim=claim,
            quote=content,
            stance="supports",
        )
        plan = create_research_plan("fixture question", ["answer the fixture"])
        plan = transition_subquestion(plan, "SQ1", "researching")
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=["S1"],
            claim_ids=[str(evidence["claim"]["claim_id"])],
        )
        snapshot = budget.snapshot()

        self.assertEqual(snapshot["relevant_searches"], 0)
        self.assertEqual(snapshot["evidence_producing_searches"], 1)
        self.assertTrue(snapshot["tool_attempts"][0]["evidence_producing_search"])
        self.assertEqual(plan["structural_subquestion_coverage"], 0.0)
        self.assertEqual(plan["status"], "partial")
        self.assertFalse(plan["subquestions"][0]["structural_closure_validated"])
        self.assertTrue(
            any(
                "relevant search" in reason
                for reason in invalid_covered_subquestions(plan, snapshot)["SQ1"]
            )
        )

    def test_fabricated_claim_and_source_ids_never_create_structural_coverage(
        self,
    ) -> None:
        plan = create_research_plan("fixture question", ["answer the fixture"])
        plan = transition_subquestion(plan, "SQ1", "researching")
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=["S1"],
            claim_ids=["C1"],
        )

        self.assertEqual(plan["structural_subquestion_coverage"], 0.0)
        self.assertEqual(plan["status"], "partial")
        self.assertFalse(plan["subquestions"][0]["structural_closure_validated"])

    def test_persisted_closure_marker_is_revalidated_against_the_ledger(
        self,
    ) -> None:
        plan = create_research_plan("fixture question", ["answer the fixture"])
        plan = transition_subquestion(plan, "SQ1", "researching")
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=["S1"],
            claim_ids=["C1"],
        )
        plan["subquestions"][0]["structural_closure_validated"] = True
        tampered = refresh_plan_status(plan)
        self.assertEqual(tampered["structural_subquestion_coverage"], 1.0)

        audited, invalid = audit_structural_subquestion_closures(tampered, {})

        self.assertIn("SQ1", invalid)
        self.assertEqual(audited["structural_subquestion_coverage"], 0.0)
        self.assertEqual(audited["status"], "partial")
        self.assertFalse(audited["subquestions"][0]["structural_closure_validated"])

    def test_ledger_audited_supported_closure_creates_structural_coverage(
        self,
    ) -> None:
        urls = ["https://one.test/page", "https://two.test/page"]
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        _record_search(
            budget,
            query="supported fixture",
            urls=urls,
            relevant_results=2,
        )
        claim_id = ""
        source_ids: list[str] = []
        for index, url in enumerate(urls, start=1):
            content = (
                f"Fixture publisher {index} independently states the supported fact."
            )
            source_id = str(budget.record_fetch(_page(url, content)))
            result = budget.record_evidence(
                source_id=source_id,
                claim="Both fixture publishers state the supported fact.",
                quote=content,
                stance="supports",
                claim_id=claim_id,
            )
            claim_id = str(result["claim"]["claim_id"])
            source_ids.append(source_id)
        plan = create_research_plan("fixture question", ["answer the fixture"])
        plan = transition_subquestion(plan, "SQ1", "researching")
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=source_ids,
            claim_ids=[claim_id],
        )

        self.assertEqual(plan["structural_subquestion_coverage"], 0.0)
        audited, invalid = audit_structural_subquestion_closures(
            plan, budget.snapshot()
        )

        self.assertEqual(invalid, {})
        self.assertEqual(audited["structural_subquestion_coverage"], 1.0)
        self.assertEqual(audited["status"], "completed")
        self.assertTrue(audited["subquestions"][0]["structural_closure_validated"])

    def test_duplicate_evidence_and_checkpoint_restore_do_not_recount_search(
        self,
    ) -> None:
        url = "https://repeat.test/page"
        content = "The repeated fixture contains one stable exact evidence excerpt."
        claim = "The repeated fixture contains one stable excerpt."
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")

        for query in ("first", "retry"):
            _record_search(
                budget,
                query=query,
                urls=[url],
                relevant_results=0,
            )
            budget.record_fetch(_page(url, content))
            budget.record_evidence(
                source_id="S1",
                claim=claim,
                quote=content,
                stance="supports",
                claim_id="C1" if query == "retry" else "",
            )

        before = budget.snapshot()
        self.assertEqual(len(before["evidence_units"]), 1)
        self.assertEqual(before["evidence_producing_searches"], 1)

        restored = ResearchBudget(EFFORT_POLICIES["low"])
        restored.restore(before, strict_policy=True)
        restored.record_fetch(_page(url, content))
        restored.record_evidence(
            source_id="S1",
            claim=claim,
            quote=content,
            stance="supports",
            claim_id="C1",
        )
        after = restored.snapshot()

        self.assertEqual(after["search_calls"], before["search_calls"])
        self.assertEqual(after["relevant_searches"], 0)
        self.assertEqual(after["evidence_producing_searches"], 1)
        self.assertEqual(len(after["tool_attempts"]), 2)
        self.assertEqual(len(after["evidence_units"]), 1)

    def test_relevant_searches_cannot_be_borrowed_across_subquestions(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["medium"])
        budget.configure_subquestions(["SQ1", "SQ2"])
        claims: list[str] = []
        sources: list[str] = []
        for subquestion_id, host in (("SQ1", "one.test"), ("SQ2", "two.test")):
            budget.activate_subquestion(subquestion_id)
            url = f"https://{host}/page"
            content = (
                f"The {subquestion_id} fixture provides a distinct supported statement."
            )
            if subquestion_id == "SQ1":
                _record_search(
                    budget,
                    query="only SQ1 searched",
                    urls=[url],
                    relevant_results=1,
                )
            source_id = str(budget.record_fetch(_page(url, content)))
            result = budget.record_evidence(
                source_id=source_id,
                claim=f"The {subquestion_id} fixture provides a supported statement.",
                quote=content,
                stance="supports",
            )
            sources.append(source_id)
            claims.append(str(result["claim"]["claim_id"]))

        plan = create_research_plan("two scopes", ["first scope", "second scope"])
        plan = transition_subquestion(plan, "SQ1", "researching")
        plan = transition_subquestion(
            plan,
            "SQ1",
            "covered",
            evidence_source_ids=[sources[0]],
            claim_ids=[claims[0]],
        )
        plan = transition_subquestion(plan, "SQ2", "researching")
        plan = transition_subquestion(
            plan,
            "SQ2",
            "covered",
            evidence_source_ids=[sources[1]],
            claim_ids=[claims[1]],
        )
        invalid = invalid_covered_subquestions(plan, budget.snapshot())

        self.assertTrue(
            any("missing a relevant search" in reason for reason in invalid["SQ2"])
        )

    def test_legacy_checkpoint_keeps_missing_relevance_unavailable(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        _record_search(
            budget,
            query="legacy nonempty",
            urls=["https://legacy.test/page"],
            relevant_results=0,
        )
        legacy = budget.snapshot()
        legacy.pop("relevant_searches")
        legacy["subquestion_usage"]["SQ1"].pop("relevant_searches")

        restored = ResearchBudget(EFFORT_POLICIES["low"])
        restored.restore(legacy, strict_policy=True)
        snapshot = restored.snapshot()

        self.assertEqual(snapshot["nonempty_searches"], 1)
        self.assertEqual(snapshot["successful_searches"], 1)
        self.assertIsNone(snapshot["relevant_searches"])
        self.assertIsNone(snapshot["subquestion_usage"]["SQ1"]["relevant_searches"])

    def test_strict_restore_rejects_counters_that_disagree_with_attempts(
        self,
    ) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        _record_search(
            budget,
            query="low relevance",
            urls=["https://noise.test/page"],
            relevant_results=0,
        )
        tampered = budget.snapshot()
        tampered["relevant_searches"] = 1
        tampered["subquestion_usage"]["SQ1"]["relevant_searches"] = 1

        restored = ResearchBudget(EFFORT_POLICIES["low"])
        with self.assertRaisesRegex(ValueError, "attempt ledger"):
            restored.restore(tampered, strict_policy=True)

    def test_strict_restore_rejects_fetch_counters_below_admitted_attempts(
        self,
    ) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        self.assertTrue(budget.reserve_fetch())
        budget.record_tool_attempt(
            tool_name="fetch_url",
            target="https://unavailable.test/page",
            payload={
                "status": "error",
                "url": "https://unavailable.test/page",
                "error": "HTTP 503",
            },
        )
        for tamper_global in (True, False):
            with self.subTest(tamper_global=tamper_global):
                tampered = budget.snapshot()
                if tamper_global:
                    tampered["fetch_calls"] = 0
                tampered["subquestion_usage"]["SQ1"]["fetch_calls"] = 0

                restored = ResearchBudget(EFFORT_POLICIES["low"])
                with self.assertRaisesRegex(ValueError, "fetch attempt ledger"):
                    restored.restore(tampered, strict_policy=True)

                self.assertEqual(restored.snapshot()["fetch_calls"], 0)

    def test_strict_restore_does_not_charge_budget_denied_fetch_attempt(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        for index in range(budget.policy.max_fetches):
            self.assertTrue(budget.reserve_fetch())
            budget.record_tool_attempt(
                tool_name="fetch_url",
                target=f"https://unavailable.test/{index}",
                payload={
                    "status": "error",
                    "url": f"https://unavailable.test/{index}",
                    "error": "HTTP 503",
                },
            )
        self.assertFalse(budget.reserve_fetch())
        denied_url = "https://unavailable.test/denied"
        budget.record_tool_attempt(
            tool_name="fetch_url",
            target=denied_url,
            payload={
                "status": "budget_exceeded",
                "url": denied_url,
                **budget.budget_denial("fetch"),
            },
        )
        snapshot = budget.snapshot()

        restored = ResearchBudget(EFFORT_POLICIES["low"])
        restored.restore(snapshot, strict_policy=True)

        self.assertEqual(restored.snapshot(), snapshot)

    def test_incomplete_legacy_ledger_makes_search_metrics_unavailable(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        _record_search(
            budget,
            query="legacy",
            urls=["https://legacy.test/page"],
            relevant_results=1,
        )
        legacy = budget.snapshot()
        legacy.pop("search_metric_semantics_version")
        legacy["tool_attempts"] = []
        legacy["next_tool_attempt_sequence"] = 1

        restored = ResearchBudget(EFFORT_POLICIES["low"])
        restored.restore(legacy, strict_policy=True)
        snapshot = restored.snapshot()

        self.assertIsNone(snapshot["provider_successes"])
        self.assertIsNone(snapshot["nonempty_searches"])
        self.assertIsNone(snapshot["successful_searches"])
        self.assertIsNone(snapshot["relevant_searches"])
        self.assertIsNone(snapshot["evidence_producing_searches"])


class SourceDiversitySemanticsTests(unittest.TestCase):
    """Verify revisions, hosts, and corroborating groups stay separate."""

    def _metrics(
        self,
        sources: list[dict[str, str]],
        evidence: list[dict[str, str]],
    ) -> dict[str, object]:
        return source_diversity_metrics(
            source_ids=[source["source_id"] for source in sources],
            sources=sources,
            evidence_units=evidence,
        )

    def test_same_host_different_pages_form_one_corroborating_group(self) -> None:
        sources = [
            {"source_id": "S1", "url": "https://publisher.test/a"},
            {"source_id": "S2", "url": "https://publisher.test/b"},
        ]
        metrics = self._metrics(
            sources,
            [
                _evidence_unit("S1", "C1", "a" * 64),
                _evidence_unit("S2", "C1", "b" * 64),
            ],
        )

        self.assertEqual(metrics["distinct_content_revision_count"], 2)
        self.assertEqual(metrics["distinct_source_host_count"], 1)
        self.assertEqual(metrics["corroborating_source_group_count"], 1)

    def test_identical_content_on_different_hosts_forms_one_group(self) -> None:
        sources = [
            {"source_id": "S1", "url": "https://one.test/a"},
            {"source_id": "S2", "url": "https://two.test/b"},
        ]
        shared_hash = "a" * 64
        metrics = self._metrics(
            sources,
            [
                _evidence_unit("S1", "C1", shared_hash),
                _evidence_unit("S2", "C1", shared_hash),
            ],
        )

        self.assertEqual(metrics["distinct_content_revision_count"], 1)
        self.assertEqual(metrics["distinct_source_host_count"], 2)
        self.assertEqual(metrics["corroborating_source_group_count"], 1)

    def test_contradicting_source_does_not_satisfy_two_source_support(self) -> None:
        sources = [
            {"source_id": "S1", "url": "https://support.test/a"},
            {"source_id": "S2", "url": "https://contradict.test/b"},
        ]
        metrics = self._metrics(
            sources,
            [
                _evidence_unit("S1", "C1", "a" * 64),
                _evidence_unit(
                    "S2",
                    "C1",
                    "b" * 64,
                    stance="contradicts",
                ),
            ],
        )

        self.assertEqual(metrics["corroborating_source_group_count"], 1)
        self.assertEqual(metrics["corroborating_source_ids"], ["S1"])

    def test_invalid_hostname_is_reported_unavailable_and_not_grouped(self) -> None:
        sources = [
            {"source_id": "S1", "url": "https://bad host.test/page"},
            {"source_id": "S2", "url": "https://[broken"},
        ]
        metrics = self._metrics(
            sources,
            [
                _evidence_unit("S1", "C1", "a" * 64),
                _evidence_unit("S2", "C1", "b" * 64),
            ],
        )

        self.assertEqual(metrics["distinct_source_host_count"], 0)
        self.assertEqual(metrics["corroborating_source_group_count"], 0)
        self.assertEqual(metrics["unavailable_source_ids"], ["S1", "S2"])


class ArtifactMetricSemanticsTests(unittest.TestCase):
    """Verify structural coverage and trace/control output are explicit."""

    def test_structural_coverage_alias_and_plan_semantics_stay_in_sync(self) -> None:
        plan = create_research_plan("fixture", ["one subquestion"])
        self.assertEqual(plan["structural_subquestion_coverage"], 0.0)
        self.assertEqual(plan["coverage"], 0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            write_plan_snapshot(path, thread_id="fixture", plan=plan, budget={})
            payload = json.loads(path.read_text())

        self.assertEqual(
            payload["plan"]["structural_subquestion_coverage"],
            payload["plan"]["coverage"],
        )
        self.assertIn(
            "deprecated alias",
            payload["metric_semantics"]["coverage"],
        )

    def test_trace_and_control_expose_distinct_search_states(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        attempt = _record_search(
            budget,
            query="noise",
            urls=["https://noise.test/page"],
            relevant_results=0,
        )
        payload = {
            **_search_payload(["https://noise.test/page"], relevant_results=0),
            **{
                key: attempt[key]
                for key in (
                    "attempt_id",
                    "outcome",
                    "failure_class",
                    "retryable",
                    "provider_success",
                    "provider_outcome",
                    "nonempty_search",
                    "relevant_search",
                    "evidence_producing_search",
                    "provider_failure",
                )
            },
        }
        trace = _build_tool_trace(
            [
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id="call-1",
                    name="web_search",
                )
            ]
        )
        plan = create_research_plan("fixture", ["one subquestion"])
        plan = transition_subquestion(plan, "SQ1", "researching")
        control = initialize_control_state(
            strategy="adaptive",
            config_fingerprint="fixture",
            hard_effort="low",
            pinned_model="fixture-model",
            pinned_topology="single",
            max_escalations=1,
        )
        assessment = build_control_assessment(
            plan=plan,
            budget=budget.snapshot(),
            control=control,
            subquestion_id="SQ1",
            cycle=1,
            new_source_ids=[],
            new_claim_ids=[],
            new_evidence_ids=[],
            new_conflict_ids=[],
            new_tool_attempt_ids=[str(attempt["attempt_id"])],
        )

        semantics = trace[0]["search_semantics"]
        self.assertEqual(semantics["provider_outcome"], "success")
        self.assertEqual(semantics["outcome"], "low_relevance")
        self.assertEqual(semantics["failure_class"], "content")
        self.assertTrue(semantics["retryable"])
        self.assertTrue(semantics["nonempty_search"])
        self.assertFalse(semantics["relevant_search"])
        self.assertFalse(semantics["evidence_producing_search"])
        self.assertEqual(assessment["provider_successes"], 1)
        self.assertEqual(assessment["nonempty_searches"], 1)
        self.assertEqual(assessment["relevant_searches"], 0)
        self.assertEqual(assessment["evidence_producing_searches"], 0)
        self.assertEqual(
            assessment["current_search_attempts"][0]["outcome"],
            "low_relevance",
        )


if __name__ == "__main__":
    unittest.main()
