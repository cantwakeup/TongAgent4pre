"""Offline tests for Stage 03C claim-to-excerpt provenance."""

from __future__ import annotations

import unittest
from copy import deepcopy

from agent_policy import EFFORT_POLICIES
from evidence_graph import (
    allowed_report_caveat_lines,
    independent_evidence_source_ids,
    report_claim_mapping_errors,
    text_sha256,
    validate_evidence_graph,
)
from search_agent import ResearchBudget


def _page(url: str, title: str, content: str, *, quality: str = "full") -> dict:
    return {
        "status": "success",
        "url": url,
        "title": title,
        "content": content,
        "content_chars": len(content),
        "evidence_quality": quality,
        "quality_reason": "test fixture" if quality == "limited" else "",
    }


def _active_budget() -> ResearchBudget:
    budget = ResearchBudget(EFFORT_POLICIES["medium"])
    budget.configure_subquestions(["SQ1", "SQ2"])
    budget.activate_subquestion("SQ1")
    return budget


class EvidenceGraphStoreTests(unittest.TestCase):
    """Verify immutable excerpts, derived statuses, and checkpoint behavior."""

    def test_exact_excerpt_creates_closed_graph_and_is_idempotent(self) -> None:
        budget = _active_budget()
        content = "BIGAI is the Beijing Institute for General Artificial Intelligence."
        source_id = budget.record_fetch(
            _page("https://official.example/about", "About", content)
        )

        first = budget.record_evidence(
            source_id=str(source_id),
            claim="BIGAI is the Beijing Institute for General Artificial Intelligence.",
            quote=content,
            stance="supports",
        )
        duplicate = budget.record_evidence(
            source_id=str(source_id),
            claim="BIGAI is the Beijing Institute for General Artificial Intelligence.",
            quote=content,
            stance="supports",
            claim_id="C1",
        )
        snapshot = budget.snapshot()

        self.assertEqual(first["claim"]["claim_id"], "C1")
        self.assertEqual(first["evidence"]["evidence_id"], "E1")
        self.assertEqual(duplicate["evidence"]["evidence_id"], "E1")
        self.assertEqual(len(snapshot["evidence_units"]), 1)
        self.assertEqual(snapshot["claims"][0]["status"], "supported")
        self.assertEqual(
            snapshot["evidence_units"][0]["quote_sha256"], text_sha256(content)
        )
        self.assertEqual(validate_evidence_graph(snapshot), [])

    def test_fabricated_excerpt_and_unknown_source_leave_graph_unchanged(self) -> None:
        budget = _active_budget()
        budget.record_fetch(
            _page(
                "https://official.example/about",
                "About",
                "The official page contains one verifiable institutional statement.",
            )
        )

        with self.assertRaisesRegex(ValueError, "not an exact excerpt"):
            budget.record_evidence(
                source_id="S1",
                claim="An invented proposition",
                quote="This sentence was never present on the fetched official page.",
                stance="supports",
            )
        with self.assertRaisesRegex(ValueError, "Unknown canonical source"):
            budget.record_evidence(
                source_id="S999",
                claim="Another invented proposition",
                quote="A sufficiently long but completely fabricated source excerpt.",
                stance="supports",
            )

        self.assertEqual(budget.snapshot()["claims"], [])
        self.assertEqual(budget.snapshot()["evidence_units"], [])

    def test_short_topic_label_is_not_accepted_as_a_canonical_claim(self) -> None:
        budget = _active_budget()
        content = "The official page publishes the institution's complete formal name."
        budget.record_fetch(_page("https://official.example/about", "About", content))

        with self.assertRaisesRegex(ValueError, "12-500"):
            budget.record_evidence(
                source_id="S1",
                claim="name",
                quote=content,
                stance="supports",
            )

        self.assertEqual(budget.snapshot()["claims"], [])

    def test_support_and_contradiction_create_one_conflict(self) -> None:
        budget = _active_budget()
        support = "The official notice states that the program opened in 2026."
        contradiction = (
            "The university notice states that the program did not open in 2026."
        )
        budget.record_fetch(
            _page("https://official.example/program", "Program", support)
        )
        budget.record_fetch(
            _page("https://university.example/notice", "Notice", contradiction)
        )
        claim = "The program opened in 2026."
        budget.record_evidence(
            source_id="S1", claim=claim, quote=support, stance="supports"
        )
        first_conflict = budget.record_evidence(
            source_id="S2",
            claim=claim,
            quote=contradiction,
            stance="contradicts",
            claim_id="C1",
        )
        repeated = budget.record_evidence(
            source_id="S2",
            claim=claim,
            quote=contradiction,
            stance="contradicts",
            claim_id="C1",
        )
        snapshot = budget.snapshot()

        self.assertEqual(snapshot["claims"][0]["status"], "contested")
        self.assertEqual(first_conflict["conflict"]["conflict_id"], "X1")
        self.assertEqual(repeated["conflict"]["conflict_id"], "X1")
        self.assertEqual(len(snapshot["conflicts"]), 1)
        self.assertEqual(validate_evidence_graph(snapshot), [])

    def test_limited_quality_is_inherited_without_upgrade(self) -> None:
        budget = _active_budget()
        content = "TongProgram-2026 lists the contact email tongprogram@bigai.ai."
        budget.record_fetch(
            _page(
                "https://official.example/tongprogram-2026/",
                "TongProgram-2026",
                content,
                quality="limited",
            )
        )
        result = budget.record_evidence(
            source_id="S1",
            claim="TongProgram-2026 lists the contact email tongprogram@bigai.ai.",
            quote=content,
            stance="supports",
        )

        self.assertEqual(result["evidence"]["evidence_quality"], "limited")
        self.assertEqual(budget.snapshot()["claims"][0]["status"], "supported")

    def test_old_source_can_be_refetched_without_changing_its_source_id(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.restore(
            {
                "successful_sources": [
                    {
                        "source_id": "S1",
                        "url": "https://official.example/about",
                        "title": "About",
                        "content_chars": 800,
                    }
                ]
            }
        )
        budget.configure_subquestions(["SQ1"])
        budget.activate_subquestion("SQ1")
        content = "The refetched official page now exposes an exact reusable passage."

        source_id = budget.record_fetch(
            _page("https://official.example/about", "About", content)
        )
        evidence = budget.record_evidence(
            source_id="S1",
            claim="The official page exposes an exact reusable passage.",
            quote=content,
            stance="supports",
        )

        self.assertEqual(source_id, "S1")
        self.assertEqual(evidence["evidence"]["source_id"], "S1")
        self.assertEqual(validate_evidence_graph(budget.snapshot()), [])

    def test_content_drift_preserves_original_and_edge_revision_hashes(self) -> None:
        budget = _active_budget()
        old = "The first fetched revision says the program starts in spring 2026."
        new = "The updated fetched revision says the program starts in autumn 2026."
        budget.record_fetch(_page("https://official.example/program", "Program", old))
        first = budget.record_evidence(
            source_id="S1",
            claim="The program starts in spring 2026.",
            quote=old,
            stance="supports",
        )
        budget.record_fetch(_page("https://official.example/program", "Program", new))
        second = budget.record_evidence(
            source_id="S1",
            claim="The program starts in autumn 2026.",
            quote=new,
            stance="supports",
        )
        source = budget.snapshot()["successful_sources"][0]

        self.assertTrue(source["content_changed"])
        self.assertEqual(source["content_sha256"], text_sha256(old))
        self.assertEqual(source["latest_content_sha256"], text_sha256(new))
        self.assertEqual(first["evidence"]["source_content_sha256"], text_sha256(old))
        self.assertEqual(second["evidence"]["source_content_sha256"], text_sha256(new))
        self.assertEqual(validate_evidence_graph(budget.snapshot()), [])

    def test_three_content_revisions_keep_every_evidence_edge_valid(self) -> None:
        budget = _active_budget()
        revisions = [
            "Revision one says the program starts in spring of 2026.",
            "Revision two says the program starts in summer of 2026.",
            "Revision three says the program starts in autumn of 2026.",
        ]

        for index, content in enumerate(revisions, start=1):
            budget.record_fetch(
                _page("https://official.example/program", "Program", content)
            )
            budget.record_evidence(
                source_id="S1",
                claim=f"Program timing statement number {index} is published.",
                quote=content,
                stance="supports",
            )

        snapshot = budget.snapshot()
        self.assertEqual(
            [
                item["content_sha256"]
                for item in snapshot["successful_sources"][0]["content_revisions"]
            ],
            [text_sha256(content) for content in revisions],
        )
        self.assertEqual(validate_evidence_graph(snapshot), [])

    def test_independent_sources_use_evidence_revisions_not_latest_alias(self) -> None:
        budget = _active_budget()
        shared = "Both source URLs initially expose this exact canonical passage."
        changed = "The second source later exposes a genuinely different passage."
        budget.record_fetch(_page("https://one.example/page", "One", shared))
        budget.record_fetch(_page("https://two.example/page", "Two", shared))
        first = budget.record_evidence(
            source_id="S1",
            claim="Both sources initially publish the same canonical statement.",
            quote=shared,
            stance="supports",
        )
        budget.record_evidence(
            source_id="S2",
            claim="Both sources initially publish the same canonical statement.",
            quote=shared,
            stance="supports",
            claim_id=first["claim"]["claim_id"],
        )
        budget.record_fetch(_page("https://two.example/page", "Two changed", changed))
        snapshot = budget.snapshot()

        self.assertIsNone(snapshot["successful_sources"][1]["duplicate_of_source_id"])
        self.assertEqual(
            independent_evidence_source_ids(
                source_ids={"S1", "S2"},
                sources=snapshot["successful_sources"],
                evidence_units=snapshot["evidence_units"],
                claim_ids={"C1"},
            ),
            ["S1"],
        )

    def test_refetch_uses_revision_quality_without_rewriting_old_evidence(self) -> None:
        budget = _active_budget()
        full = "The original full page provides enough canonical material for evidence."
        limited = "The shorter revision only publishes one narrow contact statement."
        budget.record_fetch(
            _page("https://official.example/program", "Full title", full)
        )
        first = budget.record_evidence(
            source_id="S1",
            claim="The original page provides canonical material.",
            quote=full,
            stance="supports",
        )
        budget.record_fetch(
            _page(
                "https://official.example/program",
                "Limited title",
                limited,
                quality="limited",
            )
        )
        second = budget.record_evidence(
            source_id="S1",
            claim="The newer page publishes one narrow contact statement.",
            quote=limited,
            stance="supports",
        )
        source = budget.snapshot()["successful_sources"][0]

        self.assertEqual(first["evidence"]["evidence_quality"], "full")
        self.assertEqual(second["evidence"]["evidence_quality"], "limited")
        self.assertEqual(source["evidence_quality"], "full")
        self.assertEqual(source["latest_evidence_quality"], "limited")
        self.assertEqual(source["latest_title"], "Limited title")
        self.assertEqual(validate_evidence_graph(budget.snapshot()), [])

    def test_same_content_hash_reuses_canonical_revision_metadata(self) -> None:
        budget = _active_budget()
        content = "The unchanged page body contains one canonical evidence statement."
        budget.record_fetch(
            _page("https://official.example/program", "Original title", content)
        )
        budget.record_fetch(
            _page(
                "https://official.example/program",
                "Changed title",
                content,
                quality="limited",
            )
        )

        result = budget.record_evidence(
            source_id="S1",
            claim="The page body contains one canonical evidence statement.",
            quote=content,
            stance="supports",
        )
        snapshot = budget.snapshot()

        self.assertEqual(len(snapshot["successful_sources"][0]["content_revisions"]), 1)
        self.assertEqual(result["evidence"]["title"], "Original title")
        self.assertEqual(result["evidence"]["evidence_quality"], "full")
        self.assertEqual(validate_evidence_graph(snapshot), [])

    def test_one_excerpt_cannot_receive_opposing_stances(self) -> None:
        budget = _active_budget()
        content = "The official notice states that the program opened in 2026."
        budget.record_fetch(
            _page("https://official.example/program", "Program", content)
        )
        budget.record_evidence(
            source_id="S1",
            claim="The program opened in 2026.",
            quote=content,
            stance="supports",
        )
        before = budget.snapshot()

        with self.assertRaisesRegex(ValueError, "cannot both support and contradict"):
            budget.record_evidence(
                source_id="S1",
                claim="The program opened in 2026.",
                quote=content,
                stance="contradicts",
                claim_id="C1",
            )

        self.assertEqual(budget.snapshot(), before)

    def test_new_plan_resets_claim_graph_but_preserves_source_catalog(self) -> None:
        budget = _active_budget()
        content = "The official page contains a claim that belongs only to plan one."
        budget.record_fetch(_page("https://official.example/one", "One", content))
        budget.record_evidence(
            source_id="S1",
            claim="This claim belongs only to plan one.",
            quote=content,
            stance="supports",
        )

        budget.start_new_plan()
        snapshot = budget.snapshot()

        self.assertEqual(len(snapshot["successful_sources"]), 1)
        self.assertEqual(snapshot["claims"], [])
        self.assertEqual(snapshot["evidence_units"], [])
        self.assertEqual(snapshot["conflicts"], [])

    def test_checkpoint_restore_preserves_graph_ids_and_requires_refetch(self) -> None:
        original = _active_budget()
        first_content = (
            "The first canonical excerpt is persisted in the evidence graph."
        )
        original.record_fetch(
            _page("https://official.example/one", "One", first_content)
        )
        original.record_evidence(
            source_id="S1",
            claim="The first canonical excerpt is persisted.",
            quote=first_content,
            stance="supports",
        )

        restored = ResearchBudget(EFFORT_POLICIES["medium"])
        restored.restore(original.snapshot())
        with self.assertRaisesRegex(ValueError, "refetch"):
            restored.record_evidence(
                source_id="S1",
                claim="A second claim cannot use a missing process-local page body.",
                quote="A second sufficiently long excerpt is unavailable before refetch.",
                stance="supports",
            )
        second_content = (
            "The refetched page supplies a second canonical excerpt after recovery."
        )
        restored.record_fetch(
            _page("https://official.example/one", "One", second_content)
        )
        second = restored.record_evidence(
            source_id="S1",
            claim="The refetched page supplies a second canonical excerpt.",
            quote=second_content,
            stance="supports",
        )

        snapshot = restored.snapshot()
        self.assertEqual(snapshot["claims"][0]["claim_id"], "C1")
        self.assertEqual(second["claim"]["claim_id"], "C2")
        self.assertEqual(second["evidence"]["evidence_id"], "E2")
        self.assertEqual(validate_evidence_graph(snapshot), [])

    def test_restored_graph_id_gaps_allocate_after_the_highest_id(self) -> None:
        original = _active_budget()
        quotes = [
            "Canonical excerpt number one remains available after recovery.",
            "Canonical excerpt number two may be removed during repair.",
            "Canonical excerpt number three remains available after recovery.",
            "Canonical excerpt number four is added after checkpoint recovery.",
        ]
        content = " ".join(quotes)
        original.record_fetch(_page("https://official.example/gaps", "Gaps", content))
        for index, quote in enumerate(quotes[:3], start=1):
            original.record_evidence(
                source_id="S1",
                claim=f"Canonical claim number {index} is registered.",
                quote=quote,
                stance="supports",
            )
        snapshot = original.snapshot()
        snapshot["claims"] = [
            item for item in snapshot["claims"] if item["claim_id"] != "C2"
        ]
        snapshot["evidence_units"] = [
            item for item in snapshot["evidence_units"] if item["evidence_id"] != "E2"
        ]

        restored = ResearchBudget(EFFORT_POLICIES["medium"])
        restored.restore(snapshot)
        restored.configure_subquestions(["SQ1"])
        restored.activate_subquestion("SQ1")
        restored.record_fetch(_page("https://official.example/gaps", "Gaps", content))
        added = restored.record_evidence(
            source_id="S1",
            claim="Canonical claim number four is registered.",
            quote=quotes[3],
            stance="supports",
        )

        self.assertEqual(added["claim"]["claim_id"], "C4")
        self.assertEqual(added["evidence"]["evidence_id"], "E4")
        self.assertEqual(validate_evidence_graph(restored.snapshot()), [])

    def test_integrity_validator_detects_tampered_quote_hash(self) -> None:
        budget = _active_budget()
        content = "The official page contains a stable integrity-check passage."
        budget.record_fetch(_page("https://official.example/one", "One", content))
        budget.record_evidence(
            source_id="S1",
            claim="The page contains a stable integrity-check passage.",
            quote=content,
            stance="supports",
        )
        tampered = deepcopy(budget.snapshot())
        tampered["evidence_units"][0]["quote_sha256"] = "0" * 64

        self.assertIn(
            "invalid quote hash", "; ".join(validate_evidence_graph(tampered))
        )

    def test_integrity_validator_rejects_unknown_stance(self) -> None:
        budget = _active_budget()
        content = "The official page contains one canonical support statement."
        budget.record_fetch(_page("https://official.example/one", "One", content))
        budget.record_evidence(
            source_id="S1",
            claim="The page contains one canonical support statement.",
            quote=content,
            stance="supports",
        )
        tampered = deepcopy(budget.snapshot())
        tampered["evidence_units"][0]["stance"] = "unknown"

        self.assertIn("invalid stance", "; ".join(validate_evidence_graph(tampered)))


class EvidenceReportMappingTests(unittest.TestCase):
    """Verify constrained report lines cannot relabel claims or sources."""

    def setUp(self) -> None:
        self.claims = [
            {
                "claim_id": "C1",
                "text": "BIGAI lists Zhu Songchun as its president.",
                "status": "supported",
            },
            {
                "claim_id": "C2",
                "text": "The program opened in 2026.",
                "status": "contested",
            },
        ]
        self.evidence = [
            {"claim_id": "C1", "source_id": "S1", "stance": "supports"},
            {"claim_id": "C2", "source_id": "S2", "stance": "supports"},
            {"claim_id": "C2", "source_id": "S3", "stance": "contradicts"},
        ]

    def test_supported_and_contested_claims_map_to_their_canonical_sources(
        self,
    ) -> None:
        report = """## Short Answer
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Key Findings
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Conflicts and Caveats
- The program opened in 2026. [C2][S2][S3]
## Sources
- [S1] One — https://example.com/one
- [S2] Two — https://example.com/two
- [S3] Three — https://example.com/three
"""

        errors = report_claim_mapping_errors(
            report,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )

        self.assertTrue(all(not value for value in errors.values()))

    def test_h1_heading_is_not_an_unvalidated_citation_channel(self) -> None:
        report = """# Research report [S2]
## Short Answer
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Key Findings
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Conflicts and Caveats
- The program opened in 2026. [C2][S2][S3]
## Sources
- [S1] One — https://example.com/one
- [S2] Two — https://example.com/two
- [S3] Three — https://example.com/three
"""

        errors = report_claim_mapping_errors(
            report,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )

        self.assertTrue(errors["invalid_section_structure"])

    def test_wrong_text_source_and_conflict_placement_are_rejected(self) -> None:
        report = """# Report
## Short Answer
- A locally rewritten claim. [C1][S2]
## Key Findings
- The program opened in 2026. [C2][S2]
- An uncited factual line.
## Conflicts and Caveats
- Process limitation only.
## Sources
- [S1] One — https://example.com/one
- [S2] Two — https://example.com/two
"""

        errors = report_claim_mapping_errors(
            report,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )

        self.assertTrue(errors["mismatched_claim_text_lines"])
        self.assertTrue(errors["mismatched_claim_source_pairs"])
        self.assertTrue(errors["misplaced_contested_claims"])
        self.assertTrue(errors["incomplete_conflict_lines"])
        self.assertTrue(errors["unmapped_finding_lines"])

    def test_report_rejects_crossed_pairs_and_locally_negated_claims(self) -> None:
        crossed = """## Short Answer
- BIGAI lists Zhu Songchun as its president. The program opened in 2026. [C1][S2][C2][S1]
## Key Findings
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Conflicts and Caveats
- The program opened in 2026. [C2][S2][S3]
## Sources
- [S1] One — https://example.com/one
- [S2] Two — https://example.com/two
- [S3] Three — https://example.com/three
"""
        negated = """## Short Answer
- It is false that BIGAI lists Zhu Songchun as its president. [C1][S1]
## Key Findings
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Conflicts and Caveats
- The program opened in 2026. [C2][S2][S3]
## Sources
- [S1] One — https://example.com/one
- [S2] Two — https://example.com/two
- [S3] Three — https://example.com/three
"""

        crossed_errors = report_claim_mapping_errors(
            crossed,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )
        negated_errors = report_claim_mapping_errors(
            negated,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )

        self.assertTrue(crossed_errors["multiple_claim_lines"])
        self.assertTrue(negated_errors["mismatched_claim_text_lines"])

    def test_report_requires_fixed_final_sections(self) -> None:
        report = """## Short Answer
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Key Findings
### Team
- Unmapped invented team fact.
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Conflicts and Caveats
- The program opened in 2026. [C2][S2][S3]
## Sources
- [S1] One — https://example.com/one
- [S2] Two — https://example.com/two
- [S3] Three — https://example.com/three
## Appendix
An invented fact after Sources.
"""

        errors = report_claim_mapping_errors(
            report,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )

        self.assertTrue(errors["invalid_section_structure"])
        self.assertTrue(errors["unmapped_finding_lines"])
        self.assertTrue(errors["invalid_section_lines"])

    def test_sources_section_rejects_unbulleted_fact_disguised_as_source(
        self,
    ) -> None:
        report = """## Short Answer
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Key Findings
- BIGAI lists Zhu Songchun as its president. [C1][S1]
## Conflicts and Caveats
- The program opened in 2026. [C2][S2][S3]
## Sources
[S1] INVENTED FACT OUTSIDE THE CLAIM GRAPH. One — https://example.com/one
- [S2] Two — https://example.com/two
- [S3] Three — https://example.com/three
"""

        errors = report_claim_mapping_errors(
            report,
            plan_claim_ids={"C1", "C2"},
            claims=self.claims,
            evidence_units=self.evidence,
        )

        self.assertTrue(errors["invalid_sources_section_lines"])

    def test_only_state_derived_citation_free_caveats_are_allowed(self) -> None:
        plan = {
            "status": "partial",
            "subquestions": [{"id": "SQ1", "status": "blocked", "claim_ids": []}],
        }
        allowed = allowed_report_caveat_lines(plan)
        valid_report = """## Short Answer
## Key Findings
## Conflicts and Caveats
- Research coverage is partial; unsupported subquestions: SQ1.
- No canonical claim passed the evidence gate.
## Sources
"""
        invented_report = """## Short Answer
## Key Findings
## Conflicts and Caveats
- 北京研究院已于2025年关闭。
## Sources
"""

        valid_errors = report_claim_mapping_errors(
            valid_report,
            plan_claim_ids=set(),
            claims=[],
            evidence_units=[],
            allowed_caveat_lines=allowed,
        )
        invented_errors = report_claim_mapping_errors(
            invented_report,
            plan_claim_ids=set(),
            claims=[],
            evidence_units=[],
            allowed_caveat_lines=allowed,
        )

        self.assertTrue(all(not value for value in valid_errors.values()))
        self.assertTrue(invented_errors["unauthorized_caveat_lines"])


if __name__ == "__main__":
    unittest.main()
