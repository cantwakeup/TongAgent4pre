"""Tests for strict canonical source mapping in generated reports."""

from __future__ import annotations

import unittest

from search_agent import (
    _canonical_mapping_errors,
    _canonicalize_source_section,
    _report_finding_source_ids,
)


SOURCES = [
    {
        "source_id": "S1",
        "title": "Official about page",
        "url": "https://official.example/about/",
    },
    {
        "source_id": "S2",
        "title": "Official program page",
        "url": "https://official.example/program/",
    },
]


class CanonicalSourceMappingTests(unittest.TestCase):
    """Require each source line to preserve the ledger's exact mapping."""

    def test_canonical_source_lines_pass(self) -> None:
        report = """# Report

Fact [S1]. Program details [S2].

## Sources
- [S1] Official about page — https://official.example/about/
- [S2] Official program page — https://official.example/program/
"""

        self.assertEqual(
            _canonical_mapping_errors(report, SOURCES),
            {"missing_urls": [], "mismatched_titles": []},
        )

    def test_swapped_urls_are_rejected_per_source_id(self) -> None:
        report = """## Sources
- [S1] Official about page — https://official.example/program/
- [S2] Official program page — https://official.example/about/
"""

        self.assertEqual(
            _canonical_mapping_errors(report, SOURCES)["missing_urls"],
            ["S1", "S2"],
        )

    def test_wrong_title_is_rejected_even_with_canonical_url(self) -> None:
        report = """## Sources
- [S2] TongProgram renamed locally — https://official.example/program/
"""

        self.assertEqual(
            _canonical_mapping_errors(report, [SOURCES[1]])["mismatched_titles"],
            ["S2"],
        )

    def test_simple_model_source_list_is_canonicalized_from_inline_ids(self) -> None:
        report = """## Short Answer
- Canonical claim. [C1][S2]
## Key Findings
- Canonical claim. [C1][S2]
## Conflicts and Caveats
## Sources
- [S2] Program page https://official.example/program/
"""

        canonical, changed = _canonicalize_source_section(report, SOURCES)

        self.assertTrue(changed)
        self.assertIn(
            "- [S2] Official program page — https://official.example/program/",
            canonical,
        )

    def test_source_canonicalizer_does_not_hide_appendix_prose(self) -> None:
        report = """## Short Answer
- Canonical claim. [C1][S1]
## Key Findings
- Canonical claim. [C1][S1]
## Conflicts and Caveats
## Sources
- Official about page https://official.example/about/
## Appendix
Invented prose.
"""

        canonical, changed = _canonicalize_source_section(report, SOURCES)

        self.assertFalse(changed)
        self.assertEqual(canonical, report)

    def test_source_canonicalizer_does_not_hide_url_bearing_prose(self) -> None:
        report = """## Short Answer
- Canonical claim. [C1][S1]
## Key Findings
- Canonical claim. [C1][S1]
## Conflicts and Caveats
## Sources
- [S1] Official about page https://official.example/about/
Invented appendix fact at https://attacker.example/fake.
"""

        canonical, changed = _canonicalize_source_section(report, SOURCES)

        self.assertFalse(changed)
        self.assertEqual(canonical, report)

    def test_heading_citation_does_not_create_a_cited_source(self) -> None:
        report = """# Research report [S2]
## Short Answer
- Canonical claim. [C1][S1]
## Key Findings
## Conflicts and Caveats
## Sources
- [S1] Official about page https://official.example/about/
"""

        canonical, changed = _canonicalize_source_section(report, SOURCES)

        self.assertTrue(changed)
        self.assertEqual(_report_finding_source_ids(canonical), ["S1"])
        self.assertIn("- [S1] Official about page", canonical)
        self.assertNotIn("- [S2]", canonical)


if __name__ == "__main__":
    unittest.main()
