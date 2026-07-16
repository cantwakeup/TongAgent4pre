"""Tests for strict canonical source mapping in generated reports."""

from __future__ import annotations

import unittest

from search_agent import _canonical_mapping_errors


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


if __name__ == "__main__":
    unittest.main()
