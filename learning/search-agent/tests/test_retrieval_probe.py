"""Offline tests for the retrieval-only probe artifact."""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_backend import RetrievalSession, SearchBroker
from retrieval_probe import load_probe_fixture, run_probe, write_probe_outputs
from retrieval_providers import ProviderSearchResponse, SearchResult


class ProbeProvider:
    name = "probe"

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        del max_results
        return ProviderSearchResponse(
            provider=self.name,
            query=query,
            status="success",
            http_status=200,
            results=[
                SearchResult(
                    title="Alan Turing biography",
                    url="https://en.wikipedia.org/wiki/Alan_Turing",
                    snippet="Alan Turing biography",
                    provider=self.name,
                    provider_rank=1,
                )
            ],
        )


def test_probe_computes_metrics_and_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    fixture = {
        "search_cases": [
            {
                "id": "turing",
                "query": "Alan Turing biography",
                "expected_hosts": ["en.wikipedia.org"],
            }
        ],
        "fetch_cases": [
            {
                "id": "public",
                "url": "https://example.com/",
                "expected": "success",
            },
            {
                "id": "private",
                "url": "http://127.0.0.1/private",
                "expected": "unsafe_url",
            },
        ],
    }
    session = RetrievalSession(SearchBroker([ProbeProvider()]))

    report = run_probe(
        fixture,
        session=session,
        direct_fetch=lambda url, max_chars: {
            "status": "success",
            "url": url,
            "content": "visible public content",
            "content_chars": 22,
            "content_type": "text/html",
            "max_chars": max_chars,
        },
    )
    write_probe_outputs(report, tmp_path)

    assert report["probe_kind"] == "retrieval_only_no_model"
    assert report["metrics"]["provider_success_rate"] == 1.0
    assert report["metrics"]["raw_recall_at_5"] == 1.0
    assert report["metrics"]["postrank_recall_at_5"] == 1.0
    assert report["metrics"]["fetch_success_rate"] == 1.0
    assert report["metrics"]["unsafe_url_rejection_rate"] == 1.0
    machine = json.loads(
        (tmp_path / "retrieval_probe.json").read_text(encoding="utf-8")
    )
    human = (tmp_path / "retrieval_probe.md").read_text(encoding="utf-8")
    assert machine["metrics"] == report["metrics"]
    assert "No model or Agent was invoked." in human


def test_probe_jsonl_fixture_is_independent_and_typed(tmp_path: Path) -> None:
    fixture_path = tmp_path / "probe.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "search",
                        "id": "docs",
                        "query": "Python docs",
                        "expected_hosts": ["docs.python.org"],
                    }
                ),
                json.dumps(
                    {
                        "type": "fetch",
                        "id": "html",
                        "url": "https://example.com/",
                        "expected": "success",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_probe_fixture(fixture_path)

    assert loaded["search_cases"][0]["id"] == "docs"
    assert loaded["fetch_cases"][0]["id"] == "html"
    assert "type" not in loaded["search_cases"][0]
