"""Probe search and page acquisition without invoking a model or Agent."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from retrieval_backend import RetrievalSession, build_default_session


DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent / "retrieval_fixtures" / "default_probe.json"
)
DEFAULT_OUTPUT_DIR = Path("output/retrieval_probe")


def load_probe_fixture(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load one independent JSON or JSONL retrieval fixture."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        search_cases: list[dict[str, Any]] = []
        fetch_cases: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            case_type = item.pop("type", None)
            if case_type == "search":
                search_cases.append(item)
            elif case_type == "fetch":
                fetch_cases.append(item)
            else:
                raise ValueError(
                    f"JSONL line {line_number} needs type=search or type=fetch"
                )
        return {"search_cases": search_cases, "fetch_cases": fetch_cases}

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Probe fixture must be a JSON object")
    return {
        "search_cases": _case_list(payload, "search_cases"),
        "fetch_cases": _case_list(payload, "fetch_cases"),
    }


def run_probe(
    fixture: Mapping[str, list[dict[str, Any]]],
    *,
    session: RetrievalSession | None = None,
    direct_fetch: Callable[[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run deterministic probe orchestration around injectable network adapters."""

    active_session = session or build_default_session()
    if direct_fetch is None:
        # Lazy import keeps fixture parsing and unit tests independent of the
        # application graph. This function never constructs or invokes a model.
        from search_agent import _direct_fetch_payload  # noqa: PLC0415

        direct_fetch = _direct_fetch_payload

    search_records = [
        _run_search_case(active_session, case)
        for case in fixture.get("search_cases", [])
    ]
    fetch_records = [
        _run_fetch_case(active_session, direct_fetch, case)
        for case in fixture.get("fetch_cases", [])
    ]
    metrics = _metrics(search_records, fetch_records)
    return {
        "probe_kind": "retrieval_only_no_model",
        "tavily_configured": bool(os.environ.get("TAVILY_API_KEY")),
        "search_cases": search_records,
        "fetch_cases": fetch_records,
        "metrics": metrics,
        "acceptance": {
            "search_top_5_at_least_80_percent": (
                metrics["postrank_recall_at_5"] >= 0.8
            ),
            "public_fetch_at_least_80_percent": (metrics["fetch_success_rate"] >= 0.8),
            "unsafe_rejection_100_percent": (
                metrics["unsafe_url_rejection_rate"] == 1.0
            ),
        },
    }


def write_probe_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write machine-readable observations and one compact human summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "retrieval_probe.md").write_text(
        _markdown_summary(report),
        encoding="utf-8",
    )


def _case_list(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return [dict(item) for item in value]


def _run_search_case(
    session: RetrievalSession,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    query = str(case.get("query", "")).strip()
    expected_hosts = [
        str(host).casefold().removeprefix("www.")
        for host in case.get("expected_hosts", [])
        if isinstance(host, str) and host.strip()
    ]
    execution = session.search(query, 5)
    public = execution.public_dict()
    raw_hits = [
        {
            "provider": outcome.provider,
            "provider_rank": result.provider_rank,
            "url": result.url,
        }
        for outcome in execution.provider_outcomes
        for result in outcome.results
        if result.provider_rank <= 5
        and _matches_expected_host(result.url, expected_hosts)
    ]
    postrank_hits = [
        {
            "final_rank": result.final_rank,
            "url": result.url,
        }
        for result in execution.results
        if result.final_rank <= 5 and _matches_expected_host(result.url, expected_hosts)
    ]
    return {
        "id": str(case.get("id", "")),
        "query": query,
        "expected_hosts": expected_hosts,
        "status": execution.status,
        "provider_statuses": public["provider_statuses"],
        "provider_raw_results": [
            outcome.public_dict() for outcome in execution.provider_outcomes
        ],
        "normalized_results": execution.normalized_candidates,
        "final_candidates": public["results"],
        "search_quality": execution.search_quality,
        "raw_top_5_hit": bool(raw_hits),
        "postrank_top_1_hit": any(item["final_rank"] == 1 for item in postrank_hits),
        "postrank_top_3_hit": any(item["final_rank"] <= 3 for item in postrank_hits),
        "postrank_top_5_hit": bool(postrank_hits),
        "raw_hits": raw_hits,
        "postrank_hits": postrank_hits,
    }


def _run_fetch_case(
    session: RetrievalSession,
    direct_fetch: Callable[[str, int], dict[str, Any]],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    requested_url = str(case.get("url", "")).strip()
    payload = session.fetch(
        requested_url,
        12_000,
        direct_fetch=direct_fetch,
    )
    return {
        "id": str(case.get("id", "")),
        "expected": str(case.get("expected", "success")),
        "requested_url": requested_url,
        "final_url": payload.get("final_url", payload.get("url")),
        "status": payload.get("status"),
        "acquisition_method": payload.get("acquisition_method"),
        "http_status": payload.get("http_status"),
        "content_type": payload.get("content_type"),
        "visible_content_chars": int(payload.get("content_chars", 0) or 0),
        "fallback_triggered": bool(payload.get("fallback_triggered", False)),
        "fallback_failure": payload.get("fallback_failure"),
        "failure_category": payload.get(
            "failure_taxonomy",
            payload.get("failure_type"),
        ),
        "safety_rejected": (
            payload.get("status") == "rejected"
            and payload.get("failure_taxonomy") == "unsafe_url"
        ),
    }


def _matches_expected_host(url: str, expected_hosts: list[str]) -> bool:
    host = (urlparse(url).hostname or "").casefold().removeprefix("www.")
    return any(
        host == expected or host.endswith(f".{expected}") for expected in expected_hosts
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _metrics(
    search_records: list[dict[str, Any]],
    fetch_records: list[dict[str, Any]],
) -> dict[str, float]:
    public_fetches = [item for item in fetch_records if item["expected"] == "success"]
    unsafe_fetches = [
        item for item in fetch_records if item["expected"] == "unsafe_url"
    ]
    return {
        "provider_success_rate": _ratio(
            sum(item["status"] == "success" for item in search_records),
            len(search_records),
        ),
        "raw_recall_at_5": _ratio(
            sum(bool(item["raw_top_5_hit"]) for item in search_records),
            len(search_records),
        ),
        "postrank_recall_at_5": _ratio(
            sum(bool(item["postrank_top_5_hit"]) for item in search_records),
            len(search_records),
        ),
        "fetch_success_rate": _ratio(
            sum(item["status"] == "success" for item in public_fetches),
            len(public_fetches),
        ),
        "content_extraction_success_rate": _ratio(
            sum(
                item["status"] == "success" and item["visible_content_chars"] > 0
                for item in public_fetches
            ),
            len(public_fetches),
        ),
        "unsafe_url_rejection_rate": _ratio(
            sum(bool(item["safety_rejected"]) for item in unsafe_fetches),
            len(unsafe_fetches),
        ),
    }


def _markdown_summary(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    acceptance = report["acceptance"]
    lines = [
        "# Retrieval probe",
        "",
        "No model or Agent was invoked.",
        "",
        f"- Tavily configured: `{report['tavily_configured']}`",
        f"- Provider success rate: `{metrics['provider_success_rate']:.1%}`",
        f"- Raw recall@5: `{metrics['raw_recall_at_5']:.1%}`",
        f"- Post-rank recall@5: `{metrics['postrank_recall_at_5']:.1%}`",
        f"- Fetch success rate: `{metrics['fetch_success_rate']:.1%}`",
        (
            "- Content extraction success rate: "
            f"`{metrics['content_extraction_success_rate']:.1%}`"
        ),
        (f"- Unsafe URL rejection rate: `{metrics['unsafe_url_rejection_rate']:.1%}`"),
        "",
        "## Acceptance",
        "",
    ]
    lines.extend(
        f"- {name}: `{'PASS' if passed else 'FAIL'}`"
        for name, passed in acceptance.items()
    )
    lines.extend(["", "## Cases", ""])
    lines.extend(
        (
            f"- search `{item['id']}`: status={item['status']}, "
            f"quality={item['search_quality']}, "
            f"top5={item['postrank_top_5_hit']}"
        )
        for item in report["search_cases"]
    )
    lines.extend(
        (
            f"- fetch `{item['id']}`: status={item['status']}, "
            f"method={item['acquisition_method']}, "
            f"failure={item['failure_category']}"
        )
        for item in report["fetch_cases"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the retrieval-only probe from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    fixture = load_probe_fixture(args.fixture)
    report = run_probe(fixture)
    write_probe_outputs(report, args.output_dir)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"probe_json={args.output_dir / 'retrieval_probe.json'}")
    print(f"probe_markdown={args.output_dir / 'retrieval_probe.md'}")


if __name__ == "__main__":
    main()
