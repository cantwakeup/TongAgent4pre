"""Stage D fixture-suite regressions.

These tests validate the deterministic smoke inputs only.  The process-isolated
18-run matrix is exercised explicitly by ``scripts/run_stage_d_smoke.sh`` so
the ordinary unit suite never launches an accidental benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_graph import source_diversity_metrics
from evaluation.offline import FixtureBackend
from evaluation.validate_stage_d import (
    StageDSmokeValidationError,
    _validate_command_logs,
    validate_inputs,
)


pytestmark = pytest.mark.usefixtures("socket_disabled")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "stage_d_offline_smoke.jsonl"
FIXTURES = PROJECT_ROOT / "evaluation" / "fixtures"
TRACKED_RESULTS = PROJECT_ROOT / "evaluation" / "results" / "stage_d_offline_smoke"


def test_stage_d_inputs_cover_six_truthful_closed_world_scenarios() -> None:
    summary = validate_inputs(DATASET, FIXTURES)

    assert summary == {
        "fixture_kind": "deterministic_offline_smoke",
        "benchmark_status": "smoke_only_not_a_formal_benchmark",
        "task_count": 6,
        "scenario_count": 6,
        "search_fixture_count": 8,
        "page_fixture_count": 9,
    }


def test_stage_d_irrelevant_and_retry_fixtures_execute_as_declared() -> None:
    backend = FixtureBackend.from_directory(FIXTURES)

    irrelevant = backend.search("stage d missing lunar certificate")
    assert irrelevant["status"] == "success"
    assert irrelevant["provider_success"] is True
    assert irrelevant["nonempty_search"] is True
    assert irrelevant["relevant_results"] == 0
    assert irrelevant["relevant_search"] is False

    first_fetch = backend.fetch("https://delta-archive.fixture.test/catalog")
    second_fetch = backend.fetch("https://delta-archive.fixture.test/catalog")
    assert first_fetch["status"] == "error"
    assert first_fetch["retryable"] is True
    assert first_fetch["fixture_response_index"] == 0
    assert second_fetch["status"] == "success"
    assert second_fetch["fixture_response_index"] == 1
    assert (
        "The Delta Archive lists 17 verified field notebooks" in second_fetch["content"]
    )


def test_stage_d_fixture_misses_fail_closed_without_network_fallback() -> None:
    backend = FixtureBackend.from_directory(FIXTURES)

    missing_search = backend.search("not a Stage D fixture query")
    missing_page = backend.fetch("https://unknown.fixture.test/not-present")

    assert missing_search["status"] == "fixture_not_found"
    assert missing_search["provider_outcome"] == "not_called"
    assert missing_search["provider_failure"] is False
    assert missing_page["status"] == "fixture_not_found"
    assert missing_page["provider_outcome"] == "not_called"
    assert missing_page["retryable"] is False


def test_stage_d_source_independence_rejects_same_host_and_exact_mirrors() -> None:
    same_host = source_diversity_metrics(
        source_ids={"S1", "S2"},
        sources=[
            {"source_id": "S1", "url": "https://publisher.fixture.test/a"},
            {"source_id": "S2", "url": "https://publisher.fixture.test/b"},
        ],
        evidence_units=[
            {
                "claim_id": "C1",
                "source_id": "S1",
                "stance": "supports",
                "source_content_sha256": "a" * 64,
            },
            {
                "claim_id": "C1",
                "source_id": "S2",
                "stance": "supports",
                "source_content_sha256": "b" * 64,
            },
        ],
    )
    exact_mirror = source_diversity_metrics(
        source_ids={"S1", "S2"},
        sources=[
            {"source_id": "S1", "url": "https://one.fixture.test/a"},
            {"source_id": "S2", "url": "https://two.fixture.test/b"},
        ],
        evidence_units=[
            {
                "claim_id": "C1",
                "source_id": "S1",
                "stance": "supports",
                "source_content_sha256": "c" * 64,
            },
            {
                "claim_id": "C1",
                "source_id": "S2",
                "stance": "supports",
                "source_content_sha256": "c" * 64,
            },
        ],
    )

    assert same_host["distinct_source_host_count"] == 1
    assert same_host["corroborating_source_group_count"] == 1
    assert exact_mirror["distinct_source_host_count"] == 2
    assert exact_mirror["distinct_content_revision_count"] == 1
    assert exact_mirror["corroborating_source_group_count"] == 1


def test_stage_d_strict_validator_requires_fresh_and_resume_proof(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        StageDSmokeValidationError,
        match="requires fresh and resume command logs",
    ):
        _validate_command_logs(tmp_path, require_rerun=False)


def test_stage_d_tracked_export_is_complete_and_sanitized() -> None:
    summary = json.loads((TRACKED_RESULTS / "summary.json").read_text(encoding="utf-8"))
    representative = json.loads(
        (TRACKED_RESULTS / "representative_run.json").read_text(encoding="utf-8")
    )
    exported_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(TRACKED_RESULTS.iterdir())
        if path.is_file()
    )

    assert summary["selected_result_count"] == 18
    assert len(summary["results"]) == 18
    assert summary["resume_validation"] == {
        "fresh_executed": 18,
        "fresh_skipped": 0,
        "resume_executed": 0,
        "resume_skipped": 18,
        "result_hashes_unchanged": True,
        "targeted_rerun_executed": 1,
        "strict_validation": "passed",
    }
    for row in summary["results"]:
        if row["system_id"] in {"simple_react", "vanilla_deepagents"}:
            assert row["evidence_count"] is None
            assert row["structural_subquestion_coverage"] is None
    assert representative["run_result"]["task_id"] == "fixture-nonempty-irrelevant"
    assert set(representative["companions"]) == {
        "answer.md",
        "failure.json",
        "metrics.json",
        "trace.json",
    }
    assert "/home/" not in exported_text
