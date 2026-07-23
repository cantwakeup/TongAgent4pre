"""Offline contracts for the BrowseComp resource-envelope study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_browsecomp_high_budget import (
    _completed_prefix,
    _job_plan,
    _ordered_tasks,
    run,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET = PACKAGE_ROOT / "evaluation" / "datasets" / "final_harness_benchmark_v2.jsonl"
MANIFEST = PACKAGE_ROOT / "evaluation" / "manifests" / "browsecomp_high_budget_8.json"
CONFIG = (
    PACKAGE_ROOT / "evaluation" / "configs" / "browsecomp_high_budget_gpt55.live.json"
)


def test_manifest_reuses_exact_formal_browsecomp_eight() -> None:
    resource = json.loads(MANIFEST.read_text(encoding="utf-8"))
    formal = json.loads(
        (
            PACKAGE_ROOT
            / "evaluation"
            / "manifests"
            / "final_harness_benchmark_v2.json"
        ).read_text(encoding="utf-8")
    )
    formal_tasks = next(
        item["tasks"] for item in formal["datasets"] if item["name"] == "BrowseComp"
    )

    assert resource["study_type"] == "exploratory resource-envelope study"
    assert resource["formal_result_replacement"] is False
    assert resource["selected_task_ids"] == [item["task_id"] for item in formal_tasks]
    assert resource["selected_source_indices"] == [
        item["source_index"] for item in formal_tasks
    ]
    assert resource["cost_guard"]["worst_case_total_cost"] == 11.6


def test_dry_run_is_exactly_sixteen_fair_task_major_jobs(tmp_path: Path) -> None:
    payload = run(
        dataset_path=DATASET,
        manifest_path=MANIFEST,
        config_path=CONFIG,
        output_root=tmp_path,
        experiment_id="browsecomp-high-budget-resource-envelope-v1-dry-run",
        seed=17,
        resume=False,
        dry_run=True,
        expected_head=None,
    )

    assert payload["dry_run_jobs"] == 16
    assert payload["study_type"] == "exploratory resource-envelope study"
    assert payload["worst_case_total_cost"] == 11.6
    assert payload["fairness"]["bare_standard_prompt_parity"] is True
    assert payload["fairness"]["budget"] == {
        "max_search_calls": 8,
        "max_fetch_calls": 12,
        "max_total_tool_calls": 20,
        "max_model_calls": 20,
        "max_total_tokens": 120000,
        "wall_time_seconds": 1200.0,
        "max_results_per_search": 5,
        "max_page_chars": 12000,
        "recursion_limit": 125,
    }
    first = payload["job_plan"][0]["task_id"]
    assert payload["job_plan"][:4] == [
        {"task_id": first, "system_id": "bare_simple_react"},
        {"task_id": first, "system_id": "tongagent_standard"},
        {
            "task_id": "browsecomp-02871b42634d2d7b",
            "system_id": "bare_simple_react",
        },
        {
            "task_id": "browsecomp-02871b42634d2d7b",
            "system_id": "tongagent_standard",
        },
    ]
    assert not (tmp_path / payload["experiment_id"]).exists()


def test_resume_refuses_incomplete_attempt_instead_of_creating_attempt_two(
    tmp_path: Path,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest, tasks = _ordered_tasks(DATASET, manifest, seed=17)
    plan = _job_plan(tasks)
    first = plan[0]
    attempt = tmp_path / first["system_id"] / first["task_id"] / "attempt-0001"
    attempt.mkdir(parents=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="incomplete attempt-0001"):
        _completed_prefix(
            experiment_directory=tmp_path,
            plan=plan,
            task_by_id={task.id: task for task in tasks},
            dataset_digest=digest,
            seed=17,
            git_sha="frozen",
            config_overrides=config,
        )

    assert not attempt.with_name("attempt-0002").exists()
