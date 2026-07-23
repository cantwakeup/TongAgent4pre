"""Offline contracts for the exploratory FRAMES high-budget study."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_frames_high_budget import prepare
from scripts.run_frames_high_budget import run


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_high_budget_subset_is_exact_answer_blind_parent_prefix(tmp_path: Path) -> None:
    dataset = tmp_path / "frames_high_budget_8.jsonl"
    manifest = tmp_path / "frames_high_budget_8.json"
    payload = prepare(
        parent_manifest_path=(
            PACKAGE_ROOT
            / "evaluation"
            / "manifests"
            / "final_model_harness_frames12.json"
        ),
        parent_dataset_path=(
            PACKAGE_ROOT
            / "evaluation"
            / "datasets"
            / "final_model_harness_frames12.jsonl"
        ),
        output_manifest_path=manifest,
        output_dataset_path=dataset,
    )
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]

    assert payload["benchmark_status"] == "exploratory_high_budget_benchmark"
    assert payload["selected_source_indices"] == list(range(8))
    assert payload["selected_task_ids"] == [
        f"frames-test-{index:04d}" for index in range(8)
    ]
    assert payload["selection_inputs"] == ["task_id", "question", "source_index"]
    assert "reference_answer" in payload["selection_prohibited_inputs"]
    assert payload["validation"]["historical_leakage"] == 0
    assert payload["validation"]["question_hash_match"] == "8/8"
    assert payload["validation"]["source_index_match"] == "8/8"
    assert payload["validation"]["prior_scaling_smoke_overlap"] == 2
    assert [row["source_index"] for row in rows] == list(range(8))
    assert json.loads(manifest.read_text()) == payload


def test_high_budget_dry_run_is_exactly_sixteen_task_major_jobs(
    tmp_path: Path,
) -> None:
    payload = run(
        dataset_path=(
            PACKAGE_ROOT / "evaluation" / "datasets" / "frames_high_budget_8.jsonl"
        ),
        manifest_path=(
            PACKAGE_ROOT / "evaluation" / "manifests" / "frames_high_budget_8.json"
        ),
        config_path=(
            PACKAGE_ROOT
            / "evaluation"
            / "configs"
            / "frames_high_budget_gpt55.live.json"
        ),
        output_root=tmp_path,
        seed=17,
        dry_run=True,
    )

    assert payload["dry_run_jobs"] == 16
    assert payload["benchmark_status"] == "exploratory high-budget benchmark"
    assert payload["job_plan"][:4] == [
        {"task_id": "frames-test-0000", "system_id": "bare_simple_react"},
        {"task_id": "frames-test-0000", "system_id": "tongagent_standard"},
        {"task_id": "frames-test-0001", "system_id": "bare_simple_react"},
        {"task_id": "frames-test-0001", "system_id": "tongagent_standard"},
    ]
    assert payload["fairness"]["bare_standard_prompt_parity"] is True
    assert payload["fairness"]["boundary"] == {
        "enabled": True,
        "token_trigger": 90000,
        "wall_time_trigger_seconds": 840.0,
        "remaining_model_calls_trigger": 1,
    }
    assert not (tmp_path / "frames-high-budget-completion-v1").exists()
