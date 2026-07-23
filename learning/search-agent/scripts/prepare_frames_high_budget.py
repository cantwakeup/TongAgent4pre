"""Freeze the answer-blind FRAMES-8 high-budget exploratory benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_INDICES = tuple(range(8))
MANIFEST_ID = "frames_high_budget_8"
EXPERIMENT_ID = "frames-high-budget-completion-v1"
PRIOR_SMOKE_TASK_IDS = frozenset({"frames-test-0000", "frames-test-0001"})


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(
    *,
    parent_manifest_path: Path,
    parent_dataset_path: Path,
    output_manifest_path: Path,
    output_dataset_path: Path,
) -> dict[str, Any]:
    """Select source indices 0..7 from the already-frozen FRAMES-12 bytes."""

    parent_manifest_bytes = parent_manifest_path.read_bytes()
    parent_manifest = json.loads(parent_manifest_bytes)
    if parent_manifest.get("manifest_id") != "final_model_harness_frames12":
        raise ValueError("parent manifest is not final_model_harness_frames12")

    raw_lines = [
        line
        for line in parent_dataset_path.read_bytes().splitlines(keepends=True)
        if line.strip()
    ]
    views: dict[int, dict[str, Any]] = {}
    selected_line_bytes: dict[int, bytes] = {}
    for line in raw_lines:
        payload = json.loads(line)
        source_index = payload.get("source_index")
        task_id = payload.get("id")
        question = payload.get("question")
        if (
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and isinstance(task_id, str)
            and isinstance(question, str)
        ):
            views[source_index] = {
                "task_id": task_id,
                "question": question,
                "source_index": source_index,
            }
            selected_line_bytes[source_index] = line

    if not all(index in views for index in EXPECTED_SOURCE_INDICES):
        raise ValueError("parent FRAMES-12 does not contain every source index 0..7")
    selected = [views[index] for index in EXPECTED_SOURCE_INDICES]
    selected_ids = [str(item["task_id"]) for item in selected]
    expected_ids = [f"frames-test-{index:04d}" for index in EXPECTED_SOURCE_INDICES]
    if selected_ids != expected_ids:
        raise ValueError("source-index/task-id identity mismatch")

    parent_selected_indices = parent_manifest.get("selected_source_indices")
    parent_selected_ids = parent_manifest.get("selected_task_ids")
    if not isinstance(parent_selected_indices, list) or not isinstance(
        parent_selected_ids, list
    ):
        raise ValueError("parent manifest lacks frozen selections")
    if parent_selected_indices[:8] != list(EXPECTED_SOURCE_INDICES):
        raise ValueError("parent manifest first eight source indices changed")
    if parent_selected_ids[:8] != selected_ids:
        raise ValueError("parent manifest first eight task ids changed")

    historical = set(parent_manifest.get("historical_excluded_task_ids", []))
    leakage = sorted(historical.intersection(selected_ids))
    if leakage:
        raise ValueError(f"historical task leakage detected: {leakage}")

    dataset_bytes = b"".join(
        selected_line_bytes[index] for index in EXPECTED_SOURCE_INDICES
    )
    question_hashes = {
        str(item["task_id"]): _sha256_bytes(str(item["question"]).encode("utf-8"))
        for item in selected
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "experiment_id": EXPERIMENT_ID,
        "benchmark_status": "exploratory_high_budget_benchmark",
        "parent_manifest": str(parent_manifest_path),
        "parent_manifest_sha256": _sha256_bytes(parent_manifest_bytes),
        "parent_dataset": str(parent_dataset_path),
        "parent_dataset_sha256": _sha256_bytes(parent_dataset_path.read_bytes()),
        "selection_rule": "ascending_source_index_first_8",
        "selection_inputs": ["task_id", "question", "source_index"],
        "selection_prohibited_inputs": [
            "reference_answer",
            "gold_articles",
            "source_urls",
            "historical_correctness",
            "manual_difficulty_judgment",
        ],
        "selected_count": 8,
        "selected_source_indices": list(EXPECTED_SOURCE_INDICES),
        "selected_task_ids": selected_ids,
        "question_sha256": question_hashes,
        "dataset_sha256": _sha256_bytes(dataset_bytes),
        "validation": {
            "historical_leakage": len(leakage),
            "historical_leakage_task_ids": leakage,
            "question_hash_match": "8/8",
            "source_index_match": "8/8",
            "prior_scaling_smoke_overlap": len(
                PRIOR_SMOKE_TASK_IDS.intersection(selected_ids)
            ),
            "prior_scaling_smoke_task_ids": sorted(
                PRIOR_SMOKE_TASK_IDS.intersection(selected_ids)
            ),
        },
        "matrix": {
            "systems": ["bare_simple_react", "tongagent_standard"],
            "model": "gpt-5.5",
            "tasks": 8,
            "core_jobs": 16,
            "concurrency": 1,
            "job_order": "task_id_then_system_manifest_order",
        },
        "shared_budget": {
            "total_token_ceiling": 100000,
            "max_output_tokens": 5000,
            "watchdog_seconds": 900,
            "max_search_calls": 4,
            "max_fetch_calls": 6,
            "max_model_calls": 16,
        },
        "shared_finalization_boundary": {
            "token_trigger": 90000,
            "wall_time_trigger_seconds": 840,
            "remaining_model_calls_trigger": 1,
        },
        "cost_policy": {
            "currency": "USD",
            "known_cost_cap": 6.0,
            "uncached_input_per_million": 5.0,
            "cached_input_per_million": 0.5,
            "output_per_million": 30.0,
            "stop_after_current_attempt": True,
        },
    }
    _atomic_write(output_dataset_path, dataset_bytes)
    _atomic_write(
        output_manifest_path,
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(
        parent_manifest_path=args.parent_manifest,
        parent_dataset_path=args.parent_dataset,
        output_manifest_path=args.output_manifest,
        output_dataset_path=args.output_dataset,
    )
    print(
        json.dumps(
            {
                "manifest_id": payload["manifest_id"],
                "selected_task_ids": payload["selected_task_ids"],
                "validation": payload["validation"],
                "core_jobs": payload["matrix"]["core_jobs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
