"""Freeze the answer-blind FRAMES set for the model-Harness scaling study.

Selection is intentionally isolated from reference answers and provenance URLs.
Only the stable source index, question text, and original ``reasoning_types``
metadata participate in filtering, bucketing, and ordering.  Reference answers
are attached only after the selected indices are immutable so the existing
frozen scorer can evaluate completed runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_ID = "google/frames-benchmark"
DATASET_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
SOURCE_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"
SOURCE_ROW_COUNT = 824
SELECTION_VERSION = "frames-reasoning-metadata-balanced-v1"
CATEGORY_ORDER = (
    "relation_multi_constraint",
    "numerical_temporal",
    "table_enumeration_count",
    "post_processing_mixed",
)


@dataclass(frozen=True)
class SelectionRow:
    """The complete answer-blind view available to the selector."""

    source_index: int
    question: str
    reasoning_types: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category(reasoning_types: str) -> str | None:
    tags = {item.strip() for item in reasoning_types.split("|") if item.strip()}
    if "Post processing" in tags:
        return "post_processing_mixed"
    if "Tabular reasoning" in tags:
        return "table_enumeration_count"
    if tags.intersection({"Numerical reasoning", "Temporal reasoning"}):
        return "numerical_temporal"
    if "Multiple constraints" in tags:
        return "relation_multi_constraint"
    return None


def load_selection_rows(source: Path) -> list[SelectionRow]:
    """Load only fields authorized for deterministic selection."""

    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("FRAMES source SHA-256 does not match the frozen revision")
    rows: list[SelectionRow] = []
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for ordinal, raw in enumerate(reader):
            if str(raw.get("", "")).strip() != str(ordinal):
                raise ValueError(f"non-contiguous FRAMES source index at row {ordinal}")
            rows.append(
                SelectionRow(
                    source_index=ordinal,
                    question=str(raw.get("Prompt", "")).strip(),
                    reasoning_types=str(raw.get("reasoning_types", "")).strip(),
                )
            )
    if len(rows) != SOURCE_ROW_COUNT:
        raise ValueError(
            f"expected {SOURCE_ROW_COUNT} source rows, observed {len(rows)}"
        )
    return rows


def select_indices(
    rows: list[SelectionRow], excluded_ids: set[str], *, per_category: int = 3
) -> tuple[list[SelectionRow], dict[str, list[int]]]:
    """Select the first source indices in each metadata-only category."""

    buckets: dict[str, list[SelectionRow]] = {name: [] for name in CATEGORY_ORDER}
    for row in sorted(rows, key=lambda item: item.source_index):
        task_id = f"frames-test-{row.source_index:04d}"
        category = _category(row.reasoning_types)
        if task_id in excluded_ids or not row.question or category is None:
            continue
        if len(buckets[category]) < per_category:
            buckets[category].append(row)
    short = {
        name: len(items) for name, items in buckets.items() if len(items) < per_category
    }
    if short:
        raise ValueError(f"insufficient rows for balanced selection: {short}")
    selected = sorted(
        (row for category in CATEGORY_ORDER for row in buckets[category]),
        key=lambda item: item.source_index,
    )
    return selected, {
        category: [row.source_index for row in buckets[category]]
        for category in CATEGORY_ORDER
    }


def _selected_answers(source: Path, selected_indices: set[int]) -> dict[int, str]:
    """Attach gold only after selection; never expose it to selection functions."""

    answers: dict[int, str] = {}
    with source.open(encoding="utf-8", newline="") as handle:
        for ordinal, raw in enumerate(csv.DictReader(handle, delimiter="\t")):
            if ordinal in selected_indices:
                answers[ordinal] = str(raw.get("Answer", "")).strip()
    if set(answers) != selected_indices or any(not value for value in answers.values()):
        raise ValueError("a frozen task lacks a non-empty reference answer")
    return answers


def _jsonl_bytes(selected: list[SelectionRow], answers: dict[int, str]) -> bytes:
    lines = []
    for row in selected:
        payload = {
            "id": f"frames-test-{row.source_index:04d}",
            "question": row.question,
            "reference_answer": answers[row.source_index],
            "source_dataset": DATASET_ID,
            "source_split": "test",
            "source_index": row.source_index,
            "metadata": {
                "benchmark_status": "final_model_harness_scaling_frozen",
                "dataset_revision": DATASET_REVISION,
                "reasoning_types": row.reasoning_types,
                "selection_category": _category(row.reasoning_types),
                "source_file_sha256": f"sha256:{SOURCE_SHA256}",
                "source_row_id": str(row.source_index),
            },
        }
        lines.append(
            json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )
    return ("\n".join(lines) + "\n").encode()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def prepare(
    *, source: Path, excluded_ids_path: Path, dataset: Path, manifest: Path
) -> dict[str, Any]:
    excluded_ids = {
        line.strip()
        for line in excluded_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = load_selection_rows(source)
    selected, category_indices = select_indices(rows, excluded_ids)
    answers = _selected_answers(source, {row.source_index for row in selected})
    dataset_bytes = _jsonl_bytes(selected, answers)
    selected_ids = [f"frames-test-{row.source_index:04d}" for row in selected]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "final_model_harness_frames12",
        "benchmark_status": "frozen_formal_scaling_study",
        "source_dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": f"sha256:{SOURCE_SHA256}",
        "selection_version": SELECTION_VERSION,
        "selection_inputs": ["task_id", "question", "reasoning_types"],
        "selection_prohibited_inputs": [
            "reference_answer",
            "gold_articles",
            "source_urls",
            "historical_results",
            "manual_difficulty_judgment",
        ],
        "ordering": "ascending_source_index_within_category_then_global_ascending_source_index",
        "category_priority": list(CATEGORY_ORDER),
        "category_source_indices": category_indices,
        "historical_excluded_count": len(excluded_ids),
        "historical_excluded_task_ids": sorted(excluded_ids),
        "selected_count": len(selected),
        "selected_source_indices": [row.source_index for row in selected],
        "selected_task_ids": selected_ids,
        "smoke_task_ids": selected_ids[:2],
        "dataset_sha256": f"sha256:{hashlib.sha256(dataset_bytes).hexdigest()}",
        "formal_matrix": {
            "weak_model": {
                "model": "gpt-5.4-nano",
                "systems": ["bare_simple_react", "tongagent_standard"],
                "jobs": 24,
            },
            "strong_model": {
                "model": "gpt-5.5",
                "systems": [
                    "bare_simple_react",
                    "tongagent_standard",
                    "vanilla_deepagents",
                ],
                "jobs": 36,
            },
            "total_jobs": 60,
        },
        "cost_upper_bound": {
            "currency": "USD",
            "token_ceiling_per_run": 60000,
            "max_output_tokens_per_run": 3000,
            "weak_prices_per_million": {
                "uncached_input": 0.20,
                "cached_input": 0.02,
                "output": 1.25,
            },
            "strong_prices_per_million": {
                "uncached_input": 5.0,
                "cached_input": 0.5,
                "output": 30.0,
            },
            "weak_core_24_runs": 0.3636,
            "strong_core_36_runs": 13.5,
            "core_60_runs": 13.8636,
            "cap": 15.0,
            "cap_pass": True,
        },
    }
    _atomic_write(dataset, dataset_bytes)
    _atomic_write(
        manifest,
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--excluded-ids", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(
        source=args.source,
        excluded_ids_path=args.excluded_ids,
        dataset=args.dataset,
        manifest=args.manifest,
    )
    print(
        json.dumps(
            {
                "selected_task_ids": manifest["selected_task_ids"],
                "category_source_indices": manifest["category_source_indices"],
                "core_jobs": manifest["formal_matrix"]["total_jobs"],
                "cost_upper_bound": manifest["cost_upper_bound"]["core_60_runs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
