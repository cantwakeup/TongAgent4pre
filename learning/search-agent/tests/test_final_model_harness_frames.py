from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_final_model_harness_frames import SelectionRow, select_indices


def _rows() -> list[SelectionRow]:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "evaluation" / "datasets" / "final_model_harness_frames12.jsonl"
    rows: list[SelectionRow] = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        rows.append(
            SelectionRow(
                source_index=payload["source_index"],
                question=payload["question"],
                reasoning_types=payload["metadata"]["reasoning_types"],
            )
        )
    return rows


def test_frozen_frames12_is_balanced_and_ascending() -> None:
    rows = _rows()
    assert len(rows) == 12
    assert [row.source_index for row in rows] == sorted(
        row.source_index for row in rows
    )
    categories = []
    for line in (
        (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "datasets"
            / "final_model_harness_frames12.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        categories.append(json.loads(line)["metadata"]["selection_category"])
    assert {category: categories.count(category) for category in set(categories)} == {
        "relation_multi_constraint": 3,
        "numerical_temporal": 3,
        "table_enumeration_count": 3,
        "post_processing_mixed": 3,
    }


def test_selector_has_no_reference_answer_input() -> None:
    selected, buckets = select_indices(_rows(), set(), per_category=3)
    assert len(selected) == 12
    assert all(len(indices) == 3 for indices in buckets.values())
    assert all(not hasattr(row, "reference_answer") for row in selected)
