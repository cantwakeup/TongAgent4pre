"""Deterministic, offline tests for the FRAMES five-task pilot adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.execution import load_jsonl_dataset
from evaluation.prepare_frames_pilot import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_URL,
    LICENSE,
    SOURCE_CONFIG,
    SOURCE_SHA256,
    SOURCE_SPLIT,
    SOURCE_URL,
    SourceRow,
    build_manifest,
    eligibility_failure,
    jsonl_bytes,
    load_official_rows,
    select_tasks,
)
from evaluation.schema import EvalTask


def test_eval_task_requires_complete_external_provenance() -> None:
    task = EvalTask(
        id="frames-test-0001",
        question="Question?",
        reference_answer="Answer",
        source_dataset=DATASET_ID,
        source_split="test",
        source_index=1,
    )

    assert task.source_dataset == DATASET_ID
    assert task.source_split == "test"
    assert task.source_index == 1
    with pytest.raises(ValidationError, match="must be provided together"):
        EvalTask(
            id="incomplete-provenance",
            question="Question?",
            source_dataset=DATASET_ID,
        )
    with pytest.raises(ValidationError, match="must contain text"):
        EvalTask(
            id="blank-provenance",
            question="Question?",
            source_dataset=" ",
            source_split="test",
            source_index=1,
        )


def test_frames_eligibility_rejects_only_explicit_local_artifact_dependencies() -> None:
    web_row = SourceRow(
        source_index=1,
        question="Which bay has a lighthouse with attached living quarters?",
        answer="Example Bay",
        reasoning_types="Multiple constraints",
        source_urls=("https://en.wikipedia.org/wiki/Example",),
    )
    local_row = SourceRow(
        source_index=2,
        question="What is shown in the attached image?",
        answer="Example",
        reasoning_types="Visual",
        source_urls=("https://en.wikipedia.org/wiki/Example",),
    )

    assert eligibility_failure(web_row) is None
    assert eligibility_failure(local_row) == "explicit_local_artifact_dependency"


def test_frames_selection_is_seeded_and_preserves_source_provenance() -> None:
    rows = [
        SourceRow(
            source_index=index,
            question=f"Question {index}?",
            answer=f"Answer {index}",
            reasoning_types="Test",
            source_urls=(f"https://en.wikipedia.org/wiki/Example_{index}",),
        )
        for index in range(10)
    ]

    first, rejected = select_tasks(rows, seed=17, limit=5)
    second, _ = select_tasks(rows, seed=17, limit=5)

    assert rejected == {}
    assert [task.source_index for task in first] == [7, 3, 0, 5, 1]
    assert first == second
    assert all(task.source_dataset == DATASET_ID for task in first)
    assert all(task.metadata["dataset_revision"] == DATASET_REVISION for task in first)


def test_frames_source_loader_fails_closed_on_content_drift(tmp_path: Path) -> None:
    source = tmp_path / "tiny.tsv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "Prompt",
                "Answer",
                "wikipedia_link_1",
                "reasoning_types",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "": "0",
                "Prompt": "Question?",
                "Answer": "Answer",
                "wikipedia_link_1": "https://en.wikipedia.org/wiki/Example",
                "reasoning_types": "Test",
            }
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    rows = load_official_rows(
        source,
        expected_sha256=digest,
        expected_byte_size=source.stat().st_size,
        expected_row_count=1,
    )

    assert rows[0].source_index == 0
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_official_rows(
            source,
            expected_sha256="0" * 64,
            expected_byte_size=source.stat().st_size,
            expected_row_count=1,
        )


def test_tracked_frames_pilot_matches_manifest_and_evaluation_schema() -> None:
    project = Path(__file__).resolve().parents[1]
    dataset = project / "evaluation/datasets/frames_pilot_seed17.jsonl"
    manifest_path = project / "evaluation/datasets/frames_pilot_seed17.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = load_jsonl_dataset(dataset, seed=0)

    assert manifest["source_sha256"] == f"sha256:{SOURCE_SHA256}"
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["dataset_revision"] == DATASET_REVISION
    assert manifest["dataset_url"] == DATASET_URL
    assert manifest["source_config"] == SOURCE_CONFIG
    assert manifest["source_split"] == SOURCE_SPLIT
    assert manifest["source_url"] == SOURCE_URL
    assert manifest["license"] == LICENSE
    assert manifest["total_source_rows"] == 824
    assert manifest["eligible_rows"] == 824
    assert manifest["rejected_rows"] == 0
    assert manifest["selection_seed"] == 17
    assert manifest["selected_source_indices"] == [664, 191, 123, 16, 718]
    assert loaded.total_tasks == 5
    assert sorted(task.source_index for task in loaded.selected_tasks) == sorted(
        [
            664,
            191,
            123,
            16,
            718,
        ]
    )
    assert manifest["selected_source_indices"] == [
        664,
        191,
        123,
        16,
        718,
    ]
    assert manifest["output_sha256"] == (
        f"sha256:{hashlib.sha256(dataset.read_bytes()).hexdigest()}"
    )
    assert manifest["output_sha256"] == (
        "sha256:e81379800805857fc04599c94f6174566e639d82eb0ccbd6b295ce72f328ee52"
    )


def test_manifest_hashes_exact_serialized_output() -> None:
    tasks, rejected = select_tasks(
        [
            SourceRow(
                source_index=index,
                question=f"Question {index}?",
                answer=f"Answer {index}",
                reasoning_types="Test",
                source_urls=("https://en.wikipedia.org/wiki/Example",),
            )
            for index in range(5)
        ],
        seed=17,
        limit=5,
    )
    payload = jsonl_bytes(tasks)

    manifest = build_manifest(
        tasks=tasks,
        rejected=rejected,
        output_bytes=payload,
        download_date="2026-07-19",
        total_rows=5,
    )

    assert manifest["output_sha256"] == (
        f"sha256:{hashlib.sha256(payload).hexdigest()}"
    )
