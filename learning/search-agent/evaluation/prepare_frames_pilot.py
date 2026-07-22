"""Prepare the deterministic five-task FRAMES pipeline pilot.

The official source file is deliberately supplied by the caller.  This module
never falls back to an unpinned URL, and it refuses content whose SHA-256 does
not match the reviewed FRAMES revision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .schema import EvalTask


DATASET_ID = "google/frames-benchmark"
DATASET_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
SOURCE_CONFIG = "default"
SOURCE_SPLIT = "test"
SOURCE_FILENAME = "test.tsv"
SOURCE_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"
SOURCE_BYTE_SIZE = 484_887
SOURCE_ROW_COUNT = 824
LICENSE = "apache-2.0"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SOURCE_URL = f"{DATASET_URL}/resolve/{DATASET_REVISION}/{SOURCE_FILENAME}"
DEFAULT_SEED = 17
DEFAULT_LIMIT = 5
SELECTION_ALGORITHM = "python-random-v1-shuffle-eligible-then-first-n"
ELIGIBILITY_RULE_VERSION = 1

_LOCAL_ARTIFACT_DEPENDENCY = re.compile(
    r"(?i)(?:"
    r"\b(?:attached|provided|uploaded)\s+"
    r"(?:image|photo|picture|audio|video|file|document|chart|diagram)\b|"
    r"\b(?:image|photo|picture|audio|video|chart|diagram)\s+"
    r"(?:above|below|attached|provided|uploaded)\b|"
    r"\b(?:shown|pictured|depicted)\s+(?:above|below|here)\b|"
    r"\b(?:listen to|watch)\s+(?:the|this)\s+"
    r"(?:audio|video|recording|clip)\b"
    r")"
)


@dataclass(frozen=True)
class SourceRow:
    """One validated official FRAMES row and its stable source index."""

    source_index: int
    question: str
    answer: str
    reasoning_types: str
    source_urls: tuple[str, ...]


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_rows(
    source_path: Path,
    *,
    expected_sha256: str = SOURCE_SHA256,
    expected_byte_size: int = SOURCE_BYTE_SIZE,
    expected_row_count: int = SOURCE_ROW_COUNT,
) -> list[SourceRow]:
    """Load and validate every row from the pinned official TSV."""

    observed_hash = file_sha256(source_path)
    if observed_hash != expected_sha256:
        msg = (
            "FRAMES source SHA-256 mismatch: "
            f"expected={expected_sha256} observed={observed_hash}"
        )
        raise ValueError(msg)
    observed_size = source_path.stat().st_size
    if observed_size != expected_byte_size:
        msg = (
            "FRAMES source byte-size mismatch: "
            f"expected={expected_byte_size} observed={observed_size}"
        )
        raise ValueError(msg)

    rows: list[SourceRow] = []
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {"", "Prompt", "Answer", "reasoning_types"}
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"FRAMES source lacks required columns: {names}")
        for ordinal, raw in enumerate(reader):
            raw_index = str(raw.get("", "")).strip()
            if raw_index != str(ordinal):
                msg = (
                    "FRAMES source index is not contiguous: "
                    f"row={ordinal} value={raw_index!r}"
                )
                raise ValueError(msg)
            question = str(raw.get("Prompt", "")).strip()
            answer = str(raw.get("Answer", "")).strip()
            urls = tuple(
                str(raw.get(name, "")).strip()
                for name in reader.fieldnames or ()
                if name.startswith("wikipedia_link_")
                and str(raw.get(name, "")).strip().startswith(("http://", "https://"))
            )
            rows.append(
                SourceRow(
                    source_index=ordinal,
                    question=question,
                    answer=answer,
                    reasoning_types=str(raw.get("reasoning_types", "")).strip(),
                    source_urls=urls,
                )
            )
    if len(rows) != expected_row_count:
        msg = (
            "FRAMES source row-count mismatch: "
            f"expected={expected_row_count} observed={len(rows)}"
        )
        raise ValueError(msg)
    return rows


def eligibility_failure(row: SourceRow) -> str | None:
    """Return a deterministic exclusion reason, or null for an eligible row."""

    if not row.question:
        return "empty_question"
    if not row.answer:
        return "empty_reference_answer"
    if _LOCAL_ARTIFACT_DEPENDENCY.search(row.question):
        return "explicit_local_artifact_dependency"
    if not row.source_urls:
        return "no_http_wikipedia_provenance"
    return None


def select_tasks(
    rows: list[SourceRow],
    *,
    seed: int = DEFAULT_SEED,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[EvalTask], dict[str, int]]:
    """Apply eligibility to all rows, then shuffle deterministically."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    rejected: dict[str, int] = {}
    eligible: list[SourceRow] = []
    for row in rows:
        reason = eligibility_failure(row)
        if reason is None:
            eligible.append(row)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    if len(eligible) < limit:
        raise ValueError(
            f"only {len(eligible)} eligible FRAMES rows are available for {limit} tasks"
        )

    random.Random(seed).shuffle(eligible)
    selected = eligible[:limit]
    tasks = [
        EvalTask(
            id=f"frames-test-{row.source_index:04d}",
            question=row.question,
            reference_answer=row.answer,
            source_dataset=DATASET_ID,
            source_split=SOURCE_SPLIT,
            source_index=row.source_index,
            metadata={
                "benchmark_status": "pipeline_pilot_not_a_formal_benchmark",
                "dataset_revision": DATASET_REVISION,
                "license": LICENSE,
                "reasoning_types": row.reasoning_types,
                "source_file_sha256": f"sha256:{SOURCE_SHA256}",
                "source_row_id": str(row.source_index),
                "source_urls": list(row.source_urls),
            },
        )
        for row in selected
    ]
    return tasks, rejected


def jsonl_bytes(tasks: list[EvalTask]) -> bytes:
    """Serialize tasks as canonical UTF-8 JSONL."""

    lines = [
        json.dumps(
            task.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for task in tasks
    ]
    return ("\n".join(lines) + "\n").encode()


def build_manifest(
    *,
    tasks: list[EvalTask],
    rejected: dict[str, int],
    output_bytes: bytes,
    download_date: str,
    seed: int = DEFAULT_SEED,
    total_rows: int = SOURCE_ROW_COUNT,
) -> dict[str, Any]:
    """Build the complete, non-secret selection provenance manifest."""

    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_config": SOURCE_CONFIG,
        "source_split": SOURCE_SPLIT,
        "source_filename": SOURCE_FILENAME,
        "source_url": SOURCE_URL,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "download_date": download_date,
        "source_sha256": f"sha256:{SOURCE_SHA256}",
        "source_byte_size": SOURCE_BYTE_SIZE,
        "total_source_rows": total_rows,
        "eligibility_rule": {
            "version": ELIGIBILITY_RULE_VERSION,
            "requirements": [
                "non-empty text question",
                "non-empty reference answer",
                "no explicit dependency on an attached, uploaded, or provided local artifact",
                "at least one HTTP(S) Wikipedia provenance URL",
            ],
        },
        "eligible_rows": total_rows - sum(rejected.values()),
        "rejected_rows": sum(rejected.values()),
        "rejection_distribution": dict(sorted(rejected.items())),
        "selection_seed": seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selected_count": len(tasks),
        "selected_source_indices": [task.source_index for task in tasks],
        "selected_task_ids": [task.id for task in tasks],
        "output_sha256": f"sha256:{hashlib.sha256(output_bytes).hexdigest()}",
        "benchmark_status": "pipeline_pilot_not_a_formal_benchmark",
    }


def prepare(
    *,
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    download_date: str,
    seed: int = DEFAULT_SEED,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Validate the official source and atomically write selection artifacts."""

    date.fromisoformat(download_date)
    rows = load_official_rows(source_path)
    tasks, rejected = select_tasks(rows, seed=seed, limit=limit)
    output_bytes = jsonl_bytes(tasks)
    manifest = build_manifest(
        tasks=tasks,
        rejected=rejected,
        output_bytes=output_bytes,
        download_date=download_date,
        seed=seed,
        total_rows=len(rows),
    )
    _atomic_write_bytes(output_path, output_bytes)
    _atomic_write_bytes(
        manifest_path,
        (
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    return manifest


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the deterministic five-task FRAMES pipeline pilot."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/datasets/frames_pilot_seed17.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/datasets/frames_pilot_seed17.manifest.json"),
    )
    parser.add_argument("--download-date", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser


def main() -> int:
    """CLI entry point."""

    args = _parser().parse_args()
    manifest = prepare(
        source_path=args.source,
        output_path=args.output,
        manifest_path=args.manifest,
        download_date=args.download_date,
        seed=args.seed,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "dataset_revision": manifest["dataset_revision"],
                "eligible_rows": manifest["eligible_rows"],
                "selected_count": manifest["selected_count"],
                "selected_source_indices": manifest["selected_source_indices"],
                "output_sha256": manifest["output_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
