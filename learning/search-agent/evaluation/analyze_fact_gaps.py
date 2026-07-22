"""Offline Fact-Gap analysis for a frozen permissive TongAgent experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .execution import atomic_write_json, atomic_write_text
from .fact_gap import fallback_slots, gaps_from_coverage, match_slots, repair_queries
from .systems.permissive import (
    DraftAnswer,
    ResearchNote,
    VerifiedClaim,
    _collect_typed_facts,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(experiment: Path) -> dict[str, Any]:
    """Read artifacts only; references are intentionally never accessed."""

    rows: list[dict[str, Any]] = []
    for result_path in sorted(experiment.glob("tongagent/*/attempt-*/result.json")):
        root = result_path.parent
        native = root / "native" / "tongagent"
        task = _read(root / "task.json")
        notes_data = _read(native / "research_notes.json")
        notes = [
            ResearchNote.model_validate(item) for item in notes_data["research_notes"]
        ]
        draft = DraftAnswer.model_validate(_read(native / "draft_answer.json")["draft"])
        verified = [
            VerifiedClaim.model_validate(item)
            for item in _read(native / "verified_claims.json")["verified_claims"]
        ]
        facts, _, fact_failures = _collect_typed_facts(
            notes=notes, draft=draft, verified=verified
        )
        slots = fallback_slots(
            question=str(task["question"]),
            subquestions=notes_data.get("plan", {}).get("subquestions", []),
        )
        coverage = match_slots(slots, facts)
        gaps = gaps_from_coverage(coverage)
        rows.append(
            {
                "task_id": task["id"],
                "required_fact_slots": [item.model_dump(mode="json") for item in slots],
                "typed_facts": [item.model_dump(mode="json") for item in facts],
                "fact_extraction_failures": fact_failures,
                "fact_coverage_report": [
                    item.model_dump(mode="json") for item in coverage
                ],
                "fact_gap_report": [item.model_dump(mode="json") for item in gaps],
                "repair_query_plan": [
                    query.model_dump(mode="json")
                    for gap in gaps
                    for query in repair_queries(
                        next(slot for slot in slots if slot.slot_id == gap.slot_id), gap
                    )
                ],
            }
        )
    return {"schema_version": 1, "experiment_id": experiment.name, "rows": rows}


def write_analysis(experiment: Path) -> dict[str, Any]:
    payload = analyze(experiment)
    output = experiment / "analysis" / "fact_gap"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "fact_gap_analysis.json", payload, overwrite=True)
    lines = [
        "# Fact-gap offline analysis",
        "",
        "| Task | Slots | Satisfied | Gaps | Main statuses |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        coverage = row["fact_coverage_report"]
        gaps = row["fact_gap_report"]
        statuses = ", ".join(sorted({item["status"] for item in coverage}))
        lines.append(
            f"| {row['task_id']} | {len(row['required_fact_slots'])} | "
            f"{sum(item['status'] == 'satisfied' for item in coverage)} | "
            f"{len(gaps)} | {statuses} |"
        )
    atomic_write_text(
        output / "fact_gap_analysis.md", "\n".join(lines) + "\n", overwrite=True
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(write_analysis(args.experiment), ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
