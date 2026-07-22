"""Offline replay of Typed Fact → plan → deterministic answer execution.

This module deliberately reads only frozen TongAgent artifacts.  It never
constructs a runtime, invokes a model, contacts a provider, or consults
``source_urls``.  Reference answers are inspected only after the replay answer
has been finalized, to report a post-hoc normalized exact-match field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .execution import atomic_write_json, atomic_write_text
from .schema import answer_for_exact_match, normalized_exact_match
from .systems.permissive import (
    AnswerPlan,
    DraftAnswer,
    ResearchNote,
    TypedFact,
    VerifiedClaim,
    _collect_typed_facts,
    _execute_answer_plan,
    _typed_finalize,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _offline_plan(question: str, facts: list[TypedFact]) -> AnswerPlan | None:
    """A reproducible plan inference for old artifacts without an AnswerPlan.

    It is intentionally narrow: a live workflow obtains the plan from the
    constrained structured-output call.  Replay only demonstrates whether
    archived, already-extracted facts are sufficient for deterministic work.
    """

    folded = question.casefold()
    years = [item for item in facts if item.fact_type in {"year", "date"}]
    if ("how many years" in folded or "years had passed" in folded) and len(years) >= 2:
        ordered = sorted(years[:2], key=lambda item: item.subquestion_id)
        return AnswerPlan(
            operation="date_difference",
            required_fact_ids=[item.fact_id for item in ordered],
            output_type="integer",
            output_unit="years",
        )
    if "difference" in folded and len(facts) >= 2:
        return AnswerPlan(
            operation="subtract",
            required_fact_ids=[facts[0].fact_id, facts[1].fact_id],
            output_type="number",
            output_unit=facts[0].unit,
        )
    entities = [item for item in facts if item.fact_type in {"entity", "string"}]
    if len(entities) == 1:
        return AnswerPlan(
            operation="direct_lookup",
            required_fact_ids=[entities[0].fact_id],
            output_type=entities[0].fact_type,
            output_unit=entities[0].unit,
        )
    return None


def replay(experiment: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    config = SimpleNamespace(
        permissive_workflow=SimpleNamespace(allow_low_confidence_answer=True)
    )
    for result_path in sorted(experiment.glob("tongagent/*/attempt-*/result.json")):
        root = result_path.parent
        native = root / "native" / "tongagent"
        task = _read(root / "task.json")
        notes_data = _read(native / "research_notes.json")
        draft_data = _read(native / "draft_answer.json")
        verified_data = _read(native / "verified_claims.json")
        notes = [
            ResearchNote.model_validate(item) for item in notes_data["research_notes"]
        ]
        draft = DraftAnswer.model_validate(draft_data["draft"])
        verified = [
            VerifiedClaim.model_validate(item)
            for item in verified_data["verified_claims"]
        ]
        facts, _updated, extraction_failures = _collect_typed_facts(
            notes=notes, draft=draft, verified=verified
        )
        plan = _offline_plan(str(task["question"]), facts)
        execution = _execute_answer_plan(plan=plan, facts=facts, allow_partial=True)
        required = [
            item["id"] for item in notes_data.get("plan", {}).get("subquestions", [])
        ]
        answer, status, decision = _typed_finalize(
            facts=facts,
            plan=plan,
            execution=execution,
            notes=notes,
            config=config,  # Runtime protocol only needs the documented switch.
            required_subquestion_ids=required,
        )
        # This is the first and only reference-answer use in this function.
        reference = task.get("reference_answer")
        rows.append(
            {
                "task_id": task["id"],
                "old_draft": draft.proposed_answer,
                "typed_facts": [item.model_dump(mode="json") for item in facts],
                "typed_fact_extraction_failures": extraction_failures,
                "answer_plan": plan.model_dump(mode="json") if plan else None,
                "answer_execution": execution.model_dump(mode="json"),
                "replay_answer": answer_for_exact_match(answer),
                "replay_answer_status": status,
                "replay_normalized_exact_match": normalized_exact_match(
                    answer, reference if isinstance(reference, str) else None
                ),
                "finalization_decision": decision,
            }
        )
    return {"schema_version": 1, "experiment_id": experiment.name, "rows": rows}


def write_replay(experiment: Path) -> dict[str, Any]:
    payload = replay(experiment)
    destination = experiment / "analysis" / "typed_synthesis_replay"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "replay.json", payload, overwrite=True)
    lines = [
        "# Typed synthesis offline replay",
        "",
        "| Task | Old Draft | Typed Facts | Operation | Replay Answer | Replay EM | Failure |",
        "|---|---|---:|---|---|---:|---|",
    ]
    for row in payload["rows"]:
        plan = row["answer_plan"] or {}
        execution = row["answer_execution"]
        lines.append(
            "| {task} | {draft} | {facts} | {operation} | {answer} | {em} | {failure} |".format(
                task=row["task_id"],
                draft=str(row["old_draft"] or "").replace("|", "\\|"),
                facts=len(row["typed_facts"]),
                operation=plan.get("operation", "none"),
                answer=str(row["replay_answer"] or "ABSTAIN").replace("|", "\\|"),
                em=row["replay_normalized_exact_match"],
                failure=execution.get("failure_reason", ""),
            )
        )
    atomic_write_text(
        destination / "replay.md", "\n".join(lines) + "\n", overwrite=True
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_replay(args.experiment), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
