"""Offline, reproducible evidence-funnel analysis for a frozen experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .execution import atomic_write_json, atomic_write_text
from .schema import normalize_exact_match_text
from .systems.common import current_git_sha


_FILES = (
    "research_notes.json",
    "draft_answer.json",
    "verified_claims.json",
    "evidence_graph.json",
    "finalization_decision.json",
)


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _em(answer: str | None, reference: str | None) -> bool | None:
    if answer is None or reference is None:
        return None
    return normalize_exact_match_text(answer) == normalize_exact_match_text(reference)


def analyze(experiment: Path) -> dict[str, Any]:
    """Read only canonical prior artifacts, then emit derivable aggregates."""

    rows: list[dict[str, Any]] = []
    taxonomy: Counter[str] = Counter()
    for attempt in sorted(experiment.glob("tongagent/*/attempt-*/result.json")):
        root = attempt.parent
        task = _load(root / "task.json") or {}
        result = _load(attempt) or {}
        native = root / "native" / "tongagent"
        artifacts = {name: _load(native / name) for name in _FILES}
        missing = [name for name, value in artifacts.items() if value is None]
        notes = (artifacts["research_notes.json"] or {}).get("research_notes", [])
        bundles = (artifacts["research_notes.json"] or {}).get("research_bundles", {})
        draft = (artifacts["draft_answer.json"] or {}).get("draft", {})
        verified = (artifacts["verified_claims.json"] or {}).get("verified_claims", [])
        decision = artifacts["finalization_decision.json"] or {}
        counts = Counter(str(item.get("status", "unknown")) for item in verified)
        failures = [
            failure
            for item in verified
            for failure in item.get("registration_failures", [])
            if isinstance(failure, dict)
        ]
        taxonomy.update(str(item.get("category", "unknown")) for item in failures)
        candidate_passages = sum(
            len(item.get("supporting_passages", [])) for item in notes
        )
        fetches = sum(
            len(bundle.get("sources", []))
            for values in bundles.values()
            for bundle in values
        )
        mapped = sum(bool(item.get("source_ids")) for item in draft.get("claims", []))
        exact = sum(bool(item.get("exact_quotes")) for item in verified)
        proposed = draft.get("proposed_answer")
        final = result.get("extracted_answer")
        reference = task.get("reference_answer")
        draft_em = _em(
            proposed if isinstance(proposed, str) else None,
            reference if isinstance(reference, str) else None,
        )
        final_em = _em(
            final if isinstance(final, str) else None,
            reference if isinstance(reference, str) else None,
        )
        effect = (
            "correct_answer_removed"
            if draft_em is True and final is None
            else "wrong_answer_removed"
            if proposed and draft_em is False and final is None
            else "no_answer_to_verify"
            if not proposed
            else "preserved"
        )
        rows.append(
            {
                "task_id": result.get("task_id"),
                "artifact_root": str(root),
                "missing_artifacts": missing,
                "required_sq": len(
                    (artifacts["research_notes.json"] or {})
                    .get("plan", {})
                    .get("subquestions", [])
                ),
                "completed_sq": len(notes),
                "successful_fetches": fetches,
                "research_notes": len(notes),
                "claim_candidates": sum(
                    len(item.get("claim_candidates", [])) for item in notes
                ),
                "candidate_passages": candidate_passages,
                "draft_claims": len(draft.get("claims", [])),
                "source_mapped_claims": mapped,
                "exact_quote_claims": exact,
                "verified_claims": counts["verified"],
                "partially_supported_claims": counts["partially_supported"],
                "unsupported_claims": counts["unsupported"],
                "contested_claims": counts["contested"],
                "critical_claims": len(decision.get("critical_claims", [])),
                "critical_verified": len(decision.get("verified_critical_claims", [])),
                "draft_answer": proposed,
                "draft_normalized_exact_match": draft_em,
                "final_answer": final,
                "final_normalized_exact_match": final_em,
                "verifier_effect": effect,
                "finalization_decision": decision,
                "registration_failures": failures,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": experiment.name,
        "analysis_commit": current_git_sha(),
        "rows": rows,
        "claim_failure_taxonomy": dict(sorted(taxonomy.items())),
    }


def write_analysis(experiment: Path) -> dict[str, Any]:
    payload = analyze(experiment)
    analysis = experiment / "analysis"
    rows = payload["rows"]
    funnel = {"schema_version": 1, "experiment_id": experiment.name, "rows": rows}
    delta = {
        "schema_version": 1,
        "experiment_id": experiment.name,
        "rows": [
            {
                key: row[key]
                for key in (
                    "task_id",
                    "draft_answer",
                    "draft_normalized_exact_match",
                    "final_answer",
                    "final_normalized_exact_match",
                    "verifier_effect",
                )
            }
            for row in rows
        ],
    }
    atomic_write_json(analysis / "evidence_funnel.json", funnel, overwrite=True)
    atomic_write_json(analysis / "draft_vs_final.json", delta, overwrite=True)
    atomic_write_json(
        analysis / "claim_failure_taxonomy.json",
        {
            "schema_version": 1,
            "experiment_id": experiment.name,
            "counts": payload["claim_failure_taxonomy"],
        },
        overwrite=True,
    )
    lines = [
        "# Evidence funnel",
        "",
        "| Task | Fetch | Notes | Draft | Exact | Verified | Final effect |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines += [
        f"| {r['task_id']} | {r['successful_fetches']} | {r['research_notes']} | {r['draft_claims']} | {r['exact_quote_claims']} | {r['verified_claims']} | {r['verifier_effect']} |"
        for r in rows
    ]
    atomic_write_text(
        analysis / "evidence_funnel.md", "\n".join(lines) + "\n", overwrite=True
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
