"""Offline-only selective Answer-Revise replay for frozen experiment artifacts.

The replay reads only native TongAgent notes/drafts/verifier/executor artifacts.
It performs no model calls, no provider calls, and no access to task source URLs.
Reference answers are loaded only after all policy outputs have been constructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import normalized_exact_match
from .systems.answer_revise import (
    AtomicFact,
    CandidateAnswer,
    RevisedAnswer,
    SupportCheck,
    reliability_metrics,
    selective_finalize,
)
from .systems.permissive import AnswerExecution


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _references(path: Path | None) -> dict[str, str | None]:
    """Read only id/reference_answer fields after policy construction."""

    if path is None:
        return {}
    values: dict[str, str | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        task_id = str(value.get("id", ""))
        if task_id:
            reference = value.get("reference_answer")
            values[task_id] = reference if isinstance(reference, str) else None
    return values


def replay_experiment(
    experiment: Path, *, dataset: Path | None = None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pending_scores: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for result_path in sorted(experiment.glob("tongagent/*/attempt-*/result.json")):
        result = _read(result_path, {})
        task_id = str(result.get("task_id", result_path.parents[1].name))
        native = result_path.parent / "native" / "tongagent"
        draft_payload = _read(native / "draft_answer.json", {})
        draft = dict(draft_payload.get("draft") or {})
        claims = list(draft.get("claims") or [])
        candidate = CandidateAnswer(
            answer_text=draft.get("proposed_answer"),
            rationale="Offline reconstruction from frozen DraftAnswer.",
            source_ids=sorted(
                {
                    str(source)
                    for claim in claims
                    for source in claim.get("source_ids", [])
                }
            ),
        )
        facts = [
            AtomicFact(
                fact_id=f"AF{index}",
                text=str(claim.get("text", "missing claim")),
                self_contained_text=str(claim.get("text", "missing claim")),
                critical_for_answer=bool(claim.get("critical_for_final_answer")),
                source_ids=[str(item) for item in claim.get("source_ids", [])],
                draft_claim_id=str(claim.get("claim_id", "")) or None,
                subquestion_id=str(claim.get("subquestion_id", "SQ1")),
            )
            for index, claim in enumerate(claims, start=1)
        ]
        verified = _read(native / "verified_claims.json", {}).get("verified_claims", [])
        status_by_claim = {
            str(item.get("claim_id", "")): str(item.get("status", "unsupported"))
            for item in verified
        }
        checks = []
        for fact in facts:
            source_status = status_by_claim.get(
                fact.draft_claim_id or "", "unsupported"
            )
            status = {
                "verified": "supported",
                "contested": "contradicted",
                "partially_supported": "undecidable",
                "unsupported": "irrelevant_evidence"
                if fact.source_ids
                else "undecidable",
            }.get(source_status, "undecidable")
            checks.append(
                SupportCheck(
                    fact_id=fact.fact_id,
                    status=status,
                    source_ids=fact.source_ids,
                    explanation="Offline mapping from frozen verifier status.",
                    retrieval_quality="correct"
                    if status == "supported"
                    else "incorrect"
                    if status in {"contradicted", "irrelevant_evidence"}
                    else "ambiguous",
                )
            )
        execution_payload = _read(native / "answer_execution.json", {}).get("execution")
        execution = (
            AnswerExecution.model_validate(execution_payload)
            if execution_payload
            else AnswerExecution(
                status="abstain", failure_reason="no frozen executor result"
            )
        )
        revision = RevisedAnswer(
            answer_text=execution.answer_text
            if execution.status == "success"
            else candidate.answer_text,
            retained_fact_ids=[
                item.fact_id
                for item in facts
                if next(
                    (check.status for check in checks if check.fact_id == item.fact_id),
                    "undecidable",
                )
                == "supported"
            ],
            unresolved_fact_ids=[
                item.fact_id
                for item in facts
                if next(
                    (check.status for check in checks if check.fact_id == item.fact_id),
                    "undecidable",
                )
                != "supported"
            ],
            source_ids=candidate.source_ids,
            computation_trace_id="frozen_calculation_trace"
            if execution.status == "success"
            else None,
        )
        finals = {
            name: selective_finalize(
                policy=name,
                revised=revision,
                facts=facts,
                checks=checks,
                execution=execution,
            )
            for name in ("aggressive", "balanced", "conservative")
        }
        row = {
            "task_id": task_id,
            "candidate": candidate.model_dump(mode="json"),
            "support_counts": {
                status: sum(check.status == status for check in checks)
                for status in (
                    "supported",
                    "contradicted",
                    "undecidable",
                    "irrelevant_evidence",
                )
            },
            "revised": revision.model_dump(mode="json"),
            "policies": {
                name: value.model_dump(mode="json") for name, value in finals.items()
            },
            "metrics": reliability_metrics(finals=finals, facts=facts, checks=checks),
        }
        rows.append(row)
        pending_scores.append((row, finals))
    # This is deliberately after every policy output exists.
    references = _references(dataset)
    for row, finals in pending_scores:
        reference = references.get(row["task_id"])
        for name, final in finals.items():
            row["policies"][name]["raw_exact_match"] = (
                normalized_exact_match(final.final_answer, reference)
                if reference
                else None
            )
    return {
        "schema_version": 1,
        "experiment_id": experiment.name,
        "offline": True,
        "rows": rows,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Answer-Revise offline replay: {payload['experiment_id']}",
        "",
        "| Task | Candidate | Supported/Contradicted/Undecidable | Aggressive | Balanced | Conservative |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        counts = row["support_counts"]
        policies = row["policies"]
        candidate = str((row["candidate"].get("answer_text") or "ABSTAIN"))[:80]
        lines.append(
            f"| {row['task_id']} | {candidate} | {counts['supported']}/{counts['contradicted']}/{counts['undecidable']} | {policies['aggressive']['answer_status']} | {policies['balanced']['answer_status']} | {policies['conservative']['answer_status']} |"
        )
    lines.extend(
        [
            "",
            "This replay is artifact-only: it makes no model/network call and reference answers are applied only after policy outputs are fixed.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args(argv)
    experiment = args.experiment.expanduser().resolve()
    payload = replay_experiment(experiment, dataset=args.dataset)
    analysis = experiment / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "answer_revise_replay.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (analysis / "answer_revise_replay.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    risk = {row["task_id"]: row["metrics"] for row in payload["rows"]}
    (analysis / "risk_coverage.json").write_text(
        json.dumps(
            {"schema_version": 1, "discrete_policy_points": risk, "aurc": None},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
