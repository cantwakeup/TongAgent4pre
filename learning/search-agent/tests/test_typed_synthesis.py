"""Offline unit coverage for source-bound typed answer execution."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.replay_typed_synthesis import replay
from evaluation.systems.permissive import (
    AnswerPlan,
    TypedFact,
    _execute_answer_plan,
)


def _fact(
    fact_id: str,
    fact_type: str,
    value: object,
    *,
    unit: str | None = None,
    status: str = "verified",
) -> TypedFact:
    return TypedFact(
        fact_id=fact_id,
        subquestion_id="SQ1",
        fact_type=fact_type,  # type: ignore[arg-type]
        value=value,
        unit=unit,
        source_ids=["S1"],
        claim_ids=["C1"],
        verification_status=status,  # type: ignore[arg-type]
        raw_text="Canonical fetched source text.",
    )


def test_date_difference_is_deterministic_and_records_rounding() -> None:
    execution = _execute_answer_plan(
        plan=AnswerPlan(
            operation="date_difference",
            required_fact_ids=["F1", "F2"],
            output_type="integer",
            output_unit="years",
        ),
        facts=[_fact("F1", "date", "1787-12-12"), _fact("F2", "date", "1873-07-01")],
        allow_partial=False,
    )
    assert execution.status == "success"
    assert execution.answer_value == 85
    assert execution.calculation_trace[0]["rounding"] == "completed_anniversaries"


def test_numeric_subtraction_rejects_unit_conflicts() -> None:
    execution = _execute_answer_plan(
        plan=AnswerPlan(
            operation="subtract",
            required_fact_ids=["F1", "F2"],
            output_type="integer",
            output_unit="meters",
        ),
        facts=[
            _fact("F1", "integer", 10, unit="meters"),
            _fact("F2", "integer", 3, unit="feet"),
        ],
        allow_partial=False,
    )
    assert execution.status == "abstain"
    assert execution.failure_reason == "unit_conflict"


def test_filter_and_count_preserves_included_and_excluded_items() -> None:
    execution = _execute_answer_plan(
        plan=AnswerPlan(
            operation="filter_and_count",
            required_fact_ids=["F1", "F2", "F3"],
            output_type="count",
            output_unit=None,
            parameters={"field": "year", "gte_fact_id": "F2", "lte_fact_id": "F3"},
        ),
        facts=[
            _fact(
                "F1",
                "list",
                [
                    {"name": "A", "year": 1984},
                    {"name": "B", "year": 1991},
                    {"name": "C", "year": 2000},
                ],
            ),
            _fact("F2", "year", 1985, unit="year"),
            _fact("F3", "year", 1995, unit="year"),
        ],
        allow_partial=False,
    )
    assert execution.status == "success"
    assert execution.answer_value == 1
    assert execution.calculation_trace[0]["included"] == [{"name": "B", "year": 1991}]
    assert len(execution.calculation_trace[0]["excluded"]) == 2


def test_executor_rejects_missing_and_unsupported_facts() -> None:
    missing = _execute_answer_plan(
        plan=AnswerPlan(
            operation="direct_lookup",
            required_fact_ids=["missing"],
            output_type="string",
        ),
        facts=[],
        allow_partial=False,
    )
    unsupported = _execute_answer_plan(
        plan=AnswerPlan(
            operation="direct_lookup", required_fact_ids=["F1"], output_type="string"
        ),
        facts=[_fact("F1", "string", "value", status="unsupported")],
        allow_partial=True,
    )
    assert missing.failure_reason == "missing_required_typed_fact"
    assert unsupported.failure_reason == "unsupported_or_contested_typed_fact"


def test_partially_supported_fact_is_configurable() -> None:
    plan = AnswerPlan(
        operation="direct_lookup", required_fact_ids=["F1"], output_type="string"
    )
    facts = [_fact("F1", "string", "source-bound", status="partially_supported")]
    assert (
        _execute_answer_plan(plan=plan, facts=facts, allow_partial=False).status
        == "abstain"
    )
    assert (
        _execute_answer_plan(plan=plan, facts=facts, allow_partial=True).answer_text
        == "source-bound"
    )


def test_frozen_artifact_replay_is_reproducible_without_runtime(tmp_path: Path) -> None:
    """The replay consumes canonical JSON only; it has no model/provider path."""

    root = tmp_path / "tongagent" / "controlled" / "attempt-0001"
    native = root / "native" / "tongagent"
    native.mkdir(parents=True)
    (root / "task.json").write_text(
        json.dumps(
            {
                "id": "controlled",
                "question": "How many years had passed between the two events?",
                "reference_answer": "85",
            }
        ),
        encoding="utf-8",
    )
    (root / "result.json").write_text("{}", encoding="utf-8")
    (native / "research_notes.json").write_text(
        json.dumps(
            {
                "plan": {"subquestions": [{"id": "SQ1"}, {"id": "SQ2"}]},
                "research_notes": [
                    {
                        "subquestion_id": "SQ1",
                        "source_ids": ["S1"],
                        "typed_facts": [
                            _fact("F1", "year", 1788, unit="year").model_dump()
                        ],
                    },
                    {
                        "subquestion_id": "SQ2",
                        "source_ids": ["S2"],
                        "typed_facts": [
                            _fact("F2", "year", 1873, unit="year").model_dump()
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (native / "draft_answer.json").write_text(
        json.dumps(
            {
                "draft": {
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "The first event occurred in 1788.",
                            "source_ids": ["S1"],
                            "critical_for_final_answer": True,
                            "subquestion_id": "SQ1",
                        },
                        {
                            "claim_id": "C2",
                            "text": "The second event occurred in 1873.",
                            "source_ids": ["S2"],
                            "critical_for_final_answer": True,
                            "subquestion_id": "SQ2",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (native / "verified_claims.json").write_text(
        json.dumps(
            {
                "verified_claims": [
                    {
                        "claim_id": "C1",
                        "status": "verified",
                        "source_ids": ["S1"],
                        "exact_quotes": ["The first event occurred in 1788."],
                        "explanation": "Fixture quote.",
                    },
                    {
                        "claim_id": "C2",
                        "status": "verified",
                        "source_ids": ["S2"],
                        "exact_quotes": ["The second event occurred in 1873."],
                        "explanation": "Fixture quote.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    first = replay(tmp_path)
    second = replay(tmp_path)
    assert first == second
    assert first["rows"][0]["replay_answer"] == "85 years"
    assert first["rows"][0]["replay_normalized_exact_match"] is False
