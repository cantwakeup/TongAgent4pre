"""Socket-free unit coverage for selective Answer-Revise policy semantics."""

from __future__ import annotations

from pathlib import Path

from evaluation.execution import resolve_system_config
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.schema import CompletionStatus, EvalTask
from evaluation.systems.answer_revise import (
    AtomicFact,
    CandidateAnswer,
    ShortAnswerAtomicFactAdapter,
    SupportCheck,
    reliability_metrics,
    revise_candidate,
    selective_finalize,
)
from evaluation.systems.permissive import AnswerExecution, DraftAnswer, DraftClaim
from evaluation.systems.tongagent import TongAgentRunner


def _facts() -> list[AtomicFact]:
    return [
        AtomicFact(
            fact_id="AF1",
            text="The Harbor Light Station was completed in 1901.",
            self_contained_text="The Harbor Light Station was completed in 1901.",
            critical_for_answer=True,
            source_ids=["S1"],
            subquestion_id="SQ1",
        ),
        AtomicFact(
            fact_id="AF2",
            text="The preservation file also lists 1901.",
            self_contained_text="The preservation file also lists 1901.",
            critical_for_answer=True,
            source_ids=["S2"],
            subquestion_id="SQ2",
        ),
    ]


def _execution() -> AnswerExecution:
    return AnswerExecution(
        status="success",
        answer_value=True,
        answer_text="Yes.",
        output_type="boolean",
        fact_ids=["F1", "F2"],
    )


def test_atomic_adapter_keeps_claim_source_and_self_containment() -> None:
    draft = DraftAnswer(
        claims=[
            DraftClaim(
                claim_id="D1",
                text="The Harbor Light Station was completed in 1901.",
                source_ids=["S1"],
                critical_for_final_answer=True,
                subquestion_id="SQ1",
            )
        ],
        proposed_answer="1901",
    )
    facts = ShortAnswerAtomicFactAdapter().extract(
        CandidateAnswer(answer_text="1901"), [], draft
    )
    assert facts[0].self_contained_text == draft.claims[0].text
    assert facts[0].source_ids == ["S1"]


def test_selective_policies_have_fixed_support_thresholds() -> None:
    facts = _facts()
    checks = [
        SupportCheck(
            fact_id="AF1",
            status="supported",
            source_ids=["S1"],
            explanation="quote",
            retrieval_quality="correct",
        ),
        SupportCheck(
            fact_id="AF2",
            status="undecidable",
            source_ids=["S2"],
            explanation="no quote",
            retrieval_quality="ambiguous",
        ),
    ]
    revised = revise_candidate(
        candidate=CandidateAnswer(answer_text="Yes.", source_ids=["S1", "S2"]),
        facts=facts,
        checks=checks,
        execution=_execution(),
    )
    assert (
        selective_finalize(
            policy="aggressive",
            revised=revised,
            facts=facts,
            checks=checks,
            execution=_execution(),
        ).answer_status
        == "answered"
    )
    assert (
        selective_finalize(
            policy="balanced",
            revised=revised,
            facts=facts,
            checks=checks,
            execution=_execution(),
        ).answer_status
        == "answered"
    )
    assert (
        selective_finalize(
            policy="conservative",
            revised=revised,
            facts=facts,
            checks=checks,
            execution=_execution(),
        ).answer_status
        == "abstain"
    )


def test_contradicted_critical_fact_removes_candidate_without_new_fact() -> None:
    facts = _facts()
    checks = [
        SupportCheck(
            fact_id="AF1",
            status="supported",
            source_ids=["S1"],
            explanation="quote",
            retrieval_quality="correct",
        ),
        SupportCheck(
            fact_id="AF2",
            status="contradicted",
            source_ids=["S2"],
            explanation="conflict",
            retrieval_quality="incorrect",
        ),
    ]
    revised = revise_candidate(
        candidate=CandidateAnswer(answer_text="Yes."),
        facts=facts,
        checks=checks,
        execution=_execution(),
    )
    assert revised.answer_text is None
    assert revised.removed_fact_ids == ["AF2"]


def test_reliability_reports_discrete_points_not_aurc() -> None:
    facts = _facts()
    checks = [
        SupportCheck(
            fact_id="AF1",
            status="supported",
            source_ids=["S1"],
            explanation="quote",
            retrieval_quality="correct",
        ),
        SupportCheck(
            fact_id="AF2",
            status="supported",
            source_ids=["S2"],
            explanation="quote",
            retrieval_quality="correct",
        ),
    ]
    revised = revise_candidate(
        candidate=CandidateAnswer(answer_text="Yes."),
        facts=facts,
        checks=checks,
        execution=_execution(),
    )
    finals = {
        name: selective_finalize(
            policy=name,
            revised=revised,
            facts=facts,
            checks=checks,
            execution=_execution(),
        )
        for name in ("aggressive", "balanced", "conservative")
    }
    metrics = reliability_metrics(
        finals=finals, facts=facts, checks=checks, raw_exact_match={"balanced": True}
    )
    assert metrics["balanced"]["coverage"] == 1.0
    assert metrics["balanced"]["risk"] == 0.0
    assert "aurc" not in metrics["balanced"]


def test_controlled_answer_revise_runs_research_to_three_policy_artifacts(
    tmp_path: Path,
) -> None:
    fixtures = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "fixtures"
        / "permissive_controlled"
    )
    backend = FixtureBackend.from_directory(fixtures)
    task = EvalTask(
        id="answer-revise-controlled",
        question="Where is Lunarite Mine located?",
        metadata={
            "structured_outputs": {
                "PermissivePlan": {
                    "objective": "Locate the mine.",
                    "subquestions": [
                        {
                            "id": "SQ1",
                            "question": "Where is Lunarite Mine located?",
                            "task_type": "single_fact_lookup",
                            "required_for_final_answer": True,
                        }
                    ],
                },
                "ResearchQueryDecision_SQ1_R1": {
                    "query": "Lunarite Mine location",
                    "needs_second_round": False,
                    "missing_fact": "",
                },
                "ResearchNote_SQ1": {
                    "subquestion_id": "SQ1",
                    "claim_candidates": ["Lunarite Mine is located in Alton Valley."],
                    "source_ids": ["S1"],
                    "supporting_passages": [
                        "The municipal mining archive records that Lunarite Mine is located in Alton Valley."
                    ],
                    "unresolved_points": [],
                    "confidence": "high",
                    "typed_facts": [
                        {
                            "fact_id": "F1",
                            "subquestion_id": "SQ1",
                            "fact_type": "entity",
                            "value": "Alton Valley",
                            "source_ids": ["S1"],
                            "claim_ids": [],
                            "verification_status": "partially_supported",
                            "raw_text": "Lunarite Mine is located in Alton Valley.",
                        }
                    ],
                },
                "DraftAnswer": {
                    "claims": [
                        {
                            "claim_id": "D1",
                            "text": "Lunarite Mine is located in Alton Valley.",
                            "source_ids": ["S1"],
                            "critical_for_final_answer": True,
                            "subquestion_id": "SQ1",
                        }
                    ],
                    "reasoning_steps": ["The archive gives the location."],
                    "proposed_answer": "Alton Valley.",
                    "confidence": "high",
                    "missing_information": [],
                },
                "CandidateAnswer": {
                    "answer_text": "Alton Valley.",
                    "rationale": "The archive names the valley.",
                    "source_ids": ["S1"],
                    "computation_trace_id": None,
                },
                "AnswerPlan": {
                    "operation": "direct_lookup",
                    "required_fact_ids": ["F1"],
                    "output_type": "entity",
                    "output_unit": None,
                    "parameters": {},
                },
            }
        },
    )
    config = resolve_system_config(
        "tongagent",
        "sha256:" + "a" * 64,
        17,
        tmp_path,
        fixture_directory="evaluation/fixtures/permissive_controlled",
        overrides={
            "fixture_revision": backend.revision,
            "runtime_mode": "answer_revise",
        },
    )
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, config)
    native = tmp_path / "native" / "tongagent"
    assert result.runtime_mode == "answer_revise"
    assert result.completion_status == CompletionStatus.COMPLETED
    assert (native / "candidate_answer.json").is_file()
    assert (native / "atomic_facts.json").is_file()
    assert (native / "support_checks.json").is_file()
    assert (native / "final_aggressive.json").is_file()
    assert (native / "final_balanced.json").is_file()
    assert (native / "final_conservative.json").is_file()
