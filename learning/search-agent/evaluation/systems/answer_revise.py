"""Selective candidate-answer revision workflow for TongAgent.

This is intentionally separate from the strict FSM and the optional Fact-Gap
path.  It reuses the same bounded retrieval runtime, source ledger, canonical
page cache, and EvidenceGraphStore, then applies fixed post-hoc policies to one
research trace.  No policy triggers additional searching.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from evidence_graph import validate_evidence_graph

from ..budget import ExecutionBudget
from ..config import ResolvedConfig
from ..offline import FixtureBackend
from ..schema import CompletionStatus, EvalTask, RunResult
from ..tracing import TraceCollector
from .common import (
    PreparedRuntime,
    build_run_result,
    prepare_runtime,
    write_intermediate_artifacts,
)
from .permissive import (
    AnswerExecution,
    AnswerPlan,
    DraftAnswer,
    DraftClaim,
    PermissivePlan,
    ResearchBundle,
    ResearchNote,
    ResearchSource,
    TypedFact,
    VerifiedClaim,
    _accounted_structured,
    _answer_plan,
    _citations,
    _collect_typed_facts,
    _draft,
    _execute_answer_plan,
    _note,
    _plan,
    _query_decision,
    _verify,
    _write_json,
    research_query,
)


SupportStatus = Literal[
    "supported", "contradicted", "undecidable", "irrelevant_evidence"
]
PolicyName = Literal["aggressive", "balanced", "conservative"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CandidateAnswer(_StrictModel):
    answer_text: str | None = None
    rationale: str = ""
    source_ids: list[str] = Field(default_factory=list)
    computation_trace_id: str | None = None


class AtomicFact(_StrictModel):
    fact_id: str
    text: str = Field(min_length=1)
    self_contained_text: str = Field(min_length=1)
    critical_for_answer: bool
    source_ids: list[str] = Field(default_factory=list)
    draft_claim_id: str | None = None
    subquestion_id: str = "SQ1"


class AtomicFactList(_StrictModel):
    facts: list[AtomicFact] = Field(default_factory=list)


class SupportCheck(_StrictModel):
    fact_id: str
    status: SupportStatus
    source_ids: list[str] = Field(default_factory=list)
    exact_quotes: list[str] = Field(default_factory=list)
    explanation: str
    canonical_claim_id: str | None = None
    retrieval_quality: Literal["correct", "ambiguous", "incorrect"]


class RevisedAnswer(_StrictModel):
    answer_text: str | None = None
    retained_fact_ids: list[str] = Field(default_factory=list)
    revised_fact_ids: list[str] = Field(default_factory=list)
    removed_fact_ids: list[str] = Field(default_factory=list)
    unresolved_fact_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    computation_trace_id: str | None = None


class SelectiveFinal(_StrictModel):
    policy: PolicyName
    answer_status: Literal["answered", "abstain"]
    confidence: Literal["high", "medium", "low", "none"]
    answer_text: str | None = None
    final_answer: str
    reason: str
    critical_fact_ids: list[str] = Field(default_factory=list)
    support_statuses: dict[str, SupportStatus] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    calculation_executable: bool


class AtomicFactExtractor(Protocol):
    """Adapter boundary compatible with FActScore/SAFE-style decomposition."""

    def extract(
        self,
        candidate_answer: CandidateAnswer,
        research_notes: list[ResearchNote],
        draft: DraftAnswer,
    ) -> list[AtomicFact]: ...


class ShortAnswerAtomicFactAdapter:
    """Deterministic, source-bound short-answer decomposition adapter.

    FActScore's atomic-fact notion and SAFE's self-contained fact contract are
    adopted at the interface level.  This implementation is local and does not
    vendor either project; each existing draft claim becomes one auditable fact.
    """

    def extract(
        self,
        candidate_answer: CandidateAnswer,
        research_notes: list[ResearchNote],
        draft: DraftAnswer,
    ) -> list[AtomicFact]:
        del candidate_answer, research_notes
        return [
            AtomicFact(
                fact_id=f"AF{index}",
                text=claim.text,
                self_contained_text=claim.text,
                critical_for_answer=claim.critical_for_final_answer,
                source_ids=list(dict.fromkeys(claim.source_ids)),
                draft_claim_id=claim.claim_id,
                subquestion_id=claim.subquestion_id,
            )
            for index, claim in enumerate(draft.claims, start=1)
        ]


def _candidate_answer(
    *,
    runtime: PreparedRuntime,
    task: EvalTask,
    draft: DraftAnswer,
    notes: Sequence[ResearchNote],
) -> CandidateAnswer:
    """Ask once for a source-mapped candidate; fall back to the existing draft."""

    allowed = {source for note in notes for source in note.source_ids}
    response = _accounted_structured(
        runtime=runtime,
        schema=CandidateAnswer,
        prompt=(
            "State one candidate answer using only the source-grounded draft and "
            "research notes. Do not add facts, URLs, or sources. If a key input is "
            "missing, set answer_text to null.\n\n"
            f"Question: {task.question}\nDraft: "
            + json.dumps(draft.model_dump(mode="json"), ensure_ascii=False)
        )[:14_000],
        label="tongagent.answer_revise.candidate",
        stage="final_synthesis",
    )
    if response is None:
        return CandidateAnswer(
            answer_text=draft.proposed_answer,
            rationale="Candidate reconstructed from the source-grounded draft.",
            source_ids=sorted(allowed),
        )
    candidate = CandidateAnswer.model_validate(response)
    source_ids = [source for source in candidate.source_ids if source in allowed]
    if candidate.answer_text and not source_ids:
        return CandidateAnswer(
            answer_text=draft.proposed_answer,
            rationale="Candidate source mapping was invalid; retained draft fallback.",
            source_ids=sorted(allowed),
        )
    return candidate.model_copy(update={"source_ids": list(dict.fromkeys(source_ids))})


def _support_checks(
    *,
    runtime: PreparedRuntime,
    facts: Sequence[AtomicFact],
    sources: Mapping[str, ResearchSource],
) -> tuple[list[SupportCheck], list[VerifiedClaim]]:
    """Classify source-bound atomic facts and register literal evidence post-hoc."""

    draft = DraftAnswer(
        claims=[
            DraftClaim(
                claim_id=fact.fact_id,
                text=fact.self_contained_text,
                source_ids=fact.source_ids,
                critical_for_final_answer=fact.critical_for_answer,
                subquestion_id=fact.subquestion_id,
            )
            for fact in facts
        ],
        proposed_answer=None,
    )
    verified = _verify(runtime=runtime, draft=draft, sources=sources)
    by_id = {item.claim_id: item for item in verified}
    checks: list[SupportCheck] = []
    for fact in facts:
        item = by_id.get(fact.fact_id)
        if item is None:
            checks.append(
                SupportCheck(
                    fact_id=fact.fact_id,
                    status="undecidable",
                    explanation="No post-hoc support classification was produced.",
                    retrieval_quality="ambiguous",
                )
            )
            continue
        status: SupportStatus = {
            "verified": "supported",
            "contested": "contradicted",
            "partially_supported": "undecidable",
            "unsupported": "irrelevant_evidence" if fact.source_ids else "undecidable",
        }[item.status]
        checks.append(
            SupportCheck(
                fact_id=fact.fact_id,
                status=status,
                source_ids=item.source_ids,
                exact_quotes=item.exact_quotes,
                explanation=item.explanation,
                canonical_claim_id=item.canonical_claim_id,
                retrieval_quality=(
                    "correct"
                    if status == "supported"
                    else "incorrect"
                    if status in {"contradicted", "irrelevant_evidence"}
                    else "ambiguous"
                ),
            )
        )
    return checks, verified


def revise_candidate(
    *,
    candidate: CandidateAnswer,
    facts: Sequence[AtomicFact],
    checks: Sequence[SupportCheck],
    execution: AnswerExecution,
) -> RevisedAnswer:
    """Apply a constrained RARR-style revision without introducing new facts."""

    by_fact = {item.fact_id: item for item in facts}
    by_check = {item.fact_id: item for item in checks}
    contradicted_critical = [
        fact.fact_id
        for fact in facts
        if fact.critical_for_answer
        and by_check.get(fact.fact_id)
        and by_check[fact.fact_id].status == "contradicted"
    ]
    retained = [
        fact.fact_id
        for fact in facts
        if by_check.get(fact.fact_id) and by_check[fact.fact_id].status == "supported"
    ]
    unresolved = [
        fact.fact_id
        for fact in facts
        if by_check.get(fact.fact_id)
        and by_check[fact.fact_id].status in {"undecidable", "irrelevant_evidence"}
    ]
    # A deterministic executor is the only legal way to replace a numerical
    # candidate.  Otherwise retain source-grounded candidate text unchanged.
    answer = (
        execution.answer_text
        if execution.status == "success"
        else candidate.answer_text
    )
    if contradicted_critical:
        answer = None
    source_ids = sorted(
        {source for fact_id in retained for source in by_fact[fact_id].source_ids}
    )
    return RevisedAnswer(
        answer_text=answer,
        retained_fact_ids=retained,
        revised_fact_ids=(
            retained
            if execution.status == "success" and candidate.answer_text != answer
            else []
        ),
        removed_fact_ids=contradicted_critical,
        unresolved_fact_ids=unresolved,
        source_ids=source_ids,
        computation_trace_id="calculation_trace"
        if execution.status == "success"
        else None,
    )


def selective_finalize(
    *,
    policy: PolicyName,
    revised: RevisedAnswer,
    facts: Sequence[AtomicFact],
    checks: Sequence[SupportCheck],
    execution: AnswerExecution,
) -> SelectiveFinal:
    """Produce one fixed aggressive/balanced/conservative release decision."""

    checks_by_id = {item.fact_id: item for item in checks}
    critical = [item for item in facts if item.critical_for_answer]
    statuses = {
        item.fact_id: checks_by_id.get(
            item.fact_id,
            SupportCheck(
                fact_id=item.fact_id,
                status="undecidable",
                explanation="missing check",
                retrieval_quality="ambiguous",
            ),
        ).status
        for item in critical
    }
    executable = execution.status == "success"
    nonempty = bool((revised.answer_text or "").strip())
    contradicted = any(value == "contradicted" for value in statuses.values())
    irrelevant = any(value == "irrelevant_evidence" for value in statuses.values())
    supported = any(value == "supported" for value in statuses.values())
    canonical = all(
        checks_by_id.get(item.fact_id) is not None
        and bool(checks_by_id[item.fact_id].source_ids)
        for item in critical
    )
    if policy == "aggressive":
        allowed = nonempty and executable and not contradicted
        reason = "nonempty executable answer without contradicted critical fact"
    elif policy == "balanced":
        allowed = (
            nonempty
            and executable
            and canonical
            and not contradicted
            and not irrelevant
            and supported
        )
        reason = (
            "all critical facts have canonical passages and at least one is supported"
        )
    else:
        allowed = (
            nonempty
            and executable
            and bool(critical)
            and all(value == "supported" for value in statuses.values())
        )
        reason = "all critical facts are supported and calculation is executable"
    answer = (revised.answer_text or "").strip() if allowed else None
    final_answer = f"FINAL_ANSWER: {answer}" if answer else "FINAL_ANSWER: ABSTAIN"
    return SelectiveFinal(
        policy=policy,
        answer_status="answered" if answer else "abstain",
        confidence=(
            "high"
            if policy == "conservative"
            else "medium"
            if policy == "balanced"
            else "low"
        )
        if answer
        else "none",
        answer_text=answer,
        final_answer=final_answer,
        reason=reason if answer else f"policy declined release: {reason}",
        critical_fact_ids=[item.fact_id for item in critical],
        support_statuses=statuses,
        source_ids=revised.source_ids,
        calculation_executable=executable,
    )


def reliability_metrics(
    *,
    finals: Mapping[str, SelectiveFinal],
    facts: Sequence[AtomicFact],
    checks: Sequence[SupportCheck],
    raw_exact_match: Mapping[str, bool | None] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Return discrete risk/coverage points; no continuous AURC is fabricated."""

    by_id = {item.fact_id: item for item in checks}
    critical = [item for item in facts if item.critical_for_answer]
    supported = [item for item in checks if item.status == "supported"]
    rows: dict[str, dict[str, float | None]] = {}
    for name, final in finals.items():
        answered = final.answer_status == "answered"
        em = (raw_exact_match or {}).get(name)
        accuracy = float(em) if answered and em is not None else None
        rows[name] = {
            "coverage": 1.0 if answered else 0.0,
            "selective_accuracy": accuracy,
            "risk": 1.0 - accuracy if accuracy is not None else None,
            "atomic_fact_support_rate": len(supported) / len(checks) if checks else 0.0,
            "critical_fact_support_rate": sum(
                by_id.get(item.fact_id) is not None
                and by_id[item.fact_id].status == "supported"
                for item in critical
            )
            / len(critical)
            if critical
            else 0.0,
            "unsupported_answer_rate": float(
                answered
                and any(
                    value in {"undecidable", "irrelevant_evidence"}
                    for value in final.support_statuses.values()
                )
            ),
            "contradicted_answer_rate": float(
                answered
                and any(
                    value == "contradicted" for value in final.support_statuses.values()
                )
            ),
            "citation_precision": (
                len(
                    {
                        source
                        for item in checks
                        if item.status == "supported"
                        for source in item.source_ids
                    }
                )
                / len(final.source_ids)
                if final.source_ids
                else None
            ),
            "answer_rate": 1.0 if answered else 0.0,
            "raw_exact_match": float(em) if em is not None else None,
        }
    return rows


def _answer_plan_and_execution(
    *,
    runtime: PreparedRuntime,
    task: EvalTask,
    notes: Sequence[ResearchNote],
    draft: DraftAnswer,
    verified: Sequence[VerifiedClaim],
) -> tuple[list[TypedFact], AnswerPlan | None, AnswerExecution, list[dict[str, str]]]:
    typed, _, failures = _collect_typed_facts(
        notes=notes, draft=draft, verified=verified
    )
    plan = _answer_plan(runtime=runtime, task=task, facts=typed)
    execution = (
        _execute_answer_plan(plan=plan, facts=typed, allow_partial=True)
        if plan is not None
        else AnswerExecution(status="abstain", failure_reason="missing_answer_plan")
    )
    return typed, plan, execution, failures


def run_answer_revise_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> RunResult:
    """Run RESEARCH → CANDIDATE → ATOMIC → SUPPORT → REVISION → SELECTIVE."""

    artifact_directory = Path(resolved_config.artifact_directory).expanduser()
    native = artifact_directory / "native" / "tongagent"
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    trace = TraceCollector()
    budget = ExecutionBudget(resolved_config.budget)
    run_id = f"tongagent-{task.id}-{uuid.uuid4().hex}"
    runtime: PreparedRuntime | None = None
    caught: Exception | None = None
    plan: PermissivePlan | None = None
    notes: list[ResearchNote] = []
    bundles: dict[str, list[ResearchBundle]] = {}
    sources: dict[str, ResearchSource] = {}
    draft = DraftAnswer()
    candidate = CandidateAnswer()
    facts: list[AtomicFact] = []
    checks: list[SupportCheck] = []
    verified: list[VerifiedClaim] = []
    typed: list[TypedFact] = []
    answer_plan: AnswerPlan | None = None
    execution = AnswerExecution(
        status="abstain", failure_reason="workflow_not_completed"
    )
    typed_failures: list[dict[str, str]] = []
    revision = RevisedAnswer()
    finals: dict[str, SelectiveFinal] = {}
    try:
        runtime = prepare_runtime(
            task,
            resolved_config,
            system_id="tongagent",
            execution_budget=budget,
            trace=trace,
            injected_backend=fixture_backend,
            injected_model=model,
            semantic_strategy="fixed",
            enable_tongagent_evidence_state=True,
            enable_tongagent_token_control=True,
            enable_tongagent_context_compaction=True,
            reserve_final_synthesis=False,
        )
        trace.record(
            "answer_revise_phase", phase="RESEARCH", runtime_mode="answer_revise"
        )
        plan, _ = _plan(runtime=runtime, task=task)
        runtime.research_budget.configure_subquestions(
            [item.id for item in plan.subquestions]
        )
        runtime.middleware.configure_token_partitions(
            [item.id for item in plan.subquestions]
        )
        for sq in plan.subquestions:
            runtime.research_budget.activate_subquestion(sq.id)
            runtime.middleware.activate_token_subquestion(sq.id)
            local: list[ResearchBundle] = []
            for round_number in range(1, 3):
                decision = _query_decision(
                    runtime=runtime,
                    subquestion=sq,
                    round_number=round_number,
                    previous=local[-1] if local else None,
                )
                bundle = research_query(
                    query=decision.query,
                    task_type=sq.task_type,
                    subquestion=sq,
                    search_tool=next(
                        tool for tool in runtime.tools if tool.name == "web_search"
                    ),
                    fetch_tool=next(
                        tool for tool in runtime.tools if tool.name == "fetch_url"
                    ),
                    max_sources=3,
                )
                local.append(bundle)
                sources.update({item.source_id: item for item in bundle.sources})
                trace.record(
                    "answer_revise_research_bundle",
                    subquestion_id=sq.id,
                    round=round_number,
                    source_count=len(bundle.sources),
                    broadened=bundle.broadened,
                )
                if bundle.sources and not decision.needs_second_round:
                    break
            bundles[sq.id] = local
            notes.append(_note(runtime=runtime, subquestion=sq, bundles=local))
        trace.record("answer_revise_phase", phase="CANDIDATE_ANSWER")
        draft = _draft(runtime=runtime, task=task, notes=notes)
        candidate = _candidate_answer(
            runtime=runtime, task=task, draft=draft, notes=notes
        )
        trace.record("answer_revise_phase", phase="ATOMIC_FACTS")
        facts = ShortAnswerAtomicFactAdapter().extract(candidate, notes, draft)
        trace.record("answer_revise_phase", phase="SUPPORT_CHECK")
        checks, verified = _support_checks(
            runtime=runtime, facts=facts, sources=sources
        )
        draft_claim_by_fact = {
            fact.fact_id: fact.draft_claim_id or fact.fact_id for fact in facts
        }
        verified_for_typed_facts = [
            item.model_copy(
                update={
                    "claim_id": draft_claim_by_fact.get(item.claim_id, item.claim_id)
                }
            )
            for item in verified
        ]
        typed, answer_plan, execution, typed_failures = _answer_plan_and_execution(
            runtime=runtime,
            task=task,
            notes=notes,
            draft=draft,
            verified=verified_for_typed_facts,
        )
        trace.record("answer_revise_phase", phase="REVISION")
        revision = revise_candidate(
            candidate=candidate, facts=facts, checks=checks, execution=execution
        )
        trace.record("answer_revise_phase", phase="SELECTIVE_FINALIZATION")
        finals = {
            name: selective_finalize(
                policy=cast(PolicyName, name),
                revised=revision,
                facts=facts,
                checks=checks,
                execution=execution,
            )
            for name in ("aggressive", "balanced", "conservative")
        }
    except Exception as exc:
        caught = exc
        trace.record(
            "run_exception", phase="answer_revise", exception_type=type(exc).__name__
        )
        typed_failures = []
    finally:
        default = finals.get("balanced") or SelectiveFinal(
            policy="balanced",
            answer_status="abstain",
            confidence="none",
            final_answer="FINAL_ANSWER: ABSTAIN",
            reason="workflow did not reach selective finalization",
            calculation_executable=False,
        )
        final_answer = default.final_answer
        finished_at = datetime.now(UTC)
        ledger = runtime.research_budget.snapshot() if runtime is not None else {}
        graph_errors = validate_evidence_graph(ledger) if runtime is not None else []
        checks_by_status = {
            status: sum(item.status == status for item in checks)
            for status in (
                "supported",
                "contradicted",
                "undecidable",
                "irrelevant_evidence",
            )
        }
        metrics = reliability_metrics(finals=finals, facts=facts, checks=checks)
        workflow_metrics: dict[str, Any] = {
            "runtime_mode": "answer_revise",
            "sq_research_completion_rate": (
                sum(
                    bool(note.source_ids) or bool(note.unresolved_points)
                    for note in notes
                )
                / len(plan.subquestions)
            )
            if plan and plan.subquestions
            else 0.0,
            "draft_claim_count": len(draft.claims),
            "verified_claim_rate": checks_by_status["supported"] / len(checks)
            if checks
            else 0.0,
            "critical_claim_verified_rate": (
                sum(
                    item.status == "supported"
                    and any(
                        f.fact_id == item.fact_id and f.critical_for_answer
                        for f in facts
                    )
                    for item in checks
                )
                / sum(item.critical_for_answer for item in facts)
            )
            if any(item.critical_for_answer for item in facts)
            else 0.0,
            "unsupported_claim_rate": checks_by_status["undecidable"] / len(checks)
            if checks
            else 0.0,
            "candidate_answer": candidate.answer_text,
            "answer_before_verification": candidate.answer_text,
            "answer_after_verification": final_answer,
            "atomic_fact_count": len(facts),
            "support_status_counts": checks_by_status,
            "research_rounds_per_sq": {
                key: len(value) for key, value in bundles.items()
            },
            "selective_reliability": metrics,
            "answer_policy": "balanced",
            "answer_execution_status": execution.status,
            "typed_fact_count": len(typed),
            "typed_fact_failures": typed_failures,
            "evidence_graph_errors": graph_errors,
        }
        result = build_run_result(
            run_id=run_id,
            task=task,
            resolved_config=resolved_config,
            system_id="tongagent",
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=max(0.0, time.perf_counter() - started),
            runtime=runtime,
            execution_budget=budget,
            trace=trace,
            native_output={"final_answer": final_answer},
            caught=caught,
            evidence_count=len(ledger.get("evidence_units", [])) if runtime else None,
            structural_subquestion_coverage=workflow_metrics[
                "sq_research_completion_rate"
            ]
            if plan
            else None,
            final_answer_override=final_answer,
        )
        scored_metrics = reliability_metrics(
            finals=finals,
            facts=facts,
            checks=checks,
            raw_exact_match={"balanced": result.normalized_exact_match},
        )
        workflow_metrics["selective_reliability"] = scored_metrics
        workflow_metrics.update(
            {
                f"balanced_{key}": value
                for key, value in scored_metrics.get("balanced", {}).items()
            }
        )
        if caught is None and result.completion_status == CompletionStatus.COMPLETED:
            result = result.model_copy(
                update={
                    "completion_status": CompletionStatus.COMPLETED
                    if default.answer_status == "answered"
                    else CompletionStatus.PARTIAL,
                    "citations": _citations(runtime) if runtime else [],
                    "workflow_metrics": cast("dict[str, Any]", workflow_metrics),
                }
            )
        write_intermediate_artifacts(
            artifact_directory,
            task=task,
            resolved_config=resolved_config,
            runtime=runtime,
            execution_budget=budget,
            trace=trace,
            native_output={"final_answer": final_answer},
            result=result,
            caught=caught,
        )
        _write_json(
            native / "research_notes.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "plan": plan.model_dump(mode="json") if plan else None,
                "research_notes": [item.model_dump(mode="json") for item in notes],
                "research_bundles": {
                    key: [item.model_dump(mode="json") for item in value]
                    for key, value in bundles.items()
                },
            },
        )
        _write_json(
            native / "draft_answer.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "draft": draft.model_dump(mode="json"),
            },
        )
        _write_json(
            native / "candidate_answer.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "candidate": candidate.model_dump(mode="json"),
            },
        )
        _write_json(
            native / "atomic_facts.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "facts": [item.model_dump(mode="json") for item in facts],
                "extractor": "short_answer_factscore_safe_compatible",
            },
        )
        _write_json(
            native / "support_checks.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "checks": [item.model_dump(mode="json") for item in checks],
                "verified_claims": [item.model_dump(mode="json") for item in verified],
            },
        )
        _write_json(
            native / "revision.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "revision": revision.model_dump(mode="json"),
            },
        )
        for name, final in finals.items():
            _write_json(
                native / f"final_{name}.json",
                {
                    "schema_version": 1,
                    "task_id": task.id,
                    "final": final.model_dump(mode="json"),
                },
            )
        _write_json(
            native / "evidence_graph.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "claims": ledger.get("claims", []),
                "evidence_units": ledger.get("evidence_units", []),
                "conflicts": ledger.get("conflicts", []),
                "state_integrity_errors": graph_errors,
            },
        )
        _write_json(
            native / "answer_revise_workflow.json",
            {
                "schema_version": 1,
                "runtime_mode": "answer_revise",
                "phase": "SELECTIVE_FINALIZATION",
                "workflow_metrics": workflow_metrics,
                "canonical_artifacts": {
                    "candidate": "candidate_answer.json",
                    "atomic_facts": "atomic_facts.json",
                    "support": "support_checks.json",
                    "revision": "revision.json",
                    "aggressive": "final_aggressive.json",
                    "balanced": "final_balanced.json",
                    "conservative": "final_conservative.json",
                },
            },
        )
        return RunResult.model_validate(result.model_dump(mode="python"))


def preflight_answer_revise_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Build the shared runtime without model or public-network calls."""

    budget = ExecutionBudget(resolved_config.budget)
    trace = TraceCollector()
    runtime = prepare_runtime(
        task,
        resolved_config,
        system_id="tongagent",
        execution_budget=budget,
        trace=trace,
        injected_backend=fixture_backend,
        injected_model=model,
        semantic_strategy="fixed",
        enable_tongagent_evidence_state=True,
        enable_tongagent_token_control=True,
        enable_tongagent_context_compaction=True,
        reserve_final_synthesis=False,
    )
    return {
        "task_id": task.id,
        "runtime_mode": "answer_revise",
        "runtime_tools": [tool.name for tool in runtime.tools],
        "model_invocations": 0,
        "agent_constructed": True,
    }


__all__ = [
    "AtomicFact",
    "AtomicFactExtractor",
    "CandidateAnswer",
    "RevisedAnswer",
    "SelectiveFinal",
    "ShortAnswerAtomicFactAdapter",
    "SupportCheck",
    "preflight_answer_revise_workflow",
    "reliability_metrics",
    "revise_candidate",
    "run_answer_revise_workflow",
    "selective_finalize",
]
