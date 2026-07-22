"""Minimal score-first TongAgent workflow for short-answer benchmarks.

The normal path intentionally bypasses the strict FSM, Fact-Gap, Required Fact
Slots, Evidence Graph gates, Answer-Revise, and selective finalization.  It uses
the shared retrieval runtime and budgets, then always performs one answer
synthesis when at least one fetched source is available.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from retrieval_quality import classify_query_task_type

from ..budget import ExecutionBudget
from ..config import ResolvedConfig
from ..offline import FixtureBackend
from ..schema import Citation, CompletionStatus, EvalTask, RunResult
from ..tracing import TraceCollector
from .common import (
    PreparedRuntime,
    build_run_result,
    prepare_runtime,
    write_intermediate_artifacts,
)
from .permissive import (
    PermissiveSubquestion,
    ResearchBundle,
    ResearchSource,
    TaskType,
    _accounted_structured,
    _safe_summary,
    _schema_variant,
    _write_json,
    research_query,
)


AnswerType = Literal[
    "direct",
    "number",
    "date_difference",
    "numeric_difference",
    "count",
    "filter_and_count",
    "comparison",
    "list",
    "entity",
    "boolean",
]
CalculationKind = Literal["none", "subtract", "absolute_difference", "count", "compare"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LightweightPlan(_StrictModel):
    subquestions: list[str] = Field(min_length=1, max_length=3)
    answer_type: AnswerType = "direct"
    calculation: CalculationKind = "none"


class ScoreFirstOperand(_StrictModel):
    name: str
    value: str
    source_id: str


class ScoreFirstNote(_StrictModel):
    subquestion: str
    answer_candidate: str | None = None
    relevant_passages: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    unresolved: bool = False
    operands: list[ScoreFirstOperand] = Field(default_factory=list)


class ScoreFirstAnswer(_StrictModel):
    answer_text: str | None = None
    rationale: str = ""
    source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    calculation_structured: bool = False


class DeterministicCalculation(_StrictModel):
    status: Literal["success", "unavailable", "conflict"]
    answer_text: str | None = None
    operation: CalculationKind
    operand_values: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    reason: str


_NUMBER = re.compile(r"-?\d+(?:[,.]\d+)?")
_YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_TASK_TYPES: frozenset[str] = frozenset(
    {
        "single_fact_lookup",
        "list_or_enumeration",
        "comparison",
        "date_or_numeric_lookup",
    }
)


def _task_type(question: str) -> TaskType:
    """Coerce the shared heuristic to the permissive tool's closed vocabulary."""

    value = classify_query_task_type(question)
    if value not in _TASK_TYPES:
        return "single_fact_lookup"
    return cast("TaskType", value)


def _fallback_plan(task: EvalTask) -> LightweightPlan:
    task_type = _task_type(task.question)
    answer_type: AnswerType = (
        "number" if task_type == "date_or_numeric_lookup" else "direct"
    )
    return LightweightPlan(subquestions=[task.question], answer_type=answer_type)


def _plan(*, runtime: PreparedRuntime, task: EvalTask) -> tuple[LightweightPlan, bool]:
    """Perform exactly one planning call; failure degrades to the question."""

    response = _accounted_structured(
        runtime=runtime,
        schema=LightweightPlan,
        prompt=(
            "Create 1-3 short atomic research subquestions for the question. "
            "Each subquestion must request one fact. Choose the answer type and "
            "a deterministic calculation only when clearly required. Do not "
            "produce requirements, slots, URLs, evidence schemas, or an answer.\n\n"
            f"Question: {task.question}"
        ),
        label="tongagent.score_first.plan",
        stage="planner",
    )
    if response is None:
        return _fallback_plan(task), True
    try:
        candidate = LightweightPlan.model_validate(response)
    except Exception:
        return _fallback_plan(task), True
    subquestions = [" ".join(item.split()) for item in candidate.subquestions]
    subquestions = [item for item in subquestions if item]
    if not subquestions:
        return _fallback_plan(task), True
    return candidate.model_copy(update={"subquestions": subquestions[:3]}), False


def _note_fallback(
    subquestion: str, sources: Sequence[ResearchSource]
) -> ScoreFirstNote:
    return ScoreFirstNote(
        subquestion=subquestion,
        answer_candidate=None,
        relevant_passages=[item.passage for item in sources[:2]],
        source_ids=[item.source_id for item in sources[:2]],
        unresolved=True,
    )


def _note(
    *,
    runtime: PreparedRuntime,
    subquestion_id: str,
    subquestion: str,
    sources: Sequence[ResearchSource],
) -> ScoreFirstNote:
    """Extract one compact source-bound note without evidence registration."""

    if not sources:
        return _note_fallback(subquestion, sources)
    response = _accounted_structured(
        runtime=runtime,
        schema=_schema_variant(ScoreFirstNote, subquestion_id),
        prompt=(
            "Answer the subquestion from only the fetched passages. Give a concise "
            "answer_candidate whenever the passages contain usable information. "
            "Copy at most two relevant passages and existing source IDs. For a "
            "date, number, count, or comparison, record the explicit source-bound "
            "operand values. Do not invent facts or URLs.\n\n"
            f"Subquestion: {subquestion}\nSources: "
            + json.dumps(
                [item.model_dump(mode="json") for item in sources],
                ensure_ascii=False,
            )
        )[:10_000],
        label=f"tongagent.score_first.note.{subquestion_id}",
        stage="evidence_selection",
    )
    if response is None:
        return _note_fallback(subquestion, sources)
    try:
        note = ScoreFirstNote.model_validate(response)
    except Exception:
        return _note_fallback(subquestion, sources)
    valid_ids = {item.source_id for item in sources}
    source_ids = [item for item in note.source_ids if item in valid_ids]
    passages = [
        item
        for item in note.relevant_passages
        if any(
            " ".join(item.split()) in " ".join(source.passage.split())
            for source in sources
        )
    ]
    operands = [item for item in note.operands if item.source_id in valid_ids]
    candidate = note.answer_candidate.strip() if note.answer_candidate else None
    if not candidate or not source_ids:
        return _note_fallback(subquestion, sources)
    return note.model_copy(
        update={
            "subquestion": subquestion,
            "answer_candidate": candidate,
            "source_ids": list(dict.fromkeys(source_ids)),
            "relevant_passages": passages[:2],
            "operands": operands,
            "unresolved": False,
        }
    )


def _numeric(value: str) -> float | None:
    match = _NUMBER.search(value.replace(",", ""))
    return float(match.group()) if match else None


def _year(value: str) -> int | None:
    match = _YEAR.search(value)
    return int(match.group()) if match else None


def _format_number(value: float) -> str:
    return (
        str(int(value))
        if value.is_integer()
        else f"{value:.6f}".rstrip("0").rstrip(".")
    )


def deterministic_calculation(
    plan: LightweightPlan, notes: Sequence[ScoreFirstNote]
) -> DeterministicCalculation:
    """Execute the smallest source-bound calculation possible from note operands."""

    operands = [item for note in notes for item in note.operands]
    values = [item.value for item in operands]
    sources = list(dict.fromkeys(item.source_id for item in operands))
    if plan.calculation == "none":
        return DeterministicCalculation(
            status="unavailable",
            operation="none",
            operand_values=values,
            source_ids=sources,
            reason="no deterministic operation requested",
        )
    if plan.calculation in {"subtract", "absolute_difference"}:
        if plan.answer_type == "date_difference":
            numbers = [_year(item.value) for item in operands]
        else:
            numbers = [_numeric(item.value) for item in operands]
        usable = [item for item in numbers if item is not None]
        if len(usable) < 2:
            return DeterministicCalculation(
                status="unavailable",
                operation=plan.calculation,
                operand_values=values,
                source_ids=sources,
                reason="fewer than two explicit source-bound operands",
            )
        result = float(usable[0]) - float(usable[1])
        if plan.calculation == "absolute_difference":
            result = abs(result)
        return DeterministicCalculation(
            status="success",
            answer_text=_format_number(result),
            operation=plan.calculation,
            operand_values=values,
            source_ids=sources,
            reason="computed from two explicit source-bound operands",
        )
    if plan.calculation == "count":
        if len(operands) == 1 and (number := _numeric(operands[0].value)) is not None:
            answer = _format_number(number)
        elif operands:
            answer = str(len({item.value.casefold() for item in operands}))
        else:
            return DeterministicCalculation(
                status="unavailable",
                operation="count",
                reason="no explicit source-bound count operands",
            )
        return DeterministicCalculation(
            status="success",
            answer_text=answer,
            operation="count",
            operand_values=values,
            source_ids=sources,
            reason="counted explicit source-bound operands",
        )
    if plan.calculation == "compare" and len(operands) >= 2:
        answer = (
            "yes"
            if operands[0].value.casefold() == operands[1].value.casefold()
            else "no"
        )
        return DeterministicCalculation(
            status="success",
            answer_text=answer,
            operation="compare",
            operand_values=values,
            source_ids=sources,
            reason="compared two explicit source-bound operands",
        )
    return DeterministicCalculation(
        status="unavailable",
        operation=plan.calculation,
        operand_values=values,
        source_ids=sources,
        reason="deterministic operation could not be executed",
    )


def _fallback_answer(
    notes: Sequence[ScoreFirstNote], calculation: DeterministicCalculation
) -> ScoreFirstAnswer:
    if calculation.status == "success" and calculation.answer_text:
        return ScoreFirstAnswer(
            answer_text=calculation.answer_text,
            rationale=calculation.reason,
            source_ids=calculation.source_ids,
            confidence="medium",
            calculation_structured=True,
        )
    candidates = [item for item in notes if item.answer_candidate and item.source_ids]
    if candidates:
        selected = candidates[-1]
        return ScoreFirstAnswer(
            answer_text=selected.answer_candidate,
            rationale="Best available source-grounded research note candidate.",
            source_ids=selected.source_ids,
            confidence="low" if selected.unresolved else "medium",
            calculation_structured=False,
        )
    return ScoreFirstAnswer(rationale="No successfully sourced candidate.")


def _synthesize(
    *,
    runtime: PreparedRuntime,
    task: EvalTask,
    plan: LightweightPlan,
    notes: Sequence[ScoreFirstNote],
    sources: Mapping[str, ResearchSource],
    calculation: DeterministicCalculation,
) -> ScoreFirstAnswer:
    """Always attempt one concise answer synthesis after research."""

    successful_source_ids = {source for note in notes for source in note.source_ids}
    if not successful_source_ids:
        return ScoreFirstAnswer(rationale="All subquestions lacked fetched sources.")
    payload = {
        "question": task.question,
        "plan": plan.model_dump(mode="json"),
        "notes": [item.model_dump(mode="json") for item in notes],
        "sources": [
            {
                "source_id": item.source_id,
                "title": item.title,
                "url": item.url,
                "passage": item.passage,
            }
            for item in sources.values()
            if item.source_id in successful_source_ids
        ],
        "deterministic_calculation": calculation.model_dump(mode="json"),
    }
    response = _accounted_structured(
        runtime=runtime,
        schema=ScoreFirstAnswer,
        prompt=(
            "Give the shortest direct answer to the original question using only "
            "the supplied notes, fetched passages, and deterministic calculation. "
            "Prefer the deterministic result when successful. Do not refuse merely "
            "because there is one source, no exact quote, incomplete Evidence Graph, "
            "or low confidence. Return null only if every source failed, critical "
            "operands are entirely absent, or key sources explicitly conflict.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )[:16_000],
        label="tongagent.score_first.synthesis",
        stage="final_synthesis",
    )
    if response is None:
        return _fallback_answer(notes, calculation)
    try:
        answer = ScoreFirstAnswer.model_validate(response)
    except Exception:
        return _fallback_answer(notes, calculation)
    valid_ids = set(sources)
    source_ids = [item for item in answer.source_ids if item in valid_ids]
    text = answer.answer_text.strip() if answer.answer_text else None
    if not text or not source_ids:
        return _fallback_answer(notes, calculation)
    return answer.model_copy(
        update={
            "answer_text": text,
            "source_ids": list(dict.fromkeys(source_ids)),
            "calculation_structured": calculation.status == "success",
        }
    )


def _citations(
    answer: ScoreFirstAnswer, sources: Mapping[str, ResearchSource]
) -> list[Citation]:
    return [
        Citation(
            citation_id=f"score-first-{index}",
            source_id=source.source_id,
            url=source.url,
            title=source.title or None,
            quote=source.passage[:2_000],
            metadata={"acquisition_method": source.acquisition_method},
        )
        for index, source_id in enumerate(answer.source_ids, start=1)
        if (source := sources.get(source_id)) is not None
    ]


def run_score_first_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> RunResult:
    """Run the bounded score-first path and persist auditable native artifacts."""

    artifact_directory = Path(resolved_config.artifact_directory).expanduser()
    native = artifact_directory / "native" / "tongagent"
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    trace = TraceCollector()
    budget = ExecutionBudget(resolved_config.budget)
    run_id = f"tongagent-{task.id}-{uuid.uuid4().hex}"
    runtime: PreparedRuntime | None = None
    caught: Exception | None = None
    plan = _fallback_plan(task)
    plan_fallback = True
    bundles: list[ResearchBundle] = []
    notes: list[ScoreFirstNote] = []
    sources: dict[str, ResearchSource] = {}
    calculation = DeterministicCalculation(
        status="unavailable", operation="none", reason="workflow not started"
    )
    answer = ScoreFirstAnswer(rationale="workflow not started")
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
            reserve_final_synthesis=True,
        )
        trace.record("score_first_phase", phase="LIGHTWEIGHT_PLAN")
        plan, plan_fallback = _plan(runtime=runtime, task=task)
        runtime.research_budget.configure_subquestions(
            [f"SQ{index}" for index, _ in enumerate(plan.subquestions, start=1)]
        )
        runtime.middleware.configure_token_partitions(
            [f"SQ{index}" for index, _ in enumerate(plan.subquestions, start=1)]
        )
        trace.record("score_first_phase", phase="RESEARCH")
        for index, subquestion_text in enumerate(plan.subquestions, start=1):
            subquestion_id = f"SQ{index}"
            runtime.research_budget.activate_subquestion(subquestion_id)
            runtime.middleware.activate_token_subquestion(subquestion_id)
            task_type = _task_type(subquestion_text)
            subquestion = PermissiveSubquestion(
                id=subquestion_id,
                question=subquestion_text,
                task_type=task_type,
                required_for_final_answer=True,
            )
            bundle = research_query(
                query=subquestion_text,
                task_type=task_type,
                subquestion=subquestion,
                search_tool=next(
                    tool for tool in runtime.tools if tool.name == "web_search"
                ),
                fetch_tool=next(
                    tool for tool in runtime.tools if tool.name == "fetch_url"
                ),
                max_sources=2,
            )
            bundles.append(bundle)
            sources.update({item.source_id: item for item in bundle.sources})
            notes.append(
                _note(
                    runtime=runtime,
                    subquestion_id=subquestion_id,
                    subquestion=subquestion_text,
                    sources=bundle.sources,
                )
            )
            trace.record(
                "score_first_research_complete",
                subquestion_id=subquestion_id,
                source_count=len(bundle.sources),
                broadened=bundle.broadened,
            )
        trace.record("score_first_phase", phase="OPTIONAL_DETERMINISTIC_CALCULATION")
        calculation = deterministic_calculation(plan, notes)
        trace.record("score_first_phase", phase="ANSWER_SYNTHESIS")
        answer = _synthesize(
            runtime=runtime,
            task=task,
            plan=plan,
            notes=notes,
            sources=sources,
            calculation=calculation,
        )
        trace.record(
            "score_first_phase",
            phase="FINAL_ANSWER",
            answered=bool(answer.answer_text),
        )
    except Exception as exc:
        caught = exc
        trace.record(
            "run_exception",
            phase="score_first",
            exception_type=type(exc).__name__,
            safe_summary=_safe_summary(exc),
        )
    finally:
        answer_text = (answer.answer_text or "").strip()
        final_answer = (
            f"FINAL_ANSWER: {answer_text}" if answer_text else "FINAL_ANSWER: ABSTAIN"
        )
        finished_at = datetime.now(UTC)
        mapped_source_ids = [item for item in answer.source_ids if item in sources]
        citation_coverage = (
            len(mapped_source_ids) / len(answer.source_ids)
            if answer.source_ids
            else None
        )
        posthoc_status = (
            "abstain"
            if not answer_text
            else "source_mapped"
            if citation_coverage == 1.0
            else "unsupported"
        )
        workflow_metrics: dict[str, Any] = {
            "runtime_mode": "score_first",
            "plan_fallback": plan_fallback,
            "subquestion_count": len(plan.subquestions),
            "research_note_count": len(notes),
            "successful_research_notes": sum(bool(item.source_ids) for item in notes),
            "answer_rate": 1.0 if answer_text else 0.0,
            "score_first_answer": answer_text or None,
            "calculation_status": calculation.status,
            "calculation_operation": calculation.operation,
            "posthoc_verified_status": posthoc_status,
            "posthoc_supported_claim_rate": (
                citation_coverage if answer_text else None
            ),
            "unsupported_answer_rate": (
                0.0 if posthoc_status == "source_mapped" else 1.0
            )
            if answer_text
            else None,
            "citation_coverage": citation_coverage,
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
            evidence_count=0 if runtime is not None else None,
            structural_subquestion_coverage=(
                sum(bool(item.source_ids) for item in notes) / len(plan.subquestions)
                if plan.subquestions
                else 0.0
            ),
            final_answer_override=final_answer,
        )
        if caught is None and result.completion_status == CompletionStatus.COMPLETED:
            result = result.model_copy(
                update={
                    "completion_status": (
                        CompletionStatus.COMPLETED
                        if answer_text
                        else CompletionStatus.PARTIAL
                    ),
                    "citations": _citations(answer, sources),
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
            native / "score_first_plan.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "plan": plan.model_dump(mode="json"),
                "fallback_used": plan_fallback,
            },
        )
        _write_json(
            native / "score_first_research.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "bundles": [item.model_dump(mode="json") for item in bundles],
                "notes": [item.model_dump(mode="json") for item in notes],
            },
        )
        _write_json(
            native / "score_first_calculation.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "calculation": calculation.model_dump(mode="json"),
            },
        )
        _write_json(
            native / "score_first_answer.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "answer": answer.model_dump(mode="json"),
                "final_answer": final_answer,
            },
        )
        _write_json(
            native / "posthoc_reliability.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "status": posthoc_status,
                "method": "source_mapping_only_not_claim_verification",
                "score_first_answer_unchanged": True,
                "answer_source_ids": answer.source_ids,
                "mapped_source_ids": mapped_source_ids,
                "citation_coverage": citation_coverage,
                "posthoc_supported_claim_rate": (
                    citation_coverage if answer_text else None
                ),
                "unsupported_answer_rate": workflow_metrics["unsupported_answer_rate"],
            },
        )
        return RunResult.model_validate(result.model_dump(mode="python"))


def preflight_score_first_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Construct the shared retrieval runtime without model/network calls."""

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
        reserve_final_synthesis=True,
    )
    return {
        "task_id": task.id,
        "runtime_mode": "score_first",
        "runtime_tools": [tool.name for tool in runtime.tools],
        "model_invocations": 0,
        "agent_constructed": True,
    }


__all__ = [
    "DeterministicCalculation",
    "LightweightPlan",
    "ScoreFirstAnswer",
    "ScoreFirstNote",
    "ScoreFirstOperand",
    "deterministic_calculation",
    "preflight_score_first_workflow",
    "run_score_first_workflow",
]
