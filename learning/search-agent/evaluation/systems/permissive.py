"""Permissive TongAgent research with post-hoc claim verification.

This module deliberately does not import or alter the strict FSM.  It shares
the same prepared runtime, semantic search/fetch wrappers, execution budget,
and EvidenceGraphStore, but moves Claim--Evidence--Source registration until
after all required subquestions have had a bounded chance to gather evidence.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from agent_policy import EffortPolicy
from evidence_graph import (
    MAX_QUOTE_CHARS,
    normalize_evidence_text,
    validate_evidence_graph,
)
from retrieval_quality import (
    assess_search_relevance,
    classify_query_task_type,
    deterministic_query_rewrite,
    normalize_atomic_search_query,
)

from ..budget import BudgetExceeded, ExecutionBudget
from ..config import BudgetLimits, ResolvedConfig
from ..fact_gap import (
    FactGap,
    RequiredFactSlot,
    SlotFactCandidate,
    SlotCoverage,
    fallback_slots,
    gaps_from_coverage,
    match_slots,
    repair_queries,
)
from ..offline import FixtureBackend
from ..schema import Citation, CompletionStatus, EvalTask, RunResult
from ..tracing import TraceCollector, sanitize_trace_value
from .common import (
    PreparedRuntime,
    build_run_result,
    prepare_runtime,
    write_intermediate_artifacts,
)


TaskType = Literal[
    "single_fact_lookup",
    "list_or_enumeration",
    "comparison",
    "date_or_numeric_lookup",
]
VerifiedStatus = Literal[
    "verified",
    "partially_supported",
    "unsupported",
    "contested",
]
TypedFactType = Literal[
    "integer",
    "float",
    "date",
    "year",
    "duration",
    "entity",
    "string",
    "list",
    "boolean",
]
AnswerOperation = Literal[
    "direct_lookup",
    "subtract",
    "date_difference",
    "add",
    "filter_and_count",
    "compare",
    "list_intersection",
    "boolean",
]

_MAX_SOURCES_PER_QUERY = 3
_MAX_RESEARCH_ROUNDS = 2
_PASSAGE_CHARS = 720
_DANGEROUS_QUERY = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|\b(?:javascript|data|file):)"
)
_PREFIX = re.compile(
    r"(?i)^\s*(?:search(?:\s+for)?|find|look\s+up|research|query)\s*[:\-–—]?\s*"
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
_QUOTED = re.compile(r'"([^"]+)"')
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PermissiveSubquestion(_StrictModel):
    id: str
    question: str = Field(min_length=1)
    task_type: TaskType
    required_for_final_answer: bool = True


class PermissivePlan(_StrictModel):
    objective: str = Field(min_length=1)
    subquestions: list[PermissiveSubquestion] = Field(min_length=1, max_length=3)


class RequiredFactSlotPlan(_StrictModel):
    slots: list[RequiredFactSlot] = Field(min_length=1, max_length=8)


class ResearchQueryDecision(_StrictModel):
    query: str = ""
    needs_second_round: bool = False
    missing_fact: str = ""


class ResearchSource(_StrictModel):
    source_id: str
    title: str
    url: str
    provider: str
    provider_rank: int = Field(ge=1)
    acquisition_method: str
    relevance_tier: str
    relevance_reason: str
    passage: str
    content_chars: int = Field(ge=0)
    fetch_status: str


class ResearchBundle(_StrictModel):
    query: str
    normalized_query: str
    sources: list[ResearchSource] = Field(default_factory=list)
    candidate_audit: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    broadened: bool = False
    query_validation_status: str
    query_normalizations: list[str] = Field(default_factory=list)
    query_rejection_reason: str | None = None


class ResearchNote(_StrictModel):
    subquestion_id: str
    claim_candidates: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    supporting_passages: list[str] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    typed_facts: list[TypedFact] = Field(default_factory=list)


class DraftClaim(_StrictModel):
    claim_id: str
    text: str = Field(min_length=12, max_length=500)
    source_ids: list[str] = Field(default_factory=list)
    critical_for_final_answer: bool
    subquestion_id: str


class DraftAnswer(_StrictModel):
    claims: list[DraftClaim] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    proposed_answer: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    missing_information: list[str] = Field(default_factory=list)


class VerifiedClaim(_StrictModel):
    claim_id: str
    status: VerifiedStatus
    source_ids: list[str] = Field(default_factory=list)
    exact_quotes: list[str] = Field(default_factory=list)
    explanation: str
    canonical_claim_id: str | None = None
    registration_failures: list[dict[str, str]] = Field(default_factory=list)
    typed_facts: list[TypedFact] = Field(default_factory=list)


class TypedFact(_StrictModel):
    """A source- and claim-bound value suitable for deterministic execution.

    Facts are intentionally data, not executable expressions.  The model may
    propose them in a research note, but code validates both their values and
    their existing canonical Source/Claim links before they are usable.
    """

    fact_id: str = ""
    subquestion_id: str
    fact_type: TypedFactType
    value: Any
    unit: str | None = None
    entity: str | None = None
    attribute: str = ""
    relation: str | None = None
    qualifier: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    verification_status: VerifiedStatus = "partially_supported"
    raw_text: str = ""


class AnswerPlan(_StrictModel):
    operation: AnswerOperation
    required_fact_ids: list[str] = Field(min_length=1, max_length=12)
    output_type: str = Field(min_length=1, max_length=80)
    output_unit: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnswerExecution(_StrictModel):
    """Auditable outcome of the non-LLM answer executor."""

    status: Literal["success", "abstain"]
    answer_value: Any | None = None
    answer_text: str | None = None
    output_type: str | None = None
    output_unit: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    calculation_trace: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None


_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*"
    r"(feet|foot|ft|met(?:er|re|ers|res)|m|years?|months?|days?|percent|%)?\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def _normalise_fact_value(fact: TypedFact) -> TypedFact | None:
    """Validate model-proposed values without inferring an unstated unit."""

    value = fact.value
    try:
        if fact.fact_type == "integer":
            if isinstance(value, bool):
                return None
            normalized: Any = int(str(value).replace(",", ""))
        elif fact.fact_type == "float":
            if isinstance(value, bool):
                return None
            normalized = str(Decimal(str(value).replace(",", "")).normalize())
        elif fact.fact_type == "year":
            normalized = int(str(value))
            if not 1000 <= normalized <= 2999:
                return None
            if fact.unit not in {None, "year", "years"}:
                return None
        elif fact.fact_type == "date":
            normalized = date.fromisoformat(str(value)).isoformat()
        elif fact.fact_type == "duration":
            if isinstance(value, bool) or not fact.unit:
                return None
            normalized = str(Decimal(str(value).replace(",", "")).normalize())
        elif fact.fact_type == "list":
            if not isinstance(value, list):
                return None
            normalized = value
        elif fact.fact_type == "boolean":
            if not isinstance(value, bool):
                return None
            normalized = value
        elif fact.fact_type in {"entity", "string"}:
            normalized = " ".join(str(value).split())
            if not normalized:
                return None
        else:  # pragma: no cover - Literal is enforced by Pydantic.
            return None
    except (InvalidOperation, TypeError, ValueError):
        return None
    return fact.model_copy(update={"value": normalized})


def _status_for_fact_claims(claims: Sequence[VerifiedClaim]) -> VerifiedStatus:
    statuses = {item.status for item in claims}
    if "contested" in statuses:
        return "contested"
    if "unsupported" in statuses:
        return "unsupported"
    if "verified" in statuses:
        return "verified"
    return "partially_supported"


def _fact_semantics(text: str) -> tuple[str | None, str, str | None]:
    """Conservative semantic labels used by deterministic slot matching."""

    entity_match = re.search(
        r"\b([A-Z][A-Za-z.'-]*(?:\s+(?:[A-Z][A-Za-z.'-]*|of|the|and))*)", text
    )
    entity = entity_match.group(1) if entity_match else None
    folded = text.casefold()
    attribute = next(
        (
            name
            for needle, name in (
                ("height", "height"),
                ("deep", "depth"),
                ("imprison", "imprisonment_dates"),
                ("born", "birth_date"),
                ("birthplace", "birthplace"),
                ("hometown", "hometown"),
                ("admitted", "admission_to_union"),
                ("album", "discography"),
            )
            if needle in folded
        ),
        "fact",
    )
    relation = "in" if " in " in folded else None
    return entity, attribute, relation


def _heuristic_typed_facts(
    *,
    claim: DraftClaim,
    verified_claim: VerifiedClaim,
    next_id: int,
) -> list[TypedFact]:
    """Conservative fallback for old notes which predate typed-fact output.

    The fallback never creates a value from model memory: it reads only the
    already source-bound draft claim and is useful for ordinary years, dates,
    and explicit numbers.  Rich lists remain model-proposed typed facts.
    """

    source_ids = list(verified_claim.source_ids)
    if not source_ids:
        return []
    facts: list[TypedFact] = []
    raw = claim.text
    entity, attribute, relation = _fact_semantics(raw)
    evidence = list(verified_claim.exact_quotes)
    for match in _ISO_DATE.finditer(raw):
        facts.append(
            TypedFact(
                fact_id=f"F{next_id + len(facts)}",
                subquestion_id=claim.subquestion_id,
                fact_type="date",
                value=match.group(1),
                source_ids=source_ids,
                claim_ids=[claim.claim_id],
                verification_status=verified_claim.status,
                entity=entity,
                attribute=attribute,
                relation=relation,
                qualifier={"exact_quotes": evidence},
                raw_text=raw,
            )
        )
    for match in _NUMBER_WITH_UNIT.finditer(raw):
        number, unit = match.groups()
        # A date has already been represented as one atomic fact; do not turn
        # its year/month/day fragments into unrelated numeric operands.
        if match.start() and raw[max(0, match.start() - 1) : match.start()] == "-":
            continue
        compact = number.replace(",", "")
        if "." in compact:
            fact_type: TypedFactType = "float"
            value: Any = str(Decimal(compact).normalize())
        else:
            integer = int(compact)
            fact_type = (
                "year" if unit is None and 1000 <= integer <= 2999 else "integer"
            )
            value = integer
        facts.append(
            TypedFact(
                fact_id=f"F{next_id + len(facts)}",
                subquestion_id=claim.subquestion_id,
                fact_type=fact_type,
                value=value,
                unit=(
                    unit.casefold()
                    if unit
                    else ("year" if fact_type == "year" else None)
                ),
                source_ids=source_ids,
                claim_ids=[claim.claim_id],
                verification_status=verified_claim.status,
                entity=entity,
                attribute=attribute,
                relation=relation,
                qualifier={"exact_quotes": evidence},
                raw_text=raw,
            )
        )
    return facts


def _collect_typed_facts(
    *,
    notes: Sequence[ResearchNote],
    draft: DraftAnswer,
    verified: Sequence[VerifiedClaim],
) -> tuple[list[TypedFact], list[VerifiedClaim], list[dict[str, str]]]:
    """Bind note facts to existing verified claims, then normalize them.

    This is the boundary which guarantees that no TypedFact becomes usable
    merely because an LLM printed it.  Each retained fact has an actual
    fetched source and one or more pre-existing draft/verified claims.
    """

    draft_by_sq: dict[str, list[DraftClaim]] = {}
    for claim in draft.claims:
        draft_by_sq.setdefault(claim.subquestion_id, []).append(claim)
    verified_by_id = {item.claim_id: item for item in verified}
    facts: list[TypedFact] = []
    failures: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(candidate: TypedFact, origin: str) -> None:
        source_ids = list(dict.fromkeys(candidate.source_ids))
        candidates = [
            claim
            for claim in draft_by_sq.get(candidate.subquestion_id, [])
            if set(claim.source_ids).intersection(source_ids)
            and claim.claim_id in verified_by_id
        ]
        if not source_ids or not candidates:
            failures.append(
                {
                    "category": "unmapped_typed_fact",
                    "origin": origin,
                    "fact_id": candidate.fact_id or "",
                }
            )
            return
        verified_claims = [verified_by_id[item.claim_id] for item in candidates]
        status = _status_for_fact_claims(verified_claims)
        normalized = _normalise_fact_value(
            candidate.model_copy(
                update={
                    "fact_id": candidate.fact_id or f"F{len(facts) + 1}",
                    "source_ids": source_ids,
                    "claim_ids": [item.claim_id for item in candidates],
                    "verification_status": status,
                }
            )
        )
        if normalized is None:
            failures.append(
                {
                    "category": "invalid_typed_fact_value",
                    "origin": origin,
                    "fact_id": candidate.fact_id or "",
                }
            )
            return
        key = (
            normalized.subquestion_id,
            normalized.fact_type,
            json.dumps(normalized.value, sort_keys=True, default=str),
            "|".join(sorted(normalized.source_ids)),
        )
        if key in seen:
            return
        seen.add(key)
        facts.append(normalized)

    for note in notes:
        for candidate in note.typed_facts:
            add(candidate, "research_note")
    for claim in draft.claims:
        checked = verified_by_id.get(claim.claim_id)
        if checked is None:
            continue
        for candidate in _heuristic_typed_facts(
            claim=claim, verified_claim=checked, next_id=len(facts) + 1
        ):
            add(candidate, "verified_claim_fallback")

    facts_by_claim: dict[str, list[TypedFact]] = {}
    for fact in facts:
        for claim_id in fact.claim_ids:
            facts_by_claim.setdefault(claim_id, []).append(fact)
    updated_verified = [
        item.model_copy(update={"typed_facts": facts_by_claim.get(item.claim_id, [])})
        for item in verified
    ]
    return facts, updated_verified, failures


def _usable_fact(
    fact: TypedFact,
    *,
    allow_partial: bool,
) -> bool:
    return fact.verification_status == "verified" or (
        allow_partial and fact.verification_status == "partially_supported"
    )


def _decimal_operand(fact: TypedFact) -> Decimal | None:
    if fact.fact_type not in {"integer", "float", "year", "duration"}:
        return None
    try:
        return Decimal(str(fact.value))
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal) -> str:
    integral = value.to_integral_value()
    return str(int(integral)) if value == integral else format(value.normalize(), "f")


def _execute_answer_plan(
    *,
    plan: AnswerPlan | None,
    facts: Sequence[TypedFact],
    allow_partial: bool,
) -> AnswerExecution:
    """Execute a whitelisted calculation without evaluating model-authored code."""

    if plan is None:
        return AnswerExecution(status="abstain", failure_reason="missing_answer_plan")
    by_id = {item.fact_id: item for item in facts}
    required = [by_id.get(item) for item in plan.required_fact_ids]
    if len(set(plan.required_fact_ids)) != len(plan.required_fact_ids) or any(
        item is None for item in required
    ):
        return AnswerExecution(
            status="abstain", failure_reason="missing_required_typed_fact"
        )
    operands = [cast(TypedFact, item) for item in required]
    unusable = [
        item.fact_id
        for item in operands
        if not _usable_fact(item, allow_partial=allow_partial)
    ]
    if unusable:
        return AnswerExecution(
            status="abstain",
            fact_ids=[item.fact_id for item in operands],
            failure_reason="unsupported_or_contested_typed_fact",
            calculation_trace=[{"unusable_fact_ids": unusable}],
        )
    trace: list[dict[str, Any]] = []
    try:
        if plan.operation == "direct_lookup":
            if len(operands) != 1:
                raise ValueError("direct_lookup_requires_one_fact")
            value = operands[0].value
            trace.append(
                {
                    "operation": "direct_lookup",
                    "fact_id": operands[0].fact_id,
                    "value": value,
                }
            )
        elif plan.operation in {"subtract", "add"}:
            if len(operands) != 2:
                raise ValueError("numeric_operation_requires_two_facts")
            left, right = (_decimal_operand(item) for item in operands)
            if left is None or right is None:
                raise ValueError("numeric_operand_type_mismatch")
            units = {item.unit for item in operands if item.unit}
            if len(units) > 1:
                raise ValueError("unit_conflict")
            value = left - right if plan.operation == "subtract" else left + right
            trace.append(
                {
                    "operation": plan.operation,
                    "operand_a": {
                        "fact_id": operands[0].fact_id,
                        "value": str(left),
                        "unit": operands[0].unit,
                    },
                    "operand_b": {
                        "fact_id": operands[1].fact_id,
                        "value": str(right),
                        "unit": operands[1].unit,
                    },
                    "result": str(value),
                    "unit": plan.output_unit or operands[0].unit,
                }
            )
            value = _format_decimal(value)
        elif plan.operation == "date_difference":
            if len(operands) != 2 or any(
                item.fact_type not in {"date", "year"} for item in operands
            ):
                raise ValueError("date_difference_requires_two_dates_or_years")
            if all(item.fact_type == "year" for item in operands):
                value = int(operands[1].value) - int(operands[0].value)
                rule = "calendar_year_difference"
            else:
                start = date.fromisoformat(str(operands[0].value))
                end = date.fromisoformat(str(operands[1].value))
                value = (
                    end.year
                    - start.year
                    - ((end.month, end.day) < (start.month, start.day))
                )
                rule = "completed_anniversaries"
            trace.append(
                {
                    "operation": "date_difference",
                    "start_fact": operands[0].fact_id,
                    "end_fact": operands[1].fact_id,
                    "rounding": rule,
                    "result": value,
                    "unit": "years",
                }
            )
        elif plan.operation == "filter_and_count":
            if not operands or operands[0].fact_type != "list":
                raise ValueError("filter_and_count_requires_list_fact")
            field = str(plan.parameters.get("field", "")).strip()
            if not field:
                raise ValueError("filter_and_count_requires_field")
            lower_id = str(plan.parameters.get("gte_fact_id", "")).strip()
            upper_id = str(plan.parameters.get("lte_fact_id", "")).strip()
            lower = _decimal_operand(by_id[lower_id]) if lower_id in by_id else None
            upper = _decimal_operand(by_id[upper_id]) if upper_id in by_id else None
            if lower_id and lower is None or upper_id and upper is None:
                raise ValueError("filter_bound_missing_or_non_numeric")
            included: list[Any] = []
            excluded: list[Any] = []
            for item in cast(list[Any], operands[0].value):
                if not isinstance(item, Mapping) or field not in item:
                    excluded.append(item)
                    continue
                candidate = Decimal(str(item[field]))
                if (lower is None or candidate >= lower) and (
                    upper is None or candidate <= upper
                ):
                    included.append(item)
                else:
                    excluded.append(item)
            value = len(included)
            trace.append(
                {
                    "operation": "filter_and_count",
                    "list_fact": operands[0].fact_id,
                    "field": field,
                    "lower": str(lower) if lower is not None else None,
                    "upper": str(upper) if upper is not None else None,
                    "included": included,
                    "excluded": excluded,
                    "result": value,
                }
            )
        elif plan.operation == "compare":
            if len(operands) != 2:
                raise ValueError("compare_requires_two_facts")
            direction = str(plan.parameters.get("direction", "equals"))
            left, right = operands[0].value, operands[1].value
            if direction == "equals":
                value = left == right
            elif direction == "greater_than":
                value = Decimal(str(left)) > Decimal(str(right))
            elif direction == "less_than":
                value = Decimal(str(left)) < Decimal(str(right))
            else:
                raise ValueError("unsupported_comparison_direction")
            trace.append(
                {
                    "operation": "compare",
                    "direction": direction,
                    "left_fact": operands[0].fact_id,
                    "right_fact": operands[1].fact_id,
                    "result": value,
                }
            )
        elif plan.operation == "list_intersection":
            if len(operands) != 2 or any(item.fact_type != "list" for item in operands):
                raise ValueError("list_intersection_requires_two_list_facts")
            right = {
                json.dumps(item, sort_keys=True, default=str)
                for item in operands[1].value
            }
            value = [
                item
                for item in operands[0].value
                if json.dumps(item, sort_keys=True, default=str) in right
            ]
            if plan.output_type == "count":
                value = len(value)
            trace.append(
                {
                    "operation": "list_intersection",
                    "left_fact": operands[0].fact_id,
                    "right_fact": operands[1].fact_id,
                    "result": value,
                }
            )
        elif plan.operation == "boolean":
            if len(operands) != 1 or operands[0].fact_type != "boolean":
                raise ValueError("boolean_requires_one_boolean_fact")
            value = operands[0].value
            trace.append(
                {
                    "operation": "boolean",
                    "fact_id": operands[0].fact_id,
                    "result": value,
                }
            )
        else:  # pragma: no cover - Literal is enforced by Pydantic.
            raise ValueError("unsupported_answer_operation")
    except (InvalidOperation, TypeError, ValueError) as exc:
        return AnswerExecution(
            status="abstain",
            fact_ids=[item.fact_id for item in operands],
            calculation_trace=trace,
            failure_reason=str(exc),
        )
    unit = plan.output_unit
    answer_text = str(value)
    if unit and str(unit).casefold() not in {"none", "null"}:
        answer_text = f"{answer_text} {unit}"
    return AnswerExecution(
        status="success",
        answer_value=value,
        answer_text=answer_text,
        output_type=plan.output_type,
        output_unit=unit,
        fact_ids=[item.fact_id for item in operands],
        calculation_trace=trace,
    )


@dataclass(frozen=True)
class _QueryNormalization:
    original: str
    normalized: str
    status: str
    normalizations: tuple[str, ...]
    rejection_reason: str | None = None


def _json_tool_call(tool: BaseTool, arguments: Mapping[str, Any]) -> dict[str, Any]:
    raw = tool.invoke(dict(arguments))
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"status": "error", "error": "invalid_tool_payload"}
    return dict(parsed) if isinstance(parsed, Mapping) else {"status": "error"}


def _source_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""


def _passage(content: str, query: str) -> str:
    """Return a bounded literal excerpt, biased toward query-bearing sentences."""

    text = normalize_evidence_text(content)
    if len(text) <= _PASSAGE_CHARS:
        return text
    terms = {
        token.casefold()
        for token in _WORD.findall(query)
        if len(token) >= 3
        and token.casefold()
        not in {"the", "and", "for", "from", "with", "overview", "list"}
    }
    sentences = [item.strip() for item in _SENTENCE.split(text) if item.strip()]
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (-sum(term in pair[1].casefold() for term in terms), pair[0]),
    )
    selected: list[tuple[int, str]] = []
    used = 0
    for index, sentence in ranked:
        if used and used + len(sentence) + 1 > _PASSAGE_CHARS:
            continue
        selected.append((index, sentence))
        used += len(sentence) + 1
        if used >= _PASSAGE_CHARS:
            break
    return (
        " ".join(sentence for _, sentence in sorted(selected)) or text[:_PASSAGE_CHARS]
    )


def _soft_normalize_query(
    query: str,
    *,
    subquestion: PermissiveSubquestion,
) -> _QueryNormalization:
    """Repair ordinary query quality issues without blocking research."""

    original = " ".join(str(query).split())
    normalizations: list[str] = []
    candidate = _PREFIX.sub("", original)
    if candidate != original:
        normalizations.append("removed_explanatory_prefix")
    if not candidate:
        candidate = subquestion.question
        normalizations.append("used_subquestion_fallback")
    if _DANGEROUS_QUERY.search(candidate):
        fallback = " ".join(subquestion.question.split())
        if not fallback or _DANGEROUS_QUERY.search(fallback):
            return _QueryNormalization(
                original=original,
                normalized="",
                status="rejected",
                normalizations=tuple(normalizations),
                rejection_reason="dangerous_protocol_or_url_injection",
            )
        candidate = fallback
        normalizations.append("replaced_unsafe_query_with_subquestion")
    words = _WORD.findall(candidate)
    if not words:
        return _QueryNormalization(
            original=original,
            normalized="",
            status="rejected",
            normalizations=tuple(normalizations),
            rejection_reason="no_search_terms_after_normalization",
        )
    try:
        normalized = normalize_atomic_search_query(candidate, max_words=12)
    except ValueError:
        normalized = " ".join(words[:12])
        normalizations.append("bounded_query_length")
    if subquestion.task_type == "list_or_enumeration" and not any(
        token in normalized.casefold()
        for token in ("list", "overview", "discography", "timeline")
    ):
        normalized = f"{normalized} overview".strip()
        normalizations.append("added_enumeration_hint")
    if len(_WORD.findall(normalized)) > 12:
        normalized = " ".join(_WORD.findall(normalized)[:12])
        normalizations.append("bounded_query_length")
    return _QueryNormalization(
        original=original,
        normalized=normalized,
        status="normalized" if normalizations or normalized != original else "accepted",
        normalizations=tuple(normalizations),
    )


def _broaden_query(query: str, task_type: TaskType) -> str:
    # An exact-phrase query can be broadened deterministically without an
    # extra model call.  Dropping only quote constraints keeps the entity and
    # attribute intact while allowing providers with token-only matching to
    # return a candidate on the second attempt.
    unquoted = _QUOTED.sub(lambda match: match.group(1), query)
    unquoted = " ".join(unquoted.split())
    if unquoted and unquoted != query:
        return unquoted[:180]
    rewritten = deterministic_query_rewrite(query)
    if rewritten == query and task_type == "list_or_enumeration":
        rewritten = f"{query} overview"
    if task_type == "list_or_enumeration" and "overview" not in rewritten.casefold():
        rewritten = f"{rewritten} overview"
    widened = " ".join(rewritten.split())[:180]
    return widened if widened != query else ""


def _provider_rank(value: Any, fallback: int) -> int:
    """Return a stable positive provider rank for audit and tie breaking."""

    try:
        rank = int(value)
    except (TypeError, ValueError):
        return fallback
    return rank if rank >= 1 else fallback


def _set_candidate_fetch_status(
    candidate_audit: list[dict[str, Any]], url: str, status: str
) -> None:
    """Attach acquisition outcome to the already-persisted ranking record."""

    for item in candidate_audit:
        if item.get("url") == url:
            item["fetch_status"] = status
            return


def research_query(
    *,
    query: str,
    task_type: TaskType,
    subquestion: PermissiveSubquestion,
    search_tool: BaseTool,
    fetch_tool: BaseTool,
    max_sources: int = _MAX_SOURCES_PER_QUERY,
) -> ResearchBundle:
    """Search and fetch a bounded, canonical source bundle.

    ``web_search`` and ``fetch_url`` remain the sole provider-facing tools:
    their existing security validation, cache, source ledger, and external
    budgets are therefore authoritative.  This helper never treats snippets
    as page evidence.
    """

    normalized = _soft_normalize_query(query, subquestion=subquestion)
    if normalized.rejection_reason is not None:
        return ResearchBundle(
            query=normalized.original,
            normalized_query="",
            sources=[],
            candidate_audit=[],
            failures=[{"stage": "query", "reason": normalized.rejection_reason}],
            broadened=False,
            query_validation_status=normalized.status,
            query_normalizations=list(normalized.normalizations),
            query_rejection_reason=normalized.rejection_reason,
        )

    def search_once(value: str) -> dict[str, Any]:
        return _json_tool_call(search_tool, {"query": value, "max_results": 5})

    search_payload = search_once(normalized.normalized)
    results = search_payload.get("results", [])
    broadened = False
    if not isinstance(results, list) or not results:
        broad = _broaden_query(normalized.normalized, task_type)
        if broad and broad != normalized.normalized:
            broadened = True
            search_payload = search_once(broad)
            results = search_payload.get("results", [])
            normalized = _QueryNormalization(
                original=normalized.original,
                normalized=broad,
                status="broadened",
                normalizations=(
                    *normalized.normalizations,
                    "broadened_after_no_candidates",
                ),
            )
    raw_candidates = (
        [item for item in results if isinstance(item, Mapping)]
        if isinstance(results, list)
        else []
    )
    failures: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    for provider_rank, raw_candidate in enumerate(raw_candidates, start=1):
        candidate = dict(raw_candidate)
        candidate["provider_rank"] = _provider_rank(
            candidate.get("provider_rank"), provider_rank
        )
        assessment = assess_search_relevance(normalized.normalized, candidate)
        candidate.update(
            {
                "relevance_score": assessment["score"],
                "relevance_tier": assessment["tier"],
                "relevance_reason": assessment["reason"],
                "rejection_reason": assessment["rejection_reason"],
            }
        )
        candidate_audit.append(
            {
                "url": str(candidate.get("url", "")),
                "title": str(candidate.get("title", "")),
                "provider": str(candidate.get("provider", "unknown")),
                "provider_rank": _provider_rank(candidate.get("provider_rank"), 999),
                "relevance_score": int(assessment["score"]),
                "relevance_tier": str(assessment["tier"]),
                "relevance_reason": str(assessment["reason"]),
                "rejection_reason": assessment["rejection_reason"],
                "fetch_status": "not_attempted",
            }
        )
        if assessment["tier"] == "irrelevant":
            failures.append(
                {
                    "stage": "ranking",
                    "url": str(candidate.get("url", "")),
                    "reason": str(assessment["rejection_reason"] or "irrelevant"),
                }
            )
            continue
        candidates.append(candidate)
    ranked = sorted(
        candidates,
        key=lambda item: (
            0
            if str(item.get("relevance_tier", item.get("tier", ""))) == "relevant"
            else 1
            if str(item.get("relevance_tier", item.get("tier", ""))) == "uncertain"
            else 2,
            -int(item.get("relevance_score", 0) or 0),
            int(item.get("provider_rank", 999) or 999),
        ),
    )
    sources: list[ResearchSource] = []
    seen_urls: set[str] = set()
    failed_hosts: set[str] = set()
    limit = max(1, min(int(max_sources), _MAX_SOURCES_PER_QUERY))
    # Preserve the deterministic ranking while giving a healthy second host a
    # chance before consuming the remaining quota on the first host.  This is
    # especially important after an access-blocked response; it is not a
    # provider or ranking change, only bounded candidate acquisition policy.
    primary: list[Mapping[str, Any]] = []
    deferred_same_host: list[Mapping[str, Any]] = []
    candidate_hosts: set[str] = set()
    for candidate in ranked:
        host = _source_host(str(candidate.get("url", "")))
        if host and host in candidate_hosts:
            deferred_same_host.append(candidate)
        else:
            primary.append(candidate)
            if host:
                candidate_hosts.add(host)
    for candidate in [*primary, *deferred_same_host]:
        url = str(candidate.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        host = _source_host(url)
        # Do not burn fetch quota retrying an access-blocked host while another
        # candidate host remains available.  The underlying retrieval ledger
        # retains its own cooldown, and cached provider content remains valid.
        if (
            host
            and host in failed_hosts
            and any(
                _source_host(str(item.get("url", ""))) not in {"", host}
                for item in ranked
            )
        ):
            _set_candidate_fetch_status(candidate_audit, url, "skipped_failed_host")
            continue
        seen_urls.add(url)
        payload = _json_tool_call(fetch_tool, {"url": url, "max_chars": 12_000})
        _set_candidate_fetch_status(
            candidate_audit, url, str(payload.get("status", "error"))
        )
        if payload.get("status") != "success" or not payload.get("source_id"):
            if host and str(payload.get("failure_taxonomy", "")) in {
                "access_blocked",
                "rate_limited",
            }:
                failed_hosts.add(host)
            failures.append(
                {
                    "stage": "fetch",
                    "url": url,
                    "status": str(payload.get("status", "error")),
                    "reason": str(
                        payload.get("failure_taxonomy", payload.get("error", "unknown"))
                    ),
                }
            )
            continue
        content = str(payload.get("content", ""))
        passage = _passage(content, normalized.normalized)
        if not passage:
            failures.append(
                {"stage": "fetch", "url": url, "reason": "empty_page_content"}
            )
            continue
        sources.append(
            ResearchSource(
                source_id=str(payload["source_id"]),
                title=str(
                    payload.get("title", candidate.get("title", "Untitled source"))
                ),
                url=str(payload.get("url", url)),
                provider=str(
                    candidate.get("provider", search_payload.get("provider", "unknown"))
                ),
                provider_rank=_provider_rank(candidate.get("provider_rank"), 999),
                acquisition_method=str(
                    payload.get("acquisition_method", "direct_http")
                ),
                relevance_tier=str(
                    candidate.get("relevance_tier", candidate.get("tier", "uncertain"))
                ),
                relevance_reason=str(candidate.get("relevance_reason", "unknown")),
                passage=passage,
                content_chars=int(payload.get("content_chars", len(content)) or 0),
                fetch_status="success",
            )
        )
        if len(sources) >= limit:
            break
    if search_payload.get("status") != "success":
        failures.insert(
            0,
            {
                "stage": "search",
                "status": str(search_payload.get("status", "error")),
                "reason": str(search_payload.get("error", "provider_error")),
            },
        )
    return ResearchBundle(
        query=normalized.original,
        normalized_query=normalized.normalized,
        sources=sources,
        candidate_audit=candidate_audit,
        failures=failures,
        broadened=broadened,
        query_validation_status=normalized.status,
        query_normalizations=list(normalized.normalizations),
        query_rejection_reason=None,
    )


def _repair_slot_facts(
    *,
    runtime: PreparedRuntime,
    slots: Sequence[RequiredFactSlot],
    gaps: Sequence[FactGap],
    sources: dict[str, ResearchSource],
    max_searches: int,
    max_fetches: int,
    max_queries_per_slot: int,
) -> tuple[list[TypedFact], list[dict[str, Any]]]:
    """Run bounded, slot-specific retrieval and register only canonical facts."""

    by_slot = {item.slot_id: item for item in slots}
    search_tool = next(tool for tool in runtime.tools if tool.name == "web_search")
    fetch_tool = next(tool for tool in runtime.tools if tool.name == "fetch_url")
    repairs: list[TypedFact] = []
    trace: list[dict[str, Any]] = []
    used_searches = 0
    used_fetches = 0
    for gap in gaps:
        slot = by_slot.get(gap.slot_id)
        if slot is None or not gap.repairable:
            continue
        for query in repair_queries(slot, gap)[:max_queries_per_slot]:
            if used_searches >= max_searches or used_fetches >= max_fetches:
                trace.append(
                    {
                        "slot_id": slot.slot_id,
                        "status": "repair_budget_exhausted",
                        "search_used": used_searches,
                        "fetch_used": used_fetches,
                    }
                )
                break
            if slot.subquestion_id:
                runtime.research_budget.activate_subquestion(slot.subquestion_id)
                runtime.middleware.activate_token_subquestion(slot.subquestion_id)
            bundle = research_query(
                query=query.query,
                task_type=(
                    "list_or_enumeration"
                    if slot.cardinality == "list"
                    else "date_or_numeric_lookup"
                    if slot.fact_type
                    in {"integer", "float", "year", "date", "duration"}
                    else "single_fact_lookup"
                ),
                subquestion=PermissiveSubquestion(
                    id=slot.subquestion_id or "repair",
                    question=f"{slot.entity or ''} {slot.attribute}".strip(),
                    task_type=(
                        "list_or_enumeration"
                        if slot.cardinality == "list"
                        else "date_or_numeric_lookup"
                        if slot.fact_type
                        in {"integer", "float", "year", "date", "duration"}
                        else "single_fact_lookup"
                    ),
                ),
                search_tool=search_tool,
                fetch_tool=fetch_tool,
                max_sources=2,
            )
            used_searches += 1
            used_fetches += len(bundle.sources)
            sources.update({item.source_id: item for item in bundle.sources})
            record: dict[str, Any] = {
                "slot_id": slot.slot_id,
                "query": query.model_dump(mode="json"),
                "source_ids": [item.source_id for item in bundle.sources],
                "failures": bundle.failures,
                "candidates": [],
            }
            for source in bundle.sources:
                response = _accounted_structured(
                    runtime=runtime,
                    schema=_schema_variant(
                        SlotFactCandidate, f"{slot.slot_id}_{source.source_id}"
                    ),
                    prompt=(
                        "Extract at most one fact for this RequiredFactSlot from this "
                        "canonical fetched passage. Return an empty/low-confidence value "
                        "only when the passage does not establish the slot. The exact_quote "
                        "must be a continuous literal substring. Never use outside knowledge.\n\n"
                        f"Slot: {json.dumps(slot.model_dump(mode='json'), ensure_ascii=False)}\n"
                        f"Gap: {json.dumps(gap.model_dump(mode='json'), ensure_ascii=False)}\n"
                        f"Source: {json.dumps(source.model_dump(mode='json'), ensure_ascii=False)}"
                    )[:11_000],
                    label=f"tongagent.permissive.fact_gap.extract.{slot.slot_id}",
                    stage="evidence_selection",
                )
                try:
                    candidate = (
                        SlotFactCandidate.model_validate(response) if response else None
                    )
                except Exception:
                    candidate = None
                if candidate is None or candidate.source_id != source.source_id:
                    record["candidates"].append(
                        {"source_id": source.source_id, "status": "no_valid_candidate"}
                    )
                    continue
                quote = normalize_evidence_text(candidate.exact_quote)
                if not quote or quote not in normalize_evidence_text(source.passage):
                    record["candidates"].append(
                        {"source_id": source.source_id, "status": "noncanonical_quote"}
                    )
                    continue
                raw_fact = TypedFact(
                    fact_id=f"RF{len(repairs) + 1}",
                    subquestion_id=slot.subquestion_id or "repair",
                    fact_type=candidate.fact_type,
                    value=candidate.value,
                    unit=candidate.unit,
                    entity=candidate.entity,
                    attribute=candidate.attribute,
                    relation=candidate.relation,
                    qualifier={
                        **candidate.qualifiers,
                        "exact_quotes": [quote],
                        "slot_id": slot.slot_id,
                        "complete_list": candidate.qualifiers.get(
                            "complete_list", False
                        ),
                    },
                    source_ids=[source.source_id],
                    verification_status="verified",
                    raw_text=quote,
                )
                normalized = _normalise_fact_value(raw_fact)
                if (
                    normalized is None
                    or match_slots([slot], [normalized])[0].status != "satisfied"
                ):
                    record["candidates"].append(
                        {"source_id": source.source_id, "status": "slot_mismatch"}
                    )
                    continue
                claim_text = (
                    f"{slot.entity or 'The source'} {slot.attribute.replace('_', ' ')} "
                    f"is {normalized.value}."
                )
                try:
                    evidence = runtime.research_budget.record_evidence(
                        source_id=source.source_id,
                        claim=claim_text,
                        quote=quote,
                        stance="supports",
                    )
                except Exception as exc:
                    record["candidates"].append(
                        {
                            "source_id": source.source_id,
                            "status": "registration_failed",
                            "reason": type(exc).__name__,
                        }
                    )
                    continue
                accepted = normalized.model_copy(
                    update={"claim_ids": [str(evidence["claim"]["claim_id"])]}
                )
                repairs.append(accepted)
                record["candidates"].append(
                    {
                        "source_id": source.source_id,
                        "status": "accepted",
                        "fact_id": accepted.fact_id,
                    }
                )
            trace.append(record)
            if any(item.qualifier.get("slot_id") == slot.slot_id for item in repairs):
                break
    return repairs, trace


def _schema_variant(model: type[BaseModel], suffix: str) -> type[BaseModel]:
    return create_model(f"{model.__name__}_{suffix}", __base__=model)


def _accounted_structured(
    *,
    runtime: PreparedRuntime,
    schema: type[BaseModel],
    prompt: str,
    label: str,
    stage: Literal["planner", "research_step", "evidence_selection", "final_synthesis"],
) -> BaseModel | None:
    reservation = runtime.middleware.reserve_external_model_call(
        label=label,
        request_payload={
            "prompt": prompt,
            "response_schema": schema.model_json_schema(),
        },
        stage=stage,
    )
    try:
        response = (
            runtime.middleware.model_for_stage(runtime.model, stage)
            .with_structured_output(schema, include_raw=True)
            .invoke(prompt)
        )
    except BudgetExceeded:
        runtime.middleware.cancel_external_model_call(reservation, label=label)
        raise
    except Exception as exc:
        runtime.middleware.cancel_external_model_call(reservation, label=label)
        runtime.trace.record(
            "permissive_model_call_failed",
            label=label,
            exception_type=type(exc).__name__,
        )
        return None
    raw = response.get("raw") if isinstance(response, Mapping) else response
    runtime.middleware.record_external_model_response(
        raw, label=label, reservation=reservation
    )
    try:
        parsed = response.get("parsed") if isinstance(response, Mapping) else response
        return schema.model_validate(parsed)
    except Exception as exc:
        runtime.trace.record(
            "permissive_model_output_invalid",
            label=label,
            exception_type=type(exc).__name__,
            raw_output_summary=_safe_summary(response),
        )
        return None


def _safe_summary(value: Any) -> str:
    safe = sanitize_trace_value(value)
    try:
        text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(safe)
    return " ".join(text.split())[:480]


def _fallback_plan(task: EvalTask) -> PermissivePlan:
    return PermissivePlan(
        objective=task.question,
        subquestions=[
            PermissiveSubquestion(
                id="SQ1",
                question=task.question,
                task_type=cast(TaskType, classify_query_task_type(task.question)),
                required_for_final_answer=True,
            )
        ],
    )


def _validated_plan(
    candidate: BaseModel | None, task: EvalTask
) -> PermissivePlan | None:
    if candidate is None:
        return None
    try:
        plan = PermissivePlan.model_validate(candidate)
    except Exception:
        return None
    unique: set[str] = set()
    repaired: list[PermissiveSubquestion] = []
    for index, item in enumerate(plan.subquestions[:3], start=1):
        subquestion_id = item.id.strip() or f"SQ{index}"
        if subquestion_id in unique:
            subquestion_id = f"SQ{index}"
        unique.add(subquestion_id)
        repaired.append(item.model_copy(update={"id": subquestion_id}))
    return plan.model_copy(update={"subquestions": repaired}) if repaired else None


def _plan(*, runtime: PreparedRuntime, task: EvalTask) -> tuple[PermissivePlan, bool]:
    prompt = f"""Create 1-3 atomic required web-research subquestions for this task.
Do not place an entire multi-hop calculation in one search query. For enumerations
prefer a list, overview, discography, or timeline lookup. Return the schema only.

Question: {task.question}"""
    first = _validated_plan(
        _accounted_structured(
            runtime=runtime,
            schema=PermissivePlan,
            prompt=prompt,
            label="tongagent.permissive.planner.initial",
            stage="planner",
        ),
        task,
    )
    if first is not None:
        return first, False
    repaired = _validated_plan(
        _accounted_structured(
            runtime=runtime,
            schema=_schema_variant(PermissivePlan, "Repair"),
            prompt=(
                "Return only a valid permissive research plan matching the required "
                f"schema for this question: {task.question}"
            ),
            label="tongagent.permissive.planner.repair",
            stage="planner",
        ),
        task,
    )
    return (repaired or _fallback_plan(task)), True


def _required_fact_slots(
    *, runtime: PreparedRuntime, task: EvalTask, plan: PermissivePlan
) -> tuple[list[RequiredFactSlot], bool]:
    """Plan answer inputs before retrieval, with a deterministic fallback."""

    plan_payload = plan.model_dump(mode="json")
    response = _accounted_structured(
        runtime=runtime,
        schema=RequiredFactSlotPlan,
        prompt=(
            "Identify the source-grounded facts needed to answer this question. "
            "Return RequiredFactSlot objects, never values or answers. Make date "
            "differences two slots, and make list/count requirements explicit about "
            "completeness, filters, time range, and de-duplication.\n\n"
            f"Question: {task.question}\nResearch plan: "
            + json.dumps(plan_payload, ensure_ascii=False)
        )[:10_000],
        label="tongagent.permissive.required_fact_slots.initial",
        stage="planner",
    )
    try:
        parsed = RequiredFactSlotPlan.model_validate(response) if response else None
    except Exception:
        parsed = None
    if parsed is None:
        repair = _accounted_structured(
            runtime=runtime,
            schema=_schema_variant(RequiredFactSlotPlan, "Repair"),
            prompt=(
                "Return only valid RequiredFactSlot objects for this question and "
                "research plan. Do not return fact values.\n"
                f"Question: {task.question}\nPlan: "
                + json.dumps(plan_payload, ensure_ascii=False)
            )[:8_000],
            label="tongagent.permissive.required_fact_slots.repair",
            stage="planner",
        )
        try:
            parsed = RequiredFactSlotPlan.model_validate(repair) if repair else None
        except Exception:
            parsed = None
    if parsed is not None:
        unique: set[str] = set()
        plan_ids = {item.id for item in plan.subquestions}
        slots: list[RequiredFactSlot] = []
        for index, slot in enumerate(parsed.slots, start=1):
            slot_id = slot.slot_id.strip() or f"slot-{index}"
            if slot_id in unique:
                slot_id = f"slot-{index}"
            unique.add(slot_id)
            slots.append(
                slot.model_copy(
                    update={
                        "slot_id": slot_id,
                        "subquestion_id": (
                            slot.subquestion_id
                            if slot.subquestion_id in plan_ids
                            else None
                        ),
                    }
                )
            )
        if slots:
            return slots, False
    return (
        fallback_slots(
            question=task.question,
            subquestions=[item.model_dump(mode="json") for item in plan.subquestions],
        ),
        True,
    )


def _query_decision(
    *,
    runtime: PreparedRuntime,
    subquestion: PermissiveSubquestion,
    round_number: int,
    previous: ResearchBundle | None,
) -> ResearchQueryDecision:
    previous_summary = (
        "No previous research round."
        if previous is None
        else json.dumps(
            {
                "sources": [item.model_dump(mode="json") for item in previous.sources],
                "failures": previous.failures,
            },
            ensure_ascii=False,
        )[:3_000]
    )
    schema = _schema_variant(ResearchQueryDecision, f"{subquestion.id}_R{round_number}")
    response = _accounted_structured(
        runtime=runtime,
        schema=schema,
        prompt=f"""Generate one short public-web search query for this atomic subquestion.
Use an entity plus one attribute. Do not output URLs, a final answer, or a full
reasoning chain. This is research round {round_number} of at most two.

Subquestion: {subquestion.question}
Task type: {subquestion.task_type}
Previous round: {previous_summary}""",
        label=f"tongagent.permissive.query.{subquestion.id}.r{round_number}",
        stage="research_step",
    )
    if response is None:
        return ResearchQueryDecision(
            query=subquestion.question, needs_second_round=False
        )
    return ResearchQueryDecision.model_validate(response)


def _fallback_note(
    subquestion: PermissiveSubquestion,
    bundles: Sequence[ResearchBundle],
) -> ResearchNote:
    sources = [source for bundle in bundles for source in bundle.sources]
    passages = [source.passage for source in sources]
    candidates = [
        f"Source {source.source_id} provides retrieved context for {subquestion.question}"
        for source in sources[:2]
    ]
    return ResearchNote(
        subquestion_id=subquestion.id,
        claim_candidates=candidates,
        source_ids=[source.source_id for source in sources],
        supporting_passages=passages[:3],
        unresolved_points=[] if sources else ["No successfully fetched source"],
        confidence="medium" if sources else "low",
    )


def _note(
    *,
    runtime: PreparedRuntime,
    subquestion: PermissiveSubquestion,
    bundles: Sequence[ResearchBundle],
) -> ResearchNote:
    sources = [source for bundle in bundles for source in bundle.sources]
    schema = _schema_variant(ResearchNote, subquestion.id)
    response = _accounted_structured(
        runtime=runtime,
        schema=schema,
        prompt=(
            "Create a research note from only these fetched source passages. Every "
            "claim candidate must name at least one supplied source ID. Do not invent "
            "facts or URLs. Also emit TypedFact objects only for explicit values in "
            "the supplied passages: preserve raw_text, source_ids, a precise type, "
            "and a value; include entity, attribute, relation, and qualifiers; leave "
            "fact_id and claim_ids empty because code binds them after verification. "
            "Never infer an unstated unit or list item.\n\n"
            f"Subquestion: {subquestion.question}\n"
            + json.dumps(
                [source.model_dump(mode="json") for source in sources],
                ensure_ascii=False,
            )
        )[:8_000],
        label=f"tongagent.permissive.note.{subquestion.id}",
        stage="evidence_selection",
    )
    if response is None:
        return _fallback_note(subquestion, bundles)
    note = ResearchNote.model_validate(response)
    valid_ids = {source.source_id for source in sources}
    accepted_ids = [item for item in note.source_ids if item in valid_ids]
    if sources and not accepted_ids:
        # A malformed or hallucinated source ID must not turn otherwise usable
        # fetched material into an artificial missing-SQ failure.  Preserve a
        # conservative code-generated note; its claim candidates deliberately
        # say only that the retrieved source provides context.
        return _fallback_note(subquestion, bundles)
    claims = [item for item in note.claim_candidates if item.strip() and accepted_ids]
    passages = [
        item
        for item in note.supporting_passages
        if item in {s.passage for s in sources}
    ]
    return note.model_copy(
        update={
            "subquestion_id": subquestion.id,
            "source_ids": list(dict.fromkeys(accepted_ids)),
            "claim_candidates": claims,
            "supporting_passages": passages,
        }
    )


def _fallback_draft(notes: Sequence[ResearchNote]) -> DraftAnswer:
    claims: list[DraftClaim] = []
    for note in notes:
        for index, text in enumerate(note.claim_candidates[:2], start=1):
            if len(normalize_evidence_text(text)) < 12 or not note.source_ids:
                continue
            claims.append(
                DraftClaim(
                    claim_id=f"D{len(claims) + 1}",
                    text=normalize_evidence_text(text),
                    source_ids=list(note.source_ids),
                    critical_for_final_answer=index == 1,
                    subquestion_id=note.subquestion_id,
                )
            )
    return DraftAnswer(
        claims=claims,
        reasoning_steps=[],
        proposed_answer=None,
        confidence="low",
        missing_information=[
            note.subquestion_id for note in notes if not note.source_ids
        ],
    )


def _draft(
    *, runtime: PreparedRuntime, task: EvalTask, notes: Sequence[ResearchNote]
) -> DraftAnswer:
    response = _accounted_structured(
        runtime=runtime,
        schema=DraftAnswer,
        prompt=(
            "Produce a source-grounded draft answer. Each claim needs one or more "
            "existing source IDs and its subquestion ID. Mark claims needed for the "
            "final answer as critical. Include intermediate values for calculations. "
            "Never use model memory or invent a source.\n\n"
            f"Question: {task.question}\nResearch notes:\n"
            + json.dumps(
                [item.model_dump(mode="json") for item in notes], ensure_ascii=False
            )
        )[:12_000],
        label="tongagent.permissive.draft",
        stage="final_synthesis",
    )
    if response is None:
        return _fallback_draft(notes)
    draft = DraftAnswer.model_validate(response)
    valid_by_sq = {note.subquestion_id: set(note.source_ids) for note in notes}
    claims = []
    for item in draft.claims:
        allowed_source_ids = valid_by_sq.get(item.subquestion_id, set())
        source_ids = [
            source for source in item.source_ids if source in allowed_source_ids
        ]
        if source_ids:
            claims.append(item.model_copy(update={"source_ids": source_ids}))
    if not claims and any(note.source_ids for note in notes):
        return _fallback_draft(notes)
    return draft.model_copy(update={"claims": claims})


def _answer_plan(
    *,
    runtime: PreparedRuntime,
    task: EvalTask,
    facts: Sequence[TypedFact],
) -> AnswerPlan | None:
    """Ask for a constrained plan, never an expression or computed answer."""

    if not facts:
        return None
    fact_payload = [item.model_dump(mode="json") for item in facts]
    prompt = (
        "Create an AnswerPlan using only the supplied Typed Facts. Choose one "
        "whitelisted operation and list only existing fact IDs. Do not calculate "
        "the answer, write code, use a URL, or introduce a fact. For filter_and_count "
        "use parameters.field plus optional gte_fact_id/lte_fact_id. For compare use "
        "parameters.direction of equals, greater_than, or less_than. For date "
        "differences list start then end fact IDs. Return the schema only.\n\n"
        f"Question: {task.question}\nTyped Facts:\n"
        + json.dumps(fact_payload, ensure_ascii=False)
    )[:14_000]
    response = _accounted_structured(
        runtime=runtime,
        schema=AnswerPlan,
        prompt=prompt,
        label="tongagent.permissive.answer_plan.initial",
        stage="final_synthesis",
    )
    try:
        return AnswerPlan.model_validate(response) if response is not None else None
    except Exception:
        pass
    repair = _accounted_structured(
        runtime=runtime,
        schema=_schema_variant(AnswerPlan, "Repair"),
        prompt=(
            "Return only one valid AnswerPlan. Existing fact IDs are: "
            f"{', '.join(item.fact_id for item in facts)}. Do not calculate or add facts. "
            f"Question: {task.question}"
        ),
        label="tongagent.permissive.answer_plan.repair",
        stage="final_synthesis",
    )
    try:
        return AnswerPlan.model_validate(repair) if repair is not None else None
    except Exception:
        return None


def _meaningful_tokens(value: str) -> set[str]:
    return {
        item.casefold()
        for item in _WORD.findall(value)
        if len(item) >= 3
        and item.casefold() not in {"the", "and", "with", "from", "that"}
    }


def _quote_for_claim(claim: str, source: ResearchSource) -> tuple[str | None, bool]:
    tokens = _meaningful_tokens(claim)
    claim_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", claim))
    candidates = [
        item.strip()
        for item in _SENTENCE.split(source.passage)
        if len(item.strip()) >= 12
    ]
    scored = sorted(
        candidates,
        key=lambda item: (-len(tokens.intersection(_meaningful_tokens(item))), item),
    )
    if not scored:
        return None, False
    best = _bounded_literal_quote(scored[0], tokens)
    source_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", best))
    contradicts = bool(
        claim_numbers
        and source_numbers
        and not claim_numbers.intersection(source_numbers)
    )
    if not tokens.intersection(_meaningful_tokens(best)):
        return None, contradicts
    return best, contradicts


def _bounded_literal_quote(candidate: str, claim_tokens: set[str]) -> str:
    """Keep an exact, relevant page span within EvidenceGraph quote limits.

    Some HTML-to-text pages contain navigation or tables without sentence-ending
    punctuation.  They become one enormous ``_SENTENCE`` fragment even though
    they are valid page text.  Taking a word-boundary window retains a literal
    continuous excerpt that the graph can revalidate against the canonical page;
    it does not summarize, invent, or paraphrase source content.
    """

    text = normalize_evidence_text(candidate)
    if len(text) <= MAX_QUOTE_CHARS:
        return text
    locations = [
        match.start()
        for token in sorted(claim_tokens)
        for match in re.finditer(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE)
    ]
    anchor = min(locations) if locations else 0
    start = max(0, anchor - MAX_QUOTE_CHARS // 3)
    end = min(len(text), start + MAX_QUOTE_CHARS)
    if start:
        boundary = text.find(" ", start)
        start = boundary + 1 if boundary >= 0 and boundary < end else start
    if end < len(text):
        boundary = text.rfind(" ", start, end)
        end = boundary if boundary > start else end
    excerpt = text[start:end].strip()
    # The quote invariant has a positive lower bound.  This fallback is still
    # a literal contiguous prefix, used only for pathological whitespace.
    return excerpt or text[:MAX_QUOTE_CHARS].strip()


def _verify(
    *,
    runtime: PreparedRuntime,
    draft: DraftAnswer,
    sources: Mapping[str, ResearchSource],
) -> list[VerifiedClaim]:
    verified: list[VerifiedClaim] = []
    for claim in draft.claims:
        support: list[tuple[ResearchSource, str]] = []
        contradictions: list[tuple[ResearchSource, str]] = []
        for source_id in claim.source_ids:
            source = sources.get(source_id)
            if source is None:
                continue
            quote, contradicts = _quote_for_claim(claim.text, source)
            if quote is None:
                continue
            if contradicts:
                contradictions.append((source, quote))
            else:
                support.append((source, quote))
        if support and contradictions:
            status: VerifiedStatus = "contested"
            selected = support + contradictions
            explanation = "Fetched sources contain incompatible numeric or date values."
        elif support:
            status = "verified"
            selected = support
            explanation = (
                "A continuous exact quote was found in a canonical fetched page."
            )
        elif any(source_id in sources for source_id in claim.source_ids):
            status = "partially_supported"
            selected = []
            explanation = "A fetched source is associated, but no precise continuous quote matched."
        else:
            status = "unsupported"
            selected = []
            explanation = "No canonical fetched source supports this claim."
        canonical_claim_id: str | None = None
        registration_failures: list[dict[str, str]] = []
        # Graph registration is intentionally post-hoc.  A quote was selected
        # from the page passage, and EvidenceGraphStore rechecks it against its
        # cached canonical page before assigning C#/E# IDs.
        registered: list[tuple[ResearchSource, str]] = []
        for source, quote in selected:
            try:
                # Verification runs after all SQs have been researched.  The
                # graph nevertheless owns claims by their original SQ, not by
                # whichever SQ happened to be active at workflow completion.
                runtime.research_budget.activate_subquestion(claim.subquestion_id)
                record = runtime.research_budget.record_evidence(
                    source_id=source.source_id,
                    claim=claim.text,
                    quote=quote,
                    stance="contradicts"
                    if (source, quote) in contradictions
                    else "supports",
                    claim_id=canonical_claim_id or "",
                )
            except Exception as exc:
                invariant = str(exc).splitlines()[0][:500]
                reason = (
                    "quote_too_long"
                    if len(normalize_evidence_text(quote)) > MAX_QUOTE_CHARS
                    else "canonical_registration_rejected"
                )
                failure = {
                    "category": reason,
                    "exception_type": type(exc).__name__,
                    "invariant": invariant,
                    "source_id": source.source_id,
                    "claim_id": claim.claim_id,
                    "quote_chars": str(len(normalize_evidence_text(quote))),
                    "safe_traceback": sanitize_trace_value(
                        traceback.format_exc(limit=12)
                    )[:2_000],
                }
                registration_failures.append(failure)
                runtime.trace.record(
                    "permissive_evidence_registration_failed",
                    **failure,
                )
                continue
            canonical_claim_id = str(record["claim"]["claim_id"])
            registered.append((source, quote))
        # A verifier result cannot claim exact verification unless the shared
        # EvidenceGraphStore also accepted the literal quote against its
        # canonical page cache.  This keeps the graph the final source of
        # truth without using it as a research-time gate.
        if selected and not registered:
            status = "partially_supported"
            explanation = (
                "A source passage matched, but canonical evidence registration failed."
            )
        verified.append(
            VerifiedClaim(
                claim_id=claim.claim_id,
                status=status,
                source_ids=[source.source_id for source, _ in registered]
                or list(claim.source_ids),
                exact_quotes=[quote for _, quote in registered],
                explanation=explanation,
                canonical_claim_id=canonical_claim_id,
                registration_failures=registration_failures,
            )
        )
    return verified


def _finalization_decision(
    *,
    draft: DraftAnswer,
    notes: Sequence[ResearchNote],
    verified: Sequence[VerifiedClaim],
    config: ResolvedConfig,
    required_subquestion_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Make one auditable high/low/abstain finalization decision."""

    statuses = {item.claim_id: item for item in verified}
    critical = [item for item in draft.claims if item.critical_for_final_answer]
    unsupported = [
        item.claim_id
        for item in critical
        if statuses.get(item.claim_id) is None
        or statuses[item.claim_id].status in {"unsupported", "contested"}
    ]
    weak = [
        item.claim_id
        for item in critical
        if statuses.get(item.claim_id) is not None
        and statuses[item.claim_id].status == "partially_supported"
    ]
    missing_sq = [note.subquestion_id for note in notes if not note.source_ids]
    researched_ids = {note.subquestion_id for note in notes}
    required_ids = set(required_subquestion_ids or researched_ids)
    unresearched_sq = sorted(required_ids - researched_ids)
    mapped_critical = [item.claim_id for item in critical if not item.source_ids]
    all_high = (
        bool(critical)
        and not unsupported
        and not weak
        and not missing_sq
        and not unresearched_sq
        and not mapped_critical
        and bool(draft.proposed_answer)
    )
    low = (
        bool(critical)
        and bool(weak)
        and not unsupported
        and not missing_sq
        and not unresearched_sq
        and not mapped_critical
        and any(
            statuses.get(item.claim_id) is not None
            and statuses[item.claim_id].status == "verified"
            for item in critical
        )
        and bool(draft.proposed_answer)
        and bool(draft.reasoning_steps)
        and not draft.missing_information
        and config.permissive_workflow.allow_low_confidence_answer
    )
    if not config.permissive_workflow.enable_posthoc_verifier:
        all_high = (
            bool(draft.proposed_answer) and not missing_sq and not unresearched_sq
        )
        low = False
        unsupported = []
        weak = []
    status = "answered" if all_high else "answered_low_confidence" if low else "abstain"
    return {
        "schema_version": 1,
        "answer_status": status,
        "confidence": "high" if all_high else "low" if low else "none",
        "all_required_sq_researched": not unresearched_sq,
        "missing_source_subquestions": sorted(missing_sq),
        "unresearched_subquestions": unresearched_sq,
        "critical_claims": [item.claim_id for item in critical],
        "verified_critical_claims": [
            item.claim_id
            for item in critical
            if statuses.get(item.claim_id) is not None
            and statuses[item.claim_id].status == "verified"
        ],
        "partially_supported_critical_claims": weak,
        "unsupported_or_contested_critical_claims": sorted(unsupported),
        "unmapped_critical_claims": mapped_critical,
        "low_confidence_reason": (
            "Core claims include canonical sources but one or more only have "
            "partial exact-quote verification."
            if low
            else None
        ),
    }


def _finalize(
    *,
    task: EvalTask,
    draft: DraftAnswer,
    notes: Sequence[ResearchNote],
    verified: Sequence[VerifiedClaim],
    config: ResolvedConfig,
    required_subquestion_ids: Sequence[str] | None = None,
) -> tuple[str, str, list[str]]:
    decision = _finalization_decision(
        draft=draft,
        notes=notes,
        verified=verified,
        config=config,
        required_subquestion_ids=required_subquestion_ids,
    )
    if decision["answer_status"] == "abstain":
        missing = sorted(
            set(
                list(decision["missing_source_subquestions"])
                + list(decision["unresearched_subquestions"])
                + list(decision["unsupported_or_contested_critical_claims"])
                + list(decision["unmapped_critical_claims"])
            )
        )
        return (
            "## Answer\nINSUFFICIENT_EVIDENCE\nABSTAIN\n"
            f"- Missing or unsupported: {', '.join(missing) or 'verified core evidence'}\n"
            "FINAL_ANSWER: ABSTAIN",
            "abstain",
            list(decision["unsupported_or_contested_critical_claims"]),
        )
    citations = sorted(
        {
            source_id
            for claim in draft.claims
            if claim.critical_for_final_answer
            for source_id in claim.source_ids
        }
    )
    answer = str(draft.proposed_answer).strip()
    low = decision["answer_status"] == "answered_low_confidence"
    verification_lines = [
        f"- Fully verified core claims: {', '.join(decision['verified_critical_claims']) or 'none'}",
        f"- Partially supported core claims: {', '.join(decision['partially_supported_critical_claims']) or 'none'}",
    ]
    if low:
        verification_lines.append(
            f"- Low confidence reason: {decision['low_confidence_reason']}"
        )
    return (
        "## Answer\n"
        f"{answer}\n\n"
        f"Sources: {' '.join(f'[{item}]' for item in citations)}\n"
        f"Confidence: {decision['confidence']}.\n"
        + "\n".join(verification_lines)
        + f"\nFINAL_ANSWER: {answer}",
        "answer_low_confidence" if low else "answer",
        [],
    )


def _typed_finalization_decision(
    *,
    notes: Sequence[ResearchNote],
    facts: Sequence[TypedFact],
    plan: AnswerPlan | None,
    execution: AnswerExecution,
    config: ResolvedConfig,
    required_subquestion_ids: Sequence[str],
) -> dict[str, Any]:
    """Route solely from executor output and source-bound fact statuses."""

    fact_by_id = {item.fact_id: item for item in facts}
    required = (
        [fact_by_id[item] for item in plan.required_fact_ids if item in fact_by_id]
        if plan
        else []
    )
    missing_sq = sorted(item.subquestion_id for item in notes if not item.source_ids)
    noted_ids = {item.subquestion_id for item in notes}
    unresearched = sorted(set(required_subquestion_ids) - noted_ids)
    statuses = {item.verification_status for item in required}
    all_verified = bool(required) and statuses == {"verified"}
    low = (
        bool(required)
        and "verified" in statuses
        and statuses.issubset({"verified", "partially_supported"})
        and config.permissive_workflow.allow_low_confidence_answer
    )
    executable = execution.status == "success"
    answer_status = (
        "answered"
        if executable and not missing_sq and not unresearched and all_verified
        else "answered_low_confidence"
        if executable and not missing_sq and not unresearched and low
        else "abstain"
    )
    return {
        "schema_version": 1,
        "answer_status": answer_status,
        "confidence": "high"
        if answer_status == "answered"
        else "low"
        if answer_status == "answered_low_confidence"
        else "none",
        "all_required_sq_researched": not unresearched,
        "missing_source_subquestions": missing_sq,
        "unresearched_subquestions": unresearched,
        "required_fact_ids": plan.required_fact_ids if plan else [],
        "required_fact_statuses": {
            item.fact_id: item.verification_status for item in required
        },
        "unsupported_or_contested_fact_ids": [
            item.fact_id
            for item in required
            if item.verification_status in {"unsupported", "contested"}
        ],
        "missing_answer_plan": plan is None,
        "execution_failure_reason": execution.failure_reason,
        "calculation_executed": executable,
        "low_confidence_reason": (
            "One or more calculation facts are partially supported but every "
            "required fact has a canonical source and no fact is unsupported."
            if answer_status == "answered_low_confidence"
            else None
        ),
    }


def _typed_finalize(
    *,
    facts: Sequence[TypedFact],
    plan: AnswerPlan | None,
    execution: AnswerExecution,
    notes: Sequence[ResearchNote],
    config: ResolvedConfig,
    required_subquestion_ids: Sequence[str],
) -> tuple[str, str, dict[str, Any]]:
    decision = _typed_finalization_decision(
        notes=notes,
        facts=facts,
        plan=plan,
        execution=execution,
        config=config,
        required_subquestion_ids=required_subquestion_ids,
    )
    if decision["answer_status"] == "abstain":
        missing = sorted(
            set(
                list(decision["missing_source_subquestions"])
                + list(decision["unresearched_subquestions"])
                + list(decision["unsupported_or_contested_fact_ids"])
            )
        )
        reason = decision["execution_failure_reason"] or (
            "missing_required_typed_fact"
            if decision["missing_answer_plan"]
            else "incomplete_evidence"
        )
        return (
            "## Answer\nINSUFFICIENT_EVIDENCE\nABSTAIN\n"
            f"- Missing or unsupported: {', '.join(missing) or reason}\n"
            f"- Executor: {reason}\nFINAL_ANSWER: ABSTAIN",
            "abstain",
            decision,
        )
    fact_by_id = {item.fact_id: item for item in facts}
    used = [fact_by_id[item] for item in execution.fact_ids if item in fact_by_id]
    citations = sorted({source for item in used for source in item.source_ids})
    fact_lines = [
        f"- {item.fact_id}: {item.value} {item.unit or ''}".rstrip()
        + f" [sources: {', '.join(item.source_ids)}]"
        for item in used
    ]
    answer = str(execution.answer_text or "").strip()
    low = decision["answer_status"] == "answered_low_confidence"
    return (
        "## Answer\n"
        f"{answer}\n\n"
        f"Sources: {' '.join(f'[{item}]' for item in citations)}\n"
        f"Confidence: {decision['confidence']}.\n"
        "Calculation facts:\n"
        + "\n".join(fact_lines)
        + (
            f"\n- Low confidence reason: {decision['low_confidence_reason']}"
            if low
            else ""
        )
        + f"\nFINAL_ANSWER: {answer}",
        "answer_low_confidence" if low else "answer",
        decision,
    )


def _citations(runtime: PreparedRuntime) -> list[Citation]:
    snapshot = runtime.research_budget.snapshot()
    claims = {
        str(item.get("claim_id", "")): item for item in snapshot.get("claims", [])
    }
    return [
        Citation(
            citation_id=str(item["evidence_id"]),
            source_id=str(item["source_id"]),
            url=str(item["url"]),
            title=str(item.get("title", "")) or None,
            quote=str(item["quote"]),
            claim=str(claims.get(str(item["claim_id"]), {}).get("text", "")) or None,
            metadata={"claim_id": str(item["claim_id"]), "stance": str(item["stance"])},
        )
        for item in snapshot.get("evidence_units", [])
        if isinstance(item, Mapping) and item.get("evidence_id") and item.get("url")
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fact_gap_runtime_limits(
    resolved_config: ResolvedConfig,
) -> tuple[BudgetLimits, EffortPolicy | None]:
    """Add only the declared repair allowance to TongAgent's retrieval budget."""

    settings = resolved_config.permissive_workflow
    if not settings.enable_fact_gap_retrieval:
        return resolved_config.budget, None
    base = resolved_config.budget
    limits = base.model_copy(
        update={
            "max_search_calls": base.max_search_calls
            + settings.repair_max_search_calls,
            "max_fetch_calls": base.max_fetch_calls + settings.repair_max_fetch_calls,
            "max_total_tool_calls": base.max_total_tool_calls
            + settings.repair_max_search_calls
            + settings.repair_max_fetch_calls,
        }
    )
    return (
        limits,
        EffortPolicy(
            name="high",
            max_searches=limits.max_search_calls,
            max_fetches=limits.max_fetch_calls,
            min_successful_sources=0,
            max_results_per_search=limits.max_results_per_search,
            max_chars_per_page=limits.max_page_chars,
            max_output_tokens=resolved_config.model.max_output_tokens or 1,
            max_subquestions=3,
            require_reviewer=False,
        ),
    )


def run_permissive_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> RunResult:
    """Run PLAN → RESEARCH_SQ → NOTE → DRAFT → VERIFY → FINALIZE."""

    artifact_directory = Path(resolved_config.artifact_directory).expanduser()
    native_directory = artifact_directory / "native" / "tongagent"
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    trace = TraceCollector()
    effective_limits, fact_gap_policy = _fact_gap_runtime_limits(resolved_config)
    execution_budget = ExecutionBudget(effective_limits)
    run_id = f"tongagent-{task.id}-{uuid.uuid4().hex}"
    runtime: PreparedRuntime | None = None
    caught: Exception | None = None
    plan: PermissivePlan | None = None
    notes: list[ResearchNote] = []
    bundles: dict[str, list[ResearchBundle]] = {}
    verified: list[VerifiedClaim] = []
    typed_facts: list[TypedFact] = []
    typed_fact_failures: list[dict[str, str]] = []
    required_slots: list[RequiredFactSlot] = []
    initial_coverage: list[SlotCoverage] = []
    final_coverage: list[SlotCoverage] = []
    fact_gaps: list[FactGap] = []
    repair_trace: list[dict[str, Any]] = []
    answer_plan: AnswerPlan | None = None
    answer_execution = AnswerExecution(
        status="abstain", failure_reason="workflow did not reach answer execution"
    )
    draft = DraftAnswer()
    final_answer = "FINAL_ANSWER: ABSTAIN"
    final_status = "abstain"
    removed_claims: list[str] = []
    finalization_decision: dict[str, Any] = {
        "schema_version": 1,
        "answer_status": "abstain",
        "confidence": "none",
        "reason": "workflow did not reach finalization",
    }
    try:
        runtime = prepare_runtime(
            task,
            resolved_config,
            system_id="tongagent",
            execution_budget=execution_budget,
            trace=trace,
            injected_backend=fixture_backend,
            injected_model=model,
            semantic_policy=fact_gap_policy,
            semantic_strategy="fixed",
            enable_tongagent_evidence_state=True,
            enable_tongagent_token_control=True,
            enable_tongagent_context_compaction=True,
            search_query_normalizer=None,
            reserve_final_synthesis=False,
            allow_retrieval_extension=fact_gap_policy is not None,
        )
        trace.record("permissive_phase", phase="PLAN", runtime_mode="permissive")
        plan, planner_repaired = _plan(runtime=runtime, task=task)
        runtime.research_budget.configure_subquestions(
            [item.id for item in plan.subquestions]
        )
        runtime.middleware.configure_token_partitions(
            [item.id for item in plan.subquestions]
        )
        if resolved_config.permissive_workflow.enable_fact_gap_retrieval:
            required_slots, slots_repaired = _required_fact_slots(
                runtime=runtime, task=task, plan=plan
            )
            trace.record(
                "fact_gap_slots_planned",
                slot_count=len(required_slots),
                planner_repaired=slots_repaired,
            )
        all_sources: dict[str, ResearchSource] = {}
        for subquestion in plan.subquestions:
            runtime.research_budget.activate_subquestion(subquestion.id)
            runtime.middleware.activate_token_subquestion(subquestion.id)
            trace.record(
                "permissive_phase", phase="RESEARCH_SQ", subquestion_id=subquestion.id
            )
            local_bundles: list[ResearchBundle] = []
            for round_number in range(1, _MAX_RESEARCH_ROUNDS + 1):
                decision = _query_decision(
                    runtime=runtime,
                    subquestion=subquestion,
                    round_number=round_number,
                    previous=local_bundles[-1] if local_bundles else None,
                )
                bundle = research_query(
                    query=decision.query,
                    task_type=subquestion.task_type,
                    subquestion=subquestion,
                    search_tool=next(
                        tool for tool in runtime.tools if tool.name == "web_search"
                    ),
                    fetch_tool=next(
                        tool for tool in runtime.tools if tool.name == "fetch_url"
                    ),
                    max_sources=2
                    if len(plan.subquestions) >= 3
                    else _MAX_SOURCES_PER_QUERY,
                )
                local_bundles.append(bundle)
                all_sources.update({item.source_id: item for item in bundle.sources})
                trace.record(
                    "permissive_research_bundle",
                    subquestion_id=subquestion.id,
                    round_number=round_number,
                    query_validation_status=bundle.query_validation_status,
                    query_normalizations=bundle.query_normalizations,
                    query_broadened=bundle.broadened,
                    source_count=len(bundle.sources),
                    failure_count=len(bundle.failures),
                )
                # A second, narrow round is allowed either when the model
                # explicitly identifies a missing fact or when round one
                # found no usable source.  It never blocks the next SQ.
                if round_number == _MAX_RESEARCH_ROUNDS or (
                    bundle.sources and not decision.needs_second_round
                ):
                    break
            bundles[subquestion.id] = local_bundles
            trace.record(
                "permissive_phase",
                phase="SAVE_RESEARCH_NOTE",
                subquestion_id=subquestion.id,
            )
            notes.append(
                _note(runtime=runtime, subquestion=subquestion, bundles=local_bundles)
            )
        trace.record("permissive_phase", phase="SYNTHESIZE_DRAFT")
        draft = _draft(runtime=runtime, task=task, notes=notes)
        trace.record("permissive_phase", phase="VERIFY_DRAFT")
        if runtime is not None:
            verified = _verify(runtime=runtime, draft=draft, sources=all_sources)
        typed_facts, verified, typed_fact_failures = _collect_typed_facts(
            notes=notes, draft=draft, verified=verified
        )
        if resolved_config.permissive_workflow.enable_fact_gap_retrieval:
            initial_coverage = match_slots(required_slots, typed_facts)
            fact_gaps = gaps_from_coverage(initial_coverage)
            repaired_facts, repair_trace = _repair_slot_facts(
                runtime=runtime,
                slots=required_slots,
                gaps=fact_gaps,
                sources=all_sources,
                max_searches=resolved_config.permissive_workflow.repair_max_search_calls,
                max_fetches=resolved_config.permissive_workflow.repair_max_fetch_calls,
                max_queries_per_slot=resolved_config.permissive_workflow.repair_max_queries_per_slot,
            )
            typed_facts.extend(repaired_facts)
            final_coverage = match_slots(required_slots, typed_facts)
            trace.record(
                "fact_gap_repair_complete",
                initial_gaps=len(fact_gaps),
                repaired_facts=len(repaired_facts),
                remaining_gaps=sum(
                    item.status != "satisfied" for item in final_coverage
                ),
            )
        # Answer planning is deliberately separated from draft prose.  It is
        # the only final-stage model decision and can select only source-bound
        # Fact IDs; code performs every arithmetic/list operation afterwards.
        eligible_fact_ids = (
            {
                fact_id
                for item in final_coverage
                if item.status == "satisfied"
                for fact_id in item.matching_fact_ids
            }
            if final_coverage
            else {item.fact_id for item in typed_facts}
        )
        eligible_facts = [
            item for item in typed_facts if item.fact_id in eligible_fact_ids
        ]
        if final_coverage and any(
            item.status != "satisfied" for item in final_coverage
        ):
            answer_execution = AnswerExecution(
                status="abstain", failure_reason="required_fact_slots_unresolved"
            )
        else:
            answer_plan = _answer_plan(runtime=runtime, task=task, facts=eligible_facts)
            answer_execution = _execute_answer_plan(
                plan=answer_plan,
                facts=eligible_facts,
                allow_partial=resolved_config.permissive_workflow.allow_low_confidence_answer,
            )
        trace.record("permissive_phase", phase="FINALIZE")
        final_answer, final_status, finalization_decision = _typed_finalize(
            facts=typed_facts,
            plan=answer_plan,
            execution=answer_execution,
            notes=notes,
            config=resolved_config,
            required_subquestion_ids=[item.id for item in plan.subquestions],
        )
        removed_claims = [
            item.claim_id
            for item in verified
            if item.status in {"unsupported", "contested"}
        ]
        trace.record(
            "permissive_finalized",
            answer_status=final_status,
            planner_repaired=planner_repaired,
            verifier_removed_claims=removed_claims,
            typed_fact_count=len(typed_facts),
            answer_operation=answer_plan.operation if answer_plan is not None else None,
            answer_execution_status=answer_execution.status,
        )
    except Exception as exc:
        caught = exc
        trace.record(
            "run_exception",
            phase="permissive",
            exception_type=type(exc).__name__,
        )
    finally:
        finished_at = datetime.now(UTC)
        wall_time = max(0.0, time.perf_counter() - started)
        ledger = runtime.research_budget.snapshot() if runtime is not None else {}
        graph_errors = validate_evidence_graph(ledger) if runtime is not None else []
        verified_count = sum(item.status == "verified" for item in verified)
        critical = [item for item in draft.claims if item.critical_for_final_answer]
        critical_verified = sum(
            next(
                (
                    check.status == "verified"
                    for check in verified
                    if check.claim_id == item.claim_id
                ),
                False,
            )
            for item in critical
        )
        sq_completion = (
            sum(bool(note.source_ids) or bool(note.unresolved_points) for note in notes)
            / len(plan.subquestions)
            if plan is not None and plan.subquestions
            else 0.0
        )
        workflow_metrics: dict[str, Any] = {
            "runtime_mode": "permissive",
            "sq_research_completion_rate": sq_completion,
            "draft_claim_count": len(draft.claims),
            "verified_claim_rate": verified_count / len(draft.claims)
            if draft.claims
            else 0.0,
            "critical_claim_verified_rate": critical_verified / len(critical)
            if critical
            else 0.0,
            "unsupported_claim_rate": sum(
                item.status == "unsupported" for item in verified
            )
            / len(verified)
            if verified
            else 0.0,
            "answer_before_verification": draft.proposed_answer,
            "answer_after_verification": final_answer,
            "verifier_removed_claims": removed_claims,
            "research_rounds_per_sq": {
                key: len(value) for key, value in bundles.items()
            },
            "claim_status_counts": {
                status: sum(item.status == status for item in verified)
                for status in (
                    "verified",
                    "partially_supported",
                    "unsupported",
                    "contested",
                )
            },
            "typed_fact_count": len(typed_facts),
            "typed_fact_failures": typed_fact_failures,
            "answer_operation": answer_plan.operation
            if answer_plan is not None
            else None,
            "answer_execution_status": answer_execution.status,
            "typed_calculation_success": answer_execution.status == "success",
            "required_fact_slot_count": len(required_slots),
            "initially_satisfied_slots": sum(
                item.status == "satisfied" for item in initial_coverage
            ),
            "finally_satisfied_slots": sum(
                item.status == "satisfied" for item in final_coverage
            ),
            "fact_gap_count": len(fact_gaps),
            "fact_gap_repair_count": sum(
                len(item.get("candidates", [])) for item in repair_trace
            ),
            "evidence_graph_errors": graph_errors,
        }
        native_output = {"final_answer": final_answer}
        result = build_run_result(
            run_id=run_id,
            task=task,
            resolved_config=resolved_config,
            system_id="tongagent",
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=wall_time,
            runtime=runtime,
            execution_budget=execution_budget,
            trace=trace,
            native_output=native_output,
            caught=caught,
            evidence_count=len(ledger.get("evidence_units", []))
            if runtime is not None
            else None,
            structural_subquestion_coverage=sq_completion if plan is not None else None,
            final_answer_override=final_answer,
        )
        # ``build_run_result`` is authoritative for deadline/budget terminals.
        # A deterministic ABSTAIN is only a normal partial result when no
        # terminal budget state was observed.  Overwriting a timed-out result
        # while retaining its budget snapshot violates RunResult's contract.
        if caught is None and result.completion_status == CompletionStatus.COMPLETED:
            result = result.model_copy(
                update={
                    "completion_status": (
                        CompletionStatus.PARTIAL
                        if final_status == "abstain"
                        else CompletionStatus.COMPLETED
                    ),
                    "citations": _citations(runtime) if runtime is not None else [],
                    "workflow_metrics": cast("dict[str, Any]", workflow_metrics),
                }
            )
        write_intermediate_artifacts(
            artifact_directory,
            task=task,
            resolved_config=resolved_config,
            runtime=runtime,
            execution_budget=execution_budget,
            trace=trace,
            native_output=native_output,
            result=result,
            caught=caught,
        )
        _write_json(
            native_directory / "research_notes.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "plan": plan.model_dump(mode="json") if plan is not None else None,
                "research_notes": [item.model_dump(mode="json") for item in notes],
                "research_bundles": {
                    key: [item.model_dump(mode="json") for item in value]
                    for key, value in bundles.items()
                },
            },
        )
        _write_json(
            native_directory / "draft_answer.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "draft": draft.model_dump(mode="json"),
            },
        )
        _write_json(
            native_directory / "verified_claims.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "verified_claims": [item.model_dump(mode="json") for item in verified],
            },
        )
        _write_json(
            native_directory / "typed_facts.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "typed_facts": [item.model_dump(mode="json") for item in typed_facts],
                "validation_failures": typed_fact_failures,
            },
        )
        _write_json(
            native_directory / "required_fact_slots.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "required_fact_slots": [
                    item.model_dump(mode="json") for item in required_slots
                ],
            },
        )
        _write_json(
            native_directory / "fact_coverage_report.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "coverage": [item.model_dump(mode="json") for item in initial_coverage],
            },
        )
        _write_json(
            native_directory / "fact_gap_report.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "gaps": [item.model_dump(mode="json") for item in fact_gaps],
                "gap_counts": {
                    status: sum(item.status == status for item in fact_gaps)
                    for status in sorted({item.status for item in fact_gaps})
                },
            },
        )
        _write_json(
            native_directory / "fact_gap_repair_trace.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "repair_budget": {
                    "max_search_calls": resolved_config.permissive_workflow.repair_max_search_calls,
                    "max_fetch_calls": resolved_config.permissive_workflow.repair_max_fetch_calls,
                    "max_queries_per_slot": resolved_config.permissive_workflow.repair_max_queries_per_slot,
                },
                "repairs": repair_trace,
            },
        )
        _write_json(
            native_directory / "fact_conflict_resolution.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "conflicting_slots": [
                    item.model_dump(mode="json")
                    for item in final_coverage
                    if item.status == "conflicting"
                ],
                "resolution": "no conflict promotion; unresolved conflicts remain unusable",
            },
        )
        _write_json(
            native_directory / "fact_gap_final_coverage.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "coverage": [item.model_dump(mode="json") for item in final_coverage],
            },
        )
        _write_json(
            native_directory / "answer_plan.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "answer_plan": answer_plan.model_dump(mode="json")
                if answer_plan
                else None,
            },
        )
        _write_json(
            native_directory / "calculation_trace.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "calculation_trace": answer_execution.calculation_trace,
                "fact_ids": answer_execution.fact_ids,
            },
        )
        _write_json(
            native_directory / "answer_execution.json",
            {
                "schema_version": 1,
                "task_id": task.id,
                "execution": answer_execution.model_dump(mode="json"),
            },
        )
        _write_json(
            native_directory / "evidence_graph.json",
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
            native_directory / "finalization_decision.json",
            {
                **finalization_decision,
                "schema_version": 1,
                "task_id": task.id,
                "final_answer_status": final_status,
                "final_answer": final_answer,
            },
        )
        _write_json(
            native_directory / "permissive_workflow.json",
            {
                "schema_version": 1,
                "runtime_mode": "permissive",
                "phase": "FINALIZE",
                "final_answer_status": final_status,
                "workflow_metrics": workflow_metrics,
                "state_integrity_errors": graph_errors,
                "canonical_artifacts": {
                    "research_notes": "research_notes.json",
                    "draft_answer": "draft_answer.json",
                    "verified_claims": "verified_claims.json",
                    "evidence_graph": "evidence_graph.json",
                    "finalization_decision": "finalization_decision.json",
                    "typed_facts": "typed_facts.json",
                    "answer_plan": "answer_plan.json",
                    "calculation_trace": "calculation_trace.json",
                    "answer_execution": "answer_execution.json",
                    "required_fact_slots": "required_fact_slots.json",
                    "fact_coverage": "fact_coverage_report.json",
                    "fact_gaps": "fact_gap_report.json",
                    "fact_gap_repair_trace": "fact_gap_repair_trace.json",
                    "fact_gap_final_coverage": "fact_gap_final_coverage.json",
                },
            },
        )
        return RunResult.model_validate(result.model_dump(mode="python"))


def preflight_permissive_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Prepare the permissive runtime only; no model or provider is invoked."""

    execution_budget = ExecutionBudget(resolved_config.budget)
    trace = TraceCollector()
    runtime = prepare_runtime(
        task,
        resolved_config,
        system_id="tongagent",
        execution_budget=execution_budget,
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
        "runtime_mode": "permissive",
        "runtime_tools": [tool.name for tool in runtime.tools],
        "model_invocations": 0,
        "agent_constructed": True,
    }


__all__ = [
    "AnswerExecution",
    "AnswerPlan",
    "DraftAnswer",
    "PermissivePlan",
    "PermissiveSubquestion",
    "ResearchBundle",
    "ResearchNote",
    "ResearchQueryDecision",
    "ResearchSource",
    "TypedFact",
    "VerifiedClaim",
    "preflight_permissive_workflow",
    "research_query",
    "run_permissive_workflow",
]
