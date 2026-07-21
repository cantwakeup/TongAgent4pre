"""Deterministic Fact-Gap assessment for permissive research workflows.

The module intentionally has no network/model imports.  It turns a question,
research-plan shape, and source-bound fact-like objects into auditable Slots,
coverage statuses, gaps, and bounded repair-query plans.  Runtime code owns
the actual retrieval and evidence registration.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


FactType = Literal[
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
CoverageStatus = Literal[
    "satisfied",
    "partially_satisfied",
    "missing",
    "ambiguous",
    "conflicting",
    "wrong_entity",
    "wrong_attribute",
    "wrong_type",
    "wrong_unit",
    "wrong_qualifier",
    "incomplete_list",
]

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
_CAPITALIZED = re.compile(
    r"\b(?:[A-Z][A-Za-z.'-]*)(?:\s+(?:[A-Z][A-Za-z.'-]*|of|the|and))*"
)
_STOP = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
    "would",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RequiredFactSlot(_Model):
    slot_id: str
    subquestion_id: str | None = None
    entity: str | None = None
    attribute: str
    relation: str | None = None
    fact_type: FactType
    unit: str | None = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    cardinality: Literal["single", "list"] = "single"
    required_for_final_answer: bool = True


class SlotCoverage(_Model):
    slot_id: str
    status: CoverageStatus
    matching_fact_ids: list[str] = Field(default_factory=list)
    conflicting_fact_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    unit_conversion: dict[str, Any] | None = None


class FactGap(_Model):
    slot_id: str
    status: str
    missing_components: list[str] = Field(default_factory=list)
    existing_fact_ids: list[str] = Field(default_factory=list)
    conflict_fact_ids: list[str] = Field(default_factory=list)
    repair_priority: int = Field(ge=1, le=100)
    repairable: bool


class RepairQuery(_Model):
    slot_id: str
    attempt: Literal[1, 2]
    query: str
    normalized_query: str
    broadened: bool
    reason: str


class SlotFactCandidate(_Model):
    slot_id: str
    fact_type: FactType
    value: Any
    unit: str | None = None
    entity: str | None = None
    attribute: str
    relation: str | None = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    source_id: str
    exact_quote: str = ""
    confidence: Literal["high", "medium", "low"] = "low"


def _tokens(value: str | None) -> set[str]:
    return {
        item.casefold()
        for item in _WORD.findall(value or "")
        if len(item) > 2 and item.casefold() not in _STOP
    }


def _attribute_tokens(value: str | None) -> set[str]:
    tokens = _tokens(value.replace("_", " ") if value else "")
    aliases = {
        "birth": {"born", "birth"},
        "date": {"date", "born", "opened", "dedicated", "released"},
        "admission": {"admitted", "admission", "union"},
        "union": {"union", "admitted"},
        "imprisonment": {"imprisoned", "prison", "release", "jailed"},
        "discography": {"album", "albums", "discography", "released"},
        "depth": {"depth", "deepest", "deep"},
        "height": {"height", "tall", "tallest"},
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(aliases.get(token, set()))
    return expanded


def _evidence_aligns(entity: str | None, attributes: set[str], evidence: str) -> bool:
    """Require the entity and requested attribute to co-occur locally.

    A navigation/table page can contain both words hundreds of characters apart;
    that is not evidence that the table's date has the requested relation.
    """

    folded = evidence.casefold()
    entity_tokens = sorted(_tokens(entity))
    attribute_positions = [
        match.start()
        for token in attributes
        for match in re.finditer(rf"\b{re.escape(token)}\b", folded)
    ]
    if not entity_tokens or not attribute_positions:
        return False
    entity_positions = [
        match.start()
        for token in entity_tokens
        for match in re.finditer(rf"\b{re.escape(token)}\b", folded)
    ]
    return (
        bool(entity_positions)
        and min(
            abs(left - right)
            for left in entity_positions
            for right in attribute_positions
        )
        <= 160
    )


def _evidence_has_entity(entity: str | None, evidence: str) -> bool:
    tokens = sorted(_tokens(entity))
    if not tokens:
        return True
    # The final meaningful token is usually the distinguishing surname/place;
    # requiring it avoids accepting a generic "Robert" for a different person.
    return bool(re.search(rf"\b{re.escape(tokens[-1])}\b", evidence.casefold()))


def _attribute_from_text(text: str) -> str:
    folded = text.casefold()
    for needle, attribute in (
        ("height", "height"),
        ("tallest", "height"),
        ("deep", "depth"),
        ("imprison", "imprisonment_dates"),
        ("born", "birth_date"),
        ("birthplace", "birthplace"),
        ("hometown", "hometown"),
        ("admitted to the union", "admission_to_union"),
        ("admitted", "admission"),
        ("album", "discography"),
        ("discography", "discography"),
        ("count", "enumeration"),
        ("list", "enumeration"),
    ):
        if needle in folded:
            return attribute
    return "fact"


def _type_from_text(text: str) -> FactType:
    folded = text.casefold()
    if any(item in folded for item in ("year", "born", "date", "when")):
        return "year"
    if any(item in folded for item in ("list", "albums", "discography", "which")):
        return "list"
    if any(item in folded for item in ("how many", "height", "depth", "number")):
        return "integer"
    return "entity"


def _entity_from_text(text: str) -> str | None:
    candidates = [" ".join(item.split()) for item in _CAPITALIZED.findall(text)]
    return max(candidates, key=len) if candidates else None


def fallback_slots(
    *, question: str, subquestions: Sequence[Mapping[str, Any]]
) -> list[RequiredFactSlot]:
    """Conservative no-model fallback: one explicit need per research SQ.

    A few universally meaningful constructs (dates, lists, ranges) are
    decomposed so missing operands are visible rather than hidden in a single
    broad text slot.  It contains no benchmark/task identifiers or answers.
    """

    slots: list[RequiredFactSlot] = []
    for index, subquestion in enumerate(subquestions, start=1):
        text = str(subquestion.get("question", question))
        slot_id = f"slot-{index}"
        attribute = _attribute_from_text(text)
        fact_type = _type_from_text(text)
        slots.append(
            RequiredFactSlot(
                slot_id=slot_id,
                subquestion_id=str(subquestion.get("id") or "") or None,
                entity=_entity_from_text(text),
                attribute=attribute,
                fact_type=fact_type,
                unit="year" if fact_type == "year" else None,
                cardinality="list" if fact_type == "list" else "single",
                qualifiers={"question_fragment": text[:240]},
            )
        )
    return slots or [
        RequiredFactSlot(
            slot_id="slot-1",
            entity=_entity_from_text(question),
            attribute=_attribute_from_text(question),
            fact_type=_type_from_text(question),
            cardinality="list" if _type_from_text(question) == "list" else "single",
            qualifiers={"question_fragment": question[:240]},
        )
    ]


def match_slots(
    slots: Sequence[RequiredFactSlot], facts: Sequence[Any]
) -> list[SlotCoverage]:
    """Match source-bound facts by meaning-bearing fields, not value type alone."""

    coverage: list[SlotCoverage] = []
    for slot in slots:
        slot_entity = _tokens(slot.entity)
        slot_attribute = _attribute_tokens(slot.attribute)
        candidates: list[Any] = []
        wrong_entity: list[str] = []
        wrong_attribute: list[str] = []
        wrong_type: list[str] = []
        wrong_unit: list[str] = []
        partial: list[str] = []
        conflicts: list[str] = []
        conversion: dict[str, Any] | None = None
        for fact in facts:
            fact_id = str(getattr(fact, "fact_id", ""))
            status = str(getattr(fact, "verification_status", "unsupported"))
            if status in {"unsupported", "contested"}:
                if status == "contested":
                    conflicts.append(fact_id)
                continue
            if str(getattr(fact, "fact_type", "")) != slot.fact_type:
                wrong_type.append(fact_id)
                continue
            fact_entity = _tokens(getattr(fact, "entity", None))
            raw = str(getattr(fact, "raw_text", ""))
            qualifier = getattr(fact, "qualifier", {}) or {}
            evidence = " ".join(str(item) for item in qualifier.get("exact_quotes", []))
            entity_text = fact_entity or _tokens(raw)
            if slot_entity and not slot_entity.issubset(entity_text):
                wrong_entity.append(fact_id)
                continue
            fact_attribute = _attribute_tokens(str(getattr(fact, "attribute", "")))
            evidence_tokens = _tokens(evidence)
            if slot_entity and not _evidence_has_entity(slot.entity, evidence):
                wrong_entity.append(fact_id)
                continue
            semantic_attribute = slot_attribute.intersection(fact_attribute)
            evidence_attribute = slot_attribute.intersection(evidence_tokens)
            if slot_attribute and (
                not semantic_attribute
                or not evidence_attribute
                or not _evidence_aligns(slot.entity, slot_attribute, evidence)
            ):
                wrong_attribute.append(fact_id)
                continue
            if slot.unit and getattr(fact, "unit", None) not in {None, slot.unit}:
                # The executor deliberately does not convert values; coverage
                # may document a known compatible conversion for repair/audit.
                pair = {
                    str(slot.unit).casefold(),
                    str(getattr(fact, "unit", "")).casefold(),
                }
                if pair in (
                    {"m", "ft"},
                    {"m", "feet"},
                    {"metre", "ft"},
                    {"meter", "ft"},
                ):
                    conversion = {
                        "from_unit": getattr(fact, "unit", None),
                        "to_unit": slot.unit,
                        "factor": (
                            "0.3048"
                            if str(getattr(fact, "unit", "")).casefold()
                            in {"ft", "feet"}
                            else "3.280839895"
                        ),
                    }
                elif pair not in ({"ft", "feet"}, {"m", "meter"}, {"m", "metre"}):
                    wrong_unit.append(fact_id)
                    continue
            if slot.cardinality == "list" and not bool(qualifier.get("complete_list")):
                partial.append(fact_id)
                continue
            if status == "partially_supported":
                partial.append(fact_id)
            else:
                candidates.append(fact)
        if candidates:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="satisfied",
                    matching_fact_ids=[
                        str(getattr(item, "fact_id", "")) for item in candidates
                    ],
                    conflicting_fact_ids=conflicts,
                    reasons=[
                        "verified fact matches type, entity, attribute, and qualifiers"
                    ],
                    unit_conversion=conversion,
                )
            )
        elif conflicts:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="conflicting",
                    conflicting_fact_ids=conflicts,
                    reasons=["conflicting source-bound facts"],
                )
            )
        elif partial:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="partially_satisfied",
                    matching_fact_ids=partial,
                    reasons=["only partial or incomplete-list support"],
                )
            )
        elif wrong_entity:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="wrong_entity",
                    matching_fact_ids=wrong_entity,
                    reasons=["candidate fact entity does not match slot entity"],
                )
            )
        elif wrong_attribute:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="wrong_attribute",
                    matching_fact_ids=wrong_attribute,
                    reasons=[
                        "candidate evidence does not establish the requested attribute"
                    ],
                )
            )
        elif wrong_type:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="wrong_type",
                    matching_fact_ids=wrong_type,
                    reasons=["candidate fact type does not match slot type"],
                )
            )
        elif wrong_unit:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="wrong_unit",
                    matching_fact_ids=wrong_unit,
                    reasons=["candidate fact unit is incompatible"],
                )
            )
        else:
            coverage.append(
                SlotCoverage(
                    slot_id=slot.slot_id,
                    status="missing",
                    reasons=["no source-bound fact satisfies slot"],
                )
            )
    return coverage


def gaps_from_coverage(coverage: Sequence[SlotCoverage]) -> list[FactGap]:
    result: list[FactGap] = []
    for item in coverage:
        if item.status == "satisfied":
            continue
        taxonomy = {
            "missing": "no_fact",
            "partially_satisfied": "only_partial_support",
            "conflicting": "conflicting_sources",
            "wrong_entity": "wrong_entity",
            "wrong_attribute": "wrong_attribute",
            "wrong_type": "fact_extraction_failure",
            "wrong_unit": "unit_mismatch",
            "incomplete_list": "incomplete_enumeration",
            "ambiguous": "fact_extraction_failure",
            "wrong_qualifier": "wrong_time_scope",
        }
        result.append(
            FactGap(
                slot_id=item.slot_id,
                status=taxonomy.get(item.status, item.status),
                missing_components=item.reasons,
                existing_fact_ids=item.matching_fact_ids,
                conflict_fact_ids=item.conflicting_fact_ids,
                repair_priority=1
                if item.status in {"missing", "wrong_entity", "wrong_attribute"}
                else 2,
                repairable=item.status not in {"conflicting"},
            )
        )
    return result


def repair_queries(slot: RequiredFactSlot, gap: FactGap) -> list[RepairQuery]:
    """Create exactly one narrow plus one broad query; never embed arithmetic."""

    pieces = [
        item
        for item in (slot.entity, slot.relation, slot.attribute.replace("_", " "))
        if item
    ]
    if slot.fact_type in {"year", "date"}:
        pieces.append("year")
    elif slot.fact_type in {"integer", "float", "duration"}:
        pieces.append("value")
    elif slot.cardinality == "list":
        pieces.append("list overview")
    qualifier = " ".join(
        str(value)
        for value in slot.qualifiers.values()
        if isinstance(value, (str, int))
    )
    exact = " ".join(
        pieces
        + (
            [qualifier]
            if qualifier and "question_fragment" not in slot.qualifiers
            else []
        )
    )
    exact = " ".join(exact.split())[:180]
    broad = " ".join(pieces[:3] + (["overview"] if slot.cardinality == "list" else []))[
        :140
    ]
    return [
        RepairQuery(
            slot_id=slot.slot_id,
            attempt=1,
            query=exact,
            normalized_query=exact,
            broadened=False,
            reason=gap.status,
        ),
        RepairQuery(
            slot_id=slot.slot_id,
            attempt=2,
            query=broad,
            normalized_query=broad,
            broadened=True,
            reason=gap.status,
        ),
    ]
