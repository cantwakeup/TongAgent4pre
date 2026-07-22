"""Socket-disabled coverage for Required Slots and Fact-Gap classification."""

from __future__ import annotations

from evaluation.fact_gap import (
    FactGap,
    RequiredFactSlot,
    gaps_from_coverage,
    match_slots,
    repair_queries,
)
from evaluation.systems.permissive import TypedFact


def _fact(
    fact_id: str,
    *,
    entity: str = "Aurora Bridge",
    attribute: str = "opening_date",
    fact_type: str = "year",
    value: object = 1974,
    unit: str | None = "year",
    status: str = "verified",
    quote: str = "Aurora Bridge opened in 1974.",
    qualifier: dict[str, object] | None = None,
) -> TypedFact:
    return TypedFact(
        fact_id=fact_id,
        subquestion_id="SQ1",
        fact_type=fact_type,  # type: ignore[arg-type]
        value=value,
        unit=unit,
        entity=entity,
        attribute=attribute,
        qualifier={"exact_quotes": [quote], **(qualifier or {})},
        source_ids=["S1"],
        claim_ids=["C1"],
        verification_status=status,  # type: ignore[arg-type]
        raw_text=quote,
    )


def _slot(**updates: object) -> RequiredFactSlot:
    baseline = {
        "slot_id": "slot-1",
        "subquestion_id": "SQ1",
        "entity": "Aurora Bridge",
        "attribute": "opening_date",
        "fact_type": "year",
        "unit": "year",
    }
    baseline.update(updates)
    return RequiredFactSlot(**baseline)  # type: ignore[arg-type]


def test_matcher_requires_entity_attribute_and_canonical_quote() -> None:
    assert match_slots([_slot()], [_fact("F1")])[0].status == "satisfied"
    assert (
        match_slots([_slot()], [_fact("F2", entity="Meridian Monument")])[0].status
        == "wrong_entity"
    )
    assert (
        match_slots(
            [_slot(entity="Pennsylvania", attribute="admission_to_union")],
            [
                _fact(
                    "F3",
                    entity="Pennsylvania",
                    attribute="admission_to_union",
                    quote="State date: Pennsylvania March 5, 1778.",
                )
            ],
        )[0].status
        == "wrong_attribute"
    )


def test_matcher_detects_type_partial_list_conflict_and_unit_cases() -> None:
    assert (
        match_slots([_slot()], [_fact("F1", fact_type="integer")])[0].status
        == "wrong_type"
    )
    list_slot = _slot(
        fact_type="list", attribute="discography", unit=None, cardinality="list"
    )
    incomplete = _fact(
        "F2",
        attribute="discography",
        fact_type="list",
        value=[{"year": 1991}],
        unit=None,
        quote="Aurora Bridge discography includes one album released in 1991.",
    )
    assert match_slots([list_slot], [incomplete])[0].status == "partially_satisfied"
    conflict = _fact("F3", status="contested")
    assert match_slots([_slot()], [conflict])[0].status == "conflicting"
    metres = _fact(
        "F4",
        fact_type="integer",
        value=10,
        unit="m",
        attribute="height",
        quote="Aurora Bridge height is 10 m.",
    )
    height_slot = _slot(attribute="height", fact_type="integer", unit="ft")
    converted = match_slots([height_slot], [metres])[0]
    assert converted.status == "satisfied"
    assert converted.unit_conversion is not None


def test_gap_taxonomy_and_two_bounded_repair_queries() -> None:
    coverage = match_slots([_slot()], [])
    gaps = gaps_from_coverage(coverage)
    assert gaps[0].status == "no_fact"
    assert gaps[0].repairable is True
    conflict = FactGap(
        slot_id="slot-1",
        status="conflicting_sources",
        repairable=False,
        repair_priority=1,
    )
    assert conflict.repairable is False
    queries = repair_queries(
        _slot(entity="Aurora Bridge", attribute="opening_date"), gaps[0]
    )
    assert [item.attempt for item in queries] == [1, 2]
    assert all("How many" not in item.query for item in queries)
