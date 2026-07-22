"""Controlled no-network repair from a missing Slot to canonical evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.budget import ExecutionBudget
from evaluation.execution import resolve_system_config
from evaluation.fact_gap import RequiredFactSlot, gaps_from_coverage, match_slots
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.schema import EvalTask
from evaluation.systems.common import prepare_runtime
from evaluation.systems.permissive import _repair_slot_facts
from evaluation.tracing import TraceCollector


pytestmark = pytest.mark.usefixtures("socket_disabled")


def test_controlled_missing_slot_is_repaired_and_registered(tmp_path: Path) -> None:
    url = "https://archive.fixture.test/aurora-bridge"
    backend = FixtureBackend(
        searches={
            '"Aurora Bridge" opening': {
                "results": [
                    {
                        "title": "Aurora Bridge archive",
                        "url": url,
                        "snippet": "Aurora Bridge opened in 1974.",
                    }
                ]
            },
            "Aurora Bridge opening date year": {
                "results": [
                    {
                        "title": "Aurora Bridge archive",
                        "url": url,
                        "snippet": "Aurora Bridge opened in 1974.",
                    }
                ]
            },
            "Aurora Bridge opening date": {
                "results": [
                    {
                        "title": "Aurora Bridge archive",
                        "url": url,
                        "snippet": "Aurora Bridge opened in 1974.",
                    }
                ]
            },
        },
        pages={
            url: {
                "title": "Aurora Bridge archive",
                "content": "Aurora Bridge opened in 1974. " * 50,
            }
        },
    )
    task = EvalTask(
        id="controlled-gap-repair",
        question="When did Aurora Bridge open?",
        metadata={
            "structured_outputs": {
                "SlotFactCandidate_slot-1_S1": {
                    "slot_id": "slot-1",
                    "fact_type": "year",
                    "value": 1974,
                    "unit": "year",
                    "entity": "Aurora Bridge",
                    "attribute": "opening_date",
                    "relation": None,
                    "qualifiers": {},
                    "source_id": "S1",
                    "exact_quote": "Aurora Bridge opened in 1974.",
                    "confidence": "high",
                }
            }
        },
    )
    config = resolve_system_config(
        "tongagent",
        "sha256:" + "a" * 64,
        17,
        tmp_path,
        overrides={"fixture_revision": backend.revision, "runtime_mode": "permissive"},
    )
    runtime = prepare_runtime(
        task,
        config,
        system_id="tongagent",
        execution_budget=ExecutionBudget(config.budget),
        trace=TraceCollector(),
        injected_backend=backend,
        injected_model=FixtureChatModel.from_task(task, system_id="tongagent"),
        semantic_strategy="fixed",
        enable_tongagent_evidence_state=True,
        reserve_final_synthesis=False,
    )
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.research_budget.activate_subquestion("SQ1")
    slot = RequiredFactSlot(
        slot_id="slot-1",
        subquestion_id="SQ1",
        entity="Aurora Bridge",
        attribute="opening_date",
        fact_type="year",
        unit="year",
    )
    gaps = gaps_from_coverage(match_slots([slot], []))
    facts, repair_trace = _repair_slot_facts(
        runtime=runtime,
        slots=[slot],
        gaps=gaps,
        sources={},
        max_searches=2,
        max_fetches=2,
        max_queries_per_slot=2,
    )
    assert len(facts) == 1, (repair_trace, backend.calls, backend.search_queries)
    assert match_slots([slot], facts)[0].status == "satisfied"
    assert repair_trace[0]["candidates"][0]["status"] == "accepted"
    assert len(runtime.research_budget.sources) == 1
    assert len(runtime.research_budget.evidence_graph.evidence_units) == 1
