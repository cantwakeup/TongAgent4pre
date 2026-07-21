"""Socket-disabled coverage for the post-hoc permissive TongAgent workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.budget import ExecutionBudget
from evaluation.execution import resolve_system_config
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.schema import CompletionStatus, EvalTask
from evaluation.systems.common import prepare_runtime
from evaluation.systems.permissive import (
    DraftAnswer,
    DraftClaim,
    PermissiveSubquestion,
    ResearchNote,
    ResearchSource,
    VerifiedClaim,
    _draft,
    _finalize,
    _note,
    _plan,
    _soft_normalize_query,
    _verify,
    research_query,
)
from evaluation.systems.tongagent import TongAgentRunner
from evaluation.tracing import TraceCollector


pytestmark = pytest.mark.usefixtures("socket_disabled")

_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "permissive_controlled"
)


def _backend() -> FixtureBackend:
    return FixtureBackend.from_directory(_FIXTURE_DIRECTORY)


def _config(tmp_path: Path, backend: FixtureBackend, *, mode: str = "permissive"):
    return resolve_system_config(
        "tongagent",
        "sha256:" + ("e" * 64),
        17,
        tmp_path,
        overrides={"fixture_revision": backend.revision, "runtime_mode": mode},
    )


def _river_task() -> EvalTask:
    return EvalTask(
        id="controlled-river",
        question="Which river passes through the valley containing Lunarite Mine?",
        metadata={
            "structured_outputs": {
                "PermissivePlan": {
                    "objective": "Locate Lunarite Mine and identify the valley river.",
                    "subquestions": [
                        {
                            "id": "SQ1",
                            "question": "Where is Lunarite Mine located?",
                            "task_type": "single_fact_lookup",
                            "required_for_final_answer": True,
                        },
                        {
                            "id": "SQ2",
                            "question": "Which river passes through Alton Valley?",
                            "task_type": "single_fact_lookup",
                            "required_for_final_answer": True,
                        },
                    ],
                },
                "ResearchQueryDecision_SQ1_R1": {
                    "query": "Lunarite Mine location",
                    "needs_second_round": False,
                    "missing_fact": "",
                },
                "ResearchQueryDecision_SQ2_R1": {
                    "query": "Alton Valley river",
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
                "ResearchNote_SQ2": {
                    "subquestion_id": "SQ2",
                    "claim_candidates": [
                        "The Arden River passes through Alton Valley."
                    ],
                    "source_ids": ["S2"],
                    "supporting_passages": [
                        "The regional water survey states that the Arden River passes through Alton Valley before joining the lower plain."
                    ],
                    "unresolved_points": [],
                    "confidence": "high",
                    "typed_facts": [
                        {
                            "fact_id": "F2",
                            "subquestion_id": "SQ2",
                            "fact_type": "entity",
                            "value": "Arden River",
                            "source_ids": ["S2"],
                            "claim_ids": [],
                            "verification_status": "partially_supported",
                            "raw_text": "The Arden River passes through Alton Valley.",
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
                        },
                        {
                            "claim_id": "D2",
                            "text": "The Arden River passes through Alton Valley.",
                            "source_ids": ["S2"],
                            "critical_for_final_answer": True,
                            "subquestion_id": "SQ2",
                        },
                    ],
                    "reasoning_steps": [
                        "The mine source identifies Alton Valley.",
                        "The water survey identifies the valley river.",
                    ],
                    "proposed_answer": "The Arden River.",
                    "confidence": "high",
                    "missing_information": [],
                },
                "AnswerPlan": {
                    "operation": "direct_lookup",
                    "required_fact_ids": ["F2"],
                    "output_type": "entity",
                    "output_unit": None,
                    "parameters": {},
                },
            }
        },
    )


def _additional_controlled_tasks() -> list[EvalTask]:
    """Two more local two-step cases: agreement and an explicit conflict."""

    return [
        EvalTask(
            id="controlled-dates",
            question="Did Aurora Bridge and the Meridian Monument share a year?",
            metadata={
                "structured_outputs": {
                    "PermissivePlan": {
                        "objective": "Compare the two civic dates.",
                        "subquestions": [
                            {
                                "id": "SQ1",
                                "question": "When did Aurora Bridge open?",
                                "task_type": "date_or_numeric_lookup",
                                "required_for_final_answer": True,
                            },
                            {
                                "id": "SQ2",
                                "question": "When was the Meridian Monument dedicated?",
                                "task_type": "date_or_numeric_lookup",
                                "required_for_final_answer": True,
                            },
                        ],
                    },
                    "ResearchQueryDecision_SQ1_R1": {
                        "query": "Aurora Bridge opening date",
                        "needs_second_round": False,
                        "missing_fact": "",
                    },
                    "ResearchQueryDecision_SQ2_R1": {
                        "query": "Meridian Monument dedication",
                        "needs_second_round": False,
                        "missing_fact": "",
                    },
                    "ResearchNote_SQ1": {
                        "subquestion_id": "SQ1",
                        "claim_candidates": ["Aurora Bridge opened in 1974."],
                        "source_ids": ["S1"],
                        "supporting_passages": [
                            "The engineering record states that Aurora Bridge opened in 1974 after a three-year construction programme."
                        ],
                        "unresolved_points": [],
                        "confidence": "high",
                        "typed_facts": [
                            {
                                "fact_id": "F1",
                                "subquestion_id": "SQ1",
                                "fact_type": "year",
                                "value": 1974,
                                "unit": "year",
                                "source_ids": ["S1"],
                                "claim_ids": [],
                                "verification_status": "partially_supported",
                                "raw_text": "Aurora Bridge opened in 1974.",
                            }
                        ],
                    },
                    "ResearchNote_SQ2": {
                        "subquestion_id": "SQ2",
                        "claim_candidates": [
                            "The Meridian Monument was dedicated in 1974."
                        ],
                        "source_ids": ["S2"],
                        "supporting_passages": [
                            "The civic register states that the Meridian Monument was dedicated in 1974."
                        ],
                        "unresolved_points": [],
                        "confidence": "high",
                        "typed_facts": [
                            {
                                "fact_id": "F2",
                                "subquestion_id": "SQ2",
                                "fact_type": "year",
                                "value": 1974,
                                "unit": "year",
                                "source_ids": ["S2"],
                                "claim_ids": [],
                                "verification_status": "partially_supported",
                                "raw_text": "The Meridian Monument was dedicated in 1974.",
                            }
                        ],
                    },
                    "DraftAnswer": {
                        "claims": [
                            {
                                "claim_id": "D1",
                                "text": "Aurora Bridge opened in 1974.",
                                "source_ids": ["S1"],
                                "critical_for_final_answer": True,
                                "subquestion_id": "SQ1",
                            },
                            {
                                "claim_id": "D2",
                                "text": "The Meridian Monument was dedicated in 1974.",
                                "source_ids": ["S2"],
                                "critical_for_final_answer": True,
                                "subquestion_id": "SQ2",
                            },
                        ],
                        "reasoning_steps": ["Compare the two source-grounded years."],
                        "proposed_answer": "Yes. Both events occurred in 1974.",
                        "confidence": "high",
                        "missing_information": [],
                    },
                    "AnswerPlan": {
                        "operation": "compare",
                        "required_fact_ids": ["F1", "F2"],
                        "output_type": "boolean",
                        "output_unit": None,
                        "parameters": {"direction": "equals"},
                    },
                }
            },
        ),
        EvalTask(
            id="controlled-conflict",
            question="Do the local history and preservation file agree on Harbor Light Station's completion year?",
            metadata={
                "structured_outputs": {
                    "PermissivePlan": {
                        "objective": "Compare the two Harbor Light Station records.",
                        "subquestions": [
                            {
                                "id": "SQ1",
                                "question": "What completion year does the local history give?",
                                "task_type": "date_or_numeric_lookup",
                                "required_for_final_answer": True,
                            },
                            {
                                "id": "SQ2",
                                "question": "What completion year does the preservation file give?",
                                "task_type": "date_or_numeric_lookup",
                                "required_for_final_answer": True,
                            },
                        ],
                    },
                    "ResearchQueryDecision_SQ1_R1": {
                        "query": "Harbor Light Station history",
                        "needs_second_round": False,
                        "missing_fact": "",
                    },
                    "ResearchQueryDecision_SQ2_R1": {
                        "query": "Harbor Light Station preservation",
                        "needs_second_round": False,
                        "missing_fact": "",
                    },
                    "ResearchNote_SQ1": {
                        "subquestion_id": "SQ1",
                        "claim_candidates": [
                            "The local history says Harbor Light Station was completed in 1950."
                        ],
                        "source_ids": ["S1"],
                        "supporting_passages": [
                            "The local history says that Harbor Light Station was completed in 1950."
                        ],
                        "unresolved_points": [],
                        "confidence": "medium",
                        "typed_facts": [
                            {
                                "fact_id": "F1",
                                "subquestion_id": "SQ1",
                                "fact_type": "year",
                                "value": 1950,
                                "unit": "year",
                                "source_ids": ["S1"],
                                "claim_ids": [],
                                "verification_status": "partially_supported",
                                "raw_text": "The local history says Harbor Light Station was completed in 1950.",
                            }
                        ],
                    },
                    "ResearchNote_SQ2": {
                        "subquestion_id": "SQ2",
                        "claim_candidates": [
                            "The preservation file says Harbor Light Station was completed in 1952."
                        ],
                        "source_ids": ["S2"],
                        "supporting_passages": [
                            "The preservation file says that Harbor Light Station was completed in 1952."
                        ],
                        "unresolved_points": [],
                        "confidence": "medium",
                        "typed_facts": [
                            {
                                "fact_id": "F2",
                                "subquestion_id": "SQ2",
                                "fact_type": "year",
                                "value": 1952,
                                "unit": "year",
                                "source_ids": ["S2"],
                                "claim_ids": [],
                                "verification_status": "partially_supported",
                                "raw_text": "The preservation file says Harbor Light Station was completed in 1952.",
                            }
                        ],
                    },
                    "DraftAnswer": {
                        "claims": [
                            {
                                "claim_id": "D1",
                                "text": "The local history says Harbor Light Station was completed in 1950.",
                                "source_ids": ["S1"],
                                "critical_for_final_answer": True,
                                "subquestion_id": "SQ1",
                            },
                            {
                                "claim_id": "D2",
                                "text": "The preservation file says Harbor Light Station was completed in 1952.",
                                "source_ids": ["S2"],
                                "critical_for_final_answer": True,
                                "subquestion_id": "SQ2",
                            },
                        ],
                        "reasoning_steps": ["Compare the two supported records."],
                        "proposed_answer": "No. The records conflict: 1950 versus 1952.",
                        "confidence": "medium",
                        "missing_information": [],
                    },
                    "AnswerPlan": {
                        "operation": "compare",
                        "required_fact_ids": ["F1", "F2"],
                        "output_type": "boolean",
                        "output_unit": None,
                        "parameters": {"direction": "equals"},
                    },
                }
            },
        ),
    ]


def _runtime(tmp_path: Path, task: EvalTask, backend: FixtureBackend):
    return prepare_runtime(
        task,
        _config(tmp_path, backend),
        system_id="tongagent",
        execution_budget=ExecutionBudget(_config(tmp_path, backend).budget),
        trace=TraceCollector(),
        injected_backend=backend,
        injected_model=FixtureChatModel.from_task(task, system_id="tongagent"),
        semantic_strategy="fixed",
        enable_tongagent_evidence_state=True,
        enable_tongagent_token_control=True,
        enable_tongagent_context_compaction=True,
        reserve_final_synthesis=False,
    )


def test_permissive_controlled_two_hop_workflow_verifies_before_answer(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    config = _config(tmp_path, backend)

    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, config)

    assert result.runtime_mode == "permissive"
    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.final_answer is not None
    assert "FINAL_ANSWER: Arden River" in result.final_answer
    assert result.search_calls == 2
    # The bundled acquisition also inspects one lower-ranked same-entity page
    # in the bounded two-source window; the graph only registers claims that
    # the post-hoc verifier selected.
    assert result.fetch_calls == 3
    assert result.evidence_count == 2
    assert result.structural_subquestion_coverage == 1.0
    assert result.workflow_metrics is not None
    assert result.workflow_metrics["sq_research_completion_rate"] == 1.0
    assert result.workflow_metrics["critical_claim_verified_rate"] == 1.0
    workflow = json.loads(
        (tmp_path / "native" / "tongagent" / "permissive_workflow.json").read_text()
    )
    assert workflow["phase"] == "FINALIZE"
    assert workflow["final_answer_status"] == "answer"
    artifacts = workflow["canonical_artifacts"]
    assert set(artifacts) == {
        "research_notes",
        "draft_answer",
        "verified_claims",
        "evidence_graph",
        "finalization_decision",
        "typed_facts",
        "answer_plan",
        "calculation_trace",
        "answer_execution",
    }
    for filename in artifacts.values():
        payload = json.loads((tmp_path / "native" / "tongagent" / filename).read_text())
        assert payload["schema_version"] == 1
    verified = json.loads(
        (tmp_path / "native" / "tongagent" / artifacts["verified_claims"]).read_text()
    )
    assert {item["status"] for item in verified["verified_claims"]} == {"verified"}


@pytest.mark.parametrize(
    "task", _additional_controlled_tasks(), ids=lambda item: item.id
)
def test_controlled_fixture_runs_two_more_multi_step_workflows(
    tmp_path: Path, task: EvalTask
) -> None:
    backend = _backend()
    config = _config(tmp_path, backend)
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, config)

    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.final_answer is not None
    assert "FINAL_ANSWER: ABSTAIN" not in result.final_answer
    assert result.evidence_count == 2
    assert result.workflow_metrics is not None
    assert result.workflow_metrics["critical_claim_verified_rate"] == 1.0


def test_research_query_softens_ordinary_input_and_rejects_unsafe_input(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    runtime = _runtime(tmp_path, task, backend)
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.research_budget.activate_subquestion("SQ1")
    runtime.middleware.configure_token_partitions(["SQ1"])
    runtime.middleware.activate_token_subquestion("SQ1")
    subquestion = PermissiveSubquestion(
        id="SQ1",
        question="Where is Lunarite Mine located?",
        task_type="single_fact_lookup",
        required_for_final_answer=True,
    )
    by_name = {tool.name: tool for tool in runtime.tools}
    bundle = research_query(
        query="Search for: Lunarite Mine location",
        task_type="single_fact_lookup",
        subquestion=subquestion,
        search_tool=by_name["web_search"],
        fetch_tool=by_name["fetch_url"],
    )
    assert bundle.query_validation_status == "normalized"
    assert "removed_explanatory_prefix" in bundle.query_normalizations
    assert [source.url for source in bundle.sources] == [
        "https://archive.fixture.test/lunarite-mine"
    ]
    assert bundle.candidate_audit[0]["provider_rank"] == 1
    assert bundle.candidate_audit[0]["fetch_status"] == "success"
    assert bundle.candidate_audit[1]["relevance_tier"] == "irrelevant"
    assert bundle.candidate_audit[1]["fetch_status"] == "not_attempted"
    unsafe = _soft_normalize_query(
        "file:///etc/passwd",
        subquestion=PermissiveSubquestion(
            id="bad",
            question="javascript:alert(1)",
            task_type="single_fact_lookup",
            required_for_final_answer=True,
        ),
    )
    assert unsafe.status == "rejected"
    assert unsafe.rejection_reason == "dangerous_protocol_or_url_injection"


def test_research_query_broadens_once_after_an_exact_phrase_miss(
    tmp_path: Path,
) -> None:
    url = "https://archive.fixture.test/lunarite-broad"
    backend = FixtureBackend(
        searches={
            "lunarite mine location": {
                "results": [
                    {
                        "title": "Lunarite Mine archive",
                        "url": url,
                        "snippet": "Lunarite Mine is located in Alton Valley.",
                    }
                ]
            }
        },
        pages={
            url: {
                "title": "Lunarite Mine archive",
                "content": "Lunarite Mine is located in Alton Valley. " * 20,
            }
        },
    )
    task = _river_task()
    runtime = _runtime(tmp_path, task, backend)
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.research_budget.activate_subquestion("SQ1")
    by_name = {tool.name: tool for tool in runtime.tools}
    bundle = research_query(
        query="Lunarite Mine location",
        task_type="single_fact_lookup",
        subquestion=PermissiveSubquestion(
            id="SQ1",
            question="Where is Lunarite Mine located?",
            task_type="single_fact_lookup",
            required_for_final_answer=True,
        ),
        search_tool=by_name["web_search"],
        fetch_tool=by_name["fetch_url"],
    )
    assert bundle.broadened is True
    assert bundle.normalized_query == "Lunarite Mine location"
    assert [source.url for source in bundle.sources] == [url]
    assert runtime.research_budget.snapshot()["search_calls"] == 2


def test_note_and_draft_filter_hallucinated_source_ids_without_losing_fetches(
    tmp_path: Path,
) -> None:
    backend = _backend()
    original = _river_task()
    metadata = dict(original.metadata)
    outputs = dict(metadata["structured_outputs"])
    outputs["ResearchNote_SQ1"] = {
        "subquestion_id": "SQ1",
        "claim_candidates": ["An unsupported model claim."],
        "source_ids": ["not-a-source"],
        "supporting_passages": [],
        "unresolved_points": [],
        "confidence": "high",
    }
    outputs["DraftAnswer"] = {
        "claims": [
            {
                "claim_id": "D1",
                "text": "An unsupported model claim with a fabricated source.",
                "source_ids": ["not-a-source"],
                "critical_for_final_answer": True,
                "subquestion_id": "SQ1",
            }
        ],
        "reasoning_steps": [],
        "proposed_answer": "A fabricated answer.",
        "confidence": "high",
        "missing_information": [],
    }
    task = original.model_copy(
        update={"metadata": {**metadata, "structured_outputs": outputs}}
    )
    runtime = _runtime(tmp_path, task, backend)
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.research_budget.activate_subquestion("SQ1")
    subquestion = PermissiveSubquestion(
        id="SQ1",
        question="Where is Lunarite Mine located?",
        task_type="single_fact_lookup",
        required_for_final_answer=True,
    )
    by_name = {tool.name: tool for tool in runtime.tools}
    bundle = research_query(
        query="Lunarite Mine location",
        task_type="single_fact_lookup",
        subquestion=subquestion,
        search_tool=by_name["web_search"],
        fetch_tool=by_name["fetch_url"],
    )
    note = _note(runtime=runtime, subquestion=subquestion, bundles=[bundle])
    assert note.source_ids == ["S1"]
    draft = _draft(runtime=runtime, task=task, notes=[note])
    assert draft.proposed_answer is None
    assert draft.claims[0].source_ids == ["S1"]


def test_planner_repair_then_deterministic_fallback_are_nonblocking(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task().model_copy(
        update={
            "metadata": {
                "structured_outputs": {
                    "PermissivePlan": {},
                    "PermissivePlan_Repair": {
                        "objective": "Repair plan",
                        "subquestions": [
                            {
                                "id": "SQ1",
                                "question": "Where is Lunarite Mine located?",
                                "task_type": "single_fact_lookup",
                                "required_for_final_answer": True,
                            }
                        ],
                    },
                }
            }
        }
    )
    repaired, used_fallback = _plan(
        runtime=_runtime(tmp_path, task, backend), task=task
    )
    assert used_fallback is True
    assert repaired.objective == "Repair plan"

    fallback_task = task.model_copy(update={"metadata": {"structured_outputs": {}}})
    fallback, used_fallback = _plan(
        runtime=_runtime(tmp_path / "fallback", fallback_task, backend),
        task=fallback_task,
    )
    assert used_fallback is True
    assert len(fallback.subquestions) == 1
    assert fallback.subquestions[0].question == fallback_task.question


def test_verifier_marks_conflicting_canonical_sources_and_finalizer_abstains(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    runtime = _runtime(tmp_path, task, backend)
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.research_budget.activate_subquestion("SQ1")
    runtime.middleware.configure_token_partitions(["SQ1"])
    runtime.middleware.activate_token_subquestion("SQ1")
    by_name = {tool.name: tool for tool in runtime.tools}
    subquestion = PermissiveSubquestion(
        id="SQ1",
        question="When was Harbor Light Station completed?",
        task_type="date_or_numeric_lookup",
        required_for_final_answer=True,
    )
    bundle = research_query(
        query="Harbor Light Station construction year",
        task_type="date_or_numeric_lookup",
        subquestion=subquestion,
        search_tool=by_name["web_search"],
        fetch_tool=by_name["fetch_url"],
        max_sources=2,
    )
    sources = {source.source_id: source for source in bundle.sources}
    assert len(sources) == 2
    draft = DraftAnswer(
        claims=[
            DraftClaim(
                claim_id="D1",
                text="Harbor Light Station was completed in 1950.",
                source_ids=list(sources),
                critical_for_final_answer=True,
                subquestion_id="SQ1",
            )
        ],
        proposed_answer="1950",
    )
    verified = _verify(runtime=runtime, draft=draft, sources=sources)
    assert verified[0].status == "contested"
    answer, status, removed = _finalize(
        task=task,
        draft=draft,
        notes=[ResearchNote(subquestion_id="SQ1", source_ids=list(sources))],
        verified=verified,
        config=_config(tmp_path, backend),
    )
    assert status == "abstain"
    assert removed == ["D1"]
    assert "FINAL_ANSWER: ABSTAIN" in answer


def test_finalizer_refuses_answer_without_a_critical_source_grounded_claim(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    answer, status, removed = _finalize(
        task=task,
        draft=DraftAnswer(proposed_answer="An unsupported number."),
        notes=[ResearchNote(subquestion_id="SQ1", source_ids=["S1"])],
        verified=[],
        config=_config(tmp_path, backend),
    )
    assert status == "abstain"
    assert removed == []
    assert "FINAL_ANSWER: ABSTAIN" in answer


@pytest.mark.parametrize(
    ("historical_case", "claim_text"),
    [
        (
            "frames-0123-long-wikipedia-navigation",
            "The hockey goal was scored in overtime.",
        ),
        (
            "frames-0664-long-band-navigation",
            "The band has a documented studio-album list.",
        ),
        (
            "frames-0718-long-state-admission-table",
            "Pennsylvania was admitted on March 5, 1778.",
        ),
    ],
)
def test_long_navigation_passage_registers_bounded_canonical_quote(
    tmp_path: Path, historical_case: str, claim_text: str
) -> None:
    """Regression for the three live ValueErrors caused by >500-char quotes."""

    backend = _backend()
    runtime = _runtime(tmp_path / historical_case, _river_task(), backend)
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.middleware.configure_token_partitions(["SQ1"])
    content = "navigation " * 260 + claim_text + " canonical supporting text " * 260
    runtime.research_budget.sources.append(
        {"source_id": "S-long", "url": "https://example.test/long", "title": "Long"}
    )
    runtime.research_budget.evidence_graph.cache_page("S-long", content)
    source = ResearchSource(
        source_id="S-long",
        title="Long",
        url="https://example.test/long",
        provider="fixture",
        provider_rank=1,
        acquisition_method="direct_http",
        relevance_tier="relevant",
        relevance_reason="fixture",
        passage=content,
        content_chars=len(content),
        fetch_status="success",
    )
    verified = _verify(
        runtime=runtime,
        draft=DraftAnswer(
            claims=[
                DraftClaim(
                    claim_id="D1",
                    text=claim_text,
                    source_ids=["S-long"],
                    critical_for_final_answer=True,
                    subquestion_id="SQ1",
                )
            ],
            proposed_answer="A source-grounded answer.",
            reasoning_steps=["Use the canonical source."],
        ),
        sources={"S-long": source},
    )
    assert verified[0].status == "verified"
    assert verified[0].exact_quotes
    assert len(verified[0].exact_quotes[0]) <= 500
    assert verified[0].registration_failures == []


def test_finalizer_allows_low_confidence_answer_with_verified_and_partial_core_claims(
    tmp_path: Path,
) -> None:
    backend = _backend()
    draft = DraftAnswer(
        claims=[
            DraftClaim(
                claim_id="D1",
                text="The first source establishes the required year.",
                source_ids=["S1"],
                critical_for_final_answer=True,
                subquestion_id="SQ1",
            ),
            DraftClaim(
                claim_id="D2",
                text="The second source supports the associated calculation.",
                source_ids=["S2"],
                critical_for_final_answer=True,
                subquestion_id="SQ2",
            ),
        ],
        reasoning_steps=["Use the two source-grounded values."],
        proposed_answer="42",
    )
    verified = [
        VerifiedClaim(
            claim_id="D1",
            status="verified",
            source_ids=["S1"],
            exact_quotes=["The first source establishes the required year."],
            explanation="Exact quote found.",
            canonical_claim_id="C1",
        ),
        VerifiedClaim(
            claim_id="D2",
            status="partially_supported",
            source_ids=["S2"],
            explanation="Canonical source fetched but quote is partial.",
        ),
    ]
    answer, status, removed = _finalize(
        task=_river_task(),
        draft=draft,
        notes=[
            ResearchNote(subquestion_id="SQ1", source_ids=["S1"]),
            ResearchNote(subquestion_id="SQ2", source_ids=["S2"]),
        ],
        verified=verified,
        config=_config(tmp_path, backend),
        required_subquestion_ids=["SQ1", "SQ2"],
    )
    assert status == "answer_low_confidence"
    assert removed == []
    assert "Confidence: low." in answer
    assert "Partially supported core claims: D2" in answer
    assert "FINAL_ANSWER: 42" in answer


def test_canonical_registration_rejection_is_retained_as_structured_metadata(
    tmp_path: Path,
) -> None:
    backend = _backend()
    runtime = _runtime(tmp_path, _river_task(), backend)
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.middleware.configure_token_partitions(["SQ1"])
    source = ResearchSource(
        source_id="S-mismatch",
        title="Mismatch",
        url="https://example.test/mismatch",
        provider="fixture",
        provider_rank=1,
        acquisition_method="direct_http",
        relevance_tier="relevant",
        relevance_reason="fixture",
        passage="The canonical-looking passage says the bridge opened in 1974.",
        content_chars=60,
        fetch_status="success",
    )
    runtime.research_budget.sources.append(
        {
            "source_id": "S-mismatch",
            "url": source.url,
            "title": source.title,
        }
    )
    runtime.research_budget.evidence_graph.cache_page(
        "S-mismatch", "A different canonical page does not contain that passage."
    )
    verified = _verify(
        runtime=runtime,
        draft=DraftAnswer(
            claims=[
                DraftClaim(
                    claim_id="D1",
                    text="The bridge opened in 1974.",
                    source_ids=["S-mismatch"],
                    critical_for_final_answer=True,
                    subquestion_id="SQ1",
                )
            ]
        ),
        sources={"S-mismatch": source},
    )
    assert verified[0].status == "partially_supported"
    assert len(verified[0].registration_failures) == 1
    failure = verified[0].registration_failures[0]
    assert failure["category"] == "canonical_registration_rejected"
    assert failure["exception_type"] == "EvidenceQuoteMismatch"
    assert failure["source_id"] == "S-mismatch"
    assert "EvidenceQuoteMismatch" in failure["safe_traceback"]


def test_strict_preflight_remains_available_with_controlled_fixture(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    strict_config = _config(tmp_path, backend, mode="strict")
    permissive_config = _config(tmp_path / "permissive", backend)
    assert strict_config.fairness_fingerprint != permissive_config.fairness_fingerprint
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).preflight(task, strict_config)
    assert result["agent_constructed"] is True
    assert result["model_invocations"] == 0


def test_permissive_experiment_condition_has_one_shared_fairness_fingerprint(
    tmp_path: Path,
) -> None:
    backend = _backend()
    configs = [
        resolve_system_config(
            system_id,
            "sha256:" + ("f" * 64),
            17,
            tmp_path / system_id,
            overrides={
                "fixture_revision": backend.revision,
                "runtime_mode": "permissive",
            },
        )
        for system_id in ("simple_react", "vanilla_deepagents", "tongagent")
    ]
    assert len({item.fairness_fingerprint for item in configs}) == 1
    assert all(item.runtime_mode == "permissive" for item in configs)
