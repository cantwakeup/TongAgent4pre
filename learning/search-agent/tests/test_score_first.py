"""Socket-disabled coverage for the minimal Score-First TongAgent path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from evaluation.budget import ExecutionBudget
from evaluation.execution import resolve_system_config
from evaluation.offline import FixtureBackend, FixtureChatModel
from evaluation.schema import AnswerStatus, CompletionStatus, EvalTask
from evaluation.systems.common import prepare_runtime
from evaluation.systems.permissive import PermissiveSubquestion, research_query
from evaluation.systems.score_first import (
    LightweightPlan,
    ScoreFirstNote,
    ScoreFirstOperand,
    _plan,
    deterministic_calculation,
)
from evaluation.systems.tongagent import TongAgentRunner
from evaluation.tracing import TraceCollector


pytestmark = pytest.mark.usefixtures("socket_disabled")

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "permissive_controlled"
)


def _backend() -> FixtureBackend:
    return FixtureBackend.from_directory(_FIXTURES)


def _config(tmp_path: Path, backend: FixtureBackend, *, system: str = "tongagent"):
    return resolve_system_config(
        system,
        "sha256:" + "7" * 64,
        17,
        tmp_path,
        fixture_directory="evaluation/fixtures/permissive_controlled",
        overrides={"fixture_revision": backend.revision, "runtime_mode": "score_first"},
    )


def _river_task() -> EvalTask:
    return EvalTask(
        id="score-first-controlled",
        question="Which river passes through the valley containing Lunarite Mine?",
        reference_answer="Arden River",
        metadata={
            "source_urls": ["https://must-not-reach-the-model.invalid/secret"],
            "structured_outputs": {
                "LightweightPlan": {
                    "subquestions": ["Lunarite Mine location", "Alton Valley river"],
                    "answer_type": "entity",
                    "calculation": "none",
                },
                "ScoreFirstNote_SQ1": {
                    "subquestion": "Lunarite Mine location",
                    "answer_candidate": "Alton Valley",
                    "relevant_passages": [
                        "The municipal mining archive records that Lunarite Mine is located in Alton Valley."
                    ],
                    "source_ids": ["S1"],
                    "unresolved": False,
                    "operands": [
                        {"name": "location", "value": "Alton Valley", "source_id": "S1"}
                    ],
                },
                "ScoreFirstNote_SQ2": {
                    "subquestion": "Alton Valley river",
                    "answer_candidate": "Arden River",
                    "relevant_passages": [
                        "The regional water survey states that the Arden River passes through Alton Valley before joining the lower plain."
                    ],
                    "source_ids": ["S2"],
                    "unresolved": False,
                    "operands": [
                        {"name": "river", "value": "Arden River", "source_id": "S2"}
                    ],
                },
                "ScoreFirstAnswer": {
                    "answer_text": "Arden River",
                    "rationale": "The mine is in Alton Valley, whose river is the Arden.",
                    "source_ids": ["S1", "S2"],
                    "confidence": "high",
                    "calculation_structured": False,
                },
            },
        },
    )


def test_score_first_preflight_builds_shared_runtime_without_model_calls(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    model = FixtureChatModel.from_task(task, system_id="tongagent")

    result = TongAgentRunner(fixture_backend=backend, model=model).preflight(
        task, _config(tmp_path, backend)
    )

    assert result["runtime_mode"] == "score_first"
    assert result["model_invocations"] == 0
    assert set(result["runtime_tools"]) >= {"web_search", "fetch_url"}
    assert model.runtime.snapshot() == []


def test_score_first_planner_falls_back_once_without_hidden_task_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = EvalTask(
        id="score-first-isolation",
        question="What is the public answer?",
        reference_answer="REFERENCE_SENTINEL",
        metadata={"source_urls": ["SOURCE_URL_SENTINEL"]},
    )
    prompts: list[str] = []

    def fail_once(**kwargs: object) -> BaseModel | None:
        prompts.append(str(kwargs["prompt"]))
        return None

    monkeypatch.setattr(
        "evaluation.systems.score_first._accounted_structured", fail_once
    )

    plan, fallback = _plan(runtime=object(), task=task)  # type: ignore[arg-type]

    assert fallback is True
    assert plan.subquestions == [task.question]
    assert len(prompts) == 1
    assert task.question in prompts[0]
    assert "REFERENCE_SENTINEL" not in prompts[0]
    assert "SOURCE_URL_SENTINEL" not in prompts[0]


def test_score_first_controlled_two_hop_answers_without_evidence_gate(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, _config(tmp_path, backend))

    assert result.runtime_mode == "score_first"
    assert result.completion_status == CompletionStatus.COMPLETED
    assert result.answer_status == AnswerStatus.ANSWER
    assert result.extracted_answer == "Arden River"
    assert result.normalized_exact_match is True
    assert result.evidence_count == 0
    assert result.structural_subquestion_coverage == 1.0
    assert result.search_calls == 2
    assert result.fetch_calls == 3
    assert result.workflow_metrics is not None
    assert result.workflow_metrics["answer_rate"] == 1.0
    assert result.workflow_metrics["posthoc_verified_status"] == "source_mapped"
    native = tmp_path / "native" / "tongagent"
    for name in (
        "score_first_plan.json",
        "score_first_research.json",
        "score_first_calculation.json",
        "score_first_answer.json",
        "posthoc_reliability.json",
    ):
        assert json.loads((native / name).read_text())["schema_version"] == 1


def test_score_first_one_source_can_answer_without_evidence_graph(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task().model_copy(
        update={
            "question": "Where is Lunarite Mine located?",
            "reference_answer": "Alton Valley",
            "metadata": {
                "structured_outputs": {
                    "LightweightPlan": {
                        "subquestions": ["Lunarite Mine location"],
                        "answer_type": "entity",
                        "calculation": "none",
                    },
                    "ScoreFirstNote_SQ1": _river_task().metadata["structured_outputs"][
                        "ScoreFirstNote_SQ1"
                    ],
                    "ScoreFirstAnswer": {
                        "answer_text": "Alton Valley",
                        "rationale": "The fetched archive names the valley.",
                        "source_ids": ["S1"],
                        "confidence": "medium",
                        "calculation_structured": False,
                    },
                }
            },
        }
    )
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, _config(tmp_path, backend))

    assert result.extracted_answer == "Alton Valley"
    assert result.answer_status == AnswerStatus.ANSWER
    assert result.evidence_count == 0
    assert result.citations


def test_score_first_all_sources_missing_is_nonempty_abstain(tmp_path: Path) -> None:
    backend = FixtureBackend(searches={}, pages={})
    task = EvalTask(
        id="score-first-no-source",
        question="Where is the imaginary North Beacon?",
        metadata={
            "structured_outputs": {
                "LightweightPlan": {
                    "subquestions": ["North Beacon location"],
                    "answer_type": "entity",
                    "calculation": "none",
                }
            }
        },
    )
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, _config(tmp_path, backend))

    assert result.completion_status == CompletionStatus.PARTIAL
    assert result.answer_status == AnswerStatus.ABSTAIN
    assert result.final_answer == "FINAL_ANSWER: ABSTAIN"
    assert result.workflow_metrics is not None
    assert result.workflow_metrics["answer_rate"] == 0.0


@pytest.mark.parametrize(
    ("plan", "operands", "expected"),
    [
        (
            LightweightPlan(
                subquestions=["a", "b"],
                answer_type="date_difference",
                calculation="absolute_difference",
            ),
            [("first", "1974", "S1"), ("second", "1968", "S2")],
            "6",
        ),
        (
            LightweightPlan(
                subquestions=["a", "b"],
                answer_type="numeric_difference",
                calculation="subtract",
            ),
            [("first", "125.5", "S1"), ("second", "20", "S2")],
            "105.5",
        ),
        (
            LightweightPlan(
                subquestions=["items"], answer_type="count", calculation="count"
            ),
            [("item", "Alpha", "S1"), ("item", "Beta", "S1")],
            "2",
        ),
        (
            LightweightPlan(
                subquestions=["a", "b"],
                answer_type="comparison",
                calculation="compare",
            ),
            [("first", "same", "S1"), ("second", "same", "S2")],
            "yes",
        ),
    ],
)
def test_score_first_deterministic_calculation(
    plan: LightweightPlan,
    operands: list[tuple[str, str, str]],
    expected: str,
) -> None:
    notes = [
        ScoreFirstNote(
            subquestion="fixture",
            answer_candidate="candidate",
            source_ids=sorted({source for _, _, source in operands}),
            operands=[
                ScoreFirstOperand(name=name, value=value, source_id=source)
                for name, value, source in operands
            ],
        )
    ]

    result = deterministic_calculation(plan, notes)

    assert result.status == "success"
    assert result.answer_text == expected


def test_score_first_fairness_fingerprint_matches_all_three_systems(
    tmp_path: Path,
) -> None:
    backend = _backend()
    configs = [
        _config(tmp_path / system, backend, system=system)
        for system in ("simple_react", "vanilla_deepagents", "tongagent")
    ]

    assert {config.runtime_mode for config in configs} == {"score_first"}
    assert len({config.fairness_fingerprint for config in configs}) == 1
    assert (
        len(
            {
                json.dumps(config.fairness_payload(), sort_keys=True)
                for config in configs
            }
        )
        == 1
    )


def test_score_first_source_selection_bounds_fetch_attempts_per_subquestion(
    tmp_path: Path,
) -> None:
    urls = [f"https://source-{index}.fixture.test/page" for index in range(3)]
    backend = FixtureBackend(
        searches={
            "target entity attribute": {
                "results": [
                    {
                        "title": f"Target Entity source {index}",
                        "url": url,
                        "snippet": "Target Entity attribute value.",
                    }
                    for index, url in enumerate(urls, start=1)
                ]
            }
        },
        pages={
            url: {
                "title": f"Target Entity source {index}",
                "content": "Target Entity attribute value is documented here. " * 20,
            }
            for index, url in enumerate(urls, start=1)
        },
    )
    task = EvalTask(id="bounded-selection", question="Target Entity attribute")
    config = _config(tmp_path, backend)
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
        enable_tongagent_token_control=True,
        enable_tongagent_context_compaction=True,
        reserve_final_synthesis=False,
    )
    runtime.research_budget.configure_subquestions(["SQ1"])
    runtime.research_budget.activate_subquestion("SQ1")
    tools = {tool.name: tool for tool in runtime.tools}

    bundle = research_query(
        query="Target Entity attribute",
        task_type="single_fact_lookup",
        subquestion=PermissiveSubquestion(
            id="SQ1",
            question=task.question,
            task_type="single_fact_lookup",
            required_for_final_answer=True,
        ),
        search_tool=tools["web_search"],
        fetch_tool=tools["fetch_url"],
        max_sources=2,
        max_fetch_attempts=1,
    )

    assert len(bundle.sources) == 1
    assert runtime.research_budget.snapshot()["fetch_calls"] == 1


def test_score_first_posthoc_audit_never_changes_benchmark_answer(
    tmp_path: Path,
) -> None:
    backend = _backend()
    task = _river_task()
    result = TongAgentRunner(
        fixture_backend=backend,
        model=FixtureChatModel.from_task(task, system_id="tongagent"),
    ).run(task, _config(tmp_path, backend))
    audit = json.loads(
        (tmp_path / "native" / "tongagent" / "posthoc_reliability.json").read_text()
    )

    assert audit["score_first_answer_unchanged"] is True
    assert audit["method"] == "source_mapping_only_not_claim_verification"
    assert result.final_answer == "FINAL_ANSWER: Arden River"
