"""Serializable state contracts for TongAgent's explicit research workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from deepagents.graph import DeepAgentState


SubquestionStatus = Literal["pending", "researching", "covered", "blocked"]
PlanStatus = Literal["pending", "in_progress", "completed", "partial"]
WorkflowPhase = Literal[
    "planning",
    "selecting",
    "researching",
    "evaluating",
    "reporting",
    "finished",
]


class SubQuestion(TypedDict):
    """One durable unit of work in a research plan."""

    id: str
    question: str
    rationale: str
    depends_on: list[str]
    status: SubquestionStatus
    attempts: int
    max_attempts: int
    evidence_source_ids: list[str]
    note: str


class ResearchPlan(TypedDict):
    """Machine-readable plan whose progress is persisted in checkpoints."""

    plan_id: str
    question: str
    objective: str
    planner: str
    status: PlanStatus
    coverage: float
    completion_criteria: list[str]
    subquestions: list[SubQuestion]


class BudgetState(TypedDict, total=False):
    """Serializable snapshot of the process-local research budget."""

    effort: str
    search_calls: int
    max_searches: int
    fetch_calls: int
    max_fetches: int
    min_successful_sources: int
    successful_sources: list[dict[str, Any]]
    failed_sources: list[dict[str, Any]]
    active_subquestion_id: str | None
    subquestion_limits: dict[str, dict[str, int]]
    subquestion_usage: dict[str, dict[str, int]]


class ResearchEvent(TypedDict, total=False):
    """One ordered, secret-free state transition event."""

    event_id: str
    sequence: int
    occurred_at: str
    event: str
    plan_id: str
    subquestion_id: str
    details: dict[str, Any]


class TongAgentState(DeepAgentState, total=False):
    """Deep Agent messages plus resumable TongAgent research state."""

    research_topic: str
    research_plan: ResearchPlan
    research_plan_history: list[ResearchPlan]
    research_events: list[ResearchEvent]
    active_subquestion_id: str | None
    active_source_ids_before: list[str]
    budget_state: BudgetState
    workflow_phase: WorkflowPhase
    research_cycles: int
    max_research_cycles: int
