"""Serializable state contracts for TongAgent's explicit research workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from deepagents.graph import DeepAgentState


SubquestionStatus = Literal["pending", "researching", "covered", "blocked"]
PlanStatus = Literal["pending", "in_progress", "completed", "partial"]
EvidenceStance = Literal["supports", "contradicts"]
ClaimStatus = Literal["supported", "contradicted", "contested"]
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
    claim_ids: list[str]
    conflict_ids: list[str]
    note: str


class ResearchPlan(TypedDict):
    """Machine-readable plan whose progress is persisted in checkpoints."""

    plan_id: str
    evidence_schema_version: int
    question: str
    objective: str
    planner: str
    status: PlanStatus
    coverage: float
    completion_criteria: list[str]
    subquestions: list[SubQuestion]


class ClaimRecord(TypedDict):
    """One normalized proposition linked to canonical evidence units."""

    claim_id: str
    subquestion_id: str
    text: str
    status: ClaimStatus
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    source_ids: list[str]


class EvidenceUnit(TypedDict):
    """One exact page excerpt attached to a claim with an explicit stance."""

    evidence_id: str
    claim_id: str
    subquestion_id: str
    source_id: str
    stance: EvidenceStance
    quote: str
    quote_sha256: str
    source_content_sha256: str
    url: str
    title: str
    evidence_quality: str


class ConflictRecord(TypedDict):
    """A deterministic unresolved support/contradiction pair for one claim."""

    conflict_id: str
    claim_id: str
    subquestion_id: str
    status: Literal["unresolved"]
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    source_ids: list[str]


class BudgetState(TypedDict, total=False):
    """Serializable snapshot of the process-local research budget."""

    effort: str
    search_calls: int
    successful_searches: int
    max_searches: int
    fetch_calls: int
    max_fetches: int
    min_successful_sources: int
    successful_sources: list[dict[str, Any]]
    failed_sources: list[dict[str, Any]]
    next_source_sequence: int
    active_subquestion_id: str | None
    subquestion_limits: dict[str, dict[str, int]]
    subquestion_usage: dict[str, dict[str, int]]
    evidence_graph_version: int
    claims: list[ClaimRecord]
    evidence_units: list[EvidenceUnit]
    conflicts: list[ConflictRecord]


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
    active_evidence_ids_before: list[str]
    budget_state: BudgetState
    workflow_phase: WorkflowPhase
    research_cycles: int
    max_research_cycles: int
