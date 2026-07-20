"""Serializable state contracts for TongAgent's explicit research workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from deepagents.graph import DeepAgentState


SubquestionStatus = Literal["pending", "researching", "covered", "blocked"]
PlanStatus = Literal["pending", "in_progress", "completed", "partial"]
EvidenceStance = Literal["supports", "contradicts"]
ClaimStatus = Literal["supported", "contradicted", "contested"]
ResearchStrategy = Literal["fixed", "adaptive"]
ControlAction = Literal[
    "continue",
    "expand_budget",
    "stop_subquestion",
    "finish_success",
    "finish_partial",
    "fail_closed",
]
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
    structural_closure_validated: bool
    note: str


class ResearchPlan(TypedDict):
    """Machine-readable plan whose progress is persisted in checkpoints."""

    plan_id: str
    evidence_schema_version: int
    question: str
    objective: str
    planner: str
    status: PlanStatus
    structural_subquestion_coverage: float | None
    coverage: float | None
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
    search_metric_semantics_version: int
    provider_successes: int | None
    nonempty_searches: int | None
    successful_searches: int | None
    relevant_searches: int | None
    evidence_producing_searches: int | None
    search_metric_availability: dict[str, str]
    max_searches: int
    fetch_calls: int
    max_fetches: int
    min_successful_sources: int
    successful_sources: list[dict[str, Any]]
    failed_sources: list[dict[str, Any]]
    next_source_sequence: int
    active_subquestion_id: str | None
    subquestion_limits: dict[str, dict[str, int]]
    subquestion_usage: dict[str, dict[str, int | None]]
    strategy: ResearchStrategy
    granted_searches: int
    granted_fetches: int
    reserve_searches: int
    reserve_fetches: int
    applied_grant_ids: list[str]
    applied_grants: list[dict[str, Any]]
    next_tool_attempt_sequence: int
    tool_attempts: list[ToolAttempt]
    evidence_graph_version: int
    claims: list[ClaimRecord]
    evidence_units: list[EvidenceUnit]
    conflicts: list[ConflictRecord]


class ToolAttempt(TypedDict, total=False):
    """One structured network-tool invocation, including denied attempts."""

    attempt_id: str
    sequence: int
    subquestion_id: str
    tool: Literal["web_search", "fetch_url"]
    target: str
    outcome: str
    failure_class: str
    retryable: bool
    status: str
    error: str
    provider_success: bool
    provider_outcome: Literal["success", "failure", "not_called"]
    nonempty_search: bool
    relevant_search: bool
    evidence_producing_search: bool
    provider_failure: bool
    relevant_results: int
    result_urls: list[str]
    relevant_result_urls: list[str]


class ControlAssessment(TypedDict, total=False):
    """Deterministic evidence and budget signals evaluated after one cycle."""

    subquestion_id: str
    cycle: int
    plan_status: PlanStatus
    new_source_ids: list[str]
    new_claim_ids: list[str]
    new_evidence_ids: list[str]
    new_conflict_ids: list[str]
    new_tool_attempt_ids: list[str]
    remaining_searches: int
    remaining_fetches: int
    reserve_searches: int
    reserve_fetches: int
    no_progress_streak: int
    provider_successes: int | None
    nonempty_searches: int | None
    relevant_searches: int | None
    current_subquestion_relevant_searches: int | None
    evidence_producing_searches: int | None
    search_metric_availability: dict[str, str]
    distinct_content_revision_count: int
    distinct_source_host_count: int
    corroborating_source_group_count: int
    current_search_attempts: list[dict[str, Any]]
    reason_codes: list[str]
    integrity_errors: list[str]


class ControlDecision(TypedDict, total=False):
    """One checkpointed controller decision and its budget transition."""

    decision_id: str
    sequence: int
    cycle: int
    subquestion_id: str
    action: ControlAction
    reason_codes: list[str]
    budget_before: dict[str, int]
    budget_after: dict[str, int]


class AdaptiveControlState(TypedDict, total=False):
    """Checkpointable runtime policy state for Stage 03D."""

    schema_version: int
    strategy: ResearchStrategy
    config_fingerprint: str
    hard_effort: str
    pinned_model: str
    pinned_topology: str
    max_escalations: int
    escalation_count: int
    escalations_by_subquestion: dict[str, int]
    no_progress_streak_by_subquestion: dict[str, int]
    last_assessment: ControlAssessment
    decision_history: list[ControlDecision]
    stop_reason: str


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
    active_claim_ids_before: list[str]
    active_evidence_ids_before: list[str]
    active_conflict_ids_before: list[str]
    active_tool_attempt_sequence_before: int
    budget_state: BudgetState
    adaptive_control: AdaptiveControlState
    workflow_phase: WorkflowPhase
    research_cycles: int
    max_research_cycles: int
    report_markdown: str
    compact_checkpoints: list[dict[str, Any]]
    active_token_slice_exhausted: dict[str, Any] | None
