"""Explicit, checkpointable research planning workflow for TongAgent."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from hashlib import sha256
from typing import Any, Literal, cast
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from adaptive_control import (
    build_control_assessment,
    decide_control_action,
    decision_id_for,
    initialize_control_state,
    record_control_decision,
)
from evidence_graph import (
    EVIDENCE_GRAPH_VERSION,
    EvidenceQuoteMismatch,
    allowed_report_caveat_lines,
    corroborating_evidence_source_ids,
    normalize_evidence_text,
    validate_evidence_graph,
)
from retrieval_quality import classify_query_task_type
from research_state import (
    AdaptiveControlState,
    BudgetState,
    EvidenceStance,
    ResearchEvent,
    ResearchPlan,
    ResearchStrategy,
    SubQuestion,
    SubquestionStatus,
    TongAgentState,
)
from telemetry import append_research_event


Planner = Callable[[str, int], ResearchPlan]
BudgetSnapshot = Callable[[], dict[str, Any]]
BudgetConfigure = Callable[[list[str]], None]
BudgetActivate = Callable[[str | None], None]
BudgetGrant = Callable[..., dict[str, Any]]
ReportRead = Callable[[], str]
ReportClear = Callable[[], None]
TokenBudgetSnapshot = Callable[[], dict[str, Any]]
TokenBudgetCanStart = Callable[[str], bool]
ModelBudgetSnapshot = Callable[[], dict[str, Any]]
EvidenceRecord = Callable[..., dict[str, Any]]
PlanIdFactory = Callable[[], str]
Route = Literal["research", "report"]
ControlRoute = Literal["select", "report"]
SOURCE_ID_PATTERN = re.compile(r"^S[1-9][0-9]*$")


class DraftSubquestion(BaseModel):
    """Planner-facing subquestion before durable IDs and status are assigned."""

    question: str = Field(min_length=3)
    rationale: str = ""
    depends_on: list[int] = Field(default_factory=list)


class PlanDraft(BaseModel):
    """Structured output contract used by the low-cost planning call."""

    objective: str = Field(min_length=3)
    subquestions: list[DraftSubquestion] = Field(min_length=1)
    completion_criteria: list[str] = Field(default_factory=list)


class ActiveResearchContext(BaseModel):
    """Compact code-owned state injected before one active SQ is researched."""

    plan_id: str
    active_subquestion: dict[str, Any]
    subquestion_status: str
    query_task_type: str
    remaining_search_budget: int
    remaining_fetch_budget: int
    remaining_model_calls: int | None
    relevant_sources: list[dict[str, Any]] = Field(default_factory=list)
    canonical_claims: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)


def create_research_plan(
    topic: str,
    subquestions: Sequence[DraftSubquestion | dict[str, Any] | str],
    *,
    objective: str | None = None,
    completion_criteria: Sequence[str] = (),
    planner: str = "deterministic",
    max_subquestions: int = 3,
    max_attempts: int = 2,
    plan_id_factory: PlanIdFactory | None = None,
) -> ResearchPlan:
    """Normalize planner output into a safe, durable research plan.

    Args:
        topic: Original user question.
        subquestions: Planner output without runtime-owned fields.
        objective: Optional normalized research objective.
        completion_criteria: Human-readable plan completion requirements.
        planner: Planner implementation recorded for auditability.
        max_subquestions: Hard cap on plan breadth.
        max_attempts: Per-subquestion research attempt cap.
        plan_id_factory: Optional deterministic ID factory for tests.

    Returns:
        A pending plan with stable `SQ#` IDs and no model-controlled status.

    Raises:
        ValueError: If the topic or normalized subquestion list is empty.
    """
    normalized_topic = " ".join(topic.split())
    if not normalized_topic:
        msg = "Research topic must not be empty"
        raise ValueError(msg)
    limit = max(1, max_subquestions)
    normalized: list[DraftSubquestion] = []
    seen: set[str] = set()
    for raw in subquestions:
        if isinstance(raw, str):
            item = DraftSubquestion(question=raw)
        elif isinstance(raw, DraftSubquestion):
            item = raw
        else:
            item = DraftSubquestion.model_validate(raw)
        question = " ".join(item.question.split())
        key = question.casefold()
        if not question or key in seen:
            continue
        seen.add(key)
        normalized.append(
            DraftSubquestion(
                question=question,
                rationale=" ".join(item.rationale.split()),
                depends_on=list(item.depends_on),
            )
        )
        if len(normalized) == limit:
            break
    if not normalized:
        msg = "Research plan must contain at least one unique subquestion"
        raise ValueError(msg)

    durable: list[SubQuestion] = []
    for index, item in enumerate(normalized, start=1):
        dependencies = [
            f"SQ{dependency}"
            for dependency in item.depends_on
            if 1 <= dependency < index
        ]
        durable.append(
            {
                "id": f"SQ{index}",
                "question": item.question,
                "rationale": item.rationale,
                "depends_on": dependencies,
                "status": "pending",
                "attempts": 0,
                "max_attempts": max(1, max_attempts),
                "evidence_source_ids": [],
                "claim_ids": [],
                "conflict_ids": [],
                "structural_closure_validated": False,
                "note": "",
            }
        )
    criteria = [" ".join(item.split()) for item in completion_criteria if item.strip()]
    if not criteria:
        criteria = [
            "Every subquestion is covered or explicitly blocked with a reason.",
            "The final report cites successfully fetched sources for factual claims.",
        ]
    factory = plan_id_factory or (lambda: str(uuid4()))
    return {
        "plan_id": factory(),
        "evidence_schema_version": 1,
        "question": normalized_topic,
        "objective": " ".join((objective or normalized_topic).split()),
        "planner": planner,
        "status": "pending",
        "structural_subquestion_coverage": 0.0,
        "coverage": 0.0,
        "completion_criteria": criteria,
        "subquestions": durable,
    }


def fallback_research_plan(
    topic: str,
    max_subquestions: int,
    *,
    planner: str = "deterministic-fallback",
) -> ResearchPlan:
    """Build a conservative plan when structured model planning is unavailable."""
    normalized = " ".join(topic.split())
    candidates = [
        DraftSubquestion(
            question=f"明确“{normalized}”的范围、关键概念和判断标准",
            rationale="Prevent an ambiguous question from producing an unfocused report.",
        ),
        DraftSubquestion(
            question=f"收集能够直接回答“{normalized}”的权威事实和一手证据",
            rationale="Ground the core answer in fetched evidence.",
            depends_on=[1],
        ),
        DraftSubquestion(
            question=f"检查“{normalized}”相关的限制、反例、冲突信息和不确定性",
            rationale="Expose unsupported generalizations and unresolved disagreement.",
            depends_on=[1],
        ),
    ]
    return create_research_plan(
        normalized,
        candidates,
        objective=f"形成对“{normalized}”的可引用、可审计回答",
        planner=planner,
        max_subquestions=max_subquestions,
    )


def build_model_planner(model: BaseChatModel) -> Planner:
    """Create a structured planner with a deterministic fallback.

    Args:
        model: Chat model used only for the initial plan call.

    Returns:
        Planner callable accepted by `build_research_graph`.
    """

    def plan(topic: str, max_subquestions: int) -> ResearchPlan:
        prompt = f"""Create an explicit web-research plan for the user question below.
Return between 1 and {max_subquestions} non-overlapping subquestions in dependency order.
Each subquestion must be independently researchable and materially necessary for the final answer.
Use dependencies for multi-hop questions. Do not include runtime status, source IDs, or invented facts.

User question: {topic}"""
        try:
            structured = model.with_structured_output(PlanDraft)
            raw = structured.invoke(prompt)
            draft = raw if isinstance(raw, PlanDraft) else PlanDraft.model_validate(raw)
            return create_research_plan(
                topic,
                draft.subquestions,
                objective=draft.objective,
                completion_criteria=draft.completion_criteria,
                planner="model",
                max_subquestions=max_subquestions,
            )
        except Exception as exc:  # noqa: BLE001  # Provider failures must fall back without losing the run.
            return fallback_research_plan(
                topic,
                max_subquestions,
                planner=f"deterministic-fallback:{type(exc).__name__}",
            )

    return plan


def calculate_structural_subquestion_coverage(plan: ResearchPlan) -> float | None:
    """Calculate the fraction of structurally covered subquestions.

    This is an internal workflow metric.  It does not measure answer accuracy,
    semantic facet coverage, citation entailment, or overall completeness.
    """
    if int(plan.get("evidence_schema_version", 0)) < EVIDENCE_GRAPH_VERSION:
        return None
    subquestions = plan.get("subquestions", [])
    if not subquestions:
        return 0.0
    covered = sum(
        item["status"] == "covered"
        and item.get("structural_closure_validated") is True
        and bool(item.get("claim_ids"))
        and bool(item.get("evidence_source_ids"))
        for item in subquestions
    )
    return round(covered / len(subquestions), 4)


def calculate_plan_coverage(plan: ResearchPlan) -> float | None:
    """Compatibility alias for `calculate_structural_subquestion_coverage`."""
    return calculate_structural_subquestion_coverage(plan)


def refresh_plan_status(plan: ResearchPlan) -> ResearchPlan:
    """Return a copied plan with code-derived coverage and aggregate status."""
    updated = deepcopy(plan)
    updated.setdefault("evidence_schema_version", 0)
    subquestions = updated.get("subquestions", [])
    for item in subquestions:
        item.setdefault("claim_ids", [])
        item.setdefault("conflict_ids", [])
        item.setdefault("structural_closure_validated", False)
    structural_coverage = calculate_structural_subquestion_coverage(updated)
    updated["structural_subquestion_coverage"] = structural_coverage
    # Deprecated compatibility alias retained for Stage 03A-D checkpoints.
    updated["coverage"] = structural_coverage
    statuses = {item["status"] for item in subquestions}
    schema_current = (
        int(updated.get("evidence_schema_version", 0)) >= EVIDENCE_GRAPH_VERSION
    )
    if (
        subquestions
        and statuses == {"covered"}
        and (not schema_current or structural_coverage == 1.0)
    ):
        updated["status"] = "completed"
    elif statuses.intersection({"pending", "researching"}):
        attempted = any(item["attempts"] for item in subquestions)
        updated["status"] = "in_progress" if attempted else "pending"
    else:
        updated["status"] = "partial"
    return updated


def transition_subquestion(
    plan: ResearchPlan,
    subquestion_id: str,
    status: SubquestionStatus,
    *,
    evidence_source_ids: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
    conflict_ids: Sequence[str] = (),
    note: str = "",
    increment_attempt: bool = False,
) -> ResearchPlan:
    """Apply one validated subquestion transition without mutating input state.

    Args:
        plan: Current durable plan.
        subquestion_id: Stable `SQ#` identifier.
        status: Desired next status.
        evidence_source_ids: Evidence IDs associated with a covered item.
        claim_ids: Canonical claim IDs associated with the item.
        conflict_ids: Canonical conflict IDs associated with the item.
        note: Short result or blocking explanation.
        increment_attempt: Whether this transition begins a new attempt.

    Returns:
        Updated aggregate plan.

    Raises:
        ValueError: If the item is unknown or the transition is invalid.
    """
    updated = deepcopy(plan)
    target = next(
        (item for item in updated["subquestions"] if item["id"] == subquestion_id),
        None,
    )
    if target is None:
        msg = f"Unknown subquestion: {subquestion_id}"
        raise ValueError(msg)
    current = target["status"]
    allowed: dict[SubquestionStatus, set[SubquestionStatus]] = {
        "pending": {"researching", "blocked"},
        "researching": {"pending", "covered", "blocked"},
        "covered": {"covered", "blocked"},
        "blocked": {"blocked"},
    }
    if status not in allowed[current]:
        msg = f"Invalid subquestion transition: {current} -> {status}"
        raise ValueError(msg)
    evidence = list(dict.fromkeys(evidence_source_ids))
    invalid_evidence = [
        source_id
        for source_id in evidence
        if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id)
    ]
    if invalid_evidence:
        msg = "Evidence source IDs must use the S# format"
        raise ValueError(msg)
    claims = list(dict.fromkeys(claim_ids))
    invalid_claims = [
        claim_id
        for claim_id in claims
        if not isinstance(claim_id, str) or not re.fullmatch(r"C[1-9][0-9]*", claim_id)
    ]
    if invalid_claims:
        msg = "Claim IDs must use the C# format"
        raise ValueError(msg)
    conflicts = list(dict.fromkeys(conflict_ids))
    invalid_conflicts = [
        conflict_id
        for conflict_id in conflicts
        if not isinstance(conflict_id, str)
        or not re.fullmatch(r"X[1-9][0-9]*", conflict_id)
    ]
    if invalid_conflicts:
        msg = "Conflict IDs must use the X# format"
        raise ValueError(msg)
    if status == "covered" and not evidence and not target["evidence_source_ids"]:
        msg = "Covered subquestions require at least one evidence source ID"
        raise ValueError(msg)
    if (
        status == "covered"
        and int(updated.get("evidence_schema_version", 0)) >= 1
        and not claims
        and not target.get("claim_ids", [])
    ):
        msg = "Evidence schema v1 covered subquestions require a canonical claim ID"
        raise ValueError(msg)
    if status == "blocked" and not note.strip():
        msg = "Blocked subquestions require an explanatory note"
        raise ValueError(msg)
    target["status"] = status
    identifiers_changed = bool(evidence or claims or conflicts)
    if status != "covered":
        target["structural_closure_validated"] = False
    elif identifiers_changed:
        target["structural_closure_validated"] = False
    if increment_attempt:
        target["attempts"] += 1
    if evidence:
        target["evidence_source_ids"] = list(
            dict.fromkeys([*target["evidence_source_ids"], *evidence])
        )
    if claims:
        target["claim_ids"] = list(
            dict.fromkeys([*target.get("claim_ids", []), *claims])
        )
    if conflicts:
        target["conflict_ids"] = list(
            dict.fromkeys([*target.get("conflict_ids", []), *conflicts])
        )
    if note.strip():
        target["note"] = " ".join(note.split())
    return refresh_plan_status(updated)


def select_next_subquestion(plan: ResearchPlan) -> tuple[ResearchPlan, str | None]:
    """Resume an active item or start the next dependency-ready item.

    Pending items whose prerequisites are blocked are blocked transitively rather
    than researched out of dependency order.
    """
    active = next(
        (item for item in plan["subquestions"] if item["status"] == "researching"),
        None,
    )
    if active is not None:
        return deepcopy(plan), active["id"]
    updated = deepcopy(plan)
    while True:
        blocked = {
            item["id"]
            for item in updated["subquestions"]
            if item["status"] == "blocked"
        }
        unreachable = [
            item
            for item in updated["subquestions"]
            if item["status"] == "pending"
            and set(item["depends_on"]).intersection(blocked)
        ]
        if not unreachable:
            break
        for item in unreachable:
            blocked_dependencies = sorted(set(item["depends_on"]).intersection(blocked))
            updated = transition_subquestion(
                updated,
                item["id"],
                "blocked",
                note=(
                    "Required dependencies were blocked: "
                    + ", ".join(blocked_dependencies)
                ),
            )

    covered = {
        item["id"] for item in updated["subquestions"] if item["status"] == "covered"
    }
    pending = [item for item in updated["subquestions"] if item["status"] == "pending"]
    candidate = next(
        (item for item in pending if set(item["depends_on"]).issubset(covered)),
        None,
    )
    if candidate is None:
        if pending:
            for item in pending:
                updated = transition_subquestion(
                    updated,
                    item["id"],
                    "blocked",
                    note="No dependency-ready research path remains.",
                )
        return refresh_plan_status(updated), None
    updated = transition_subquestion(
        updated,
        candidate["id"],
        "researching",
        increment_attempt=True,
    )
    return updated, candidate["id"]


def invalid_covered_subquestions(
    plan: ResearchPlan, snapshot: dict[str, Any]
) -> dict[str, list[str]]:
    """Audit every covered SQ against the checkpointed claim graph."""
    if int(plan.get("evidence_schema_version", 0)) < 1:
        return {}
    claims_by_id = {
        str(item.get("claim_id", "")): item for item in snapshot.get("claims", [])
    }
    evidence_units = snapshot.get("evidence_units", [])
    conflicts = snapshot.get("conflicts", [])
    graph_integrity_errors = validate_evidence_graph(snapshot)
    invalid: dict[str, list[str]] = {}
    for item in plan.get("subquestions", []):
        if item.get("status") != "covered":
            continue
        subquestion_id = str(item.get("id", ""))
        claim_ids = list(dict.fromkeys(item.get("claim_ids", [])))
        claim_id_set = set(claim_ids)
        reasons: list[str] = []
        if graph_integrity_errors:
            reasons.append(
                "evidence graph integrity validation failed: "
                + "; ".join(graph_integrity_errors)
            )
        scoped_usage = snapshot.get("subquestion_usage", {}).get(subquestion_id, {})
        scoped_relevant = scoped_usage.get("relevant_searches")
        if scoped_relevant is None:
            reasons.append("relevant search metric is unavailable for this subquestion")
        elif int(scoped_relevant) < 1:
            reasons.append("missing a relevant search in this subquestion")
        if not claim_ids:
            reasons.append("missing canonical claims")
        valid_claim_ids = {
            claim_id
            for claim_id in claim_ids
            if (
                (claim := claims_by_id.get(claim_id)) is not None
                and claim.get("subquestion_id") == subquestion_id
                and claim.get("status") in {"supported", "contested"}
                and claim.get("supporting_evidence_ids")
            )
        }
        if valid_claim_ids != claim_id_set:
            reasons.append("unknown, cross-SQ, or unsupported claims")
        attached_units = [
            unit
            for unit in evidence_units
            if str(unit.get("claim_id", "")) in claim_id_set
        ]
        derived_source_ids = {str(unit.get("source_id", "")) for unit in attached_units}
        if set(item.get("evidence_source_ids", [])) != derived_source_ids:
            reasons.append("source IDs do not match claim edges")
        derived_conflict_ids = {
            str(conflict.get("conflict_id", ""))
            for conflict in conflicts
            if str(conflict.get("claim_id", "")) in claim_id_set
        }
        if set(item.get("conflict_ids", [])) != derived_conflict_ids:
            reasons.append("conflict IDs do not match contested claims")
        if reasons:
            invalid[subquestion_id] = reasons
    subquestions = plan.get("subquestions", [])
    if subquestions and all(item.get("status") == "covered" for item in subquestions):
        plan_source_ids = {
            str(source_id)
            for item in subquestions
            for source_id in item.get("evidence_source_ids", [])
        }
        plan_claim_ids = {
            str(claim_id)
            for item in subquestions
            for claim_id in item.get("claim_ids", [])
        }
        corroborating_count = len(
            corroborating_evidence_source_ids(
                source_ids=plan_source_ids,
                sources=snapshot.get("successful_sources", []),
                evidence_units=evidence_units,
                claim_ids=plan_claim_ids,
            )
        )
        minimum_sources = int(snapshot.get("min_successful_sources", 0))
        if corroborating_count < minimum_sources:
            last_id = str(subquestions[-1].get("id", ""))
            invalid.setdefault(last_id, []).append(
                f"only {corroborating_count} of {minimum_sources} required "
                "corroborating source groups"
            )
        required_searches = min(
            int(snapshot.get("max_searches", 0)), max(1, len(subquestions))
        )
        raw_relevant_searches = snapshot.get("relevant_searches")
        if raw_relevant_searches is None:
            last_id = str(subquestions[-1].get("id", ""))
            invalid.setdefault(last_id, []).append(
                "the relevant search metric is unavailable"
            )
        elif int(raw_relevant_searches) < required_searches:
            relevant_searches = int(raw_relevant_searches)
            last_id = str(subquestions[-1].get("id", ""))
            invalid.setdefault(last_id, []).append(
                f"only {relevant_searches} of {required_searches} required relevant searches"
            )
    return invalid


def audit_structural_subquestion_closures(
    plan: ResearchPlan, snapshot: dict[str, Any]
) -> tuple[ResearchPlan, dict[str, list[str]]]:
    """Recompute durable closure-validation markers from the canonical ledger."""
    invalid = invalid_covered_subquestions(plan, snapshot)
    audited = deepcopy(plan)
    graph_required = int(audited.get("evidence_schema_version", 0)) >= 1
    for item in audited.get("subquestions", []):
        item["structural_closure_validated"] = bool(
            graph_required
            and item.get("status") == "covered"
            and str(item.get("id", "")) not in invalid
        )
    return refresh_plan_status(audited), invalid


def build_source_ledger_tool(budget_snapshot: BudgetSnapshot) -> BaseTool:
    """Expose the canonical ID/title/URL mapping shared by every agent role."""

    @tool("get_source_ledger")
    def get_source_ledger() -> str:
        """Return fetched sources, evidence quality, and the active SQ budget."""
        snapshot = budget_snapshot()
        active = snapshot.get("active_subquestion_id")
        return json.dumps(
            {
                "successful_sources": snapshot.get("successful_sources", []),
                "active_subquestion_id": active,
                "subquestion_limits": snapshot.get("subquestion_limits", {}).get(
                    active, {}
                ),
                "subquestion_usage": snapshot.get("subquestion_usage", {}).get(
                    active, {}
                ),
                "evidence_graph_counts": {
                    "claims": len(snapshot.get("claims", [])),
                    "evidence_units": len(snapshot.get("evidence_units", [])),
                    "conflicts": len(snapshot.get("conflicts", [])),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    return get_source_ledger


def _evidence_retry_identity(source_id: str, claim: str, quote: str) -> tuple[str, str]:
    """Return stable, content-safe identifiers for one evidence attempt."""

    claim_digest = sha256(
        normalize_evidence_text(claim).casefold().encode()
    ).hexdigest()
    quote_digest = sha256(normalize_evidence_text(quote).encode()).hexdigest()
    evidence_key = f"{source_id}:{claim_digest}"
    return evidence_key, f"{evidence_key}:{quote_digest}:quote_mismatch"


def _active_canonical_state(
    snapshot: dict[str, Any],
    active_subquestion_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    """Return supported claims, their evidence, source IDs, and conflict IDs."""

    claims = [
        dict(item)
        for item in snapshot.get("claims", [])
        if item.get("subquestion_id") == active_subquestion_id
        and item.get("status") in {"supported", "contested"}
    ]
    claim_ids = {str(item.get("claim_id", "")) for item in claims}
    evidence = [
        dict(item)
        for item in snapshot.get("evidence_units", [])
        if str(item.get("claim_id", "")) in claim_ids
    ]
    source_ids = list(
        dict.fromkeys(
            str(item.get("source_id", "")) for item in evidence if item.get("source_id")
        )
    )
    conflict_ids = [
        str(item.get("conflict_id", ""))
        for item in snapshot.get("conflicts", [])
        if str(item.get("claim_id", "")) in claim_ids
    ]
    return claims, evidence, source_ids, conflict_ids


def _unresolved_active_requirements(
    snapshot: dict[str, Any],
    active_subquestion_id: str,
) -> list[str]:
    """Describe the minimum deterministic gates still missing for one SQ."""

    claims, _, source_ids, _ = _active_canonical_state(
        snapshot,
        active_subquestion_id,
    )
    scoped = snapshot.get("subquestion_usage", {}).get(active_subquestion_id, {})
    unresolved: list[str] = []
    if int(scoped.get("relevant_searches") or 0) < 1:
        unresolved.append("one relevant search")
    if not source_ids:
        unresolved.append("one successfully fetched canonical source")
    if not claims:
        unresolved.append("one supported canonical claim with exact evidence")
    return unresolved


def _auto_close_after_evidence(
    *,
    runtime: ToolRuntime,
    snapshot: dict[str, Any],
    events: list[ResearchEvent],
) -> tuple[ResearchPlan | None, list[ResearchEvent], list[str]]:
    """Close a locally supported SQ without another model-owned state update."""

    active_id = str(runtime.state.get("active_subquestion_id") or "")
    plan = cast("ResearchPlan", runtime.state.get("research_plan", {}))
    if not active_id or not plan:
        return None, events, ["active subquestion state"]
    unresolved = _unresolved_active_requirements(snapshot, active_id)
    if unresolved:
        return None, events, unresolved
    claims, _, source_ids, conflict_ids = _active_canonical_state(snapshot, active_id)
    claim_ids = [str(item.get("claim_id", "")) for item in claims]
    try:
        updated = transition_subquestion(
            plan,
            active_id,
            "covered",
            evidence_source_ids=source_ids,
            claim_ids=claim_ids,
            conflict_ids=conflict_ids,
            note="Minimum canonical evidence was registered; closure was automatic.",
        )
        updated, closure_errors = audit_structural_subquestion_closures(
            updated,
            snapshot,
        )
    except ValueError as exc:
        return None, events, [str(exc)]
    if active_id in closure_errors:
        return None, events, closure_errors[active_id]
    next_events = append_research_event(
        events,
        "subquestion_updated",
        plan_id=updated["plan_id"],
        subquestion_id=active_id,
        details={
            "status": "covered",
            "automatic": True,
            "evidence_source_ids": source_ids,
            "claim_ids": claim_ids,
            "conflict_ids": conflict_ids,
        },
    )
    return updated, next_events, []


def build_evidence_graph_tools(
    evidence_record: EvidenceRecord,
    budget_snapshot: BudgetSnapshot,
    *,
    auto_update_subquestion: bool = False,
) -> list[BaseTool]:
    """Build tools for exact-excerpt registration and graph inspection."""

    @tool("record_evidence")
    def record_evidence(
        source_id: str,
        claim: str,
        quote: str,
        runtime: ToolRuntime,
        stance: EvidenceStance = "supports",
        claim_id: str = "",
    ) -> str | Command:
        """Link an exact source excerpt to a code-assigned canonical claim.

        The claim must be a self-contained report-ready proposition with a
        subject and predicate, never a topic label or field name. Omit claim_id
        to create it. Pass claim_id only when reusing an existing C# returned by
        a successful earlier call; never invent a C#.
        """
        events = cast("list[ResearchEvent]", runtime.state.get("research_events", []))
        try:
            result = evidence_record(
                source_id=source_id,
                claim=claim,
                quote=quote,
                stance=stance,
                claim_id=claim_id,
            )
        except EvidenceQuoteMismatch as exc:
            evidence_key, signature = _evidence_retry_identity(source_id, claim, quote)
            prior_failures = [
                item
                for item in events
                if item.get("event") == "evidence_registration_failed"
                and item.get("details", {}).get("evidence_key") == evidence_key
            ]
            duplicate = any(
                item.get("details", {}).get("signature") == signature
                for item in prior_failures
            )
            retryable = not duplicate and not prior_failures
            status = "duplicate_retry_blocked" if duplicate else "quote_mismatch"
            payload = {
                "status": status,
                "error": str(exc),
                "retryable": retryable,
                "candidate_quotes": exc.candidate_quotes,
                "normalized_similarity": exc.normalized_similarity,
                "retry_limit": 1,
            }
            next_events = append_research_event(
                events,
                "evidence_registration_failed",
                plan_id=str(runtime.state.get("research_plan", {}).get("plan_id", "")),
                subquestion_id=str(runtime.state.get("active_subquestion_id") or ""),
                details={
                    "failure_type": status,
                    "evidence_key": evidence_key,
                    "signature": signature,
                    "retryable": retryable,
                    "candidate_count": len(exc.candidate_quotes),
                    "normalized_similarity": exc.normalized_similarity,
                },
            )
            content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            return Command(
                update={
                    "research_events": next_events,
                    "messages": [
                        ToolMessage(
                            content=content,
                            tool_call_id=runtime.tool_call_id,
                            name="record_evidence",
                            status="error",
                        )
                    ],
                }
            )
        except ValueError as exc:
            payload = {"status": "error", "error": str(exc), "retryable": False}
            next_events = append_research_event(
                events,
                "evidence_registration_failed",
                plan_id=str(runtime.state.get("research_plan", {}).get("plan_id", "")),
                subquestion_id=str(runtime.state.get("active_subquestion_id") or ""),
                details={
                    "failure_type": "validation_error",
                    "retryable": False,
                },
            )
            return Command(
                update={
                    "research_events": next_events,
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            tool_call_id=runtime.tool_call_id,
                            name="record_evidence",
                            status="error",
                        )
                    ],
                }
            )

        snapshot = budget_snapshot()
        active_id = str(runtime.state.get("active_subquestion_id") or "")
        registered_events = append_research_event(
            events,
            "evidence_registered",
            plan_id=str(runtime.state.get("research_plan", {}).get("plan_id", "")),
            subquestion_id=active_id,
            details={
                "claim_id": result.get("claim", {}).get("claim_id"),
                "evidence_id": result.get("evidence", {}).get("evidence_id"),
                "source_id": source_id,
            },
        )
        updated_plan: ResearchPlan | None = None
        unresolved = _unresolved_active_requirements(snapshot, active_id)
        if auto_update_subquestion:
            updated_plan, registered_events, unresolved = _auto_close_after_evidence(
                runtime=runtime,
                snapshot=snapshot,
                events=registered_events,
            )
        payload = {
            "status": "success",
            **result,
            "subquestion_auto_updated": updated_plan is not None,
            "next_state": "covered" if updated_plan is not None else "researching",
            "unresolved_requirements": unresolved,
        }
        update: dict[str, Any] = {
            "research_events": registered_events,
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    tool_call_id=runtime.tool_call_id,
                    name="record_evidence",
                )
            ],
        }
        if updated_plan is not None:
            update["research_plan"] = updated_plan
        return Command(update=update)

    @tool("get_evidence_graph")
    def get_evidence_graph() -> str:
        """Return canonical claims, exact excerpts, and unresolved conflicts."""
        snapshot = budget_snapshot()
        return json.dumps(
            {
                "evidence_graph_version": snapshot.get("evidence_graph_version", 0),
                "claims": snapshot.get("claims", []),
                "evidence_units": snapshot.get("evidence_units", []),
                "conflicts": snapshot.get("conflicts", []),
            },
            ensure_ascii=False,
            indent=2,
        )

    return [record_evidence, get_evidence_graph]


def _has_current_researcher_delegation(
    messages: Sequence[Any], subquestion_id: str
) -> bool:
    """Return whether the current SQ has a completed researcher task."""
    start: int | None = None
    for index, message in enumerate(messages):
        message_id = (
            message.get("id", "")
            if isinstance(message, dict)
            else getattr(message, "id", "") or ""
        )
        if str(message_id).startswith("research-step-"):
            start = index
    if start is None:
        return False
    pending_call_ids: set[str] = set()
    expected_marker = f"[SQ:{subquestion_id}]"
    for message in messages[start:]:
        tool_calls = (
            message.get("tool_calls", [])
            if isinstance(message, dict)
            else getattr(message, "tool_calls", []) or []
        )
        for call in tool_calls:
            args = (
                call.get("args", {})
                if isinstance(call, dict)
                else getattr(call, "args", {}) or {}
            )
            name = (
                call.get("name", "")
                if isinstance(call, dict)
                else getattr(call, "name", "") or ""
            )
            call_id = (
                call.get("id", "")
                if isinstance(call, dict)
                else getattr(call, "id", "") or ""
            )
            description = str(args.get("description", "")).lstrip()
            if (
                name == "task"
                and args.get("subagent_type") == "researcher"
                and description.startswith(expected_marker)
                and call_id
            ):
                pending_call_ids.add(str(call_id))
        tool_call_id = (
            message.get("tool_call_id", "")
            if isinstance(message, dict)
            else getattr(message, "tool_call_id", "") or ""
        )
        if str(tool_call_id) not in pending_call_ids:
            continue
        status = (
            message.get("status", "success")
            if isinstance(message, dict)
            else getattr(message, "status", "success") or "success"
        )
        if status == "success":
            return True
        pending_call_ids.discard(str(tool_call_id))
    return False


def build_research_state_tools(
    budget_snapshot: BudgetSnapshot, *, require_researcher: bool = False
) -> list[BaseTool]:
    """Build tools that expose and persist ledger-validated plan progress."""

    @tool("get_research_plan")
    def get_research_plan(runtime: ToolRuntime) -> str:
        """Return the active structured plan and subquestion statuses."""
        plan = runtime.state.get("research_plan", {})
        return json.dumps(plan, ensure_ascii=False, indent=2)

    @tool("update_subquestion")
    def update_subquestion(
        subquestion_id: str,
        status: Literal["covered", "blocked"],
        evidence_source_ids: list[str],
        note: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """Mark the active subquestion covered or blocked with explicit evidence."""
        plan = cast("ResearchPlan", runtime.state.get("research_plan", {}))
        if not plan:
            return json.dumps(
                {"status": "error", "error": "No active research plan"},
                ensure_ascii=False,
            )
        active_subquestion_id = runtime.state.get("active_subquestion_id")
        if subquestion_id != active_subquestion_id:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Only the active subquestion can be updated: "
                        f"active={active_subquestion_id or 'none'}"
                    ),
                },
                ensure_ascii=False,
            )
        if require_researcher and not _has_current_researcher_delegation(
            runtime.state.get("messages", []), str(active_subquestion_id)
        ):
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "Multi-agent research requires task(subagent_type='researcher') "
                        "during the current research step before update_subquestion. "
                        f"Prefix its description with [SQ:{active_subquestion_id}], "
                        "wait for its successful result, then retry; do not claim "
                        "that network tools are unavailable"
                    ),
                },
                ensure_ascii=False,
            )
        snapshot = budget_snapshot()
        ledger = snapshot.get("successful_sources", [])
        ledger_by_id = {str(item.get("source_id", "")): dict(item) for item in ledger}
        requested_ids = set(evidence_source_ids)
        unknown_ids = sorted(requested_ids - set(ledger_by_id))
        if unknown_ids:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "Evidence IDs are not present in the successful-source "
                        f"ledger: {', '.join(unknown_ids)}"
                    ),
                },
                ensure_ascii=False,
            )
        graph_required = int(plan.get("evidence_schema_version", 0)) >= 1
        graph_claims = [
            dict(item)
            for item in snapshot.get("claims", [])
            if item.get("subquestion_id") == active_subquestion_id
        ]
        eligible_claims = [
            item
            for item in graph_claims
            if item.get("status") in {"supported", "contested"}
        ]
        active_claims = eligible_claims if status == "covered" else graph_claims
        active_claim_ids = [str(item["claim_id"]) for item in active_claims]
        active_claim_id_set = set(active_claim_ids)
        active_conflicts = [
            dict(item)
            for item in snapshot.get("conflicts", [])
            if item.get("claim_id") in active_claim_id_set
        ]
        active_conflict_ids = [str(item["conflict_id"]) for item in active_conflicts]
        active_units = [
            dict(item)
            for item in snapshot.get("evidence_units", [])
            if item.get("claim_id") in active_claim_id_set
        ]
        graph_source_ids = {str(item.get("source_id", "")) for item in active_units}
        if graph_required and status == "covered" and not eligible_claims:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "Covered status requires at least one supported canonical "
                        "claim for the active subquestion"
                    ),
                },
                ensure_ascii=False,
            )
        if graph_required and status == "covered":
            scoped_usage = snapshot.get("subquestion_usage", {}).get(
                str(active_subquestion_id), {}
            )
            raw_scoped_relevant = scoped_usage.get("relevant_searches")
            if raw_scoped_relevant is None:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "Covered status requires an available per-subquestion "
                            "relevant-search metric; legacy non-empty searches are "
                            "not upgraded to relevant"
                        ),
                    },
                    ensure_ascii=False,
                )
            if int(raw_scoped_relevant) < 1:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "Covered status requires at least one relevant web_search "
                            "within the active subquestion; non-empty low-relevance "
                            "searches do not satisfy this gate"
                        ),
                    },
                    ensure_ascii=False,
                )
        if graph_required and requested_ids != graph_source_ids:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "Evidence source IDs must exactly match the active "
                        "claim-evidence graph: expected "
                        + ", ".join(sorted(graph_source_ids))
                    ),
                },
                ensure_ascii=False,
            )
        other_subquestions = [
            item
            for item in plan.get("subquestions", [])
            if item.get("id") != subquestion_id
        ]
        would_finish_plan = status == "covered" and all(
            item.get("status") == "covered" for item in other_subquestions
        )
        if graph_required and would_finish_plan:
            candidate_source_ids = {
                str(source_id)
                for item in other_subquestions
                for source_id in item.get("evidence_source_ids", [])
            }.union(requested_ids)
            candidate_claim_ids = {
                str(claim_id)
                for item in other_subquestions
                for claim_id in item.get("claim_ids", [])
            }.union(active_claim_id_set)
            corroborating_source_ids = corroborating_evidence_source_ids(
                source_ids=candidate_source_ids,
                sources=ledger,
                evidence_units=snapshot.get("evidence_units", []),
                claim_ids=candidate_claim_ids,
            )
            minimum_sources = int(snapshot.get("min_successful_sources", 0))
            if len(corroborating_source_ids) < minimum_sources:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "The final covered update requires at least "
                            f"{minimum_sources} corroborating source groups; "
                            f"currently {len(corroborating_source_ids)}. Continue "
                            "researching and register a claim from another "
                            "host with non-duplicate captured content"
                        ),
                    },
                    ensure_ascii=False,
                )
            required_searches = min(
                int(snapshot.get("max_searches", 0)),
                max(1, len(plan.get("subquestions", []))),
            )
            raw_relevant_searches = snapshot.get("relevant_searches")
            if raw_relevant_searches is None:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "The final covered update requires an available "
                            "relevant-search metric; legacy non-empty searches are "
                            "not upgraded to relevant"
                        ),
                    },
                    ensure_ascii=False,
                )
            relevant_searches = int(raw_relevant_searches)
            if relevant_searches < required_searches:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "The final covered update requires at least "
                            f"{required_searches} relevant web searches; currently "
                            f"{relevant_searches}. Run a relevant web_search, "
                            "then retry this update"
                        ),
                    },
                    ensure_ascii=False,
                )
        if graph_required and status == "blocked":
            candidate_source_ids = {
                str(source_id)
                for item in other_subquestions
                for source_id in item.get("evidence_source_ids", [])
            }.union(graph_source_ids)
            candidate_claim_ids = {
                str(claim_id)
                for item in other_subquestions
                for claim_id in item.get("claim_ids", [])
            }.union(str(item["claim_id"]) for item in eligible_claims)
            corroborating_count = len(
                corroborating_evidence_source_ids(
                    source_ids=candidate_source_ids,
                    sources=ledger,
                    evidence_units=snapshot.get("evidence_units", []),
                    claim_ids=candidate_claim_ids,
                )
            )
            minimum_sources = int(snapshot.get("min_successful_sources", 0))
            required_searches = min(
                int(snapshot.get("max_searches", 0)),
                max(1, len(plan.get("subquestions", []))),
            )
            relevant_searches = int(snapshot.get("relevant_searches") or 0)
            active_limits = snapshot.get("subquestion_limits", {}).get(
                active_subquestion_id, {}
            )
            active_usage = snapshot.get("subquestion_usage", {}).get(
                active_subquestion_id, {}
            )
            remaining_fetches = max(
                0,
                int(active_limits.get("max_fetches", snapshot.get("max_fetches", 0)))
                - int(active_usage.get("fetch_calls", snapshot.get("fetch_calls", 0))),
            )
            remaining_searches = max(
                0,
                int(active_limits.get("max_searches", snapshot.get("max_searches", 0)))
                - int(
                    active_usage.get("search_calls", snapshot.get("search_calls", 0))
                ),
            )
            if snapshot.get("strategy") == "adaptive":
                remaining_fetches += max(0, int(snapshot.get("reserve_fetches", 0)))
                remaining_searches += max(0, int(snapshot.get("reserve_searches", 0)))
            source_gap_recoverable = (
                corroborating_count < minimum_sources and remaining_fetches > 0
            )
            search_gap_recoverable = (
                relevant_searches < required_searches and remaining_searches > 0
            )
            claim_gap_recoverable = not eligible_claims and (
                remaining_searches > 0 or remaining_fetches > 0
            )
            if (
                source_gap_recoverable
                or search_gap_recoverable
                or claim_gap_recoverable
            ):
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "Blocked status is premature: the active SQ still has "
                            "recoverable policy gaps and reserved budget remains. "
                            "Continue web_search/fetch_url/record_evidence, or stop "
                            "without updating so the bounded outer retry can resume"
                        ),
                    },
                    ensure_ascii=False,
                )
        if not graph_required:
            previous_ids = set(runtime.state.get("active_source_ids_before", []))
            current_step_ids = set(ledger_by_id) - previous_ids
            if requested_ids and not requested_ids.intersection(current_step_ids):
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "Evidence IDs attached to an update must include at least "
                            "one source fetched during the active research step"
                        ),
                    },
                    ensure_ascii=False,
                )
            if status == "covered" and not requested_ids.intersection(current_step_ids):
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "Covered status requires at least one successful source "
                            "fetched during the active research step"
                        ),
                    },
                    ensure_ascii=False,
                )
        try:
            updated = transition_subquestion(
                plan,
                subquestion_id,
                status,
                evidence_source_ids=evidence_source_ids,
                claim_ids=active_claim_ids,
                conflict_ids=active_conflict_ids,
                note=note,
            )
            updated, closure_errors = audit_structural_subquestion_closures(
                updated, snapshot
            )
            if status == "covered" and subquestion_id in closure_errors:
                msg = "Covered state failed the canonical closure audit: " + "; ".join(
                    closure_errors[subquestion_id]
                )
                raise ValueError(msg)
        except ValueError as exc:
            return json.dumps(
                {"status": "error", "error": str(exc)}, ensure_ascii=False
            )
        events = cast("list[ResearchEvent]", runtime.state.get("research_events", []))
        next_events = append_research_event(
            events,
            "subquestion_updated",
            plan_id=updated["plan_id"],
            subquestion_id=subquestion_id,
            details={
                "status": status,
                "evidence_source_ids": evidence_source_ids,
                "claim_ids": active_claim_ids,
                "conflict_ids": active_conflict_ids,
            },
        )
        content = json.dumps(
            {
                "status": "success",
                "subquestion_id": subquestion_id,
                "new_status": status,
                "structural_subquestion_coverage": updated[
                    "structural_subquestion_coverage"
                ],
                "coverage": updated["coverage"],
                "canonical_claims": active_claims,
                "canonical_conflicts": active_conflicts,
                "canonical_sources": [
                    ledger_by_id[source_id]
                    for source_id in evidence_source_ids
                    if source_id in ledger_by_id
                ],
            },
            ensure_ascii=False,
        )
        return Command(
            update={
                "research_plan": updated,
                "research_events": next_events,
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    return [get_research_plan, update_subquestion]


def _compact_research_history(
    state: TongAgentState,
    *,
    max_chars: int = 480,
) -> str:
    """Summarize durable control history without replaying prior SQ messages."""

    events = list(state.get("research_events", []))
    recent_events = [
        {
            "event": item.get("event"),
            "subquestion_id": item.get("subquestion_id"),
        }
        for item in events[-4:]
    ]
    decisions = list(state.get("adaptive_control", {}).get("decision_history", []))
    last_decision = decisions[-1] if decisions else {}
    summary = json.dumps(
        {
            "event_count": len(events),
            "recent_events": recent_events,
            "last_control": {
                "action": last_decision.get("action"),
                "reason_codes": list(last_decision.get("reason_codes", []))[:4],
            },
            "compact_checkpoint_count": len(state.get("compact_checkpoints", [])),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return summary[:max_chars]


def build_active_research_context(
    *,
    plan: ResearchPlan,
    active_subquestion_id: str,
    budget: dict[str, Any],
    model_budget: dict[str, Any] | None = None,
) -> ActiveResearchContext:
    """Derive the only current-SQ state needed by the research model."""

    active = next(
        item
        for item in plan.get("subquestions", [])
        if item.get("id") == active_subquestion_id
    )
    limits = budget.get("subquestion_limits", {}).get(active_subquestion_id, {})
    usage = budget.get("subquestion_usage", {}).get(active_subquestion_id, {})
    claims, _, source_ids, _ = _active_canonical_state(
        budget,
        active_subquestion_id,
    )
    source_id_set = set(source_ids)
    sources = [
        {
            "source_id": item.get("source_id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "evidence_quality": item.get("evidence_quality"),
            "acquisition_method": item.get("acquisition_method"),
        }
        for item in budget.get("successful_sources", [])
        if str(item.get("source_id", "")) in source_id_set
    ]
    remaining_model_calls = None
    if model_budget is not None:
        raw_remaining = model_budget.get("remaining_model_calls")
        if isinstance(raw_remaining, int) and not isinstance(raw_remaining, bool):
            remaining_model_calls = max(0, raw_remaining)
    return ActiveResearchContext(
        plan_id=str(plan.get("plan_id", "")),
        active_subquestion={
            "id": active.get("id"),
            "question": active.get("question"),
            "rationale": active.get("rationale"),
            "depends_on": list(active.get("depends_on", [])),
            "attempts": active.get("attempts"),
        },
        subquestion_status=str(active.get("status", "")),
        query_task_type=classify_query_task_type(str(active.get("question", ""))),
        remaining_search_budget=max(
            0,
            int(limits.get("max_searches", 0)) - int(usage.get("search_calls", 0)),
        ),
        remaining_fetch_budget=max(
            0,
            int(limits.get("max_fetches", 0)) - int(usage.get("fetch_calls", 0)),
        ),
        remaining_model_calls=remaining_model_calls,
        relevant_sources=sources,
        canonical_claims=[
            {
                "claim_id": item.get("claim_id"),
                "text": item.get("text"),
                "status": item.get("status"),
                "source_ids": list(item.get("source_ids", [])),
            }
            for item in claims
        ],
        unresolved_requirements=_unresolved_active_requirements(
            budget,
            active_subquestion_id,
        ),
    )


def _query_task_guidance(task_type: str) -> str:
    if task_type == "list_or_enumeration":
        return (
            "The first query must target a complete list, table, discography, "
            "chronology, timeline, or overview. Do not start from one member; "
            "narrow only after the overview exposes a specific gap."
        )
    if task_type == "comparison":
        return (
            "Search each compared entity and attribute separately; do not put "
            "the final comparison or arithmetic into either query."
        )
    if task_type == "date_or_numeric_lookup":
        return (
            "Use one entity plus the exact date, depth, height, or numeric attribute."
        )
    return "Use one entity plus one factual attribute."


def build_research_graph(
    *,
    research_agent: Any,
    report_agent: Any | None = None,
    planner: Planner,
    budget_snapshot: BudgetSnapshot,
    budget_configure: BudgetConfigure | None = None,
    budget_activate: BudgetActivate | None = None,
    budget_grant: BudgetGrant | None = None,
    report_read: ReportRead | None = None,
    report_clear: ReportClear | None = None,
    checkpointer: Any | None,
    max_subquestions: int,
    max_research_cycles: int,
    interrupt_before: list[str] | None = None,
    require_researcher: bool = False,
    strategy: ResearchStrategy = "fixed",
    config_fingerprint: str = "",
    hard_effort: str = "",
    pinned_model: str = "",
    pinned_topology: str = "",
    max_escalations: int = 0,
    token_budget_snapshot: TokenBudgetSnapshot | None = None,
    token_budget_can_start: TokenBudgetCanStart | None = None,
    model_budget_snapshot: ModelBudgetSnapshot | None = None,
) -> Any:
    """Compile the outer plan-select-research-evaluate-report workflow.

    Args:
        research_agent: Uncheckpointed Deep Agent subgraph or compatible node.
        report_agent: Optional synthesis-only agent without research-state tools.
        planner: Injectable structured planner.
        budget_snapshot: Callable returning the latest serializable ledger.
        budget_configure: Optional callback that partitions the plan budget.
        budget_activate: Optional callback that selects the active SQ budget.
        budget_grant: Optional idempotent adaptive reserve-release callback.
        report_read: Optional callback that snapshots `/report.md` into state.
        report_clear: Optional callback that removes any pre-synthesis report.
        checkpointer: Checkpointer owned exclusively by the outer graph.
        max_subquestions: Hard plan breadth limit.
        max_research_cycles: Loop limit protecting against stalled model behavior.
        interrupt_before: Optional node interrupts used by recovery tests.
        require_researcher: Force one researcher task call per active SQ.
        strategy: Fixed baseline or deterministic adaptive control.
        config_fingerprint: Pinned policy identity used for safe resume.
        hard_effort: User-selected hard effort ceiling.
        pinned_model: Model identity that cannot change inside this plan.
        pinned_topology: Topology identity that cannot change inside this plan.
        max_escalations: Maximum reserve releases for the whole plan.
        model_budget_snapshot: Optional evaluation model-call budget telemetry.

    Returns:
        Compiled checkpointable research graph.
    """
    if report_agent is None:
        msg = "A separate synthesis-only report_agent is required"
        raise ValueError(msg)

    def plan_node(state: TongAgentState) -> dict[str, Any]:
        topic = " ".join(state.get("research_topic", "").split())
        existing = state.get("research_plan")
        resuming_plan = bool(
            existing and existing["status"] in {"pending", "in_progress"}
        )
        events = list(state.get("research_events", []))
        history = list(state.get("research_plan_history", []))
        if resuming_plan and existing:
            plan = refresh_plan_status(existing)
        else:
            if existing:
                history.append(deepcopy(existing))
            plan = refresh_plan_status(planner(topic, max_subquestions))
        if budget_configure is not None:
            budget_configure([item["id"] for item in plan["subquestions"]])
        if budget_activate is not None:
            budget_activate(None)
        budget = budget_snapshot()
        if resuming_plan:
            plan, closure_errors = audit_structural_subquestion_closures(plan, budget)
            events = append_research_event(
                events,
                "plan_resumed",
                plan_id=plan["plan_id"],
                details={
                    "structural_subquestion_coverage": plan[
                        "structural_subquestion_coverage"
                    ],
                    "coverage": plan["coverage"],
                    "closure_errors": closure_errors,
                },
            )
        else:
            events = append_research_event(
                events,
                "plan_created",
                plan_id=plan["plan_id"],
                details={
                    "planner": plan["planner"],
                    "subquestions": len(plan["subquestions"]),
                },
            )
        saved_control = state.get("adaptive_control") if resuming_plan else None
        if saved_control:
            control = deepcopy(saved_control)
            saved_fingerprint = str(control.get("config_fingerprint", ""))
            if (
                config_fingerprint
                and saved_fingerprint
                and saved_fingerprint != config_fingerprint
            ):
                msg = "Pending research plan was created with a different policy"
                raise ValueError(msg)
        else:
            control = initialize_control_state(
                strategy=strategy,
                config_fingerprint=config_fingerprint,
                hard_effort=hard_effort,
                pinned_model=pinned_model,
                pinned_topology=pinned_topology,
                max_escalations=max_escalations,
            )
        return {
            "research_plan": plan,
            "research_plan_history": history,
            "research_events": events,
            "active_subquestion_id": None,
            "active_source_ids_before": [],
            "active_claim_ids_before": [],
            "active_evidence_ids_before": [],
            "active_conflict_ids_before": [],
            "active_tool_attempt_sequence_before": int(
                budget.get("next_tool_attempt_sequence", 1)
            ),
            "budget_state": cast("BudgetState", budget),
            "adaptive_control": cast("AdaptiveControlState", control),
            "workflow_phase": "selecting",
            "research_cycles": 0,
            "max_research_cycles": max_research_cycles,
            "report_markdown": "",
            "compact_checkpoints": (
                list(state.get("compact_checkpoints", [])) if resuming_plan else []
            ),
            "active_token_slice_exhausted": None,
        }

    def select_node(state: TongAgentState) -> dict[str, Any]:
        previous_plan = state["research_plan"]
        preselection_budget = budget_snapshot()
        audited_plan, rejected_covered = audit_structural_subquestion_closures(
            previous_plan, preselection_budget
        )
        for subquestion_id, reasons in rejected_covered.items():
            audited_plan = transition_subquestion(
                audited_plan,
                subquestion_id,
                "blocked",
                note="Covered state rejected: " + "; ".join(reasons),
            )
        plan, active = select_next_subquestion(audited_plan)
        token_blocked: list[str] = []
        while (
            active
            and token_budget_can_start is not None
            and not token_budget_can_start(active)
        ):
            token_blocked.append(active)
            plan = transition_subquestion(
                plan,
                active,
                "blocked",
                note=(
                    "The required subquestion token partition has insufficient "
                    "capacity for another compact research turn."
                ),
            )
            plan, active = select_next_subquestion(plan)
        if budget_activate is not None:
            budget_activate(active)
        budget = budget_snapshot()
        events = list(state.get("research_events", []))
        previous_statuses = {
            item["id"]: item["status"] for item in previous_plan["subquestions"]
        }
        for item in plan["subquestions"]:
            if (
                item["status"] == "blocked"
                and previous_statuses.get(item["id"]) == "covered"
            ):
                events = append_research_event(
                    events,
                    "covered_state_rejected",
                    plan_id=plan["plan_id"],
                    subquestion_id=item["id"],
                    details={"note": item["note"]},
                )
            if (
                item["status"] == "blocked"
                and previous_statuses.get(item["id"]) == "pending"
            ):
                events = append_research_event(
                    events,
                    "subquestion_dependency_blocked",
                    plan_id=plan["plan_id"],
                    subquestion_id=item["id"],
                    details={"note": item["note"]},
                )
        events = append_research_event(
            events,
            "subquestion_selected" if active else "plan_research_complete",
            plan_id=plan["plan_id"],
            subquestion_id=active or "",
            details={
                "structural_subquestion_coverage": plan[
                    "structural_subquestion_coverage"
                ],
                "coverage": plan["coverage"],
                "budget_limits": budget.get("subquestion_limits", {}).get(active, {}),
            },
        )
        for subquestion_id in token_blocked:
            events = append_research_event(
                events,
                "subquestion_token_partition_exhausted",
                plan_id=plan["plan_id"],
                subquestion_id=subquestion_id,
                details={
                    "token_partition": (
                        token_budget_snapshot()
                        if token_budget_snapshot is not None
                        else {}
                    )
                },
            )
        sources = budget.get("successful_sources", [])
        source_ids = [str(item.get("source_id", "")) for item in sources]
        evidence_ids = [
            str(item.get("evidence_id", ""))
            for item in budget.get("evidence_units", [])
        ]
        claim_ids = [str(item.get("claim_id", "")) for item in budget.get("claims", [])]
        conflict_ids = [
            str(item.get("conflict_id", "")) for item in budget.get("conflicts", [])
        ]
        return {
            "research_plan": plan,
            "research_events": events,
            "active_subquestion_id": active,
            "active_source_ids_before": source_ids,
            "active_claim_ids_before": claim_ids,
            "active_evidence_ids_before": evidence_ids,
            "active_conflict_ids_before": conflict_ids,
            "active_tool_attempt_sequence_before": int(
                budget.get("next_tool_attempt_sequence", 1)
            ),
            "budget_state": cast("BudgetState", budget),
            "workflow_phase": "researching" if active else "reporting",
            "active_token_slice_exhausted": None,
        }

    def route_after_select(state: TongAgentState) -> Route:
        return "research" if state.get("active_subquestion_id") else "report"

    def prepare_research_node(state: TongAgentState) -> dict[str, Any]:
        active_id = state["active_subquestion_id"]
        active = next(
            item
            for item in state["research_plan"]["subquestions"]
            if item["id"] == active_id
        )
        budget = budget_snapshot()
        active_context = build_active_research_context(
            plan=state["research_plan"],
            active_subquestion_id=active_id,
            budget=budget,
            model_budget=(
                model_budget_snapshot() if model_budget_snapshot is not None else None
            ),
        )
        token_partition = (
            token_budget_snapshot() if token_budget_snapshot is not None else {}
        )
        event_summary = _compact_research_history(state)
        delegation_instruction = (
            "MULTI MODE: your very next tool call MUST be task with "
            "subagent_type='researcher' and its description MUST start exactly "
            f"with '[SQ:{active_id}]'. Give it this full active SQ and require "
            "web_search, fetch_url, record_evidence, and canonical C/E/S IDs in "
            "its return. Do not call update_subquestion or claim network tools are "
            "unavailable before that task returns.\n\n"
            if require_researcher
            else ""
        )
        content = f"""[RESEARCH STEP]
Root question: {state["research_plan"]["question"]}
ACTIVE RESEARCH CONTEXT:
{active_context.model_dump_json()}

Token partition for this SQ: {json.dumps(token_partition.get("buckets", {}).get(f"sq:{active_id}", {}), ensure_ascii=False)}
RESEARCH EVENT SUMMARY:
{event_summary}

Query strategy: {_query_task_guidance(active_context.query_task_type)}

{delegation_instruction}Research only this active subquestion. The compact context above is authoritative for normal execution; do not call get_research_plan, get_source_ledger, or get_evidence_graph unless it is internally inconsistent. Every web_search query must target one atomic fact, normally `entity name + attribute`, use roughly 8-12 English words or fewer, and must not copy the full subquestion or include the final multi-hop calculation. Search different entities in separate calls. After each successful fetch, call record_evidence for every factual proposition you may report, copying an exact 12-800 character excerpt from that fetched page. The claim argument must itself be a self-contained report-ready sentence with subject and predicate, never a label such as "official name" or "contact email"; quote is the separate exact page excerpt that supports it. OMIT claim_id when creating a new claim: the tool assigns C#. Pass claim_id only to add evidence to a C# already returned by a successful record_evidence call; never invent C#. On quote_mismatch, use at most one returned candidate quote correction; never resubmit an identical rejected call. A successful record_evidence deterministically updates coverage when the minimum gates are met. If it returns subquestion_auto_updated=true, stop this research turn immediately without reading state or calling update_subquestion. A source alone cannot cover an SQ. Search snippets never receive source IDs or evidence units. If no supported claim can be registered within the reserved budget, end the turn so the bounded controller can route it. Do not write the final report during this step."""
        message_id = (
            f"research-step-{state['research_plan']['plan_id']}-{active['id']}-"
            f"{active['attempts']}"
        )
        return {
            "messages": [
                HumanMessage(content=content, id=message_id),
            ],
            "workflow_phase": "researching",
        }

    def evaluate_node(state: TongAgentState) -> dict[str, Any]:
        plan = deepcopy(state["research_plan"])
        active_id = state.get("active_subquestion_id")
        budget = budget_snapshot()
        current_ids = [
            str(item.get("source_id", ""))
            for item in budget.get("successful_sources", [])
        ]
        previous_ids = set(state.get("active_source_ids_before", []))
        new_ids = [
            source_id for source_id in current_ids if source_id not in previous_ids
        ]
        previous_evidence_ids = set(state.get("active_evidence_ids_before", []))
        new_evidence_ids = [
            str(item.get("evidence_id", ""))
            for item in budget.get("evidence_units", [])
            if str(item.get("evidence_id", "")) not in previous_evidence_ids
        ]
        previous_claim_ids = set(state.get("active_claim_ids_before", []))
        new_claim_ids = [
            str(item.get("claim_id", ""))
            for item in budget.get("claims", [])
            if str(item.get("claim_id", "")) not in previous_claim_ids
        ]
        previous_conflict_ids = set(state.get("active_conflict_ids_before", []))
        new_conflict_ids = [
            str(item.get("conflict_id", ""))
            for item in budget.get("conflicts", [])
            if str(item.get("conflict_id", "")) not in previous_conflict_ids
        ]
        first_attempt_sequence = int(
            state.get("active_tool_attempt_sequence_before", 1)
        )
        new_tool_attempt_ids = [
            str(item.get("attempt_id", ""))
            for item in budget.get("tool_attempts", [])
            if int(item.get("sequence", 0)) >= first_attempt_sequence
        ]
        token_slice_exhausted = state.get("active_token_slice_exhausted")
        if active_id:
            active = next(
                item for item in plan["subquestions"] if item["id"] == active_id
            )
            if active["status"] == "researching":
                active_claims = [
                    dict(item)
                    for item in budget.get("claims", [])
                    if item.get("subquestion_id") == active_id
                    and item.get("status") in {"supported", "contested"}
                ]
                active_claim_ids = {
                    str(item.get("claim_id", "")) for item in active_claims
                }
                active_evidence = [
                    dict(item)
                    for item in budget.get("evidence_units", [])
                    if str(item.get("claim_id", "")) in active_claim_ids
                ]
                active_source_ids = list(
                    dict.fromkeys(
                        str(item.get("source_id", ""))
                        for item in active_evidence
                        if item.get("source_id")
                    )
                )
                active_conflict_ids = [
                    str(item.get("conflict_id", ""))
                    for item in budget.get("conflicts", [])
                    if str(item.get("claim_id", "")) in active_claim_ids
                ]
                active_relevant = int(
                    budget.get("subquestion_usage", {})
                    .get(active_id, {})
                    .get("relevant_searches")
                    or 0
                )
                if active_claim_ids and active_source_ids and active_relevant >= 1:
                    candidate = transition_subquestion(
                        plan,
                        active_id,
                        "covered",
                        evidence_source_ids=active_source_ids,
                        claim_ids=sorted(active_claim_ids),
                        conflict_ids=active_conflict_ids,
                        note=(
                            "Minimum canonical evidence was recorded and the "
                            "outer graph deterministically closed this subquestion."
                        ),
                    )
                    candidate, candidate_errors = audit_structural_subquestion_closures(
                        candidate,
                        budget,
                    )
                    if active_id not in candidate_errors:
                        plan = candidate
                    else:
                        plan = transition_subquestion(
                            plan,
                            active_id,
                            "pending",
                            note=(
                                "Canonical evidence exists but closure still needs: "
                                + "; ".join(candidate_errors[active_id])
                            ),
                        )
                elif (
                    token_slice_exhausted
                    and active_claim_ids
                    and active_source_ids
                    and active_relevant >= 1
                ):
                    plan = transition_subquestion(
                        plan,
                        active_id,
                        "covered",
                        evidence_source_ids=active_source_ids,
                        claim_ids=sorted(active_claim_ids),
                        conflict_ids=active_conflict_ids,
                        note=(
                            "The local token slice ended after the minimum "
                            "canonical evidence for this subquestion was recorded."
                        ),
                    )
                elif token_slice_exhausted:
                    plan = transition_subquestion(
                        plan,
                        active_id,
                        "blocked",
                        note=(
                            "The local token slice ended before minimum canonical "
                            "evidence could be recorded."
                        ),
                    )
                elif (
                    strategy == "fixed" and active["attempts"] >= active["max_attempts"]
                ):
                    plan = transition_subquestion(
                        plan,
                        active_id,
                        "blocked",
                        note="No new successful evidence after the maximum attempts.",
                    )
                else:
                    plan = transition_subquestion(
                        plan,
                        active_id,
                        "pending",
                        note=(
                            "Adaptive controller will decide whether to retry or "
                            "release reserve budget."
                            if strategy == "adaptive"
                            else "No new successful evidence; retry is allowed."
                        ),
                    )
        plan, rejected_covered = audit_structural_subquestion_closures(plan, budget)
        for subquestion_id, reasons in rejected_covered.items():
            if strategy == "adaptive":
                reopened = deepcopy(plan)
                target = next(
                    item
                    for item in reopened["subquestions"]
                    if item["id"] == subquestion_id
                )
                target["status"] = "pending"
                target["evidence_source_ids"] = []
                target["claim_ids"] = []
                target["conflict_ids"] = []
                target["structural_closure_validated"] = False
                target["note"] = "Covered state rejected: " + "; ".join(reasons)
                plan = refresh_plan_status(reopened)
            else:
                plan = transition_subquestion(
                    plan,
                    subquestion_id,
                    "blocked",
                    note="Covered state rejected: " + "; ".join(reasons),
                )
        plan = refresh_plan_status(plan)
        cycles = int(state.get("research_cycles", 0)) + 1
        control = deepcopy(
            state.get("adaptive_control")
            or initialize_control_state(
                strategy=strategy,
                config_fingerprint=config_fingerprint,
                hard_effort=hard_effort,
                pinned_model=pinned_model,
                pinned_topology=pinned_topology,
                max_escalations=max_escalations,
            )
        )
        assessment = build_control_assessment(
            plan=plan,
            budget=budget,
            control=control,
            subquestion_id=active_id or "",
            cycle=cycles,
            new_source_ids=new_ids,
            new_claim_ids=new_claim_ids,
            new_evidence_ids=new_evidence_ids,
            new_conflict_ids=new_conflict_ids,
            new_tool_attempt_ids=new_tool_attempt_ids,
        )
        control["last_assessment"] = assessment
        events = append_research_event(
            list(state.get("research_events", [])),
            "coverage_evaluated",
            plan_id=plan["plan_id"],
            subquestion_id=active_id or "",
            details={
                "structural_subquestion_coverage": plan[
                    "structural_subquestion_coverage"
                ],
                "coverage": plan["coverage"],
                "plan_status": plan["status"],
                "new_source_ids": new_ids,
                "new_claim_ids": new_claim_ids,
                "new_evidence_ids": new_evidence_ids,
                "new_conflict_ids": new_conflict_ids,
                "new_tool_attempt_ids": new_tool_attempt_ids,
                "cycle": cycles,
            },
        )
        if token_slice_exhausted:
            events = append_research_event(
                events,
                "subquestion_token_slice_stopped",
                plan_id=plan["plan_id"],
                subquestion_id=active_id or "",
                details={
                    "denial": token_slice_exhausted,
                    "token_partition": (
                        token_budget_snapshot()
                        if token_budget_snapshot is not None
                        else {}
                    ),
                },
            )
        compact_checkpoints = list(state.get("compact_checkpoints", []))
        active_after = next(
            (item for item in plan["subquestions"] if item.get("id") == active_id),
            {},
        )
        compact_checkpoints.append(
            {
                "cycle": cycles,
                "subquestion_id": active_id,
                "status": active_after.get("status"),
                "source_ids": list(active_after.get("evidence_source_ids", [])),
                "claim_ids": list(active_after.get("claim_ids", [])),
                "conflict_ids": list(active_after.get("conflict_ids", [])),
                "new_source_ids": new_ids,
                "new_claim_ids": new_claim_ids,
                "new_evidence_ids": new_evidence_ids,
                "token_partition": (
                    token_budget_snapshot() if token_budget_snapshot is not None else {}
                ),
            }
        )
        events = append_research_event(
            events,
            "subquestion_compact_checkpoint",
            plan_id=plan["plan_id"],
            subquestion_id=active_id or "",
            details={
                "cycle": cycles,
                "status": active_after.get("status"),
                "source_ids": list(active_after.get("evidence_source_ids", [])),
                "claim_ids": list(active_after.get("claim_ids", [])),
            },
        )
        return {
            "research_plan": plan,
            "research_events": events,
            "budget_state": cast("BudgetState", budget),
            "adaptive_control": cast("AdaptiveControlState", control),
            "active_subquestion_id": None,
            "research_cycles": cycles,
            "workflow_phase": "evaluating",
            "compact_checkpoints": compact_checkpoints,
            "active_token_slice_exhausted": None,
        }

    def route_after_evaluate(state: TongAgentState) -> str:
        if strategy == "adaptive":
            return "control"
        plan = state["research_plan"]
        unfinished = any(
            item["status"] in {"pending", "researching"}
            for item in plan["subquestions"]
        )
        within_limit = int(state.get("research_cycles", 0)) < int(
            state.get("max_research_cycles", max_research_cycles)
        )
        return "research" if unfinished and within_limit else "report"

    def adaptive_control_node(state: TongAgentState) -> dict[str, Any]:
        plan = deepcopy(state["research_plan"])
        control = deepcopy(state["adaptive_control"])
        assessment = deepcopy(control.get("last_assessment", {}))
        cycle = int(state.get("research_cycles", 0))
        cycle_limit = int(state.get("max_research_cycles", max_research_cycles))
        action, reasons = decide_control_action(
            plan=plan,
            control=control,
            assessment=assessment,
            cycle_limit_reached=cycle >= cycle_limit,
        )
        sequence = len(control.get("decision_history", [])) + 1
        subquestion_id = str(assessment.get("subquestion_id", ""))
        decision_id = decision_id_for(plan["plan_id"], cycle, sequence, subquestion_id)
        budget_before_snapshot = budget_snapshot()
        budget_before = {
            "search_calls": int(budget_before_snapshot.get("search_calls", 0)),
            "fetch_calls": int(budget_before_snapshot.get("fetch_calls", 0)),
            "granted_searches": int(budget_before_snapshot.get("granted_searches", 0)),
            "granted_fetches": int(budget_before_snapshot.get("granted_fetches", 0)),
            "reserve_searches": int(budget_before_snapshot.get("reserve_searches", 0)),
            "reserve_fetches": int(budget_before_snapshot.get("reserve_fetches", 0)),
        }
        grant_result: dict[str, Any] = {}
        if action == "expand_budget":
            if budget_grant is None or not subquestion_id:
                action = "stop_subquestion"
                reasons = [*reasons, "budget_grant_unavailable"]
            else:
                search_retry_needed = "search_retry_needed" in reasons
                fetch_retry_needed = "fetch_retry_needed" in reasons
                needs_search = (
                    search_retry_needed
                    or ("relevant_search_gap" in reasons)
                    or (
                        "provider_failure" in reasons
                        and not search_retry_needed
                        and not fetch_retry_needed
                    )
                )
                needs_evidence = any(
                    item in reasons
                    for item in ("claim_gap", "corroborating_source_gap")
                )
                grant_result = budget_grant(
                    decision_id,
                    subquestion_id,
                    search_delta=(
                        1
                        if needs_search or (needs_evidence and not fetch_retry_needed)
                        else 0
                    ),
                    fetch_delta=1 if needs_evidence or fetch_retry_needed else 0,
                )
                if not grant_result.get("applied") and not grant_result.get(
                    "idempotent_replay"
                ):
                    action = "stop_subquestion"
                    reasons = [*reasons, "hard_ceiling_reached"]
                else:
                    target = next(
                        item
                        for item in plan["subquestions"]
                        if item["id"] == subquestion_id
                    )
                    target["max_attempts"] = max(
                        int(target["max_attempts"]), int(target["attempts"]) + 1
                    )
                    plan = refresh_plan_status(plan)
        if action == "stop_subquestion" and subquestion_id:
            target = next(
                (item for item in plan["subquestions"] if item["id"] == subquestion_id),
                None,
            )
            if target is not None and target["status"] in {"pending", "researching"}:
                plan = transition_subquestion(
                    plan,
                    subquestion_id,
                    "blocked",
                    note="Adaptive controller stopped research: "
                    + ", ".join(dict.fromkeys(reasons)),
                )
        if action == "fail_closed":
            plan = deepcopy(plan)
            for target in plan["subquestions"]:
                target["status"] = "blocked"
                target["evidence_source_ids"] = []
                target["claim_ids"] = []
                target["conflict_ids"] = []
                target["structural_closure_validated"] = False
                target["note"] = (
                    "Evidence integrity validation failed; research stopped."
                )
            plan = refresh_plan_status(plan)
        budget_after_snapshot = budget_snapshot()
        budget_after = {
            "search_calls": int(budget_after_snapshot.get("search_calls", 0)),
            "fetch_calls": int(budget_after_snapshot.get("fetch_calls", 0)),
            "granted_searches": int(budget_after_snapshot.get("granted_searches", 0)),
            "granted_fetches": int(budget_after_snapshot.get("granted_fetches", 0)),
            "reserve_searches": int(budget_after_snapshot.get("reserve_searches", 0)),
            "reserve_fetches": int(budget_after_snapshot.get("reserve_fetches", 0)),
        }
        if action == "expand_budget" and grant_result:
            original_before = grant_result.get("budget_before")
            original_after = grant_result.get("budget_after")
            if isinstance(original_before, dict) and isinstance(original_after, dict):
                for field_name in (
                    "granted_searches",
                    "granted_fetches",
                    "reserve_searches",
                    "reserve_fetches",
                ):
                    budget_before[field_name] = int(original_before[field_name])
                    budget_after[field_name] = int(original_after[field_name])
        control = record_control_decision(
            control,
            decision_id=decision_id,
            action=action,
            assessment=assessment,
            reason_codes=list(dict.fromkeys(reasons)),
            budget_before=budget_before,
            budget_after=budget_after,
        )
        events = append_research_event(
            list(state.get("research_events", [])),
            "control_assessed",
            plan_id=plan["plan_id"],
            subquestion_id=subquestion_id,
            details={
                "decision_id": decision_id,
                "action": action,
                "reason_codes": list(dict.fromkeys(reasons)),
                "assessment": assessment,
            },
        )
        action_event = {
            "expand_budget": "budget_expanded",
            "continue": "adaptive_continued",
            "stop_subquestion": "adaptive_stopped",
            "finish_success": "adaptive_stopped",
            "finish_partial": "adaptive_stopped",
            "fail_closed": "control_integrity_failed",
        }[action]
        events = append_research_event(
            events,
            action_event,
            plan_id=plan["plan_id"],
            subquestion_id=subquestion_id,
            details={
                "decision_id": decision_id,
                "action": action,
                "grant": grant_result,
                "budget_before": budget_before,
                "budget_after": budget_after,
            },
        )
        return {
            "research_plan": refresh_plan_status(plan),
            "research_events": events,
            "budget_state": cast("BudgetState", budget_after_snapshot),
            "adaptive_control": cast("AdaptiveControlState", control),
            "workflow_phase": "evaluating",
        }

    def route_after_control(state: TongAgentState) -> ControlRoute:
        control = state["adaptive_control"]
        history = control.get("decision_history", [])
        action = history[-1].get("action") if history else "finish_partial"
        if action in {"finish_success", "finish_partial", "fail_closed"}:
            return "report"
        if int(state.get("research_cycles", 0)) >= int(
            state.get("max_research_cycles", max_research_cycles)
        ):
            return "report"
        return "select"

    def prepare_report_node(state: TongAgentState) -> dict[str, Any]:
        if report_clear is not None:
            report_clear()
        if budget_activate is not None:
            budget_activate(None)
        budget = budget_snapshot()
        plan, _ = audit_structural_subquestion_closures(state["research_plan"], budget)
        if int(state.get("research_cycles", 0)) >= int(
            state.get("max_research_cycles", max_research_cycles)
        ):
            for item in list(plan["subquestions"]):
                if item["status"] in {"pending", "researching"}:
                    plan = transition_subquestion(
                        plan,
                        item["id"],
                        "blocked",
                        note="The explicit research cycle limit was reached.",
                    )
        plan = refresh_plan_status(plan)
        integrity_errors = validate_evidence_graph(budget)
        control_history = state.get("adaptive_control", {}).get("decision_history", [])
        integrity_fail_closed = bool(integrity_errors) or bool(
            control_history and control_history[-1].get("action") == "fail_closed"
        )
        if integrity_fail_closed:
            plan = deepcopy(plan)
            for target in plan["subquestions"]:
                target["status"] = "blocked"
                target["evidence_source_ids"] = []
                target["claim_ids"] = []
                target["conflict_ids"] = []
                target["structural_closure_validated"] = False
                target["note"] = (
                    "Evidence integrity validation failed; research stopped."
                )
            plan = refresh_plan_status(plan)
        events = append_research_event(
            list(state.get("research_events", [])),
            "report_requested",
            plan_id=plan["plan_id"],
            details={
                "structural_subquestion_coverage": plan[
                    "structural_subquestion_coverage"
                ],
                "coverage": plan["coverage"],
                "status": plan["status"],
                "integrity_fail_closed": integrity_fail_closed,
                "integrity_errors": integrity_errors,
            },
        )
        report_plan = {
            "plan_id": plan.get("plan_id"),
            "question": plan.get("question"),
            "objective": plan.get("objective"),
            "status": plan.get("status"),
            "structural_subquestion_coverage": plan.get(
                "structural_subquestion_coverage"
            ),
            "subquestions": [
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "status": item.get("status"),
                    "evidence_source_ids": item.get("evidence_source_ids", []),
                    "claim_ids": item.get("claim_ids", []),
                    "conflict_ids": item.get("conflict_ids", []),
                    "note": item.get("note", ""),
                }
                for item in plan.get("subquestions", [])
            ],
        }
        plan_json = json.dumps(report_plan, ensure_ascii=False, separators=(",", ":"))
        report_sources = (
            []
            if integrity_fail_closed
            else [
                {
                    "source_id": item.get("source_id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "evidence_quality": item.get("evidence_quality"),
                    "quality_reason": item.get("quality_reason", ""),
                    "acquisition_method": item.get("acquisition_method"),
                }
                for item in budget.get("successful_sources", [])
                if isinstance(item, dict)
            ]
        )
        ledger_json = json.dumps(report_sources, ensure_ascii=False, indent=2)
        report_graph = (
            {"claims": [], "evidence_units": [], "conflicts": []}
            if integrity_fail_closed
            else {
                "claims": [
                    {
                        "claim_id": item.get("claim_id"),
                        "subquestion_id": item.get("subquestion_id"),
                        "text": item.get("text"),
                        "status": item.get("status"),
                        "source_ids": item.get("source_ids", []),
                    }
                    for item in budget.get("claims", [])
                    if isinstance(item, dict)
                ],
                "evidence_units": [
                    {
                        "evidence_id": item.get("evidence_id"),
                        "claim_id": item.get("claim_id"),
                        "subquestion_id": item.get("subquestion_id"),
                        "source_id": item.get("source_id"),
                        "stance": item.get("stance"),
                        "quote": item.get("quote"),
                    }
                    for item in budget.get("evidence_units", [])
                    if isinstance(item, dict)
                ],
                "conflicts": [
                    {
                        "conflict_id": item.get("conflict_id"),
                        "claim_id": item.get("claim_id"),
                        "status": item.get("status"),
                        "source_ids": item.get("source_ids", []),
                    }
                    for item in budget.get("conflicts", [])
                    if isinstance(item, dict)
                ],
            }
        )
        evidence_graph_json = json.dumps(
            report_graph,
            ensure_ascii=False,
            indent=2,
        )
        allowed_caveats = sorted(
            allowed_report_caveat_lines(plan, integrity_failure=integrity_fail_closed)
        )
        allowed_caveats_json = json.dumps(
            allowed_caveats,
            ensure_ascii=False,
            indent=2,
        )
        fail_closed_instruction = (
            "Evidence integrity validation failed. Do not report or cite any "
            "canonical fact; leave the factual sections empty and copy only the "
            "matching allowed caveat under Conflicts and Caveats.\n\n"
            if integrity_fail_closed
            else ""
        )
        content = f"""[FINAL SYNTHESIS]
{fail_closed_instruction}You MUST call write_file to create the final `/report.md` for the root question using only the canonical evidence graph below. Use exactly these four H2 section headings in this order with Sources last: `## Short Answer`, `## Key Findings`, `## Conflicts and Caveats`, and `## Sources`; do not add other headings. Every non-empty finding line must contain exactly one canonical `claim.text` copied byte-for-byte from the graph plus its `[C#]` and linked `[S#]`; copy `claim.text`, NOT the evidence quote. The only valid shape is `- <exact claim.text> [C#][S#]`: add no prefix, suffix, emphasis, or local paraphrase. Every canonical claim_id attached to a covered SQ in PLAN MUST appear at least once, even when two claim texts look redundant. Put contested claims only in Conflicts and Caveats and cite both supporting and contradicting sources. Citation-free text in Conflicts and Caveats is forbidden unless the complete stripped line is copied byte-for-byte from ALLOWED CAVEAT LINES below; if that list is empty, leave the section empty unless it contains a canonical contested claim. Never attach a supported claim ID to caveat text. Never present contradicted-only claims as facts. Every Sources line MUST have exactly this shape: `- [S#] <exact canonical title> — <canonical URL>`. If there are no reportable claims, leave Short Answer and Key Findings empty and use only an applicable allowed caveat. Search snippets and failed fetches are not evidence. If the plan is partial, use only the exact applicable partial-coverage caveat without claiming blocked work was completed.

PLAN:
{plan_json}

CANONICAL SOURCE LEDGER:
{ledger_json}

CANONICAL EVIDENCE GRAPH:
{evidence_graph_json}

ALLOWED CAVEAT LINES:
{allowed_caveats_json}"""
        return {
            "messages": [
                HumanMessage(
                    content=content,
                    id=f"report-step-{plan['plan_id']}-{state.get('research_cycles', 0)}",
                )
            ],
            "research_plan": plan,
            "research_events": events,
            "budget_state": cast("BudgetState", budget),
            "workflow_phase": "reporting",
            "report_markdown": "",
        }

    def finish_node(state: TongAgentState) -> dict[str, Any]:
        budget = budget_snapshot()
        plan, _ = audit_structural_subquestion_closures(state["research_plan"], budget)
        events = append_research_event(
            list(state.get("research_events", [])),
            "run_finished",
            plan_id=plan["plan_id"],
            details={
                "structural_subquestion_coverage": plan[
                    "structural_subquestion_coverage"
                ],
                "coverage": plan["coverage"],
                "status": plan["status"],
            },
        )
        return {
            "research_plan": plan,
            "research_events": events,
            "budget_state": cast("BudgetState", budget),
            "workflow_phase": "finished",
        }

    def invoke_inner_agent(
        state: TongAgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run the uncheckpointed inner graph and atomically expose its ledger."""
        try:
            result = research_agent.invoke(state, config=config)
        except Exception as exc:
            resource = getattr(exc, "resource", None)
            resource_value = getattr(resource, "value", resource)
            if str(resource_value) != "subquestion_slice":
                raise
            events = append_research_event(
                list(state.get("research_events", [])),
                "subquestion_token_slice_exhausted",
                plan_id=state["research_plan"]["plan_id"],
                subquestion_id=str(state.get("active_subquestion_id") or ""),
                details={
                    "attempted": getattr(exc, "attempted", {}),
                    "token_partition": (
                        token_budget_snapshot()
                        if token_budget_snapshot is not None
                        else {}
                    ),
                },
            )
            return {
                "research_events": events,
                "budget_state": cast("BudgetState", budget_snapshot()),
                "report_markdown": "",
                "active_token_slice_exhausted": getattr(exc, "attempted", {}),
            }
        return {
            **result,
            "budget_state": cast("BudgetState", budget_snapshot()),
            "report_markdown": "",
            "active_token_slice_exhausted": None,
        }

    def invoke_report_agent(
        state: TongAgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run synthesis without exposing untrusted research-turn messages."""
        report_state = dict(state)
        report_state["messages"] = [state["messages"][-1]]
        result = report_agent.invoke(report_state, config=config)
        report_markdown = report_read() if report_read is not None else ""
        return {
            **result,
            "budget_state": cast("BudgetState", budget_snapshot()),
            "report_markdown": report_markdown,
        }

    builder = StateGraph(TongAgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("select", select_node)
    builder.add_node("prepare_research", prepare_research_node)
    builder.add_node("research_agent", invoke_inner_agent)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("adaptive_control", adaptive_control_node)
    builder.add_node("prepare_report", prepare_report_node)
    builder.add_node("report_agent", invoke_report_agent)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "select")
    builder.add_conditional_edges(
        "select",
        route_after_select,
        {"research": "prepare_research", "report": "prepare_report"},
    )
    builder.add_edge("prepare_research", "research_agent")
    builder.add_edge("research_agent", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "research": "select",
            "report": "prepare_report",
            "control": "adaptive_control",
        },
    )
    builder.add_conditional_edges(
        "adaptive_control",
        route_after_control,
        {"select": "select", "report": "prepare_report"},
    )
    builder.add_edge("prepare_report", "report_agent")
    builder.add_edge("report_agent", "finish")
    builder.add_edge("finish", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        name="tongagent-research-workflow",
    )
