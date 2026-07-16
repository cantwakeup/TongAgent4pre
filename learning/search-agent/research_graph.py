"""Explicit, checkpointable research planning workflow for TongAgent."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
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

from evidence_graph import independent_evidence_source_ids
from research_state import (
    BudgetState,
    EvidenceStance,
    ResearchEvent,
    ResearchPlan,
    SubQuestion,
    SubquestionStatus,
    TongAgentState,
)
from telemetry import append_research_event


Planner = Callable[[str, int], ResearchPlan]
BudgetSnapshot = Callable[[], dict[str, Any]]
BudgetConfigure = Callable[[list[str]], None]
BudgetActivate = Callable[[str | None], None]
EvidenceRecord = Callable[..., dict[str, Any]]
PlanIdFactory = Callable[[], str]
Route = Literal["research", "report"]
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


def calculate_plan_coverage(plan: ResearchPlan) -> float:
    """Calculate deterministic covered-subquestion coverage."""
    subquestions = plan.get("subquestions", [])
    if not subquestions:
        return 0.0
    covered = sum(item["status"] == "covered" for item in subquestions)
    return round(covered / len(subquestions), 4)


def refresh_plan_status(plan: ResearchPlan) -> ResearchPlan:
    """Return a copied plan with code-derived coverage and aggregate status."""
    updated = deepcopy(plan)
    updated.setdefault("evidence_schema_version", 0)
    subquestions = updated.get("subquestions", [])
    for item in subquestions:
        item.setdefault("claim_ids", [])
        item.setdefault("conflict_ids", [])
    updated["coverage"] = calculate_plan_coverage(updated)
    statuses = {item["status"] for item in subquestions}
    if subquestions and statuses == {"covered"}:
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
    invalid: dict[str, list[str]] = {}
    for item in plan.get("subquestions", []):
        if item.get("status") != "covered":
            continue
        subquestion_id = str(item.get("id", ""))
        claim_ids = list(dict.fromkeys(item.get("claim_ids", [])))
        claim_id_set = set(claim_ids)
        reasons: list[str] = []
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
        independent_count = len(
            independent_evidence_source_ids(
                source_ids=plan_source_ids,
                sources=snapshot.get("successful_sources", []),
                evidence_units=evidence_units,
                claim_ids=plan_claim_ids,
            )
        )
        minimum_sources = int(snapshot.get("min_successful_sources", 0))
        if independent_count < minimum_sources:
            last_id = str(subquestions[-1].get("id", ""))
            invalid.setdefault(last_id, []).append(
                f"only {independent_count} of {minimum_sources} required independent sources"
            )
        required_searches = min(
            int(snapshot.get("max_searches", 0)), max(1, len(subquestions))
        )
        successful_searches = int(
            snapshot.get("successful_searches", snapshot.get("search_calls", 0))
        )
        if successful_searches < required_searches:
            last_id = str(subquestions[-1].get("id", ""))
            invalid.setdefault(last_id, []).append(
                f"only {successful_searches} of {required_searches} required successful searches"
            )
    return invalid


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


def build_evidence_graph_tools(
    evidence_record: EvidenceRecord, budget_snapshot: BudgetSnapshot
) -> list[BaseTool]:
    """Build tools for exact-excerpt registration and graph inspection."""

    @tool("record_evidence")
    def record_evidence(
        source_id: str,
        claim: str,
        quote: str,
        stance: EvidenceStance = "supports",
        claim_id: str = "",
    ) -> str:
        """Link an exact source excerpt to a code-assigned canonical claim.

        The claim must be a self-contained report-ready proposition with a
        subject and predicate, never a topic label or field name. Omit claim_id
        to create it. Pass claim_id only when reusing an existing C# returned by
        a successful earlier call; never invent a C#.
        """
        try:
            result = evidence_record(
                source_id=source_id,
                claim=claim,
                quote=quote,
                stance=stance,
                claim_id=claim_id,
            )
        except ValueError as exc:
            result = {"status": "error", "error": str(exc)}
        else:
            result = {"status": "success", **result}
        return json.dumps(result, ensure_ascii=False, indent=2)

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
            independent_source_ids = independent_evidence_source_ids(
                source_ids=candidate_source_ids,
                sources=ledger,
                evidence_units=snapshot.get("evidence_units", []),
                claim_ids=candidate_claim_ids,
            )
            minimum_sources = int(snapshot.get("min_successful_sources", 0))
            if len(independent_source_ids) < minimum_sources:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "The final covered update requires at least "
                            f"{minimum_sources} independent canonical sources; "
                            f"currently {len(independent_source_ids)}. Continue "
                            "researching and register a claim from another "
                            "independent fetched page"
                        ),
                    },
                    ensure_ascii=False,
                )
            required_searches = min(
                int(snapshot.get("max_searches", 0)),
                max(1, len(plan.get("subquestions", []))),
            )
            successful_searches = int(
                snapshot.get("successful_searches", snapshot.get("search_calls", 0))
            )
            if successful_searches < required_searches:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            "The final covered update requires at least "
                            f"{required_searches} successful web searches; currently "
                            f"{successful_searches}. Run a relevant web_search, "
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
            independent_count = len(
                independent_evidence_source_ids(
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
            successful_searches = int(
                snapshot.get("successful_searches", snapshot.get("search_calls", 0))
            )
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
            source_gap_recoverable = (
                independent_count < minimum_sources and remaining_fetches > 0
            )
            search_gap_recoverable = (
                successful_searches < required_searches and remaining_searches > 0
            )
            if source_gap_recoverable or search_gap_recoverable:
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


def build_research_graph(
    *,
    research_agent: Any,
    planner: Planner,
    budget_snapshot: BudgetSnapshot,
    budget_configure: BudgetConfigure | None = None,
    budget_activate: BudgetActivate | None = None,
    checkpointer: Any | None,
    max_subquestions: int,
    max_research_cycles: int,
    interrupt_before: list[str] | None = None,
    require_researcher: bool = False,
) -> Any:
    """Compile the outer plan-select-research-evaluate-report workflow.

    Args:
        research_agent: Uncheckpointed Deep Agent subgraph or compatible node.
        planner: Injectable structured planner.
        budget_snapshot: Callable returning the latest serializable ledger.
        budget_configure: Optional callback that partitions the plan budget.
        budget_activate: Optional callback that selects the active SQ budget.
        checkpointer: Checkpointer owned exclusively by the outer graph.
        max_subquestions: Hard plan breadth limit.
        max_research_cycles: Loop limit protecting against stalled model behavior.
        interrupt_before: Optional node interrupts used by recovery tests.
        require_researcher: Force one researcher task call per active SQ.

    Returns:
        Compiled checkpointable research graph.
    """

    def plan_node(state: TongAgentState) -> dict[str, Any]:
        topic = " ".join(state.get("research_topic", "").split())
        existing = state.get("research_plan")
        events = list(state.get("research_events", []))
        history = list(state.get("research_plan_history", []))
        if existing and existing["status"] in {"pending", "in_progress"}:
            plan = refresh_plan_status(existing)
            events = append_research_event(
                events,
                "plan_resumed",
                plan_id=plan["plan_id"],
                details={"coverage": plan["coverage"]},
            )
        else:
            if existing:
                history.append(deepcopy(existing))
            plan = refresh_plan_status(planner(topic, max_subquestions))
            events = append_research_event(
                events,
                "plan_created",
                plan_id=plan["plan_id"],
                details={
                    "planner": plan["planner"],
                    "subquestions": len(plan["subquestions"]),
                },
            )
        if budget_configure is not None:
            budget_configure([item["id"] for item in plan["subquestions"]])
        if budget_activate is not None:
            budget_activate(None)
        budget = budget_snapshot()
        return {
            "research_plan": plan,
            "research_plan_history": history,
            "research_events": events,
            "active_subquestion_id": None,
            "active_source_ids_before": [],
            "active_evidence_ids_before": [],
            "budget_state": cast("BudgetState", budget),
            "workflow_phase": "selecting",
            "research_cycles": 0,
            "max_research_cycles": max_research_cycles,
        }

    def select_node(state: TongAgentState) -> dict[str, Any]:
        previous_plan = state["research_plan"]
        audited_plan = deepcopy(previous_plan)
        preselection_budget = budget_snapshot()
        rejected_covered = invalid_covered_subquestions(
            audited_plan, preselection_budget
        )
        for subquestion_id, reasons in rejected_covered.items():
            audited_plan = transition_subquestion(
                audited_plan,
                subquestion_id,
                "blocked",
                note="Covered state rejected: " + "; ".join(reasons),
            )
        plan, active = select_next_subquestion(audited_plan)
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
                "coverage": plan["coverage"],
                "budget_limits": budget.get("subquestion_limits", {}).get(active, {}),
            },
        )
        sources = budget.get("successful_sources", [])
        source_ids = [str(item.get("source_id", "")) for item in sources]
        evidence_ids = [
            str(item.get("evidence_id", ""))
            for item in budget.get("evidence_units", [])
        ]
        return {
            "research_plan": plan,
            "research_events": events,
            "active_subquestion_id": active,
            "active_source_ids_before": source_ids,
            "active_evidence_ids_before": evidence_ids,
            "budget_state": cast("BudgetState", budget),
            "workflow_phase": "researching" if active else "reporting",
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
        limits = budget.get("subquestion_limits", {}).get(active_id, {})
        usage = budget.get("subquestion_usage", {}).get(active_id, {})
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
Active subquestion: {active["id"]} — {active["question"]}
Rationale: {active["rationale"] or "Required for plan coverage."}
Reserved budget for this SQ: {json.dumps({"limits": limits, "usage": usage}, ensure_ascii=False)}

{delegation_instruction}Research only this active subquestion. Read the explicit plan with get_research_plan when useful. After each successful fetch, call record_evidence for every factual proposition you may report, copying an exact 12-800 character excerpt from that fetched page. The claim argument must itself be a self-contained report-ready sentence with subject and predicate, never a label such as "official name" or "contact email"; quote is the separate exact page excerpt that supports it. OMIT claim_id when creating a new claim: the tool assigns C#. Pass claim_id only to add evidence to a C# already returned by a successful record_evidence call; never invent C#. If a tool call fails, read its error, correct the arguments, and retry within budget. Then call get_evidence_graph and pass update_subquestion exactly the [S#] IDs linked to this SQ's canonical [C#] claims. The final covered update also enforces the policy's minimum web-search and independent-source counts; if it reports either minimum is unmet, continue researching and retry instead of blocking. A source alone cannot cover an SQ. Search snippets never receive source IDs or evidence units. If no supported claim can be registered within the reserved budget, mark the SQ blocked. Do not write the final report during this step."""
        message_id = (
            f"research-step-{state['research_plan']['plan_id']}-{active['id']}-"
            f"{active['attempts']}"
        )
        return {
            "messages": [HumanMessage(content=content, id=message_id)],
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
        if active_id:
            active = next(
                item for item in plan["subquestions"] if item["id"] == active_id
            )
            if active["status"] == "researching":
                if active["attempts"] >= active["max_attempts"]:
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
                        note="No new successful evidence; retry is allowed.",
                    )
        for subquestion_id, reasons in invalid_covered_subquestions(
            plan, budget
        ).items():
            plan = transition_subquestion(
                plan,
                subquestion_id,
                "blocked",
                note="Covered state rejected: " + "; ".join(reasons),
            )
        plan = refresh_plan_status(plan)
        cycles = int(state.get("research_cycles", 0)) + 1
        events = append_research_event(
            list(state.get("research_events", [])),
            "coverage_evaluated",
            plan_id=plan["plan_id"],
            subquestion_id=active_id or "",
            details={
                "coverage": plan["coverage"],
                "plan_status": plan["status"],
                "new_source_ids": new_ids,
                "new_evidence_ids": new_evidence_ids,
                "cycle": cycles,
            },
        )
        return {
            "research_plan": plan,
            "research_events": events,
            "budget_state": cast("BudgetState", budget),
            "active_subquestion_id": None,
            "research_cycles": cycles,
            "workflow_phase": "evaluating",
        }

    def route_after_evaluate(state: TongAgentState) -> Route:
        plan = state["research_plan"]
        unfinished = any(
            item["status"] in {"pending", "researching"}
            for item in plan["subquestions"]
        )
        within_limit = int(state.get("research_cycles", 0)) < int(
            state.get("max_research_cycles", max_research_cycles)
        )
        return "research" if unfinished and within_limit else "report"

    def prepare_report_node(state: TongAgentState) -> dict[str, Any]:
        if budget_activate is not None:
            budget_activate(None)
        plan = deepcopy(state["research_plan"])
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
        events = append_research_event(
            list(state.get("research_events", [])),
            "report_requested",
            plan_id=plan["plan_id"],
            details={"coverage": plan["coverage"], "status": plan["status"]},
        )
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        budget = budget_snapshot()
        ledger_json = json.dumps(
            budget.get("successful_sources", []), ensure_ascii=False, indent=2
        )
        evidence_graph_json = json.dumps(
            {
                "claims": budget.get("claims", []),
                "evidence_units": budget.get("evidence_units", []),
                "conflicts": budget.get("conflicts", []),
            },
            ensure_ascii=False,
            indent=2,
        )
        content = f"""[FINAL SYNTHESIS]
You MUST call write_file to create the final `/report.md` for the root question using only the canonical evidence graph below. Use exactly these four H2 section headings in this order with Sources last: `## Short Answer`, `## Key Findings`, `## Conflicts and Caveats`, and `## Sources`; do not add other headings. Every non-empty finding line must contain exactly one canonical `claim.text` copied byte-for-byte from the graph plus its `[C#]` and linked `[S#]`; copy `claim.text`, NOT the evidence quote. The only valid shape is `- <exact claim.text> [C#][S#]`: add no prefix, suffix, emphasis, or local paraphrase. Every canonical claim_id attached to a covered SQ in PLAN MUST appear at least once, even when two claim texts look redundant. Put contested claims only in Conflicts and Caveats and cite both supporting and contradicting sources. A generic process or evidence-quality caveat in that section must contain no `[C#]` or `[S#]`; never attach a supported claim ID to locally written caveat text. Never present contradicted-only claims as facts. Every Sources line MUST have exactly this shape: `- [S#] <exact canonical title> — <canonical URL>`. If there are no reportable claims, leave Short Answer and Key Findings empty and explain the limitation only under Conflicts and Caveats. Search snippets and failed fetches are not evidence. If the plan is partial, write an honest partial report without claiming blocked work was completed.

PLAN:
{plan_json}

CANONICAL SOURCE LEDGER:
{ledger_json}

CANONICAL EVIDENCE GRAPH:
{evidence_graph_json}"""
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
        }

    def finish_node(state: TongAgentState) -> dict[str, Any]:
        plan = refresh_plan_status(state["research_plan"])
        budget = budget_snapshot()
        events = append_research_event(
            list(state.get("research_events", [])),
            "run_finished",
            plan_id=plan["plan_id"],
            details={"coverage": plan["coverage"], "status": plan["status"]},
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
        result = research_agent.invoke(state, config=config)
        return {
            **result,
            "budget_state": cast("BudgetState", budget_snapshot()),
        }

    builder = StateGraph(TongAgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("select", select_node)
    builder.add_node("prepare_research", prepare_research_node)
    builder.add_node("research_agent", invoke_inner_agent)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("prepare_report", prepare_report_node)
    builder.add_node("report_agent", invoke_inner_agent)
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
        {"research": "select", "report": "prepare_report"},
    )
    builder.add_edge("prepare_report", "report_agent")
    builder.add_edge("report_agent", "finish")
    builder.add_edge("finish", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        name="tongagent-research-workflow",
    )
