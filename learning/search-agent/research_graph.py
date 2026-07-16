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

from research_state import (
    BudgetState,
    ResearchEvent,
    ResearchPlan,
    SubQuestion,
    SubquestionStatus,
    TongAgentState,
)
from telemetry import append_research_event


Planner = Callable[[str, int], ResearchPlan]
BudgetSnapshot = Callable[[], dict[str, Any]]
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
    subquestions = updated.get("subquestions", [])
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
    note: str = "",
    increment_attempt: bool = False,
) -> ResearchPlan:
    """Apply one validated subquestion transition without mutating input state.

    Args:
        plan: Current durable plan.
        subquestion_id: Stable `SQ#` identifier.
        status: Desired next status.
        evidence_source_ids: Evidence IDs associated with a covered item.
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
        "covered": {"covered"},
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
    if status == "covered" and not evidence and not target["evidence_source_ids"]:
        msg = "Covered subquestions require at least one evidence source ID"
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


def build_research_state_tools(budget_snapshot: BudgetSnapshot) -> list[BaseTool]:
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
        if status == "covered":
            ledger_ids = {
                str(item.get("source_id", ""))
                for item in budget_snapshot().get("successful_sources", [])
            }
            requested_ids = set(evidence_source_ids)
            unknown_ids = sorted(requested_ids - ledger_ids)
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
            previous_ids = set(runtime.state.get("active_source_ids_before", []))
            current_step_ids = ledger_ids - previous_ids
            if not requested_ids.intersection(current_step_ids):
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
            details={"status": status, "evidence_source_ids": evidence_source_ids},
        )
        content = json.dumps(
            {
                "status": "success",
                "subquestion_id": subquestion_id,
                "new_status": status,
                "coverage": updated["coverage"],
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
    checkpointer: Any | None,
    max_subquestions: int,
    max_research_cycles: int,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Compile the outer plan-select-research-evaluate-report workflow.

    Args:
        research_agent: Uncheckpointed Deep Agent subgraph or compatible node.
        planner: Injectable structured planner.
        budget_snapshot: Callable returning the latest serializable ledger.
        checkpointer: Checkpointer owned exclusively by the outer graph.
        max_subquestions: Hard plan breadth limit.
        max_research_cycles: Loop limit protecting against stalled model behavior.
        interrupt_before: Optional node interrupts used by recovery tests.

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
        return {
            "research_plan": plan,
            "research_plan_history": history,
            "research_events": events,
            "active_subquestion_id": None,
            "active_source_ids_before": [],
            "budget_state": cast("BudgetState", budget_snapshot()),
            "workflow_phase": "selecting",
            "research_cycles": 0,
            "max_research_cycles": max_research_cycles,
        }

    def select_node(state: TongAgentState) -> dict[str, Any]:
        previous_plan = state["research_plan"]
        plan, active = select_next_subquestion(state["research_plan"])
        events = list(state.get("research_events", []))
        previous_statuses = {
            item["id"]: item["status"] for item in previous_plan["subquestions"]
        }
        for item in plan["subquestions"]:
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
            details={"coverage": plan["coverage"]},
        )
        sources = budget_snapshot().get("successful_sources", [])
        source_ids = [str(item.get("source_id", "")) for item in sources]
        return {
            "research_plan": plan,
            "research_events": events,
            "active_subquestion_id": active,
            "active_source_ids_before": source_ids,
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
        content = f"""[RESEARCH STEP]
Root question: {state["research_plan"]["question"]}
Active subquestion: {active["id"]} — {active["question"]}
Rationale: {active["rationale"] or "Required for plan coverage."}

Research only this active subquestion. Read the explicit plan with get_research_plan when useful. Gather successfully fetched evidence, then call update_subquestion with status=covered and the relevant [S#] IDs. If it cannot be answered within the active budget, mark it blocked with a concrete reason. Do not write the final report during this step."""
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
        content = f"""[FINAL SYNTHESIS]
You MUST call write_file to create the final `/report.md` for the root question using the accumulated fetched evidence and the explicit plan below. Include a short answer, key findings, caveats, unresolved or blocked subquestions, and a Sources section with full URLs. If the plan is partial, still write an honest partial report explaining the evidence gap. Do not claim that blocked work was completed.

{plan_json}"""
        return {
            "messages": [
                HumanMessage(
                    content=content,
                    id=f"report-step-{plan['plan_id']}-{state.get('research_cycles', 0)}",
                )
            ],
            "research_plan": plan,
            "research_events": events,
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
