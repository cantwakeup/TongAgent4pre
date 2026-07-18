"""Deterministic Stage 03D assessment and bounded control decisions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from evidence_graph import (
    source_diversity_metrics,
    validate_evidence_graph,
)
from research_state import (
    AdaptiveControlState,
    ControlAction,
    ControlAssessment,
    ControlDecision,
    ResearchPlan,
    ResearchStrategy,
)


CONTROL_SCHEMA_VERSION = 1


def initialize_control_state(
    *,
    strategy: ResearchStrategy,
    config_fingerprint: str,
    hard_effort: str,
    pinned_model: str,
    pinned_topology: str,
    max_escalations: int,
) -> AdaptiveControlState:
    """Create fresh checkpointable controller state for one research plan."""
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "strategy": strategy,
        "config_fingerprint": config_fingerprint,
        "hard_effort": hard_effort,
        "pinned_model": pinned_model,
        "pinned_topology": pinned_topology,
        "max_escalations": max(0, max_escalations),
        "escalation_count": 0,
        "escalations_by_subquestion": {},
        "no_progress_streak_by_subquestion": {},
        "decision_history": [],
        "stop_reason": "",
    }


def _plan_evidence_ids(plan: ResearchPlan) -> tuple[set[str], set[str]]:
    source_ids = {
        str(source_id)
        for item in plan.get("subquestions", [])
        for source_id in item.get("evidence_source_ids", [])
    }
    claim_ids = {
        str(claim_id)
        for item in plan.get("subquestions", [])
        for claim_id in item.get("claim_ids", [])
    }
    return source_ids, claim_ids


def build_control_assessment(
    *,
    plan: ResearchPlan,
    budget: dict[str, Any],
    control: AdaptiveControlState,
    subquestion_id: str,
    cycle: int,
    new_source_ids: list[str],
    new_claim_ids: list[str],
    new_evidence_ids: list[str],
    new_conflict_ids: list[str],
    new_tool_attempt_ids: list[str],
) -> ControlAssessment:
    """Reduce plan, evidence, attempts, and remaining budget into typed signals."""
    limits = budget.get("subquestion_limits", {}).get(subquestion_id, {})
    usage = budget.get("subquestion_usage", {}).get(subquestion_id, {})
    remaining_searches = max(
        0,
        int(limits.get("max_searches", 0)) - int(usage.get("search_calls", 0)),
    )
    remaining_fetches = max(
        0,
        int(limits.get("max_fetches", 0)) - int(usage.get("fetch_calls", 0)),
    )
    reserve_searches = max(0, int(budget.get("reserve_searches", 0)))
    reserve_fetches = max(0, int(budget.get("reserve_fetches", 0)))
    target = next(
        (
            item
            for item in plan.get("subquestions", [])
            if item.get("id") == subquestion_id
        ),
        None,
    )
    eligible_claims = [
        item
        for item in budget.get("claims", [])
        if item.get("subquestion_id") == subquestion_id
        and item.get("status") in {"supported", "contested"}
    ]
    source_ids, claim_ids = _plan_evidence_ids(plan)
    diversity = source_diversity_metrics(
        source_ids=source_ids,
        sources=budget.get("successful_sources", []),
        evidence_units=budget.get("evidence_units", []),
        claim_ids=claim_ids,
    )
    corroborating_count = int(diversity["corroborating_source_group_count"])
    minimum_sources = int(budget.get("min_successful_sources", 0))
    raw_relevant_searches = budget.get("relevant_searches")
    relevant_searches = (
        int(raw_relevant_searches) if raw_relevant_searches is not None else None
    )
    raw_scoped_relevant = usage.get("relevant_searches")
    scoped_relevant_searches = (
        int(raw_scoped_relevant) if raw_scoped_relevant is not None else None
    )
    current_attempt_ids = set(new_tool_attempt_ids)
    current_attempts = [
        item
        for item in budget.get("tool_attempts", [])
        if str(item.get("attempt_id", "")) in current_attempt_ids
    ]
    reason_codes: list[str] = []
    integrity_errors = validate_evidence_graph(budget)
    if integrity_errors:
        reason_codes.append("integrity_failure")
    if plan.get("status") == "completed":
        reason_codes.append("plan_completed")
    elif target is not None and target.get("status") not in {"covered"}:
        if not eligible_claims:
            reason_codes.append("claim_gap")
        if corroborating_count < minimum_sources:
            reason_codes.append("corroborating_source_gap")
        if (scoped_relevant_searches or 0) < 1:
            reason_codes.append("relevant_search_gap")
    if new_conflict_ids:
        reason_codes.append("new_conflict")
    if any(
        item.get("failure_class") in {"network", "provider"}
        for item in current_attempts
    ):
        reason_codes.append("provider_failure")
    for attempt in current_attempts:
        tool_name = str(attempt.get("tool", ""))
        outcome = str(attempt.get("outcome", ""))
        retryable = bool(attempt.get("retryable", False))
        retry_needed = outcome == "budget_exceeded" or (
            retryable and outcome != "success"
        )
        if tool_name == "web_search" and retry_needed:
            reason_codes.append("search_retry_needed")
            if outcome == "low_relevance":
                reason_codes.append("search_quality_gap")
                reason_codes.append("low_relevance_candidates")
            if outcome == "empty_results":
                reason_codes.append("search_empty_results")
        if tool_name == "fetch_url" and retry_needed:
            reason_codes.append("fetch_retry_needed")
    made_progress = bool(new_source_ids or new_claim_ids or new_evidence_ids)
    streaks = control.get("no_progress_streak_by_subquestion", {})
    previous_streak = int(streaks.get(subquestion_id, 0))
    no_progress_streak = 0 if made_progress else previous_streak + 1
    if not made_progress and target is not None and target.get("status") != "covered":
        reason_codes.append("no_progress")
    if remaining_searches == 0 and remaining_fetches == 0:
        reason_codes.append("slice_exhausted")
    if reserve_searches == 0 and reserve_fetches == 0:
        reason_codes.append("hard_ceiling_reached")
    return {
        "subquestion_id": subquestion_id,
        "cycle": cycle,
        "plan_status": plan.get("status", "partial"),
        "new_source_ids": list(new_source_ids),
        "new_claim_ids": list(new_claim_ids),
        "new_evidence_ids": list(new_evidence_ids),
        "new_conflict_ids": list(new_conflict_ids),
        "new_tool_attempt_ids": list(new_tool_attempt_ids),
        "remaining_searches": remaining_searches,
        "remaining_fetches": remaining_fetches,
        "reserve_searches": reserve_searches,
        "reserve_fetches": reserve_fetches,
        "no_progress_streak": no_progress_streak,
        "provider_successes": budget.get("provider_successes"),
        "nonempty_searches": budget.get("nonempty_searches"),
        "relevant_searches": relevant_searches,
        "current_subquestion_relevant_searches": scoped_relevant_searches,
        "evidence_producing_searches": budget.get("evidence_producing_searches"),
        "search_metric_availability": dict(
            budget.get("search_metric_availability", {})
        ),
        "distinct_content_revision_count": int(
            diversity["distinct_content_revision_count"]
        ),
        "distinct_source_host_count": int(diversity["distinct_source_host_count"]),
        "corroborating_source_group_count": corroborating_count,
        "current_search_attempts": [
            {
                "attempt_id": item.get("attempt_id"),
                "provider_outcome": item.get("provider_outcome"),
                "nonempty_search": item.get("nonempty_search"),
                "relevant_search": item.get("relevant_search"),
                "evidence_producing_search": item.get("evidence_producing_search"),
                "outcome": item.get("outcome"),
                "failure_class": item.get("failure_class"),
            }
            for item in current_attempts
            if item.get("tool") == "web_search"
        ],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "integrity_errors": integrity_errors,
    }


def decide_control_action(
    *,
    plan: ResearchPlan,
    control: AdaptiveControlState,
    assessment: ControlAssessment,
    cycle_limit_reached: bool,
) -> tuple[ControlAction, list[str]]:
    """Choose a bounded action without consulting a model or mutating state."""
    reasons = list(assessment.get("reason_codes", []))
    if assessment.get("integrity_errors"):
        return "fail_closed", [*reasons, "integrity_failure"]
    if plan.get("status") == "completed":
        return "finish_success", [*reasons, "plan_completed"]
    unfinished = [
        item
        for item in plan.get("subquestions", [])
        if item.get("status") in {"pending", "researching"}
    ]
    if not unfinished:
        return "finish_partial", [*reasons, "no_researchable_subquestions"]
    subquestion_id = str(assessment.get("subquestion_id", ""))
    target = next(
        (
            item
            for item in plan.get("subquestions", [])
            if item.get("id") == subquestion_id
        ),
        None,
    )
    if cycle_limit_reached:
        if target is not None and target.get("status") in {"pending", "researching"}:
            return "stop_subquestion", [*reasons, "cycle_ceiling_reached"]
        return "finish_partial", [*reasons, "cycle_ceiling_reached"]
    if target is None or target.get("status") == "covered":
        return "continue", [*reasons, "next_subquestion"]
    if target.get("status") == "blocked":
        return "stop_subquestion", [*reasons, "subquestion_blocked"]

    no_progress_streak = int(assessment.get("no_progress_streak", 0))
    if no_progress_streak >= 2 and not assessment.get("new_tool_attempt_ids"):
        return "stop_subquestion", [*reasons, "no_progress_ceiling_reached"]

    remaining_searches = int(assessment.get("remaining_searches", 0))
    remaining_fetches = int(assessment.get("remaining_fetches", 0))
    search_retry_needed = "search_retry_needed" in reasons
    fetch_retry_needed = "fetch_retry_needed" in reasons
    needs_search = "relevant_search_gap" in reasons or (
        "provider_failure" in reasons
        and not search_retry_needed
        and not fetch_retry_needed
    )
    needs_evidence = any(
        reason in reasons for reason in ("claim_gap", "corroborating_source_gap")
    )
    current_candidate_fetch_can_help = (
        "low_relevance_candidates" in reasons
        and needs_evidence
        and remaining_fetches > 0
    )
    if search_retry_needed or fetch_retry_needed:
        current_budget_can_help = current_candidate_fetch_can_help or (
            (not search_retry_needed or remaining_searches > 0)
            and (not fetch_retry_needed or remaining_fetches > 0)
        )
    elif needs_search:
        current_budget_can_help = remaining_searches > 0
    elif needs_evidence:
        current_budget_can_help = remaining_searches > 0 or remaining_fetches > 0
    else:
        current_budget_can_help = remaining_searches > 0 or remaining_fetches > 0
    if current_budget_can_help:
        return "continue", [*reasons, "current_slice_remaining"]

    # Newly registered state gets one model-only update/extraction retry.
    if (
        assessment.get("new_source_ids")
        or assessment.get("new_claim_ids")
        or assessment.get("new_evidence_ids")
    ) and no_progress_streak <= 1:
        return "continue", [*reasons, "state_update_retry"]

    escalation_count = int(control.get("escalation_count", 0))
    max_escalations = int(control.get("max_escalations", 0))
    reserve_searches = int(assessment.get("reserve_searches", 0))
    reserve_fetches = int(assessment.get("reserve_fetches", 0))
    reserve_candidate_fetch_can_help = (
        "low_relevance_candidates" in reasons and needs_evidence and reserve_fetches > 0
    )
    if search_retry_needed or fetch_retry_needed:
        reserve_can_help = reserve_candidate_fetch_can_help or (
            (not search_retry_needed or reserve_searches > 0)
            and (not fetch_retry_needed or reserve_fetches > 0)
        )
    elif needs_search:
        reserve_can_help = reserve_searches > 0
    elif needs_evidence:
        reserve_can_help = reserve_searches > 0 or reserve_fetches > 0
    else:
        reserve_can_help = reserve_searches > 0 or reserve_fetches > 0
    if escalation_count < max_escalations and reserve_can_help:
        return "expand_budget", [*reasons, "recoverable_budget_gap"]
    if escalation_count >= max_escalations:
        reasons.append("escalation_ceiling_reached")
    if not reserve_can_help:
        reasons.append("hard_ceiling_reached")
    return "stop_subquestion", list(dict.fromkeys(reasons))


def record_control_decision(
    control: AdaptiveControlState,
    *,
    decision_id: str,
    action: ControlAction,
    assessment: ControlAssessment,
    reason_codes: list[str],
    budget_before: dict[str, int],
    budget_after: dict[str, int],
) -> AdaptiveControlState:
    """Append one idempotent decision to checkpoint state."""
    updated = deepcopy(control)
    history = list(updated.get("decision_history", []))
    if any(item.get("decision_id") == decision_id for item in history):
        return updated
    sequence = len(history) + 1
    subquestion_id = str(assessment.get("subquestion_id", ""))
    decision: ControlDecision = {
        "decision_id": decision_id,
        "sequence": sequence,
        "cycle": int(assessment.get("cycle", 0)),
        "subquestion_id": subquestion_id,
        "action": action,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "budget_before": dict(budget_before),
        "budget_after": dict(budget_after),
    }
    history.append(decision)
    updated["decision_history"] = history
    updated["last_assessment"] = deepcopy(assessment)
    streaks = dict(updated.get("no_progress_streak_by_subquestion", {}))
    if subquestion_id:
        streaks[subquestion_id] = int(assessment.get("no_progress_streak", 0))
    updated["no_progress_streak_by_subquestion"] = streaks
    if action == "expand_budget":
        updated["escalation_count"] = int(updated.get("escalation_count", 0)) + 1
        by_subquestion = dict(updated.get("escalations_by_subquestion", {}))
        by_subquestion[subquestion_id] = int(by_subquestion.get(subquestion_id, 0)) + 1
        updated["escalations_by_subquestion"] = by_subquestion
    if action in {
        "stop_subquestion",
        "finish_success",
        "finish_partial",
        "fail_closed",
    }:
        canonical_stop_reasons = {
            "finish_success": "plan_completed",
            "fail_closed": "integrity_failure",
        }
        updated["stop_reason"] = canonical_stop_reasons.get(
            action,
            str(reason_codes[-1] if reason_codes else action),
        )
    return updated


def decision_id_for(
    plan_id: str, cycle: int, sequence: int, subquestion_id: str
) -> str:
    """Return a deterministic grant/idempotency key for one controller step."""
    return f"{plan_id}:cycle-{cycle}:decision-{sequence}:{subquestion_id or 'plan'}"
