"""Unit tests for deterministic Stage 03D control and bounded grants."""

from __future__ import annotations

from copy import deepcopy
import unittest

from adaptive_control import (
    build_control_assessment,
    decide_control_action,
    decision_id_for,
    initialize_control_state,
    record_control_decision,
)
from agent_policy import EFFORT_POLICIES
from research_state import (
    AdaptiveControlState,
    ControlAssessment,
    ResearchPlan,
)
from search_agent import ResearchBudget


def _plan(
    *,
    status: str = "in_progress",
    subquestion_status: str = "researching",
) -> ResearchPlan:
    """Return the smallest plan needed by the pure controller."""
    return {
        "plan_id": "plan-adaptive",
        "evidence_schema_version": 1,
        "question": "What is supported?",
        "objective": "Collect enough independent evidence.",
        "planner": "test",
        "status": status,
        "coverage": 1.0 if status == "completed" else 0.0,
        "completion_criteria": ["Support the claim."],
        "subquestions": [
            {
                "id": "SQ1",
                "question": "Establish the primary fact.",
                "rationale": "The answer depends on it.",
                "depends_on": [],
                "status": subquestion_status,
                "attempts": 1,
                "max_attempts": 2,
                "evidence_source_ids": [],
                "claim_ids": [],
                "conflict_ids": [],
                "note": "",
            }
        ],
    }


def _control(*, max_escalations: int = 1) -> AdaptiveControlState:
    """Return a pinned controller state with a deterministic identity."""
    return initialize_control_state(
        strategy="adaptive",
        config_fingerprint="fingerprint",
        hard_effort="medium",
        pinned_model="model",
        pinned_topology="multi",
        max_escalations=max_escalations,
    )


def _assessment(
    *,
    remaining_searches: int = 0,
    remaining_fetches: int = 0,
    reserve_searches: int = 0,
    reserve_fetches: int = 0,
    reason_codes: list[str] | None = None,
    integrity_errors: list[str] | None = None,
) -> ControlAssessment:
    """Return explicit control inputs without involving graph side effects."""
    return {
        "subquestion_id": "SQ1",
        "cycle": 2,
        "plan_status": "in_progress",
        "new_source_ids": [],
        "new_claim_ids": [],
        "new_evidence_ids": [],
        "new_conflict_ids": [],
        "new_tool_attempt_ids": [],
        "remaining_searches": remaining_searches,
        "remaining_fetches": remaining_fetches,
        "reserve_searches": reserve_searches,
        "reserve_fetches": reserve_fetches,
        "no_progress_streak": 1,
        "reason_codes": list(reason_codes or []),
        "integrity_errors": list(integrity_errors or []),
    }


class AdaptiveControlDecisionTests(unittest.TestCase):
    """Verify that the controller is deterministic and fail-closed."""

    def test_completed_plan_finishes_successfully(self) -> None:
        action, reasons = decide_control_action(
            plan=_plan(status="completed", subquestion_status="covered"),
            control=_control(),
            assessment=_assessment(),
            cycle_limit_reached=False,
        )

        self.assertEqual(action, "finish_success")
        self.assertIn("plan_completed", reasons)

    def test_remaining_current_slice_continues_without_escalation(self) -> None:
        action, reasons = decide_control_action(
            plan=_plan(),
            control=_control(),
            assessment=_assessment(
                remaining_searches=1,
                reason_codes=["claim_gap"],
            ),
            cycle_limit_reached=False,
        )

        self.assertEqual(action, "continue")
        self.assertIn("current_slice_remaining", reasons)

    def test_exhausted_slice_with_reserve_expands_budget(self) -> None:
        action, reasons = decide_control_action(
            plan=_plan(),
            control=_control(),
            assessment=_assessment(
                reserve_searches=1,
                reserve_fetches=1,
                reason_codes=["claim_gap", "slice_exhausted"],
            ),
            cycle_limit_reached=False,
        )

        self.assertEqual(action, "expand_budget")
        self.assertIn("recoverable_budget_gap", reasons)

    def test_search_outcome_chooses_candidate_fetch_or_search_reserve(self) -> None:
        cases = (
            ("low_relevance", True, "content", "continue"),
            ("empty_results", True, "content", "expand_budget"),
            ("budget_exceeded", False, "budget", "expand_budget"),
        )
        for outcome, retryable, failure_class, expected_action in cases:
            with self.subTest(outcome=outcome):
                control = _control(max_escalations=2)
                assessment = build_control_assessment(
                    plan=_plan(),
                    budget={
                        "subquestion_limits": {
                            "SQ1": {"max_searches": 1, "max_fetches": 1}
                        },
                        "subquestion_usage": {
                            "SQ1": {"search_calls": 1, "fetch_calls": 0}
                        },
                        "reserve_searches": 1,
                        "reserve_fetches": 2,
                        "max_searches": 2,
                        "successful_searches": 1,
                        "min_successful_sources": 2,
                        "successful_sources": [],
                        "claims": [],
                        "evidence_units": [],
                        "conflicts": [],
                        "tool_attempts": [
                            {
                                "attempt_id": "A1",
                                "tool": "web_search",
                                "outcome": outcome,
                                "failure_class": failure_class,
                                "retryable": retryable,
                            }
                        ],
                    },
                    control=control,
                    subquestion_id="SQ1",
                    cycle=1,
                    new_source_ids=[],
                    new_claim_ids=[],
                    new_evidence_ids=[],
                    new_conflict_ids=[],
                    new_tool_attempt_ids=["A1"],
                )

                action, _ = decide_control_action(
                    plan=_plan(),
                    control=control,
                    assessment=assessment,
                    cycle_limit_reached=False,
                )

                self.assertEqual(assessment["remaining_searches"], 0)
                self.assertEqual(assessment["remaining_fetches"], 1)
                self.assertIn("search_retry_needed", assessment["reason_codes"])
                self.assertEqual(
                    "search_quality_gap" in assessment["reason_codes"],
                    outcome == "low_relevance",
                )
                self.assertEqual(
                    "low_relevance_candidates" in assessment["reason_codes"],
                    outcome == "low_relevance",
                )
                self.assertEqual(
                    "search_empty_results" in assessment["reason_codes"],
                    outcome == "empty_results",
                )
                self.assertEqual(action, expected_action)

    def test_low_relevance_candidate_can_survive_a_retryable_fetch_failure(
        self,
    ) -> None:
        control = _control(max_escalations=1)
        assessment = build_control_assessment(
            plan=_plan(),
            budget={
                "subquestion_limits": {"SQ1": {"max_searches": 1, "max_fetches": 2}},
                "subquestion_usage": {"SQ1": {"search_calls": 1, "fetch_calls": 1}},
                "reserve_searches": 0,
                "reserve_fetches": 0,
                "max_searches": 1,
                "successful_searches": 1,
                "min_successful_sources": 1,
                "successful_sources": [],
                "claims": [],
                "evidence_units": [],
                "conflicts": [],
                "tool_attempts": [
                    {
                        "attempt_id": "A1",
                        "tool": "web_search",
                        "outcome": "low_relevance",
                        "failure_class": "content",
                        "retryable": True,
                    },
                    {
                        "attempt_id": "A2",
                        "tool": "fetch_url",
                        "outcome": "network_error",
                        "failure_class": "network",
                        "retryable": True,
                    },
                ],
            },
            control=control,
            subquestion_id="SQ1",
            cycle=1,
            new_source_ids=[],
            new_claim_ids=[],
            new_evidence_ids=[],
            new_conflict_ids=[],
            new_tool_attempt_ids=["A1", "A2"],
        )

        action, reasons = decide_control_action(
            plan=_plan(),
            control=control,
            assessment=assessment,
            cycle_limit_reached=False,
        )

        self.assertEqual(assessment["remaining_searches"], 0)
        self.assertEqual(assessment["remaining_fetches"], 1)
        self.assertIn("search_retry_needed", reasons)
        self.assertIn("fetch_retry_needed", reasons)
        self.assertEqual(action, "continue")
        self.assertIn("current_slice_remaining", reasons)

    def test_escalation_or_hard_budget_ceiling_stops_subquestion(self) -> None:
        escalation_control = _control(max_escalations=1)
        escalation_control["escalation_count"] = 1
        cases = (
            (
                escalation_control,
                _assessment(
                    reserve_searches=1,
                    reason_codes=["claim_gap", "slice_exhausted"],
                ),
                "escalation_ceiling_reached",
            ),
            (
                _control(max_escalations=2),
                _assessment(reason_codes=["claim_gap", "slice_exhausted"]),
                "hard_ceiling_reached",
            ),
        )

        for control, assessment, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                action, reasons = decide_control_action(
                    plan=_plan(),
                    control=control,
                    assessment=assessment,
                    cycle_limit_reached=False,
                )
                self.assertEqual(action, "stop_subquestion")
                self.assertIn(expected_reason, reasons)

    def test_integrity_error_fails_closed_before_other_actions(self) -> None:
        action, reasons = decide_control_action(
            plan=_plan(),
            control=_control(),
            assessment=_assessment(
                remaining_searches=1,
                integrity_errors=["evidence E1 references unknown source S9"],
            ),
            cycle_limit_reached=False,
        )

        self.assertEqual(action, "fail_closed")
        self.assertIn("integrity_failure", reasons)

    def test_cycle_ceiling_after_covered_sq_finishes_partial_not_continue(self) -> None:
        plan = _plan(subquestion_status="covered")
        second = deepcopy(plan["subquestions"][0])
        second["id"] = "SQ2"
        second["status"] = "pending"
        plan["subquestions"].append(second)

        action, reasons = decide_control_action(
            plan=plan,
            control=_control(),
            assessment=_assessment(),
            cycle_limit_reached=True,
        )

        self.assertEqual(action, "finish_partial")
        self.assertIn("cycle_ceiling_reached", reasons)

    def test_recording_is_immutable_and_idempotent_by_decision_id(self) -> None:
        control = _control(max_escalations=2)
        assessment = _assessment(
            reserve_searches=2,
            reason_codes=["claim_gap", "slice_exhausted"],
        )
        decision_id = decision_id_for("plan-adaptive", 2, 1, "SQ1")
        updated = record_control_decision(
            control,
            decision_id=decision_id,
            action="expand_budget",
            assessment=assessment,
            reason_codes=["claim_gap", "recoverable_budget_gap"],
            budget_before={"searches": 1, "fetches": 1},
            budget_after={"searches": 2, "fetches": 1},
        )
        replayed = record_control_decision(
            updated,
            decision_id=decision_id,
            action="expand_budget",
            assessment=assessment,
            reason_codes=["claim_gap", "recoverable_budget_gap"],
            budget_before={"searches": 1, "fetches": 1},
            budget_after={"searches": 2, "fetches": 1},
        )

        self.assertEqual(control["decision_history"], [])
        self.assertEqual(len(updated["decision_history"]), 1)
        self.assertEqual(updated["escalation_count"], 1)
        self.assertEqual(updated["escalations_by_subquestion"], {"SQ1": 1})
        self.assertEqual(replayed, updated)
        self.assertEqual(
            decision_id,
            decision_id_for("plan-adaptive", 2, 1, "SQ1"),
        )
        self.assertNotEqual(
            decision_id,
            decision_id_for("plan-adaptive", 2, 2, "SQ1"),
        )

    def test_success_stop_reason_is_not_overwritten_by_incidental_gap(self) -> None:
        finished = record_control_decision(
            _control(),
            decision_id="finish-decision",
            action="finish_success",
            assessment=_assessment(),
            reason_codes=["plan_completed", "slice_exhausted"],
            budget_before={},
            budget_after={},
        )

        self.assertEqual(finished["stop_reason"], "plan_completed")


class AdaptiveBudgetTests(unittest.TestCase):
    """Verify reserve release, hard caps, and strict checkpoint restore."""

    def test_adaptive_configuration_exposes_baseline_and_keeps_reserve(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["medium"], strategy="adaptive")
        budget.configure_subquestions(["SQ1", "SQ2"])

        snapshot = budget.snapshot()
        self.assertEqual(
            snapshot["subquestion_limits"],
            {
                "SQ1": {"max_searches": 1, "max_fetches": 1},
                "SQ2": {"max_searches": 1, "max_fetches": 1},
            },
        )
        self.assertEqual(snapshot["granted_searches"], 2)
        self.assertEqual(snapshot["granted_fetches"], 2)
        self.assertEqual(snapshot["reserve_searches"], 2)
        self.assertEqual(snapshot["reserve_fetches"], 4)

    def test_grants_are_monotonic_idempotent_and_hard_capped(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        budget.configure_subquestions(["SQ1"])
        baseline = budget.snapshot()["subquestion_limits"]["SQ1"]

        first = budget.grant_subquestion(
            "decision-1",
            "SQ1",
            search_delta=1,
            fetch_delta=1,
        )
        after_first = budget.snapshot()["subquestion_limits"]["SQ1"]
        replay = budget.grant_subquestion(
            "decision-1",
            "SQ1",
            search_delta=99,
            fetch_delta=99,
        )
        second = budget.grant_subquestion(
            "decision-2",
            "SQ1",
            search_delta=99,
            fetch_delta=99,
        )
        final = budget.snapshot()

        self.assertTrue(first["applied"])
        self.assertGreaterEqual(after_first["max_searches"], baseline["max_searches"])
        self.assertGreaterEqual(after_first["max_fetches"], baseline["max_fetches"])
        self.assertFalse(replay["applied"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["before"], first["before"])
        self.assertEqual(replay["after"], first["after"])
        self.assertEqual(replay["added"], first["added"])
        self.assertTrue(second["applied"])
        self.assertEqual(
            final["subquestion_limits"]["SQ1"]["max_searches"],
            EFFORT_POLICIES["low"].max_searches,
        )
        self.assertEqual(
            final["subquestion_limits"]["SQ1"]["max_fetches"],
            EFFORT_POLICIES["low"].max_fetches,
        )
        self.assertEqual(final["reserve_searches"], 0)
        self.assertEqual(final["reserve_fetches"], 0)
        self.assertEqual(final["applied_grant_ids"], ["decision-1", "decision-2"])

    def test_one_grant_cannot_release_multiple_reserve_units(self) -> None:
        budget = ResearchBudget(EFFORT_POLICIES["medium"], strategy="adaptive")
        budget.configure_subquestions(["SQ1"])

        result = budget.grant_subquestion(
            "decision-1",
            "SQ1",
            search_delta=99,
            fetch_delta=99,
        )
        snapshot = budget.snapshot()

        self.assertEqual(result["added"], {"searches": 1, "fetches": 1})
        self.assertEqual(
            snapshot["subquestion_limits"]["SQ1"],
            {"max_searches": 2, "max_fetches": 2},
        )

    def test_strict_restore_accepts_matching_adaptive_checkpoint(self) -> None:
        original = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        original.configure_subquestions(["SQ1"])
        original.activate_subquestion("SQ1")
        self.assertTrue(original.reserve_search())
        original.record_tool_attempt(
            tool_name="web_search",
            target="checkpoint fixture",
            payload={
                "status": "success",
                "results": [
                    {
                        "url": "https://checkpoint.test/page",
                        "relevance_score": 100,
                    }
                ],
                "relevant_results": 1,
            },
        )
        snapshot = original.snapshot()

        restored = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        restored.restore(snapshot, strict_policy=True)

        self.assertEqual(restored.snapshot(), snapshot)

    def test_strict_restore_preserves_durable_grant_records(self) -> None:
        original = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        original.configure_subquestions(["SQ1"])
        first = original.grant_subquestion(
            "decision-1",
            "SQ1",
            search_delta=1,
            fetch_delta=1,
        )
        snapshot = original.snapshot()
        restored = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")

        restored.restore(snapshot, strict_policy=True)
        replay = restored.grant_subquestion(
            "decision-1",
            "SQ1",
            search_delta=1,
            fetch_delta=1,
        )

        self.assertEqual(restored.snapshot(), snapshot)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["budget_before"], first["budget_before"])
        self.assertEqual(replay["budget_after"], first["budget_after"])

    def test_strict_restore_rejects_policy_drift_and_invalid_grants(self) -> None:
        original = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        original.configure_subquestions(["SQ1"])
        snapshot = original.snapshot()

        cases: list[tuple[ResearchBudget, dict[str, object], str]] = [
            (
                ResearchBudget(EFFORT_POLICIES["medium"], strategy="adaptive"),
                snapshot,
                "effort does not match",
            ),
            (
                ResearchBudget(EFFORT_POLICIES["low"], strategy="fixed"),
                snapshot,
                "strategy does not match",
            ),
        ]
        over_cap = deepcopy(snapshot)
        over_cap["subquestion_limits"]["SQ1"]["max_searches"] = 3
        cases.append(
            (
                ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive"),
                over_cap,
                "search grants exceed",
            )
        )
        over_usage = deepcopy(snapshot)
        over_usage["subquestion_usage"]["SQ1"]["search_calls"] = 2
        cases.append(
            (
                ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive"),
                over_usage,
                "search usage exceeds grant",
            )
        )
        fractional_grant = deepcopy(snapshot)
        fractional_grant["subquestion_limits"]["SQ1"]["max_searches"] = 1.5
        cases.append(
            (
                ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive"),
                fractional_grant,
                "is not an integer",
            )
        )
        forged_multi_release = deepcopy(snapshot)
        forged_multi_release["subquestion_limits"]["SQ1"] = {
            "max_searches": 2,
            "max_fetches": 3,
        }
        forged_multi_release["applied_grant_ids"] = ["decision-1"]
        cases.append(
            (
                ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive"),
                forged_multi_release,
                "one-step grant history",
            )
        )

        for restored, candidate, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    restored.restore(candidate, strict_policy=True)

    def test_strict_restore_rejects_negative_counters_before_budget_use(self) -> None:
        original = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")
        original.configure_subquestions(["SQ1"])
        original.grant_subquestion(
            "decision-1",
            "SQ1",
            search_delta=1,
            fetch_delta=1,
        )
        corrupted = original.snapshot()
        corrupted["search_calls"] = -1
        corrupted["subquestion_usage"]["SQ1"]["search_calls"] = -1
        restored = ResearchBudget(EFFORT_POLICIES["low"], strategy="adaptive")

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            restored.restore(corrupted, strict_policy=True)

        self.assertEqual(restored.snapshot()["search_calls"], 0)


if __name__ == "__main__":
    unittest.main()
