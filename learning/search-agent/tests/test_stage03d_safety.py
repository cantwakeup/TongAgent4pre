"""Safety regressions for Stage 03D report review and run isolation."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from search_agent import (
    _adaptive_audit_errors,
    _build_tool_trace,
    _create_run_output_dir,
    _pending_budget_scope_errors,
    _successful_delegation_before_final_write,
    _successful_final_report_write_position,
)


def _task_call(call_id: str, *, subagent_type: str = "reviewer") -> dict[str, object]:
    """Return one compact report-phase delegation trace event."""
    return {
        "event": "tool_call",
        "name": "task",
        "id": call_id,
        "args": {"subagent_type": subagent_type},
        "phase": "report",
    }


def _task_result(
    call_id: str,
    *,
    status: str = "success",
    content_chars: int = 12,
) -> dict[str, object]:
    """Return the matching compact task result trace event."""
    return {
        "event": "tool_result",
        "name": "task",
        "tool_call_id": call_id,
        "content_chars": content_chars,
        "status": status,
        "phase": "report",
    }


def _write_call(
    file_path: str = "/report.md",
    *,
    call_id: str = "write-report",
    phase: str = "report",
) -> dict[str, object]:
    """Return one final report write trace event."""
    return {
        "event": "tool_call",
        "name": "write_file",
        "id": call_id,
        "args": {"file_path": file_path},
        "phase": phase,
    }


def _write_result(
    call_id: str = "write-report",
    *,
    status: str = "success",
    phase: str = "report",
) -> dict[str, object]:
    """Return the matching report write result."""
    return {
        "event": "tool_result",
        "name": "write_file",
        "tool_call_id": call_id,
        "content_chars": 12,
        "status": status,
        "phase": phase,
    }


class Stage03DSafetyTests(unittest.TestCase):
    """Lock down reviewer ordering, trace fidelity, and artifact isolation."""

    def test_report_reviewer_requires_success_before_final_write(self) -> None:
        successful = [
            _task_call("review-ok"),
            _task_result("review-ok"),
            _write_call(),
            _write_result(),
        ]
        failed = [
            _task_call("review-error"),
            _task_result("review-error", status="error"),
            _write_call(),
            _write_result(),
        ]
        reviewer_after_write = [
            _write_call(),
            _write_result(),
            _task_call("review-too-late"),
            _task_result("review-too-late"),
        ]
        report_then_review_then_notes = [
            _write_call(),
            _write_result(),
            _task_call("review-after-report"),
            _task_result("review-after-report"),
            _write_call("/notes.md", call_id="write-notes"),
            _write_result("write-notes"),
        ]
        failed_final_write = [
            _write_call(call_id="write-draft"),
            _write_result("write-draft"),
            _task_call("review-ok"),
            _task_result("review-ok"),
            _write_call(call_id="write-final"),
            _write_result("write-final", status="error"),
        ]

        self.assertTrue(
            _successful_delegation_before_final_write(successful, "reviewer")
        )
        self.assertFalse(_successful_delegation_before_final_write(failed, "reviewer"))
        self.assertFalse(
            _successful_delegation_before_final_write(
                reviewer_after_write,
                "reviewer",
            )
        )
        self.assertFalse(
            _successful_delegation_before_final_write(
                report_then_review_then_notes,
                "reviewer",
            )
        )
        self.assertFalse(
            _successful_delegation_before_final_write(
                failed_final_write,
                "reviewer",
            )
        )

    def test_final_report_write_must_succeed_in_report_phase(self) -> None:
        successful = [_write_call(), _write_result()]
        research_only = [
            _write_call(phase="research"),
            _write_result(phase="research"),
        ]
        failed_final = [
            _write_call(call_id="draft"),
            _write_result("draft"),
            _write_call(call_id="final"),
            _write_result("final", status="error"),
        ]

        self.assertEqual(_successful_final_report_write_position(successful), 0)
        self.assertIsNone(_successful_final_report_write_position(research_only))
        self.assertIsNone(_successful_final_report_write_position(failed_final))

    def test_tool_trace_preserves_tool_result_status_and_phase(self) -> None:
        messages = [
            HumanMessage(content="research", id="research-step-plan-SQ1"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fetch_url",
                        "args": {"url": "https://example.com"},
                        "id": "fetch-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="network failure",
                tool_call_id="fetch-1",
                name="fetch_url",
                status="error",
            ),
            HumanMessage(content="report", id="report-step-plan"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"subagent_type": "reviewer"},
                        "id": "review-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="review passed",
                tool_call_id="review-1",
                name="task",
                status="success",
            ),
        ]

        results = [
            event
            for event in _build_tool_trace(messages)
            if event["event"] == "tool_result"
        ]

        self.assertEqual(
            [(event["status"], event["phase"]) for event in results],
            [("error", "research"), ("success", "report")],
        )

    def test_run_output_dirs_isolate_threads_runs_and_sanitize_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "artifacts"
            unsafe_thread = "../../outside/中文 thread"

            first_run = _create_run_output_dir(base, unsafe_thread)
            second_run = _create_run_output_dir(base, unsafe_thread)
            colliding_prefix_run = _create_run_output_dir(
                base,
                "outside 中文/thread",
            )

            runs_root = (base / "runs").resolve()
            for run_dir in (first_run, second_run, colliding_prefix_run):
                self.assertTrue(run_dir.is_dir())
                run_dir.resolve().relative_to(runs_root)
                self.assertRegex(
                    run_dir.parent.name,
                    r"^[A-Za-z0-9._-]+-[0-9a-f]{8}$",
                )
                self.assertRegex(
                    run_dir.name,
                    re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8}$"),
                )

            self.assertEqual(first_run.parent, second_run.parent)
            self.assertNotEqual(first_run, second_run)
            self.assertNotEqual(first_run.parent, colliding_prefix_run.parent)
            self.assertNotIn("..", first_run.parent.name)
            self.assertNotIn("/", first_run.parent.name)
            self.assertNotIn("中文", first_run.parent.name)

    def test_adaptive_grant_audit_rejects_same_count_wrong_id(self) -> None:
        control = {
            "strategy": "adaptive",
            "config_fingerprint": "fingerprint",
            "pinned_model": "model",
            "pinned_topology": "single",
            "max_escalations": 2,
            "escalation_count": 1,
            "escalations_by_subquestion": {"SQ1": 1},
            "decision_history": [
                {
                    "decision_id": "decision-1",
                    "sequence": 1,
                    "cycle": 1,
                    "subquestion_id": "SQ1",
                    "action": "expand_budget",
                    "budget_before": {
                        "granted_searches": 1,
                        "granted_fetches": 1,
                        "reserve_searches": 1,
                        "reserve_fetches": 2,
                    },
                    "budget_after": {
                        "granted_searches": 2,
                        "granted_fetches": 2,
                        "reserve_searches": 0,
                        "reserve_fetches": 1,
                    },
                }
            ],
        }
        ledger = {
            "max_searches": 2,
            "max_fetches": 3,
            "subquestion_limits": {"SQ1": {"max_searches": 2, "max_fetches": 2}},
            "granted_searches": 2,
            "granted_fetches": 2,
            "reserve_searches": 0,
            "reserve_fetches": 1,
        }

        errors = _adaptive_audit_errors(
            adaptive_control=control,
            ledger={**ledger, "applied_grant_ids": ["different-decision"]},
            config_fingerprint="fingerprint",
            model_name="model",
            topology="single",
            max_escalations=2,
        )

        self.assertIn(
            "Adaptive budget grant IDs do not match controller expansion decisions",
            errors,
        )
        self.assertEqual(
            _adaptive_audit_errors(
                adaptive_control=control,
                ledger={**ledger, "applied_grant_ids": ["decision-1"]},
                config_fingerprint="fingerprint",
                model_name="model",
                topology="single",
                max_escalations=2,
            ),
            [],
        )

    def test_pending_adaptive_audit_allows_pristine_baseline_without_decisions(
        self,
    ) -> None:
        control = {
            "strategy": "adaptive",
            "config_fingerprint": "fingerprint",
            "pinned_model": "model",
            "pinned_topology": "single",
            "max_escalations": 2,
            "escalation_count": 0,
            "escalations_by_subquestion": {},
            "decision_history": [],
        }
        ledger = {
            "max_searches": 2,
            "max_fetches": 3,
            "subquestion_limits": {"SQ1": {"max_searches": 1, "max_fetches": 1}},
            "granted_searches": 1,
            "granted_fetches": 1,
            "reserve_searches": 1,
            "reserve_fetches": 2,
            "applied_grant_ids": [],
        }

        pending_errors = _adaptive_audit_errors(
            adaptive_control=control,
            ledger=ledger,
            config_fingerprint="fingerprint",
            model_name="model",
            topology="single",
            max_escalations=2,
            require_decision_history=False,
        )
        final_errors = _adaptive_audit_errors(
            adaptive_control=control,
            ledger=ledger,
            config_fingerprint="fingerprint",
            model_name="model",
            topology="single",
            max_escalations=2,
        )

        self.assertEqual(pending_errors, [])
        self.assertIn(
            "Adaptive controller finished without a decision history", final_errors
        )

    def test_pending_budget_scopes_must_match_plan_subquestions(self) -> None:
        plan = {"subquestions": [{"id": "SQ1"}]}
        mismatched = {
            "subquestion_limits": {"SQX": {"max_searches": 1, "max_fetches": 1}},
            "subquestion_usage": {"SQX": {"search_calls": 1, "fetch_calls": 0}},
            "active_subquestion_id": "SQX",
        }
        matching = {
            "subquestion_limits": {"SQ1": {"max_searches": 1, "max_fetches": 1}},
            "subquestion_usage": {"SQ1": {"search_calls": 1, "fetch_calls": 0}},
            "active_subquestion_id": "SQ1",
        }

        errors = _pending_budget_scope_errors(plan, mismatched)

        self.assertIn("Pending budget scopes do not match the research plan", errors)
        self.assertIn(
            "Pending active budget scope does not belong to the research plan", errors
        )
        self.assertEqual(_pending_budget_scope_errors(plan, matching), [])

    def test_adaptive_grant_audit_rejects_reversed_grant_order(self) -> None:
        control = {
            "strategy": "adaptive",
            "config_fingerprint": "fingerprint",
            "pinned_model": "model",
            "pinned_topology": "single",
            "max_escalations": 2,
            "escalation_count": 2,
            "escalations_by_subquestion": {"SQ1": 2},
            "decision_history": [
                {
                    "decision_id": "decision-1",
                    "sequence": 1,
                    "cycle": 1,
                    "subquestion_id": "SQ1",
                    "action": "expand_budget",
                    "budget_before": {
                        "granted_searches": 1,
                        "granted_fetches": 1,
                        "reserve_searches": 1,
                        "reserve_fetches": 2,
                    },
                    "budget_after": {
                        "granted_searches": 2,
                        "granted_fetches": 2,
                        "reserve_searches": 0,
                        "reserve_fetches": 1,
                    },
                },
                {
                    "decision_id": "decision-2",
                    "sequence": 2,
                    "cycle": 2,
                    "subquestion_id": "SQ1",
                    "action": "expand_budget",
                    "budget_before": {
                        "granted_searches": 2,
                        "granted_fetches": 2,
                        "reserve_searches": 0,
                        "reserve_fetches": 1,
                    },
                    "budget_after": {
                        "granted_searches": 2,
                        "granted_fetches": 3,
                        "reserve_searches": 0,
                        "reserve_fetches": 0,
                    },
                },
            ],
        }

        errors = _adaptive_audit_errors(
            adaptive_control=control,
            ledger={
                "max_searches": 2,
                "max_fetches": 3,
                "subquestion_limits": {"SQ1": {"max_searches": 2, "max_fetches": 3}},
                "granted_searches": 2,
                "granted_fetches": 3,
                "reserve_searches": 0,
                "reserve_fetches": 0,
                "applied_grant_ids": ["decision-2", "decision-1"],
            },
            config_fingerprint="fingerprint",
            model_name="model",
            topology="single",
            max_escalations=2,
        )

        self.assertIn(
            "Adaptive budget grant IDs do not match controller expansion decisions",
            errors,
        )

    def test_adaptive_grant_audit_rejects_multi_release_for_one_decision(self) -> None:
        control = {
            "strategy": "adaptive",
            "config_fingerprint": "fingerprint",
            "pinned_model": "model",
            "pinned_topology": "single",
            "max_escalations": 1,
            "escalation_count": 1,
            "escalations_by_subquestion": {"SQ1": 1},
            "decision_history": [
                {
                    "decision_id": "decision-1",
                    "sequence": 1,
                    "cycle": 1,
                    "subquestion_id": "SQ1",
                    "action": "expand_budget",
                    "budget_before": {
                        "granted_searches": 1,
                        "granted_fetches": 1,
                        "reserve_searches": 1,
                        "reserve_fetches": 2,
                    },
                    "budget_after": {
                        "granted_searches": 2,
                        "granted_fetches": 3,
                        "reserve_searches": 0,
                        "reserve_fetches": 0,
                    },
                }
            ],
        }

        errors = _adaptive_audit_errors(
            adaptive_control=control,
            ledger={
                "max_searches": 2,
                "max_fetches": 3,
                "subquestion_limits": {"SQ1": {"max_searches": 2, "max_fetches": 3}},
                "granted_searches": 2,
                "granted_fetches": 3,
                "reserve_searches": 0,
                "reserve_fetches": 0,
                "applied_grant_ids": ["decision-1"],
            },
            config_fingerprint="fingerprint",
            model_name="model",
            topology="single",
            max_escalations=1,
        )

        self.assertIn(
            "Adaptive expansion decision 1 is not a one-step SQ grant",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
