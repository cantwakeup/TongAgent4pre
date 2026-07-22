"""End-to-end controlled fault-injection regression."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.fault_injection import run_fault_injection


pytestmark = pytest.mark.usefixtures("socket_disabled")


def test_tongagent_standard_recovers_and_resumes_without_duplicate_fetch(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run_fault_injection(
        manifest_path=(
            root / "evaluation" / "manifests" / "final_fault_injection.json"
        ),
        output_directory=tmp_path / "fault-evaluation",
    )

    assert summary["run_count"] == 16
    bare = summary["systems"]["bare_simple_react"]
    standard = summary["systems"]["tongagent_standard"]
    assert standard["recovery_success_rate"] >= 0.8
    assert standard["trace_completeness_rate"] == 1.0
    assert standard["valid_terminal_rate"] > bare["valid_terminal_rate"]
    resume_rows = [
        row
        for row in summary["rows"]
        if row["system_id"] == "tongagent_standard"
        and row["fault"] == "process_interruption_and_resume"
    ]
    assert len(resume_rows) == 2
    assert all(row["checkpoint_restored"] is True for row in resume_rows)
    assert all(row["duplicate_fetch_calls"] == 0 for row in resume_rows)
