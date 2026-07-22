"""Live-style startup construction regressions for the TongAgent adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.startup_preflight import DEFAULT_TASK_IDS, preflight_tongagent_tasks


pytestmark = pytest.mark.usefixtures("socket_disabled")


def test_live_tongagent_startup_preflight_builds_all_frames_graphs() -> None:
    """Exercise runner → runtime → phase decider → real graph construction.

    This is intentionally not a unit call to the phase-decider helper and it
    does not mock ``build_agent``.  A fixture model prevents provider access;
    graph construction must nevertheless reach the same live E1 builder path
    before the first agent invoke.
    """

    project = Path(__file__).resolve().parents[1]
    overrides = json.loads(
        (project / "evaluation/configs/live_pilot.example.json").read_text(
            encoding="utf-8"
        )
    )
    result = preflight_tongagent_tasks(
        dataset=project / "evaluation/datasets/frames_pilot_seed17.jsonl",
        config_overrides=overrides,
        task_ids=DEFAULT_TASK_IDS,
        seed=17,
    )

    assert result["failed"] == 0
    assert result["passed"] == len(DEFAULT_TASK_IDS)
    for outcome in result["outcomes"]:
        assert outcome["status"] == "passed"
        assert outcome["agent_constructed"] is True
        assert outcome["phase_decider_configured"] is True
        assert outcome["model_invocations"] == 0
        # The live runtime reserves final synthesis while it is built, but the
        # injected model must not be invoked before the graph starts.
        assert outcome["reserved_model_calls"] >= 1
