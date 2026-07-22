"""Offline preflight tests for the one-shot controlled live pilot config."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.execution import resolve_system_config
from evaluation.systems.tongagent import _resolve_options


def test_controlled_live_config_has_shared_hard_boundaries(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    config_path = project / "evaluation/configs/live_pilot.example.json"
    overrides = json.loads(config_path.read_text(encoding="utf-8"))
    configs = [
        resolve_system_config(
            system_id,
            "sha256:" + "1" * 64,
            17,
            tmp_path / system_id / "pilot-python-313" / "attempt-0001",
            overrides=overrides,
        )
        for system_id in (
            "simple_react",
            "vanilla_deepagents",
            "tongagent",
        )
    ]

    assert {config.fairness_fingerprint for config in configs} == {
        configs[0].fairness_fingerprint
    }
    assert {(config.model.provider, config.model.name) for config in configs} == {
        ("openai", "gpt-5.4-nano")
    }
    assert all(config.model.max_output_tokens == 5_000 for config in configs)
    assert all(config.model.credential_env == ["OPENAI_API_KEY"] for config in configs)
    assert all(
        config.tools.search_backend == "tongagent-web-search" for config in configs
    )
    assert all(
        config.tools.fetch_backend == "tongagent-fetch-url" for config in configs
    )
    assert all(config.judge is None for config in configs)
    assert all(config.budget.max_search_calls == 4 for config in configs)
    assert all(config.budget.max_fetch_calls == 6 for config in configs)
    assert all(config.budget.max_total_tool_calls == 12 for config in configs)
    assert all(config.budget.max_model_calls == 16 for config in configs)
    assert all(config.budget.max_total_tokens == 100_000 for config in configs)
    assert all(config.budget.wall_time_seconds <= 600 for config in configs)
    _, policy = _resolve_options(configs[2])
    assert policy.max_output_tokens == configs[2].model.max_output_tokens


def test_frames_pilot_config_caps_fifteen_run_token_cost(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    config_path = project / "evaluation/configs/frames_pilot_seed17.example.json"
    overrides = json.loads(config_path.read_text(encoding="utf-8"))
    configs = [
        resolve_system_config(
            system_id,
            "sha256:" + "2" * 64,
            17,
            tmp_path / system_id / "frames-test-0001" / "attempt-0001",
            overrides=overrides,
        )
        for system_id in (
            "simple_react",
            "vanilla_deepagents",
            "tongagent",
        )
    ]

    assert {config.fairness_fingerprint for config in configs} == {
        configs[0].fairness_fingerprint
    }
    assert all(config.budget.max_total_tokens == 50_000 for config in configs)
    assert all(config.budget.max_total_tool_calls == 12 for config in configs)
    # FRAMES uses a fair evaluation-only provider cap; Stage 03D's underlying
    # medium effort policy remains unchanged.
    assert all(config.model.max_output_tokens == 1_024 for config in configs)
    options, policy = _resolve_options(configs[2])
    assert options["effort"] == "medium"
    assert options["mode"] == "single"
    assert options["strategy"] == "adaptive"
    assert options["max_escalations"] == 2
    assert policy.max_output_tokens == 5_000
