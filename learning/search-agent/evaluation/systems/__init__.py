"""System adapters exposed through the unified evaluation protocol."""

from __future__ import annotations

from .base import SystemRunner
from .common import EvaluationMiddleware
from .simple_react import SimpleReactRunner
from .vanilla_deepagents import VanillaDeepAgentsRunner


def get_runner(system_id: str) -> SystemRunner:
    """Construct one registered runner without hidden runtime dependencies."""

    if system_id == SimpleReactRunner.system_id:
        return SimpleReactRunner()
    if system_id == VanillaDeepAgentsRunner.system_id:
        return VanillaDeepAgentsRunner()
    if system_id == "tongagent":
        # B3 is kept lazy because it imports the full TongAgent graph.
        from .tongagent import TongAgentRunner  # noqa: PLC0415

        return TongAgentRunner()
    msg = f"Unknown evaluation system: {system_id}"
    raise ValueError(msg)


__all__ = [
    "EvaluationMiddleware",
    "SimpleReactRunner",
    "SystemRunner",
    "VanillaDeepAgentsRunner",
    "get_runner",
]
