"""Shared structural interface implemented by every evaluated system."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import ResolvedConfig
from ..schema import EvalTask, RunResult


@runtime_checkable
class SystemRunner(Protocol):
    """Run one task under a fully resolved common configuration."""

    def run(self, task: EvalTask, resolved_config: ResolvedConfig) -> RunResult:
        """Execute one isolated attempt and return its canonical result."""
        ...
