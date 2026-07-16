"""Durable artifact helpers for TongAgent research state and events."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_state import ResearchEvent, ResearchPlan


Clock = Callable[[], datetime]


def append_research_event(
    events: list[ResearchEvent],
    event: str,
    *,
    plan_id: str = "",
    subquestion_id: str = "",
    details: dict[str, Any] | None = None,
    clock: Clock | None = None,
) -> list[ResearchEvent]:
    """Return a new event list with one ordered transition appended.

    Args:
        events: Existing canonical event history from checkpoint state.
        event: Stable event type.
        plan_id: Associated research plan ID when available.
        subquestion_id: Associated subquestion ID when available.
        details: Small JSON-serializable event payload.
        clock: Optional clock used by deterministic tests.

    Returns:
        A copied event list containing the new event.
    """
    sequence = len(events) + 1
    now = (clock or (lambda: datetime.now(UTC)))()
    item: ResearchEvent = {
        "event_id": f"E{sequence:04d}",
        "sequence": sequence,
        "occurred_at": now.isoformat(),
        "event": event,
        "plan_id": plan_id,
        "subquestion_id": subquestion_id,
        "details": dict(details or {}),
    }
    return [*events, item]


def write_plan_snapshot(
    path: Path,
    *,
    thread_id: str,
    plan: ResearchPlan,
    budget: dict[str, Any],
) -> None:
    """Atomically write the latest durable plan snapshot.

    Args:
        path: Destination JSON file.
        thread_id: LangGraph thread that owns the plan.
        plan: Latest plan state.
        budget: Latest serializable budget snapshot.
    """
    payload = {"thread_id": thread_id, "plan": plan, "budget": budget}
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_event_log(path: Path, events: list[ResearchEvent]) -> None:
    """Atomically materialize canonical checkpoint events as JSONL.

    Args:
        path: Destination JSONL file.
        events: Ordered canonical event history.
    """
    content = "".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        for event in events
    )
    _atomic_write(path, content)


def _atomic_write(path: Path, content: str) -> None:
    """Replace a text artifact without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
