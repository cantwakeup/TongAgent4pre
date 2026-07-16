"""Tests for Stage 03A plan and event artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from research_graph import create_research_plan
from telemetry import append_research_event, write_event_log, write_plan_snapshot


class ResearchArtifactTests(unittest.TestCase):
    """Verify deterministic event ordering and complete plan snapshots."""

    def test_plan_and_event_artifacts_are_valid_and_replace_atomically(self) -> None:
        plan = create_research_plan(
            "测试中文研究问题",
            ["收集主要事实"],
            plan_id_factory=lambda: "plan-artifact",
        )
        fixed_time = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
        events = append_research_event(
            [],
            "plan_created",
            plan_id=plan["plan_id"],
            clock=lambda: fixed_time,
        )
        events = append_research_event(
            events,
            "subquestion_selected",
            plan_id=plan["plan_id"],
            subquestion_id="SQ1",
            clock=lambda: fixed_time,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "plan.json"
            events_path = root / "events.jsonl"
            write_plan_snapshot(
                plan_path,
                thread_id="thread-artifact",
                plan=plan,
                budget={"search_calls": 1},
            )
            write_event_log(events_path, events)

            snapshot = json.loads(plan_path.read_text())
            rows = [json.loads(line) for line in events_path.read_text().splitlines()]

        self.assertEqual(snapshot["thread_id"], "thread-artifact")
        self.assertEqual(snapshot["plan"]["question"], "测试中文研究问题")
        self.assertEqual(snapshot["budget"]["search_calls"], 1)
        self.assertEqual([row["sequence"] for row in rows], [1, 2])
        self.assertEqual([row["event_id"] for row in rows], ["E0001", "E0002"])


if __name__ == "__main__":
    unittest.main()
