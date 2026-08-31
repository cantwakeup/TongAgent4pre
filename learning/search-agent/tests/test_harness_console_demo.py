"""Validate the committed offline Harness Console presentation bundle."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = REPOSITORY_ROOT / "demo" / "harness-console" / "data" / "demo_bundle.json"


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


class HarnessConsoleDemoTests(unittest.TestCase):
    """Protect the demo's provenance, parity, and frozen result semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    def test_bundle_excludes_runtime_secrets_and_reference_answer(self) -> None:
        rendered = json.dumps(self.bundle)

        self.assertFalse(_contains_key(self.bundle, "reference_answer"))
        self.assertFalse(self.bundle["provenance"]["contains_reference_answer"])
        self.assertFalse(self.bundle["provenance"]["network_required"])
        self.assertNotIn("OPENAI_API_KEY", rendered)
        self.assertNotIn("trycloudflare.com", rendered)

    def test_replay_preserves_transparent_answer_parity(self) -> None:
        run = self.bundle["run"]

        self.assertTrue(run["answer_unchanged"])
        self.assertEqual(run["raw_model_answer"], run["final_answer"])
        self.assertTrue(run["standard_em"])
        self.assertEqual(
            [event["kind"] for event in self.bundle["timeline"]],
            ["system", "runtime", "model", "search", "model", "fetch", "model"],
        )

    def test_replay_choices_cover_success_recovery_and_budget_stop(self) -> None:
        replays = self.bundle["replays"]

        self.assertEqual(
            [replay["label"] for replay in replays],
            ["顺利完成", "改写查询后完成", "预算保护停止", "中断后恢复"],
        )
        self.assertEqual(
            [replay["run"]["completion_status"] for replay in replays],
            ["completed", "completed", "budget_exhausted", "completed"],
        )
        self.assertEqual(
            [len(replay["timeline"]) for replay in replays], [7, 13, 16, 8]
        )
        for replay in replays:
            self.assertEqual(
                replay["run"]["raw_model_answer"],
                replay["run"]["final_answer"],
            )
            self.assertTrue(replay["artifact_directory"])
            for event in replay["timeline"]:
                self.assertTrue(event["module"])
                self.assertTrue(event["function"])

        recovery_titles = {event["title"] for event in replays[1]["timeline"]}
        stopped_titles = {event["title"] for event in replays[2]["timeline"]}
        checkpoint_titles = {event["title"] for event in replays[3]["timeline"]}
        self.assertIn("Agent 改写查询后重新搜索", recovery_titles)
        self.assertIn("重复抓取被安全阻止", recovery_titles)
        self.assertIn("预算保护触发", stopped_titles)
        self.assertIn("任务按预算策略安全停止", stopped_titles)
        self.assertIn("受控实验模拟进程中断", checkpoint_titles)
        self.assertIn("从检查点恢复执行", checkpoint_titles)
        self.assertTrue(replays[3]["run"]["checkpoint"]["restored"])

    def test_fetch_events_disclose_their_artifact_urls(self) -> None:
        for replay in self.bundle["replays"]:
            successful_fetches = [
                event
                for event in replay["timeline"]
                if event["kind"] == "fetch" and event["status"] == "success"
            ]
            self.assertTrue(successful_fetches)
            self.assertTrue(all(event["url"] for event in successful_fetches))

    def test_bundle_retains_complete_frozen_comparisons(self) -> None:
        faults = self.bundle["fault_lab"]
        benchmark = self.bundle["benchmark"]

        self.assertEqual(len(faults["scenarios"]), 4)
        self.assertEqual(
            faults["systems"]["bare_simple_react"]["recovery_success_rate"],
            0.5,
        )
        self.assertEqual(
            faults["systems"]["tongagent_standard"]["recovery_success_rate"],
            1.0,
        )
        self.assertEqual(len(benchmark["systems"]), 3)
        self.assertFalse(benchmark["claims"]["accuracy_superiority"])
        self.assertTrue(benchmark["claims"]["operational_reliability_superiority"])


if __name__ == "__main__":
    unittest.main()
