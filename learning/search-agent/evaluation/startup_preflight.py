"""Construct live TongAgent graphs without invoking models or network tools."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .execution import load_jsonl_dataset, resolve_system_config
from .offline import FixtureChatModel
from .systems.tongagent import TongAgentRunner
from .tracing import sanitize_trace_value


DEFAULT_TASK_IDS = (
    "frames-test-0664",
    "frames-test-0191",
    "frames-test-0123",
    "frames-test-0016",
    "frames-test-0718",
)


def preflight_tongagent_tasks(
    *,
    dataset: str | Path,
    config_overrides: Mapping[str, Any],
    task_ids: Sequence[str] = DEFAULT_TASK_IDS,
    seed: int = 17,
) -> dict[str, Any]:
    """Build one live runtime/graph per requested task without ``invoke``.

    The injected fixture model is construction-only: it prevents a real
    provider request while preserving the production ``TongAgentRunner`` →
    ``prepare_runtime`` → ``build_agent`` route.  The task question is the
    only task field that reaches graph construction; no reference answer or
    benchmark URL is used as a model or retrieval input.
    """

    loaded = load_jsonl_dataset(dataset, seed=seed)
    tasks_by_id = {task.id: task for task in loaded.selected_tasks}
    requested_ids = tuple(task_ids)
    missing = [task_id for task_id in requested_ids if task_id not in tasks_by_id]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"requested startup-preflight task ids are missing: {names}")

    outcomes: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="tongagent-startup-preflight-") as directory:
        root = Path(directory)
        for task_id in requested_ids:
            task = tasks_by_id[task_id]
            config = resolve_system_config(
                "tongagent",
                loaded.digest,
                seed,
                root / task_id,
                overrides=config_overrides,
            )
            model = FixtureChatModel.from_task(task, system_id="tongagent")
            try:
                outcome = TongAgentRunner(model=model).preflight(task, config)
            except Exception as exc:
                outcomes.append(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "message": sanitize_trace_value(str(exc)),
                    }
                )
                continue
            if model.call_history:
                outcomes.append(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "exception_type": "UnexpectedModelInvocation",
                        "message": "startup preflight invoked its fixture model",
                    }
                )
                continue
            outcomes.append({"task_id": task_id, "status": "passed", **outcome})

    return {
        "dataset_digest": loaded.digest,
        "seed": seed,
        "task_ids": list(requested_ids),
        "outcomes": outcomes,
        "passed": sum(item["status"] == "passed" for item in outcomes),
        "failed": sum(item["status"] != "passed" for item in outcomes),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.startup_preflight",
        description="Build live TongAgent graphs without model or network invocation",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--task-ids", nargs="+", default=list(DEFAULT_TASK_IDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a JSON object")
    result = preflight_tongagent_tasks(
        dataset=args.dataset,
        config_overrides=payload,
        task_ids=args.task_ids,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
