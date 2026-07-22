"""Command-line entry point for unified B1/B2/B3 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .aggregate import aggregate_experiment
from .execution import SYSTEM_IDS, run_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.cli",
        description=(
            "Run process-isolated, fingerprinted evaluation attempts or "
            "summarize an existing experiment."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run selected systems on JSONL tasks")
    run.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEM_IDS,
        default=list(SYSTEM_IDS),
        help="registered system ids (default: all)",
    )
    run.add_argument("--dataset", type=Path, required=True, help="UTF-8 task JSONL")
    run.add_argument("--limit", type=_nonnegative_int)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluations"),
        help="evaluation output root",
    )
    run.add_argument(
        "--experiment",
        default=None,
        help="safe experiment id (default: UTC timestamp)",
    )
    resume = run.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="skip matching terminal results (default)",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="create a new attempt for every selected pair",
    )
    run.set_defaults(resume=True)
    run.add_argument(
        "--rerun",
        action="store_true",
        help="always create a new attempt; never skip a terminal result",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the schedule without creating attempts",
    )
    run.add_argument(
        "--config",
        type=Path,
        help=(
            "optional JSON overrides for backend/model/tools/budget and "
            "per-system system_options; defaults are deterministic fixtures"
        ),
    )

    summarize = subcommands.add_parser(
        "summarize",
        help="regenerate strict JSON, CSV, and Markdown summaries",
    )
    summarize.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluations"),
        help="evaluation output root",
    )
    summarize.add_argument("--experiment", required=True, help="experiment id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        return _summarize_command(args)
    except Exception as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


def _run_command(args: argparse.Namespace) -> int:
    experiment_id = args.experiment or datetime.now(UTC).strftime("eval-%Y%m%dT%H%M%SZ")
    overrides = _load_config(args.config) if args.config is not None else None
    report = run_dataset(
        args.dataset,
        systems=args.systems,
        limit=args.limit,
        seed=args.seed,
        output_directory=args.output,
        experiment_id=experiment_id,
        resume=args.resume,
        rerun=args.rerun,
        dry_run=args.dry_run,
        config_overrides=overrides,
    )
    response: dict[str, Any] = {
        "experiment_id": report.experiment_id,
        "experiment_directory": str(report.experiment_directory),
        "dataset_digest": report.dataset_digest,
        "executed": report.executed,
        "skipped": report.skipped,
        "dry_run": report.dry_run,
        "jobs": [
            {
                "system_id": outcome.system_id,
                "task_id": outcome.task_id,
                "action": outcome.action,
                "attempt_directory": str(outcome.attempt_directory),
                "completion_status": (
                    outcome.result.completion_status.value
                    if outcome.result is not None
                    else None
                ),
                "failure_type": (
                    outcome.result.failure_type.value
                    if outcome.result is not None
                    and outcome.result.failure_type is not None
                    else None
                ),
            }
            for outcome in report.outcomes
        ],
    }
    if not args.dry_run:
        response["summary"] = aggregate_experiment(report.experiment_directory)
    print(
        json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _summarize_command(args: argparse.Namespace) -> int:
    experiment_directory = args.output.expanduser().resolve() / args.experiment
    payload = aggregate_experiment(experiment_directory)
    print(
        json.dumps(
            {
                "experiment_id": payload["experiment_id"],
                "selected_result_count": payload["selected_result_count"],
                "incomplete_attempt_count": payload["incomplete_attempt_count"],
                "fairness_fingerprint": payload["fairness_fingerprint"],
                "summary_json": str(experiment_directory / "summary.json"),
                "summary_csv": str(experiment_directory / "summary.csv"),
                "summary_markdown": str(experiment_directory / "summary.md"),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_config(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a JSON object")
    return payload


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
