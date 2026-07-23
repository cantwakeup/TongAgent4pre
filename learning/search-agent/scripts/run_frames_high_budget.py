"""Run the frozen FRAMES high-budget exploratory matrix sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evaluation.aggregate import aggregate_experiment
from evaluation.execution import (
    atomic_write_json,
    load_jsonl_dataset,
    resolve_git_sha,
    resolve_system_config,
    run_evaluation,
)
from evaluation.schema import EvalTask, RunResult
from evaluation.systems.transparent_react import TRANSPARENT_REACT_SYSTEM_PROMPT


EXPERIMENT_ID = "frames-high-budget-completion-v1"
SYSTEMS = ("bare_simple_react", "tongagent_standard")
COST_CAP_USD = 6.0
UNCACHED_INPUT_PER_TOKEN = 5.0 / 1_000_000
CACHED_INPUT_PER_TOKEN = 0.5 / 1_000_000
OUTPUT_PER_TOKEN = 30.0 / 1_000_000
WORST_CASE_RUN_COST = 0.625


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _ordered_tasks(
    dataset_path: Path, manifest: Mapping[str, Any]
) -> tuple[str, tuple[EvalTask, ...]]:
    loaded = load_jsonl_dataset(dataset_path, seed=17)
    selected_ids = manifest.get("selected_task_ids")
    selected_indices = manifest.get("selected_source_indices")
    if not isinstance(selected_ids, list) or not isinstance(selected_indices, list):
        raise ValueError("manifest lacks selected task identities")
    by_id = {task.id: task for task in loaded.selected_tasks}
    if set(by_id) != set(selected_ids):
        raise ValueError("manifest/dataset task IDs differ")
    ordered = tuple(by_id[str(task_id)] for task_id in selected_ids)
    observed_indices = [task.metadata.get("source_row_id") for task in ordered]
    if observed_indices != [str(value) for value in selected_indices]:
        raise ValueError("manifest/dataset source indices differ")
    return loaded.digest, ordered


def _job_plan(tasks: tuple[EvalTask, ...]) -> list[dict[str, str]]:
    return [
        {"task_id": task.id, "system_id": system_id}
        for task in tasks
        for system_id in SYSTEMS
    ]


def _known_cost(result: RunResult) -> float | None:
    usage = result.token_usage
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = min(usage.input_tokens, usage.cached_input_tokens or 0)
    uncached = usage.input_tokens - cached
    return (
        uncached * UNCACHED_INPUT_PER_TOKEN
        + cached * CACHED_INPUT_PER_TOKEN
        + usage.output_tokens * OUTPUT_PER_TOKEN
    )


@contextmanager
def _launcher_lock(experiment_directory: Path) -> Iterator[None]:
    experiment_directory.mkdir(parents=True, exist_ok=True)
    path = experiment_directory / ".high-budget-launcher.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"high-budget launcher already active: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "experiment_id": EXPERIMENT_ID}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def _validate_fairness(
    *,
    dataset_digest: str,
    seed: int,
    config_overrides: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    configs = [
        resolve_system_config(
            system_id,
            dataset_digest,
            seed,
            output_root / "preflight" / system_id / "attempt-0001",
            overrides=config_overrides,
        )
        for system_id in SYSTEMS
    ]
    if len({item.fairness_fingerprint for item in configs}) != 1:
        raise ValueError("Bare and Standard high-budget configs are not fair")
    first = configs[0]
    boundary = first.system_options.get("high_budget_finalization")
    if not isinstance(boundary, Mapping):
        raise ValueError("high-budget finalization boundary is missing")
    return {
        "schema_version": 1,
        "policy_prompt_sha256": hashlib.sha256(
            TRANSPARENT_REACT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "bare_standard_prompt_parity": True,
        "fairness_fingerprint": first.fairness_fingerprint,
        "model": first.model.model_dump(mode="json"),
        "tools": first.tools.model_dump(mode="json"),
        "budget": first.budget.model_dump(mode="json"),
        "seed": seed,
        "boundary": dict(boundary),
    }


def run(
    *,
    dataset_path: Path,
    manifest_path: Path,
    config_path: Path,
    output_root: Path,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("manifest_id") != "frames_high_budget_8":
        raise ValueError("unexpected high-budget manifest")
    digest, tasks = _ordered_tasks(dataset_path, manifest)
    config_overrides = _load_json(config_path)
    fairness = _validate_fairness(
        dataset_digest=digest,
        seed=seed,
        config_overrides=config_overrides,
        output_root=output_root,
    )
    plan = _job_plan(tasks)
    if len(plan) != 16:
        raise ValueError("high-budget core schedule must contain exactly 16 jobs")
    if dry_run:
        return {
            "experiment_id": EXPERIMENT_ID,
            "benchmark_status": "exploratory high-budget benchmark",
            "dry_run_jobs": len(plan),
            "job_plan": plan,
            "fairness": fairness,
            "known_cost_cap": COST_CAP_USD,
            "worst_case_run_cost": WORST_CASE_RUN_COST,
        }

    experiment_directory = output_root.resolve() / EXPERIMENT_ID
    if experiment_directory.exists():
        raise RuntimeError(
            f"experiment directory already exists; refusing attempt-0002: "
            f"{experiment_directory}"
        )
    package_root = Path(__file__).resolve().parents[1]
    git_sha = resolve_git_sha(package_root)
    task_by_id = {task.id: task for task in tasks}
    completed: list[dict[str, Any]] = []
    known_cost = 0.0
    conservative_cost = 0.0
    stop_reason: str | None = None

    def config_factory(
        system_id: str,
        dataset_digest: str,
        selected_seed: int,
        attempt_directory: Path,
    ):
        return resolve_system_config(
            system_id,
            dataset_digest,
            selected_seed,
            attempt_directory,
            overrides=config_overrides,
        )

    with _launcher_lock(experiment_directory):
        atomic_write_json(
            experiment_directory / "exploratory_manifest.json",
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "benchmark_status": "exploratory high-budget benchmark",
                "git_sha": git_sha,
                "dataset_digest": digest,
                "manifest": str(manifest_path.resolve()),
                "config": str(config_path.resolve()),
                "job_plan": plan,
                "fairness": fairness,
                "known_cost_cap": COST_CAP_USD,
                "worst_case_run_cost": WORST_CASE_RUN_COST,
            },
        )
        for job_index, job in enumerate(plan, start=1):
            if conservative_cost + WORST_CASE_RUN_COST > COST_CAP_USD:
                stop_reason = "pre_dispatch_worst_case_cost_guard"
                break
            task = task_by_id[job["task_id"]]
            report = run_evaluation(
                tasks=(task,),
                dataset_digest=digest,
                systems=(job["system_id"],),
                seed=seed,
                output_directory=output_root,
                experiment_id=EXPERIMENT_ID,
                resume=True,
                rerun=False,
                dry_run=False,
                config_factory=config_factory,
                git_sha=git_sha,
            )
            outcome = report.outcomes[0]
            result = outcome.result
            if result is None:
                raise RuntimeError("executed high-budget job returned no result")
            cost = _known_cost(result)
            if cost is None:
                conservative_cost += WORST_CASE_RUN_COST
            else:
                known_cost += cost
                conservative_cost += cost
            completed.append(
                {
                    "job_index": job_index,
                    "task_id": result.task_id,
                    "system_id": result.system_id,
                    "completion_status": result.completion_status.value,
                    "known_cost": cost,
                    "known_cost_cumulative": known_cost,
                    "conservative_cost_cumulative": conservative_cost,
                    "artifact_directory": result.artifact_directory,
                }
            )
            state = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "completed_jobs": completed,
                "completed_count": len(completed),
                "planned_count": len(plan),
                "known_cost": known_cost,
                "conservative_cost": conservative_cost,
                "known_cost_cap": COST_CAP_USD,
                "stop_reason": None,
            }
            atomic_write_json(
                experiment_directory / "high_budget_run_state.json",
                state,
                overwrite=True,
            )
            print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
            if known_cost >= COST_CAP_USD:
                stop_reason = "known_cost_cap_reached"
                break

        final_state = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "completed_jobs": completed,
            "completed_count": len(completed),
            "planned_count": len(plan),
            "known_cost": known_cost,
            "conservative_cost": conservative_cost,
            "known_cost_cap": COST_CAP_USD,
            "stop_reason": stop_reason,
            "core_complete": len(completed) == len(plan),
        }
        atomic_write_json(
            experiment_directory / "high_budget_run_state.json",
            final_state,
            overwrite=True,
        )
        if completed:
            aggregate_experiment(experiment_directory)
    return final_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/evaluations"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = run(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        config_path=args.config,
        output_root=args.output,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
