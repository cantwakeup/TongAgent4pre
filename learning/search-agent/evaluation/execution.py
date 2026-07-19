"""Process-isolated execution and crash-safe artifact management.

The orchestration layer deliberately does not import any system adapter.  Every
task/system pair is executed by a fresh ``sys.executable -m evaluation.worker``
process, so adapter imports and TongAgent's global harness configuration cannot
leak between comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from .config import ResolvedConfig
from .judging import ensure_judge_available
from .schema import (
    CompletionStatus,
    EvalTask,
    FailureDetail,
    FailureType,
    RunResult,
    json_ready,
    normalized_exact_match,
    parse_eval_task_jsonl_line,
)
from .tracing import sanitize_trace_value


SYSTEM_IDS = ("simple_react", "vanilla_deepagents", "tongagent")
DEFAULT_FIXTURE_DIRECTORY = "evaluation/fixtures"
DEFAULT_FIXTURE_REVISION = "offline-fixtures-v1"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ATTEMPT_NAME = re.compile(r"^attempt-(\d{4,})$")
_TERMINAL_STATUSES = frozenset(CompletionStatus)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class EvaluationExecutionError(RuntimeError):
    """Base class for orchestration failures."""


class EvaluationStateError(EvaluationExecutionError):
    """Raised when persisted state is corrupt or belongs to another config."""


class FairnessMismatchError(EvaluationExecutionError):
    """Raised before execution when systems do not share a fairness fingerprint."""


class WorkerJob(BaseModel):
    """Strict, self-contained instruction consumed by one worker process."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    run_id: str
    git_sha: str
    task_file: Literal["task.json"] = "task.json"
    config_file: Literal["config.json"] = "config.json"
    result_file: Literal["result.json"] = "result.json"


@dataclass(frozen=True)
class LoadedDataset:
    """Validated task selection plus the identity of the complete JSONL file."""

    path: Path
    digest: str
    total_tasks: int
    selected_tasks: tuple[EvalTask, ...]
    seed: int
    limit: int | None


@dataclass(frozen=True)
class JobOutcome:
    """Outcome of scheduling one task/system pair."""

    system_id: str
    task_id: str
    action: Literal["executed", "skipped", "dry_run"]
    attempt_directory: Path
    result: RunResult | None


@dataclass(frozen=True)
class ExecutionReport:
    """In-memory report returned after a dataset execution pass."""

    experiment_id: str
    experiment_directory: Path
    dataset_digest: str
    outcomes: tuple[JobOutcome, ...]

    @property
    def executed(self) -> int:
        return sum(item.action == "executed" for item in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(item.action == "skipped" for item in self.outcomes)

    @property
    def dry_run(self) -> int:
        return sum(item.action == "dry_run" for item in self.outcomes)


ConfigFactory = Callable[[str, str, int, Path], ResolvedConfig]


def load_jsonl_dataset(
    path: str | Path,
    *,
    seed: int = 0,
    limit: int | None = None,
) -> LoadedDataset:
    """Load, validate, fingerprint, deterministically shuffle, and limit JSONL.

    The digest covers the exact bytes of the complete dataset, not merely the
    selected subset.  Selection uses a local PRNG and therefore never mutates
    global random state.
    """

    dataset_path = Path(path).expanduser().resolve()
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or null")
    raw = dataset_path.read_bytes()
    digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"dataset must be UTF-8 JSONL: {dataset_path}"
        raise ValueError(msg) from exc

    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        task = parse_eval_task_jsonl_line(line, line_number=line_number)
        if task.id in seen_ids:
            msg = f"duplicate evaluation task id {task.id!r} at line {line_number}"
            raise ValueError(msg)
        seen_ids.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"dataset contains no tasks: {dataset_path}")

    selected = list(tasks)
    random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    return LoadedDataset(
        path=dataset_path,
        digest=digest,
        total_tasks=len(tasks),
        selected_tasks=tuple(selected),
        seed=seed,
        limit=limit,
    )


def resolve_system_config(
    system_id: str,
    dataset_digest: str,
    seed: int,
    artifact_directory: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    fixture_directory: str = DEFAULT_FIXTURE_DIRECTORY,
) -> ResolvedConfig:
    """Build a strict, secret-free config from safe offline defaults.

    ``fixture_directory`` is a portable repository-relative logical path.  An
    absolute or parent-traversing path is rejected so machine-specific paths do
    not enter configuration fingerprints.
    """

    _validate_component(system_id, kind="system id")
    fixture_directory = _validate_fixture_directory(fixture_directory)
    fixture_revision = _discover_fixture_revision(fixture_directory)
    defaults: dict[str, Any] = {
        "backend_kind": "fixture",
        "fixture_revision": fixture_revision,
        "model": {
            "provider": "fixture",
            "name": "fixture-chat-v1",
            "temperature": 0.0,
            # Medium-effort TongAgent requires 5k; the same cap is exposed to
            # B1/B2 so B3 never receives a hidden output-context advantage.
            "max_output_tokens": 5_000,
            "credential_env": [],
            "parameters": {},
        },
        "tools": {
            "search_backend": "fixture-search",
            "fetch_backend": "fixture-fetch",
            "parameters": {},
        },
        "budget": {
            "max_search_calls": 4,
            "max_fetch_calls": 6,
            "max_total_tool_calls": 30,
            "max_model_calls": 30,
            # Fixture smoke uses the same hard ceiling for all systems.  The
            # medium B3 control loop can legitimately exceed 50k prompt tokens.
            "max_total_tokens": 100_000,
            "wall_time_seconds": 60.0,
            "max_results_per_search": 5,
            "max_page_chars": 12_000,
            "recursion_limit": 125,
        },
        "judge": None,
        "system_options": _default_system_options(system_id, fixture_directory),
    }
    override_payload = dict(overrides or {})
    forbidden = {
        "artifact_directory",
        "config_fingerprint",
        "dataset_digest",
        "fairness_fingerprint",
        "schema_version",
        "seed",
        "system_id",
    }
    conflicts = forbidden.intersection(override_payload)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(
            f"orchestration-owned config fields cannot be overridden: {names}"
        )

    fixture_revision_overridden = "fixture_revision" in override_payload
    per_system_options = override_payload.pop("system_options", None)
    merged = _deep_merge(defaults, override_payload)
    if per_system_options is not None:
        if not isinstance(per_system_options, Mapping):
            raise ValueError("system_options override must be a JSON object")
        option_keys = set(per_system_options)
        if option_keys and option_keys.issubset(set(SYSTEM_IDS)):
            selected_options = per_system_options.get(system_id, {})
            if not isinstance(selected_options, Mapping):
                raise ValueError(f"system_options.{system_id} must be a JSON object")
        else:
            selected_options = per_system_options
        merged["system_options"] = _deep_merge(
            defaults["system_options"],
            dict(selected_options),
        )

    options = merged.get("system_options")
    if isinstance(options, Mapping) and "fixture_dir" in options:
        options = dict(options)
        options["fixture_dir"] = _validate_fixture_directory(
            str(options["fixture_dir"])
        )
        merged["system_options"] = options
        if merged.get("backend_kind") == "fixture" and not fixture_revision_overridden:
            merged["fixture_revision"] = _discover_fixture_revision(
                options["fixture_dir"]
            )
    _reject_secret_fields(merged)

    payload = {
        **merged,
        "system_id": system_id,
        "dataset_digest": dataset_digest,
        "seed": seed,
        "artifact_directory": str(artifact_directory.resolve()),
    }
    return ResolvedConfig.model_validate(payload)


def run_dataset(
    dataset: str | Path,
    *,
    systems: Sequence[str] = SYSTEM_IDS,
    limit: int | None = None,
    seed: int = 0,
    output_directory: str | Path = "output/evaluations",
    experiment_id: str,
    resume: bool = True,
    rerun: bool = False,
    dry_run: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
    config_factory: ConfigFactory | None = None,
    worker_module: str = "evaluation.worker",
    worker_cwd: str | Path | None = None,
    subprocess_timeout_seconds: float | None = None,
    git_sha: str | None = None,
) -> ExecutionReport:
    """Load a JSONL dataset and execute every selected task/system pair."""

    loaded = load_jsonl_dataset(dataset, seed=seed, limit=limit)

    if config_factory is None:

        def factory(
            system_id: str,
            digest: str,
            selected_seed: int,
            attempt_directory: Path,
        ) -> ResolvedConfig:
            return resolve_system_config(
                system_id,
                digest,
                selected_seed,
                attempt_directory,
                overrides=config_overrides,
            )

        config_factory = factory

    return run_evaluation(
        tasks=loaded.selected_tasks,
        dataset_digest=loaded.digest,
        systems=systems,
        seed=seed,
        output_directory=output_directory,
        experiment_id=experiment_id,
        resume=resume,
        rerun=rerun,
        dry_run=dry_run,
        config_factory=config_factory,
        worker_module=worker_module,
        worker_cwd=worker_cwd,
        subprocess_timeout_seconds=subprocess_timeout_seconds,
        git_sha=git_sha,
    )


def run_evaluation(
    *,
    tasks: Sequence[EvalTask],
    dataset_digest: str,
    systems: Sequence[str],
    seed: int,
    output_directory: str | Path,
    experiment_id: str,
    resume: bool = True,
    rerun: bool = False,
    dry_run: bool = False,
    config_factory: ConfigFactory,
    worker_module: str = "evaluation.worker",
    worker_cwd: str | Path | None = None,
    subprocess_timeout_seconds: float | None = None,
    git_sha: str | None = None,
) -> ExecutionReport:
    """Execute one process-isolated attempt for every task/system pair."""

    _validate_component(experiment_id, kind="experiment id")
    if not systems:
        raise ValueError("at least one system is required")
    normalized_systems = tuple(dict.fromkeys(systems))
    for system_id in normalized_systems:
        _validate_component(system_id, kind="system id")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("tasks must have unique ids")
    if subprocess_timeout_seconds is not None and subprocess_timeout_seconds <= 0:
        raise ValueError("subprocess timeout must be positive")

    output_root = Path(output_directory).expanduser().resolve()
    experiment_directory = output_root / experiment_id
    worker_directory = (
        Path(worker_cwd).expanduser().resolve() if worker_cwd else _PACKAGE_ROOT
    )
    current_git_sha = git_sha or resolve_git_sha(_PACKAGE_ROOT)

    # Resolve all prospective configs first.  This prevents any filesystem
    # mutation or model/tool call when a comparison is unfair.
    prospective: dict[tuple[str, str], tuple[Path, ResolvedConfig]] = {}
    fairness: set[str] = set()
    for system_id in normalized_systems:
        for task in tasks:
            task_root = experiment_directory / system_id / task.id
            attempt = task_root / _prospective_attempt_name(task_root)
            config = config_factory(system_id, dataset_digest, seed, attempt)
            _validate_config_identity(
                config,
                system_id=system_id,
                dataset_digest=dataset_digest,
                seed=seed,
                attempt_directory=attempt,
            )
            ensure_judge_available(config)
            prospective[(system_id, task.id)] = (attempt, config)
            fairness.add(config.fairness_fingerprint)
    if len(fairness) > 1:
        raise FairnessMismatchError(
            "systems/tasks resolved to different fairness fingerprints; "
            "shared model, tools, dataset, backend, seed, and budgets must match"
        )

    outcomes: list[JobOutcome] = []
    for system_id in normalized_systems:
        for task in tasks:
            task_root = experiment_directory / system_id / task.id
            prospective_attempt, prospective_config = prospective[(system_id, task.id)]
            terminal = _inspect_existing_attempts(
                task_root,
                task=task,
                expected_config=prospective_config,
                expected_git_sha=current_git_sha,
            )
            if resume and not rerun and terminal is not None:
                outcomes.append(
                    JobOutcome(
                        system_id=system_id,
                        task_id=task.id,
                        action="skipped",
                        attempt_directory=Path(terminal.artifact_directory),
                        result=terminal,
                    )
                )
                continue

            if dry_run:
                outcomes.append(
                    JobOutcome(
                        system_id=system_id,
                        task_id=task.id,
                        action="dry_run",
                        attempt_directory=prospective_attempt,
                        result=None,
                    )
                )
                continue

            attempt_directory = _create_next_attempt(task_root)
            config = config_factory(
                system_id,
                dataset_digest,
                seed,
                attempt_directory,
            )
            _validate_config_identity(
                config,
                system_id=system_id,
                dataset_digest=dataset_digest,
                seed=seed,
                attempt_directory=attempt_directory,
            )
            if fairness and config.fairness_fingerprint not in fairness:
                raise FairnessMismatchError(
                    "config changed after fairness preflight; refusing execution"
                )
            job = WorkerJob(
                run_id=f"run-{uuid.uuid4().hex}",
                git_sha=current_git_sha,
            )
            _write_attempt_inputs(attempt_directory, task, config, job)
            result = _run_worker_subprocess(
                attempt_directory,
                task=task,
                config=config,
                job=job,
                worker_module=worker_module,
                worker_cwd=worker_directory,
                timeout_seconds=subprocess_timeout_seconds,
            )
            outcomes.append(
                JobOutcome(
                    system_id=system_id,
                    task_id=task.id,
                    action="executed",
                    attempt_directory=attempt_directory,
                    result=result,
                )
            )

    return ExecutionReport(
        experiment_id=experiment_id,
        experiment_directory=experiment_directory,
        dataset_digest=dataset_digest,
        outcomes=tuple(outcomes),
    )


def sanitized_subprocess_env(
    *,
    seed: int,
    backend_kind: str,
    credential_env: Sequence[str] = (),
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an inherited environment with implicit API/model/tracing state removed."""

    if backend_kind not in {"fixture", "live"}:
        raise ValueError(f"unsupported evaluation backend_kind: {backend_kind!r}")
    if backend_kind == "fixture" and credential_env:
        raise ValueError("fixture subprocesses cannot receive credential_env")
    forbidden_allowlist = [
        name for name in credential_env if _is_implicit_environment_key(name.upper())
    ]
    if forbidden_allowlist:
        names = ", ".join(sorted(forbidden_allowlist))
        raise ValueError(
            f"credential_env cannot restore implicit runtime state: {names}"
        )
    env = dict(os.environ if source is None else source)
    allowlisted_values = {
        name: env[name]
        for name in credential_env
        if backend_kind == "live" and name in env
    }
    for key in tuple(env):
        upper = key.upper()
        if _is_sensitive_environment_key(upper) or re.search(
            r"(?i)\bbearer\s+\S+",
            env[key],
        ):
            env.pop(key, None)
    if backend_kind == "live":
        env.update(allowlisted_values)
    env["PYTHONHASHSEED"] = str(seed)
    env["TONGAGENT_EVALUATION_BACKEND"] = backend_kind
    if backend_kind == "fixture":
        env["TONGAGENT_OFFLINE"] = "1"
    return env


def persist_terminal_result(
    attempt_directory: Path,
    result: RunResult,
) -> None:
    """Write standardized artifacts and ``result.json`` as the final marker."""

    attempt = attempt_directory.resolve()
    if Path(result.artifact_directory).resolve() != attempt:
        raise EvaluationStateError(
            "result artifact_directory does not match the active attempt"
        )
    native = attempt / "native"
    native.mkdir(exist_ok=True)
    answer = result.final_answer or ""
    if answer and not answer.endswith("\n"):
        answer += "\n"
    # Adapters may emit richer native/intermediate artifacts before returning.
    # Canonical top-level views are refreshed atomically; only result.json is
    # immutable and unique.
    atomic_write_text(attempt / "answer.md", answer, overwrite=True)
    atomic_write_json(
        attempt / "trace.json",
        _canonical_trace_payload(result, native),
        overwrite=True,
    )
    atomic_write_json(
        attempt / "metrics.json",
        _canonical_metrics_payload(result),
        overwrite=True,
    )
    atomic_write_json(
        attempt / "failure.json",
        _canonical_failure_payload(result),
        overwrite=True,
    )
    # This is the only completion marker and must be persisted last.
    atomic_write_json(attempt / "result.json", json_ready(result))


def build_failure_result(
    *,
    task: EvalTask,
    config: ResolvedConfig,
    run_id: str,
    git_sha: str,
    started_at: datetime,
    started_monotonic: float,
    completion_status: CompletionStatus,
    failure_type: FailureType,
    message: str,
    stage: str,
    details: Mapping[str, JsonValue] | None = None,
) -> RunResult:
    """Construct a canonical terminal result for worker/process failures."""

    finished_at = datetime.now(UTC)
    safe_message = sanitize_trace_value(message)
    if not isinstance(safe_message, str) or not safe_message:
        safe_message = "evaluation worker failed"
    safe_details = sanitize_trace_value(dict(details or {}))
    if not isinstance(safe_details, dict):
        safe_details = {}
    failure = FailureDetail(
        failure_type=failure_type,
        message=safe_message,
        stage=stage,
        retryable=False,
        details=safe_details,
    )
    return RunResult(
        run_id=run_id,
        task_id=task.id,
        system_id=config.system_id,
        git_sha=git_sha,
        resolved_config=config,
        config_fingerprint=config.config_fingerprint,
        fairness_fingerprint=config.fairness_fingerprint,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=max(0.0, monotonic() - started_monotonic),
        final_answer=None,
        citations=[],
        tool_calls=[],
        # The parent cannot observe how far a timed-out or crashed worker ran.
        # Unknown counters must remain null rather than masquerading as known
        # zero usage.
        search_calls=None,
        fetch_calls=None,
        relevant_searches=None,
        evidence_count=None,
        structural_subquestion_coverage=None,
        token_usage=None,
        estimated_cost=None,
        completion_status=completion_status,
        failure_type=failure_type,
        failure=failure,
        artifact_directory=str(Path(config.artifact_directory).resolve()),
        fixture_smoke=config.backend_kind == "fixture",
        normalized_exact_match=normalized_exact_match(
            None,
            task.reference_answer,
        ),
        judge_score=None,
    )


def validate_persisted_result(
    result_path: Path,
    *,
    task: EvalTask | None = None,
    system_id: str | None = None,
    expected_config: ResolvedConfig | None = None,
    expected_git_sha: str | None = None,
) -> RunResult:
    """Strictly validate one completion marker and its directory identity."""

    if result_path.is_symlink() or not result_path.is_file():
        raise EvaluationStateError(
            f"result marker is not a regular file: {result_path}"
        )
    try:
        result = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvaluationStateError(
            f"invalid result marker {result_path}: {exc}"
        ) from exc
    attempt = result_path.parent.resolve()
    if Path(result.artifact_directory).resolve() != attempt:
        raise EvaluationStateError(
            f"result artifact directory mismatch in {result_path}"
        )
    if Path(result.resolved_config.artifact_directory).resolve() != attempt:
        raise EvaluationStateError(
            f"resolved config artifact directory mismatch in {result_path}"
        )
    if task is not None and result.task_id != task.id:
        raise EvaluationStateError(f"task identity mismatch in {result_path}")
    if system_id is not None and result.system_id != system_id:
        raise EvaluationStateError(f"system identity mismatch in {result_path}")
    if expected_git_sha is not None and result.git_sha != expected_git_sha:
        raise EvaluationStateError(f"git SHA mismatch in {result_path}")
    if expected_config is not None:
        if result.config_fingerprint != expected_config.config_fingerprint:
            raise EvaluationStateError(
                f"config fingerprint mismatch in {result_path}; "
                "use a new experiment id for a changed configuration"
            )
        if result.fairness_fingerprint != expected_config.fairness_fingerprint:
            raise EvaluationStateError(
                f"fairness fingerprint mismatch in {result_path}"
            )
    if result.completion_status not in _TERMINAL_STATUSES:
        raise EvaluationStateError(f"non-terminal result marker: {result_path}")
    _validate_companion_artifacts(result_path, result)
    return result


def _canonical_trace_payload(result: RunResult, native: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": result.run_id,
        "task_id": result.task_id,
        "system_id": result.system_id,
        "trace_scope": "canonical_tool_calls",
        "native_trace_jsonl": (
            "native/trace.jsonl" if (native / "trace.jsonl").is_file() else None
        ),
        "tool_calls": [
            item.model_dump(mode="json", exclude_none=False)
            for item in result.tool_calls
        ],
    }


def _canonical_metrics_payload(result: RunResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": result.run_id,
        "completion_status": result.completion_status.value,
        "wall_time_seconds": result.wall_time_seconds,
        "search_calls": result.search_calls,
        "fetch_calls": result.fetch_calls,
        "relevant_searches": result.relevant_searches,
        "evidence_count": result.evidence_count,
        "structural_subquestion_coverage": result.structural_subquestion_coverage,
        "token_usage": (
            result.token_usage.model_dump(mode="json", exclude_none=False)
            if result.token_usage is not None
            else None
        ),
        "estimated_cost": result.estimated_cost,
        "normalized_exact_match": result.normalized_exact_match,
        "judge_score": result.judge_score,
        "judge_result": (
            result.judge_result.model_dump(mode="json", exclude_none=False)
            if result.judge_result is not None
            else None
        ),
    }


def _canonical_failure_payload(result: RunResult) -> dict[str, Any]:
    return {
        "failure_type": (
            result.failure_type.value if result.failure_type is not None else None
        ),
        "failure": (
            result.failure.model_dump(mode="json", exclude_none=False)
            if result.failure is not None
            else None
        ),
    }


def _validate_companion_artifacts(result_path: Path, result: RunResult) -> None:
    """Require the atomically published marker's canonical companion set."""

    attempt = result_path.parent
    native = attempt / "native"
    if native.is_symlink() or not native.is_dir():
        raise EvaluationStateError(
            f"terminal result is missing a regular native directory: {result_path}"
        )
    answer_path = attempt / "answer.md"
    if answer_path.is_symlink() or not answer_path.is_file():
        raise EvaluationStateError(
            f"terminal result is missing answer.md: {result_path}"
        )
    expected_answer = result.final_answer or ""
    if expected_answer and not expected_answer.endswith("\n"):
        expected_answer += "\n"
    if answer_path.read_text(encoding="utf-8") != expected_answer:
        raise EvaluationStateError(f"answer.md/result mismatch in {result_path}")

    expected_payloads = {
        "trace.json": _canonical_trace_payload(result, native),
        "metrics.json": _canonical_metrics_payload(result),
        "failure.json": _canonical_failure_payload(result),
    }
    for name, expected in expected_payloads.items():
        companion = attempt / name
        if companion.is_symlink() or not companion.is_file():
            raise EvaluationStateError(
                f"terminal result is missing regular {name}: {result_path}"
            )
        try:
            actual = json.loads(companion.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationStateError(
                f"invalid {name} companion for {result_path}: {exc}"
            ) from exc
        if actual != expected:
            raise EvaluationStateError(f"{name}/result mismatch in {result_path}")


def atomic_write_json(
    path: Path,
    payload: JsonValue | dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write JSON while retaining explicit null fields."""

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    atomic_write_text(path, encoded, overwrite=overwrite)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one UTF-8 file, refusing accidental replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise EvaluationStateError(f"refusing to overwrite {path}") from exc
            temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_git_sha(cwd: Path) -> str:
    """Resolve the exact local commit without consulting a remote."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not sha:
        return "unknown"
    return sha


def _run_worker_subprocess(
    attempt_directory: Path,
    *,
    task: EvalTask,
    config: ResolvedConfig,
    job: WorkerJob,
    worker_module: str,
    worker_cwd: Path,
    timeout_seconds: float | None,
) -> RunResult:
    command = [
        sys.executable,
        "-m",
        worker_module,
        "--job",
        str((attempt_directory / "job.json").resolve()),
    ]
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else max(1.0, config.budget.wall_time_seconds + 5.0)
    )
    env = sanitized_subprocess_env(
        seed=config.seed,
        backend_kind=config.backend_kind,
        credential_env=config.model.credential_env,
    )
    stdout_path = attempt_directory / "worker.stdout.log"
    stderr_path = attempt_directory / "worker.stderr.log"
    started_at = datetime.now(UTC)
    started_monotonic = monotonic()
    timed_out = False
    return_code: int | None = None
    with (
        stdout_path.open("xb") as stdout_handle,
        stderr_path.open("xb") as stderr_handle,
    ):
        try:
            completed = subprocess.run(
                command,
                cwd=worker_cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                timeout=effective_timeout,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True

    result_path = attempt_directory / "result.json"
    if result_path.exists():
        return validate_persisted_result(
            result_path,
            task=task,
            system_id=config.system_id,
            expected_config=config,
            expected_git_sha=job.git_sha,
        )

    if timed_out:
        result = build_failure_result(
            task=task,
            config=config,
            run_id=job.run_id,
            git_sha=job.git_sha,
            started_at=started_at,
            started_monotonic=started_monotonic,
            completion_status=CompletionStatus.TIMED_OUT,
            failure_type=FailureType.DEADLINE_EXCEEDED,
            message=f"worker exceeded subprocess deadline of {effective_timeout:g}s",
            stage="subprocess",
            details={"timeout_seconds": effective_timeout},
        )
    else:
        result = build_failure_result(
            task=task,
            config=config,
            run_id=job.run_id,
            git_sha=job.git_sha,
            started_at=started_at,
            started_monotonic=started_monotonic,
            completion_status=CompletionStatus.FAILED,
            failure_type=FailureType.RUNNER_ERROR,
            message=f"worker exited without a result marker (exit code {return_code})",
            stage="subprocess",
            details={"return_code": return_code},
        )
    persist_terminal_result(attempt_directory, result)
    return result


def _write_attempt_inputs(
    attempt_directory: Path,
    task: EvalTask,
    config: ResolvedConfig,
    job: WorkerJob,
) -> None:
    atomic_write_json(attempt_directory / "task.json", json_ready(task))
    atomic_write_json(
        attempt_directory / "config.json",
        config.model_dump(mode="json", exclude_none=False),
    )
    # Job is published last among inputs so a worker never sees partial inputs.
    atomic_write_json(
        attempt_directory / "job.json",
        job.model_dump(mode="json", exclude_none=False),
    )


def _inspect_existing_attempts(
    task_root: Path,
    *,
    task: EvalTask,
    expected_config: ResolvedConfig,
    expected_git_sha: str,
) -> RunResult | None:
    attempts = _attempt_directories(task_root)
    latest: RunResult | None = None
    for index, attempt in enumerate(attempts):
        result_path = attempt / "result.json"
        if not result_path.exists() and not result_path.is_symlink():
            continue
        result = validate_persisted_result(
            result_path,
            task=task,
            system_id=expected_config.system_id,
            expected_config=expected_config,
        )
        # Validate every historical marker fail-closed, but resume only skips
        # when the newest attempt itself is terminal.  A newer interrupted
        # rerun is preserved and followed by a fresh attempt.
        if index == len(attempts) - 1 and result.git_sha == expected_git_sha:
            latest = result
    return latest


def _attempt_directories(task_root: Path) -> list[Path]:
    if not task_root.exists():
        return []
    if task_root.is_symlink() or not task_root.is_dir():
        raise EvaluationStateError(f"task output root is not a directory: {task_root}")
    attempts: list[tuple[int, Path]] = []
    for child in task_root.iterdir():
        match = _ATTEMPT_NAME.fullmatch(child.name)
        if match is None:
            continue
        if child.is_symlink() or not child.is_dir():
            raise EvaluationStateError(f"attempt is not a regular directory: {child}")
        attempts.append((int(match.group(1)), child))
    return [item[1] for item in sorted(attempts)]


def _prospective_attempt_name(task_root: Path) -> str:
    attempts = _attempt_directories(task_root)
    next_number = 1
    if attempts:
        match = _ATTEMPT_NAME.fullmatch(attempts[-1].name)
        assert match is not None
        next_number = int(match.group(1)) + 1
    return f"attempt-{next_number:04d}"


def _create_next_attempt(task_root: Path) -> Path:
    task_root.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = task_root / _prospective_attempt_name(task_root)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate.resolve()


def _validate_config_identity(
    config: ResolvedConfig,
    *,
    system_id: str,
    dataset_digest: str,
    seed: int,
    attempt_directory: Path,
) -> None:
    if config.system_id != system_id:
        raise EvaluationExecutionError("config factory returned the wrong system_id")
    if config.dataset_digest != dataset_digest:
        raise EvaluationExecutionError(
            "config factory returned the wrong dataset_digest"
        )
    if config.seed != seed:
        raise EvaluationExecutionError("config factory returned the wrong seed")
    if Path(config.artifact_directory).resolve() != attempt_directory.resolve():
        raise EvaluationExecutionError(
            "config factory returned the wrong artifact_directory"
        )


def _default_system_options(
    system_id: str,
    fixture_directory: str,
) -> dict[str, JsonValue]:
    options: dict[str, JsonValue] = {"fixture_dir": fixture_directory}
    if system_id == "tongagent":
        options.update(
            {
                "effort": "medium",
                "mode": "single",
                "strategy": "adaptive",
                "max_escalations": 2,
            }
        )
    return options


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _reject_secret_fields(payload: Any, *, path: str = "config") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if _looks_like_secret_key(str(key)):
                raise ValueError(
                    f"secrets are forbidden in persisted config: {path}.{key}"
                )
            _reject_secret_fields(value, path=f"{path}.{key}")
    elif isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        for index, value in enumerate(payload):
            _reject_secret_fields(value, path=f"{path}[{index}]")
    elif isinstance(payload, str) and re.search(r"(?i)\bbearer\s+\S+", payload):
        raise ValueError(
            f"bearer credentials are forbidden in persisted config: {path}"
        )


def _validate_fixture_directory(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "." in candidate.parts
    ):
        raise ValueError(
            "fixture_dir must be a non-empty repository-relative path without traversal"
        )
    return candidate.as_posix()


def _discover_fixture_revision(fixture_directory: str) -> str:
    fixture_root = _PACKAGE_ROOT / fixture_directory
    if not (fixture_root / "manifest.json").is_file():
        return DEFAULT_FIXTURE_REVISION
    # Import only the deterministic provider, never a system adapter.  The
    # revision is a canonical content hash and therefore identifies fixture
    # content without putting an absolute machine path in a fingerprint.
    from .offline import FixtureBackend

    return FixtureBackend.from_directory(fixture_root).revision


def _validate_component(value: str, *, kind: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe {kind}: {value!r}")


def _is_sensitive_environment_key(upper: str) -> bool:
    exact = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "MODEL",
        "MODEL_NAME",
        "MODEL_PROVIDER",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "PYTHONPATH",
        "WORKER_MODEL",
        "AUTHORIZATION",
        "COOKIE",
        "HTTP_COOKIE",
        "X_API_KEY",
    }
    if _is_implicit_environment_key(upper) or upper in exact:
        return True
    return (
        upper.endswith(
            (
                "_API_KEY",
                "_ACCESS_TOKEN",
                "_AUTH_TOKEN",
                "_PASSWORD",
                "_SECRET",
                "_TOKEN",
                "_CREDENTIAL",
                "_CREDENTIALS",
            )
        )
        or "API_KEY" in upper
        or "AUTHORIZATION" in upper
        or "COOKIE" in upper
        or "CREDENTIAL" in upper
        or upper.endswith("_KEY")
        or "_AUTH_" in upper
        or "TRACING" in upper
        or upper.endswith("_MODEL")
    )


def _is_implicit_environment_key(upper: str) -> bool:
    if upper in {
        "MODEL",
        "MODEL_NAME",
        "MODEL_PROVIDER",
        "PYTHONPATH",
        "WORKER_MODEL",
    }:
        return True
    if upper.startswith(("LANGCHAIN_", "LANGSMITH_", "OTEL_", "TRACE_", "TRACING_")):
        return True
    return any(
        marker in upper
        for marker in ("API_BASE", "API_VERSION", "BASE_URL", "ENDPOINT")
    ) or upper.endswith(("_MODEL", "_MODEL_NAME", "_MODEL_PROVIDER"))


def _looks_like_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]", "", key.upper())
    return any(
        marker in normalized
        for marker in (
            "APIKEY",
            "ACCESSTOKEN",
            "AUTHTOKEN",
            "AUTHORIZATION",
            "PASSWORD",
            "SECRET",
            "COOKIE",
        )
    )


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
