"""Fresh-process worker for exactly one evaluation task/system attempt."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from .config import ResolvedConfig
from .execution import (
    EvaluationStateError,
    WorkerJob,
    build_failure_result,
    persist_terminal_result,
)
from .judging import EvaluationJudge, apply_evaluation_judge
from .schema import (
    CompletionStatus,
    EvalTask,
    FailureDetail,
    FailureType,
    RunResult,
    normalized_exact_match,
)
from .tracing import sanitize_trace_value


class _InvalidRunnerOutput(ValueError):
    """Internal marker for adapter output that violates the shared contract."""


def run_worker(
    job_path: str | Path,
    *,
    judge: EvaluationJudge | None = None,
) -> RunResult:
    """Execute a single job and publish ``result.json`` only after all artifacts."""

    path = Path(job_path).expanduser().resolve()
    attempt_directory = path.parent
    job = WorkerJob.model_validate_json(path.read_text(encoding="utf-8"))
    task = EvalTask.model_validate_json(
        (attempt_directory / job.task_file).read_text(encoding="utf-8")
    )
    config = ResolvedConfig.model_validate_json(
        (attempt_directory / job.config_file).read_text(encoding="utf-8")
    )
    if Path(config.artifact_directory).resolve() != attempt_directory:
        raise EvaluationStateError(
            "worker config artifact_directory does not match the job directory"
        )
    result_path = attempt_directory / job.result_file
    if result_path.exists() or result_path.is_symlink():
        raise EvaluationStateError(f"result marker already exists: {result_path}")
    (attempt_directory / "native").mkdir(exist_ok=True)

    started_at = datetime.now(UTC)
    started_monotonic = monotonic()
    try:
        # Deliberately late: the parent orchestrator never imports adapters, and
        # each worker gets a clean module/global state.
        from .systems import get_runner

        runner = get_runner(config.system_id)
        raw_result = runner.run(task, config)
        result = _canonicalize_runner_result(
            raw_result,
            task=task,
            config=config,
            run_id=job.run_id,
            git_sha=job.git_sha,
        )
    except Exception as exc:
        result = build_failure_result(
            task=task,
            config=config,
            run_id=job.run_id,
            git_sha=job.git_sha,
            started_at=started_at,
            started_monotonic=started_monotonic,
            completion_status=CompletionStatus.FAILED,
            failure_type=_classify_runner_exception(exc),
            message=_safe_exception_message(exc),
            stage="runner",
            details={"exception_type": type(exc).__name__},
        )
    else:
        # A judge scores usable system output; it must not overwrite an
        # already-terminal system failure and thereby corrupt failure
        # attribution in aggregate reports.
        if result.failure_type is None:
            try:
                result = apply_evaluation_judge(
                    task,
                    result,
                    config,
                    judge=judge,
                )
            except Exception as exc:
                result = _preserve_result_with_judge_failure(result, exc)
    persist_terminal_result(attempt_directory, result)
    return result


def _canonicalize_runner_result(
    raw_result: RunResult | dict[str, Any],
    *,
    task: EvalTask,
    config: ResolvedConfig,
    run_id: str,
    git_sha: str,
) -> RunResult:
    """Validate adapter output and own orchestration-level identity/EM fields."""

    try:
        result = (
            raw_result
            if isinstance(raw_result, RunResult)
            else RunResult.model_validate(raw_result)
        )
    except Exception as exc:
        raise _InvalidRunnerOutput(f"runner returned invalid RunResult: {exc}") from exc
    if result.task_id != task.id:
        raise _InvalidRunnerOutput(
            f"runner returned task_id {result.task_id!r}, expected {task.id!r}"
        )
    if result.system_id != config.system_id:
        raise _InvalidRunnerOutput(
            f"runner returned system_id {result.system_id!r}, "
            f"expected {config.system_id!r}"
        )
    if result.config_fingerprint != config.config_fingerprint:
        raise _InvalidRunnerOutput("runner returned a mismatched config fingerprint")
    if result.fairness_fingerprint != config.fairness_fingerprint:
        raise _InvalidRunnerOutput("runner returned a mismatched fairness fingerprint")
    if (
        Path(result.artifact_directory).resolve()
        != Path(config.artifact_directory).resolve()
    ):
        raise _InvalidRunnerOutput("runner returned a mismatched artifact directory")

    updates: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": git_sha,
        "resolved_config": config,
        "config_fingerprint": config.config_fingerprint,
        "fairness_fingerprint": config.fairness_fingerprint,
        "artifact_directory": str(Path(config.artifact_directory).resolve()),
        "fixture_smoke": config.backend_kind == "fixture",
        "normalized_exact_match": normalized_exact_match(
            result.final_answer,
            task.reference_answer,
        ),
    }
    if result.failure_type is not None and result.failure is None:
        updates["failure"] = FailureDetail(
            failure_type=result.failure_type,
            message="runner returned a terminal failure without details",
            stage="runner",
            retryable=False,
            details={},
        )
    return RunResult.model_validate(
        result.model_copy(update=updates).model_dump(mode="python")
    )


def _classify_runner_exception(exc: Exception) -> FailureType:
    """Map known typed adapter errors without importing adapters eagerly."""

    if isinstance(exc, _InvalidRunnerOutput):
        return FailureType.INVALID_OUTPUT
    failure_type = getattr(exc, "failure_type", None)
    if isinstance(failure_type, FailureType):
        return failure_type
    if isinstance(failure_type, str):
        try:
            return FailureType(failure_type)
        except ValueError:
            pass
    return FailureType.RUNNER_ERROR


def _preserve_result_with_judge_failure(
    result: RunResult,
    exc: Exception,
) -> RunResult:
    """Mark judge failure without discarding the system's answer and metrics."""

    original_failure = (
        result.failure_type.value if result.failure_type is not None else None
    )
    failure = FailureDetail(
        failure_type=FailureType.JUDGE_ERROR,
        message=_safe_exception_message(exc),
        stage="judge",
        retryable=False,
        details={
            "exception_type": type(exc).__name__,
            "system_completion_status": result.completion_status.value,
            "system_failure_type": original_failure,
        },
    )
    return RunResult.model_validate(
        result.model_copy(
            update={
                "completion_status": CompletionStatus.FAILED,
                "failure_type": FailureType.JUDGE_ERROR,
                "failure": failure,
                "judge_score": None,
                "judge_result": None,
            }
        ).model_dump(mode="python")
    )


def _safe_exception_message(exc: Exception) -> str:
    raw = str(exc).strip() or type(exc).__name__
    sanitized = sanitize_trace_value(raw)
    if isinstance(sanitized, str) and sanitized:
        return sanitized
    return "evaluation runner failed"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute exactly one process-isolated evaluation job."
    )
    parser.add_argument("--job", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_worker(args.job)
    except Exception as exc:
        # Invalid/missing job inputs cannot safely produce a canonical result,
        # so leave the attempt incomplete for the parent to classify.
        message = _safe_exception_message(exc)
        print(f"evaluation worker failed before completion: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
