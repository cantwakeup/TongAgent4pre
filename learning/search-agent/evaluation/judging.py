"""Process-local, explicit evaluation-judge extension point."""

from __future__ import annotations

from threading import RLock
from typing import Protocol, runtime_checkable

from .config import EvaluationJudgeConfig, ResolvedConfig
from .schema import EvalTask, JudgeResult, RunResult


class JudgeUnavailableError(ValueError):
    """Raised when a configured judge has no matching registered implementation."""


@runtime_checkable
class EvaluationJudge(Protocol):
    """Score one canonical result under an explicit judge identity."""

    id: str
    version: str

    def evaluate(
        self,
        task: EvalTask,
        result: RunResult,
        config: EvaluationJudgeConfig,
    ) -> JudgeResult:
        """Return one validated score without mutating the system result."""
        ...


_REGISTRY: dict[tuple[str, str], tuple[EvaluationJudge, bool]] = {}
_REGISTRY_LOCK = RLock()


def register_evaluation_judge(
    judge: EvaluationJudge,
    *,
    subprocess_safe: bool = False,
) -> None:
    """Register an implementation under its immutable identity.

    Dynamic objects default to process-local. A future built-in judge may opt
    into ``subprocess_safe`` only when the same registration occurs at import
    time in every fresh evaluation worker.
    """

    if not isinstance(judge, EvaluationJudge):
        raise TypeError("judge must implement EvaluationJudge")
    key = (judge.id, judge.version)
    if not all(isinstance(item, str) and item.strip() for item in key):
        raise ValueError("judge id and version must contain non-whitespace text")
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and existing[0] is not judge:
            raise ValueError(f"judge implementation already registered: {key!r}")
        _REGISTRY[key] = (judge, subprocess_safe)


def resolve_evaluation_judge(
    config: EvaluationJudgeConfig,
    *,
    require_subprocess_safe: bool = False,
) -> EvaluationJudge:
    """Resolve a configured judge or fail instead of silently emitting null."""

    key = (config.id, config.version)
    with _REGISTRY_LOCK:
        registration = _REGISTRY.get(key)
    if registration is None:
        raise JudgeUnavailableError(
            f"no evaluation judge registered for id={config.id!r} "
            f"version={config.version!r}"
        )
    judge, subprocess_safe = registration
    if require_subprocess_safe and not subprocess_safe:
        raise JudgeUnavailableError(
            f"evaluation judge id={config.id!r} version={config.version!r} "
            "is only process-local and cannot run in fresh workers"
        )
    return judge


def ensure_judge_available(config: ResolvedConfig) -> None:
    """Fail preflight when a non-null judge cannot be executed."""

    if config.judge is not None:
        resolve_evaluation_judge(config.judge, require_subprocess_safe=True)


def apply_evaluation_judge(
    task: EvalTask,
    result: RunResult,
    resolved_config: ResolvedConfig,
    *,
    judge: EvaluationJudge | None = None,
) -> RunResult:
    """Apply an explicit judge, retaining null when judge configuration is null."""

    judge_config = resolved_config.judge
    if judge_config is None:
        if judge is not None:
            raise ValueError("judge implementation supplied while config.judge is null")
        return result.model_copy(update={"judge_score": None, "judge_result": None})
    selected = judge or resolve_evaluation_judge(judge_config)
    if (selected.id, selected.version) != (judge_config.id, judge_config.version):
        raise ValueError("judge implementation identity does not match config.judge")
    judged = selected.evaluate(task, result, judge_config)
    return result.model_copy(
        update={"judge_score": judged.score, "judge_result": judged}
    )
