"""Unified contracts and runtime primitives for TongAgent evaluation."""

from .budget import (
    BudgetExceeded,
    BudgetResource,
    BudgetSnapshot,
    ExecutionBudget,
    ModelCallReservation,
    ModelCallSettlement,
)
from .config import (
    BudgetLimits,
    EvaluationJudgeConfig,
    EvaluationModelConfig,
    ResolvedConfig,
    SharedToolConfig,
    canonical_json,
)
from .judging import (
    EvaluationJudge,
    JudgeUnavailableError,
    apply_evaluation_judge,
    register_evaluation_judge,
)
from .schema import (
    AnswerStatus,
    Citation,
    CompletionStatus,
    EvalTask,
    FailureDetail,
    FailureType,
    JudgeResult,
    RunResult,
    TokenUsage,
    ToolCall,
    ToolCallStatus,
    TraceEvent,
    normalize_exact_match_text,
    normalized_exact_match,
    parse_eval_task_jsonl_line,
)
from .tracing import TraceCollector, sanitize_trace_value

__all__ = [
    "BudgetExceeded",
    "BudgetResource",
    "BudgetLimits",
    "BudgetSnapshot",
    "AnswerStatus",
    "Citation",
    "CompletionStatus",
    "EvalTask",
    "EvaluationJudge",
    "EvaluationJudgeConfig",
    "EvaluationModelConfig",
    "ExecutionBudget",
    "FailureDetail",
    "FailureType",
    "JudgeResult",
    "JudgeUnavailableError",
    "ModelCallReservation",
    "ModelCallSettlement",
    "ResolvedConfig",
    "RunResult",
    "SharedToolConfig",
    "SystemRunner",
    "TokenUsage",
    "ToolCall",
    "ToolCallStatus",
    "TraceCollector",
    "TraceEvent",
    "apply_evaluation_judge",
    "canonical_json",
    "normalize_exact_match_text",
    "normalized_exact_match",
    "parse_eval_task_jsonl_line",
    "register_evaluation_judge",
    "sanitize_trace_value",
]


def __getattr__(name: str) -> object:
    """Load the runner protocol lazily to keep import dependencies acyclic."""
    if name == "SystemRunner":
        from .systems.base import SystemRunner

        return SystemRunner
    raise AttributeError(name)
