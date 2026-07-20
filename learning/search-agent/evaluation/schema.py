"""Strict, JSON-serializable contracts for evaluation inputs and results."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from .budget import BudgetResource, BudgetSnapshot
from .config import ResolvedConfig


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINAL_ANSWER_LINE = re.compile(r"(?im)^[ \t]*FINAL_ANSWER[ \t]*:[ \t]*(.*?)[ \t]*$")
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Coverage = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    """Base class shared by persisted evaluation schemas."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )


class EvalTask(StrictModel):
    """One safe, self-contained JSONL evaluation task."""

    id: NonEmptyString
    question: NonEmptyString
    reference_answer: str | None = None
    source_dataset: NonEmptyString | None = None
    source_split: NonEmptyString | None = None
    source_index: NonNegativeInt | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        """Reject IDs that cannot safely form one task-directory component."""
        if not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
            msg = (
                "task id must be 1-128 ASCII letters, digits, '.', '_' or '-', "
                "beginning with a letter or digit"
            )
            raise ValueError(msg)
        return value

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject whitespace-only questions without rewriting user text."""
        if not value.strip():
            msg = "question must contain non-whitespace text"
            raise ValueError(msg)
        return value

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, value: str | None) -> str | None:
        """A present reference must contain text; absence is represented by null."""
        if value is not None and not value.strip():
            msg = "reference_answer must be null or contain non-whitespace text"
            raise ValueError(msg)
        return value

    @field_validator("source_dataset", "source_split")
    @classmethod
    def validate_source_identity(cls, value: str | None) -> str | None:
        """Reject whitespace-only external provenance identifiers."""

        if value is not None and not value.strip():
            raise ValueError("source provenance identifiers must contain text")
        return value

    @model_validator(mode="after")
    def validate_source_provenance(self) -> EvalTask:
        """Require external dataset provenance to be complete or wholly absent."""

        provenance = (
            self.source_dataset,
            self.source_split,
            self.source_index,
        )
        if any(item is not None for item in provenance) and not all(
            item is not None for item in provenance
        ):
            msg = (
                "source_dataset, source_split, and source_index must be "
                "provided together"
            )
            raise ValueError(msg)
        return self


class FailureType(StrEnum):
    """Stable, aggregate-friendly failure classes."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    ACCESS_BLOCKED = "access_blocked"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    DNS_REJECTED = "dns_rejected"
    FETCH_ERROR = "fetch_error"
    FIXTURE_NOT_FOUND = "fixture_not_found"
    INTERRUPTED = "interrupted"
    INVALID_OUTPUT = "invalid_output"
    INVALID_TASK = "invalid_task"
    JUDGE_ERROR = "judge_error"
    MODEL_ERROR = "model_error"
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMITED = "rate_limited"
    RUNNER_ERROR = "runner_error"
    SEARCH_ERROR = "search_error"
    SECURITY_REJECTED = "security_rejected"
    TOOL_ERROR = "tool_error"


class CompletionStatus(StrEnum):
    """Final state of an individual task/system attempt."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


class AnswerStatus(StrEnum):
    """Whether the strict final-answer contract yielded a candidate."""

    ANSWER = "answer"
    ABSTAIN = "abstain"
    EMPTY = "empty"


class ToolCallStatus(StrEnum):
    """Outcome of one attempted model-facing tool call."""

    SUCCESS = "success"
    ERROR = "error"
    BUDGET_EXCEEDED = "budget_exceeded"
    NOT_CALLED = "not_called"


class Citation(StrictModel):
    """A citation emitted in the final answer.

    ``source_id`` and ``quote`` are nullable because B1 and B2 do not own a
    TongAgent evidence graph. Fixture and TongAgent runners should populate
    them whenever provenance is available.
    """

    citation_id: NonEmptyString
    url: NonEmptyString
    source_id: str | None = None
    title: str | None = None
    quote: str | None = None
    claim: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FailureDetail(StrictModel):
    """Machine-readable failure information for one run or tool call."""

    failure_type: FailureType
    message: NonEmptyString
    stage: str | None = None
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ToolCall(StrictModel):
    """Sanitized trace record for a single tool invocation."""

    call_id: NonEmptyString
    tool_name: NonEmptyString
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: NonNegativeFloat | None = None
    status: ToolCallStatus
    result: JsonValue | None = None
    failure: FailureDetail | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timing_and_failure(self) -> ToolCall:
        """Keep timing and error state internally coherent."""
        if self.finished_at is not None and self.finished_at < self.started_at:
            msg = "finished_at cannot precede started_at"
            raise ValueError(msg)
        if self.status == ToolCallStatus.ERROR and self.failure is None:
            msg = "error tool calls require failure details"
            raise ValueError(msg)
        if (
            self.status not in {ToolCallStatus.ERROR, ToolCallStatus.BUDGET_EXCEEDED}
            and self.failure is not None
        ):
            msg = "non-error tool calls cannot contain failure details"
            raise ValueError(msg)
        return self


class TokenUsage(StrictModel):
    """Provider token accounting.

    The whole object is null when the provider exposes no token information.
    Individual nullable fields permit honest partial accounting.
    """

    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None
    cached_input_tokens: NonNegativeInt | None = None
    reasoning_tokens: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_known_usage(self) -> TokenUsage:
        """Disallow a non-null object that contains no actual information."""
        values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_input_tokens,
            self.reasoning_tokens,
        )
        if all(value is None for value in values):
            msg = "unknown token usage must be represented by null"
            raise ValueError(msg)
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            msg = "total_tokens cannot be smaller than input_tokens + output_tokens"
            raise ValueError(msg)
        return self


class TraceEvent(StrictModel):
    """One ordered, already-sanitized runner trace event."""

    sequence: Annotated[int, Field(ge=1)]
    event_type: NonEmptyString
    occurred_at: datetime
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class JudgeResult(StrictModel):
    """Normalized output of an optional, separately identified judge."""

    score: Coverage
    rationale: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RunResult(StrictModel):
    """Canonical result written for every task/system attempt.

    Optional benchmark metrics remain explicit JSON ``null`` when unavailable.
    In particular, baseline systems must not invent TongAgent evidence or
    structural-coverage values.
    """

    run_id: NonEmptyString
    task_id: NonEmptyString
    system_id: NonEmptyString
    git_sha: NonEmptyString
    resolved_config: ResolvedConfig
    config_fingerprint: NonEmptyString
    fairness_fingerprint: NonEmptyString
    started_at: datetime
    finished_at: datetime
    wall_time_seconds: NonNegativeFloat
    final_answer: str | None
    extracted_answer: str | None = None
    answer_status: AnswerStatus = AnswerStatus.EMPTY
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    external_retrieval_calls: NonNegativeInt | None = None
    internal_tool_calls: NonNegativeInt | None = None
    search_calls: NonNegativeInt | None
    fetch_calls: NonNegativeInt | None
    relevant_searches: NonNegativeInt | None
    evidence_count: NonNegativeInt | None
    structural_subquestion_coverage: Coverage | None
    token_usage: TokenUsage | None
    estimated_cost: NonNegativeFloat | None
    completion_status: CompletionStatus
    failure_type: FailureType | None
    failure: FailureDetail | None = None
    budget_resource: BudgetResource | None = None
    budget_snapshot: BudgetSnapshot | None = None
    budget_accounting_version: Literal[2] | None = None
    artifact_directory: NonEmptyString
    fixture_smoke: bool
    normalized_exact_match: bool | None
    judge_score: Coverage | None
    judge_result: JudgeResult | None = None

    @model_validator(mode="after")
    def validate_result_coherence(self) -> RunResult:
        """Reject mismatched identities, fingerprints, timing, and failures."""
        if self.system_id != self.resolved_config.system_id:
            msg = "system_id must match resolved_config.system_id"
            raise ValueError(msg)
        if self.config_fingerprint != self.resolved_config.config_fingerprint:
            msg = "config_fingerprint must match the resolved configuration"
            raise ValueError(msg)
        if self.fairness_fingerprint != self.resolved_config.fairness_fingerprint:
            msg = "fairness_fingerprint must match the resolved configuration"
            raise ValueError(msg)
        if self.finished_at < self.started_at:
            msg = "finished_at cannot precede started_at"
            raise ValueError(msg)
        if self.completion_status == CompletionStatus.COMPLETED:
            if self.final_answer is None:
                msg = "completed runs require a final_answer"
                raise ValueError(msg)
            if self.failure_type is not None or self.failure is not None:
                msg = "completed runs cannot contain failure information"
                raise ValueError(msg)
        if (
            self.completion_status
            in {
                CompletionStatus.FAILED,
                CompletionStatus.BUDGET_EXHAUSTED,
                CompletionStatus.TIMED_OUT,
                CompletionStatus.INTERRUPTED,
            }
            and self.failure_type is None
        ):
            msg = "non-completed terminal runs require failure_type"
            raise ValueError(msg)
        if self.failure is not None:
            if self.failure_type != self.failure.failure_type:
                msg = "failure_type must match failure.failure_type"
                raise ValueError(msg)
        if self.judge_result is None and self.judge_score is not None:
            msg = "judge_score requires judge_result"
            raise ValueError(msg)
        if (
            self.judge_result is not None
            and self.judge_score != self.judge_result.score
        ):
            msg = "judge_score must equal judge_result.score"
            raise ValueError(msg)
        if (
            self.search_calls is not None
            and self.relevant_searches is not None
            and self.relevant_searches > self.search_calls
        ):
            msg = "relevant_searches cannot exceed search_calls"
            raise ValueError(msg)
        if (
            self.external_retrieval_calls is not None
            and self.search_calls is not None
            and self.fetch_calls is not None
            and self.external_retrieval_calls != self.search_calls + self.fetch_calls
        ):
            msg = "external_retrieval_calls must equal search_calls + fetch_calls"
            raise ValueError(msg)
        parsed_status, parsed_answer = extract_answer_contract(self.final_answer)
        if (
            "answer_status" in self.model_fields_set
            and self.answer_status != parsed_status
        ):
            raise ValueError("answer_status must match the FINAL_ANSWER contract")
        if (
            "extracted_answer" in self.model_fields_set
            and self.extracted_answer != parsed_answer
        ):
            raise ValueError("extracted_answer must match the FINAL_ANSWER contract")
        self.answer_status = parsed_status
        self.extracted_answer = parsed_answer
        if (
            self.completion_status == CompletionStatus.BUDGET_EXHAUSTED
            and self.failure_type != FailureType.BUDGET_EXHAUSTED
        ):
            raise ValueError(
                "budget_exhausted completion requires budget_exhausted failure_type"
            )
        if (
            self.completion_status == CompletionStatus.TIMED_OUT
            and self.failure_type != FailureType.DEADLINE_EXCEEDED
        ):
            raise ValueError(
                "timed_out completion requires deadline_exceeded failure_type"
            )
        budget_terminal = self.completion_status in {
            CompletionStatus.BUDGET_EXHAUSTED,
            CompletionStatus.TIMED_OUT,
        }
        if not budget_terminal and (
            self.budget_resource is not None or self.budget_snapshot is not None
        ):
            raise ValueError(
                "budget resource fields are only valid for budget/deadline terminals"
            )
        if (
            self.budget_accounting_version == 2
            and self.completion_status == CompletionStatus.BUDGET_EXHAUSTED
            and (self.budget_resource is None or self.budget_snapshot is None)
        ):
            msg = "budget_exhausted runs require budget_resource and budget_snapshot"
            raise ValueError(msg)
        if self.budget_accounting_version == 2 and (
            self.external_retrieval_calls is None or self.internal_tool_calls is None
        ):
            msg = "budget accounting v2 requires external/internal tool counters"
            raise ValueError(msg)
        if self.budget_accounting_version == 2 and (
            self.search_calls is None or self.fetch_calls is None
        ):
            raise ValueError("budget accounting v2 requires search/fetch counters")
        if self.budget_accounting_version == 2 and any(
            call.status == ToolCallStatus.BUDGET_EXCEEDED and call.failure is None
            for call in self.tool_calls
        ):
            raise ValueError("v2 budget-exceeded tool calls require failure details")
        if (
            self.budget_accounting_version == 2
            and self.completion_status == CompletionStatus.BUDGET_EXHAUSTED
            and self.budget_resource == BudgetResource.DEADLINE
        ):
            raise ValueError("deadline exhaustion must use timed_out completion status")
        if (
            self.budget_accounting_version == 2
            and self.completion_status == CompletionStatus.TIMED_OUT
            and (
                self.budget_resource != BudgetResource.DEADLINE
                or self.budget_snapshot is None
            )
        ):
            raise ValueError("v2 timed_out runs require deadline resource and snapshot")
        if (
            self.budget_accounting_version == 2
            and self.completion_status == CompletionStatus.TIMED_OUT
            and self.budget_snapshot is not None
            and not self.budget_snapshot.deadline_exceeded
        ):
            raise ValueError("v2 timed_out runs require a deadline-exceeded snapshot")
        if self.budget_accounting_version == 2 and self.budget_snapshot is not None:
            snapshot = self.budget_snapshot
            observed = (
                self.search_calls,
                self.fetch_calls,
                self.external_retrieval_calls,
                self.internal_tool_calls,
            )
            denied_at = (
                snapshot.search_calls,
                snapshot.fetch_calls,
                snapshot.external_retrieval_calls,
                snapshot.internal_tool_calls,
            )
            if observed != denied_at:
                raise ValueError(
                    "top-level tool counters must match the denial-time snapshot"
                )
        return self


def extract_answer_contract(
    final_answer: object,
) -> tuple[AnswerStatus, str | None]:
    """Parse exactly one explicit ``FINAL_ANSWER:`` line.

    Explanations and limitation text outside the marker are never promoted to
    candidate answers.
    """

    if not isinstance(final_answer, str) or not final_answer.strip():
        return AnswerStatus.EMPTY, None
    matches = _FINAL_ANSWER_LINE.findall(final_answer)
    if len(matches) != 1:
        return AnswerStatus.EMPTY, None
    value = matches[0].strip()
    if not value:
        return AnswerStatus.EMPTY, None
    if value.casefold() == "abstain":
        return AnswerStatus.ABSTAIN, None
    return AnswerStatus.ANSWER, value


def answer_for_exact_match(final_answer: str | None) -> str | None:
    """Use strict extracted answers while retaining legacy fixture behavior."""

    status, answer = extract_answer_contract(final_answer)
    if _FINAL_ANSWER_LINE.search(final_answer or "") is not None:
        return answer if status == AnswerStatus.ANSWER else None
    return final_answer


def normalize_exact_match_text(value: str) -> str:
    """Normalize text conservatively for whole-string exact matching.

    This intentionally performs only Unicode NFKC normalization, case folding,
    and whitespace collapsing. It does not drop punctuation, articles, or
    answer prefixes, and therefore cannot turn a substring into a match.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def normalized_exact_match(
    prediction: str | None,
    reference_answer: str | None,
) -> bool | None:
    """Return conservative normalized whole-string equality.

    A missing reference makes the metric unavailable (``None``). A missing
    prediction against a present reference is an observed non-match.
    """

    if reference_answer is None:
        return None
    scoring_prediction = answer_for_exact_match(prediction)
    if scoring_prediction is None:
        return False
    return normalize_exact_match_text(scoring_prediction) == normalize_exact_match_text(
        reference_answer
    )


def parse_eval_task_jsonl_line(
    line: str, *, line_number: int | None = None
) -> EvalTask:
    """Parse one non-blank JSONL record with strict schema validation."""

    if not line.strip():
        location = f" at line {line_number}" if line_number is not None else ""
        msg = f"blank JSONL record{location}"
        raise ValueError(msg)
    try:
        return EvalTask.model_validate_json(line)
    except Exception as exc:
        if line_number is None:
            raise
        msg = f"invalid evaluation task at line {line_number}: {exc}"
        raise ValueError(msg) from exc


def json_ready(model: BaseModel) -> dict[str, Any]:
    """Return a JSON-safe dump that retains explicit null fields."""

    return model.model_dump(mode="json", exclude_none=False)
