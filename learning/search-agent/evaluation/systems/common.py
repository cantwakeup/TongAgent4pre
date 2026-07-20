"""Shared runtime, accounting, and artifact helpers for evaluated systems.

The baseline adapters deliberately remain thin.  This module owns the pieces
that must be identical across systems:

* one hard :class:`ExecutionBudget` for model and tool calls;
* one sanitized :class:`TraceCollector`;
* TongAgent's production search/fetch semantic wrappers, with raw providers
  injected underneath them and an evidence-free ledger for B1/B2;
* conservative extraction of answers, citations, token usage, and failures;
* standard, atomic intermediate artifacts.

``result.json`` is intentionally not written here.  The experiment executor
writes it last, after validating every intermediate artifact, so its presence
remains the sole completion marker used by safe resume.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import JsonValue

from agent_policy import EffortPolicy

from ..budget import (
    BudgetExceeded,
    BudgetResource,
    BudgetSnapshot,
    ExecutionBudget,
    ModelCallReservation,
)
from ..config import ResolvedConfig
from ..offline import FixtureBackend, FixtureChatModel
from ..schema import (
    Citation,
    CompletionStatus,
    EvalTask,
    FailureDetail,
    FailureType,
    RunResult,
    TokenUsage,
    ToolCall,
    ToolCallStatus,
    answer_for_exact_match,
    extract_answer_contract,
    normalized_exact_match,
)
from ..tracing import TraceCollector, sanitize_trace_value
from ..token_control import (
    DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS,
    ModelStage,
    PartitionDenial,
    PartitionReservation,
    StageTokenController,
)


SYSTEM_SIMPLE_REACT = "simple_react"
SYSTEM_VANILLA_DEEPAGENTS = "vanilla_deepagents"
_URL = re.compile(r"https?://[^\s<>()\[\]{}\"']+")
_SOURCE_ID = re.compile(r"(?<![A-Za-z0-9])S[1-9][0-9]*(?![A-Za-z0-9])")
_HEX_SHA = re.compile(r"^[0-9a-f]{7,64}$")
SemanticStrategy = Literal["fixed", "adaptive"]
_FINAL_SYNTHESIS_INPUT_TOKEN_RESERVE = 4_096
_FINAL_SYNTHESIS_DRAFT_CHARS = 6_000
_TONGAGENT_PAGE_CONTEXT_CHARS = 3_500
_TONGAGENT_SEARCH_SNIPPET_CHARS = 320


class _BaselineNoEvidenceState:
    """Explicitly disable Evidence Graph state for the B1/B2 baselines.

    The three evaluated systems reuse the exact same production search/fetch
    wrappers.  Those wrappers accept an injected retrieval ledger, but the
    production ``ResearchBudget`` normally creates an ``EvidenceGraphStore``.
    B1/B2 need the shared search semantics and counters without silently
    acquiring TongAgent evidence state, so they inject this deliberately empty
    state object instead.  No evidence tools are exposed to either baseline.
    """

    def cache_page(
        self,
        source_id: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del source_id, content, metadata

    @staticmethod
    def snapshot() -> dict[str, Any]:
        return {"evidence_graph_available": False}

    @staticmethod
    def reset() -> None:
        return None

    @staticmethod
    def restore(snapshot: Mapping[str, Any]) -> None:
        del snapshot


@dataclass(frozen=True)
class PreparedRuntime:
    """Dependencies shared by one system/task attempt."""

    model: BaseChatModel
    tools: tuple[BaseTool, ...]
    research_budget: Any
    execution_budget: ExecutionBudget
    middleware: EvaluationMiddleware
    trace: TraceCollector
    fixture_backend: FixtureBackend | None


class EvaluationMiddleware(AgentMiddleware):
    """Enforce and observe the shared model/tool budget.

    A single instance may be installed in both a DeepAgents parent and its
    native general-purpose subagent.  Its mutable state is therefore protected
    by a re-entrant lock and all counters ultimately delegate to the
    thread-safe :class:`ExecutionBudget`.
    """

    def __init__(
        self,
        execution_budget: ExecutionBudget,
        trace: TraceCollector,
        *,
        max_output_tokens: int,
        reserve_final_synthesis: bool = False,
        stage_output_caps: Mapping[ModelStage, int] | None = None,
        enable_context_compaction: bool = False,
        enable_token_partitions: bool = False,
    ) -> None:
        super().__init__()
        self.execution_budget = execution_budget
        self.trace = trace
        self._max_output_tokens = max_output_tokens
        self._stage_output_caps: dict[ModelStage, int] = {}
        if stage_output_caps is not None:
            configured_caps = dict(DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS)
            configured_caps.update(stage_output_caps)
            self._stage_output_caps = {
                stage: min(max_output_tokens, cap)
                for stage, cap in configured_caps.items()
            }
        self._enable_context_compaction = enable_context_compaction
        self._token_controller = (
            StageTokenController(
                total_token_limit=execution_budget.limits.max_total_tokens,
                stage_output_caps=cast(
                    "Mapping[ModelStage, int]",
                    self._stage_output_caps,
                ),
            )
            if enable_token_partitions
            else None
        )
        self._lock = RLock()
        self._tool_calls: list[ToolCall] = []
        self._failures: list[FailureDetail] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._cached_input_tokens = 0
        self._reasoning_tokens = 0
        self._known_usage_fields: set[str] = set()
        self._budget_denied = False
        self._budget_failure: BudgetExceeded | None = None
        self._final_evidence_fragments: list[str] = []
        self._final_synthesis_reservation: ModelCallReservation | None = None
        self._partition_reservations: dict[int, PartitionReservation] = {}
        self._reservation_context: dict[int, dict[str, Any]] = {}
        self._model_call_sequence = 0
        self._tools_since_model_call: list[str] = []
        if reserve_final_synthesis:
            final_cap = self.stage_output_cap("final_extractor")
            token_reservation = final_cap + _FINAL_SYNTHESIS_INPUT_TOKEN_RESERVE
            partition = self._reserve_partition(
                stage="final_extractor",
                token_reservation=token_reservation,
                active_subquestion_id=None,
            )
            reservation = execution_budget.require_model_call(
                token_reservation=token_reservation
            )
            if partition is not None:
                self._partition_reservations[reservation.reservation_id] = partition
            self._final_synthesis_reservation = reservation
            self.trace.record(
                "final_synthesis_reserved",
                token_reservation=token_reservation,
                stage="final_extractor",
                stage_output_cap=final_cap,
                budget=execution_budget.snapshot(),
                token_partition=self.token_partition_snapshot(),
            )

    @property
    def budget_denied(self) -> bool:
        """Whether any model, token, deadline, or tool reservation was denied."""

        with self._lock:
            return self._budget_denied

    @property
    def budget_failure(self) -> BudgetExceeded | None:
        """Return the exact terminal execution-budget denial, if any."""

        with self._lock:
            return self._budget_failure

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Return a detached canonical tool trace."""

        with self._lock:
            return [item.model_copy(deep=True) for item in self._tool_calls]

    @property
    def failures(self) -> list[FailureDetail]:
        """Return detached observed failures, including handled tool failures."""

        with self._lock:
            return [item.model_copy(deep=True) for item in self._failures]

    @property
    def token_usage(self) -> TokenUsage | None:
        """Return provider-reported usage, or null when no field was reported."""

        with self._lock:
            if not self._known_usage_fields:
                return None
            values: dict[str, int | None] = {
                "input_tokens": (
                    self._input_tokens
                    if "input_tokens" in self._known_usage_fields
                    else None
                ),
                "output_tokens": (
                    self._output_tokens
                    if "output_tokens" in self._known_usage_fields
                    else None
                ),
                "total_tokens": (
                    self._total_tokens
                    if "total_tokens" in self._known_usage_fields
                    else None
                ),
                "cached_input_tokens": (
                    self._cached_input_tokens
                    if "cached_input_tokens" in self._known_usage_fields
                    else None
                ),
                "reasoning_tokens": (
                    self._reasoning_tokens
                    if "reasoning_tokens" in self._known_usage_fields
                    else None
                ),
            }
        return TokenUsage.model_validate(values)

    def stage_output_cap(self, stage: ModelStage) -> int:
        """Return the auditable cap used for one TongAgent model stage."""

        return int(self._stage_output_caps.get(stage, self._max_output_tokens))

    def configure_token_partitions(self, subquestion_ids: Sequence[str]) -> None:
        """Partition the global token ceiling after a durable plan exists."""

        controller = self._token_controller
        if controller is None:
            return
        controller.configure(subquestion_ids)
        self.trace.record(
            "token_partitions_configured",
            subquestion_ids=list(subquestion_ids),
            token_partition=controller.snapshot(),
        )

    def activate_token_subquestion(self, subquestion_id: str | None) -> None:
        """Select the SQ bucket used by subsequent model reservations."""

        controller = self._token_controller
        if controller is None:
            return
        controller.activate(subquestion_id)
        self.trace.record(
            "token_partition_activated",
            active_subquestion_id=subquestion_id,
            token_partition=controller.snapshot(),
        )

    def can_start_token_subquestion(self, subquestion_id: str) -> bool:
        controller = self._token_controller
        return (
            True
            if controller is None
            else controller.can_start_subquestion(subquestion_id)
        )

    def token_partition_snapshot(self) -> dict[str, Any]:
        controller = self._token_controller
        return {} if controller is None else controller.snapshot()

    def model_for_stage(
        self,
        model: BaseChatModel,
        stage: ModelStage,
    ) -> BaseChatModel:
        """Clone models that expose a max-token field for out-of-stack calls."""

        cap = self.stage_output_cap(stage)
        fields = getattr(type(model), "model_fields", {})
        if "max_tokens" not in fields:
            return model
        return cast(
            "BaseChatModel",
            model.model_copy(update={"max_tokens": cap}, deep=False),
        )

    def reserve_external_model_call(
        self,
        *,
        label: str,
        request_payload: Any,
        stage: ModelStage | None = None,
    ) -> ModelCallReservation:
        """Reserve a model call made outside a LangChain agent middleware stack.

        TongAgent's explicit planner is one such call.  The caller must pair a
        successful reservation with :meth:`record_external_model_response`.
        """

        resolved_stage = stage or _stage_for_external_label(label)
        context = _external_context_profile(request_payload)
        return self._reserve_model_call(
            label=label,
            estimated_input_tokens=_estimate_external_input_tokens(request_payload),
            stage=resolved_stage,
            active_subquestion_id=(
                self._token_controller.active_subquestion_id
                if self._token_controller is not None
                else None
            ),
            context_profile=context,
        )

    def cancel_external_model_call(
        self,
        reservation: ModelCallReservation,
        *,
        label: str,
    ) -> None:
        """Release a planner reservation after provider failure."""

        settlement = self.execution_budget.cancel_model_call(reservation)
        self._cancel_partition_reservation(reservation)
        context = self._reservation_context.pop(reservation.reservation_id, {})
        self.trace.record(
            "model_call_cancelled",
            label=label,
            call_sequence=context.get("call_sequence"),
            stage=context.get("stage"),
            active_subquestion_id=context.get("active_subquestion_id"),
            settlement=settlement,
            budget=self.execution_budget.snapshot(),
            token_partition=self.token_partition_snapshot(),
        )

    def record_external_model_response(
        self,
        response: Any,
        *,
        label: str,
        reservation: ModelCallReservation,
    ) -> None:
        """Account token metadata for an externally executed model response."""

        self._record_model_response(
            response,
            label=label,
            reservation=reservation,
        )

    def external_tool_guard(self, tool_name: str) -> dict[str, JsonValue] | None:
        """Reserve provider-facing retrieval after semantic slice checks pass."""

        try:
            self.execution_budget.require_tool(tool_name)
        except BudgetExceeded as exc:
            with self._lock:
                self._budget_denied = True
                self._budget_failure = exc
            snapshot = exc.snapshot or self.execution_budget.snapshot()
            self.trace.record(
                "external_retrieval_budget_exceeded",
                tool_name=tool_name,
                budget_resource=exc.resource,
                attempted=exc.attempted,
                budget=snapshot,
            )
            return {
                "status": "budget_exceeded",
                "outcome": "budget_exceeded",
                "reason": "execution_budget_exceeded",
                "budget_resource": exc.resource.value,
                "budget_snapshot": cast(
                    "JsonValue",
                    snapshot.model_dump(mode="json"),
                ),
                "attempted": cast(
                    "JsonValue",
                    sanitize_trace_value(exc.attempted),
                ),
                "provider_success": False,
                "provider_outcome": "not_called",
                "retryable": False,
            }
        return None

    def finalize_answer(
        self,
        model: BaseChatModel,
        *,
        question: str,
        draft: str | None,
        required_evidence_complete: bool = True,
        missing_subquestions: Sequence[str] = (),
    ) -> str:
        """Use the held common reservation for one strict final synthesis."""

        with self._lock:
            reservation = self._final_synthesis_reservation
            self._final_synthesis_reservation = None
        if reservation is None:
            return draft or ""

        clean_draft = _remove_final_answer_lines(draft or "")
        if not required_evidence_complete:
            settlement = self.execution_budget.cancel_model_call(reservation)
            self._cancel_partition_reservation(reservation)
            missing = ", ".join(item for item in missing_subquestions if item) or (
                "one or more required subquestions"
            )
            self.trace.record(
                "final_synthesis_skipped",
                stage="final_extractor",
                reason="required_subquestions_incomplete",
                missing_subquestions=list(missing_subquestions),
                settlement=settlement,
                budget=self.execution_budget.snapshot(),
                token_partition=self.token_partition_snapshot(),
            )
            explanation = (
                f"INSUFFICIENT_EVIDENCE: missing required subquestions: {missing}"
            )
            return f"{explanation}\nFINAL_ANSWER: ABSTAIN"

        with self._lock:
            evidence_context = "\n\n".join(self._final_evidence_fragments)
        prompt = (
            "Question:\n"
            f"{question[:2_000]}\n\n"
            "Agent draft and retrieved findings:\n"
            f"{clean_draft[:_FINAL_SYNTHESIS_DRAFT_CHARS]}\n\n"
            "Retrieved evidence context (may be partial):\n"
            f"{evidence_context[:_FINAL_SYNTHESIS_DRAFT_CHARS]}\n\n"
            "Return exactly one line in this form:\n"
            "FINAL_ANSWER: <concise core answer>\n"
            "Use FINAL_ANSWER: ABSTAIN only when the draft contains no usable "
            "evidence. With partial usable evidence, perform a best-effort "
            "synthesis. Never use a browsing limitation or request for more "
            "information as the candidate answer."
        )
        label = "evaluation.final_synthesis"
        stage: ModelStage = "final_extractor"
        cap = self.stage_output_cap(stage)
        context = _external_context_profile(
            [
                SystemMessage(content="strict answer extractor"),
                HumanMessage(content=prompt),
            ]
        )
        reservation_context = self._reservation_context.setdefault(
            reservation.reservation_id,
            {},
        )
        self._model_call_sequence += 1
        reservation_context.update(
            {
                "call_sequence": self._model_call_sequence,
                "stage": stage,
                "active_subquestion_id": None,
                "stage_output_cap": cap,
                "context": context,
                "tools_since_previous_call": self._consume_tools_since_model_call(),
            }
        )
        self.trace.record(
            "model_call_started",
            label=label,
            call_sequence=self._model_call_sequence,
            stage=stage,
            active_subquestion_id=None,
            estimated_input_tokens=_estimate_external_input_tokens(
                [
                    SystemMessage(content="strict answer extractor"),
                    HumanMessage(content=prompt),
                ]
            ),
            max_output_tokens=cap,
            token_reservation=reservation.reserved_tokens,
            context=context,
            tools_since_previous_call=reservation_context["tools_since_previous_call"],
            budget=self.execution_budget.snapshot(),
            token_partition=self.token_partition_snapshot(),
        )
        started = time.perf_counter()
        try:
            response = self.model_for_stage(model, stage).invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a strict answer extractor. Do not add "
                            "explanation, citations, or formatting outside the "
                            "single FINAL_ANSWER line."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception:
            self.cancel_external_model_call(reservation, label=label)
            raise

        duration = max(0.0, time.perf_counter() - started)
        try:
            self.record_external_model_response(
                response,
                label=label,
                reservation=reservation,
            )
        except BudgetExceeded:
            # Actual provider usage is already settled and the exact token
            # denial recorded. Preserve the obtained final answer.
            pass
        marker = _strict_final_marker(_message_text(response))
        self.trace.record(
            "final_synthesis_finished",
            call_sequence=self._model_call_sequence,
            stage=stage,
            duration_seconds=duration,
            marker=marker,
            budget=self.execution_budget.snapshot(),
            token_partition=self.token_partition_snapshot(),
        )
        prefix = clean_draft.rstrip()
        return f"{prefix}\n\n{marker}".lstrip() if prefix else marker

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Reserve, trace, and account one synchronous model call."""

        label = _model_label(request)
        stage = _stage_for_model_request(request)
        active_subquestion_id = _active_subquestion_id(request)
        original_context = _model_context_profile(request)
        effective_request = (
            _compact_tongagent_model_request(request)
            if self._enable_context_compaction
            else request
        )
        cap = self.stage_output_cap(stage)
        model_settings = dict(effective_request.model_settings)
        model_settings["max_tokens"] = cap
        effective_request = effective_request.override(model_settings=model_settings)
        effective_context = _model_context_profile(effective_request)
        reservation = self._reserve_model_call(
            label=label,
            estimated_input_tokens=_estimate_model_request_input_tokens(
                effective_request
            ),
            stage=stage,
            active_subquestion_id=active_subquestion_id,
            context_profile=effective_context,
            original_context_profile=original_context,
        )
        started = time.perf_counter()
        try:
            response = handler(effective_request)
        except Exception as exc:
            settlement = self.execution_budget.cancel_model_call(reservation)
            self._cancel_partition_reservation(reservation)
            context = self._reservation_context.pop(reservation.reservation_id, {})
            duration = max(0.0, time.perf_counter() - started)
            failure = _failure_from_exception(
                exc,
                default_type=FailureType.MODEL_ERROR,
                stage="model_call",
            )
            self._append_failure(failure)
            self.trace.record(
                "model_call_failed",
                label=label,
                call_sequence=context.get("call_sequence"),
                stage=stage,
                active_subquestion_id=active_subquestion_id,
                duration_seconds=duration,
                failure=failure,
                settlement=settlement,
                budget=self.execution_budget.snapshot(),
                token_partition=self.token_partition_snapshot(),
            )
            raise
        duration = max(0.0, time.perf_counter() - started)
        reservation_context = dict(
            self._reservation_context.get(reservation.reservation_id, {})
        )
        try:
            self._record_model_response(
                response,
                label=label,
                reservation=reservation,
            )
        except BudgetExceeded:
            self.trace.record(
                "model_call_token_budget_exceeded",
                label=label,
                duration_seconds=duration,
                token_usage=self.token_usage,
            )
            raise
        self.trace.record(
            "model_call_finished",
            label=label,
            call_sequence=reservation_context.get("call_sequence"),
            stage=stage,
            active_subquestion_id=active_subquestion_id,
            duration_seconds=duration,
            token_usage=_usage_from_response(response),
            context=effective_context,
            original_context=original_context,
            stage_output_cap=cap,
            budget=self.execution_budget.snapshot(),
            token_partition=self.token_partition_snapshot(),
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Reserve, trace, and account one synchronous tool call."""

        raw_call = request.tool_call
        tool_name = str(raw_call.get("name") or getattr(request.tool, "name", "tool"))
        call_id = str(raw_call.get("id") or f"eval-tool-{uuid.uuid4().hex}")
        arguments = _sanitized_arguments(raw_call.get("args", {}))
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        denial: BudgetExceeded | None = None
        if not _is_external_retrieval_tool(tool_name):
            try:
                self.execution_budget.require_tool(tool_name)
            except BudgetExceeded as exc:
                denial = exc
        if denial is not None:
            duration = max(0.0, time.perf_counter() - started)
            finished_at = datetime.now(UTC)
            with self._lock:
                self._budget_denied = True
                self._budget_failure = denial
            snapshot = denial.snapshot or self.execution_budget.snapshot()
            failure = _failure_from_exception(
                denial,
                default_type=FailureType.BUDGET_EXHAUSTED,
                stage=tool_name,
            )
            tool_call = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                status=ToolCallStatus.BUDGET_EXCEEDED,
                result={
                    "status": "budget_exceeded",
                    "tool": tool_name,
                },
                failure=failure,
                metadata={
                    "budget_resource": denial.resource.value,
                    "execution_budget": snapshot.model_dump(mode="json"),
                },
            )
            self._append_tool_call(tool_call)
            self._record_tool_since_model_call(tool_name)
            self.trace.record(
                "tool_call_budget_exceeded",
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                budget_resource=denial.resource,
                budget=snapshot,
                attempted=denial.attempted,
            )
            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "budget_exceeded",
                        "tool": tool_name,
                        "error": "evaluation_budget_exhausted",
                    },
                    sort_keys=True,
                ),
                tool_call_id=call_id,
                name=tool_name,
                status="error",
            )

        self.trace.record(
            "tool_call_started",
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        try:
            response = handler(request)
        except Exception as exc:
            duration = max(0.0, time.perf_counter() - started)
            finished_at = datetime.now(UTC)
            failure = _failure_from_exception(
                exc,
                default_type=_tool_failure_type(tool_name),
                stage=tool_name,
            )
            tool_call = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                status=ToolCallStatus.ERROR,
                result=None,
                failure=failure,
            )
            self._append_tool_call(tool_call)
            self._append_failure(failure)
            self._record_tool_since_model_call(tool_name)
            self.trace.record(
                "tool_call_failed",
                call_id=call_id,
                tool_name=tool_name,
                duration_seconds=duration,
                failure=failure,
            )
            raise

        duration = max(0.0, time.perf_counter() - started)
        finished_at = datetime.now(UTC)
        fragment = _ephemeral_tool_evidence(tool_name, response)
        if fragment:
            with self._lock:
                self._final_evidence_fragments.append(fragment)
        parsed_result = _tool_response_value(response)
        semantic_failure = _semantic_tool_failure(tool_name, parsed_result)
        if (
            semantic_failure is not None
            and semantic_failure.failure_type == FailureType.BUDGET_EXHAUSTED
            and semantic_failure.details.get("budget_resource")
            != BudgetResource.SUBQUESTION_SLICE.value
        ):
            raw_resource = semantic_failure.details.get("budget_resource")
            try:
                resource = BudgetResource(str(raw_resource))
            except ValueError:
                resource = (
                    BudgetResource.SEARCH
                    if _tool_failure_type(tool_name) == FailureType.SEARCH_ERROR
                    else BudgetResource.FETCH
                )
            snapshot = self.execution_budget.snapshot()
            with self._lock:
                self._budget_denied = True
                if self._budget_failure is None:
                    self._budget_failure = BudgetExceeded(
                        resource,
                        snapshot=snapshot,
                        attempted={
                            "tool_name": tool_name,
                            "semantic_result": sanitize_trace_value(parsed_result),
                        },
                    )
        if semantic_failure is None:
            status = ToolCallStatus.SUCCESS
        elif (
            semantic_failure.failure_type == FailureType.BUDGET_EXHAUSTED
            and semantic_failure.details.get("budget_resource")
            != BudgetResource.SUBQUESTION_SLICE.value
        ):
            status = ToolCallStatus.BUDGET_EXCEEDED
        else:
            status = ToolCallStatus.ERROR
        metadata = _tool_result_metadata(parsed_result)
        tool_call = ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            status=status,
            result=parsed_result,
            failure=semantic_failure,
            metadata=metadata,
        )
        self._append_tool_call(tool_call)
        self._record_tool_since_model_call(tool_name)
        if semantic_failure is not None:
            self._append_failure(semantic_failure)
        self.trace.record(
            "tool_call_finished",
            call_id=call_id,
            tool_name=tool_name,
            duration_seconds=duration,
            status=status,
            provider_status=metadata.get("provider_status"),
            result=parsed_result,
            failure=semantic_failure,
        )
        return response

    def _reserve_model_call(
        self,
        *,
        label: str,
        estimated_input_tokens: int,
        stage: ModelStage,
        active_subquestion_id: str | None,
        context_profile: Mapping[str, Any],
        original_context_profile: Mapping[str, Any] | None = None,
    ) -> ModelCallReservation:
        stage_output_cap = self.stage_output_cap(stage)
        token_reservation = estimated_input_tokens + stage_output_cap
        tools_since_previous_call = self._consume_tools_since_model_call()
        partition: PartitionReservation | None = None
        controller = self._token_controller
        if controller is not None:
            partition_result = controller.reserve(
                stage=stage,
                token_reservation=token_reservation,
                subquestion_id=active_subquestion_id,
            )
            if isinstance(partition_result, PartitionDenial):
                snapshot = self.execution_budget.snapshot()
                attempted = {
                    "token_reservation": token_reservation,
                    "available_tokens": partition_result.available_tokens,
                    "estimated_input_tokens": estimated_input_tokens,
                    "max_output_tokens": stage_output_cap,
                    "stage": stage,
                    "active_subquestion_id": active_subquestion_id,
                    "partition_bucket": partition_result.bucket,
                    "token_partition": partition_result.snapshot,
                }
                self.trace.record(
                    "model_call_partition_exceeded",
                    label=label,
                    stage=stage,
                    active_subquestion_id=active_subquestion_id,
                    budget_resource=BudgetResource.SUBQUESTION_SLICE,
                    attempted=attempted,
                    context=context_profile,
                    original_context=original_context_profile,
                    tools_since_previous_call=tools_since_previous_call,
                    budget=snapshot,
                    token_partition=partition_result.snapshot,
                )
                raise BudgetExceeded(
                    BudgetResource.SUBQUESTION_SLICE,
                    snapshot=snapshot,
                    attempted=attempted,
                )
            partition = partition_result
        try:
            reservation = self.execution_budget.require_model_call(
                token_reservation=token_reservation
            )
        except BudgetExceeded as exc:
            if partition is not None and controller is not None:
                controller.cancel(partition)
            with self._lock:
                self._budget_denied = True
                self._budget_failure = exc
            snapshot = exc.snapshot or self.execution_budget.snapshot()
            self.trace.record(
                "model_call_budget_exceeded",
                label=label,
                budget_resource=exc.resource,
                attempted={
                    **exc.attempted,
                    "estimated_input_tokens": estimated_input_tokens,
                    "max_output_tokens": stage_output_cap,
                    "stage": stage,
                    "active_subquestion_id": active_subquestion_id,
                },
                context=context_profile,
                original_context=original_context_profile,
                tools_since_previous_call=tools_since_previous_call,
                budget=snapshot,
                token_partition=self.token_partition_snapshot(),
            )
            raise
        else:
            self._model_call_sequence += 1
            call_sequence = self._model_call_sequence
            if partition is not None:
                self._partition_reservations[reservation.reservation_id] = partition
            self._reservation_context[reservation.reservation_id] = {
                "call_sequence": call_sequence,
                "stage": stage,
                "active_subquestion_id": active_subquestion_id,
                "stage_output_cap": stage_output_cap,
                "context": dict(context_profile),
                "original_context": (
                    dict(original_context_profile)
                    if original_context_profile is not None
                    else None
                ),
                "tools_since_previous_call": tools_since_previous_call,
            }
            self.trace.record(
                "model_call_started",
                label=label,
                call_sequence=call_sequence,
                stage=stage,
                active_subquestion_id=active_subquestion_id,
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=stage_output_cap,
                token_reservation=token_reservation,
                context=context_profile,
                original_context=original_context_profile,
                tools_since_previous_call=tools_since_previous_call,
                budget=self.execution_budget.snapshot(),
                token_partition=self.token_partition_snapshot(),
            )
            return reservation

    def _record_model_response(
        self,
        response: Any,
        *,
        label: str,
        reservation: ModelCallReservation,
    ) -> None:
        context = dict(self._reservation_context.get(reservation.reservation_id, {}))
        usage = _usage_from_response(response)
        if usage is None:
            settlement = self.execution_budget.settle_model_call(
                reservation,
                actual_tokens=None,
                charge_reservation_if_unknown=True,
            )
            self._settle_partition_reservation(
                reservation,
                actual_tokens=None,
                charge_reservation_if_unknown=True,
            )
            self.trace.record(
                "model_token_usage_unavailable",
                label=label,
                call_sequence=context.get("call_sequence"),
                stage=context.get("stage"),
                active_subquestion_id=context.get("active_subquestion_id"),
                settlement=settlement,
                budget=self.execution_budget.snapshot(),
                token_partition=self.token_partition_snapshot(),
            )
            self._reservation_context.pop(reservation.reservation_id, None)
            return
        self._merge_usage(usage)
        accounted_total = usage.get("total_tokens")
        if accounted_total is None:
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if input_tokens is not None and output_tokens is not None:
                accounted_total = input_tokens + output_tokens
        if accounted_total is None:
            settlement = self.execution_budget.settle_model_call(
                reservation,
                actual_tokens=None,
                charge_reservation_if_unknown=True,
            )
            self._settle_partition_reservation(
                reservation,
                actual_tokens=None,
                charge_reservation_if_unknown=True,
            )
            self.trace.record(
                "model_token_budget_unverifiable",
                label=label,
                call_sequence=context.get("call_sequence"),
                stage=context.get("stage"),
                active_subquestion_id=context.get("active_subquestion_id"),
                usage=usage,
                settlement=settlement,
                budget=self.execution_budget.snapshot(),
                token_partition=self.token_partition_snapshot(),
            )
            self._reservation_context.pop(reservation.reservation_id, None)
            return
        settlement = self.execution_budget.settle_model_call(
            reservation,
            actual_tokens=accounted_total,
        )
        self._settle_partition_reservation(
            reservation,
            actual_tokens=accounted_total,
            charge_reservation_if_unknown=False,
        )
        self.trace.record(
            "model_token_settled",
            label=label,
            call_sequence=context.get("call_sequence"),
            stage=context.get("stage"),
            active_subquestion_id=context.get("active_subquestion_id"),
            stage_output_cap=context.get("stage_output_cap"),
            context=context.get("context"),
            original_context=context.get("original_context"),
            tools_since_previous_call=context.get("tools_since_previous_call", []),
            usage=usage,
            settlement=settlement,
            budget=self.execution_budget.snapshot(),
            token_partition=self.token_partition_snapshot(),
        )
        self._reservation_context.pop(reservation.reservation_id, None)
        if not settlement.token_budget_exceeded:
            return
        with self._lock:
            self._budget_denied = True
        denial = BudgetExceeded(
            BudgetResource.TOKEN,
            snapshot=self.execution_budget.snapshot(),
            attempted={
                "label": label,
                "actual_tokens": accounted_total,
                "reserved_tokens": reservation.reserved_tokens,
            },
        )
        with self._lock:
            self._budget_failure = denial
        raise denial

    def _merge_usage(self, usage: Mapping[str, int | None]) -> None:
        with self._lock:
            for field_name, attribute in (
                ("input_tokens", "_input_tokens"),
                ("output_tokens", "_output_tokens"),
                ("total_tokens", "_total_tokens"),
                ("cached_input_tokens", "_cached_input_tokens"),
                ("reasoning_tokens", "_reasoning_tokens"),
            ):
                value = usage.get(field_name)
                if value is None:
                    continue
                self._known_usage_fields.add(field_name)
                setattr(self, attribute, int(getattr(self, attribute)) + value)

    def _reserve_partition(
        self,
        *,
        stage: ModelStage,
        token_reservation: int,
        active_subquestion_id: str | None,
    ) -> PartitionReservation | None:
        controller = self._token_controller
        if controller is None:
            return None
        result = controller.reserve(
            stage=stage,
            token_reservation=token_reservation,
            subquestion_id=active_subquestion_id,
        )
        if isinstance(result, PartitionDenial):
            raise ValueError(f"initial {stage} reservation exceeds token partition")
        return result

    def _cancel_partition_reservation(
        self,
        reservation: ModelCallReservation,
    ) -> None:
        partition = self._partition_reservations.pop(
            reservation.reservation_id,
            None,
        )
        if partition is not None and self._token_controller is not None:
            self._token_controller.cancel(partition)

    def _settle_partition_reservation(
        self,
        reservation: ModelCallReservation,
        *,
        actual_tokens: int | None,
        charge_reservation_if_unknown: bool,
    ) -> None:
        partition = self._partition_reservations.pop(
            reservation.reservation_id,
            None,
        )
        if partition is not None and self._token_controller is not None:
            self._token_controller.settle(
                partition,
                actual_tokens=actual_tokens,
                charge_reservation_if_unknown=charge_reservation_if_unknown,
            )

    def _record_tool_since_model_call(self, tool_name: str) -> None:
        with self._lock:
            self._tools_since_model_call.append(tool_name)

    def _consume_tools_since_model_call(self) -> list[str]:
        with self._lock:
            tools = list(self._tools_since_model_call)
            self._tools_since_model_call.clear()
            return tools

    def _append_tool_call(self, tool_call: ToolCall) -> None:
        with self._lock:
            self._tool_calls.append(tool_call)

    def _append_failure(self, failure: FailureDetail) -> None:
        with self._lock:
            self._failures.append(failure)


class _AccountedSummaryModel(BaseChatModel):
    """Route DeepAgents' out-of-stack summary calls through the shared budget."""

    wrapped_model: BaseChatModel
    evaluation_middleware: EvaluationMiddleware
    label: str = "deepagents.summarization"

    @property
    def _llm_type(self) -> str:
        return "evaluation-accounted-summary"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError("summary accounting requires invoke or ainvoke")

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        reservation = self.evaluation_middleware.reserve_external_model_call(
            label=self.label,
            request_payload=input,
            stage="control_status",
        )
        try:
            response = self.evaluation_middleware.model_for_stage(
                self.wrapped_model,
                "control_status",
            ).invoke(
                input,
                config=config,
                stop=stop,
                **kwargs,
            )
        except Exception:
            self.evaluation_middleware.cancel_external_model_call(
                reservation,
                label=self.label,
            )
            raise
        self.evaluation_middleware.record_external_model_response(
            response,
            label=self.label,
            reservation=reservation,
        )
        return response

    async def ainvoke(
        self,
        input: Any,
        config: Any | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        reservation = self.evaluation_middleware.reserve_external_model_call(
            label=self.label,
            request_payload=input,
            stage="control_status",
        )
        try:
            response = await self.evaluation_middleware.model_for_stage(
                self.wrapped_model,
                "control_status",
            ).ainvoke(
                input,
                config=config,
                stop=stop,
                **kwargs,
            )
        except Exception:
            self.evaluation_middleware.cancel_external_model_call(
                reservation,
                label=self.label,
            )
            raise
        self.evaluation_middleware.record_external_model_response(
            response,
            label=self.label,
            reservation=reservation,
        )
        return response


class EvaluationSummarizationMiddleware(SummarizationMiddleware):
    """Native DeepAgents compaction whose hidden model call is fully accounted."""

    @property
    def name(self) -> str:
        # Replace DeepAgents' built-in slot in both parent and subagent stacks.
        return "SummarizationMiddleware"

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        self._raise_if_budget_denied()
        summary = super()._create_summary(messages_to_summarize)
        self._raise_if_budget_denied()
        return _safe_summary_error(summary)

    async def _acreate_summary(
        self,
        messages_to_summarize: list[AnyMessage],
    ) -> str:
        self._raise_if_budget_denied()
        summary = await super()._acreate_summary(messages_to_summarize)
        self._raise_if_budget_denied()
        return _safe_summary_error(summary)

    def _raise_if_budget_denied(self) -> None:
        model = self.model
        if not isinstance(model, _AccountedSummaryModel):
            raise RuntimeError("evaluation summarizer lost its accounted model")
        middleware = model.evaluation_middleware
        if not middleware.budget_denied:
            return
        failure = middleware.budget_failure
        if failure is not None:
            raise failure
        snapshot = middleware.execution_budget.snapshot()
        raise BudgetExceeded(
            (
                BudgetResource.DEADLINE
                if snapshot.deadline_exceeded
                else BudgetResource.TOKEN
            ),
            snapshot=snapshot,
        )


def build_evaluation_summarization_middleware(
    model: BaseChatModel,
    backend: Any,
    evaluation_middleware: EvaluationMiddleware,
) -> EvaluationSummarizationMiddleware:
    """Build a drop-in replacement for DeepAgents' native summary middleware."""

    defaults = compute_summarization_defaults(model)
    accounted_model = _AccountedSummaryModel(
        wrapped_model=model,
        evaluation_middleware=evaluation_middleware,
        profile=model.profile,
    )
    return EvaluationSummarizationMiddleware(
        model=accounted_model,
        backend=backend,
        trigger=defaults["trigger"],
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
        truncate_args_settings=defaults["truncate_args_settings"],
    )


def _safe_summary_error(summary: str) -> str:
    if summary.startswith("Error generating summary:"):
        return "Error generating summary: provider call failed"
    return summary


def prepare_runtime(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    system_id: str,
    execution_budget: ExecutionBudget,
    trace: TraceCollector,
    injected_backend: FixtureBackend | None = None,
    injected_model: BaseChatModel | None = None,
    semantic_policy: EffortPolicy | None = None,
    semantic_strategy: SemanticStrategy = "fixed",
    enable_tongagent_evidence_state: bool = False,
    enable_tongagent_token_control: bool = False,
    enable_tongagent_context_compaction: bool = False,
    search_query_normalizer: Callable[[str], str] | None = None,
) -> PreparedRuntime:
    """Resolve model and raw providers, then install shared semantic wrappers."""

    if resolved_config.system_id != system_id:
        msg = f"{system_id} runner received config for {resolved_config.system_id}"
        raise ValueError(msg)

    fixture_backend: FixtureBackend | None = None
    if resolved_config.backend_kind == "fixture":
        fixture_backend = injected_backend or _load_fixture_backend(resolved_config)
        if fixture_backend.revision != resolved_config.fixture_revision:
            msg = (
                "fixture_revision does not match fixture content: "
                f"configured={resolved_config.fixture_revision!r} "
                f"actual={fixture_backend.revision!r}"
            )
            raise ValueError(msg)
        fixture_backend.reset()
        raw_tools = fixture_backend.as_tools()
        model = injected_model or FixtureChatModel.from_task(
            task,
            system_id=system_id,
        )
    else:
        if injected_backend is not None:
            msg = "A fixture backend cannot be injected into a live run"
            raise ValueError(msg)
        raw_tools = _live_raw_tools()
        model = injected_model or _live_model(resolved_config)

    middleware = EvaluationMiddleware(
        execution_budget,
        trace,
        max_output_tokens=resolved_config.model.max_output_tokens or 1,
        reserve_final_synthesis=resolved_config.backend_kind == "live",
        stage_output_caps=(
            DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS
            if enable_tongagent_token_control
            else None
        ),
        enable_context_compaction=enable_tongagent_context_compaction,
        enable_token_partitions=enable_tongagent_token_control,
    )
    semantic_tools, research_budget = _semantic_network_tools(
        resolved_config,
        raw_tools,
        policy=semantic_policy,
        strategy=semantic_strategy,
        enable_tongagent_evidence_state=enable_tongagent_evidence_state,
        external_guard=middleware.external_tool_guard,
        search_query_normalizer=search_query_normalizer,
    )
    trace.record(
        "runtime_prepared",
        system_id=system_id,
        backend_kind=resolved_config.backend_kind,
        fixture_revision=resolved_config.fixture_revision,
        model={
            "provider": resolved_config.model.provider,
            "name": resolved_config.model.name,
        },
        tools=[item.name for item in semantic_tools],
        semantic_strategy=semantic_strategy,
        semantic_policy=research_budget.policy,
        budget=resolved_config.budget,
    )
    return PreparedRuntime(
        model=cast("BaseChatModel", model),
        tools=tuple(semantic_tools),
        research_budget=research_budget,
        execution_budget=execution_budget,
        middleware=middleware,
        trace=trace,
        fixture_backend=fixture_backend,
    )


def run_graph_system(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    system_id: str,
    graph_factory: Callable[[PreparedRuntime, Path], Any],
    injected_backend: FixtureBackend | None = None,
    injected_model: BaseChatModel | None = None,
) -> RunResult:
    """Execute one graph adapter and return a complete canonical result."""

    artifact_directory = Path(resolved_config.artifact_directory).expanduser()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    trace = TraceCollector()
    execution_budget = ExecutionBudget(resolved_config.budget)
    run_id = f"{system_id}-{task.id}-{uuid.uuid4().hex}"
    runtime: PreparedRuntime | None = None
    native_output: Any = None
    caught: Exception | None = None
    trace.record(
        "run_started",
        run_id=run_id,
        task_id=task.id,
        system_id=system_id,
        config_fingerprint=resolved_config.config_fingerprint,
        fairness_fingerprint=resolved_config.fairness_fingerprint,
    )
    try:
        runtime = prepare_runtime(
            task,
            resolved_config,
            system_id=system_id,
            execution_budget=execution_budget,
            trace=trace,
            injected_backend=injected_backend,
            injected_model=injected_model,
        )
        graph = graph_factory(runtime, artifact_directory)
        native_output = graph.invoke(
            {"messages": [{"role": "user", "content": task.question}]},
            config={"recursion_limit": resolved_config.budget.recursion_limit},
        )
    except Exception as exc:  # Canonical failure conversion happens below.
        caught = exc
        trace.record(
            "run_exception",
            exception_type=type(exc).__name__,
            message=_safe_message(exc),
        )

    final_answer_override: str | None = None
    if runtime is not None and resolved_config.backend_kind == "live":
        draft = extract_final_answer(native_output)
        try:
            final_answer_override = runtime.middleware.finalize_answer(
                runtime.model,
                question=task.question,
                draft=draft,
            )
        except Exception as exc:
            if caught is None:
                caught = exc
                trace.record(
                    "run_exception",
                    exception_type=type(exc).__name__,
                    message=_safe_message(exc),
                    stage="final_synthesis",
                )
            final_answer_override = _attach_final_marker(draft, "ABSTAIN")

    finished_at = datetime.now(UTC)
    wall_time_seconds = max(0.0, time.perf_counter() - started)
    result = build_run_result(
        run_id=run_id,
        task=task,
        resolved_config=resolved_config,
        system_id=system_id,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time_seconds,
        runtime=runtime,
        execution_budget=execution_budget,
        trace=trace,
        native_output=native_output,
        caught=caught,
        final_answer_override=final_answer_override,
    )
    write_intermediate_artifacts(
        artifact_directory,
        task=task,
        resolved_config=resolved_config,
        runtime=runtime,
        execution_budget=execution_budget,
        trace=trace,
        native_output=native_output,
        result=result,
        caught=caught,
    )
    return result


def build_run_result(
    *,
    run_id: str,
    task: EvalTask,
    resolved_config: ResolvedConfig,
    system_id: str,
    started_at: datetime,
    finished_at: datetime,
    wall_time_seconds: float,
    runtime: PreparedRuntime | None,
    execution_budget: ExecutionBudget,
    trace: TraceCollector,
    native_output: Any,
    caught: Exception | None,
    evidence_count: int | None = None,
    structural_subquestion_coverage: float | None = None,
    final_answer_override: str | None = None,
) -> RunResult:
    """Build a truthful result from canonical runtime observations."""

    del trace
    final_answer = (
        final_answer_override
        if final_answer_override is not None
        else extract_final_answer(native_output)
    )
    tool_calls = runtime.middleware.tool_calls if runtime is not None else []
    token_usage = runtime.middleware.token_usage if runtime is not None else None
    research_snapshot = (
        runtime.research_budget.snapshot() if runtime is not None else {}
    )
    execution_snapshot = execution_budget.snapshot()
    completion_status, failure = _completion_and_failure(
        caught=caught,
        final_answer=final_answer,
        runtime=runtime,
        deadline_exceeded=execution_snapshot.deadline_exceeded,
    )
    budget_resource, budget_snapshot = _terminal_budget_observation(
        completion_status=completion_status,
        failure=failure,
        caught=caught,
        runtime=runtime,
        execution_snapshot=execution_snapshot,
    )
    citations = extract_citations(final_answer, tool_calls)
    return RunResult(
        run_id=run_id,
        task_id=task.id,
        system_id=system_id,
        git_sha=current_git_sha(),
        resolved_config=resolved_config,
        config_fingerprint=resolved_config.config_fingerprint,
        fairness_fingerprint=resolved_config.fairness_fingerprint,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time_seconds,
        final_answer=final_answer,
        citations=citations,
        tool_calls=tool_calls,
        external_retrieval_calls=execution_snapshot.external_retrieval_calls,
        internal_tool_calls=execution_snapshot.internal_tool_calls,
        search_calls=execution_snapshot.search_calls,
        fetch_calls=execution_snapshot.fetch_calls,
        relevant_searches=_nonnegative_metric(
            research_snapshot.get("relevant_searches"),
            upper_bound=execution_snapshot.search_calls,
        ),
        evidence_count=evidence_count,
        structural_subquestion_coverage=structural_subquestion_coverage,
        token_usage=token_usage,
        estimated_cost=None,
        completion_status=completion_status,
        failure_type=failure.failure_type if failure is not None else None,
        failure=failure,
        budget_resource=budget_resource,
        budget_snapshot=budget_snapshot,
        budget_accounting_version=2,
        artifact_directory=resolved_config.artifact_directory,
        fixture_smoke=resolved_config.backend_kind == "fixture",
        normalized_exact_match=normalized_exact_match(
            answer_for_exact_match(final_answer),
            task.reference_answer,
        ),
        judge_score=None,
    )


def extract_final_answer(native_output: Any) -> str | None:
    """Extract only a terminal model answer, never an intermediate tool call."""

    if native_output is None:
        return None
    if isinstance(native_output, str):
        return native_output.strip() or None
    if isinstance(native_output, Mapping):
        explicit = native_output.get("final_answer")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        structured = native_output.get("structured_response")
        if isinstance(structured, str) and structured.strip():
            return structured.strip()
        messages = native_output.get("messages")
        if isinstance(messages, Sequence) and not isinstance(
            messages, (str, bytes, bytearray)
        ):
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        continue
                    text = _message_text(message)
                    if text:
                        return text
                elif isinstance(message, Mapping):
                    role = str(message.get("role", message.get("type", "")))
                    if role not in {"assistant", "ai"} or message.get("tool_calls"):
                        continue
                    text = _content_text(message.get("content"))
                    if text:
                        return text
    return None


def extract_citations(
    final_answer: str | None,
    tool_calls: Sequence[ToolCall],
) -> list[Citation]:
    """Return only final-answer citations backed by successful fetch calls."""

    if not final_answer:
        return []
    fetched: list[dict[str, str | None]] = []
    for call in tool_calls:
        if call.tool_name not in {"fetch_url", "fetch", "open_page", "open_url"}:
            continue
        if call.status != ToolCallStatus.SUCCESS:
            continue
        result = call.result if isinstance(call.result, Mapping) else {}
        url = (
            result.get("url")
            or result.get("requested_url")
            or call.arguments.get("url")
        )
        if not isinstance(url, str) or not url:
            continue
        source_id = result.get("source_id")
        title = result.get("title")
        fetched.append(
            {
                "url": url,
                "source_id": source_id if isinstance(source_id, str) else None,
                "title": title if isinstance(title, str) else None,
            }
        )

    answer_urls = {
        item.rstrip(".,;:!?，。；：！？") for item in _URL.findall(final_answer)
    }
    answer_source_ids = set(_SOURCE_ID.findall(final_answer))
    citations: list[Citation] = []
    seen: set[tuple[str, str | None]] = set()
    for source in fetched:
        url = cast("str", source["url"])
        source_id = source["source_id"]
        if url not in answer_urls and source_id not in answer_source_ids:
            continue
        key = (url, source_id)
        if key in seen:
            continue
        seen.add(key)
        citation_id = source_id or f"CIT{len(citations) + 1}"
        citations.append(
            Citation(
                citation_id=citation_id,
                url=url,
                source_id=source_id,
                title=source["title"],
            )
        )
    return citations


@lru_cache(maxsize=1)
def current_git_sha() -> str:
    """Resolve the checkout commit without consulting a remote."""

    supplied = os.environ.get("TONGAGENT_GIT_SHA", "").strip().casefold()
    if _HEX_SHA.fullmatch(supplied):
        return supplied
    repo_root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip().casefold()
    return sha if result.returncode == 0 and _HEX_SHA.fullmatch(sha) else "unknown"


def _load_fixture_backend(resolved_config: ResolvedConfig) -> FixtureBackend:
    fixture_dir = resolved_config.system_options.get("fixture_dir")
    if not isinstance(fixture_dir, str) or not fixture_dir.strip():
        msg = "fixture runs require system_options.fixture_dir"
        raise ValueError(msg)
    fixture_path = Path(fixture_dir)
    if not fixture_path.is_absolute():
        fixture_path = Path(__file__).resolve().parents[2] / fixture_path
    return FixtureBackend.from_directory(fixture_path)


def _live_raw_tools() -> list[BaseTool]:
    # Kept lazy so importing the baseline contracts never initializes the
    # TongAgent application graph or any online client.
    from search_agent import create_retrieval_tools  # noqa: PLC0415

    return list(create_retrieval_tools())


def _live_model(resolved_config: ResolvedConfig) -> BaseChatModel:
    kwargs = dict(resolved_config.model.parameters)
    if resolved_config.model.temperature is not None:
        kwargs.setdefault("temperature", resolved_config.model.temperature)
    if resolved_config.model.max_output_tokens is not None:
        kwargs["max_tokens"] = resolved_config.model.max_output_tokens
    return cast(
        "BaseChatModel",
        init_chat_model(
            resolved_config.model.name,
            model_provider=resolved_config.model.provider,
            **kwargs,
        ),
    )


def _semantic_network_tools(
    resolved_config: ResolvedConfig,
    raw_tools: Sequence[BaseTool],
    *,
    policy: EffortPolicy | None = None,
    strategy: SemanticStrategy = "fixed",
    enable_tongagent_evidence_state: bool = False,
    external_guard: Callable[[str], Mapping[str, Any] | None] | None = None,
    search_query_normalizer: Callable[[str], str] | None = None,
) -> tuple[list[BaseTool], Any]:
    by_name = {item.name: item for item in raw_tools}
    raw_search = by_name.get("web_search")
    raw_fetch = by_name.get("fetch_url")
    if raw_search is None or raw_fetch is None:
        msg = "raw provider must expose web_search and fetch_url"
        raise ValueError(msg)

    # This is the only TongAgent application import needed by B1/B2: both
    # baselines deliberately reuse the production provider semantics instead
    # of copying search/fetch wrappers into the evaluation package.
    from search_agent import ResearchBudget, build_budgeted_tools  # noqa: PLC0415

    limits = resolved_config.budget
    if policy is None:
        policy = EffortPolicy(
            name="low",
            max_searches=limits.max_search_calls,
            max_fetches=limits.max_fetch_calls,
            min_successful_sources=0,
            max_results_per_search=limits.max_results_per_search,
            max_chars_per_page=limits.max_page_chars,
            max_output_tokens=resolved_config.model.max_output_tokens or 1,
            max_subquestions=1,
            require_reviewer=False,
        )
    elif (
        policy.max_searches != limits.max_search_calls
        or policy.max_fetches != limits.max_fetch_calls
        or policy.max_results_per_search != limits.max_results_per_search
        or policy.max_chars_per_page != limits.max_page_chars
    ):
        msg = "semantic policy network limits must equal the shared evaluation budget"
        raise ValueError(msg)
    budget = (
        None
        if enable_tongagent_evidence_state
        else ResearchBudget(
            policy,
            strategy=strategy,
            evidence_graph=_BaselineNoEvidenceState(),
        )
    )
    return build_budgeted_tools(
        policy,
        strategy=strategy,
        raw_search_tool=raw_search,
        raw_fetch_tool=raw_fetch,
        budget=budget,
        external_guard=external_guard,
        search_query_normalizer=search_query_normalizer,
    )


def _stage_for_external_label(label: str) -> ModelStage:
    normalized = label.casefold()
    if "planner" in normalized:
        return "planner"
    if "summar" in normalized:
        return "control_status"
    if "final" in normalized:
        return "final_extractor"
    return "control_status"


def _stage_for_model_request(request: ModelRequest[Any]) -> ModelStage:
    human_marker = ""
    for message in reversed(request.messages):
        if isinstance(message, HumanMessage):
            human_marker = str(message.id or "")
            break
    if human_marker.startswith("report-step-"):
        return "final_synthesis"
    last_tool = next(
        (
            str(message.name or "")
            for message in reversed(request.messages)
            if isinstance(message, ToolMessage)
        ),
        "",
    )
    if last_tool in {"fetch_url", "record_evidence"}:
        return "evidence_selection"
    if last_tool in {
        "get_research_plan",
        "get_source_ledger",
        "get_evidence_graph",
        "update_subquestion",
    }:
        return "control_status"
    return "research_step"


def _active_subquestion_id(request: ModelRequest[Any]) -> str | None:
    state = request.state
    if not isinstance(state, Mapping):
        return None
    value = state.get("active_subquestion_id")
    return str(value) if isinstance(value, str) and value else None


def _external_context_profile(payload: Any) -> dict[str, int]:
    chars = _stable_payload_chars(payload)
    message_count = (
        len(payload)
        if isinstance(payload, Sequence)
        and not isinstance(payload, (str, bytes, bytearray))
        else 1
    )
    return {
        "message_count": message_count,
        "input_chars": chars,
        "system_prompt_chars": 0,
        "plan_chars": 0,
        "event_chars": 0,
        "ledger_chars": 0,
        "evidence_graph_chars": 0,
        "page_content_chars": 0,
        "tool_message_chars": 0,
        "other_history_chars": chars,
        "tool_schema_chars": 0,
    }


def _model_context_profile(request: ModelRequest[Any]) -> dict[str, int]:
    system_prompt_chars = (
        _stable_payload_chars(request.system_message.content)
        if request.system_message is not None
        else 0
    )
    tool_message_chars = sum(
        _stable_payload_chars(message)
        for message in request.messages
        if isinstance(message, ToolMessage)
    )
    other_history_chars = sum(
        _stable_payload_chars(message)
        for message in request.messages
        if not isinstance(message, ToolMessage)
    )
    tool_schema_chars = sum(_stable_payload_chars(tool) for tool in request.tools)
    page_content_chars = sum(
        _fetch_tool_message_content_chars(message)
        for message in request.messages
        if isinstance(message, ToolMessage) and message.name == "fetch_url"
    )
    plan_chars = sum(
        _stable_payload_chars(message.content)
        for message in request.messages
        if isinstance(message, ToolMessage)
        and message.name in {"get_research_plan", "update_subquestion"}
    )
    ledger_chars = sum(
        _stable_payload_chars(message.content)
        for message in request.messages
        if isinstance(message, ToolMessage) and message.name == "get_source_ledger"
    )
    evidence_graph_chars = sum(
        _stable_payload_chars(message.content)
        for message in request.messages
        if isinstance(message, ToolMessage)
        and message.name in {"get_evidence_graph", "record_evidence"}
    )
    event_chars = 0
    for message in request.messages:
        if not isinstance(message, HumanMessage) or not isinstance(
            message.content, str
        ):
            continue
        plan_chars += _marked_section_chars(
            message.content,
            "PLAN:\n",
            "\n\nCANONICAL SOURCE LEDGER:",
        )
        ledger_chars += _marked_section_chars(
            message.content,
            "CANONICAL SOURCE LEDGER:\n",
            "\n\nCANONICAL EVIDENCE GRAPH:",
        )
        evidence_graph_chars += _marked_section_chars(
            message.content,
            "CANONICAL EVIDENCE GRAPH:\n",
            "\n\nALLOWED CAVEAT LINES:",
        )
        event_chars += _marked_section_chars(
            message.content,
            "RESEARCH EVENT SUMMARY:\n",
            "\n\n",
        )
    payload: list[Any] = []
    if request.system_message is not None:
        payload.append(request.system_message)
    payload.extend(request.messages)
    payload.extend(request.tools)
    if request.response_format is not None:
        payload.append(request.response_format)
    return {
        "message_count": len(request.messages),
        "input_chars": _stable_payload_chars(payload),
        "system_prompt_chars": system_prompt_chars,
        "plan_chars": plan_chars,
        "event_chars": event_chars,
        "ledger_chars": ledger_chars,
        "evidence_graph_chars": evidence_graph_chars,
        "page_content_chars": page_content_chars,
        "tool_message_chars": tool_message_chars,
        "other_history_chars": other_history_chars,
        "tool_schema_chars": tool_schema_chars,
    }


def _marked_section_chars(content: str, start: str, end: str) -> int:
    start_index = content.find(start)
    if start_index < 0:
        return 0
    value_start = start_index + len(start)
    end_index = content.find(end, value_start)
    if end_index < 0:
        end_index = len(content)
    return max(0, end_index - value_start)


def _compact_tongagent_model_request(
    request: ModelRequest[Any],
) -> ModelRequest[Any]:
    active_question = _active_subquestion_question(request.state)
    messages = list(request.messages)
    if not isinstance(request.model, FixtureChatModel):
        marker_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
                and (
                    str(messages[index].id or "").startswith("research-step-")
                    or str(messages[index].id or "").startswith("report-step-")
                )
            ),
            0,
        )
        messages = messages[marker_index:]
    compacted: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = _compact_tool_message_content(
                message.name or "",
                message.content,
                state=request.state,
                active_question=active_question,
            )
            compacted.append(message.model_copy(update={"content": content}))
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = []
            for call in message.tool_calls:
                args = dict(call.get("args", {}))
                for key, limit in (("quote", 500), ("claim", 500), ("content", 500)):
                    if isinstance(args.get(key), str) and len(args[key]) > limit:
                        args[key] = args[key][:limit]
                calls.append({**call, "args": args})
            compacted.append(message.model_copy(update={"tool_calls": calls}))
            continue
        compacted.append(message)
    return request.override(messages=compacted)


def _compact_tool_message_content(
    tool_name: str,
    content: Any,
    *,
    state: Mapping[str, Any],
    active_question: str,
) -> Any:
    if not isinstance(content, str):
        return content
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if not isinstance(payload, Mapping):
        return content
    if tool_name == "web_search":
        results = []
        for item in payload.get("results", [])[:5]:
            if not isinstance(item, Mapping):
                continue
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": str(item.get("snippet", ""))[
                        :_TONGAGENT_SEARCH_SNIPPET_CHARS
                    ],
                    "provider": item.get("provider"),
                    "provider_rank": item.get("provider_rank"),
                    "relevance_score": item.get("relevance_score"),
                    "relevance_tier": item.get("relevance_tier"),
                    "relevance_reason": item.get("relevance_reason"),
                }
            )
        compact = {
            "status": payload.get("status"),
            "query": payload.get("query"),
            "original_query": payload.get("original_query"),
            "normalized_query": payload.get("normalized_query"),
            "search_quality": payload.get("search_quality"),
            "provider_success": payload.get("provider_success"),
            "relevant_results": payload.get("relevant_results"),
            "uncertain_results": payload.get("uncertain_results"),
            "results": results,
        }
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if tool_name == "fetch_url":
        page = str(payload.get("content", ""))
        excerpt = _select_relevant_page_excerpt(page, active_question)
        compact = {
            key: payload.get(key)
            for key in (
                "status",
                "source_id",
                "title",
                "url",
                "original_url",
                "final_url",
                "acquisition_method",
                "content_provider",
                "evidence_quality",
                "failure_taxonomy",
            )
            if key in payload
        }
        compact.update(
            {
                "content": excerpt,
                "content_chars": len(excerpt),
                "original_content_chars": payload.get("content_chars", len(page)),
                "content_compacted_for_model": len(excerpt) < len(page),
            }
        )
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if tool_name == "get_research_plan":
        plan = payload
        if isinstance(plan, Mapping):
            active_id = state.get("active_subquestion_id")
            subquestions = [
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "status": item.get("status"),
                    "depends_on": item.get("depends_on", []),
                    "evidence_source_ids": item.get("evidence_source_ids", []),
                    "claim_ids": item.get("claim_ids", []),
                    "note": item.get("note", ""),
                    "active": item.get("id") == active_id,
                }
                for item in plan.get("subquestions", [])
                if isinstance(item, Mapping)
            ]
            return json.dumps(
                {
                    "plan_id": plan.get("plan_id"),
                    "objective": plan.get("objective"),
                    "status": plan.get("status"),
                    "structural_subquestion_coverage": plan.get(
                        "structural_subquestion_coverage"
                    ),
                    "subquestions": subquestions,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
    if tool_name == "get_source_ledger":
        ledger = payload
        if isinstance(ledger, Mapping):
            active_id = ledger.get(
                "active_subquestion_id",
                state.get("active_subquestion_id"),
            )
            sources = [
                {
                    "source_id": item.get("source_id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "evidence_quality": item.get("evidence_quality"),
                    "acquisition_method": item.get("acquisition_method"),
                }
                for item in ledger.get("successful_sources", [])
                if isinstance(item, Mapping)
            ]
            return json.dumps(
                {
                    "active_subquestion_id": active_id,
                    "successful_sources": sources,
                    "subquestion_limits": ledger.get("subquestion_limits", {}),
                    "subquestion_usage": ledger.get("subquestion_usage", {}),
                    "evidence_graph_counts": ledger.get(
                        "evidence_graph_counts",
                        {},
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
    if tool_name == "get_evidence_graph":
        ledger = payload
        if isinstance(ledger, Mapping):
            active_id = state.get("active_subquestion_id")
            claims = [
                item
                for item in ledger.get("claims", [])
                if isinstance(item, Mapping) and item.get("subquestion_id") == active_id
            ]
            claim_ids = {str(item.get("claim_id")) for item in claims}
            evidence = [
                item
                for item in ledger.get("evidence_units", [])
                if isinstance(item, Mapping) and str(item.get("claim_id")) in claim_ids
            ]
            conflicts = [
                item
                for item in ledger.get("conflicts", [])
                if isinstance(item, Mapping) and str(item.get("claim_id")) in claim_ids
            ]
            return json.dumps(
                {
                    "claims": claims,
                    "evidence_units": evidence,
                    "conflicts": conflicts,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
    return content


def _active_subquestion_question(state: Mapping[str, Any]) -> str:
    active_id = state.get("active_subquestion_id")
    plan = state.get("research_plan", {})
    if not isinstance(plan, Mapping):
        return ""
    for item in plan.get("subquestions", []):
        if isinstance(item, Mapping) and item.get("id") == active_id:
            return str(item.get("question", ""))
    return ""


def _select_relevant_page_excerpt(content: str, query: str) -> str:
    if len(content) <= _TONGAGENT_PAGE_CONTEXT_CHARS:
        return content
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", query)
        if token.casefold()
        not in {
            "the",
            "and",
            "for",
            "from",
            "that",
            "this",
            "what",
            "when",
            "which",
            "with",
            "into",
            "does",
            "were",
            "was",
        }
    }
    fragments = [
        item.strip()
        for item in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9])", content)
        if item.strip()
    ]
    ranked = sorted(
        enumerate(fragments),
        key=lambda pair: (
            -sum(term in pair[1].casefold() for term in terms),
            pair[0],
        ),
    )
    selected: list[tuple[int, str]] = []
    used = 0
    for index, fragment in ranked:
        if used >= _TONGAGENT_PAGE_CONTEXT_CHARS:
            break
        remaining = _TONGAGENT_PAGE_CONTEXT_CHARS - used
        piece = fragment[:remaining]
        if not piece:
            continue
        selected.append((index, piece))
        used += len(piece) + 1
    if not selected:
        return content[:_TONGAGENT_PAGE_CONTEXT_CHARS]
    return "\n".join(piece for _, piece in sorted(selected))


def _fetch_tool_message_content_chars(message: ToolMessage) -> int:
    if not isinstance(message.content, str):
        return 0
    try:
        payload = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return 0
    return len(str(payload.get("content", ""))) if isinstance(payload, Mapping) else 0


def _model_label(request: ModelRequest[Any]) -> str:
    model = request.model
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _estimate_model_request_input_tokens(request: ModelRequest[Any]) -> int:
    """Deterministically estimate prompt/tool-schema tokens before provider I/O."""

    payload: list[Any] = []
    if request.system_message is not None:
        payload.append(request.system_message)
    payload.extend(request.messages)
    payload.extend(request.tools)
    if request.response_format is not None:
        payload.append(request.response_format)
    # Four tokens of structural chat overhead per top-level request item.
    return max(1, (_stable_payload_chars(payload) + 3) // 4 + 4 * len(payload))


def _estimate_external_input_tokens(payload: Any) -> int:
    """Use the same deterministic estimate for planner calls outside middleware."""

    return max(1, (_stable_payload_chars(payload) + 3) // 4)


def _stable_payload_chars(value: Any) -> int:
    """Approximate serialized character count without unstable object reprs."""

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (bool, int, float)):
        return len(str(value))
    if isinstance(value, BaseMessage):
        return (
            len(type(value).__name__)
            + _stable_payload_chars(value.content)
            + _stable_payload_chars(getattr(value, "tool_calls", None))
        )
    if isinstance(value, BaseTool):
        schema: Any = getattr(value, "args", None)
        return (
            len(value.name)
            + len(value.description or "")
            + _stable_payload_chars(schema)
        )
    if isinstance(value, Mapping):
        return sum(
            len(str(key)) + _stable_payload_chars(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return sum(_stable_payload_chars(item) for item in value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _stable_payload_chars(model_dump(mode="json"))
        except (TypeError, ValueError):
            pass
    return len(type(value).__name__)


def _usage_from_response(response: Any) -> dict[str, int | None] | None:
    if isinstance(response, AIMessage):
        messages: Sequence[Any] = [response]
    else:
        candidate = getattr(response, "result", None)
        messages = candidate if isinstance(candidate, Sequence) else []
    aggregate: dict[str, int] = {}
    observed: set[str] = set()
    for message in messages:
        metadata = getattr(message, "usage_metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        details = metadata.get("input_token_details")
        output_details = metadata.get("output_token_details")
        values = {
            "input_tokens": _optional_nonnegative_int(metadata.get("input_tokens")),
            "output_tokens": _optional_nonnegative_int(metadata.get("output_tokens")),
            "total_tokens": _optional_nonnegative_int(metadata.get("total_tokens")),
            "cached_input_tokens": _optional_nonnegative_int(
                details.get("cache_read")
                if isinstance(details, Mapping)
                else metadata.get("cached_input_tokens")
            ),
            "reasoning_tokens": _optional_nonnegative_int(
                output_details.get("reasoning")
                if isinstance(output_details, Mapping)
                else metadata.get("reasoning_tokens")
            ),
        }
        for field_name, value in values.items():
            if value is None:
                continue
            observed.add(field_name)
            aggregate[field_name] = aggregate.get(field_name, 0) + value
    if not observed:
        return None
    return {
        field_name: aggregate.get(field_name) if field_name in observed else None
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        )
    }


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _sanitized_arguments(value: Any) -> dict[str, JsonValue]:
    sanitized = sanitize_trace_value(value)
    if isinstance(sanitized, dict):
        return sanitized
    return {"value": sanitized}


def _tool_response_value(response: ToolMessage | Command[Any]) -> JsonValue:
    if isinstance(response, ToolMessage):
        raw: Any = response.content
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                pass
    else:
        raw = response
    return sanitize_trace_value(raw)


def _ephemeral_tool_evidence(
    tool_name: str,
    response: ToolMessage | Command[Any],
) -> str:
    """Build bounded in-memory synthesis context without persisting page bodies."""

    if not isinstance(response, ToolMessage):
        return ""
    raw: Any = response.content
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ""
    if not isinstance(raw, Mapping):
        return ""
    status = str(raw.get("status", "success")).casefold()
    if status not in {"success", "ok", "completed"}:
        return ""
    normalized = tool_name.strip().casefold()
    pieces: list[str] = []
    if normalized in {"web_search", "search"}:
        results = raw.get("results")
        if not isinstance(results, Sequence) or isinstance(
            results,
            (str, bytes, bytearray),
        ):
            return ""
        for item in list(results)[:3]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title", "")).strip()[:300]
            snippet = str(item.get("snippet", "")).strip()[:800]
            url = str(item.get("url", "")).strip()[:1_000]
            if title or snippet:
                pieces.append(f"SEARCH: {title}\n{snippet}\n{url}".strip())
    elif normalized in {"fetch_url", "fetch", "open_page", "open_url"}:
        content = raw.get("content") or raw.get("page_content")
        if not isinstance(content, str) or not content.strip():
            return ""
        title = str(raw.get("title", "")).strip()[:300]
        url = str(raw.get("url") or raw.get("requested_url") or "").strip()[:1_000]
        pieces.append(f"FETCH: {title}\n{content.strip()[:2_500]}\n{url}".strip())
    if not pieces:
        return ""
    sanitized = sanitize_trace_value(
        "\n\n".join(pieces),
        max_text_chars=6_000,
    )
    return sanitized if isinstance(sanitized, str) else ""


def _semantic_tool_failure(
    tool_name: str,
    result: JsonValue,
) -> FailureDetail | None:
    if not isinstance(result, Mapping):
        return None
    raw_status = result.get("status")
    status = str(raw_status).casefold() if raw_status is not None else "success"
    if status in {"success", "ok", "completed"}:
        return None
    if status == "budget_exceeded":
        failure_type = FailureType.BUDGET_EXHAUSTED
    elif status == "fixture_not_found":
        failure_type = FailureType.FIXTURE_NOT_FOUND
    else:
        failure_type = _semantic_failure_type(tool_name, result)
    raw_message = result.get("error") or result.get("message") or status
    message = str(raw_message).strip() or status
    details: dict[str, JsonValue] = {"provider_status": status}
    reason = result.get("reason")
    if status == "budget_exceeded":
        declared_resource = result.get("budget_resource")
        if isinstance(declared_resource, str):
            try:
                details["budget_resource"] = BudgetResource(declared_resource).value
            except ValueError:
                declared_resource = None
        if not isinstance(declared_resource, str):
            details["budget_resource"] = (
                BudgetResource.SUBQUESTION_SLICE.value
                if reason in {"subquestion_budget_exceeded", "no_active_subquestion"}
                else (
                    BudgetResource.SEARCH.value
                    if _tool_failure_type(tool_name) == FailureType.SEARCH_ERROR
                    else BudgetResource.FETCH.value
                )
            )
        for key in (
            "active_subquestion_id",
            "budget_snapshot",
            "subquestion_limits",
            "subquestion_usage",
        ):
            if key in result:
                details[key] = cast("JsonValue", result[key])
    taxonomy = result.get("failure_taxonomy") or result.get("failure_type")
    if isinstance(taxonomy, str):
        details["failure_taxonomy"] = taxonomy
    return FailureDetail(
        failure_type=failure_type,
        message=message,
        stage=tool_name,
        retryable=bool(result.get("retryable", False)),
        details=details,
    )


def _semantic_failure_type(
    tool_name: str,
    result: Mapping[str, Any],
) -> FailureType:
    taxonomy = result.get("failure_taxonomy") or result.get("failure_type")
    normalized = str(taxonomy or "").strip().casefold()
    aliases = {
        "access_blocked": FailureType.ACCESS_BLOCKED,
        "http_403": FailureType.ACCESS_BLOCKED,
        "rate_limited": FailureType.RATE_LIMITED,
        "http_429": FailureType.RATE_LIMITED,
        "network_timeout": FailureType.NETWORK_TIMEOUT,
        "timeout": FailureType.NETWORK_TIMEOUT,
        "dns_rejected": FailureType.DNS_REJECTED,
        "dns_error": FailureType.DNS_REJECTED,
        "dns_rebinding": FailureType.SECURITY_REJECTED,
        "redirect_rejected": FailureType.SECURITY_REJECTED,
        "unsafe_url": FailureType.SECURITY_REJECTED,
        "security_rejected": FailureType.SECURITY_REJECTED,
        "ssrf_rejected": FailureType.SECURITY_REJECTED,
        "private_address": FailureType.SECURITY_REJECTED,
        "userinfo_rejected": FailureType.SECURITY_REJECTED,
    }
    return aliases.get(normalized, _tool_failure_type(tool_name))


def _tool_result_metadata(result: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(result, Mapping):
        return {}
    metadata: dict[str, JsonValue] = {}
    status = result.get("status")
    if isinstance(status, (str, int, float, bool)) or status is None:
        metadata["provider_status"] = status
    for key in (
        "provider_success",
        "provider_outcome",
        "nonempty_search",
        "relevant_search",
        "evidence_producing_search",
        "retryable",
        "attempt_id",
        "source_id",
        "budget_resource",
    ):
        value = result.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
    return metadata


def _tool_failure_type(tool_name: str) -> FailureType:
    normalized = tool_name.casefold()
    if normalized in {"web_search", "search"}:
        return FailureType.SEARCH_ERROR
    if normalized in {"fetch_url", "fetch", "open_page", "open_url"}:
        return FailureType.FETCH_ERROR
    return FailureType.TOOL_ERROR


def _is_external_retrieval_tool(tool_name: str) -> bool:
    return tool_name.strip().casefold() in {
        "web_search",
        "search",
        "fetch_url",
        "fetch",
        "open_page",
        "open_url",
    }


def _completion_and_failure(
    *,
    caught: Exception | None,
    final_answer: str | None,
    runtime: PreparedRuntime | None,
    deadline_exceeded: bool,
) -> tuple[CompletionStatus, FailureDetail | None]:
    if caught is not None:
        failure = (
            FailureDetail(
                failure_type=FailureType.DEADLINE_EXCEEDED,
                message=_safe_message(caught),
                stage="run",
                retryable=True,
                details={"exception_type": type(caught).__name__},
            )
            if isinstance(caught, TimeoutError) and deadline_exceeded
            else _failure_from_exception(
                caught,
                default_type=FailureType.RUNNER_ERROR,
                stage="run",
            )
        )
        if failure.failure_type == FailureType.BUDGET_EXHAUSTED:
            return CompletionStatus.BUDGET_EXHAUSTED, failure
        if failure.failure_type == FailureType.DEADLINE_EXCEEDED:
            return CompletionStatus.TIMED_OUT, failure
        return CompletionStatus.FAILED, failure
    if deadline_exceeded:
        return (
            CompletionStatus.TIMED_OUT,
            FailureDetail(
                failure_type=FailureType.DEADLINE_EXCEEDED,
                message="Evaluation wall-time budget was exceeded",
                stage="run",
                retryable=True,
            ),
        )
    if runtime is not None and runtime.middleware.budget_denied:
        denial = runtime.middleware.budget_failure
        if denial is not None:
            status = (
                CompletionStatus.TIMED_OUT
                if denial.resource == BudgetResource.DEADLINE
                else CompletionStatus.BUDGET_EXHAUSTED
            )
            return (
                status,
                _failure_from_exception(
                    denial,
                    default_type=FailureType.BUDGET_EXHAUSTED,
                    stage="run",
                ),
            )
        return (
            CompletionStatus.BUDGET_EXHAUSTED,
            FailureDetail(
                failure_type=FailureType.BUDGET_EXHAUSTED,
                message="Evaluation model, token, or tool budget was exhausted",
                stage="run",
                retryable=False,
            ),
        )
    if runtime is not None:
        fixture_failure = next(
            (
                item
                for item in runtime.middleware.failures
                if item.failure_type == FailureType.FIXTURE_NOT_FOUND
            ),
            None,
        )
        if fixture_failure is not None:
            return CompletionStatus.FAILED, fixture_failure
    if final_answer is None:
        return (
            CompletionStatus.FAILED,
            FailureDetail(
                failure_type=FailureType.INVALID_OUTPUT,
                message="Agent returned no terminal answer",
                stage="final_answer",
                retryable=False,
            ),
        )
    return CompletionStatus.COMPLETED, None


def _terminal_budget_observation(
    *,
    completion_status: CompletionStatus,
    failure: FailureDetail | None,
    caught: Exception | None,
    runtime: PreparedRuntime | None,
    execution_snapshot: BudgetSnapshot,
) -> tuple[BudgetResource | None, BudgetSnapshot | None]:
    """Return the exact denial captured when a terminal resource was refused."""

    if completion_status not in {
        CompletionStatus.BUDGET_EXHAUSTED,
        CompletionStatus.TIMED_OUT,
    }:
        return None, None
    if completion_status == CompletionStatus.TIMED_OUT:
        if (
            isinstance(caught, BudgetExceeded)
            and caught.resource == BudgetResource.DEADLINE
        ):
            return BudgetResource.DEADLINE, caught.snapshot or execution_snapshot
        return BudgetResource.DEADLINE, execution_snapshot
    if isinstance(caught, BudgetExceeded):
        return caught.resource, caught.snapshot or execution_snapshot
    if isinstance(caught, GraphRecursionError):
        return BudgetResource.RECURSION, execution_snapshot
    if runtime is not None and runtime.middleware.budget_failure is not None:
        denial = runtime.middleware.budget_failure
        return denial.resource, denial.snapshot or execution_snapshot
    if completion_status == CompletionStatus.BUDGET_EXHAUSTED and failure is not None:
        raw = failure.details.get("budget_resource")
        if isinstance(raw, str):
            try:
                return BudgetResource(raw), execution_snapshot
            except ValueError:
                pass
    return None, None


def _failure_from_exception(
    exc: Exception,
    *,
    default_type: FailureType,
    stage: str,
) -> FailureDetail:
    if isinstance(exc, BudgetExceeded):
        failure_type = (
            FailureType.DEADLINE_EXCEEDED
            if exc.resource == BudgetResource.DEADLINE
            else FailureType.BUDGET_EXHAUSTED
        )
        retryable = False
    elif isinstance(exc, GraphRecursionError):
        failure_type = FailureType.BUDGET_EXHAUSTED
        retryable = False
    elif isinstance(exc, TimeoutError):
        failure_type = default_type
        retryable = True
    elif type(exc).__name__ == "FixtureFormatError":
        failure_type = FailureType.INVALID_OUTPUT
        retryable = False
    else:
        failure_type = default_type
        retryable = False
    details: dict[str, JsonValue] = {"exception_type": type(exc).__name__}
    if isinstance(exc, BudgetExceeded):
        details["budget_resource"] = exc.resource.value
        if exc.snapshot is not None:
            details["budget_snapshot"] = cast(
                "JsonValue",
                exc.snapshot.model_dump(mode="json"),
            )
        if exc.attempted:
            details["attempted"] = cast(
                "JsonValue",
                sanitize_trace_value(exc.attempted),
            )
    elif isinstance(exc, GraphRecursionError):
        details["budget_resource"] = BudgetResource.RECURSION.value
    return FailureDetail(
        failure_type=failure_type,
        message=_safe_message(exc),
        stage=stage,
        retryable=retryable,
        details=details,
    )


def _safe_message(exc: BaseException) -> str:
    raw = str(exc).strip() or type(exc).__name__
    sanitized = sanitize_trace_value(raw, max_text_chars=1_000)
    return sanitized if isinstance(sanitized, str) else type(exc).__name__


def _nonnegative_metric(value: Any, *, upper_bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return min(max(0, value), max(0, upper_bound))


def _message_text(message: BaseMessage) -> str | None:
    return _content_text(message.content)


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(part.strip() for part in parts if part.strip())
        return joined or None
    return None


def _remove_final_answer_lines(value: str) -> str:
    return re.sub(
        r"(?im)^[ \t]*FINAL_ANSWER[ \t]*:[^\n\r]*(?:\r?\n)?",
        "",
        value,
    ).strip()


def _strict_final_marker(value: str | None) -> str:
    status, answer = extract_answer_contract(value)
    if status.value == "answer" and answer is not None:
        return f"FINAL_ANSWER: {answer}"
    return "FINAL_ANSWER: ABSTAIN"


def _attach_final_marker(draft: str | None, value: str) -> str:
    prefix = _remove_final_answer_lines(draft or "")
    marker = f"FINAL_ANSWER: {value}"
    return f"{prefix}\n\n{marker}".lstrip() if prefix else marker


def write_intermediate_artifacts(
    artifact_directory: Path,
    *,
    task: EvalTask,
    resolved_config: ResolvedConfig,
    runtime: PreparedRuntime | None,
    execution_budget: ExecutionBudget,
    trace: TraceCollector,
    native_output: Any,
    result: RunResult,
    caught: Exception | None,
) -> None:
    native_directory = artifact_directory / "native"
    research_budget = (
        runtime.research_budget.snapshot() if runtime is not None else None
    )
    native_payload = {
        "task_id": task.id,
        "system_id": resolved_config.system_id,
        "output": sanitize_trace_value(native_output),
        "exception": (
            None
            if caught is None
            else {
                "type": type(caught).__name__,
                "message": _safe_message(caught),
            }
        ),
    }
    budget_payload = {
        "execution": execution_budget.snapshot().model_dump(mode="json"),
        "research": sanitize_trace_value(research_budget),
    }
    tool_payload = [
        item.model_dump(mode="json", exclude_none=False) for item in result.tool_calls
    ]
    _atomic_write_json(
        native_directory / "task.json",
        task.model_dump(mode="json", exclude_none=False),
    )
    _atomic_write_json(
        native_directory / "resolved_config.json",
        resolved_config.model_dump(mode="json", exclude_none=False),
    )
    _atomic_write_json(native_directory / "native.json", native_payload)
    _atomic_write_json(native_directory / "budget.json", budget_payload)
    _atomic_write_json(native_directory / "tool_calls.json", tool_payload)
    _atomic_write_text(native_directory / "trace.jsonl", trace.jsonl())
    if result.final_answer is not None:
        answer = result.final_answer.rstrip() + "\n"
        _atomic_write_text(native_directory / "answer.md", answer)


def _atomic_write_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    _atomic_write_text(path, serialized + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
