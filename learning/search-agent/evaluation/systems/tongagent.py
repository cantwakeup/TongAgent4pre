"""B3: the production Stage 03D TongAgent workflow.

This adapter deliberately builds the application through
``search_agent.build_agent``.  It injects the same model, fixture/live raw
providers, semantic wrappers, execution budget, and trace middleware used by
the other baselines.  It never calls the application's ``ChatOpenAI``/``.env``
initialization path, and an offline fixture can never fall back to the network;
explicit live configurations are resolved only by the shared evaluation
runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_policy import EFFORT_POLICIES, EffortName, EffortPolicy
from evidence_graph import (
    allowed_report_caveat_lines,
    corroborating_evidence_source_ids,
    report_claim_mapping_errors,
    validate_evidence_graph,
)
from research_graph import (
    PlanDraft,
    create_research_plan,
    fallback_research_plan,
    invalid_covered_subquestions,
)
from search_agent import (
    AgentRuntimeDependencies,
    _adaptive_audit_errors,
    _build_tool_trace,
    _canonical_mapping_errors,
    _canonicalize_source_section,
    _report_finding_source_ids,
    _restore_checkpointed_report,
    _successful_final_report_write_position,
    build_agent,
)
from telemetry import write_event_log, write_plan_snapshot

from ..budget import BudgetExceeded, ExecutionBudget
from ..config import ResolvedConfig
from ..offline import FixtureBackend
from ..schema import (
    Citation,
    CompletionStatus,
    EvalTask,
    FailureDetail,
    FailureType,
    RunResult,
)
from ..tracing import TraceCollector, sanitize_trace_value
from .common import (
    PreparedRuntime,
    _attach_final_marker,
    build_evaluation_summarization_middleware,
    build_run_result,
    prepare_runtime,
    write_intermediate_artifacts,
)


SYSTEM_TONGAGENT = "tongagent"
_PLANNER_LABEL = "tongagent.planner"
_MODE = "single"
_STRATEGY = "adaptive"
_MAX_ESCALATIONS = 2
_DEFAULT_EFFORT: EffortName = "medium"


class TongAgentRunner:
    """Run B3 through the full production Stage 03D outer graph."""

    system_id = SYSTEM_TONGAGENT

    def __init__(
        self,
        *,
        fixture_backend: FixtureBackend | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self._fixture_backend = fixture_backend
        self._model = model

    def run(self, task: EvalTask, resolved_config: ResolvedConfig) -> RunResult:
        """Execute one isolated, checkpointed B3 attempt."""

        options, policy = _resolve_options(resolved_config)
        artifact_directory = Path(resolved_config.artifact_directory).expanduser()
        tongagent_directory = artifact_directory / "native" / "tongagent"
        tongagent_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tongagent_directory / "checkpoint.sqlite"
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        trace = TraceCollector()
        execution_budget = ExecutionBudget(resolved_config.budget)
        run_id = f"{self.system_id}-{task.id}-{uuid.uuid4().hex}"
        runtime: PreparedRuntime | None = None
        native_state: dict[str, Any] | None = None
        control_fingerprint = ""
        caught: Exception | None = None
        trace.record(
            "run_started",
            run_id=run_id,
            task_id=task.id,
            system_id=self.system_id,
            phase="runner",
            config_fingerprint=resolved_config.config_fingerprint,
            fairness_fingerprint=resolved_config.fairness_fingerprint,
        )

        try:
            runtime = prepare_runtime(
                task,
                resolved_config,
                system_id=self.system_id,
                execution_budget=execution_budget,
                trace=trace,
                injected_backend=self._fixture_backend,
                injected_model=self._model,
                semantic_policy=policy,
                semantic_strategy=_STRATEGY,
                enable_tongagent_evidence_state=True,
            )
            planner = _build_accounted_planner(
                task=task,
                runtime=runtime,
                fixture=resolved_config.backend_kind == "fixture",
            )
            summarization = build_evaluation_summarization_middleware(
                runtime.model,
                FilesystemBackend(
                    root_dir=tongagent_directory,
                    virtual_mode=True,
                ),
                runtime.middleware,
            )
            dependencies = AgentRuntimeDependencies(
                model=runtime.model,
                reviewer_model=runtime.model,
                network_tools=runtime.tools,
                budget=runtime.research_budget,
                planner=planner,
                middleware=(runtime.middleware, summarization),
            )
            thread_id = f"eval-{run_id}"
            with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
                bundle = build_agent(
                    output_dir=tongagent_directory,
                    model_name=resolved_config.model.name,
                    worker_model_name=resolved_config.model.name,
                    effort=cast("EffortName", options["effort"]),
                    mode=_MODE,
                    strategy=_STRATEGY,
                    max_escalations=_MAX_ESCALATIONS,
                    topic=task.question,
                    checkpointer=checkpointer,
                    runtime_dependencies=dependencies,
                )
                control_fingerprint = bundle.config_fingerprint
                trace.record(
                    "tongagent_graph_started",
                    phase="main",
                    topology=bundle.topology,
                    strategy=bundle.strategy,
                    effort=bundle.policy.name,
                    max_escalations=bundle.max_escalations,
                    checkpoint=str(checkpoint_path),
                )
                native_state = cast(
                    "dict[str, Any]",
                    bundle.agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": task.question,
                                }
                            ],
                            "research_topic": task.question,
                        },
                        config={
                            "configurable": {"thread_id": thread_id},
                            "recursion_limit": (resolved_config.budget.recursion_limit),
                        },
                    ),
                )
                trace.record(
                    "tongagent_graph_finished",
                    phase="main",
                    plan_status=_plan_value(native_state, "status"),
                    workflow_phase=native_state.get("workflow_phase"),
                )
        except Exception as exc:  # Canonical conversion is performed below.
            caught = exc
            trace.record(
                "run_exception",
                phase="runner",
                exception_type=type(exc).__name__,
                message=_safe_exception_message(exc),
            )

        ledger = runtime.research_budget.snapshot() if runtime is not None else {}
        plan = _native_plan(native_state)
        report_path = tongagent_directory / "report.md"
        if native_state is not None:
            _restore_checkpointed_report(report_path, native_state)
        report = (
            report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        )
        sources = _mapping_list(ledger.get("successful_sources"))
        if report:
            report, canonicalized = _canonicalize_source_section(report, sources)
            if canonicalized:
                report_path.write_text(report, encoding="utf-8")
        else:
            canonicalized = False

        fatal_errors, completeness_errors = _validate_native_run(
            plan=plan,
            report=report,
            native_state=native_state,
            ledger=ledger,
            policy=policy,
            model_name=resolved_config.model.name,
            control_fingerprint=control_fingerprint,
        )
        evidence_count = (
            len(_mapping_list(ledger.get("evidence_units")))
            if runtime is not None
            else None
        )
        structural_coverage = _structural_coverage(plan)
        canonical_output = {
            "final_answer": report or None,
            "tongagent_state": native_state,
        }
        final_answer_override: str | None = None
        if runtime is not None and resolved_config.backend_kind == "live":
            try:
                final_answer_override = runtime.middleware.finalize_answer(
                    runtime.model,
                    question=task.question,
                    draft=report or None,
                )
            except Exception as exc:
                if caught is None:
                    caught = exc
                    trace.record(
                        "run_exception",
                        phase="final_synthesis",
                        exception_type=type(exc).__name__,
                        message=_safe_exception_message(exc),
                    )
                final_answer_override = _attach_final_marker(report or None, "ABSTAIN")
        finished_at = datetime.now(UTC)
        wall_time_seconds = max(0.0, time.perf_counter() - started)
        base_result = build_run_result(
            run_id=run_id,
            task=task,
            resolved_config=resolved_config,
            system_id=self.system_id,
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=wall_time_seconds,
            runtime=runtime,
            execution_budget=execution_budget,
            trace=trace,
            native_output=canonical_output,
            caught=caught,
            evidence_count=evidence_count,
            structural_subquestion_coverage=structural_coverage,
            final_answer_override=final_answer_override,
        )
        result = _apply_native_completion(
            base_result,
            report=report,
            citations=_native_citations(report, ledger),
            plan=plan,
            fatal_errors=fatal_errors,
            completeness_errors=completeness_errors,
        )
        trace.record(
            "run_finished",
            phase="runner",
            completion_status=result.completion_status,
            fatal_errors=fatal_errors,
            completeness_errors=completeness_errors,
            evidence_count=evidence_count,
            structural_subquestion_coverage=structural_coverage,
        )
        write_intermediate_artifacts(
            artifact_directory,
            task=task,
            resolved_config=resolved_config,
            runtime=runtime,
            execution_budget=execution_budget,
            trace=trace,
            native_output=canonical_output,
            result=result,
            caught=caught,
        )
        _write_tongagent_artifacts(
            tongagent_directory,
            thread_id=f"eval-{run_id}",
            plan=plan,
            native_state=native_state,
            ledger=ledger,
            report=report,
            fatal_errors=fatal_errors,
            completeness_errors=completeness_errors,
            source_section_canonicalized=canonicalized,
        )
        return result


def _resolve_options(
    resolved_config: ResolvedConfig,
) -> tuple[dict[str, Any], EffortPolicy]:
    """Resolve the fixed B3 definition and reject hidden budget drift."""

    if resolved_config.system_id != SYSTEM_TONGAGENT:
        raise ValueError(
            f"tongagent runner received config for {resolved_config.system_id}"
        )
    raw_options = dict(resolved_config.system_options)
    effort = raw_options.get("effort", _DEFAULT_EFFORT)
    if effort not in EFFORT_POLICIES:
        raise ValueError(f"unsupported TongAgent effort: {effort!r}")
    mode = raw_options.get("mode", _MODE)
    strategy = raw_options.get("strategy", _STRATEGY)
    max_escalations = raw_options.get("max_escalations", _MAX_ESCALATIONS)
    if mode != _MODE:
        raise ValueError(
            "B3 main baseline requires mode=single; multi is a separately "
            "accounted future ablation"
        )
    if strategy != _STRATEGY:
        raise ValueError("B3 main baseline requires strategy=adaptive")
    if max_escalations != _MAX_ESCALATIONS:
        raise ValueError("B3 main baseline requires max_escalations=2")
    policy = EFFORT_POLICIES[cast("EffortName", effort)]
    return {
        **raw_options,
        "effort": effort,
        "mode": mode,
        "strategy": strategy,
        "max_escalations": max_escalations,
    }, policy


def _build_accounted_planner(
    *,
    task: EvalTask,
    runtime: PreparedRuntime,
    fixture: bool,
) -> Callable[[str, int], dict[str, Any]]:
    """Build the structured planner and account its out-of-graph model call."""

    def plan(topic: str, max_subquestions: int) -> dict[str, Any]:
        prompt = f"""Create an explicit web-research plan for the user question below.
Return between 1 and {max_subquestions} non-overlapping subquestions in dependency order.
Each subquestion must be independently researchable and materially necessary for the final answer.
Use dependencies for multi-hop questions. Do not include runtime status, source IDs, or invented facts.

User question: {topic}"""
        reservation = runtime.middleware.reserve_external_model_call(
            label=_PLANNER_LABEL,
            request_payload={
                "prompt": prompt,
                "response_schema": PlanDraft.model_json_schema(),
            },
        )
        try:
            structured = runtime.model.with_structured_output(
                PlanDraft,
                include_raw=True,
            )
            response = structured.invoke(prompt)
        except BudgetExceeded:
            runtime.middleware.cancel_external_model_call(
                reservation,
                label=_PLANNER_LABEL,
            )
            raise
        except Exception as exc:  # Match production's deterministic fallback.
            runtime.middleware.cancel_external_model_call(
                reservation,
                label=_PLANNER_LABEL,
            )
            runtime.trace.record(
                "planner_model_call_failed",
                phase="planner",
                label=_PLANNER_LABEL,
                exception_type=type(exc).__name__,
                message=_safe_exception_message(exc),
            )
            return cast(
                "dict[str, Any]",
                fallback_research_plan(
                    topic,
                    max_subquestions,
                    planner=f"deterministic-fallback:{type(exc).__name__}",
                ),
            )

        raw_response = (
            response.get("raw") if isinstance(response, Mapping) else response
        )
        runtime.middleware.record_external_model_response(
            raw_response,
            label=_PLANNER_LABEL,
            reservation=reservation,
        )
        try:
            parsed = (
                response.get("parsed") if isinstance(response, Mapping) else response
            )
            draft = (
                parsed
                if isinstance(parsed, PlanDraft)
                else PlanDraft.model_validate(parsed)
            )
        except Exception as exc:
            runtime.trace.record(
                "planner_output_invalid",
                phase="planner",
                label=_PLANNER_LABEL,
                exception_type=type(exc).__name__,
                message=_safe_exception_message(exc),
            )
            return cast(
                "dict[str, Any]",
                fallback_research_plan(
                    topic,
                    max_subquestions,
                    planner=f"deterministic-fallback:{type(exc).__name__}",
                ),
            )

        plan_id_factory: Callable[[], str] | None = None
        if fixture:
            digest = hashlib.sha256(f"{task.id}\0{topic}".encode("utf-8")).hexdigest()[
                :16
            ]

            def deterministic_plan_id() -> str:
                return f"fixture-plan-{digest}"

            plan_id_factory = deterministic_plan_id
        durable = create_research_plan(
            topic,
            draft.subquestions,
            objective=draft.objective,
            completion_criteria=draft.completion_criteria,
            planner="fixture-model" if fixture else "model",
            max_subquestions=max_subquestions,
            plan_id_factory=plan_id_factory,
        )
        runtime.trace.record(
            "planner_model_call_finished",
            phase="planner",
            label=_PLANNER_LABEL,
            planner=durable["planner"],
            plan_id=durable["plan_id"],
            subquestions=len(durable["subquestions"]),
            token_usage=runtime.middleware.token_usage,
        )
        return cast("dict[str, Any]", durable)

    return plan


def _validate_native_run(
    *,
    plan: dict[str, Any] | None,
    report: str,
    native_state: dict[str, Any] | None,
    ledger: dict[str, Any],
    policy: EffortPolicy,
    model_name: str,
    control_fingerprint: str,
) -> tuple[list[str], list[str]]:
    """Separate integrity/output failures from honest partial completion."""

    fatal: list[str] = []
    incomplete: list[str] = []
    if plan is None:
        fatal.append("missing durable research plan")
        return fatal, incomplete

    graph_errors = validate_evidence_graph(ledger)
    if graph_errors:
        fatal.extend(f"evidence graph: {item}" for item in graph_errors)
    invalid_covered = invalid_covered_subquestions(plan, ledger)
    for subquestion_id, errors in invalid_covered.items():
        fatal.append(
            f"invalid covered subquestion {subquestion_id}: " + "; ".join(errors)
        )
    adaptive_control = (
        dict(native_state.get("adaptive_control", {}))
        if native_state is not None
        else {}
    )
    fatal.extend(
        _adaptive_audit_errors(
            adaptive_control=adaptive_control,
            ledger=ledger,
            config_fingerprint=control_fingerprint,
            model_name=model_name,
            topology=_MODE,
            max_escalations=_MAX_ESCALATIONS,
        )
    )
    messages = _native_messages(native_state)
    native_trace = _build_tool_trace(messages)
    if _successful_final_report_write_position(native_trace) is None:
        fatal.append("missing successful report-phase write to /report.md")
    if not report:
        fatal.append("missing report.md")
        return list(dict.fromkeys(fatal)), incomplete

    claims = _mapping_list(ledger.get("claims"))
    evidence_units = _mapping_list(ledger.get("evidence_units"))
    plan_claim_ids = {
        str(claim_id)
        for item in _mapping_list(plan.get("subquestions"))
        for claim_id in _string_list(item.get("claim_ids"))
    }
    mapping_errors = report_claim_mapping_errors(
        report,
        plan_claim_ids=plan_claim_ids,
        claims=claims,
        evidence_units=evidence_units,
        allowed_caveat_lines=allowed_report_caveat_lines(
            plan,
            integrity_failure=bool(graph_errors),
        ),
    )
    for label, values in mapping_errors.items():
        if values:
            fatal.append(f"report {label}: {', '.join(values)}")

    cited_ids = set(_report_finding_source_ids(report))
    sources = _mapping_list(ledger.get("successful_sources"))
    cited_sources = [
        item for item in sources if str(item.get("source_id", "")) in cited_ids
    ]
    canonical_errors = _canonical_mapping_errors(report, cited_sources)
    for label, values in canonical_errors.items():
        if values:
            fatal.append(f"report source {label}: {', '.join(values)}")

    if plan.get("status") != "completed":
        incomplete.append(
            f"research plan is not completed: status={plan.get('status', 'missing')}"
        )
    if _structural_coverage(plan) != 1.0:
        incomplete.append(
            f"structural_subquestion_coverage is not 1.0: {_structural_coverage(plan)}"
        )
    required_searches = min(
        policy.max_searches,
        max(1, len(_mapping_list(plan.get("subquestions")))),
    )
    relevant_searches = ledger.get("relevant_searches")
    if not isinstance(relevant_searches, int) or isinstance(relevant_searches, bool):
        incomplete.append("relevant_searches is unavailable")
    elif relevant_searches < required_searches:
        incomplete.append(
            f"relevant_searches={relevant_searches} below {required_searches}"
        )
    corroborating_ids = corroborating_evidence_source_ids(
        source_ids=cited_ids,
        sources=sources,
        evidence_units=evidence_units,
        claim_ids=plan_claim_ids,
    )
    if len(corroborating_ids) < policy.min_successful_sources:
        incomplete.append(
            "report corroborating source groups="
            f"{len(corroborating_ids)} below {policy.min_successful_sources}"
        )
    return list(dict.fromkeys(fatal)), list(dict.fromkeys(incomplete))


def _apply_native_completion(
    base: RunResult,
    *,
    report: str,
    citations: list[Citation],
    plan: dict[str, Any] | None,
    fatal_errors: list[str],
    completeness_errors: list[str],
) -> RunResult:
    """Make plan/report integrity authoritative over a terminal chat message."""

    payload = base.model_dump(mode="python", exclude_none=False)
    # The canonical output may append a strict FINAL_ANSWER contract to the
    # native report. Keep that output while validating the native report
    # independently below.
    payload["citations"] = citations
    if base.completion_status != CompletionStatus.COMPLETED:
        return RunResult.model_validate(payload)
    if fatal_errors:
        failure = FailureDetail(
            failure_type=FailureType.INVALID_OUTPUT,
            message="TongAgent native output failed validation",
            stage="tongagent_validation",
            retryable=False,
            details={
                "errors": cast(
                    "list[str]",
                    sanitize_trace_value(fatal_errors),
                )
            },
        )
        payload.update(
            {
                "completion_status": CompletionStatus.FAILED,
                "failure_type": FailureType.INVALID_OUTPUT,
                "failure": failure,
            }
        )
    elif (
        plan is not None
        and plan.get("status") == "completed"
        and not completeness_errors
    ):
        payload.update(
            {
                "completion_status": CompletionStatus.COMPLETED,
                "failure_type": None,
                "failure": None,
            }
        )
    else:
        payload.update(
            {
                "completion_status": CompletionStatus.PARTIAL,
                "failure_type": None,
                "failure": None,
            }
        )
    return RunResult.model_validate(payload)


def _native_citations(
    report: str,
    ledger: Mapping[str, Any],
) -> list[Citation]:
    """Materialize cited Claim–Evidence–Source edges with exact quotes."""

    if not report:
        return []
    cited_source_ids = set(_report_finding_source_ids(report))
    report_claim_ids = {match for match in re.findall(r"\[(C[1-9][0-9]*)\]", report)}
    claims = {
        str(item.get("claim_id", "")): item
        for item in _mapping_list(ledger.get("claims"))
    }
    citations: list[Citation] = []
    for evidence in _mapping_list(ledger.get("evidence_units")):
        source_id = str(evidence.get("source_id", ""))
        claim_id = str(evidence.get("claim_id", ""))
        if source_id not in cited_source_ids or claim_id not in report_claim_ids:
            continue
        evidence_id = str(evidence.get("evidence_id", ""))
        url = str(evidence.get("url", ""))
        quote = str(evidence.get("quote", ""))
        if not evidence_id or not url or not quote:
            continue
        claim = claims.get(claim_id, {})
        citations.append(
            Citation(
                citation_id=evidence_id,
                url=url,
                source_id=source_id,
                title=str(evidence.get("title", "")) or None,
                quote=quote,
                claim=str(claim.get("text", "")) or None,
                metadata={
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "stance": str(evidence.get("stance", "")),
                },
            )
        )
    return citations


def _write_tongagent_artifacts(
    directory: Path,
    *,
    thread_id: str,
    plan: dict[str, Any] | None,
    native_state: dict[str, Any] | None,
    ledger: dict[str, Any],
    report: str,
    fatal_errors: list[str],
    completeness_errors: list[str],
    source_section_canonicalized: bool,
) -> None:
    """Persist the native Stage 03D state beside its SQLite checkpoint."""

    if plan is not None:
        write_plan_snapshot(
            directory / "plan.json",
            thread_id=thread_id,
            plan=cast("Any", plan),
            budget=ledger,
        )
    events = (
        list(native_state.get("research_events", []))
        if native_state is not None
        else []
    )
    write_event_log(directory / "events.jsonl", cast("Any", events))
    control = (
        dict(native_state.get("adaptive_control", {}))
        if native_state is not None
        else {}
    )
    _atomic_json(directory / "control.json", control)
    _atomic_json(directory / "sources.json", ledger)
    _atomic_json(
        directory / "evidence.json",
        {
            "evidence_graph_version": ledger.get("evidence_graph_version"),
            "sources": _mapping_list(ledger.get("successful_sources")),
            "claims": _mapping_list(ledger.get("claims")),
            "evidence_units": _mapping_list(ledger.get("evidence_units")),
            "conflicts": _mapping_list(ledger.get("conflicts")),
            "integrity_errors": validate_evidence_graph(ledger),
        },
    )
    _atomic_json(
        directory / "validation.json",
        {
            "status": "failed" if fatal_errors else "passed",
            "fatal_errors": fatal_errors,
            "completeness_errors": completeness_errors,
            "source_section_canonicalized": source_section_canonicalized,
        },
    )
    if report:
        _atomic_text(directory / "report.md", report.rstrip() + "\n")


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _native_plan(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    value = state.get("research_plan")
    return dict(value) if isinstance(value, Mapping) else None


def _native_messages(state: Mapping[str, Any] | None) -> list[BaseMessage]:
    if state is None:
        return []
    messages = state.get("messages")
    if not isinstance(messages, Sequence) or isinstance(
        messages,
        (str, bytes, bytearray),
    ):
        return []
    return [item for item in messages if isinstance(item, BaseMessage)]


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return []
    return [str(item) for item in value]


def _structural_coverage(plan: Mapping[str, Any] | None) -> float | None:
    if plan is None:
        return None
    value = plan.get("structural_subquestion_coverage")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _plan_value(state: Mapping[str, Any], key: str) -> Any:
    plan = state.get("research_plan")
    return plan.get(key) if isinstance(plan, Mapping) else None


def _safe_exception_message(exc: BaseException) -> str:
    value = sanitize_trace_value(str(exc).strip() or type(exc).__name__)
    return value if isinstance(value, str) else type(exc).__name__


__all__ = ["SYSTEM_TONGAGENT", "TongAgentRunner"]
