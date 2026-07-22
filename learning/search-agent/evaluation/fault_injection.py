"""Controlled, offline fault-injection evaluation for the transparent harness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from .config import (
    BudgetLimits,
    EvaluationModelConfig,
    ResolvedConfig,
    SharedToolConfig,
)
from .execution import atomic_write_json, persist_terminal_result
from .offline import FixtureBackend, FixtureChatModel
from .schema import CompletionStatus, EvalTask, RunResult
from .systems import BareSimpleReactRunner, TongAgentStandardRunner
from .systems.common import run_graph_system
from .systems.transparent_react import build_transparent_react_graph


FAULTS = (
    "fetch_first_attempt_failure",
    "temporary_model_connection_failure",
    "process_interruption_and_resume",
    "near_budget_exhaustion",
)
SYSTEMS = ("bare_simple_react", "tongagent_standard")


@dataclass
class _FailureCounter:
    remaining: int
    lock: Any = field(default_factory=Lock)

    def consume(self) -> bool:
        with self.lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True


class _TransientConnectionModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: BaseChatModel
    counter: _FailureCounter = Field(exclude=True)

    @property
    def _llm_type(self) -> str:
        return "fault-injected-connection-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self.model_copy(
            update={
                "wrapped": self.wrapped.bind_tools(
                    tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            },
            deep=False,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        if self.counter.consume():
            raise ConnectionError("temporary injected model connection failure")
        message = self.wrapped.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])


def run_fault_injection(
    *,
    manifest_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run all preregistered controlled cases without network access."""

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("fault manifest cases must be a list")
    counts = Counter(str(case.get("fault")) for case in cases if isinstance(case, dict))
    if counts != Counter({fault: 2 for fault in FAULTS}):
        raise ValueError("fault manifest must contain exactly two cases per fault")

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("fault case must be an object")
        for system_id in SYSTEMS:
            rows.append(_run_case(case, system_id=system_id, output=output))

    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["task_id"], {})[row["system_id"]] = row
    for compared in by_case.values():
        bare = compared["bare_simple_react"]
        standard = compared["tongagent_standard"]
        standard["additional_tokens"] = _numeric_delta(
            standard.get("total_tokens"), bare.get("total_tokens")
        )
        standard["additional_wall_time"] = _numeric_delta(
            standard.get("wall_time_seconds"), bare.get("wall_time_seconds")
        )
        bare["additional_tokens"] = 0
        bare["additional_wall_time"] = 0.0

    standard_rows = [row for row in rows if row["system_id"] == "tongagent_standard"]
    bare_rows = [row for row in rows if row["system_id"] == "bare_simple_react"]
    summary = {
        "schema_version": 1,
        "manifest": manifest_file.name,
        "run_count": len(rows),
        "rows": rows,
        "systems": {
            "bare_simple_react": _summarize(bare_rows),
            "tongagent_standard": _summarize(standard_rows),
        },
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def _run_case(
    case: Mapping[str, Any],
    *,
    system_id: str,
    output: Path,
) -> dict[str, Any]:
    task_id = str(case["task_id"])
    fault = str(case["fault"])
    answer = str(case["answer"])
    task = _task(task_id, answer=answer, fault=fault)
    backend = _backend(task_id, answer=answer, fault=fault)
    attempt = output / system_id / task_id / "attempt-0001"
    config = _config(
        system_id=system_id,
        backend=backend,
        artifact_directory=attempt,
        near_budget=fault == "near_budget_exhaustion",
    )

    checkpoint_restored = False
    provider_search_calls = 0
    provider_fetch_calls = 0
    if fault == "process_interruption_and_resume":
        if system_id == "tongagent_standard":
            TongAgentStandardRunner(
                fixture_backend=backend,
                interrupt_after_tools=True,
            ).run(task, config)
            result = TongAgentStandardRunner(fixture_backend=backend).run(task, config)
            checkpoint_restored = _checkpoint_restored(attempt)
            provider_fetch_calls = int(result.fetch_calls or 0)
        else:
            interrupted_directory = attempt / "interrupted"
            interrupted_config = config.model_copy(
                update={"artifact_directory": str(interrupted_directory)}
            )
            interrupted_config = ResolvedConfig.model_validate(
                interrupted_config.model_dump(
                    exclude={"config_fingerprint", "fairness_fingerprint"}
                )
            )
            interrupted = run_graph_system(
                task,
                interrupted_config,
                system_id=system_id,
                graph_factory=lambda runtime, _: build_transparent_react_graph(
                    runtime,
                    interrupt_after_tools=True,
                    name="fault-injected-bare-interruption",
                ),
                injected_backend=backend,
                finalize_live_answer=False,
            )
            restart_directory = attempt / "restart"
            restart_config = config.model_copy(
                update={"artifact_directory": str(restart_directory)}
            )
            restart_config = ResolvedConfig.model_validate(
                restart_config.model_dump(
                    exclude={"config_fingerprint", "fairness_fingerprint"}
                )
            )
            result = BareSimpleReactRunner(fixture_backend=backend).run(
                task,
                restart_config,
            )
            provider_fetch_calls = int(interrupted.fetch_calls or 0) + int(
                result.fetch_calls or 0
            )
            attempt = restart_directory
    else:
        model: BaseChatModel | None = None
        if fault == "temporary_model_connection_failure":
            fixture_model = FixtureChatModel.from_task(task, system_id=system_id)
            model = _TransientConnectionModel(
                wrapped=fixture_model,
                counter=_FailureCounter(1),
                profile=fixture_model.profile,
            )
        runner = (
            TongAgentStandardRunner(fixture_backend=backend, model=model)
            if system_id == "tongagent_standard"
            else BareSimpleReactRunner(fixture_backend=backend, model=model)
        )
        result = runner.run(task, config)
        provider_search_calls = int(result.search_calls or 0)
        provider_fetch_calls = int(result.fetch_calls or 0)

    persist_terminal_result(attempt, result)
    valid_terminal = (
        result.completion_status == CompletionStatus.COMPLETED
        and result.answer_rate is True
    )
    recovery_success = _recovery_success(
        fault,
        result=result,
        checkpoint_restored=checkpoint_restored,
    )
    row = {
        "task_id": task_id,
        "fault": fault,
        "system_id": system_id,
        "recovery_success": recovery_success,
        "valid_terminal_result": valid_terminal,
        "duplicate_search_calls": max(
            0,
            provider_search_calls - _unique_tool_calls(result, "web_search"),
        ),
        "duplicate_fetch_calls": max(
            0,
            provider_fetch_calls - _unique_tool_calls(result, "fetch_url"),
        ),
        "checkpoint_restored": checkpoint_restored,
        "trace_complete": _trace_complete(attempt, result),
        "total_tokens": (
            result.token_usage.total_tokens if result.token_usage is not None else None
        ),
        "wall_time_seconds": result.wall_time_seconds,
        "additional_tokens": None,
        "additional_wall_time": None,
        "completion_status": result.completion_status.value,
        "failure_type": (
            result.failure_type.value if result.failure_type is not None else None
        ),
        "search_calls": provider_search_calls,
        "fetch_calls": provider_fetch_calls,
        "artifact_directory": str(attempt),
    }
    return row


def _task(task_id: str, *, answer: str, fault: str) -> EvalTask:
    url = f"https://fixture.test/{task_id}"
    query = f"fixture {task_id}"
    script: list[dict[str, Any]] = []
    if fault in {"fetch_first_attempt_failure", "near_budget_exhaustion"}:
        script.append({"tool": "web_search", "args": {"query": query}})
    if fault != "temporary_model_connection_failure":
        script.append({"tool": "fetch_url", "args": {"url": url}})
    return EvalTask(
        id=task_id,
        question=f"Which city is stated by the controlled source for {task_id}?",
        reference_answer=answer,
        metadata={
            "answer": f"FINAL_ANSWER: {answer}",
            "research_script": script,
            "strict_fixture_tools": True,
        },
    )


def _backend(task_id: str, *, answer: str, fault: str) -> FixtureBackend:
    url = f"https://fixture.test/{task_id}"
    query = f"fixture {task_id}"
    success = {
        "status": "success",
        "title": f"Controlled source for {task_id}",
        "content": f"The stated city is {answer}. " * 30,
    }
    page: Any = success
    if fault == "fetch_first_attempt_failure":
        page = {
            "responses": [
                {
                    "status": "error",
                    "url": url,
                    "error": "temporary injected fetch failure",
                    "retryable": True,
                },
                success,
            ]
        }
    return FixtureBackend(
        searches={
            query: {
                "status": "success",
                "results": [
                    {
                        "title": f"Controlled source for {task_id}",
                        "url": url,
                        "snippet": f"The stated city is {answer}.",
                        "relevance_score": 100,
                    }
                ],
            }
        },
        pages={url: page},
    )


def _config(
    *,
    system_id: str,
    backend: FixtureBackend,
    artifact_directory: Path,
    near_budget: bool,
) -> ResolvedConfig:
    return ResolvedConfig(
        system_id=system_id,
        dataset_digest="sha256:final-fault-injection-v1",
        backend_kind="fixture",
        fixture_revision=backend.revision,
        model=EvaluationModelConfig(
            provider="fixture",
            name="fixture-chat-model",
            temperature=0.0,
            max_output_tokens=128,
        ),
        tools=SharedToolConfig(
            search_backend="fixture",
            fetch_backend="fixture",
        ),
        budget=BudgetLimits(
            max_search_calls=1,
            max_fetch_calls=2,
            max_total_tool_calls=3,
            max_model_calls=3 if near_budget else 8,
            max_total_tokens=20_000,
            wall_time_seconds=30.0,
            max_results_per_search=3,
            max_page_chars=2_000,
        ),
        runtime_mode="tongagent_standard",
        seed=17,
        system_options={"fixture_dir": "evaluation/fixtures"},
        artifact_directory=str(artifact_directory),
    )


def _checkpoint_restored(attempt: Path) -> bool:
    path = attempt / "native" / "checkpoint_manifest.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("checkpoint_restored") is True


def _recovery_success(
    fault: str,
    *,
    result: RunResult,
    checkpoint_restored: bool,
) -> bool:
    if fault == "fetch_first_attempt_failure":
        return any(
            call.tool_name == "fetch_url" and call.status.value == "success"
            for call in result.tool_calls
        )
    if (
        fault == "process_interruption_and_resume"
        and result.system_id == "tongagent_standard"
    ):
        return result.answer_rate is True and checkpoint_restored
    return result.answer_rate is True


def _unique_tool_calls(result: RunResult, tool_name: str) -> int:
    keys = {
        json.dumps(call.arguments, sort_keys=True)
        for call in result.tool_calls
        if call.tool_name == tool_name
    }
    return len(keys)


def _trace_complete(attempt: Path, result: RunResult) -> bool:
    trace = attempt / "native" / "trace.jsonl"
    if not trace.is_file():
        return False
    text = trace.read_text(encoding="utf-8")
    required_tools = len(result.tool_calls)
    return (
        "run_started" in text
        and "model_call_finished" in text
        and text.count("tool_call_finished") >= required_tools
    )


def _numeric_delta(value: Any, baseline: Any) -> float | int | None:
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return value - baseline
    return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "runs": count,
        "recovery_success_rate": (
            sum(row["recovery_success"] is True for row in rows) / count
        ),
        "valid_terminal_rate": (
            sum(row["valid_terminal_result"] is True for row in rows) / count
        ),
        "trace_completeness_rate": (
            sum(row["trace_complete"] is True for row in rows) / count
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_fault_injection(
        manifest_path=args.manifest,
        output_directory=args.output,
    )
    print(json.dumps(summary["systems"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
