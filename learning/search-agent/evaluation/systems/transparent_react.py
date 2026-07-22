"""One ReAct policy exposed bare and with TongAgent's transparent harness."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from ..config import ResolvedConfig
from ..execution import atomic_write_json, atomic_write_text
from ..offline import FixtureBackend
from ..schema import EvalTask, RunResult, TokenUsage, ToolCall
from .common import (
    SYSTEM_BARE_SIMPLE_REACT,
    SYSTEM_TONGAGENT_STANDARD,
    PreparedRuntime,
    run_graph_system,
)


TRANSPARENT_REACT_SYSTEM_PROMPT = """You are a web research assistant.

Use web_search to locate relevant results and fetch_url to inspect pages. Work
iteratively until you can answer the user's question or the shared budget is
exhausted. Base the answer on fetched page text; do not invent facts or URLs.

When finished, output exactly one line and nothing else:
FINAL_ANSWER: <short answer>

Use FINAL_ANSWER: ABSTAIN only when the available research cannot support an
answer.
"""


class _CheckpointedGraph:
    """Own one SQLite connection for a checkpointed graph invocation."""

    def __init__(
        self,
        runtime: PreparedRuntime,
        artifact_directory: Path,
        *,
        interrupt_after_tools: bool,
    ) -> None:
        native = artifact_directory / "native"
        native.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = native / "checkpoint.sqlite"
        self.restoring = self.checkpoint_path.is_file() and (
            self.checkpoint_path.stat().st_size > 0
        )
        self._connection = sqlite3.connect(
            self.checkpoint_path,
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(self._connection)
        self._graph = build_transparent_react_graph(
            runtime,
            checkpointer=checkpointer,
            interrupt_after_tools=interrupt_after_tools,
            name="evaluation-tongagent-standard",
        )
        self._manifest_path = native / "checkpoint_manifest.json"

    def invoke(self, value: Any, *, config: Mapping[str, Any]) -> Any:
        """Resume an existing checkpoint or start one new policy trajectory."""

        try:
            output = self._graph.invoke(
                None if self.restoring else value, config=config
            )
        finally:
            self._connection.close()
            atomic_write_json(
                self._manifest_path,
                {
                    "schema_version": 1,
                    "checkpoint_file": self.checkpoint_path.name,
                    "checkpoint_restored": self.restoring,
                    "thread_id": str(
                        dict(config.get("configurable", {})).get("thread_id", "")
                    ),
                },
                overwrite=True,
            )
        return output


def build_transparent_react_graph(
    runtime: PreparedRuntime,
    *,
    checkpointer: Any = None,
    interrupt_after_tools: bool = False,
    name: str = "evaluation-transparent-react",
) -> Any:
    """Build the exact policy graph shared by Bare and TongAgent Standard."""

    return create_agent(
        model=runtime.model,
        tools=list(runtime.tools),
        system_prompt=TRANSPARENT_REACT_SYSTEM_PROMPT,
        middleware=[runtime.middleware],
        checkpointer=checkpointer,
        interrupt_after=["tools"] if interrupt_after_tools else None,
        name=name,
    )


class BareSimpleReactRunner:
    """Run the frozen policy without checkpoint, retry, trace audit, or rewriting."""

    system_id = SYSTEM_BARE_SIMPLE_REACT

    def __init__(
        self,
        *,
        fixture_backend: FixtureBackend | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self._fixture_backend = fixture_backend
        self._model = model

    def run(self, task: EvalTask, resolved_config: ResolvedConfig) -> RunResult:
        """Execute one bare policy trajectory."""

        return run_graph_system(
            task,
            resolved_config,
            system_id=self.system_id,
            graph_factory=self._build_graph,
            injected_backend=self._fixture_backend,
            injected_model=self._model,
            finalize_live_answer=False,
        )

    @staticmethod
    def _build_graph(runtime: PreparedRuntime, artifact_directory: Path) -> Any:
        del artifact_directory
        return build_transparent_react_graph(
            runtime,
            name="evaluation-bare-simple-react",
        )


class TongAgentStandardRunner:
    """Run the same policy with non-invasive checkpoint, retry, and audit services."""

    system_id = SYSTEM_TONGAGENT_STANDARD

    def __init__(
        self,
        *,
        fixture_backend: FixtureBackend | None = None,
        model: BaseChatModel | None = None,
        interrupt_after_tools: bool = False,
    ) -> None:
        self._fixture_backend = fixture_backend
        self._model = model
        self._interrupt_after_tools = interrupt_after_tools

    def run(self, task: EvalTask, resolved_config: ResolvedConfig) -> RunResult:
        """Execute one checkpointed trajectory and write a post-hoc audit."""

        thread_id = _checkpoint_thread_id(task, resolved_config)
        artifact_directory = Path(resolved_config.artifact_directory)
        native = artifact_directory / "native"
        prior_trace = _read_text_if_present(native / "trace.jsonl")
        prior_state = _read_resume_state(native / "standard_resume_state.json")
        result = run_graph_system(
            task,
            resolved_config,
            system_id=self.system_id,
            graph_factory=self._build_graph,
            injected_backend=self._fixture_backend,
            injected_model=self._model,
            finalize_live_answer=False,
            max_model_connection_retries=1,
            max_retrieval_connection_retries=1,
            graph_invoke_config={"configurable": {"thread_id": thread_id}},
        )
        result = _merge_resumed_result(result, prior_state)
        current_trace = _read_text_if_present(native / "trace.jsonl")
        if prior_trace:
            atomic_write_text(
                native / "trace.jsonl",
                prior_trace.rstrip() + "\n" + current_trace.lstrip(),
                overwrite=True,
            )
        atomic_write_json(
            native / "standard_resume_state.json",
            {
                "schema_version": 1,
                "wall_time_seconds": result.wall_time_seconds,
                "tool_calls": [
                    call.model_dump(mode="json", exclude_none=False)
                    for call in result.tool_calls
                ],
                "search_calls": result.search_calls,
                "fetch_calls": result.fetch_calls,
                "external_retrieval_calls": result.external_retrieval_calls,
                "internal_tool_calls": result.internal_tool_calls,
                "relevant_searches": result.relevant_searches,
                "token_usage": (
                    result.token_usage.model_dump(mode="json", exclude_none=False)
                    if result.token_usage is not None
                    else None
                ),
            },
            overwrite=True,
        )
        _write_posthoc_audit(artifact_directory, result)
        return result

    def _build_graph(
        self,
        runtime: PreparedRuntime,
        artifact_directory: Path,
    ) -> _CheckpointedGraph:
        return _CheckpointedGraph(
            runtime,
            artifact_directory,
            interrupt_after_tools=self._interrupt_after_tools,
        )


def _checkpoint_thread_id(task: EvalTask, config: ResolvedConfig) -> str:
    identity = f"{task.id}\0{config.config_fingerprint}".encode()
    return f"tongagent-standard-{hashlib.sha256(identity).hexdigest()[:24]}"


def _write_posthoc_audit(artifact_directory: Path, result: RunResult) -> None:
    """Persist provenance observations without accepting or rejecting the answer."""

    native = artifact_directory / "native"
    successful_fetches: list[dict[str, Any]] = []
    for call in result.tool_calls:
        if call.tool_name != "fetch_url" or call.status.value != "success":
            continue
        payload = call.result if isinstance(call.result, Mapping) else {}
        successful_fetches.append(
            {
                "call_id": call.call_id,
                "source_id": payload.get("source_id"),
                "title": payload.get("title"),
                "url": payload.get("url") or call.arguments.get("url"),
                "content_sha256": payload.get("content_sha256"),
                "provider_status": call.metadata.get("provider_status"),
            }
        )
    evidence_graph = {
        "evidence_graph_version": 1,
        "claims": [],
        "evidence_units": [],
        "conflicts": [],
        "audit_mode": "post_hoc_non_blocking",
    }
    atomic_write_json(
        native / "source_ledger.json",
        {
            "schema_version": 1,
            "successful_fetches": successful_fetches,
        },
        overwrite=True,
    )
    atomic_write_json(
        native / "evidence_graph.json",
        evidence_graph,
        overwrite=True,
    )
    atomic_write_json(
        native / "posthoc_audit.json",
        {
            "schema_version": 1,
            "mode": "post_hoc_non_blocking",
            "raw_model_answer": result.raw_model_answer,
            "final_answer": result.final_answer,
            "answer_unchanged": result.raw_model_answer == result.final_answer,
            "citations": [
                citation.model_dump(mode="json", exclude_none=False)
                for citation in result.citations
            ],
            "source_ledger": "source_ledger.json",
            "evidence_graph": "evidence_graph.json",
        },
        overwrite=True,
    )


def _read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _read_resume_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _merge_resumed_result(
    result: RunResult,
    prior: Mapping[str, Any] | None,
) -> RunResult:
    if prior is None:
        return result
    prior_calls = [
        ToolCall.model_validate_json(json.dumps(item))
        for item in prior.get("tool_calls", [])
        if isinstance(item, Mapping)
    ]
    token_usage = _merge_token_usage(prior.get("token_usage"), result.token_usage)
    updates = {
        "wall_time_seconds": float(prior.get("wall_time_seconds", 0.0))
        + result.wall_time_seconds,
        "tool_calls": [*prior_calls, *result.tool_calls],
        "search_calls": int(prior.get("search_calls") or 0)
        + int(result.search_calls or 0),
        "fetch_calls": int(prior.get("fetch_calls") or 0)
        + int(result.fetch_calls or 0),
        "external_retrieval_calls": int(prior.get("external_retrieval_calls") or 0)
        + int(result.external_retrieval_calls or 0),
        "internal_tool_calls": int(prior.get("internal_tool_calls") or 0)
        + int(result.internal_tool_calls or 0),
        "relevant_searches": int(prior.get("relevant_searches") or 0)
        + int(result.relevant_searches or 0),
        "token_usage": token_usage,
    }
    return RunResult.model_validate(
        result.model_copy(update=updates).model_dump(mode="python")
    )


def _merge_token_usage(
    prior: Any,
    current: TokenUsage | None,
) -> TokenUsage | None:
    if not isinstance(prior, Mapping):
        return current
    previous = TokenUsage.model_validate(prior)
    if current is None:
        return previous
    values: dict[str, int | None] = {}
    for field_name in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    ):
        left = getattr(previous, field_name)
        right = getattr(current, field_name)
        values[field_name] = (
            None if left is None and right is None else int(left or 0) + int(right or 0)
        )
    return TokenUsage.model_validate(values)


__all__ = [
    "BareSimpleReactRunner",
    "TongAgentStandardRunner",
    "TRANSPARENT_REACT_SYSTEM_PROMPT",
    "build_transparent_react_graph",
]
