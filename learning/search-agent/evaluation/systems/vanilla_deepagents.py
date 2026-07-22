"""B2: pinned upstream DeepAgents baseline without TongAgent research state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.language_models import BaseChatModel

from ..config import ResolvedConfig
from ..offline import FixtureBackend
from ..schema import EvalTask, RunResult
from .common import (
    SYSTEM_VANILLA_DEEPAGENTS,
    PreparedRuntime,
    build_evaluation_summarization_middleware,
    run_graph_system,
)


_SYSTEM_PROMPT = """Research the user's question with web_search and fetch_url.

Use the native DeepAgents planning, filesystem, and general-purpose delegation
facilities when useful. Base the final answer on fetched page text and cite
source IDs while researching. If retrieval fails or the shared budget is
exhausted, stop honestly. When finished, output exactly one line and nothing
else: FINAL_ANSWER: <short answer>. Use FINAL_ANSWER: ABSTAIN only when the
available research cannot support an answer.
"""


class VanillaDeepAgentsRunner:
    """Run B2 through the repository-pinned ``deepagents==0.6.12`` API."""

    system_id = SYSTEM_VANILLA_DEEPAGENTS

    def __init__(
        self,
        *,
        fixture_backend: FixtureBackend | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self._fixture_backend = fixture_backend
        self._model = model

    def run(self, task: EvalTask, resolved_config: ResolvedConfig) -> RunResult:
        """Execute one isolated B2 attempt."""

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
    def _build_graph(runtime: PreparedRuntime, artifact_directory: Path):
        workspace = artifact_directory / "native" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
        summarization = build_evaluation_summarization_middleware(
            runtime.model,
            backend,
            runtime.middleware,
        )
        shared_middleware = [runtime.middleware, summarization]

        # Explicitly mirror the native general-purpose subagent so the same
        # accounting and native summarization instances cover work inside it.
        # Passing this spec does not add a TongAgent role or research graph.
        general_purpose: dict[str, Any] = {
            **GENERAL_PURPOSE_SUBAGENT,
            "model": runtime.model,
            "tools": list(runtime.tools),
            "middleware": shared_middleware,
        }
        return create_deep_agent(
            model=runtime.model,
            tools=list(runtime.tools),
            system_prompt=_SYSTEM_PROMPT,
            middleware=shared_middleware,
            subagents=[general_purpose],
            backend=backend,
            name="evaluation-vanilla-deepagents",
        )


__all__ = ["VanillaDeepAgentsRunner"]
