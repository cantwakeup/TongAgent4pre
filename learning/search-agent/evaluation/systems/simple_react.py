"""B1: minimal LangChain ReAct baseline."""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ..config import ResolvedConfig
from ..offline import FixtureBackend
from ..schema import EvalTask, RunResult
from .common import (
    SYSTEM_SIMPLE_REACT,
    PreparedRuntime,
    run_graph_system,
)


_SYSTEM_PROMPT = """You are a minimal web research assistant.

Use web_search to locate relevant results and fetch_url to inspect pages.
Answer the user's question directly from the retrieved page text. Cite fetched
sources with their source IDs and full URLs. If retrieval fails or the budget
is exhausted, state the limitation and stop.
"""


class SimpleReactRunner:
    """Run B1 with only a model and the shared search/fetch tools."""

    system_id = SYSTEM_SIMPLE_REACT

    def __init__(
        self,
        *,
        fixture_backend: FixtureBackend | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self._fixture_backend = fixture_backend
        self._model = model

    def run(self, task: EvalTask, resolved_config: ResolvedConfig) -> RunResult:
        """Execute one isolated B1 attempt."""

        return run_graph_system(
            task,
            resolved_config,
            system_id=self.system_id,
            graph_factory=self._build_graph,
            injected_backend=self._fixture_backend,
            injected_model=self._model,
        )

    @staticmethod
    def _build_graph(runtime: PreparedRuntime, artifact_directory: Path):
        del artifact_directory
        return create_agent(
            model=runtime.model,
            tools=list(runtime.tools),
            system_prompt=_SYSTEM_PROMPT,
            middleware=[runtime.middleware],
            name="evaluation-simple-react",
        )


__all__ = ["SimpleReactRunner"]
