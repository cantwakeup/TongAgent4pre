"""Regression tests for durable LangGraph threads and per-run audit traces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from search_agent import _messages_since_checkpoint


def _reply(state: MessagesState) -> dict[str, list[AIMessage]]:
    """Return a deterministic message whose ID changes on a resumed turn."""
    message_count = len(state["messages"])
    return {
        "messages": [
            AIMessage(
                content=f"reply after {message_count} messages",
                id=f"reply-{message_count}",
            )
        ]
    }


def _build_graph(checkpointer: SqliteSaver) -> Any:
    """Compile a minimal message graph against the supplied durable saver."""
    builder = StateGraph(MessagesState)
    builder.add_node("reply", _reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=checkpointer)


class CheckpointTests(unittest.TestCase):
    """Verify SQLite recovery and exclusion of old messages from a new trace."""

    def test_same_thread_recovers_history_but_trace_keeps_current_turn_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "checkpoints.sqlite"
            config = {"configurable": {"thread_id": "research-thread"}}

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                graph = _build_graph(checkpointer)
                graph.invoke(
                    {"messages": [HumanMessage(content="first", id="human-1")]},
                    config=config,
                )

            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                graph = _build_graph(checkpointer)
                checkpoint = graph.get_state(config)
                previous = checkpoint.values["messages"]
                previous_ids = {message.id for message in previous if message.id}
                result = graph.invoke(
                    {"messages": [HumanMessage(content="second", id="human-2")]},
                    config=config,
                )

            all_ids = [message.id for message in result["messages"]]
            current = _messages_since_checkpoint(result["messages"], previous_ids)
            current_ids = [message.id for message in current]

            self.assertEqual(all_ids, ["human-1", "reply-1", "human-2", "reply-3"])
            self.assertEqual(current_ids, ["human-2", "reply-3"])


if __name__ == "__main__":
    unittest.main()
