"""Regression tests for concise streaming output."""

from __future__ import annotations

import io
import unittest
from collections.abc import Iterator
from contextlib import redirect_stdout
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, ToolMessageChunk

from search_agent import _stream_agent


class _FakeAgent:
    """Yield representative model, tool, and state events without network access."""

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        del args, kwargs
        yield "messages", (
            ToolMessageChunk(
                content="SECRET PAGE BODY",
                tool_call_id="call-1",
                name="fetch_url",
                id="tool-1",
            ),
            {"langgraph_node": "tools"},
        )
        yield "messages", (
            AIMessageChunk(content="WRONG NODE BODY", id="ai-wrong"),
            {"langgraph_node": "tools"},
        )
        yield "messages", (
            AIMessageChunk(content="hello", id="ai-final"),
            {"langgraph_node": "model"},
        )
        yield "values", {
            "messages": [
                AIMessage(
                    content="",
                    id="ai-call",
                    tool_calls=[
                        {
                            "name": "fetch_url",
                            "args": {"url": "https://example.com"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="SECRET PAGE BODY",
                    tool_call_id="call-1",
                    name="fetch_url",
                    id="tool-1",
                ),
                AIMessage(content="hello", id="ai-final"),
            ]
        }


class StreamAgentTests(unittest.TestCase):
    """Verify that live output is useful without dumping tool payloads."""

    def test_stream_hides_tool_content_and_keeps_events(self) -> None:
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            state = _stream_agent(_FakeAgent(), "test")

        output = buffer.getvalue()
        self.assertEqual(state["messages"][-1].content, "hello")
        self.assertIn("[model] hello", output)
        self.assertIn("[tool call] fetch_url", output)
        self.assertIn("[tool result] fetch_url (16 characters)", output)
        self.assertNotIn("SECRET PAGE BODY", output)
        self.assertNotIn("WRONG NODE BODY", output)


if __name__ == "__main__":
    unittest.main()
