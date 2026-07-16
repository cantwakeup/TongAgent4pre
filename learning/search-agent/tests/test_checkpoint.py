"""Regression tests for durable LangGraph threads and per-run audit traces."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_policy import EFFORT_POLICIES
from research_graph import (
    create_research_plan,
    select_next_subquestion,
    transition_subquestion,
)
from search_agent import (
    AgentBundle,
    ResearchBudget,
    _aggregate_message_usage,
    _execute_cli,
    _messages_since_checkpoint,
    _turn_inputs,
)


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

    def test_pending_node_resumes_with_none_before_accepting_new_input(self) -> None:
        same_question = _turn_inputs(
            "original topic",
            ["follow-up"],
            resume_pending=True,
            active_question="original topic",
        )
        new_question = _turn_inputs(
            "new topic",
            [],
            resume_pending=True,
            active_question="original topic",
        )

        self.assertEqual(same_question, [None, "follow-up"])
        self.assertEqual(new_question, [None, "new topic"])

    def test_api_usage_aggregates_checkpointed_ai_messages(self) -> None:
        usage = _aggregate_message_usage(
            [
                HumanMessage(content="question"),
                AIMessage(
                    content="first",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                        "input_token_details": {"cache_read": 4},
                    },
                ),
                AIMessage(
                    content="second",
                    usage_metadata={
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "total_tokens": 23,
                    },
                ),
            ]
        )

        self.assertEqual(
            usage,
            {
                "model_calls": 2,
                "input_tokens": 30,
                "output_tokens": 5,
                "total_tokens": 35,
                "cache_read_tokens": 4,
            },
        )

    def test_execute_cli_continues_pending_node_with_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            pending = create_research_plan(
                "original topic",
                ["Establish the facts"],
                plan_id_factory=lambda: "plan-cli-resume",
            )
            pending["evidence_schema_version"] = 0
            pending, _ = select_next_subquestion(pending)
            completed = transition_subquestion(
                pending,
                "SQ1",
                "covered",
                evidence_source_ids=["S1", "S2"],
                note="covered before report",
            )
            budget = ResearchBudget(EFFORT_POLICIES["low"])
            budget.restore(
                {
                    "search_calls": 1,
                    "fetch_calls": 2,
                    "successful_sources": [
                        {
                            "source_id": "S1",
                            "url": "https://example.com/one",
                            "title": "One",
                            "content_chars": 800,
                        },
                        {
                            "source_id": "S2",
                            "url": "https://example.com/two",
                            "title": "Two",
                            "content_chars": 800,
                        },
                    ],
                    "failed_sources": [],
                }
            )

            class _PendingAgent:
                def __init__(self) -> None:
                    self.inputs: list[dict[str, Any] | None] = []

                def get_state(self, config: dict[str, Any]) -> Any:
                    del config
                    return SimpleNamespace(
                        values={
                            "messages": [
                                HumanMessage(
                                    content="[FINAL SYNTHESIS]",
                                    id="report-step-plan-cli-resume-1",
                                )
                            ],
                            "research_plan": completed,
                            "budget_state": budget.snapshot(),
                        },
                        next=("report_agent",),
                    )

                def invoke(
                    self, payload: dict[str, Any] | None, *, config: dict[str, Any]
                ) -> dict[str, Any]:
                    del config
                    self.inputs.append(payload)
                    report = (
                        "# Report\n\nSources\n\n"
                        "[S1] One — https://example.com/one\n\n"
                        "[S2] Two — https://example.com/two\n"
                    )
                    (output_dir / "report.md").write_text(report)
                    return {
                        "messages": [
                            HumanMessage(
                                content="[FINAL SYNTHESIS]",
                                id="report-step-plan-cli-resume-1",
                            ),
                            AIMessage(
                                content="",
                                id="write-call-message",
                                tool_calls=[
                                    {
                                        "name": "write_file",
                                        "args": {
                                            "file_path": "/report.md",
                                            "content": report,
                                        },
                                        "id": "write-call",
                                        "type": "tool_call",
                                    }
                                ],
                            ),
                            ToolMessage(
                                content="written",
                                tool_call_id="write-call",
                                name="write_file",
                                id="write-result",
                            ),
                        ],
                        "research_plan": completed,
                        "research_events": [],
                    }

            fake_agent = _PendingAgent()
            bundle = AgentBundle(
                agent=fake_agent,
                budget=budget,
                policy=EFFORT_POLICIES["low"],
                mode="single",
                topology="single",
            )
            args = Namespace(
                model="test-model",
                effort="low",
                worker_model="test-worker",
                mode="single",
                topic="original topic",
                follow_up=[],
                no_stream=True,
                print_report=False,
            )

            with patch("search_agent.build_agent", return_value=bundle):
                _execute_cli(
                    args,
                    output_dir=output_dir,
                    checkpoint_path=Path(temp_dir) / "checkpoint.sqlite",
                    checkpointer=object(),
                    thread_id="cli-resume",
                )

            self.assertEqual(fake_agent.inputs, [None])


if __name__ == "__main__":
    unittest.main()
