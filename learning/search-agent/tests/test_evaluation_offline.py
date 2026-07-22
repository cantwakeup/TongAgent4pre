"""Offline regression tests for the production fixture runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from evaluation.offline import (
    FixtureBackend,
    FixtureChatModel,
    FixtureFormatError,
    normalize_query,
    normalize_url,
)


pytestmark = pytest.mark.usefixtures("socket_disabled")


def _write_fixture(
    root: Path,
    *,
    searches: dict[str, Any],
    pages: dict[str, Any],
) -> FixtureBackend:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixture_id": "test-fixture",
                "search_file": "search.json",
                "page_file": "pages.json",
            }
        ),
        encoding="utf-8",
    )
    (root / "search.json").write_text(json.dumps(searches), encoding="utf-8")
    (root / "pages.json").write_text(json.dumps(pages), encoding="utf-8")
    return FixtureBackend.from_directory(root)


def test_normalized_exact_match_and_unknown_items_fail_closed(tmp_path: Path) -> None:
    backend = _write_fixture(
        tmp_path,
        searches={
            "Alpha   Research": {
                "status": "success",
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://EXAMPLE.test:443/fact#section",
                        "snippet": "An exact fixture result.",
                        "relevance_score": 30,
                    }
                ],
            }
        },
        pages={
            "https://example.test/fact": {
                "status": "success",
                "title": "Alpha fact",
                "content": "Alpha is a deterministic fixture fact.",
            }
        },
    )

    assert normalize_query("  ALPHA\u3000research ") == "alpha research"
    assert (
        normalize_url("HTTPS://EXAMPLE.TEST:443/fact#ignored")
        == "https://example.test/fact"
    )
    found = backend.search("  ALPHA\u3000research ")
    assert found["status"] == "success"
    assert found["relevant_search"] is True
    fetched = backend.fetch("HTTPS://EXAMPLE.TEST:443/fact#ignored")
    assert fetched["content"] == "Alpha is a deterministic fixture fact."

    missing_search = backend.search("alpha research extra")
    missing_page = backend.fetch("https://example.test/facts")
    assert missing_search["status"] == "fixture_not_found"
    assert missing_search["provider_outcome"] == "not_called"
    assert missing_page["status"] == "fixture_not_found"
    assert missing_page["retryable"] is False
    assert backend.calls[-2]["fixture_found"] is False
    assert backend.calls[-1]["fixture_found"] is False


def test_page_retry_sequence_is_per_url_and_repeats_final_response(
    tmp_path: Path,
) -> None:
    backend = _write_fixture(
        tmp_path,
        searches={},
        pages={
            "https://retry.test/page": {
                "responses": [
                    {
                        "status": "error",
                        "url": "https://retry.test/page",
                        "error": "temporary_fixture_failure",
                        "retryable": True,
                    },
                    {
                        "status": "success",
                        "url": "https://retry.test/page",
                        "title": "Recovered",
                        "content": "The fixture succeeds on its second response.",
                    },
                ]
            },
            "https://other.test/page": {
                "status": "success",
                "content": "Independent URL cursor.",
            },
        },
    )

    first = backend.fetch("https://retry.test/page")
    other = backend.fetch("https://other.test/page")
    second = backend.fetch("https://retry.test/page")
    third = backend.fetch("https://retry.test/page")

    assert first["status"] == "error"
    assert first["fixture_response_index"] == 0
    assert other["fixture_response_index"] == 0
    assert second["status"] == "success"
    assert second["fixture_response_index"] == 1
    assert third == second
    assert [call["response_index"] for call in backend.calls] == [0, 0, 1, 1]


def test_bind_tools_returns_isolated_clones_with_shared_unique_id_runtime() -> None:
    @tool
    def web_search(query: str) -> str:
        """Return a local search marker."""

        return query

    @tool
    def fetch_url(url: str) -> str:
        """Return a local fetch marker."""

        return url

    task = {
        "id": "clone-test",
        "question": "What does the fixture say?",
        "metadata": {
            "answer": "Done.",
            "research_script": [
                {"tool": "web_search", "args": {"query": "fixture query"}},
                {"tool": "fetch_url", "args": {"url": "https://fixture.test/"}},
            ],
        },
    }
    model = FixtureChatModel.from_task(task, system_id="simple_react")
    search_clone = model.bind_tools([web_search])
    fetch_clone = model.bind_tools([fetch_url])

    assert isinstance(search_clone, FixtureChatModel)
    assert isinstance(fetch_clone, FixtureChatModel)
    assert search_clone is not model
    assert fetch_clone is not model
    assert search_clone is not fetch_clone
    assert model.bound_tool_names == ()
    assert search_clone.bound_tool_names == ("web_search",)
    assert fetch_clone.bound_tool_names == ("fetch_url",)
    assert search_clone.runtime is fetch_clone.runtime is model.runtime

    messages = [HumanMessage(content=task["question"])]
    search_message = search_clone.invoke(messages)
    fetch_message = fetch_clone.invoke(messages)
    assert search_message.tool_calls[0]["name"] == "web_search"
    assert fetch_message.tool_calls[0]["name"] == "fetch_url"
    assert search_message.tool_calls[0]["id"] != fetch_message.tool_calls[0]["id"]
    assert search_message.usage_metadata is not None
    assert fetch_message.usage_metadata is not None
    assert len(model.call_history) == 2


def test_fixture_model_drives_minimal_create_agent_loop_deterministically(
    tmp_path: Path,
) -> None:
    backend = _write_fixture(
        tmp_path,
        searches={
            "fixture alpha": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://fixture.test/alpha",
                        "snippet": "Alpha is verified.",
                        "relevance_score": 40,
                    }
                ]
            }
        },
        pages={
            "https://fixture.test/alpha": {
                "title": "Alpha",
                "content": "Alpha is verified by this deterministic fixture.",
            }
        },
    )
    task = {
        "id": "baseline-loop",
        "question": "Is Alpha verified?",
        "metadata": {
            "answer": "Yes. Alpha is verified.",
            "research_script": [
                {
                    "tool": "web_search",
                    "args": {"query": "fixture alpha", "max_results": 3},
                },
                {
                    "tool": "fetch_url",
                    "args": {
                        "url": "https://fixture.test/alpha",
                        "max_chars": 1000,
                    },
                },
            ],
        },
    }

    def run_once() -> tuple[list[str], list[str], str]:
        local_backend = FixtureBackend(
            searches={
                "fixture alpha": {
                    "results": [
                        {
                            "title": "Alpha",
                            "url": "https://fixture.test/alpha",
                            "snippet": "Alpha is verified.",
                            "relevance_score": 40,
                        }
                    ]
                }
            },
            pages={
                "https://fixture.test/alpha": {
                    "title": "Alpha",
                    "content": "Alpha is verified by this deterministic fixture.",
                }
            },
        )
        model = FixtureChatModel.from_task(task, system_id="simple_react")
        agent = create_agent(model=model, tools=local_backend.as_tools())
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task["question"]}]}
        )
        ai_messages = [
            message for message in result["messages"] if isinstance(message, AIMessage)
        ]
        tool_names = [
            call["name"] for message in ai_messages for call in message.tool_calls
        ]
        tool_ids = [
            call["id"] for message in ai_messages for call in message.tool_calls
        ]
        return tool_names, tool_ids, str(ai_messages[-1].content)

    first = run_once()
    second = run_once()
    assert first == second
    assert first[0] == ["web_search", "fetch_url"]
    assert len(first[1]) == len(set(first[1])) == 2
    assert first[2] == task["metadata"]["answer"]
    assert [call["tool"] for call in backend.calls] == []


def test_structured_planner_and_duplicate_normalized_keys(tmp_path: Path) -> None:
    task = {
        "id": "planner",
        "question": "Research one fixture claim",
        "metadata": {
            "planner": {
                "objective": "Verify one fixture claim",
                "subquestions": [
                    {
                        "question": "What does the fixture state?",
                        "rationale": "Obtain the exact statement.",
                        "depends_on": [],
                    }
                ],
            }
        },
    }
    model = FixtureChatModel.from_task(task, system_id="tongagent")
    durable = model.deterministic_planner(task["question"], 2)
    assert durable["planner"] == "fixture"
    assert durable["plan_id"].startswith("fixture-plan-")
    assert durable["subquestions"][0]["status"] == "pending"

    with pytest.raises(FixtureFormatError, match="same exact-match key"):
        FixtureBackend(
            searches={
                "Alpha query": {"results": []},
                " alpha   QUERY ": {"results": []},
            },
            pages={},
        )


def test_fixture_model_gates_tongagent_actions_by_phase_and_outer_cycle() -> None:
    """A stop action ends one inner loop without consuming future actions."""

    @tool
    def web_search(query: str) -> str:
        """Return a local search marker."""

        return query

    @tool
    def fetch_url(url: str) -> str:
        """Return a local fetch marker."""

        return url

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Return a local write marker."""

        return f"{file_path}:{content}"

    task = {
        "id": "phase-cycle",
        "question": "Run the adaptive fixture",
        "metadata": {
            "answer": "Cycle finished.",
            "strict_fixture_tools": True,
            "research_script": [
                {
                    "tool": "web_search",
                    "phase": "research",
                    "research_cycle": 1,
                    "stop_cycle": True,
                    "args": {"query": "first cycle"},
                },
                {
                    "tool": "fetch_url",
                    "phase": "research",
                    "research_cycle": 2,
                    "args": {"url": "https://fixture.test/fact"},
                },
                {
                    "tool": "write_file",
                    "phase": "report",
                    "args": {"file_path": "/report.md", "content": "final report"},
                },
            ],
        },
    }
    model = FixtureChatModel.from_task(task, system_id="tongagent")
    bound = model.bind_tools([web_search, fetch_url, write_file])
    assert isinstance(bound, FixtureChatModel)

    first_messages = [
        HumanMessage(
            id="research-step-plan-SQ1-1",
            content="[RESEARCH STEP]\nFirst outer cycle.",
        )
    ]
    first = bound.invoke(first_messages)
    assert first.tool_calls[0]["name"] == "web_search"
    first_messages.extend(
        [
            first,
            ToolMessage(
                content="first cycle",
                name="web_search",
                tool_call_id=first.tool_calls[0]["id"],
            ),
        ]
    )
    stopped = bound.invoke(first_messages)
    assert stopped.tool_calls == []

    second_messages = [
        *first_messages,
        stopped,
        HumanMessage(
            id="research-step-plan-SQ1-2",
            content="[RESEARCH STEP]\nSecond outer cycle.",
        ),
    ]
    second = bound.invoke(second_messages)
    assert second.tool_calls[0]["name"] == "fetch_url"
    second_messages.extend(
        [
            second,
            ToolMessage(
                content="fixture page",
                name="fetch_url",
                tool_call_id=second.tool_calls[0]["id"],
            ),
        ]
    )
    research_finished = bound.invoke(second_messages)
    assert research_finished.tool_calls == []

    report = bound.invoke(
        [
            *second_messages,
            research_finished,
            HumanMessage(
                id="report-step-plan-2",
                content="[FINAL SYNTHESIS]\nWrite the final report.",
            ),
        ]
    )
    assert report.tool_calls[0]["name"] == "write_file"
