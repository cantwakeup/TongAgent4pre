"""Validate the deterministic Stage D fixture smoke suite and its artifacts.

This module is intentionally narrower than the generic evaluation contracts.
It checks that the six fixed offline scenarios actually exercise the behavior
claimed by Stage D, and that a completed B1/B2/B3 smoke experiment retained
truthful, provenance-backed results.  It never invokes a model or network
provider.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .aggregate import FIXTURE_SMOKE_DISCLAIMER, aggregate_experiment
from .execution import validate_persisted_result


SYSTEM_IDS = ("simple_react", "vanilla_deepagents", "tongagent")
SCENARIOS = (
    "one-hop",
    "two-source",
    "nonempty-irrelevant",
    "conflict",
    "fetch-retry",
    "budget-resume",
)
FIXTURE_KIND = "deterministic_offline_smoke"
BENCHMARK_STATUS = "smoke_only_not_a_formal_benchmark"
MIN_RELEVANCE_SCORE = 20
BASELINE_SCENARIO_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "one-hop": {
        "search_calls": 1,
        "fetch_calls": 2,
        "relevant_searches": 1,
        "citation_count": 2,
    },
    "two-source": {
        "search_calls": 2,
        "fetch_calls": 2,
        "relevant_searches": 2,
        "citation_count": 2,
    },
    "nonempty-irrelevant": {
        "search_calls": 1,
        "fetch_calls": 0,
        "relevant_searches": 0,
        "citation_count": 0,
    },
    "conflict": {
        "search_calls": 2,
        "fetch_calls": 3,
        "relevant_searches": 2,
        "citation_count": 3,
    },
    "fetch-retry": {
        "search_calls": 1,
        "fetch_calls": 3,
        "relevant_searches": 1,
        "citation_count": 2,
        "failed_fetches": 1,
    },
    "budget-resume": {
        "search_calls": 4,
        "fetch_calls": 0,
        "relevant_searches": 4,
        "citation_count": 0,
    },
}
STANDARD_ATTEMPT_FILES = (
    "answer.md",
    "failure.json",
    "metrics.json",
    "result.json",
    "trace.json",
    "native/budget.json",
    "native/native.json",
    "native/resolved_config.json",
    "native/tool_calls.json",
    "native/trace.jsonl",
)


class StageDSmokeValidationError(AssertionError):
    """Raised when Stage D inputs or generated artifacts contradict the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageDSmokeValidationError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StageDSmokeValidationError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StageDSmokeValidationError(
            f"invalid JSON file {path}: {exc.msg}"
        ) from exc


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StageDSmokeValidationError(f"missing dataset: {path}") from exc
    tasks: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        _require(bool(line.strip()), f"blank JSONL record at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StageDSmokeValidationError(
                f"invalid dataset JSON at line {line_number}: {exc.msg}"
            ) from exc
        _require(
            isinstance(value, dict),
            f"dataset line {line_number} must be a JSON object",
        )
        tasks.append(value)
    return tasks


def _fixture_host(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").rstrip(".").casefold()
    return hostname == "fixture.test" or hostname.endswith(".fixture.test")


def _page_contents(entry: Any) -> list[str]:
    if isinstance(entry, Mapping) and "responses" in entry:
        responses = entry["responses"]
    elif isinstance(entry, list):
        responses = entry
    else:
        responses = [entry]
    if not isinstance(responses, list):
        return []
    return [
        str(response.get("content", ""))
        for response in responses
        if isinstance(response, Mapping)
        and response.get("status", "success") == "success"
    ]


def _actions(task: Mapping[str, Any], system_id: str) -> list[dict[str, Any]]:
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    scripts = metadata.get("research_script")
    if isinstance(scripts, Mapping):
        selected = scripts.get(system_id, scripts.get("default"))
    else:
        selected = scripts
    if not isinstance(selected, list):
        return []
    return [dict(item) for item in selected if isinstance(item, Mapping)]


def _scenario_task(tasks: Sequence[dict[str, Any]], scenario: str) -> dict[str, Any]:
    matches = [
        task
        for task in tasks
        if isinstance(task.get("metadata"), Mapping)
        and task["metadata"].get("scenario") == scenario
    ]
    _require(len(matches) == 1, f"scenario {scenario!r} must appear exactly once")
    return matches[0]


def validate_inputs(
    dataset_path: str | Path,
    fixture_directory: str | Path,
) -> dict[str, Any]:
    """Validate the six-task closed-world fixture suite without executing it."""

    dataset = Path(dataset_path).expanduser().resolve()
    fixtures = Path(fixture_directory).expanduser().resolve()
    manifest = _read_json(fixtures / "manifest.json")
    searches = _read_json(fixtures / "search.json")
    pages = _read_json(fixtures / "pages.json")
    tasks = _read_tasks(dataset)

    _require(isinstance(manifest, dict), "fixture manifest must be an object")
    _require(manifest.get("schema_version") == 1, "fixture schema_version must be 1")
    _require(
        manifest.get("fixture_kind") == FIXTURE_KIND,
        "fixture manifest must identify deterministic offline smoke data",
    )
    _require(
        manifest.get("benchmark_status") == BENCHMARK_STATUS,
        "fixture manifest must say smoke_only_not_a_formal_benchmark",
    )
    _require(
        manifest.get("search_file") == "search.json"
        and manifest.get("page_file") == "pages.json",
        "fixture manifest must point at the canonical search/page files",
    )
    _require(isinstance(searches, dict), "search fixture must be an object")
    _require(isinstance(pages, dict), "page fixture must be an object")
    _require(len(tasks) == len(SCENARIOS), "Stage D dataset must contain six tasks")

    ids = [task.get("id") for task in tasks]
    _require(
        all(isinstance(task_id, str) and task_id for task_id in ids),
        "every Stage D task needs a non-empty id",
    )
    _require(len(set(ids)) == len(ids), "Stage D task ids must be unique")
    actual_scenarios = [
        task.get("metadata", {}).get("scenario")
        if isinstance(task.get("metadata"), Mapping)
        else None
        for task in tasks
    ]
    _require(
        set(actual_scenarios) == set(SCENARIOS),
        f"Stage D scenarios differ: {actual_scenarios!r}",
    )

    successful_page_text = {url: _page_contents(entry) for url, entry in pages.items()}
    all_page_text = "\n".join(
        content for contents in successful_page_text.values() for content in contents
    )
    for url in pages:
        _require(_fixture_host(url), f"page fixture is not offline-only: {url}")
        contents = successful_page_text[url]
        _require(
            bool(contents) and min(len(content) for content in contents) >= 500,
            f"successful fixture page does not pass the full-content gate: {url}",
        )

    for query, response in searches.items():
        _require(
            isinstance(response, Mapping),
            f"search response for {query!r} must be an object",
        )
        results = response.get("results", [])
        _require(
            isinstance(results, list),
            f"search response for {query!r} must contain a result list",
        )
        relevant_results = sum(
            isinstance(item, Mapping)
            and isinstance(item.get("relevance_score", 0), int)
            and not isinstance(item.get("relevance_score", 0), bool)
            and item.get("relevance_score", 0) >= MIN_RELEVANCE_SCORE
            for item in results
        )
        if "nonempty_search" in response:
            _require(
                response["nonempty_search"] is bool(results),
                f"search {query!r} has inconsistent nonempty_search",
            )
        if "relevant_results" in response:
            _require(
                response["relevant_results"] == relevant_results,
                f"search {query!r} has inconsistent relevant_results",
            )
        if "relevant_search" in response:
            _require(
                response["relevant_search"] is (relevant_results > 0),
                f"search {query!r} has inconsistent relevant_search",
            )
        for result in results:
            _require(
                isinstance(result, Mapping),
                f"search result for {query!r} must be an object",
            )
            url = result.get("url")
            _require(
                isinstance(url, str) and _fixture_host(url),
                f"search fixture is not offline-only: {query!r} -> {url!r}",
            )

    allowed_tools = {
        "fetch_url",
        "record_evidence",
        "update_subquestion",
        "web_search",
        "write_file",
    }
    for task in tasks:
        metadata = task.get("metadata")
        _require(isinstance(metadata, Mapping), f"task {task.get('id')} lacks metadata")
        _require(
            metadata.get("fixture_kind") == FIXTURE_KIND,
            f"task {task.get('id')} has the wrong fixture_kind",
        )
        _require(
            metadata.get("benchmark_status") == BENCHMARK_STATUS,
            f"task {task.get('id')} does not explicitly say smoke-only",
        )
        _require(
            task.get("reference_answer") is None,
            f"task {task.get('id')} must leave external answer scoring unavailable",
        )
        expected = metadata.get("expected")
        _require(
            isinstance(expected, Mapping) and set(expected) == set(SYSTEM_IDS),
            f"task {task.get('id')} needs expectations for all three systems",
        )
        for baseline_id in SYSTEM_IDS[:2]:
            baseline_expected = expected[baseline_id]
            _require(
                isinstance(baseline_expected, Mapping),
                f"task {task.get('id')} expectation for {baseline_id} is invalid",
            )
            _require(
                "evidence_count" not in baseline_expected
                and "structural_subquestion_coverage" not in baseline_expected,
                f"task {task.get('id')} invents TongAgent metrics for {baseline_id}",
            )

        for system_id in SYSTEM_IDS:
            actions = _actions(task, system_id)
            _require(
                bool(actions),
                f"task {task.get('id')} has no script for {system_id}",
            )
            for action in actions:
                tool = action.get("tool")
                args = action.get("args")
                _require(
                    tool in allowed_tools,
                    f"task {task.get('id')} scripts unsupported tool {tool!r}",
                )
                _require(
                    isinstance(args, Mapping),
                    f"task {task.get('id')} tool {tool!r} needs object args",
                )
                if tool == "web_search":
                    _require(
                        args.get("query") in searches,
                        f"task {task.get('id')} uses unknown fixture query "
                        f"{args.get('query')!r}",
                    )
                elif tool == "fetch_url":
                    url = args.get("url")
                    if isinstance(url, str) and url.startswith("${"):
                        continue
                    _require(
                        url in pages,
                        f"task {task.get('id')} fetches unknown fixture URL {url!r}",
                    )
                elif tool == "record_evidence":
                    quote = args.get("quote")
                    if isinstance(quote, str) and quote.startswith("${"):
                        continue
                    _require(
                        isinstance(quote, str) and quote in all_page_text,
                        f"task {task.get('id')} records a non-verbatim quote",
                    )

        values = metadata.get("template_values", {})
        _require(
            isinstance(values, Mapping),
            f"task {task.get('id')} template_values must be an object",
        )
        for key, quote in values.items():
            if not str(key).startswith("quote"):
                continue
            suffix = str(key).removeprefix("quote")
            url = values.get(f"url{suffix}")
            _require(
                isinstance(quote, str)
                and isinstance(url, str)
                and quote in "\n".join(successful_page_text.get(url, [])),
                f"task {task.get('id')} template {key} lacks exact URL provenance",
            )

    one_hop = _scenario_task(tasks, "one-hop")
    _require(
        sum(
            action.get("tool") == "web_search"
            for action in _actions(one_hop, "simple_react")
        )
        == 1,
        "one-hop scenario must use one baseline search",
    )

    two_source = _scenario_task(tasks, "two-source")
    two_source_queries = {
        action.get("args", {}).get("query")
        for action in _actions(two_source, "simple_react")
        if action.get("tool") == "web_search"
    }
    _require(
        len(two_source_queries) == 2,
        "two-source scenario must retrieve through two explicit searches",
    )

    irrelevant = _scenario_task(tasks, "nonempty-irrelevant")
    irrelevant_query = next(
        action["args"]["query"]
        for action in _actions(irrelevant, "simple_react")
        if action.get("tool") == "web_search"
    )
    irrelevant_results = searches[irrelevant_query]["results"]
    _require(bool(irrelevant_results), "irrelevant scenario must be non-empty")
    _require(
        all(
            item.get("relevance_score", 0) < MIN_RELEVANCE_SCORE
            for item in irrelevant_results
        ),
        "irrelevant scenario must contain no relevant result",
    )
    _require(
        all(
            action.get("tool")
            not in {"fetch_url", "record_evidence", "update_subquestion"}
            for action in _actions(irrelevant, "tongagent")
        ),
        "irrelevant TongAgent scenario must not fabricate evidence",
    )

    conflict = _scenario_task(tasks, "conflict")
    stances = {
        action.get("args", {}).get("stance")
        for action in _actions(conflict, "tongagent")
        if action.get("tool") == "record_evidence"
    }
    _require(
        {"supports", "contradicts"}.issubset(stances),
        "conflict scenario must preserve support and contradiction edges",
    )

    retry = _scenario_task(tasks, "fetch-retry")
    retry_values = retry["metadata"]["template_values"]
    retry_url = retry_values["url1"]
    retry_responses = pages[retry_url].get("responses")
    _require(
        isinstance(retry_responses, list)
        and len(retry_responses) >= 2
        and retry_responses[0].get("status") != "success"
        and retry_responses[0].get("retryable") is True
        and retry_responses[1].get("status") == "success",
        "fetch-retry page must fail retryably before succeeding",
    )
    for system_id in SYSTEM_IDS:
        rendered_retry_url = "${url1}"
        retry_calls = [
            action
            for action in _actions(retry, system_id)
            if action.get("tool") == "fetch_url"
            and action.get("args", {}).get("url") == rendered_retry_url
        ]
        _require(
            len(retry_calls) == 2,
            f"fetch-retry scenario must retry the same URL for {system_id}",
        )

    budget = _scenario_task(tasks, "budget-resume")
    for system_id in SYSTEM_IDS:
        search_count = sum(
            action.get("tool") == "web_search" for action in _actions(budget, system_id)
        )
        _require(
            search_count == 5,
            f"budget scenario must attempt five searches for {system_id}",
        )

    expected_cycles = {
        "one-hop": [1, 1, 1, 3, 3, 3, None],
        "two-source": [1, 1, 1, 3, 3, 3, 3, None],
        "nonempty-irrelevant": [1, None],
        "conflict": [1, 1, 1, 3, 3, 3, 5, 5, 5, None],
        "fetch-retry": [1, 1, 2, 2, 4, 4, 4, None],
        "budget-resume": [1, 2, 2, 2, 2, None],
    }
    expected_stop_indexes = {
        "one-hop": [2],
        "two-source": [2],
        "nonempty-irrelevant": [0],
        "conflict": [2, 5],
        "fetch-retry": [1, 3],
        "budget-resume": [0],
    }
    for scenario in SCENARIOS:
        scenario_actions = _actions(_scenario_task(tasks, scenario), "tongagent")
        observed_cycles = [action.get("research_cycle") for action in scenario_actions]
        observed_stops = [
            index
            for index, action in enumerate(scenario_actions)
            if action.get("stop_cycle") is True
        ]
        _require(
            observed_cycles == expected_cycles[scenario]
            and observed_stops == expected_stop_indexes[scenario],
            f"{scenario} fixture script does not match the controller cycle schedule",
        )

    return {
        "fixture_kind": FIXTURE_KIND,
        "benchmark_status": BENCHMARK_STATUS,
        "task_count": len(tasks),
        "scenario_count": len(actual_scenarios),
        "search_fixture_count": len(searches),
        "page_fixture_count": len(pages),
    }


def _attempt_number(path: Path) -> int:
    suffix = path.name.removeprefix("attempt-")
    return int(suffix) if suffix.isdigit() else -1


def _latest_result(
    task_directory: Path,
    *,
    system_id: str,
) -> tuple[Path, dict[str, Any]]:
    attempts = sorted(
        (
            child
            for child in task_directory.glob("attempt-*")
            if child.is_dir() and (child / "result.json").is_file()
        ),
        key=_attempt_number,
    )
    _require(bool(attempts), f"no completed attempt in {task_directory}")
    latest = attempts[-1]
    result = validate_persisted_result(
        latest / "result.json",
        system_id=system_id,
    ).model_dump(mode="json", exclude_none=False)
    return latest, result


def _failed_tool_calls(result: Mapping[str, Any], tool_name: str) -> int:
    calls = result.get("tool_calls", [])
    if not isinstance(calls, list):
        return 0
    return sum(
        isinstance(call, Mapping)
        and call.get("tool_name") == tool_name
        and call.get("status") != "success"
        for call in calls
    )


def _validate_expected(
    *,
    task: Mapping[str, Any],
    system_id: str,
    attempt: Path,
    result: Mapping[str, Any],
) -> None:
    metadata = task["metadata"]
    expected = dict(metadata["expected"][system_id])
    if system_id in SYSTEM_IDS[:2]:
        expected.update(BASELINE_SCENARIO_EXPECTATIONS[metadata["scenario"]])
    evidence_path = attempt / "native" / "tongagent" / "evidence.json"
    evidence = (
        _read_json(evidence_path)
        if system_id == "tongagent" and evidence_path.is_file()
        else {}
    )
    for key, value in expected.items():
        if key == "citation_count":
            observed = len(result.get("citations", []))
        elif key == "conflict_count":
            observed = len(evidence.get("conflicts", []))
        elif key == "failed_fetches":
            observed = _failed_tool_calls(result, "fetch_url")
        else:
            observed = result.get(key)
        _require(
            observed == value,
            f"{system_id}/{task['id']} expected {key}={value!r}, observed {observed!r}",
        )


def _validate_b3_provenance(
    *,
    task: Mapping[str, Any],
    attempt: Path,
    result: Mapping[str, Any],
    pages: Mapping[str, Any],
) -> None:
    citations = result.get("citations", [])
    _require(isinstance(citations, list), f"{task['id']} citations must be a list")
    evidence_path = attempt / "native" / "tongagent" / "evidence.json"
    _require(evidence_path.is_file(), f"{task['id']} lacks TongAgent evidence.json")
    evidence = _read_json(evidence_path)
    _require(
        evidence.get("integrity_errors") == [],
        f"{task['id']} evidence graph has integrity errors",
    )
    for citation in citations:
        _require(isinstance(citation, Mapping), f"{task['id']} citation is invalid")
        url = citation.get("url")
        quote = citation.get("quote")
        _require(
            isinstance(url, str) and url in pages,
            f"{task['id']} citation URL is absent from fixture pages: {url!r}",
        )
        _require(
            isinstance(quote, str)
            and quote
            and quote in "\n".join(_page_contents(pages[url])),
            f"{task['id']} citation is not an exact fixture quote",
        )

    scenario = task["metadata"]["scenario"]
    if scenario == "nonempty-irrelevant":
        _require(
            result.get("completion_status") == "partial"
            and result.get("relevant_searches") == 0
            and result.get("evidence_count") == 0
            and result.get("structural_subquestion_coverage") == 0.0
            and not citations,
            "irrelevant non-empty retrieval falsely completed TongAgent",
        )
    elif scenario == "budget-resume":
        _require(
            result.get("completion_status") == "budget_exhausted",
            "budget scenario must terminate as budget_exhausted",
        )
        search_calls = [
            call
            for call in result.get("tool_calls", [])
            if isinstance(call, Mapping) and call.get("tool_name") == "web_search"
        ]
        semantic_denials = [
            call
            for call in search_calls
            if isinstance(call.get("result"), Mapping)
            and call["result"].get("reason") == "subquestion_budget_exceeded"
        ]
        global_denials = [
            call for call in search_calls if call.get("status") == "budget_exceeded"
        ]
        _require(
            len(search_calls) == 5
            and len(semantic_denials) == 3
            and len(global_denials) == 1,
            "budget scenario must distinguish provider, subquestion, and global "
            "search-budget outcomes",
        )


def _tool_calls(
    result: Mapping[str, Any],
    tool_name: str,
) -> list[Mapping[str, Any]]:
    calls = result.get("tool_calls", [])
    if not isinstance(calls, list):
        return []
    return [
        call
        for call in calls
        if isinstance(call, Mapping) and call.get("tool_name") == tool_name
    ]


def _validate_runtime_scenarios(
    *,
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    attempts: Mapping[tuple[str, str], Path],
) -> None:
    """Check the semantic outcomes that make the six fixtures meaningful."""

    for system_id in SYSTEM_IDS:
        irrelevant = results[(system_id, "fixture-nonempty-irrelevant")]
        searches = _tool_calls(irrelevant, "web_search")
        _require(
            len(searches) == 1,
            f"{system_id} irrelevant scenario must contain one search attempt",
        )
        search = searches[0]
        metadata = search.get("metadata")
        payload = search.get("result")
        _require(
            isinstance(metadata, Mapping)
            and metadata.get("provider_success") is True
            and metadata.get("nonempty_search") is True
            and metadata.get("relevant_search") is False,
            f"{system_id} did not preserve irrelevant-search semantic metadata",
        )
        _require(
            isinstance(payload, Mapping)
            and payload.get("provider_success") is True
            and payload.get("nonempty_search") is True
            and payload.get("relevant_search") is False
            and payload.get("relevant_results") == 0,
            f"{system_id} irrelevant-search payload is semantically inconsistent",
        )

        retry = results[(system_id, "fixture-fetch-retry")]
        fetches = _tool_calls(retry, "fetch_url")
        _require(
            retry.get("fetch_calls") == 3 and len(fetches) == 3,
            f"{system_id} retry must consume three fetch calls without refund",
        )
        first_payload = fetches[0].get("result")
        second_payload = fetches[1].get("result")
        _require(
            fetches[0].get("status") == "error"
            and isinstance(first_payload, Mapping)
            and first_payload.get("failure_class") == "timeout"
            and first_payload.get("retryable") is True
            and first_payload.get("fixture_response_index") == 0,
            f"{system_id} retry failure taxonomy is incorrect",
        )
        _require(
            fetches[1].get("status") == "success"
            and isinstance(second_payload, Mapping)
            and second_payload.get("fixture_response_index") == 1
            and second_payload.get("requested_url")
            == first_payload.get("requested_url")
            and all(call.get("status") != "budget_exceeded" for call in fetches),
            f"{system_id} did not retry the failed fixture fetch exactly once",
        )

        exhausted = results[(system_id, "fixture-budget-resume")]
        _require(
            exhausted.get("completion_status") == "budget_exhausted"
            and exhausted.get("failure_type") == "budget_exhausted"
            and exhausted.get("search_calls") == 4,
            f"{system_id} budget exhaustion was disguised as normal completion",
        )

    conflict_pair = ("tongagent", "fixture-conflict")
    conflict_attempt = attempts[conflict_pair]
    conflict_result = results[conflict_pair]
    conflict_evidence = _read_json(
        conflict_attempt / "native" / "tongagent" / "evidence.json"
    )
    conflicts = conflict_evidence.get("conflicts", [])
    claims = conflict_evidence.get("claims", [])
    _require(
        isinstance(conflicts, list)
        and len(conflicts) == 1
        and conflicts[0].get("status") == "unresolved"
        and isinstance(claims, list)
        and len(claims) == 1
        and claims[0].get("status") == "contested"
        and "## Conflicts and Caveats" in str(conflict_result.get("final_answer", "")),
        "TongAgent conflict was flattened or omitted from the report state",
    )

    two_source_attempt = attempts[("tongagent", "fixture-two-source")]
    budget = _read_json(two_source_attempt / "native" / "budget.json")
    research = budget.get("research", {})
    diversity = (
        research.get("source_diversity", {}) if isinstance(research, Mapping) else {}
    )
    _require(
        isinstance(diversity, Mapping)
        and diversity.get("distinct_source_host_count") == 2
        and diversity.get("distinct_content_revision_count") == 2
        and diversity.get("corroborating_source_group_count") == 2,
        "TongAgent two-source fixture did not retain two independent groups",
    )


def _validate_command_logs(
    experiment: Path,
    *,
    require_rerun: bool,
) -> None:
    fresh_path = experiment / "fresh-run.json"
    resume_path = experiment / "resume-run.json"
    rerun_path = experiment / "rerun-run.json"
    _require(
        fresh_path.is_file() and resume_path.is_file(),
        "strict Stage D validation requires fresh and resume command logs",
    )
    fresh = _read_json(fresh_path)
    _require(
        fresh.get("executed") == 18
        and fresh.get("skipped") == 0
        and fresh.get("dry_run") == 0
        and isinstance(fresh.get("jobs"), list)
        and len(fresh["jobs"]) == 18,
        "fresh command log does not prove 18 new workers",
    )
    resume = _read_json(resume_path)
    _require(
        resume.get("executed") == 0
        and resume.get("skipped") == 18
        and resume.get("dry_run") == 0
        and isinstance(resume.get("jobs"), list)
        and len(resume["jobs"]) == 18,
        "resume command log does not prove 18 safe skips",
    )
    before = experiment / "fresh-results.sha256"
    after = experiment / "resume-results.sha256"
    _require(
        before.is_file()
        and after.is_file()
        and before.read_bytes() == after.read_bytes(),
        "resume changed a terminal result.json",
    )
    manifest_entries: list[tuple[str, str]] = []
    for line in before.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        _require(len(fields) == 2, "invalid result hash manifest entry")
        digest, raw_relative = fields
        relative = raw_relative.lstrip("*").removeprefix("./")
        relative_path = Path(relative)
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative_path.name == "result.json",
            "result hash manifest must contain safe relative SHA-256 entries",
        )
        result_path = experiment / relative_path
        _require(
            result_path.is_file()
            and hashlib.sha256(result_path.read_bytes()).hexdigest() == digest,
            f"terminal result changed after fresh execution: {relative}",
        )
        manifest_entries.append((digest, relative))
    _require(
        len(manifest_entries) == 18
        and len({relative for _, relative in manifest_entries}) == 18,
        "fresh result hash manifest must cover exactly 18 distinct results",
    )
    if require_rerun:
        _require(rerun_path.is_file(), "rerun command log is required")
        rerun = _read_json(rerun_path)
        _require(
            rerun.get("executed") == 1
            and rerun.get("skipped") == 0
            and isinstance(rerun.get("jobs"), list)
            and len(rerun["jobs"]) == 1
            and rerun["jobs"][0].get("system_id") == "simple_react"
            and rerun["jobs"][0].get("task_id") == "fixture-fetch-retry",
            "targeted rerun must execute exactly one new worker",
        )
        rerun_task = experiment / "simple_react" / "fixture-fetch-retry"
        _require(
            (rerun_task / "attempt-0001" / "result.json").is_file()
            and (rerun_task / "attempt-0002" / "result.json").is_file(),
            "targeted rerun must preserve attempt-0001 and add attempt-0002",
        )


def validate_experiment(
    experiment_directory: str | Path,
    dataset_path: str | Path,
    fixture_directory: str | Path,
    *,
    require_rerun: bool = False,
) -> dict[str, Any]:
    """Validate a generated 18-result B1/B2/B3 Stage D smoke experiment."""

    experiment = Path(experiment_directory).expanduser().resolve()
    dataset = Path(dataset_path).expanduser().resolve()
    fixtures = Path(fixture_directory).expanduser().resolve()
    input_summary = validate_inputs(dataset, fixtures)
    tasks = _read_tasks(dataset)
    task_by_id = {str(task["id"]): task for task in tasks}
    pages = _read_json(fixtures / "pages.json")

    for name in ("summary.json", "summary.csv", "summary.md"):
        _require((experiment / name).is_file(), f"missing aggregate artifact: {name}")
    summary = _read_json(experiment / "summary.json")
    _require(isinstance(summary, dict), "summary.json must be an object")
    recomputed_summary = aggregate_experiment(experiment, write=False)
    persisted_comparable = {
        key: value for key, value in summary.items() if key != "generated_at"
    }
    recomputed_comparable = {
        key: value for key, value in recomputed_summary.items() if key != "generated_at"
    }
    _require(
        persisted_comparable == recomputed_comparable,
        "summary.json does not match strict aggregation of terminal results",
    )
    _require(
        summary.get("selected_result_count") == len(tasks) * len(SYSTEM_IDS),
        "summary must select exactly 18 task/system results",
    )
    _require(
        summary.get("incomplete_attempt_count") == 0,
        "canonical smoke experiment contains incomplete attempts",
    )
    _require(
        isinstance(summary.get("fairness_fingerprint"), str)
        and bool(summary["fairness_fingerprint"]),
        "summary lacks one shared fairness fingerprint",
    )

    rows = summary.get("results")
    _require(isinstance(rows, list) and len(rows) == 18, "summary rows must total 18")
    row_by_pair = {
        (str(row.get("system_id")), str(row.get("task_id"))): row
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_pairs = {
        (system_id, task_id) for system_id in SYSTEM_IDS for task_id in task_by_id
    }
    _require(
        set(row_by_pair) == expected_pairs,
        "summary does not contain the complete B1/B2/B3 x six task matrix",
    )

    with (experiment / "summary.csv").open(newline="", encoding="utf-8") as handle:
        csv_reader = csv.DictReader(handle)
        csv_rows = list(csv_reader)
    _require(len(csv_rows) == 18, "summary.csv must contain 18 result rows")
    expected_csv_rows = {
        (str(row["system_id"]), str(row["task_id"])): {
            key: (
                ""
                if value is None
                else (
                    str(value).lower()
                    if isinstance(value, bool)
                    else (
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                        )
                        if isinstance(value, (dict, list))
                        else str(value)
                    )
                )
            )
            for key, value in row.items()
        }
        for row in recomputed_summary["results"]
    }
    actual_csv_rows = {
        (str(row.get("system_id")), str(row.get("task_id"))): row for row in csv_rows
    }
    _require(
        actual_csv_rows == expected_csv_rows,
        "summary.csv does not match strict aggregation of terminal results",
    )
    markdown = (experiment / "summary.md").read_text(encoding="utf-8")
    _require(
        FIXTURE_SMOKE_DISCLAIMER in markdown
        and all(system_id in markdown for system_id in SYSTEM_IDS)
        and all(task_id in markdown for task_id in task_by_id),
        "summary.md lacks the exact smoke disclaimer or complete matrix",
    )

    results: dict[tuple[str, str], dict[str, Any]] = {}
    selected_attempts: dict[tuple[str, str], Path] = {}
    rerun_attempts = 0
    for system_id, task_id in sorted(expected_pairs):
        task = task_by_id[task_id]
        task_directory = experiment / system_id / task_id
        latest, result = _latest_result(
            task_directory,
            system_id=system_id,
        )
        attempts = [
            path
            for path in task_directory.glob("attempt-*")
            if path.is_dir() and (path / "result.json").is_file()
        ]
        if len(attempts) > 1:
            rerun_attempts += 1
        row = row_by_pair[(system_id, task_id)]
        _require(
            row.get("attempt") == latest.name,
            f"summary did not select latest attempt for {system_id}/{task_id}",
        )
        _require(
            result.get("system_id") == system_id and result.get("task_id") == task_id,
            f"result identity mismatch in {latest}",
        )
        _require(
            result.get("fixture_smoke") is True,
            f"{system_id}/{task_id} is not marked fixture_smoke",
        )
        config = result.get("resolved_config")
        _require(
            isinstance(config, Mapping)
            and config.get("backend_kind") == "fixture"
            and config.get("model", {}).get("provider") == "fixture"
            and config.get("tools", {}).get("search_backend") == "fixture-search"
            and config.get("tools", {}).get("fetch_backend") == "fixture-fetch",
            f"{system_id}/{task_id} did not remain fully offline",
        )
        _require(
            result.get("fairness_fingerprint") == summary["fairness_fingerprint"],
            f"{system_id}/{task_id} fairness fingerprint drifted",
        )
        _require(
            result.get("normalized_exact_match") is None
            and result.get("judge_score") is None
            and result.get("estimated_cost") is None,
            f"{system_id}/{task_id} invented an unavailable external metric",
        )
        for relative in STANDARD_ATTEMPT_FILES:
            _require(
                (latest / relative).is_file(),
                f"{system_id}/{task_id} lacks standard artifact {relative}",
            )
        _validate_expected(
            task=task,
            system_id=system_id,
            attempt=latest,
            result=result,
        )
        if system_id in SYSTEM_IDS[:2]:
            _require(
                result.get("evidence_count") is None
                and result.get("structural_subquestion_coverage") is None,
                f"{system_id}/{task_id} invented TongAgent-only metrics",
            )
        else:
            _validate_b3_provenance(
                task=task,
                attempt=latest,
                result=result,
                pages=pages,
            )
        results[(system_id, task_id)] = result
        selected_attempts[(system_id, task_id)] = latest

    run_ids = [str(result.get("run_id", "")) for result in results.values()]
    _require(
        all(run_ids) and len(set(run_ids)) == 18,
        "every selected system/task result must have a distinct non-empty run_id",
    )
    _validate_runtime_scenarios(results=results, attempts=selected_attempts)
    if require_rerun:
        _require(
            rerun_attempts >= 1,
            "rerun validation requested but no second terminal attempt exists",
        )
    _validate_command_logs(experiment, require_rerun=require_rerun)

    system_summaries = summary.get("systems")
    _require(
        isinstance(system_summaries, list) and len(system_summaries) == 3,
        "summary must contain three system comparisons",
    )
    for system_summary in system_summaries:
        _require(
            isinstance(system_summary, Mapping),
            "system summary entry must be an object",
        )
        system_id = str(system_summary.get("system_id"))
        selected = [
            result
            for (candidate, _), result in results.items()
            if candidate == system_id
        ]
        completion_distribution = Counter(
            str(result.get("completion_status")) for result in selected
        )
        failure_distribution = Counter(
            str(result.get("failure_type"))
            for result in selected
            if result.get("failure_type") is not None
        )
        _require(
            system_summary.get("completion_distribution")
            == dict(sorted(completion_distribution.items())),
            f"{system_id} completion distribution is inconsistent",
        )
        _require(
            system_summary.get("failure_distribution")
            == dict(sorted(failure_distribution.items())),
            f"{system_id} failure distribution is inconsistent",
        )

    return {
        **input_summary,
        "experiment_directory": str(experiment),
        "selected_result_count": 18,
        "system_count": 3,
        "rerun_task_count": rerun_attempts,
        "fairness_fingerprint": summary["fairness_fingerprint"],
        "validation_status": "passed",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate deterministic Stage D fixture smoke inputs/results."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--experiment-directory", type=Path)
    parser.add_argument("--require-rerun", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.experiment_directory is None:
        payload = validate_inputs(args.dataset, args.fixtures)
    else:
        payload = validate_experiment(
            args.experiment_directory,
            args.dataset,
            args.fixtures,
            require_rerun=args.require_rerun,
        )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
