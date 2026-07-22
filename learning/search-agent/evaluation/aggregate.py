"""Strict aggregation for process-isolated evaluation attempts."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from .execution import (
    EvaluationStateError,
    FairnessMismatchError,
    atomic_write_json,
    atomic_write_text,
    validate_persisted_result,
)
from .schema import RunResult


_RESULT_COLUMNS = (
    "system_id",
    "runtime_mode",
    "task_id",
    "backend_kind",
    "fixture_smoke",
    "attempt",
    "run_id",
    "completion_status",
    "failure_type",
    "budget_resource",
    "answer_status",
    "extracted_answer",
    "raw_whole_string_em",
    "standard_normalized_em",
    "normalized_exact_match",
    "judge_score",
    "wall_time_seconds",
    "tool_calls",
    "external_retrieval_calls",
    "internal_tool_calls",
    "search_calls",
    "fetch_calls",
    "relevant_searches",
    "citations",
    "evidence_count",
    "structural_subquestion_coverage",
    "sq_research_completion_rate",
    "draft_claim_count",
    "verified_claim_rate",
    "critical_claim_verified_rate",
    "unsupported_claim_rate",
    "coverage",
    "selective_accuracy",
    "risk",
    "atomic_fact_support_rate",
    "critical_fact_support_rate",
    "unsupported_answer_rate",
    "contradicted_answer_rate",
    "citation_precision",
    "answer_rate",
    "total_tokens",
    "estimated_cost",
    "config_fingerprint",
    "fairness_fingerprint",
    "artifact_directory",
)
FIXTURE_SMOKE_DISCLAIMER = (
    "Deterministic offline fixture smoke test; not a formal benchmark result."
)


def aggregate_experiment(
    experiment_directory: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Select latest valid terminal attempts and emit JSON/CSV/Markdown.

    Corrupt result markers are errors rather than silently omitted data.  A
    summary also refuses to mix fairness fingerprints, which prevents invalid
    B1/B2/B3 comparisons even if attempt directories were assembled manually.
    """

    experiment = Path(experiment_directory).expanduser().resolve()
    selected, incomplete_attempts = _select_latest_results(experiment)
    fingerprints = {result.fairness_fingerprint for _, result in selected}
    if len(fingerprints) > 1:
        raise FairnessMismatchError(
            "cannot aggregate mixed fairness fingerprints in one experiment"
        )
    fairness_fingerprint = next(iter(fingerprints), None)
    git_shas = {result.git_sha for _, result in selected}
    if len(git_shas) > 1:
        raise EvaluationStateError(
            "cannot aggregate results produced by mixed git SHAs"
        )
    git_sha = next(iter(git_shas), None)
    backend_kinds = {result.resolved_config.backend_kind for _, result in selected}
    backend_kind = next(iter(backend_kinds), None)
    runtime_modes = {result.runtime_mode for _, result in selected}
    if len(runtime_modes) > 1:
        raise FairnessMismatchError("cannot aggregate mixed runtime modes")
    runtime_mode = next(iter(runtime_modes), None)
    rows = [
        _result_row(result, attempt_directory.name)
        for attempt_directory, result in selected
    ]
    systems = _system_summaries(selected)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "fairness_fingerprint": fairness_fingerprint,
        "git_sha": git_sha,
        "backend_kind": backend_kind,
        "runtime_mode": runtime_mode,
        "fixture_smoke": (
            backend_kind == "fixture" if backend_kind is not None else None
        ),
        "benchmark_disclaimer": (
            FIXTURE_SMOKE_DISCLAIMER if backend_kind == "fixture" else None
        ),
        "selected_result_count": len(selected),
        "incomplete_attempt_count": incomplete_attempts,
        "results": rows,
        "systems": systems,
    }
    if write:
        experiment.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            experiment / "summary.json",
            payload,
            overwrite=True,
        )
        atomic_write_text(
            experiment / "summary.csv",
            _render_csv(rows),
            overwrite=True,
        )
        atomic_write_text(
            experiment / "summary.md",
            _render_markdown(payload),
            overwrite=True,
        )
    return payload


def _select_latest_results(
    experiment: Path,
) -> tuple[list[tuple[Path, RunResult]], int]:
    if not experiment.exists():
        return [], 0
    if experiment.is_symlink() or not experiment.is_dir():
        raise EvaluationStateError(
            f"experiment output is not a directory: {experiment}"
        )
    selected: list[tuple[Path, RunResult]] = []
    incomplete = 0
    for system_directory in sorted(experiment.iterdir(), key=lambda item: item.name):
        if not system_directory.is_dir() or system_directory.name.startswith("."):
            continue
        if system_directory.is_symlink():
            raise EvaluationStateError(
                f"system output is a symlink: {system_directory}"
            )
        for task_directory in sorted(
            system_directory.iterdir(), key=lambda item: item.name
        ):
            if not task_directory.is_dir() or task_directory.name.startswith("."):
                continue
            if task_directory.is_symlink():
                raise EvaluationStateError(
                    f"task output is a symlink: {task_directory}"
                )
            terminal: tuple[Path, RunResult] | None = None
            for attempt in _sorted_attempts(task_directory):
                result_path = attempt / "result.json"
                if not result_path.exists() and not result_path.is_symlink():
                    incomplete += 1
                    continue
                result = validate_persisted_result(
                    result_path,
                    system_id=system_directory.name,
                )
                if result.task_id != task_directory.name:
                    raise EvaluationStateError(
                        f"task directory/result mismatch in {result_path}"
                    )
                terminal = (attempt, result)
            if terminal is not None:
                selected.append(terminal)
    selected.sort(key=lambda item: (item[1].system_id, item[1].task_id))
    return selected, incomplete


def _sorted_attempts(task_directory: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for child in task_directory.iterdir():
        if not child.name.startswith("attempt-"):
            continue
        suffix = child.name.removeprefix("attempt-")
        if not suffix.isdigit():
            continue
        if child.is_symlink() or not child.is_dir():
            raise EvaluationStateError(f"invalid attempt directory: {child}")
        numbered.append((int(suffix), child))
    return [item[1] for item in sorted(numbered)]


def _result_row(result: RunResult, attempt: str) -> dict[str, Any]:
    workflow = result.workflow_metrics or {}
    return {
        "system_id": result.system_id,
        "runtime_mode": result.runtime_mode,
        "task_id": result.task_id,
        "backend_kind": result.resolved_config.backend_kind,
        "fixture_smoke": result.fixture_smoke,
        "attempt": attempt,
        "run_id": result.run_id,
        "completion_status": result.completion_status.value,
        "failure_type": (
            result.failure_type.value if result.failure_type is not None else None
        ),
        "budget_resource": (
            result.budget_resource.value if result.budget_resource is not None else None
        ),
        "answer_status": result.answer_status.value,
        "extracted_answer": result.extracted_answer,
        "raw_whole_string_em": result.raw_whole_string_em,
        "standard_normalized_em": result.standard_normalized_em,
        "normalized_exact_match": result.normalized_exact_match,
        "judge_score": result.judge_score,
        "wall_time_seconds": result.wall_time_seconds,
        "tool_calls": len(result.tool_calls),
        "external_retrieval_calls": result.external_retrieval_calls,
        "internal_tool_calls": result.internal_tool_calls,
        "search_calls": result.search_calls,
        "fetch_calls": result.fetch_calls,
        "relevant_searches": result.relevant_searches,
        "citations": len(result.citations),
        "evidence_count": result.evidence_count,
        "structural_subquestion_coverage": (result.structural_subquestion_coverage),
        "sq_research_completion_rate": workflow.get("sq_research_completion_rate"),
        "draft_claim_count": workflow.get("draft_claim_count"),
        "verified_claim_rate": workflow.get("verified_claim_rate"),
        "critical_claim_verified_rate": workflow.get("critical_claim_verified_rate"),
        "unsupported_claim_rate": workflow.get("unsupported_claim_rate"),
        "coverage": workflow.get("balanced_coverage"),
        "selective_accuracy": workflow.get("balanced_selective_accuracy"),
        "risk": workflow.get("balanced_risk"),
        "atomic_fact_support_rate": workflow.get("balanced_atomic_fact_support_rate"),
        "critical_fact_support_rate": workflow.get(
            "balanced_critical_fact_support_rate"
        ),
        "unsupported_answer_rate": workflow.get("balanced_unsupported_answer_rate"),
        "contradicted_answer_rate": workflow.get("balanced_contradicted_answer_rate"),
        "citation_precision": workflow.get("balanced_citation_precision"),
        "answer_rate": result.answer_rate,
        "total_tokens": (
            result.token_usage.total_tokens if result.token_usage is not None else None
        ),
        "estimated_cost": result.estimated_cost,
        "config_fingerprint": result.config_fingerprint,
        "fairness_fingerprint": result.fairness_fingerprint,
        "artifact_directory": result.artifact_directory,
    }


def _system_summaries(
    selected: list[tuple[Path, RunResult]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for _, result in selected:
        grouped[result.system_id].append(result)
    summaries: list[dict[str, Any]] = []
    for system_id in sorted(grouped):
        results = grouped[system_id]
        exact_values = [
            result.normalized_exact_match
            for result in results
            if result.normalized_exact_match is not None
        ]
        raw_exact_values = [
            result.raw_whole_string_em
            for result in results
            if result.raw_whole_string_em is not None
        ]
        standard_exact_values = [
            result.standard_normalized_em
            for result in results
            if result.standard_normalized_em is not None
        ]
        answer_values = [
            result.answer_rate for result in results if result.answer_rate is not None
        ]
        token_values = [
            result.token_usage.total_tokens
            for result in results
            if result.token_usage is not None
            and result.token_usage.total_tokens is not None
        ]
        search_values = [
            result.search_calls for result in results if result.search_calls is not None
        ]
        fetch_values = [
            result.fetch_calls for result in results if result.fetch_calls is not None
        ]
        relevant_values = [
            result.relevant_searches
            for result in results
            if result.relevant_searches is not None
        ]
        external_values = [
            result.external_retrieval_calls
            for result in results
            if result.external_retrieval_calls is not None
        ]
        internal_values = [
            result.internal_tool_calls
            for result in results
            if result.internal_tool_calls is not None
        ]
        cost_values = [
            result.estimated_cost
            for result in results
            if result.estimated_cost is not None
        ]
        workflow_values = {
            key: [
                value
                for result in results
                for value in [_numeric_workflow_metric(result, key)]
                if value is not None
            ]
            for key in (
                "sq_research_completion_rate",
                "draft_claim_count",
                "verified_claim_rate",
                "critical_claim_verified_rate",
                "unsupported_claim_rate",
                "balanced_coverage",
                "balanced_selective_accuracy",
                "balanced_risk",
                "balanced_atomic_fact_support_rate",
                "balanced_critical_fact_support_rate",
                "balanced_unsupported_answer_rate",
                "balanced_contradicted_answer_rate",
                "balanced_citation_precision",
                "balanced_answer_rate",
            )
        }
        failure_distribution = Counter(
            result.failure_type.value
            for result in results
            if result.failure_type is not None
        )
        completion_distribution = Counter(
            result.completion_status.value for result in results
        )
        summaries.append(
            {
                "system_id": system_id,
                "backend_kind": results[0].resolved_config.backend_kind,
                "runtime_mode": results[0].runtime_mode,
                "fixture_smoke": results[0].fixture_smoke,
                "runs": len(results),
                "completion_distribution": dict(
                    sorted(completion_distribution.items())
                ),
                "failure_distribution": dict(sorted(failure_distribution.items())),
                "exact_match_available": len(exact_values),
                "exact_matches": sum(value is True for value in exact_values),
                "normalized_exact_match_rate": (
                    sum(value is True for value in exact_values) / len(exact_values)
                    if exact_values
                    else None
                ),
                "raw_whole_string_em_rate": (
                    sum(value is True for value in raw_exact_values)
                    / len(raw_exact_values)
                    if raw_exact_values
                    else None
                ),
                "standard_normalized_em_rate": (
                    sum(value is True for value in standard_exact_values)
                    / len(standard_exact_values)
                    if standard_exact_values
                    else None
                ),
                "answer_rate": (
                    sum(value is True for value in answer_values) / len(answer_values)
                    if answer_values
                    else None
                ),
                "total_tool_calls": sum(len(result.tool_calls) for result in results),
                "mean_tool_calls": fmean(len(result.tool_calls) for result in results),
                "known_external_retrieval_runs": len(external_values),
                "total_external_retrieval_calls": (
                    sum(external_values)
                    if len(external_values) == len(results)
                    else None
                ),
                "known_internal_tool_runs": len(internal_values),
                "total_internal_tool_calls": (
                    sum(internal_values)
                    if len(internal_values) == len(results)
                    else None
                ),
                "known_search_call_runs": len(search_values),
                "total_search_calls": (
                    sum(search_values) if len(search_values) == len(results) else None
                ),
                "known_fetch_call_runs": len(fetch_values),
                "total_fetch_calls": (
                    sum(fetch_values) if len(fetch_values) == len(results) else None
                ),
                "known_relevant_search_runs": len(relevant_values),
                "total_relevant_searches": (
                    sum(relevant_values)
                    if len(relevant_values) == len(results)
                    else None
                ),
                "mean_wall_time_seconds": fmean(
                    result.wall_time_seconds for result in results
                ),
                "total_wall_time_seconds": sum(
                    result.wall_time_seconds for result in results
                ),
                "known_total_token_runs": len(token_values),
                "total_tokens": (
                    sum(token_values) if len(token_values) == len(results) else None
                ),
                "known_cost_runs": len(cost_values),
                "estimated_cost": (
                    sum(cost_values) if len(cost_values) == len(results) else None
                ),
                "workflow_metric_runs": {
                    key: len(values) for key, values in workflow_values.items()
                },
                "mean_workflow_metrics": {
                    key: fmean(values) if values else None
                    for key, values in workflow_values.items()
                },
            }
        )
    return summaries


def _numeric_workflow_metric(result: RunResult, key: str) -> float | None:
    """Return one optional numeric permissive metric without coercing N/A to 0."""

    value = (result.workflow_metrics or {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _render_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: "" if row.get(key) is None else _csv_value(row.get(key))
                for key in _RESULT_COLUMNS
            }
        )
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    return value


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation summary: {payload['experiment_id']}",
        "",
    ]
    if payload["fixture_smoke"] is True:
        lines.extend(
            [
                f"> {FIXTURE_SMOKE_DISCLAIMER}",
                "",
            ]
        )
    lines.extend(
        [
            f"- Backend kind: `{payload['backend_kind'] or '—'}`",
            f"- Selected terminal results: {payload['selected_result_count']}",
            f"- Incomplete attempts retained: {payload['incomplete_attempt_count']}",
            f"- Fairness fingerprint: `{payload['fairness_fingerprint'] or '—'}`",
            f"- Git SHA: `{payload['git_sha'] or '—'}`",
            "",
            "## System comparison",
            "",
            "| System | Backend | Smoke | Runs | Completed | Not completed | Raw EM | "
            "Standard EM | Answer rate | "
            "Mean tools | External retrieval | Internal tools | "
            "Mean wall time (s) | Failure distribution |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for system in payload["systems"]:
        completions = system["completion_distribution"]
        completed = completions.get("completed", 0)
        failed = sum(
            count for status, count in completions.items() if status != "completed"
        )
        failures = system["failure_distribution"]
        failure_text = (
            ", ".join(f"{key}: {value}" for key, value in failures.items())
            if failures
            else "—"
        )
        lines.append(
            "| {system} | {backend} | {smoke} | {runs} | {completed} | "
            "{failed} | {raw_exact} | {standard_exact} | {answer_rate} | "
            "{tools} | {external} | {internal} | {wall} | {failures} |".format(
                system=system["system_id"],
                backend=system["backend_kind"],
                smoke=_markdown_value(system["fixture_smoke"]),
                runs=system["runs"],
                completed=completed,
                failed=failed,
                raw_exact=_markdown_value(system["raw_whole_string_em_rate"]),
                standard_exact=_markdown_value(system["standard_normalized_em_rate"]),
                answer_rate=_markdown_value(system["answer_rate"]),
                tools=_markdown_value(system["mean_tool_calls"]),
                external=_markdown_value(system["total_external_retrieval_calls"]),
                internal=_markdown_value(system["total_internal_tool_calls"]),
                wall=_markdown_value(system["mean_wall_time_seconds"]),
                failures=failure_text,
            )
        )
    lines.extend(
        [
            "",
            "## Selected task results",
            "",
            "| System | Task | Attempt | Status | Failure | Budget | Answer status | "
            "Raw EM | Standard EM | Answer | Tools | External | Internal | Wall time (s) | Evidence | "
            "Structural coverage | Tokens |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            "| {system} | {task} | {attempt} | {status} | {failure} | "
            "{budget} | {answer_status} | {raw_exact} | {standard_exact} | {answered} | {tools} | {external} | "
            "{internal} | {wall} | {evidence} | {coverage} | "
            "{tokens} |".format(
                system=row["system_id"],
                task=row["task_id"],
                attempt=row["attempt"],
                status=row["completion_status"],
                failure=_markdown_value(row["failure_type"]),
                budget=_markdown_value(row["budget_resource"]),
                answer_status=row["answer_status"],
                raw_exact=_markdown_value(row["raw_whole_string_em"]),
                standard_exact=_markdown_value(row["standard_normalized_em"]),
                answered=_markdown_value(row["answer_rate"]),
                tools=row["tool_calls"],
                external=_markdown_value(row["external_retrieval_calls"]),
                internal=_markdown_value(row["internal_tool_calls"]),
                wall=_markdown_value(row["wall_time_seconds"]),
                evidence=_markdown_value(row["evidence_count"]),
                coverage=_markdown_value(row["structural_subquestion_coverage"]),
                tokens=_markdown_value(row["total_tokens"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
