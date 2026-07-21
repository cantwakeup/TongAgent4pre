"""Permissive TongAgent research with post-hoc claim verification.

This module deliberately does not import or alter the strict FSM.  It shares
the same prepared runtime, semantic search/fetch wrappers, execution budget,
and EvidenceGraphStore, but moves Claim--Evidence--Source registration until
after all required subquestions have had a bounded chance to gather evidence.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from evidence_graph import normalize_evidence_text, validate_evidence_graph
from retrieval_quality import (
    assess_search_relevance,
    classify_query_task_type,
    deterministic_query_rewrite,
    normalize_atomic_search_query,
)

from ..budget import BudgetExceeded, ExecutionBudget
from ..config import ResolvedConfig
from ..offline import FixtureBackend
from ..schema import Citation, CompletionStatus, EvalTask, RunResult
from ..tracing import TraceCollector, sanitize_trace_value
from .common import (
    PreparedRuntime,
    build_run_result,
    prepare_runtime,
    write_intermediate_artifacts,
)


TaskType = Literal[
    "single_fact_lookup",
    "list_or_enumeration",
    "comparison",
    "date_or_numeric_lookup",
]
VerifiedStatus = Literal[
    "verified",
    "partially_supported",
    "unsupported",
    "contested",
]

_MAX_SOURCES_PER_QUERY = 3
_MAX_RESEARCH_ROUNDS = 2
_PASSAGE_CHARS = 720
_DANGEROUS_QUERY = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|\b(?:javascript|data|file):)"
)
_PREFIX = re.compile(
    r"(?i)^\s*(?:search(?:\s+for)?|find|look\s+up|research|query)\s*[:\-–—]?\s*"
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
_QUOTED = re.compile(r'"([^"]+)"')
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PermissiveSubquestion(_StrictModel):
    id: str
    question: str = Field(min_length=1)
    task_type: TaskType
    required_for_final_answer: bool = True


class PermissivePlan(_StrictModel):
    objective: str = Field(min_length=1)
    subquestions: list[PermissiveSubquestion] = Field(min_length=1, max_length=3)


class ResearchQueryDecision(_StrictModel):
    query: str = ""
    needs_second_round: bool = False
    missing_fact: str = ""


class ResearchSource(_StrictModel):
    source_id: str
    title: str
    url: str
    provider: str
    provider_rank: int = Field(ge=1)
    acquisition_method: str
    relevance_tier: str
    relevance_reason: str
    passage: str
    content_chars: int = Field(ge=0)
    fetch_status: str


class ResearchBundle(_StrictModel):
    query: str
    normalized_query: str
    sources: list[ResearchSource] = Field(default_factory=list)
    candidate_audit: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    broadened: bool = False
    query_validation_status: str
    query_normalizations: list[str] = Field(default_factory=list)
    query_rejection_reason: str | None = None


class ResearchNote(_StrictModel):
    subquestion_id: str
    claim_candidates: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    supporting_passages: list[str] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"


class DraftClaim(_StrictModel):
    claim_id: str
    text: str = Field(min_length=12, max_length=500)
    source_ids: list[str] = Field(default_factory=list)
    critical_for_final_answer: bool
    subquestion_id: str


class DraftAnswer(_StrictModel):
    claims: list[DraftClaim] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    proposed_answer: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    missing_information: list[str] = Field(default_factory=list)


class VerifiedClaim(_StrictModel):
    claim_id: str
    status: VerifiedStatus
    source_ids: list[str] = Field(default_factory=list)
    exact_quotes: list[str] = Field(default_factory=list)
    explanation: str
    canonical_claim_id: str | None = None


@dataclass(frozen=True)
class _QueryNormalization:
    original: str
    normalized: str
    status: str
    normalizations: tuple[str, ...]
    rejection_reason: str | None = None


def _json_tool_call(tool: BaseTool, arguments: Mapping[str, Any]) -> dict[str, Any]:
    raw = tool.invoke(dict(arguments))
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"status": "error", "error": "invalid_tool_payload"}
    return dict(parsed) if isinstance(parsed, Mapping) else {"status": "error"}


def _source_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""


def _passage(content: str, query: str) -> str:
    """Return a bounded literal excerpt, biased toward query-bearing sentences."""

    text = normalize_evidence_text(content)
    if len(text) <= _PASSAGE_CHARS:
        return text
    terms = {
        token.casefold()
        for token in _WORD.findall(query)
        if len(token) >= 3
        and token.casefold()
        not in {"the", "and", "for", "from", "with", "overview", "list"}
    }
    sentences = [item.strip() for item in _SENTENCE.split(text) if item.strip()]
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (-sum(term in pair[1].casefold() for term in terms), pair[0]),
    )
    selected: list[tuple[int, str]] = []
    used = 0
    for index, sentence in ranked:
        if used and used + len(sentence) + 1 > _PASSAGE_CHARS:
            continue
        selected.append((index, sentence))
        used += len(sentence) + 1
        if used >= _PASSAGE_CHARS:
            break
    return (
        " ".join(sentence for _, sentence in sorted(selected)) or text[:_PASSAGE_CHARS]
    )


def _soft_normalize_query(
    query: str,
    *,
    subquestion: PermissiveSubquestion,
) -> _QueryNormalization:
    """Repair ordinary query quality issues without blocking research."""

    original = " ".join(str(query).split())
    normalizations: list[str] = []
    candidate = _PREFIX.sub("", original)
    if candidate != original:
        normalizations.append("removed_explanatory_prefix")
    if not candidate:
        candidate = subquestion.question
        normalizations.append("used_subquestion_fallback")
    if _DANGEROUS_QUERY.search(candidate):
        fallback = " ".join(subquestion.question.split())
        if not fallback or _DANGEROUS_QUERY.search(fallback):
            return _QueryNormalization(
                original=original,
                normalized="",
                status="rejected",
                normalizations=tuple(normalizations),
                rejection_reason="dangerous_protocol_or_url_injection",
            )
        candidate = fallback
        normalizations.append("replaced_unsafe_query_with_subquestion")
    words = _WORD.findall(candidate)
    if not words:
        return _QueryNormalization(
            original=original,
            normalized="",
            status="rejected",
            normalizations=tuple(normalizations),
            rejection_reason="no_search_terms_after_normalization",
        )
    try:
        normalized = normalize_atomic_search_query(candidate, max_words=12)
    except ValueError:
        normalized = " ".join(words[:12])
        normalizations.append("bounded_query_length")
    if subquestion.task_type == "list_or_enumeration" and not any(
        token in normalized.casefold()
        for token in ("list", "overview", "discography", "timeline")
    ):
        normalized = f"{normalized} overview".strip()
        normalizations.append("added_enumeration_hint")
    if len(_WORD.findall(normalized)) > 12:
        normalized = " ".join(_WORD.findall(normalized)[:12])
        normalizations.append("bounded_query_length")
    return _QueryNormalization(
        original=original,
        normalized=normalized,
        status="normalized" if normalizations or normalized != original else "accepted",
        normalizations=tuple(normalizations),
    )


def _broaden_query(query: str, task_type: TaskType) -> str:
    # An exact-phrase query can be broadened deterministically without an
    # extra model call.  Dropping only quote constraints keeps the entity and
    # attribute intact while allowing providers with token-only matching to
    # return a candidate on the second attempt.
    unquoted = _QUOTED.sub(lambda match: match.group(1), query)
    unquoted = " ".join(unquoted.split())
    if unquoted and unquoted != query:
        return unquoted[:180]
    rewritten = deterministic_query_rewrite(query)
    if rewritten == query and task_type == "list_or_enumeration":
        rewritten = f"{query} overview"
    if task_type == "list_or_enumeration" and "overview" not in rewritten.casefold():
        rewritten = f"{rewritten} overview"
    widened = " ".join(rewritten.split())[:180]
    return widened if widened != query else ""


def _provider_rank(value: Any, fallback: int) -> int:
    """Return a stable positive provider rank for audit and tie breaking."""

    try:
        rank = int(value)
    except (TypeError, ValueError):
        return fallback
    return rank if rank >= 1 else fallback


def _set_candidate_fetch_status(
    candidate_audit: list[dict[str, Any]], url: str, status: str
) -> None:
    """Attach acquisition outcome to the already-persisted ranking record."""

    for item in candidate_audit:
        if item.get("url") == url:
            item["fetch_status"] = status
            return


def research_query(
    *,
    query: str,
    task_type: TaskType,
    subquestion: PermissiveSubquestion,
    search_tool: BaseTool,
    fetch_tool: BaseTool,
    max_sources: int = _MAX_SOURCES_PER_QUERY,
) -> ResearchBundle:
    """Search and fetch a bounded, canonical source bundle.

    ``web_search`` and ``fetch_url`` remain the sole provider-facing tools:
    their existing security validation, cache, source ledger, and external
    budgets are therefore authoritative.  This helper never treats snippets
    as page evidence.
    """

    normalized = _soft_normalize_query(query, subquestion=subquestion)
    if normalized.rejection_reason is not None:
        return ResearchBundle(
            query=normalized.original,
            normalized_query="",
            sources=[],
            candidate_audit=[],
            failures=[{"stage": "query", "reason": normalized.rejection_reason}],
            broadened=False,
            query_validation_status=normalized.status,
            query_normalizations=list(normalized.normalizations),
            query_rejection_reason=normalized.rejection_reason,
        )

    def search_once(value: str) -> dict[str, Any]:
        return _json_tool_call(search_tool, {"query": value, "max_results": 5})

    search_payload = search_once(normalized.normalized)
    results = search_payload.get("results", [])
    broadened = False
    if not isinstance(results, list) or not results:
        broad = _broaden_query(normalized.normalized, task_type)
        if broad and broad != normalized.normalized:
            broadened = True
            search_payload = search_once(broad)
            results = search_payload.get("results", [])
            normalized = _QueryNormalization(
                original=normalized.original,
                normalized=broad,
                status="broadened",
                normalizations=(
                    *normalized.normalizations,
                    "broadened_after_no_candidates",
                ),
            )
    raw_candidates = (
        [item for item in results if isinstance(item, Mapping)]
        if isinstance(results, list)
        else []
    )
    failures: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    for provider_rank, raw_candidate in enumerate(raw_candidates, start=1):
        candidate = dict(raw_candidate)
        candidate["provider_rank"] = _provider_rank(
            candidate.get("provider_rank"), provider_rank
        )
        assessment = assess_search_relevance(normalized.normalized, candidate)
        candidate.update(
            {
                "relevance_score": assessment["score"],
                "relevance_tier": assessment["tier"],
                "relevance_reason": assessment["reason"],
                "rejection_reason": assessment["rejection_reason"],
            }
        )
        candidate_audit.append(
            {
                "url": str(candidate.get("url", "")),
                "title": str(candidate.get("title", "")),
                "provider": str(candidate.get("provider", "unknown")),
                "provider_rank": _provider_rank(candidate.get("provider_rank"), 999),
                "relevance_score": int(assessment["score"]),
                "relevance_tier": str(assessment["tier"]),
                "relevance_reason": str(assessment["reason"]),
                "rejection_reason": assessment["rejection_reason"],
                "fetch_status": "not_attempted",
            }
        )
        if assessment["tier"] == "irrelevant":
            failures.append(
                {
                    "stage": "ranking",
                    "url": str(candidate.get("url", "")),
                    "reason": str(assessment["rejection_reason"] or "irrelevant"),
                }
            )
            continue
        candidates.append(candidate)
    ranked = sorted(
        candidates,
        key=lambda item: (
            0
            if str(item.get("relevance_tier", item.get("tier", ""))) == "relevant"
            else 1
            if str(item.get("relevance_tier", item.get("tier", ""))) == "uncertain"
            else 2,
            -int(item.get("relevance_score", 0) or 0),
            int(item.get("provider_rank", 999) or 999),
        ),
    )
    sources: list[ResearchSource] = []
    seen_urls: set[str] = set()
    failed_hosts: set[str] = set()
    limit = max(1, min(int(max_sources), _MAX_SOURCES_PER_QUERY))
    # Preserve the deterministic ranking while giving a healthy second host a
    # chance before consuming the remaining quota on the first host.  This is
    # especially important after an access-blocked response; it is not a
    # provider or ranking change, only bounded candidate acquisition policy.
    primary: list[Mapping[str, Any]] = []
    deferred_same_host: list[Mapping[str, Any]] = []
    candidate_hosts: set[str] = set()
    for candidate in ranked:
        host = _source_host(str(candidate.get("url", "")))
        if host and host in candidate_hosts:
            deferred_same_host.append(candidate)
        else:
            primary.append(candidate)
            if host:
                candidate_hosts.add(host)
    for candidate in [*primary, *deferred_same_host]:
        url = str(candidate.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        host = _source_host(url)
        # Do not burn fetch quota retrying an access-blocked host while another
        # candidate host remains available.  The underlying retrieval ledger
        # retains its own cooldown, and cached provider content remains valid.
        if (
            host
            and host in failed_hosts
            and any(
                _source_host(str(item.get("url", ""))) not in {"", host}
                for item in ranked
            )
        ):
            _set_candidate_fetch_status(candidate_audit, url, "skipped_failed_host")
            continue
        seen_urls.add(url)
        payload = _json_tool_call(fetch_tool, {"url": url, "max_chars": 12_000})
        _set_candidate_fetch_status(
            candidate_audit, url, str(payload.get("status", "error"))
        )
        if payload.get("status") != "success" or not payload.get("source_id"):
            if host and str(payload.get("failure_taxonomy", "")) in {
                "access_blocked",
                "rate_limited",
            }:
                failed_hosts.add(host)
            failures.append(
                {
                    "stage": "fetch",
                    "url": url,
                    "status": str(payload.get("status", "error")),
                    "reason": str(
                        payload.get("failure_taxonomy", payload.get("error", "unknown"))
                    ),
                }
            )
            continue
        content = str(payload.get("content", ""))
        passage = _passage(content, normalized.normalized)
        if not passage:
            failures.append(
                {"stage": "fetch", "url": url, "reason": "empty_page_content"}
            )
            continue
        sources.append(
            ResearchSource(
                source_id=str(payload["source_id"]),
                title=str(
                    payload.get("title", candidate.get("title", "Untitled source"))
                ),
                url=str(payload.get("url", url)),
                provider=str(
                    candidate.get("provider", search_payload.get("provider", "unknown"))
                ),
                provider_rank=_provider_rank(candidate.get("provider_rank"), 999),
                acquisition_method=str(
                    payload.get("acquisition_method", "direct_http")
                ),
                relevance_tier=str(
                    candidate.get("relevance_tier", candidate.get("tier", "uncertain"))
                ),
                relevance_reason=str(candidate.get("relevance_reason", "unknown")),
                passage=passage,
                content_chars=int(payload.get("content_chars", len(content)) or 0),
                fetch_status="success",
            )
        )
        if len(sources) >= limit:
            break
    if search_payload.get("status") != "success":
        failures.insert(
            0,
            {
                "stage": "search",
                "status": str(search_payload.get("status", "error")),
                "reason": str(search_payload.get("error", "provider_error")),
            },
        )
    return ResearchBundle(
        query=normalized.original,
        normalized_query=normalized.normalized,
        sources=sources,
        candidate_audit=candidate_audit,
        failures=failures,
        broadened=broadened,
        query_validation_status=normalized.status,
        query_normalizations=list(normalized.normalizations),
        query_rejection_reason=None,
    )


def _schema_variant(model: type[BaseModel], suffix: str) -> type[BaseModel]:
    return create_model(f"{model.__name__}_{suffix}", __base__=model)


def _accounted_structured(
    *,
    runtime: PreparedRuntime,
    schema: type[BaseModel],
    prompt: str,
    label: str,
    stage: Literal["planner", "research_step", "evidence_selection", "final_synthesis"],
) -> BaseModel | None:
    reservation = runtime.middleware.reserve_external_model_call(
        label=label,
        request_payload={
            "prompt": prompt,
            "response_schema": schema.model_json_schema(),
        },
        stage=stage,
    )
    try:
        response = (
            runtime.middleware.model_for_stage(runtime.model, stage)
            .with_structured_output(schema, include_raw=True)
            .invoke(prompt)
        )
    except BudgetExceeded:
        runtime.middleware.cancel_external_model_call(reservation, label=label)
        raise
    except Exception as exc:
        runtime.middleware.cancel_external_model_call(reservation, label=label)
        runtime.trace.record(
            "permissive_model_call_failed",
            label=label,
            exception_type=type(exc).__name__,
        )
        return None
    raw = response.get("raw") if isinstance(response, Mapping) else response
    runtime.middleware.record_external_model_response(
        raw, label=label, reservation=reservation
    )
    try:
        parsed = response.get("parsed") if isinstance(response, Mapping) else response
        return schema.model_validate(parsed)
    except Exception as exc:
        runtime.trace.record(
            "permissive_model_output_invalid",
            label=label,
            exception_type=type(exc).__name__,
            raw_output_summary=_safe_summary(response),
        )
        return None


def _safe_summary(value: Any) -> str:
    safe = sanitize_trace_value(value)
    try:
        text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(safe)
    return " ".join(text.split())[:480]


def _fallback_plan(task: EvalTask) -> PermissivePlan:
    return PermissivePlan(
        objective=task.question,
        subquestions=[
            PermissiveSubquestion(
                id="SQ1",
                question=task.question,
                task_type=cast(TaskType, classify_query_task_type(task.question)),
                required_for_final_answer=True,
            )
        ],
    )


def _validated_plan(
    candidate: BaseModel | None, task: EvalTask
) -> PermissivePlan | None:
    if candidate is None:
        return None
    try:
        plan = PermissivePlan.model_validate(candidate)
    except Exception:
        return None
    unique: set[str] = set()
    repaired: list[PermissiveSubquestion] = []
    for index, item in enumerate(plan.subquestions[:3], start=1):
        subquestion_id = item.id.strip() or f"SQ{index}"
        if subquestion_id in unique:
            subquestion_id = f"SQ{index}"
        unique.add(subquestion_id)
        repaired.append(item.model_copy(update={"id": subquestion_id}))
    return plan.model_copy(update={"subquestions": repaired}) if repaired else None


def _plan(*, runtime: PreparedRuntime, task: EvalTask) -> tuple[PermissivePlan, bool]:
    prompt = f"""Create 1-3 atomic required web-research subquestions for this task.
Do not place an entire multi-hop calculation in one search query. For enumerations
prefer a list, overview, discography, or timeline lookup. Return the schema only.

Question: {task.question}"""
    first = _validated_plan(
        _accounted_structured(
            runtime=runtime,
            schema=PermissivePlan,
            prompt=prompt,
            label="tongagent.permissive.planner.initial",
            stage="planner",
        ),
        task,
    )
    if first is not None:
        return first, False
    repaired = _validated_plan(
        _accounted_structured(
            runtime=runtime,
            schema=_schema_variant(PermissivePlan, "Repair"),
            prompt=(
                "Return only a valid permissive research plan matching the required "
                f"schema for this question: {task.question}"
            ),
            label="tongagent.permissive.planner.repair",
            stage="planner",
        ),
        task,
    )
    return (repaired or _fallback_plan(task)), True


def _query_decision(
    *,
    runtime: PreparedRuntime,
    subquestion: PermissiveSubquestion,
    round_number: int,
    previous: ResearchBundle | None,
) -> ResearchQueryDecision:
    previous_summary = (
        "No previous research round."
        if previous is None
        else json.dumps(
            {
                "sources": [item.model_dump(mode="json") for item in previous.sources],
                "failures": previous.failures,
            },
            ensure_ascii=False,
        )[:3_000]
    )
    schema = _schema_variant(ResearchQueryDecision, f"{subquestion.id}_R{round_number}")
    response = _accounted_structured(
        runtime=runtime,
        schema=schema,
        prompt=f"""Generate one short public-web search query for this atomic subquestion.
Use an entity plus one attribute. Do not output URLs, a final answer, or a full
reasoning chain. This is research round {round_number} of at most two.

Subquestion: {subquestion.question}
Task type: {subquestion.task_type}
Previous round: {previous_summary}""",
        label=f"tongagent.permissive.query.{subquestion.id}.r{round_number}",
        stage="research_step",
    )
    if response is None:
        return ResearchQueryDecision(
            query=subquestion.question, needs_second_round=False
        )
    return ResearchQueryDecision.model_validate(response)


def _fallback_note(
    subquestion: PermissiveSubquestion,
    bundles: Sequence[ResearchBundle],
) -> ResearchNote:
    sources = [source for bundle in bundles for source in bundle.sources]
    passages = [source.passage for source in sources]
    candidates = [
        f"Source {source.source_id} provides retrieved context for {subquestion.question}"
        for source in sources[:2]
    ]
    return ResearchNote(
        subquestion_id=subquestion.id,
        claim_candidates=candidates,
        source_ids=[source.source_id for source in sources],
        supporting_passages=passages[:3],
        unresolved_points=[] if sources else ["No successfully fetched source"],
        confidence="medium" if sources else "low",
    )


def _note(
    *,
    runtime: PreparedRuntime,
    subquestion: PermissiveSubquestion,
    bundles: Sequence[ResearchBundle],
) -> ResearchNote:
    sources = [source for bundle in bundles for source in bundle.sources]
    schema = _schema_variant(ResearchNote, subquestion.id)
    response = _accounted_structured(
        runtime=runtime,
        schema=schema,
        prompt=(
            "Create a research note from only these fetched source passages. Every "
            "claim candidate must name at least one supplied source ID. Do not invent "
            "facts or URLs.\n\n"
            f"Subquestion: {subquestion.question}\n"
            + json.dumps(
                [source.model_dump(mode="json") for source in sources],
                ensure_ascii=False,
            )
        )[:8_000],
        label=f"tongagent.permissive.note.{subquestion.id}",
        stage="evidence_selection",
    )
    if response is None:
        return _fallback_note(subquestion, bundles)
    note = ResearchNote.model_validate(response)
    valid_ids = {source.source_id for source in sources}
    accepted_ids = [item for item in note.source_ids if item in valid_ids]
    if sources and not accepted_ids:
        # A malformed or hallucinated source ID must not turn otherwise usable
        # fetched material into an artificial missing-SQ failure.  Preserve a
        # conservative code-generated note; its claim candidates deliberately
        # say only that the retrieved source provides context.
        return _fallback_note(subquestion, bundles)
    claims = [item for item in note.claim_candidates if item.strip() and accepted_ids]
    passages = [
        item
        for item in note.supporting_passages
        if item in {s.passage for s in sources}
    ]
    return note.model_copy(
        update={
            "subquestion_id": subquestion.id,
            "source_ids": list(dict.fromkeys(accepted_ids)),
            "claim_candidates": claims,
            "supporting_passages": passages,
        }
    )


def _fallback_draft(notes: Sequence[ResearchNote]) -> DraftAnswer:
    claims: list[DraftClaim] = []
    for note in notes:
        for index, text in enumerate(note.claim_candidates[:2], start=1):
            if len(normalize_evidence_text(text)) < 12 or not note.source_ids:
                continue
            claims.append(
                DraftClaim(
                    claim_id=f"D{len(claims) + 1}",
                    text=normalize_evidence_text(text),
                    source_ids=list(note.source_ids),
                    critical_for_final_answer=index == 1,
                    subquestion_id=note.subquestion_id,
                )
            )
    return DraftAnswer(
        claims=claims,
        reasoning_steps=[],
        proposed_answer=None,
        confidence="low",
        missing_information=[
            note.subquestion_id for note in notes if not note.source_ids
        ],
    )


def _draft(
    *, runtime: PreparedRuntime, task: EvalTask, notes: Sequence[ResearchNote]
) -> DraftAnswer:
    response = _accounted_structured(
        runtime=runtime,
        schema=DraftAnswer,
        prompt=(
            "Produce a source-grounded draft answer. Each claim needs one or more "
            "existing source IDs and its subquestion ID. Mark claims needed for the "
            "final answer as critical. Include intermediate values for calculations. "
            "Never use model memory or invent a source.\n\n"
            f"Question: {task.question}\nResearch notes:\n"
            + json.dumps(
                [item.model_dump(mode="json") for item in notes], ensure_ascii=False
            )
        )[:12_000],
        label="tongagent.permissive.draft",
        stage="final_synthesis",
    )
    if response is None:
        return _fallback_draft(notes)
    draft = DraftAnswer.model_validate(response)
    valid_by_sq = {note.subquestion_id: set(note.source_ids) for note in notes}
    claims = []
    for item in draft.claims:
        allowed_source_ids = valid_by_sq.get(item.subquestion_id, set())
        source_ids = [
            source for source in item.source_ids if source in allowed_source_ids
        ]
        if source_ids:
            claims.append(item.model_copy(update={"source_ids": source_ids}))
    if not claims and any(note.source_ids for note in notes):
        return _fallback_draft(notes)
    return draft.model_copy(update={"claims": claims})


def _meaningful_tokens(value: str) -> set[str]:
    return {
        item.casefold()
        for item in _WORD.findall(value)
        if len(item) >= 3
        and item.casefold() not in {"the", "and", "with", "from", "that"}
    }


def _quote_for_claim(claim: str, source: ResearchSource) -> tuple[str | None, bool]:
    tokens = _meaningful_tokens(claim)
    claim_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", claim))
    candidates = [
        item.strip()
        for item in _SENTENCE.split(source.passage)
        if len(item.strip()) >= 12
    ]
    scored = sorted(
        candidates,
        key=lambda item: (-len(tokens.intersection(_meaningful_tokens(item))), item),
    )
    if not scored:
        return None, False
    best = scored[0]
    source_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", best))
    contradicts = bool(
        claim_numbers
        and source_numbers
        and not claim_numbers.intersection(source_numbers)
    )
    if not tokens.intersection(_meaningful_tokens(best)):
        return None, contradicts
    return best, contradicts


def _verify(
    *,
    runtime: PreparedRuntime,
    draft: DraftAnswer,
    sources: Mapping[str, ResearchSource],
) -> list[VerifiedClaim]:
    verified: list[VerifiedClaim] = []
    for claim in draft.claims:
        support: list[tuple[ResearchSource, str]] = []
        contradictions: list[tuple[ResearchSource, str]] = []
        for source_id in claim.source_ids:
            source = sources.get(source_id)
            if source is None:
                continue
            quote, contradicts = _quote_for_claim(claim.text, source)
            if quote is None:
                continue
            if contradicts:
                contradictions.append((source, quote))
            else:
                support.append((source, quote))
        if support and contradictions:
            status: VerifiedStatus = "contested"
            selected = support + contradictions
            explanation = "Fetched sources contain incompatible numeric or date values."
        elif support:
            status = "verified"
            selected = support
            explanation = (
                "A continuous exact quote was found in a canonical fetched page."
            )
        elif any(source_id in sources for source_id in claim.source_ids):
            status = "partially_supported"
            selected = []
            explanation = "A fetched source is associated, but no precise continuous quote matched."
        else:
            status = "unsupported"
            selected = []
            explanation = "No canonical fetched source supports this claim."
        canonical_claim_id: str | None = None
        # Graph registration is intentionally post-hoc.  A quote was selected
        # from the page passage, and EvidenceGraphStore rechecks it against its
        # cached canonical page before assigning C#/E# IDs.
        registered: list[tuple[ResearchSource, str]] = []
        for source, quote in selected:
            try:
                record = runtime.research_budget.record_evidence(
                    source_id=source.source_id,
                    claim=claim.text,
                    quote=quote,
                    stance="contradicts"
                    if (source, quote) in contradictions
                    else "supports",
                    claim_id=canonical_claim_id or "",
                )
            except Exception as exc:
                runtime.trace.record(
                    "permissive_evidence_registration_failed",
                    draft_claim_id=claim.claim_id,
                    source_id=source.source_id,
                    exception_type=type(exc).__name__,
                )
                continue
            canonical_claim_id = str(record["claim"]["claim_id"])
            registered.append((source, quote))
        # A verifier result cannot claim exact verification unless the shared
        # EvidenceGraphStore also accepted the literal quote against its
        # canonical page cache.  This keeps the graph the final source of
        # truth without using it as a research-time gate.
        if selected and not registered:
            status = "partially_supported"
            explanation = (
                "A source passage matched, but canonical evidence registration failed."
            )
        verified.append(
            VerifiedClaim(
                claim_id=claim.claim_id,
                status=status,
                source_ids=[source.source_id for source, _ in registered]
                or list(claim.source_ids),
                exact_quotes=[quote for _, quote in registered],
                explanation=explanation,
                canonical_claim_id=canonical_claim_id,
            )
        )
    return verified


def _finalize(
    *,
    task: EvalTask,
    draft: DraftAnswer,
    notes: Sequence[ResearchNote],
    verified: Sequence[VerifiedClaim],
    config: ResolvedConfig,
) -> tuple[str, str, list[str]]:
    statuses = {item.claim_id: item for item in verified}
    critical = [item for item in draft.claims if item.critical_for_final_answer]
    unsupported = [
        item.claim_id
        for item in critical
        if statuses.get(item.claim_id) is None
        or statuses[item.claim_id].status in {"unsupported", "contested"}
    ]
    weak = [
        item.claim_id
        for item in critical
        if statuses.get(item.claim_id) is not None
        and statuses[item.claim_id].status == "partially_supported"
    ]
    missing_sq = [note.subquestion_id for note in notes if not note.source_ids]
    # The model may not quietly evade verification by emitting no critical
    # claims.  A final answer needs at least one source-grounded core claim.
    allowed = (
        bool(critical)
        and not unsupported
        and not missing_sq
        and bool(draft.proposed_answer)
    )
    if weak and config.permissive_workflow.require_exact_quote_for_core_claims:
        allowed = False
    if weak and not config.permissive_workflow.allow_low_confidence_answer:
        allowed = False
    if not config.permissive_workflow.enable_posthoc_verifier:
        allowed = bool(draft.proposed_answer) and not missing_sq
        unsupported = []
        weak = []
    if not allowed:
        missing = (
            ", ".join(sorted(set(missing_sq + unsupported))) or "verified core evidence"
        )
        return (
            "## Answer\nINSUFFICIENT_EVIDENCE\nABSTAIN\n"
            f"- Missing or unsupported: {missing}\n"
            "FINAL_ANSWER: ABSTAIN",
            "abstain",
            unsupported,
        )
    citations = sorted(
        {source_id for claim in critical for source_id in claim.source_ids}
    )
    confidence = "low" if weak else draft.confidence
    caveat = (
        "\n- Low confidence: one or more core claims have source support but no exact quote."
        if weak
        else ""
    )
    answer = str(draft.proposed_answer).strip()
    return (
        "## Answer\n"
        f"{answer}\n\n"
        f"Sources: {' '.join(f'[{item}]' for item in citations)}\n"
        f"Confidence: {confidence}.{caveat}\n"
        f"FINAL_ANSWER: {answer}",
        "answer_low_confidence" if weak else "answer",
        [],
    )


def _citations(runtime: PreparedRuntime) -> list[Citation]:
    snapshot = runtime.research_budget.snapshot()
    claims = {
        str(item.get("claim_id", "")): item for item in snapshot.get("claims", [])
    }
    return [
        Citation(
            citation_id=str(item["evidence_id"]),
            source_id=str(item["source_id"]),
            url=str(item["url"]),
            title=str(item.get("title", "")) or None,
            quote=str(item["quote"]),
            claim=str(claims.get(str(item["claim_id"]), {}).get("text", "")) or None,
            metadata={"claim_id": str(item["claim_id"]), "stance": str(item["stance"])},
        )
        for item in snapshot.get("evidence_units", [])
        if isinstance(item, Mapping) and item.get("evidence_id") and item.get("url")
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_permissive_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> RunResult:
    """Run PLAN → RESEARCH_SQ → NOTE → DRAFT → VERIFY → FINALIZE."""

    artifact_directory = Path(resolved_config.artifact_directory).expanduser()
    native_directory = artifact_directory / "native" / "tongagent"
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    trace = TraceCollector()
    execution_budget = ExecutionBudget(resolved_config.budget)
    run_id = f"tongagent-{task.id}-{uuid.uuid4().hex}"
    runtime: PreparedRuntime | None = None
    caught: Exception | None = None
    plan: PermissivePlan | None = None
    notes: list[ResearchNote] = []
    bundles: dict[str, list[ResearchBundle]] = {}
    verified: list[VerifiedClaim] = []
    draft = DraftAnswer()
    final_answer = "FINAL_ANSWER: ABSTAIN"
    final_status = "abstain"
    removed_claims: list[str] = []
    try:
        runtime = prepare_runtime(
            task,
            resolved_config,
            system_id="tongagent",
            execution_budget=execution_budget,
            trace=trace,
            injected_backend=fixture_backend,
            injected_model=model,
            semantic_policy=None,
            semantic_strategy="fixed",
            enable_tongagent_evidence_state=True,
            enable_tongagent_token_control=True,
            enable_tongagent_context_compaction=True,
            search_query_normalizer=None,
            reserve_final_synthesis=False,
        )
        trace.record("permissive_phase", phase="PLAN", runtime_mode="permissive")
        plan, planner_repaired = _plan(runtime=runtime, task=task)
        runtime.research_budget.configure_subquestions(
            [item.id for item in plan.subquestions]
        )
        runtime.middleware.configure_token_partitions(
            [item.id for item in plan.subquestions]
        )
        all_sources: dict[str, ResearchSource] = {}
        for subquestion in plan.subquestions:
            runtime.research_budget.activate_subquestion(subquestion.id)
            runtime.middleware.activate_token_subquestion(subquestion.id)
            trace.record(
                "permissive_phase", phase="RESEARCH_SQ", subquestion_id=subquestion.id
            )
            local_bundles: list[ResearchBundle] = []
            for round_number in range(1, _MAX_RESEARCH_ROUNDS + 1):
                decision = _query_decision(
                    runtime=runtime,
                    subquestion=subquestion,
                    round_number=round_number,
                    previous=local_bundles[-1] if local_bundles else None,
                )
                bundle = research_query(
                    query=decision.query,
                    task_type=subquestion.task_type,
                    subquestion=subquestion,
                    search_tool=next(
                        tool for tool in runtime.tools if tool.name == "web_search"
                    ),
                    fetch_tool=next(
                        tool for tool in runtime.tools if tool.name == "fetch_url"
                    ),
                    max_sources=2
                    if len(plan.subquestions) >= 3
                    else _MAX_SOURCES_PER_QUERY,
                )
                local_bundles.append(bundle)
                all_sources.update({item.source_id: item for item in bundle.sources})
                trace.record(
                    "permissive_research_bundle",
                    subquestion_id=subquestion.id,
                    round_number=round_number,
                    query_validation_status=bundle.query_validation_status,
                    query_normalizations=bundle.query_normalizations,
                    query_broadened=bundle.broadened,
                    source_count=len(bundle.sources),
                    failure_count=len(bundle.failures),
                )
                # A second, narrow round is allowed either when the model
                # explicitly identifies a missing fact or when round one
                # found no usable source.  It never blocks the next SQ.
                if round_number == _MAX_RESEARCH_ROUNDS or (
                    bundle.sources and not decision.needs_second_round
                ):
                    break
            bundles[subquestion.id] = local_bundles
            trace.record(
                "permissive_phase",
                phase="SAVE_RESEARCH_NOTE",
                subquestion_id=subquestion.id,
            )
            notes.append(
                _note(runtime=runtime, subquestion=subquestion, bundles=local_bundles)
            )
        trace.record("permissive_phase", phase="SYNTHESIZE_DRAFT")
        draft = _draft(runtime=runtime, task=task, notes=notes)
        trace.record("permissive_phase", phase="VERIFY_DRAFT")
        if runtime is not None:
            verified = _verify(runtime=runtime, draft=draft, sources=all_sources)
        trace.record("permissive_phase", phase="FINALIZE")
        final_answer, final_status, removed_claims = _finalize(
            task=task,
            draft=draft,
            notes=notes,
            verified=verified,
            config=resolved_config,
        )
        trace.record(
            "permissive_finalized",
            answer_status=final_status,
            planner_repaired=planner_repaired,
            verifier_removed_claims=removed_claims,
        )
    except Exception as exc:
        caught = exc
        trace.record(
            "run_exception",
            phase="permissive",
            exception_type=type(exc).__name__,
        )
    finally:
        finished_at = datetime.now(UTC)
        wall_time = max(0.0, time.perf_counter() - started)
        ledger = runtime.research_budget.snapshot() if runtime is not None else {}
        graph_errors = validate_evidence_graph(ledger) if runtime is not None else []
        verified_count = sum(item.status == "verified" for item in verified)
        critical = [item for item in draft.claims if item.critical_for_final_answer]
        critical_verified = sum(
            next(
                (
                    check.status == "verified"
                    for check in verified
                    if check.claim_id == item.claim_id
                ),
                False,
            )
            for item in critical
        )
        sq_completion = (
            sum(bool(note.source_ids) or bool(note.unresolved_points) for note in notes)
            / len(plan.subquestions)
            if plan is not None and plan.subquestions
            else 0.0
        )
        workflow_metrics: dict[str, Any] = {
            "runtime_mode": "permissive",
            "sq_research_completion_rate": sq_completion,
            "draft_claim_count": len(draft.claims),
            "verified_claim_rate": verified_count / len(draft.claims)
            if draft.claims
            else 0.0,
            "critical_claim_verified_rate": critical_verified / len(critical)
            if critical
            else 0.0,
            "unsupported_claim_rate": sum(
                item.status == "unsupported" for item in verified
            )
            / len(verified)
            if verified
            else 0.0,
            "answer_before_verification": draft.proposed_answer,
            "answer_after_verification": final_answer,
            "verifier_removed_claims": removed_claims,
            "research_rounds_per_sq": {
                key: len(value) for key, value in bundles.items()
            },
            "claim_status_counts": {
                status: sum(item.status == status for item in verified)
                for status in (
                    "verified",
                    "partially_supported",
                    "unsupported",
                    "contested",
                )
            },
            "evidence_graph_errors": graph_errors,
        }
        native_output = {"final_answer": final_answer}
        result = build_run_result(
            run_id=run_id,
            task=task,
            resolved_config=resolved_config,
            system_id="tongagent",
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=wall_time,
            runtime=runtime,
            execution_budget=execution_budget,
            trace=trace,
            native_output=native_output,
            caught=caught,
            evidence_count=len(ledger.get("evidence_units", []))
            if runtime is not None
            else None,
            structural_subquestion_coverage=sq_completion if plan is not None else None,
            final_answer_override=final_answer,
        )
        if caught is None:
            result = result.model_copy(
                update={
                    "completion_status": (
                        CompletionStatus.PARTIAL
                        if final_status == "abstain"
                        else CompletionStatus.COMPLETED
                    ),
                    "citations": _citations(runtime) if runtime is not None else [],
                    "workflow_metrics": cast("dict[str, Any]", workflow_metrics),
                }
            )
        write_intermediate_artifacts(
            artifact_directory,
            task=task,
            resolved_config=resolved_config,
            runtime=runtime,
            execution_budget=execution_budget,
            trace=trace,
            native_output=native_output,
            result=result,
            caught=caught,
        )
        _write_json(
            native_directory / "permissive_workflow.json",
            {
                "runtime_mode": "permissive",
                "plan": plan.model_dump(mode="json") if plan is not None else None,
                "research_bundles": {
                    key: [item.model_dump(mode="json") for item in value]
                    for key, value in bundles.items()
                },
                "research_notes": [item.model_dump(mode="json") for item in notes],
                "draft": draft.model_dump(mode="json"),
                "verified_claims": [item.model_dump(mode="json") for item in verified],
                "final_answer_status": final_status,
                "workflow_metrics": workflow_metrics,
                "phase": "FINALIZE",
                "state_integrity_errors": graph_errors,
            },
        )
        return RunResult.model_validate(result.model_dump(mode="python"))


def preflight_permissive_workflow(
    task: EvalTask,
    resolved_config: ResolvedConfig,
    *,
    fixture_backend: FixtureBackend | None = None,
    model: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Prepare the permissive runtime only; no model or provider is invoked."""

    execution_budget = ExecutionBudget(resolved_config.budget)
    trace = TraceCollector()
    runtime = prepare_runtime(
        task,
        resolved_config,
        system_id="tongagent",
        execution_budget=execution_budget,
        trace=trace,
        injected_backend=fixture_backend,
        injected_model=model,
        semantic_strategy="fixed",
        enable_tongagent_evidence_state=True,
        enable_tongagent_token_control=True,
        enable_tongagent_context_compaction=True,
        reserve_final_synthesis=False,
    )
    return {
        "task_id": task.id,
        "runtime_mode": "permissive",
        "runtime_tools": [tool.name for tool in runtime.tools],
        "model_invocations": 0,
        "agent_constructed": True,
    }


__all__ = [
    "DraftAnswer",
    "PermissivePlan",
    "PermissiveSubquestion",
    "ResearchBundle",
    "ResearchNote",
    "ResearchQueryDecision",
    "ResearchSource",
    "VerifiedClaim",
    "preflight_permissive_workflow",
    "research_query",
    "run_permissive_workflow",
]
