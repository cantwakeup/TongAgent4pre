"""Provider routing, deterministic fusion, and run-scoped page acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from retrieval_providers import (
    ProviderSearchResponse,
    SearchProvider,
    SearchResult,
    default_search_providers,
    fetch_mediawiki_content,
    parse_wikipedia_url,
)
from retrieval_quality import assess_search_relevance, deterministic_query_rewrite
from retrieval_safety import (
    URLValidationError,
    canonicalize_http_url,
    validate_public_url,
)


RRF_CONSTANT = 60
MAX_CACHED_CONTENT_CHARS = 1_000_000
MAX_RETURNED_CONTENT_CHARS = 20_000
_TIER_ORDER = {"relevant": 2, "uncertain": 1, "irrelevant": 0}


class _VisibleTextParser(HTMLParser):
    """Extract inert visible text from provider-supplied HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


@dataclass
class FusedSearchResult:
    """One canonical URL after multi-provider fusion."""

    title: str
    url: str
    snippet: str
    provider: str
    provider_rank: int
    providers: list[str]
    provider_ranks: dict[str, int]
    raw_content: str | None
    metadata: dict[str, Any]
    relevance_score: int
    relevance_tier: str
    relevance_reason: str
    rejection_reason: str | None
    relevance_details: dict[str, Any]
    rrf_score: float
    final_rank: int = 0

    def public_dict(self) -> dict[str, Any]:
        """Return a compact candidate record safe for the Agent and probe."""

        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
            "engine": self.provider,
            "provider_rank": self.provider_rank,
            "providers": list(self.providers),
            "provider_ranks": dict(self.provider_ranks),
            "raw_content_available": bool(
                self.raw_content and self.raw_content.strip()
            ),
            "relevance_score": self.relevance_score,
            "relevance_tier": self.relevance_tier,
            "relevance_reason": self.relevance_reason,
            "rejection_reason": self.rejection_reason,
            "relevance_details": dict(self.relevance_details),
            "rrf_score": round(self.rrf_score, 8),
            "final_rank": self.final_rank,
            "metadata": dict(self.metadata),
        }


@dataclass
class SearchExecution:
    """Auditable result of one query across the provider chain."""

    query: str
    status: str
    provider_outcomes: list[ProviderSearchResponse]
    normalized_candidates: list[dict[str, Any]]
    results: list[FusedSearchResult]
    relevant_results: int
    uncertain_results: int
    rejected_results: int
    search_quality: str
    fallback_reasons: list[str]

    def public_dict(self) -> dict[str, Any]:
        """Return the stable JSON contract consumed by retrieval tools."""

        provider_success = any(
            outcome.status in {"success", "empty"} for outcome in self.provider_outcomes
        )
        return {
            "status": self.status,
            "query": self.query,
            "results": [result.public_dict() for result in self.results],
            "normalized_candidates": list(self.normalized_candidates),
            "provider_statuses": [
                outcome.public_dict() for outcome in self.provider_outcomes
            ],
            "engines": [
                outcome.provider
                for outcome in self.provider_outcomes
                if outcome.results
            ],
            "fallback_reason": (
                self.fallback_reasons[0] if self.fallback_reasons else None
            ),
            "fallback_reasons": list(self.fallback_reasons),
            "search_quality": self.search_quality,
            "relevant_results": self.relevant_results,
            "uncertain_results": self.uncertain_results,
            "rejected_results": self.rejected_results,
            "provider_success": provider_success,
            "nonempty_search": bool(self.results),
            "relevant_search": self.relevant_results > 0,
            "provider_failure": not provider_success,
            **(
                {"error": "all_search_providers_failed"}
                if self.status == "error"
                else {}
            ),
        }


@dataclass(frozen=True)
class CachedPage:
    """Provider-supplied raw content retained only for one retrieval session."""

    url: str
    title: str
    content: str
    provider: str
    stored_at: str


class SearchBroker:
    """Run the provider chain and fuse candidates deterministically."""

    def __init__(self, providers: Sequence[SearchProvider] | None = None) -> None:
        self.providers = list(providers or default_search_providers())

    def search(self, query: str, max_results: int) -> SearchExecution:
        """Search providers in order until enough relevant candidates exist."""

        result_limit = min(max(max_results, 1), 8)
        outcomes: list[ProviderSearchResponse] = []
        gathered: list[SearchResult] = []
        fallback_reasons: list[str] = []
        stopped = False

        for index, provider in enumerate(self.providers):
            if stopped:
                outcomes.append(
                    ProviderSearchResponse(
                        provider=provider.name,
                        query=query,
                        status="skipped",
                        failure_category="sufficient_candidates",
                    )
                )
                continue
            provider_query = (
                deterministic_query_rewrite(query)
                if provider.name == "bing_rss"
                else query
            )
            try:
                outcome = provider.search(provider_query, result_limit)
            except Exception as exc:
                outcome = ProviderSearchResponse(
                    provider=provider.name,
                    query=provider_query,
                    status="error",
                    failure_category="provider_error",
                    error_type=type(exc).__name__,
                )
            outcomes.append(outcome)
            gathered.extend(outcome.results)

            provisional, _ = _fuse_results(query, gathered)
            relevant_count = sum(
                result.relevance_tier == "relevant" for result in provisional
            )
            required_relevant = min(2, result_limit)
            if relevant_count >= required_relevant:
                stopped = True
            elif index + 1 < len(self.providers):
                fallback_reasons.append(_fallback_reason(outcome, relevant_count))

        fused, normalized = _fuse_results(query, gathered)
        relevant = [result for result in fused if result.relevance_tier == "relevant"]
        uncertain = [result for result in fused if result.relevance_tier == "uncertain"]
        irrelevant = [
            result for result in fused if result.relevance_tier == "irrelevant"
        ]
        if relevant:
            selected = [*relevant, *uncertain][:result_limit]
            quality = "relevant"
        elif uncertain:
            selected = uncertain[: min(2, result_limit)]
            quality = "uncertain_only"
        else:
            selected = []
            quality = "no_candidates"
        for rank, result in enumerate(selected, 1):
            result.final_rank = rank

        provider_success = any(
            outcome.status in {"success", "empty"} for outcome in outcomes
        )
        status = "success" if provider_success else "error"
        return SearchExecution(
            query=query,
            status=status,
            provider_outcomes=outcomes,
            normalized_candidates=normalized,
            results=selected,
            relevant_results=sum(
                result.relevance_tier == "relevant" for result in selected
            ),
            uncertain_results=sum(
                result.relevance_tier == "uncertain" for result in selected
            ),
            rejected_results=len(irrelevant)
            + sum(item.get("rejection_reason") == "unsafe_url" for item in normalized),
            search_quality=quality,
            fallback_reasons=fallback_reasons,
        )


class RetrievalSession:
    """Own one run's search broker, raw-content cache, and host cooldown."""

    def __init__(
        self,
        broker: SearchBroker | None = None,
        *,
        url_validator: Callable[[str], object] = validate_public_url,
        mediawiki_fetcher: Callable[[str, int], dict[str, Any]] = (
            fetch_mediawiki_content
        ),
    ) -> None:
        self.broker = broker or SearchBroker()
        self._url_validator = url_validator
        self._mediawiki_fetcher = mediawiki_fetcher
        self._cache: dict[str, CachedPage] = {}
        self._blocked_hosts: dict[str, str] = {}

    def search(self, query: str, max_results: int) -> SearchExecution:
        """Search and cache provider raw content without exposing it as evidence."""

        execution = self.broker.search(query, max_results)
        for outcome in execution.provider_outcomes:
            for result in outcome.results:
                self._cache_result(result)
        return execution

    def has_cached_content(self, url: str) -> bool:
        """Return whether this run has usable provider content for `url`."""

        try:
            canonical = canonicalize_http_url(url)
        except URLValidationError:
            return False
        cached = self._cache.get(canonical)
        return bool(cached and cached.content.strip())

    def fetch(
        self,
        url: str,
        max_chars: int,
        *,
        direct_fetch: Callable[[str, int], dict[str, Any]],
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Acquire one page via cache, direct HTTP, then MediaWiki fallback."""

        char_limit = min(max(max_chars, 1_000), MAX_RETURNED_CONTENT_CHARS)
        try:
            canonical = canonicalize_http_url(url)
        except URLValidationError as exc:
            return _unsafe_url_failure(url, exc)

        cached = self._cache.get(canonical) if use_cache else None
        if cached is not None and cached.content.strip():
            try:
                self._url_validator(canonical)
            except URLValidationError as exc:
                return _unsafe_url_failure(url, exc)
            return _cached_fetch_payload(cached, url, char_limit)

        host = _normalized_host(canonical)
        if host and host in self._blocked_hosts:
            taxonomy = self._blocked_hosts[host]
            return {
                "status": "suppressed",
                "url": canonical,
                "requested_url": url,
                "original_url": url,
                "final_url": canonical,
                "error": "host_in_access_cooldown",
                "failure_taxonomy": taxonomy,
                "failure_type": taxonomy,
                "retryable": False,
                "switch_source": True,
                "retry_with_another_source": True,
                "acquisition_method": "direct_http",
                "fallback_triggered": False,
            }

        try:
            direct = direct_fetch(canonical, char_limit)
        except Exception as exc:
            direct = {
                "status": "error",
                "url": canonical,
                "error": type(exc).__name__,
                "failure_taxonomy": "provider_error",
                "failure_type": "provider_error",
                "retryable": True,
            }
        direct.setdefault("requested_url", url)
        direct.setdefault("original_url", url)
        direct.setdefault("final_url", str(direct.get("url", canonical)))
        direct.setdefault("acquisition_method", "direct_http")
        direct.setdefault("content_provider", "direct_http")
        direct.setdefault("fallback_triggered", False)
        if direct.get("status") == "success":
            return direct

        taxonomy = str(
            direct.get(
                "failure_taxonomy",
                direct.get("failure_type", ""),
            )
        )
        wikipedia = parse_wikipedia_url(canonical)
        if wikipedia is not None and taxonomy in {
            "access_blocked",
            "rate_limited",
        }:
            fallback = self._mediawiki_fetcher(canonical, char_limit)
            fallback.setdefault("requested_url", url)
            fallback.setdefault("original_url", url)
            fallback.setdefault("final_url", canonical)
            fallback["fallback_triggered"] = True
            fallback["fallback_from"] = {
                "acquisition_method": "direct_http",
                "failure_taxonomy": taxonomy,
                "http_status": direct.get("http_status"),
            }
            if fallback.get("status") == "success":
                return fallback
            direct["fallback_triggered"] = True
            direct["fallback_failure"] = {
                "failure_taxonomy": fallback.get(
                    "failure_taxonomy",
                    fallback.get("failure_type", "provider_error"),
                ),
                "http_status": fallback.get("http_status"),
                "acquisition_method": "mediawiki_api",
                "details": fallback.get("wikimedia_core_failure"),
            }

        if taxonomy in {"access_blocked", "rate_limited"} and host:
            self._blocked_hosts[host] = taxonomy
        return direct

    def _cache_result(self, result: SearchResult) -> None:
        """Store normalized provider content under its canonical public URL."""

        if not isinstance(result.raw_content, str) or not result.raw_content.strip():
            return
        try:
            canonical = canonicalize_http_url(result.url)
        except URLValidationError:
            return
        normalized = normalize_provider_content(result.raw_content)
        if not normalized:
            return
        existing = self._cache.get(canonical)
        if existing is not None and len(existing.content) >= len(normalized):
            return
        self._cache[canonical] = CachedPage(
            url=canonical,
            title=result.title,
            content=normalized,
            provider=result.provider,
            stored_at=datetime.now(UTC).isoformat(),
        )


def build_default_session() -> RetrievalSession:
    """Create one isolated retrieval session for one Agent run."""

    return RetrievalSession()


def normalize_provider_content(content: str) -> str:
    """Normalize untrusted provider text without interpreting instructions."""

    bounded = content.replace("\x00", "")[:MAX_CACHED_CONTENT_CHARS]
    if "<html" in bounded[:500].casefold() or "<body" in bounded[:500].casefold():
        parser = _VisibleTextParser()
        try:
            parser.feed(bounded)
            bounded = "\n".join(parser.parts)
        except (ValueError, TypeError):
            return ""
    return "\n".join(line.strip() for line in bounded.splitlines() if line.strip())


def _fuse_results(
    query: str,
    results: Sequence[SearchResult],
) -> tuple[list[FusedSearchResult], list[dict[str, Any]]]:
    """Deduplicate URLs and rank them with relevance-aware RRF."""

    merged: dict[str, FusedSearchResult] = {}
    normalized: list[dict[str, Any]] = []
    for result in results:
        raw = {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "provider_rank": result.provider_rank,
        }
        try:
            canonical = canonicalize_http_url(result.url)
        except URLValidationError:
            normalized.append(
                {
                    **result.public_dict(),
                    "relevance_score": 0,
                    "relevance_tier": "irrelevant",
                    "relevance_reason": "unsafe_url",
                    "rejection_reason": "unsafe_url",
                }
            )
            continue
        relevance = assess_search_relevance(query, raw)
        candidate = {
            **result.public_dict(),
            "url": canonical,
            "relevance_score": int(relevance["score"]),
            "relevance_tier": str(relevance["tier"]),
            "relevance_reason": str(relevance["reason"]),
            "rejection_reason": relevance["rejection_reason"],
            "relevance_details": {
                key: relevance[key]
                for key in (
                    "gate",
                    "matched_terms",
                    "matched_entity_terms",
                    "matched_years",
                    "matched_numbers",
                    "provider_rank_adjustment",
                )
            },
        }
        normalized.append(candidate)
        rrf = 1.0 / (RRF_CONSTANT + max(result.provider_rank, 1))
        existing = merged.get(canonical)
        if existing is None:
            merged[canonical] = FusedSearchResult(
                title=result.title,
                url=canonical,
                snippet=result.snippet,
                provider=result.provider,
                provider_rank=result.provider_rank,
                providers=[result.provider],
                provider_ranks={result.provider: result.provider_rank},
                raw_content=result.raw_content,
                metadata=dict(result.metadata),
                relevance_score=int(relevance["score"]),
                relevance_tier=str(relevance["tier"]),
                relevance_reason=str(relevance["reason"]),
                rejection_reason=(
                    str(relevance["rejection_reason"])
                    if relevance["rejection_reason"] is not None
                    else None
                ),
                relevance_details=dict(candidate["relevance_details"]),
                rrf_score=rrf,
            )
            continue
        existing.rrf_score += rrf
        if result.provider not in existing.providers:
            existing.providers.append(result.provider)
        existing.provider_ranks[result.provider] = min(
            result.provider_rank,
            existing.provider_ranks.get(result.provider, result.provider_rank),
        )
        if result.raw_content and (
            not existing.raw_content
            or len(result.raw_content) > len(existing.raw_content)
        ):
            existing.raw_content = result.raw_content
            existing.provider = result.provider
            existing.provider_rank = result.provider_rank
            existing.metadata = dict(result.metadata)
        if int(relevance["score"]) > existing.relevance_score:
            existing.relevance_score = int(relevance["score"])
            existing.relevance_tier = str(relevance["tier"])
            existing.relevance_reason = str(relevance["reason"])
            existing.rejection_reason = (
                str(relevance["rejection_reason"])
                if relevance["rejection_reason"] is not None
                else None
            )
            existing.relevance_details = dict(candidate["relevance_details"])

    fused = sorted(
        merged.values(),
        key=lambda item: (
            -_TIER_ORDER.get(item.relevance_tier, 0),
            -item.relevance_score,
            -item.rrf_score,
            min(item.provider_ranks.values()),
            item.url,
        ),
    )
    return fused, normalized


def _fallback_reason(
    outcome: ProviderSearchResponse,
    relevant_count: int,
) -> str:
    if outcome.status == "not_configured":
        return f"{outcome.provider}_not_configured"
    if outcome.status == "skipped":
        category = outcome.failure_category or "not_applicable"
        return f"{outcome.provider}_{category}"
    if outcome.status == "error":
        category = outcome.failure_category or "provider_error"
        return f"{outcome.provider}_{category}"
    if not outcome.results:
        return f"{outcome.provider}_empty"
    if relevant_count == 0:
        return f"{outcome.provider}_low_relevance"
    return f"{outcome.provider}_insufficient_relevant"


def _cached_fetch_payload(
    cached: CachedPage,
    requested_url: str,
    char_limit: int,
) -> dict[str, Any]:
    content = cached.content[:char_limit]
    truncated = len(cached.content) > len(content)
    return {
        "status": "success",
        "url": cached.url,
        "requested_url": requested_url,
        "original_url": requested_url,
        "final_url": cached.url,
        "title": cached.title,
        "content": content,
        "content_chars": len(content),
        "content_length": len(cached.content),
        "observed_content_length": len(cached.content),
        "content_length_scope": "provider_raw_content_normalized_text",
        "downloaded_bytes": 0,
        "downloaded_bytes_scope": "provider_supplied_content",
        "http_content_length": None,
        "http_content_encoding": None,
        "http_status": None,
        "content_type": "text/plain",
        "fetched_at": datetime.now(UTC).isoformat(),
        "truncated": truncated,
        "truncation_reasons": (["returned_character_limit"] if truncated else []),
        "acquisition_method": "provider_raw_content",
        "content_provider": cached.provider,
        "provider_cached_at": cached.stored_at,
        "fallback_triggered": False,
        "untrusted_content": True,
    }


def _unsafe_url_failure(url: str, error: URLValidationError) -> dict[str, Any]:
    return {
        "status": "rejected",
        "url": url,
        "requested_url": url,
        "original_url": url,
        "final_url": None,
        "error": str(error),
        "failure_taxonomy": "unsafe_url",
        "failure_type": "unsafe_url",
        "security_reason": error.taxonomy,
        "retryable": False,
        "switch_source": True,
        "retry_with_another_source": True,
        "fallback_triggered": False,
    }


def _normalized_host(url: str) -> str:
    return (urlparse(url).hostname or "").casefold().removeprefix("www.")


__all__ = [
    "CachedPage",
    "FusedSearchResult",
    "RetrievalSession",
    "SearchBroker",
    "SearchExecution",
    "build_default_session",
    "normalize_provider_content",
]
