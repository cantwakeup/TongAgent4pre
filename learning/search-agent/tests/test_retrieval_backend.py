"""Offline tests for the unified retrieval provider and acquisition backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import httpx

import retrieval_providers
from agent_policy import EffortPolicy
from evaluation.systems.common import _live_raw_tools
from retrieval_backend import RetrievalSession, SearchBroker
from retrieval_providers import (
    MediaWikiSearchProvider,
    ProviderRequestError,
    ProviderSearchResponse,
    SearchResult,
    TavilySearchProvider,
    default_search_providers,
    fetch_mediawiki_content,
)
from retrieval_quality import assess_search_relevance
from retrieval_safety import ValidatedURL
from search_agent import build_budgeted_tools, create_retrieval_tools


@dataclass
class StubProvider:
    """Return one fixed provider response and retain call observations."""

    name: str
    response: ProviderSearchResponse
    calls: int = 0

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        self.calls += 1
        assert max_results <= 8
        return ProviderSearchResponse(
            provider=self.name,
            query=query,
            status=self.response.status,
            results=list(self.response.results[:max_results]),
            http_status=self.response.http_status,
            failure_category=self.response.failure_category,
            error_type=self.response.error_type,
            metadata=dict(self.response.metadata),
        )


def _result(
    *,
    provider: str,
    rank: int,
    url: str = "https://en.wikipedia.org/wiki/Alan_Turing",
    title: str = "Alan Turing biography",
    snippet: str = "Alan Turing was a British mathematician.",
    raw_content: str | None = None,
) -> SearchResult:
    return SearchResult(
        title=title,
        url=url,
        snippet=snippet,
        provider=provider,
        provider_rank=rank,
        raw_content=raw_content,
    )


def _response(
    provider: str,
    results: list[SearchResult],
    *,
    status: str = "success",
    failure_category: str | None = None,
) -> ProviderSearchResponse:
    return ProviderSearchResponse(
        provider=provider,
        query="",
        status=status,
        results=results,
        failure_category=failure_category,
    )


def _public_validator(url: str) -> object:
    assert url.startswith("https://")
    return object()


class _ProviderResponse:
    is_redirect = False
    status_code = 200
    headers = {"content-type": "application/json"}
    request = httpx.Request("GET", "https://provider.example/api")

    def __enter__(self) -> _ProviderResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self) -> object:
        yield b"{}"


class _ProviderClient:
    options: dict[str, Any] = {}
    request_url = ""

    def __init__(self, *args: object, **kwargs: Any) -> None:
        del args
        type(self).options = kwargs

    def __enter__(self) -> _ProviderClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def stream(self, method: str, url: str, **kwargs: Any) -> _ProviderResponse:
        del method, kwargs
        type(self).request_url = url
        return _ProviderResponse()


def test_tavily_success_preserves_rank_and_raw_content_without_exposing_key() -> None:
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.tavily.com/search"),
        json={
            "results": [
                {
                    "title": "Python documentation",
                    "url": "https://docs.python.org/3/",
                    "content": "Official Python documentation",
                    "raw_content": "<html><body>Python reference</body></html>",
                    "score": 0.94,
                }
            ]
        },
    )
    provider = TavilySearchProvider("test-secret")

    with patch("retrieval_providers._request", return_value=response):
        outcome = provider.search("Python official documentation", 5)

    assert outcome.status == "success"
    assert outcome.http_status == 200
    assert outcome.results[0].provider_rank == 1
    assert outcome.results[0].raw_content == (
        "<html><body>Python reference</body></html>"
    )
    assert "test-secret" not in json.dumps(outcome.public_dict())


def test_fixed_provider_proxy_uses_logical_host_after_dual_public_validation() -> None:
    validated = ValidatedURL(
        url="https://provider.example/api",
        hostname="provider.example",
        port=443,
        proxy_url="http://127.0.0.1:17898",
        addresses=("93.184.216.34",),
        selected_address="93.184.216.34",
        resolution_mode="proxy_doh",
    )

    with (
        patch("retrieval_providers.validate_public_url", return_value=validated),
        patch("retrieval_providers.revalidate_public_url") as revalidate,
        patch("retrieval_providers.build_pinned_request") as pinned,
        patch("retrieval_providers.httpx.Client", _ProviderClient),
    ):
        response = retrieval_providers._request(  # noqa: SLF001
            "GET",
            validated.url,
        )

    assert response.json() == {}
    assert _ProviderClient.request_url == validated.url
    assert _ProviderClient.options["proxy"] == validated.proxy_url
    assert _ProviderClient.options["trust_env"] is False
    pinned.assert_not_called()
    revalidate.assert_called_once_with(validated)


def test_default_provider_order_and_unconfigured_tavily_are_stable() -> None:
    providers = default_search_providers()

    assert [provider.name for provider in providers] == [
        "tavily",
        "mediawiki",
        "duckduckgo_html",
        "bing_rss",
    ]
    outcome = TavilySearchProvider("").search("Python documentation", 5)
    assert outcome.status == "not_configured"
    assert outcome.failure_category == "not_configured"


def test_mediawiki_search_normalizes_provider_rank_and_canonical_url() -> None:
    response = httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            "https://en.wikipedia.org/w/api.php",
        ),
        json={
            "query": {
                "search": [
                    {
                        "pageid": 42,
                        "title": "Alan Turing",
                        "snippet": "<span>British</span> mathematician",
                    }
                ]
            }
        },
    )

    with patch("retrieval_providers._request", return_value=response):
        outcome = MediaWikiSearchProvider().search("Alan Turing biography", 5)

    assert outcome.status == "success"
    assert outcome.results[0].provider_rank == 1
    assert outcome.results[0].url == ("https://en.wikipedia.org/wiki/Alan_Turing")
    assert outcome.results[0].snippet == "British mathematician"


def test_mediawiki_search_skips_explicit_documentation_intent() -> None:
    with patch("retrieval_providers._request") as request:
        outcome = MediaWikiSearchProvider().search(
            "Python official documentation",
            5,
        )

    assert outcome.status == "skipped"
    assert outcome.failure_category == "not_applicable"
    assert outcome.metadata["reason"] == "non_encyclopedic_query"
    request.assert_not_called()


def test_mediawiki_content_adapter_returns_plaintext_acquisition() -> None:
    response = httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            "https://en.wikipedia.org/w/api.php",
        ),
        headers={"content-type": "application/json"},
        json={
            "query": {
                "pages": {
                    "42": {
                        "pageid": 42,
                        "title": "Alan Turing",
                        "extract": "Alan Turing was a British mathematician.",
                    }
                }
            }
        },
    )

    with patch("retrieval_providers._request", return_value=response):
        result = fetch_mediawiki_content(
            "https://en.wikipedia.org/wiki/Alan_Turing",
            12_000,
        )

    assert result["status"] == "success"
    assert result["acquisition_method"] == "mediawiki_api"
    assert result["final_url"] == ("https://en.wikipedia.org/wiki/Alan_Turing")
    assert "British mathematician" in result["content"]


def test_mediawiki_action_403_falls_back_to_official_core_api() -> None:
    core_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            ("https://api.wikimedia.org/core/v1/wikipedia/en/page/Alan_Turing/html"),
        ),
        headers={"content-type": "text/html; charset=utf-8"},
        text=(
            "<html><head><title>Alan Turing</title></head>"
            "<body><p>Alan Turing was a British mathematician.</p></body></html>"
        ),
    )

    with patch(
        "retrieval_providers._request",
        side_effect=[
            ProviderRequestError("access_blocked", http_status=403),
            core_response,
        ],
    ):
        result = fetch_mediawiki_content(
            "https://en.wikipedia.org/wiki/Alan_Turing",
            12_000,
        )

    assert result["status"] == "success"
    assert result["acquisition_method"] == "mediawiki_api"
    assert result["mediawiki_api_variant"] == "wikimedia_core"
    assert result["mediawiki_fallback_from"]["http_status"] == 403


def test_tavily_failure_falls_back_without_interrupting_search() -> None:
    tavily = StubProvider(
        "tavily",
        _response(
            "tavily",
            [],
            status="error",
            failure_category="provider_error",
        ),
    )
    mediawiki = StubProvider(
        "mediawiki",
        _response(
            "mediawiki",
            [_result(provider="mediawiki", rank=1)],
        ),
    )
    broker = SearchBroker([tavily, mediawiki])

    execution = broker.search("Alan Turing biography", 5)

    assert execution.status == "success"
    assert execution.results[0].provider == "mediawiki"
    assert execution.fallback_reasons == ["tavily_provider_error"]
    assert tavily.calls == mediawiki.calls == 1


def test_multi_provider_dedup_preserves_ranks_and_rrf_is_deterministic() -> None:
    first = StubProvider(
        "first",
        _response("first", [_result(provider="first", rank=2)]),
    )
    second = StubProvider(
        "second",
        _response(
            "second",
            [
                _result(provider="second", rank=1),
                _result(
                    provider="second",
                    rank=2,
                    url="https://www.turing.org.uk/scrapbook/bio.html",
                    title="Alan Turing biography archive",
                ),
            ],
        ),
    )
    broker = SearchBroker([first, second])

    initial = broker.search("Alan Turing biography", 5)
    repeated = broker.search("Alan Turing biography", 5)

    assert [item.url for item in initial.results] == [
        item.url for item in repeated.results
    ]
    wikipedia = next(
        item
        for item in initial.results
        if item.url == "https://en.wikipedia.org/wiki/Alan_Turing"
    )
    assert wikipedia.providers == ["first", "second"]
    assert wikipedia.provider_ranks == {"first": 2, "second": 1}
    assert wikipedia.rrf_score > 1 / 61


def test_generic_words_are_irrelevant_but_entity_match_is_relevant() -> None:
    generic = assess_search_relevance(
        "new red school",
        {
            "title": "New red school building",
            "url": "https://example.com/",
            "snippet": "School news",
        },
    )
    entity = assess_search_relevance(
        "Nelson Mandela imprisonment",
        {
            "title": "Nelson Mandela imprisonment",
            "url": "https://example.org/mandela",
            "snippet": "Mandela prison history",
        },
    )

    assert generic["tier"] == "irrelevant"
    assert entity["tier"] == "relevant"


def test_uncertain_candidate_is_retained_when_no_relevant_result_exists() -> None:
    provider = StubProvider(
        "fallback",
        _response(
            "fallback",
            [
                _result(
                    provider="fallback",
                    rank=1,
                    url="https://example.org/mandela-record",
                    title="Mandela prison record",
                    snippet="Archival entry",
                )
            ],
        ),
    )

    execution = SearchBroker([provider]).search(
        "Nelson Mandela imprisonment Robben Island 1964",
        5,
    )

    assert execution.search_quality == "uncertain_only"
    assert execution.relevant_results == 0
    assert execution.uncertain_results == 1
    assert execution.results[0].relevance_tier == "uncertain"


def test_all_provider_failures_return_auditable_empty_result() -> None:
    providers = [
        StubProvider(
            name,
            _response(
                name,
                [],
                status="error",
                failure_category="provider_error",
            ),
        )
        for name in ("tavily", "mediawiki")
    ]

    execution = SearchBroker(providers).search("Alan Turing biography", 5)
    payload = execution.public_dict()

    assert payload["status"] == "error"
    assert payload["error"] == "all_search_providers_failed"
    assert payload["results"] == []
    assert [item["status"] for item in payload["provider_statuses"]] == [
        "error",
        "error",
    ]


def test_provider_raw_content_cache_is_only_acquired_through_fetch() -> None:
    content = "Alan Turing evidence. " * 40
    provider = StubProvider(
        "tavily",
        _response(
            "tavily",
            [_result(provider="tavily", rank=1, raw_content=content)],
        ),
    )
    session = RetrievalSession(
        SearchBroker([provider]),
        url_validator=_public_validator,
    )
    search_tool, fetch_tool = create_retrieval_tools(session)
    policy = EffortPolicy(
        name="low",
        max_searches=1,
        max_fetches=1,
        min_successful_sources=0,
        max_results_per_search=5,
        max_chars_per_page=2_000,
        max_output_tokens=64,
        max_subquestions=1,
        require_reviewer=False,
    )
    tools, budget = build_budgeted_tools(
        policy,
        raw_search_tool=search_tool,
        raw_fetch_tool=fetch_tool,
    )
    search = next(item for item in tools if item.name == "web_search")
    fetch = next(item for item in tools if item.name == "fetch_url")

    searched = json.loads(search.invoke({"query": "Alan Turing biography"}))
    assert searched["status"] == "success"
    assert budget.snapshot()["successful_sources"] == []

    fetched = json.loads(
        fetch.invoke({"url": "https://en.wikipedia.org/wiki/Alan_Turing"})
    )
    snapshot = budget.snapshot()
    assert fetched["status"] == "success"
    assert fetched["acquisition_method"] == "provider_raw_content"
    assert fetched["source_id"] == "S1"
    assert snapshot["fetch_calls"] == 1
    source = snapshot["successful_sources"][0]
    assert source["acquisition_method"] == "provider_raw_content"
    assert source["content_provider"] == "tavily"


def test_cache_hit_validates_url_and_never_calls_direct_fetch() -> None:
    provider = StubProvider(
        "tavily",
        _response(
            "tavily",
            [
                _result(
                    provider="tavily",
                    rank=1,
                    raw_content="<html><body>trusted acquisition text</body></html>",
                )
            ],
        ),
    )
    session = RetrievalSession(
        SearchBroker([provider]),
        url_validator=_public_validator,
    )
    session.search("Alan Turing biography", 5)
    direct_calls = 0

    def direct_fetch(url: str, max_chars: int) -> dict[str, Any]:
        nonlocal direct_calls
        del url, max_chars
        direct_calls += 1
        return {"status": "error", "failure_taxonomy": "provider_error"}

    result = session.fetch(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        12_000,
        direct_fetch=direct_fetch,
    )

    assert result["status"] == "success"
    assert result["acquisition_method"] == "provider_raw_content"
    assert result["untrusted_content"] is True
    assert direct_calls == 0


def test_direct_fetch_success_keeps_acquisition_metadata() -> None:
    session = RetrievalSession(url_validator=_public_validator)

    result = session.fetch(
        "https://example.com/",
        12_000,
        direct_fetch=lambda url, max_chars: {
            "status": "success",
            "url": url,
            "content": "visible text",
            "content_chars": 12,
            "max_chars": max_chars,
        },
    )

    assert result["status"] == "success"
    assert result["acquisition_method"] == "direct_http"
    assert result["content_provider"] == "direct_http"
    assert result["original_url"] == "https://example.com/"


def test_wikipedia_access_block_uses_mediawiki_fallback() -> None:
    mediawiki_calls = 0

    def mediawiki(url: str, max_chars: int) -> dict[str, Any]:
        nonlocal mediawiki_calls
        mediawiki_calls += 1
        return {
            "status": "success",
            "url": url,
            "content": "MediaWiki extract",
            "content_chars": 17,
            "max_chars": max_chars,
            "acquisition_method": "mediawiki_api",
            "content_provider": "mediawiki",
        }

    session = RetrievalSession(mediawiki_fetcher=mediawiki)
    result = session.fetch(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        12_000,
        direct_fetch=lambda url, max_chars: {
            "status": "error",
            "url": url,
            "http_status": 403,
            "failure_taxonomy": "access_blocked",
        },
    )

    assert result["status"] == "success"
    assert result["acquisition_method"] == "mediawiki_api"
    assert result["fallback_triggered"] is True
    assert result["fallback_from"]["http_status"] == 403
    assert mediawiki_calls == 1


def test_mediawiki_fallback_failure_activates_host_cooldown() -> None:
    direct_calls = 0

    def blocked(url: str, max_chars: int) -> dict[str, Any]:
        nonlocal direct_calls
        del max_chars
        direct_calls += 1
        return {
            "status": "error",
            "url": url,
            "http_status": 429,
            "failure_taxonomy": "rate_limited",
        }

    session = RetrievalSession(
        mediawiki_fetcher=lambda url, max_chars: {
            "status": "error",
            "url": url,
            "max_chars": max_chars,
            "failure_taxonomy": "provider_error",
        }
    )
    first = session.fetch(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        12_000,
        direct_fetch=blocked,
    )
    second = session.fetch(
        "https://en.wikipedia.org/wiki/Ada_Lovelace",
        12_000,
        direct_fetch=blocked,
    )

    assert first["fallback_triggered"] is True
    assert first["fallback_failure"]["failure_taxonomy"] == "provider_error"
    assert second["status"] == "suppressed"
    assert second["switch_source"] is True
    assert direct_calls == 1


def test_non_wikipedia_access_block_never_calls_mediawiki() -> None:
    mediawiki_calls = 0

    def mediawiki(url: str, max_chars: int) -> dict[str, Any]:
        nonlocal mediawiki_calls
        del url, max_chars
        mediawiki_calls += 1
        return {"status": "error"}

    session = RetrievalSession(mediawiki_fetcher=mediawiki)
    result = session.fetch(
        "https://example.com/private",
        12_000,
        direct_fetch=lambda url, max_chars: {
            "status": "error",
            "url": url,
            "max_chars": max_chars,
            "failure_taxonomy": "access_blocked",
        },
    )

    assert result["failure_taxonomy"] == "access_blocked"
    assert result["fallback_triggered"] is False
    assert mediawiki_calls == 0


def test_private_ipv4_and_ipv6_are_rejected_before_direct_fetch() -> None:
    direct_calls = 0

    def direct_fetch(url: str, max_chars: int) -> dict[str, Any]:
        nonlocal direct_calls
        del url, max_chars
        direct_calls += 1
        return {"status": "success"}

    session = RetrievalSession()
    for url in ("http://10.0.0.8/private", "http://[fd00::8]/private"):
        result = session.fetch(url, 12_000, direct_fetch=direct_fetch)
        assert result["status"] == "rejected"
        assert result["failure_taxonomy"] == "unsafe_url"
    assert direct_calls == 0


def test_all_live_systems_receive_fresh_tools_from_same_backend() -> None:
    first = _live_raw_tools()
    second = _live_raw_tools()

    assert [item.name for item in first] == ["web_search", "fetch_url"]
    assert {item.metadata["retrieval_backend"] for item in [*first, *second]} == {
        "unified"
    }
    assert (
        first[0].metadata["retrieval_session"] is first[1].metadata["retrieval_session"]
    )
    assert (
        first[0].metadata["retrieval_session"]
        is not second[0].metadata["retrieval_session"]
    )
