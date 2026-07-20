"""Search-provider adapters with one auditable response contract."""

from __future__ import annotations

import html
import os
import re
import socket
import ssl
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx

from retrieval_safety import (
    URLValidationError,
    build_pinned_request,
    revalidate_public_url,
    validate_public_url,
)


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
BING_SEARCH_URL = "https://www.bing.com/search"
DEFAULT_USER_AGENT = (
    "TongAgentResearch/0.2 "
    "(https://github.com/cantwakeup/TongAgent4pre; retrieval backend)"
)
MAX_PROVIDER_RESPONSE_BYTES = 5_000_000
MAX_MEDIAWIKI_RESPONSE_BYTES = 2_000_000
_NON_ENCYCLOPEDIC_QUERY = re.compile(
    r"\b(?:api\s+reference|documentation|docs|manual|sdk)\b",
    flags=re.IGNORECASE,
)


@dataclass
class SearchResult:
    """One normalized provider result before cross-provider fusion."""

    title: str
    url: str
    snippet: str
    provider: str
    provider_rank: int
    raw_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        """Return the provider observation without copying raw page text."""

        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
            "provider_rank": self.provider_rank,
            "raw_content_available": bool(
                self.raw_content and self.raw_content.strip()
            ),
            "raw_content_chars": (
                len(self.raw_content) if isinstance(self.raw_content, str) else 0
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class ProviderSearchResponse:
    """All observable state from one provider invocation."""

    provider: str
    query: str
    status: str
    results: list[SearchResult] = field(default_factory=list)
    http_status: int | None = None
    failure_category: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        """Return a secret-free machine-readable provider record."""

        return {
            "provider": self.provider,
            "query": self.query,
            "status": self.status,
            "http_status": self.http_status,
            "failure_category": self.failure_category,
            "error_type": self.error_type,
            "result_count": len(self.results),
            "raw_results": [result.public_dict() for result in self.results],
            "metadata": dict(self.metadata),
        }


class SearchProvider(Protocol):
    """Protocol implemented by every retrieval search adapter."""

    name: str

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        """Search for `query` and preserve the provider's original ranking."""


class ProviderRequestError(RuntimeError):
    """Secret-free provider request failure."""

    def __init__(
        self,
        category: str,
        *,
        http_status: int | None = None,
        error_type: str | None = None,
    ) -> None:
        self.category = category
        self.http_status = http_status
        self.error_type = error_type or category
        super().__init__(category)


class _DuckDuckGoResultParser(HTMLParser):
    """Parse DuckDuckGo's HTML-only result list."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None

    @staticmethod
    def _result_url(href: str) -> str:
        absolute = "https:" + href if href.startswith("//") else href
        parsed = urlparse(absolute)
        hostname = (parsed.hostname or "").casefold()
        if hostname == "duckduckgo.com" or hostname.endswith(".duckduckgo.com"):
            redirected = parse_qs(parsed.query).get("uddg")
            if redirected:
                return unquote(redirected[0])
        return absolute

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if "result__a" in classes:
            if self._current and self._current.get("title"):
                self.results.append(self._current)
            self._current = {
                "title": "",
                "url": self._result_url(values.get("href") or ""),
                "snippet": "",
            }
            self._capture = "title"
        elif "result__snippet" in classes and self._current is not None:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._current is None or self._capture is None:
            return
        text = " ".join(data.split())
        if text:
            existing = self._current[self._capture]
            self._current[self._capture] = f"{existing} {text}".strip()

    def close(self) -> None:
        super().close()
        if self._current and self._current.get("title"):
            self.results.append(self._current)
            self._current = None


class _VisiblePageParser(HTMLParser):
    """Extract inert visible text and a title from provider HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        self.parts.append(text)
        if self._in_title:
            self.title_parts.append(text)


def configured_user_agent() -> str:
    """Return a configurable, non-secret user agent."""

    return os.environ.get("SEARCH_AGENT_USER_AGENT", "").strip() or DEFAULT_USER_AGENT


def _request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    max_bytes: int = MAX_PROVIDER_RESPONSE_BYTES,
) -> httpx.Response:
    """Send one DNS-pinned public request without following redirects."""

    try:
        validated = validate_public_url(url)
        request_headers = {
            "User-Agent": configured_user_agent(),
            "Accept-Encoding": "identity",
            **(headers or {}),
        }
        if validated.proxy_url is None:
            pinned = build_pinned_request(validated)
            request_url = pinned.url
            request_headers = {**pinned.headers, **request_headers}
            client_options: dict[str, Any] = {
                "transport": pinned.transport,
                "trust_env": False,
            }
            request_extensions = pinned.extensions
        else:
            # Provider endpoints are constructed by these adapters rather than
            # accepted from Agent input. Some HTTP proxies reject CONNECT by IP,
            # so keep the logical hostname only after dual public DoH checks.
            # Arbitrary page fetches continue to use strict IP pinning.
            request_url = validated.url
            client_options = {
                "proxy": validated.proxy_url,
                "trust_env": False,
            }
            request_extensions = {}
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            **client_options,
        ) as client:
            with client.stream(
                method,
                request_url,
                params=params,
                data=data,
                json=json_body,
                headers=request_headers,
                extensions=request_extensions,
            ) as response:
                if response.is_redirect:
                    raise ProviderRequestError(
                        "provider_redirect",
                        http_status=response.status_code,
                    )
                response.raise_for_status()
                chunks: list[bytes] = []
                downloaded = 0
                for chunk in response.iter_bytes():
                    remaining = max_bytes - downloaded
                    if remaining <= 0 or len(chunk) > remaining:
                        raise ProviderRequestError(
                            "provider_error",
                            http_status=response.status_code,
                            error_type="ResponseTooLarge",
                        )
                    chunks.append(chunk)
                    downloaded += len(chunk)
                buffered = httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=b"".join(chunks),
                    request=response.request,
                )
            revalidate_public_url(validated)
            return buffered
    except ProviderRequestError:
        raise
    except URLValidationError as exc:
        category = {
            "dns_rejected": "dns_error",
            "dns_rebinding": "unsafe_url",
            "redirect_rejected": "unsafe_url",
            "ssrf_rejected": "unsafe_url",
        }.get(exc.taxonomy, exc.taxonomy)
        raise ProviderRequestError(
            category,
            error_type=type(exc).__name__,
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        category = (
            "rate_limited"
            if status == 429
            else (
                "access_blocked" if status in {401, 403, 407, 451} else "provider_error"
            )
        )
        raise ProviderRequestError(
            category,
            http_status=status,
            error_type=type(exc).__name__,
        ) from exc
    except httpx.TimeoutException as exc:
        raise ProviderRequestError(
            "timeout",
            error_type=type(exc).__name__,
        ) from exc
    except httpx.RequestError as exc:
        if any(isinstance(item, socket.gaierror) for item in _exception_chain(exc)):
            category = "dns_error"
        else:
            category = "tls_error" if _contains_tls_error(exc) else "provider_error"
        raise ProviderRequestError(
            category,
            error_type=type(exc).__name__,
        ) from exc


def _contains_tls_error(error: BaseException) -> bool:
    """Return whether a bounded exception chain contains a TLS failure."""

    current: BaseException | None = error
    for _ in range(8):
        if current is None:
            return False
        if (
            isinstance(current, ssl.SSLError)
            or "ssl" in type(current).__name__.casefold()
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Return one bounded provider exception chain."""

    chain = [error]
    while len(chain) < 8:
        next_error = chain[-1].__cause__ or chain[-1].__context__
        if next_error is None or next_error in chain:
            break
        chain.append(next_error)
    return chain


def _json_payload(response: httpx.Response) -> dict[str, Any]:
    """Parse a provider response without exposing its body on failure."""

    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderRequestError(
            "parse_error",
            http_status=response.status_code,
            error_type=type(exc).__name__,
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderRequestError(
            "parse_error",
            http_status=response.status_code,
            error_type="NonObjectJSON",
        )
    return payload


def _failure_response(
    provider: str,
    query: str,
    error: Exception,
) -> ProviderSearchResponse:
    """Convert any provider error to an auditable, secret-free record."""

    if isinstance(error, ProviderRequestError):
        return ProviderSearchResponse(
            provider=provider,
            query=query,
            status="error",
            http_status=error.http_status,
            failure_category=error.category,
            error_type=error.error_type,
        )
    return ProviderSearchResponse(
        provider=provider,
        query=query,
        status="error",
        failure_category="provider_error",
        error_type=type(error).__name__,
    )


class TavilySearchProvider:
    """Official Tavily Search API adapter with optional raw page content."""

    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get("TAVILY_API_KEY")
        )

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        """Search Tavily and retain raw content in memory only."""

        if not self._api_key:
            return ProviderSearchResponse(
                provider=self.name,
                query=query,
                status="not_configured",
                failure_category="not_configured",
            )
        try:
            response = _request(
                "POST",
                TAVILY_SEARCH_URL,
                json_body={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_raw_content": True,
                    "search_depth": "advanced",
                },
            )
            payload = _json_payload(response)
            raw_results = payload.get("results")
            if not isinstance(raw_results, list):
                raise ProviderRequestError(
                    "parse_error",
                    http_status=response.status_code,
                    error_type="MissingResults",
                )
            results: list[SearchResult] = []
            for rank, raw in enumerate(raw_results[:max_results], 1):
                if not isinstance(raw, dict):
                    continue
                url = str(raw.get("url", "")).strip()
                if not url:
                    continue
                score = raw.get("score")
                results.append(
                    SearchResult(
                        title=str(raw.get("title", "")).strip(),
                        url=url,
                        snippet=str(raw.get("content", "")).strip(),
                        provider=self.name,
                        provider_rank=rank,
                        raw_content=(
                            raw.get("raw_content")
                            if isinstance(raw.get("raw_content"), str)
                            else None
                        ),
                        metadata={
                            **(
                                {"provider_score": score}
                                if isinstance(score, (int, float))
                                and not isinstance(score, bool)
                                else {}
                            ),
                        },
                    )
                )
            return ProviderSearchResponse(
                provider=self.name,
                query=query,
                status="success" if results else "empty",
                results=results,
                http_status=response.status_code,
                metadata={"include_raw_content": True},
            )
        except Exception as exc:
            return _failure_response(self.name, query, exc)


class MediaWikiSearchProvider:
    """MediaWiki Search API adapter for encyclopedic entity discovery."""

    name = "mediawiki"

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        """Search a language-appropriate Wikipedia project."""

        if _NON_ENCYCLOPEDIC_QUERY.search(query):
            return ProviderSearchResponse(
                provider=self.name,
                query=query,
                status="skipped",
                failure_category="not_applicable",
                metadata={"reason": "non_encyclopedic_query"},
            )
        language = "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"
        endpoint = f"https://{language}.wikipedia.org/w/api.php"
        try:
            response = _request(
                "GET",
                endpoint,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": max_results,
                    "utf8": 1,
                    "format": "json",
                },
                headers={"Api-User-Agent": configured_user_agent()},
            )
            payload = _json_payload(response)
            query_payload = payload.get("query")
            raw_results = (
                query_payload.get("search") if isinstance(query_payload, dict) else None
            )
            if not isinstance(raw_results, list):
                raise ProviderRequestError(
                    "parse_error",
                    http_status=response.status_code,
                    error_type="MissingSearchResults",
                )
            results: list[SearchResult] = []
            for rank, raw in enumerate(raw_results[:max_results], 1):
                if not isinstance(raw, dict):
                    continue
                title = str(raw.get("title", "")).strip()
                if not title:
                    continue
                page_title = quote(title.replace(" ", "_"), safe="():,")
                results.append(
                    SearchResult(
                        title=title,
                        url=(f"https://{language}.wikipedia.org/wiki/{page_title}"),
                        snippet=_plain_snippet(str(raw.get("snippet", ""))),
                        provider=self.name,
                        provider_rank=rank,
                        metadata={
                            "language": language,
                            "project": "wikipedia",
                            **(
                                {"page_id": int(raw["pageid"])}
                                if isinstance(raw.get("pageid"), int)
                                else {}
                            ),
                        },
                    )
                )
            return ProviderSearchResponse(
                provider=self.name,
                query=query,
                status="success" if results else "empty",
                results=results,
                http_status=response.status_code,
                metadata={"language": language, "project": "wikipedia"},
            )
        except Exception as exc:
            return _failure_response(self.name, query, exc)


class DuckDuckGoHTMLProvider:
    """DuckDuckGo HTML adapter retained as a no-key fallback."""

    name = "duckduckgo_html"

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        """Fetch and parse DuckDuckGo's HTML results."""

        try:
            response = _request(
                "POST",
                DUCKDUCKGO_SEARCH_URL,
                data={"q": query, "kl": "wt-wt"},
            )
            parser = _DuckDuckGoResultParser()
            parser.feed(response.text)
            parser.close()
            results = [
                SearchResult(
                    title=item["title"],
                    url=item["url"],
                    snippet=item["snippet"],
                    provider=self.name,
                    provider_rank=rank,
                )
                for rank, item in enumerate(parser.results[:max_results], 1)
                if item.get("url")
            ]
            return ProviderSearchResponse(
                provider=self.name,
                query=query,
                status="success" if results else "empty",
                results=results,
                http_status=response.status_code,
            )
        except Exception as exc:
            return _failure_response(self.name, query, exc)


class BingRSSProvider:
    """Bing RSS adapter retained only as the last no-key fallback."""

    name = "bing_rss"

    def search(self, query: str, max_results: int) -> ProviderSearchResponse:
        """Fetch and parse Bing's RSS response."""

        try:
            response = _request(
                "GET",
                BING_SEARCH_URL,
                params={
                    "format": "rss",
                    "q": query,
                    "cc": "us",
                    "mkt": "en-US",
                    "setlang": "en-US",
                },
            )
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as exc:
                raise ProviderRequestError(
                    "parse_error",
                    http_status=response.status_code,
                    error_type=type(exc).__name__,
                ) from exc
            results = [
                SearchResult(
                    title=item.findtext("title", default="").strip(),
                    url=item.findtext("link", default="").strip(),
                    snippet=item.findtext("description", default="").strip(),
                    provider=self.name,
                    provider_rank=rank,
                )
                for rank, item in enumerate(
                    root.findall("./channel/item")[:max_results],
                    1,
                )
                if item.findtext("link", default="").strip()
            ]
            return ProviderSearchResponse(
                provider=self.name,
                query=query,
                status="success" if results else "empty",
                results=results,
                http_status=response.status_code,
            )
        except Exception as exc:
            return _failure_response(self.name, query, exc)


def default_search_providers() -> list[SearchProvider]:
    """Return the configured provider chain in reliability order."""

    return [
        TavilySearchProvider(),
        MediaWikiSearchProvider(),
        DuckDuckGoHTMLProvider(),
        BingRSSProvider(),
    ]


def _plain_snippet(value: str) -> str:
    """Convert MediaWiki's small HTML snippet to visible text."""

    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(without_tags).split())


def parse_wikipedia_url(url: str) -> dict[str, str] | None:
    """Extract language, project, and title from a canonical Wikipedia URL."""

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    labels = hostname.split(".")
    if (
        parsed.scheme not in {"http", "https"}
        or len(labels) != 3
        or labels[1:] != ["wikipedia", "org"]
        or not parsed.path.startswith("/wiki/")
    ):
        return None
    encoded_title = parsed.path.removeprefix("/wiki/")
    if not encoded_title:
        return None
    return {
        "language": labels[0],
        "project": "wikipedia",
        "title": unquote(encoded_title).replace("_", " "),
    }


def fetch_mediawiki_content(url: str, max_chars: int) -> dict[str, Any]:
    """Fetch plain page text through MediaWiki after an HTML access failure."""

    parsed = parse_wikipedia_url(url)
    if parsed is None:
        return {
            "status": "error",
            "url": url,
            "error": "Not a supported Wikipedia page URL",
            "failure_taxonomy": "provider_error",
            "failure_type": "provider_error",
            "retryable": False,
        }
    endpoint = f"https://{parsed['language']}.wikipedia.org/w/api.php"
    try:
        response = _request(
            "GET",
            endpoint,
            params={
                "action": "query",
                "prop": "extracts|info",
                "explaintext": 1,
                "redirects": 1,
                "inprop": "url",
                "titles": parsed["title"],
                "format": "json",
                "utf8": 1,
            },
            headers={"Api-User-Agent": configured_user_agent()},
            max_bytes=MAX_MEDIAWIKI_RESPONSE_BYTES,
        )
        payload = _json_payload(response)
        query_payload = payload.get("query")
        pages = query_payload.get("pages") if isinstance(query_payload, dict) else None
        if not isinstance(pages, dict):
            raise ProviderRequestError(
                "parse_error",
                http_status=response.status_code,
                error_type="MissingPages",
            )
        page = next(
            (
                item
                for item in pages.values()
                if isinstance(item, dict)
                and isinstance(item.get("extract"), str)
                and item["extract"].strip()
            ),
            None,
        )
        if page is None:
            missing_extract = {
                "status": "error",
                "url": url,
                "error": "MediaWiki API returned no readable page extract",
                "failure_taxonomy": "insufficient_content",
                "failure_type": "insufficient_content",
                "retryable": False,
            }
            return _mediawiki_core_or_failure(
                url,
                parsed,
                max_chars,
                missing_extract,
            )
        normalized = "\n".join(
            line.strip() for line in str(page["extract"]).splitlines() if line.strip()
        )
        observed = len(normalized)
        content = normalized[:max_chars]
        return {
            "status": "success",
            "url": url,
            "original_url": url,
            "final_url": url,
            "title": str(page.get("title", parsed["title"])),
            "content": content,
            "content_chars": len(content),
            "content_length": observed,
            "observed_content_length": observed,
            "content_length_scope": "mediawiki_plaintext_extract",
            "downloaded_bytes": len(response.content),
            "downloaded_bytes_scope": "mediawiki_api_response_bytes",
            "http_content_length": len(response.content),
            "http_content_encoding": None,
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "fetched_at": datetime.now(UTC).isoformat(),
            "truncated": observed > len(content),
            "truncation_reasons": (
                ["returned_character_limit"] if observed > len(content) else []
            ),
            "acquisition_method": "mediawiki_api",
            "content_provider": "mediawiki",
            "fallback_triggered": True,
            "mediawiki": parsed,
            "mediawiki_api_variant": "action_api",
        }
    except Exception as exc:
        failure = _failure_response("mediawiki", parsed["title"], exc)
        action_failure = {
            "status": "error",
            "url": url,
            "error": failure.failure_category or "provider_error",
            "failure_taxonomy": failure.failure_category or "provider_error",
            "failure_type": failure.failure_category or "provider_error",
            "http_status": failure.http_status,
            "retryable": failure.failure_category in {"timeout", "rate_limited"},
            "acquisition_method": "mediawiki_api",
            "content_provider": "mediawiki",
            "fallback_triggered": True,
        }
        return _mediawiki_core_or_failure(
            url,
            parsed,
            max_chars,
            action_failure,
        )


def _mediawiki_core_or_failure(
    url: str,
    parsed: dict[str, str],
    max_chars: int,
    action_failure: dict[str, Any],
) -> dict[str, Any]:
    """Try Wikimedia's official Core API after Action API acquisition fails."""

    language = quote(parsed["language"], safe="")
    title = quote(parsed["title"].replace(" ", "_"), safe="")
    endpoint = (
        f"https://api.wikimedia.org/core/v1/wikipedia/{language}/page/{title}/html"
    )
    try:
        response = _request(
            "GET",
            endpoint,
            headers={
                "Accept": "text/html",
                "Api-User-Agent": configured_user_agent(),
            },
            max_bytes=MAX_MEDIAWIKI_RESPONSE_BYTES,
        )
        parser = _VisiblePageParser()
        try:
            parser.feed(response.text)
            parser.close()
        except (TypeError, ValueError) as exc:
            raise ProviderRequestError(
                "parse_error",
                http_status=response.status_code,
                error_type=type(exc).__name__,
            ) from exc
        normalized = "\n".join(parser.parts)
        if not normalized.strip():
            raise ProviderRequestError(
                "insufficient_content",
                http_status=response.status_code,
                error_type="EmptyCoreHTML",
            )
        observed = len(normalized)
        content = normalized[:max_chars]
        return {
            "status": "success",
            "url": url,
            "original_url": url,
            "final_url": url,
            "title": " ".join(parser.title_parts).strip() or parsed["title"],
            "content": content,
            "content_chars": len(content),
            "content_length": observed,
            "observed_content_length": observed,
            "content_length_scope": "wikimedia_core_normalized_visible_text",
            "downloaded_bytes": len(response.content),
            "downloaded_bytes_scope": "wikimedia_core_response_bytes",
            "http_content_length": len(response.content),
            "http_content_encoding": None,
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "fetched_at": datetime.now(UTC).isoformat(),
            "truncated": observed > len(content),
            "truncation_reasons": (
                ["returned_character_limit"] if observed > len(content) else []
            ),
            "acquisition_method": "mediawiki_api",
            "content_provider": "mediawiki",
            "fallback_triggered": True,
            "mediawiki": parsed,
            "mediawiki_api_variant": "wikimedia_core",
            "mediawiki_fallback_from": {
                "failure_taxonomy": action_failure.get(
                    "failure_taxonomy",
                    action_failure.get("failure_type"),
                ),
                "http_status": action_failure.get("http_status"),
                "variant": "action_api",
            },
        }
    except Exception as exc:
        failure = _failure_response("wikimedia_core", parsed["title"], exc)
        action_failure["mediawiki_api_variant"] = "action_api"
        action_failure["wikimedia_core_failure"] = {
            "failure_taxonomy": failure.failure_category or "provider_error",
            "http_status": failure.http_status,
            "error_type": failure.error_type,
        }
        return action_failure


__all__ = [
    "BingRSSProvider",
    "DuckDuckGoHTMLProvider",
    "MediaWikiSearchProvider",
    "ProviderSearchResponse",
    "SearchProvider",
    "SearchResult",
    "TavilySearchProvider",
    "configured_user_agent",
    "default_search_providers",
    "fetch_mediawiki_content",
    "parse_wikipedia_url",
]
