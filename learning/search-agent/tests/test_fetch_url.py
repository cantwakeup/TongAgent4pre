"""Regression tests for recoverable page-fetch failures."""

from __future__ import annotations

import json
import ssl
import unittest
from unittest.mock import Mock, patch

import httpx

from agent_policy import EFFORT_POLICIES
import retrieval_safety
from retrieval_safety import (
    URLValidationError,
    ValidatedURL,
    build_pinned_request,
    validate_public_url,
)
from search_agent import ResearchBudget, _fetch_public_url, fetch_url


class _ForbiddenResponse:
    """Act like a streamed HTTP response that returns status 403."""

    is_redirect = False
    headers = {"content-type": "text/html"}

    def __enter__(self) -> _ForbiddenResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def raise_for_status(self) -> None:
        request = httpx.Request("GET", "https://example.com/private")
        response = httpx.Response(403, request=request)
        msg = "Forbidden"
        raise httpx.HTTPStatusError(msg, request=request, response=response)


class _ForbiddenClient:
    """Provide the small part of the httpx streaming interface used by the tool."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> _ForbiddenClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def stream(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _ForbiddenResponse:
        del method, url, kwargs
        return _ForbiddenResponse()


class _ContentResponse:
    """Provide deterministic streamed text with configurable headers."""

    is_redirect = False
    encoding = "utf-8"

    def __init__(
        self,
        body: bytes,
        content_length: str | None,
        *,
        content_encoding: str | None = None,
    ) -> None:
        self._body = body
        self.headers = {"content-type": "text/plain"}
        if content_length is not None:
            self.headers["content-length"] = content_length
        if content_encoding is not None:
            self.headers["content-encoding"] = content_encoding

    def __enter__(self) -> _ContentResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self) -> object:
        yield self._body


class _ContentClient:
    """Provide one deterministic response to the fetch implementation."""

    def __init__(self, response: _ContentResponse) -> None:
        self.response = response

    def __enter__(self) -> _ContentClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def stream(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _ContentResponse:
        del method, url, kwargs
        return self.response


class _RedirectResponse:
    """Return one redirect without opening a real socket."""

    is_redirect = True
    headers = {"location": "http://127.0.0.1/private"}

    def __enter__(self) -> _RedirectResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _RedirectClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> _RedirectClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def stream(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _RedirectResponse:
        del method, url, kwargs
        return _RedirectResponse()


class _DoHResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "Status": 0,
            "Answer": [{"type": 1, "data": "10.0.0.8"}],
        }


class _DoHClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> _DoHClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def get(self, *args: object, **kwargs: object) -> _DoHResponse:
        del args, kwargs
        return _DoHResponse()


class _TimedOutDoHClient(_DoHClient):
    def get(self, *args: object, **kwargs: object) -> _DoHResponse:
        del args, kwargs
        request = httpx.Request("GET", "https://resolver.invalid/dns-query")
        raise httpx.ReadTimeout("timed out", request=request)


class _RateLimitedDoHResponse:
    def raise_for_status(self) -> None:
        request = httpx.Request("GET", "https://resolver.invalid/dns-query")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError(
            "rate limited",
            request=request,
            response=response,
        )


class _RateLimitedDoHClient(_DoHClient):
    def get(self, *args: object, **kwargs: object) -> _RateLimitedDoHResponse:
        del args, kwargs
        return _RateLimitedDoHResponse()


class FetchUrlTests(unittest.TestCase):
    """Verify that one inaccessible page does not raise out of the tool."""

    @patch("search_agent._validate_public_url")
    @patch("search_agent.httpx.Client", _ForbiddenClient)
    def test_http_403_becomes_a_tool_result(self, validate_url: object) -> None:
        del validate_url

        result = fetch_url.invoke({"url": "https://example.com/private"})

        self.assertEqual(
            json.loads(result),
            {
                "status": "error",
                "url": "https://example.com/private",
                "error": "HTTP 403",
                "http_status": 403,
                "failure_taxonomy": "access_blocked",
                "failure_type": "access_blocked",
                "retryable": False,
                "switch_source": True,
                "retry_with_another_source": True,
            },
        )

    def test_http_429_and_timeout_have_distinct_taxonomy(self) -> None:
        request = httpx.Request("GET", "https://example.com/page")
        rate_response = httpx.Response(429, request=request)
        rate_error = httpx.HTTPStatusError(
            "rate limited",
            request=request,
            response=rate_response,
        )
        timeout_error = httpx.ReadTimeout("timed out", request=request)

        with patch("search_agent._fetch_public_url", side_effect=rate_error):
            rate_limited = json.loads(
                fetch_url.invoke({"url": "https://example.com/page"})
            )
        with patch("search_agent._fetch_public_url", side_effect=timeout_error):
            timed_out = json.loads(
                fetch_url.invoke({"url": "https://example.com/page"})
            )

        self.assertEqual(rate_limited["failure_taxonomy"], "rate_limited")
        self.assertTrue(rate_limited["retryable"])
        self.assertEqual(timed_out["failure_taxonomy"], "timeout")
        self.assertTrue(timed_out["retryable"])

    def test_dns_rejection_keeps_machine_taxonomy(self) -> None:
        failure = URLValidationError(
            "Could not resolve host: example.invalid",
            taxonomy="dns_rejected",
        )
        with patch("search_agent._fetch_public_url", side_effect=failure):
            result = json.loads(
                fetch_url.invoke({"url": "https://example.invalid/page"})
            )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["failure_taxonomy"], "dns_rejected")
        self.assertFalse(result["retryable"])

    def test_proxy_resolver_timeout_and_rate_limit_keep_exact_taxonomy(self) -> None:
        probes = (
            (_TimedOutDoHClient, "timeout", True),
            (_RateLimitedDoHClient, "rate_limited", True),
        )

        for client_type, taxonomy, retryable in probes:
            with (
                self.subTest(taxonomy=taxonomy),
                patch("retrieval_safety.httpx.Client", client_type),
                self.assertRaises(URLValidationError) as raised,
            ):
                retrieval_safety._doh_public_addresses(  # noqa: SLF001
                    "public.example",
                    "http://127.0.0.1:17898",
                )
            self.assertEqual(raised.exception.taxonomy, taxonomy)
            self.assertEqual(raised.exception.retryable, retryable)

    def test_validation_timeout_is_retryable_not_a_security_rejection(self) -> None:
        error = URLValidationError(
            "resolver timed out",
            taxonomy="timeout",
            retryable=True,
        )

        with patch("search_agent._fetch_public_url", side_effect=error):
            result = json.loads(fetch_url.invoke({"url": "https://public.example"}))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failure_taxonomy"], "timeout")
        self.assertTrue(result["retryable"])

    @patch("search_agent._validate_public_url")
    def test_character_truncation_records_full_observation_and_partial_hash(
        self,
        validate_url: object,
    ) -> None:
        del validate_url
        body = ("x" * 1_500).encode()
        client = _ContentClient(_ContentResponse(body, str(len(body))))

        with patch("search_agent.httpx.Client", return_value=client):
            result = _fetch_public_url("https://fixture.test/page", 1_000)

        self.assertEqual(result["content_chars"], 1_000)
        self.assertEqual(result["content_length"], 1_500)
        self.assertEqual(result["observed_content_length"], 1_500)
        self.assertEqual(result["downloaded_bytes"], 1_500)
        self.assertEqual(result["http_content_length"], 1_500)
        self.assertTrue(result["truncated"])
        self.assertEqual(
            result["truncation_reasons"],
            ["returned_character_limit"],
        )

        budget = ResearchBudget(EFFORT_POLICIES["low"])
        budget.record_fetch(result)
        revision = budget.snapshot()["successful_sources"][0]["content_revisions"][0]
        self.assertEqual(
            revision["content_sha256_scope"],
            "returned_normalized_visible_text",
        )
        self.assertFalse(revision["content_sha256_complete"])

    @patch("search_agent._validate_public_url")
    def test_download_byte_truncation_keeps_full_content_length_unknown(
        self,
        validate_url: object,
    ) -> None:
        del validate_url
        body = b"0123456789abcdefghij"
        client = _ContentClient(_ContentResponse(body, str(len(body))))

        with (
            patch("search_agent.MAX_DOWNLOAD_BYTES", 10),
            patch("search_agent.httpx.Client", return_value=client),
        ):
            result = _fetch_public_url("https://fixture.test/page", 12_000)

        self.assertEqual(result["downloaded_bytes"], 10)
        self.assertEqual(result["observed_content_length"], 10)
        self.assertIsNone(result["content_length"])
        self.assertEqual(result["http_content_length"], 20)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reasons"], ["download_byte_limit"])

    @patch("search_agent._validate_public_url")
    def test_missing_or_invalid_http_content_length_is_null_not_zero(
        self,
        validate_url: object,
    ) -> None:
        del validate_url
        body = b"complete fixture text"

        for header in (None, "not-an-integer", "-1"):
            with self.subTest(header=header):
                client = _ContentClient(_ContentResponse(body, header))
                with patch("search_agent.httpx.Client", return_value=client):
                    result = _fetch_public_url(
                        "https://fixture.test/page",
                        12_000,
                    )

                self.assertIsNone(result["http_content_length"])
                self.assertEqual(result["content_length"], len(body))
                self.assertFalse(result["truncated"])

    @patch("search_agent._validate_public_url")
    def test_compressed_content_length_is_not_compared_to_decoded_bytes(
        self,
        validate_url: object,
    ) -> None:
        del validate_url
        client = _ContentClient(
            _ContentResponse(
                b"x",
                "25",
                content_encoding="gzip",
            )
        )

        with patch("search_agent.httpx.Client", return_value=client):
            result = _fetch_public_url("https://fixture.test/page", 12_000)

        self.assertEqual(result["downloaded_bytes"], 1)
        self.assertEqual(
            result["downloaded_bytes_scope"],
            "httpx_decoded_response_bytes",
        )
        self.assertEqual(result["http_content_length"], 25)
        self.assertEqual(result["http_content_encoding"], "gzip")
        self.assertFalse(result["truncated"])

    def test_proxy_doh_ignores_synthetic_system_dns_and_keeps_public_answers(
        self,
    ) -> None:
        public = ("208.80.154.224",)
        with (
            patch(
                "retrieval_safety._proxy_url_for",
                return_value="http://127.0.0.1:17898",
            ),
            patch(
                "retrieval_safety._doh_public_addresses",
                side_effect=[public, public],
            ),
            patch(
                "retrieval_safety.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("2001::1", 0))],
            ) as system_dns,
        ):
            validated = validate_public_url(
                "https://en.wikipedia.org/wiki/Public_example"
            )

        system_dns.assert_not_called()
        self.assertEqual(validated.addresses, public)
        self.assertEqual(validated.resolution_mode, "proxy_doh")

    @patch("retrieval_safety.httpx.Client", _DoHClient)
    def test_proxy_doh_rejects_private_answers(self) -> None:
        with self.assertRaisesRegex(URLValidationError, "non-public"):
            retrieval_safety._doh_public_addresses(  # noqa: SLF001
                "public-looking.example",
                "http://127.0.0.1:17898",
            )

    def test_proxy_doh_rejects_answer_drift_as_rebinding(self) -> None:
        with (
            patch(
                "retrieval_safety._proxy_url_for",
                return_value="http://127.0.0.1:17898",
            ),
            patch(
                "retrieval_safety._doh_public_addresses",
                side_effect=[
                    ("93.184.216.34",),
                    ("93.184.216.35",),
                ],
            ),
            self.assertRaises(URLValidationError) as raised,
        ):
            validate_public_url("https://public-looking.example/page")

        self.assertEqual(raised.exception.taxonomy, "dns_rebinding")

    def test_userinfo_and_private_literals_are_rejected_before_proxy(self) -> None:
        for url in (
            "https://user:password@example.com/page",
            "http://127.0.0.1/private",
            "http://[::1]/private",
            "http://169.254.169.254/latest/meta-data",
            "http://[64:ff9b::7f00:1]/nat64-loopback",
            "http://[ff02::1]/multicast",
        ):
            with self.subTest(url=url), self.assertRaises(URLValidationError):
                validate_public_url(url)

    def test_resolver_rejects_nat64_and_multicast_answers(self) -> None:
        for address in ("64:ff9b::7f00:1", "ff02::1"):
            with (
                self.subTest(address=address),
                self.assertRaises(URLValidationError),
            ):
                retrieval_safety._validate_address_set(  # noqa: SLF001
                    "public.example",
                    {address},
                )

    def test_direct_https_request_pins_ip_but_preserves_host_and_sni(self) -> None:
        validated = ValidatedURL(
            url="https://evidence.example/path?q=1",
            hostname="evidence.example",
            port=443,
            proxy_url=None,
            addresses=("93.184.216.34",),
            selected_address="93.184.216.34",
            resolution_mode="system_dns",
        )

        request = build_pinned_request(validated)
        try:
            self.assertEqual(request.url, "https://93.184.216.34/path?q=1")
            self.assertEqual(request.headers["Host"], "evidence.example")
            self.assertEqual(
                request.extensions["sni_hostname"],
                "evidence.example",
            )
        finally:
            request.transport.close()

    def test_proxy_https_request_pins_connect_ip_and_preserves_sni(self) -> None:
        validated = ValidatedURL(
            url="https://evidence.example/path",
            hostname="evidence.example",
            port=443,
            proxy_url="http://127.0.0.1:17898",
            addresses=("93.184.216.34",),
            selected_address="93.184.216.34",
            resolution_mode="proxy_doh",
        )

        request = build_pinned_request(validated)
        try:
            self.assertEqual(request.url, "https://93.184.216.34/path")
            self.assertEqual(request.headers["Host"], "evidence.example")
            self.assertEqual(
                request.extensions["sni_hostname"],
                "evidence.example",
            )
        finally:
            request.transport.close()

        wrapped = Mock()
        wrapped.start_tls.return_value = Mock()
        stream = retrieval_safety._SNIPinningStream(  # noqa: SLF001
            wrapped,
            selected_address="93.184.216.34",
            server_hostname="evidence.example",
        )
        context = ssl.create_default_context()
        stream.start_tls(context, "93.184.216.34")
        wrapped.start_tls.assert_called_once_with(
            context,
            "evidence.example",
            None,
        )

    @patch("search_agent.httpx.Client", _RedirectClient)
    def test_redirect_to_private_is_revalidated_and_rejected(self) -> None:
        public = object()
        with patch(
            "search_agent._validate_public_url",
            side_effect=[
                public,
                URLValidationError("Refusing non-public address: 127.0.0.1"),
            ],
        ):
            result = json.loads(
                fetch_url.invoke({"url": "https://public.example/start"})
            )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["failure_taxonomy"], "redirect_rejected")

    @patch("search_agent.httpx.Client", _ForbiddenClient)
    def test_target_http_status_is_not_masked_by_post_response_dns_failure(
        self,
    ) -> None:
        validation = ValidatedURL(
            url="https://public.example/page",
            hostname="public.example",
            port=443,
            proxy_url=None,
            addresses=("93.184.216.34",),
            selected_address="93.184.216.34",
            resolution_mode="system_dns",
        )
        with (
            patch("search_agent._validate_public_url", return_value=validation),
            patch(
                "search_agent.revalidate_public_url",
                side_effect=URLValidationError(
                    "resolver timed out",
                    taxonomy="timeout",
                    retryable=True,
                ),
            ) as revalidate,
        ):
            result = json.loads(fetch_url.invoke({"url": validation.url}))

        revalidate.assert_not_called()
        self.assertEqual(result["failure_taxonomy"], "access_blocked")
        self.assertEqual(result["http_status"], 403)


if __name__ == "__main__":
    unittest.main()
