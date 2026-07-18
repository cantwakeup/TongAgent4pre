"""Regression tests for recoverable page-fetch failures."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from agent_policy import EFFORT_POLICIES
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

    def stream(self, method: str, url: str) -> _ForbiddenResponse:
        del method, url
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

    def stream(self, method: str, url: str) -> _ContentResponse:
        del method, url
        return self.response


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
                "retry_with_another_source": True,
            },
        )

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


if __name__ == "__main__":
    unittest.main()
