"""Regression tests for recoverable page-fetch failures."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from search_agent import fetch_url


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


if __name__ == "__main__":
    unittest.main()
