"""Proxy-aware URL validation with direct pinning and DNS drift checks."""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.request import getproxies_environment, proxy_bypass_environment

import httpcore
import httpx
from httpcore._backends.base import SOCKET_OPTION, NetworkBackend, NetworkStream
from httpcore._backends.sync import SyncBackend


_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")
_LOCAL_HOST_SUFFIXES = (
    ".home.arpa",
    ".internal",
    ".intranet",
    ".lan",
    ".local",
    ".localhost",
)
_DOH_ENDPOINTS = (
    ("cloudflare", "https://cloudflare-dns.com/dns-query"),
    ("google", "https://dns.google/resolve"),
)
_UNSAFE_IPV6_TRANSLATION_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)
_SECURITY_TAXONOMIES = frozenset(
    {
        "dns_rebinding",
        "redirect_rejected",
        "ssrf_rejected",
    }
)


class URLValidationError(ValueError):
    """A safe, machine-classified URL rejection."""

    def __init__(
        self,
        message: str,
        *,
        taxonomy: str = "ssrf_rejected",
        retryable: bool = False,
    ) -> None:
        self.taxonomy = taxonomy
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class ValidatedURL:
    """Logical URL and a DNS-pinned address selected for one request."""

    url: str
    hostname: str
    port: int
    proxy_url: str | None
    addresses: tuple[str, ...]
    selected_address: str
    resolution_mode: str


@dataclass(frozen=True)
class PinnedRequest:
    """Direct-IP request details or a fixed-proxy request guarded by DoH checks."""

    url: str
    headers: dict[str, str]
    extensions: dict[str, Any]
    transport: httpx.BaseTransport


def canonicalize_http_url(url: str) -> str:
    """Normalize an HTTP URL for validation and duplicate suppression."""

    if _CONTROL_OR_SPACE.search(url) or "\\" in url:
        raise URLValidationError("URL contains whitespace, control, or backslash")
    parsed = urlparse(url)
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise URLValidationError("Only http:// and https:// URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise URLValidationError("URLs containing userinfo are not allowed")
    if not parsed.hostname:
        raise URLValidationError("URL must include a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise URLValidationError("URL contains an invalid port") from exc

    raw_hostname = parsed.hostname.rstrip(".")
    if "%" in raw_hostname:
        raise URLValidationError("URL hostname contains an invalid escape or zone ID")
    try:
        hostname = raw_hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise URLValidationError("URL hostname is not valid IDNA") from exc
    if not hostname:
        raise URLValidationError("URL must include a hostname")

    _reject_local_hostname(hostname)
    _reject_non_public_literal(hostname)

    default_port = 80 if scheme == "http" else 443
    port_text = f":{port}" if port is not None and port != default_port else ""
    bracketed = f"[{hostname}]" if ":" in hostname else hostname
    return urlunparse(
        (
            scheme,
            f"{bracketed}{port_text}",
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",
        )
    )


def validate_public_url(url: str) -> ValidatedURL:
    """Resolve twice through the active route and pin one public address."""

    canonical = canonicalize_http_url(url)
    parsed = urlparse(canonical)
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    literal = _ip_literal(hostname)
    proxy_url = _proxy_url_for(parsed.scheme, hostname)

    if literal is not None:
        addresses = (str(literal),)
        resolution_mode = "literal"
    elif proxy_url is None:
        first = public_addresses(hostname)
        second = public_addresses(hostname)
        addresses = _stable_answers(hostname, first, second)
        resolution_mode = "system_dns"
    else:
        first = _doh_public_addresses(hostname, proxy_url)
        second = _doh_public_addresses(hostname, proxy_url)
        addresses = _stable_answers(hostname, first, second)
        resolution_mode = "proxy_doh"

    return ValidatedURL(
        url=canonical,
        hostname=hostname,
        port=port,
        proxy_url=proxy_url,
        addresses=addresses,
        selected_address=addresses[0],
        resolution_mode=resolution_mode,
    )


def build_pinned_request(validated: ValidatedURL) -> PinnedRequest:
    """Build a request whose direct or proxy CONNECT target is the public IP."""

    parsed = urlparse(validated.url)
    default_port = 443 if parsed.scheme == "https" else 80
    address = validated.selected_address
    bracketed = f"[{address}]" if ":" in address else address
    port_text = f":{validated.port}" if validated.port != default_port else ""
    request_url = urlunparse(
        (
            parsed.scheme,
            f"{bracketed}{port_text}",
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",
        )
    )
    logical_host = (
        f"[{validated.hostname}]" if ":" in validated.hostname else validated.hostname
    )
    host_text = (
        f"{logical_host}:{validated.port}"
        if validated.port != default_port
        else logical_host
    )
    extensions: dict[str, Any] = (
        {"sni_hostname": validated.hostname} if parsed.scheme == "https" else {}
    )
    transport: httpx.BaseTransport = (
        httpx.HTTPTransport(trust_env=False)
        if validated.proxy_url is None
        else _PinnedProxyTransport(
            validated.proxy_url,
            selected_address=validated.selected_address,
            server_hostname=validated.hostname,
        )
    )
    return PinnedRequest(
        url=request_url,
        headers={"Host": host_text},
        extensions=extensions,
        transport=transport,
    )


def revalidate_public_url(validated: ValidatedURL) -> None:
    """Reject proxy/system DNS drift observed once the response is established."""

    if validated.resolution_mode == "literal":
        return
    if validated.proxy_url is None:
        observed = public_addresses(validated.hostname)
    else:
        observed = _doh_public_addresses(validated.hostname, validated.proxy_url)
    if observed != validated.addresses:
        raise URLValidationError(
            f"DNS answers changed during request for {validated.hostname}",
            taxonomy="dns_rebinding",
        )


def public_addresses(hostname: str) -> tuple[str, ...]:
    """Return every stable-system-DNS address after public-IP validation."""

    try:
        raw = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise URLValidationError(
            f"Could not resolve host: {hostname}",
            taxonomy="dns_rejected",
        ) from exc
    return _validate_address_set(hostname, raw)


def _doh_public_addresses(hostname: str, proxy_url: str) -> tuple[str, ...]:
    """Resolve a target through fixed public DoH using the already-fixed proxy."""

    last_error: URLValidationError | None = None
    for provider, endpoint in _DOH_ENDPOINTS:
        try:
            answers: set[str] = set()
            with httpx.Client(
                timeout=10,
                proxy=proxy_url,
                trust_env=False,
                follow_redirects=False,
                headers={"Accept": "application/dns-json"},
            ) as client:
                for record_type in ("A", "AAAA"):
                    response = client.get(
                        endpoint,
                        params={"name": hostname, "type": record_type},
                    )
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        taxonomy, retryable = _http_failure_taxonomy(status)
                        raise URLValidationError(
                            f"DoH resolver {provider} returned HTTP {status}",
                            taxonomy=taxonomy,
                            retryable=retryable,
                        ) from exc
                    try:
                        payload = response.json()
                    except (TypeError, ValueError) as exc:
                        raise URLValidationError(
                            f"DoH resolver {provider} returned invalid JSON",
                            taxonomy="network_error",
                            retryable=True,
                        ) from exc
                    if not isinstance(payload, dict):
                        raise URLValidationError(
                            f"DoH resolver {provider} returned an invalid payload",
                            taxonomy="network_error",
                            retryable=True,
                        )
                    if payload.get("Status") != 0:
                        raise URLValidationError(
                            f"DoH resolver {provider} rejected the DNS query",
                            taxonomy="dns_rejected",
                        )
                    raw_answers = payload.get("Answer", [])
                    if not isinstance(raw_answers, list):
                        raise URLValidationError(
                            f"DoH resolver {provider} returned an invalid answer",
                            taxonomy="network_error",
                            retryable=True,
                        )
                    for answer in raw_answers:
                        if not isinstance(answer, dict):
                            continue
                        candidate = str(answer.get("data", "")).strip()
                        try:
                            ipaddress.ip_address(candidate)
                        except ValueError:
                            continue
                        answers.add(candidate)
            return _validate_address_set(hostname, answers)
        except URLValidationError as exc:
            if exc.taxonomy in _SECURITY_TAXONOMIES:
                raise
            last_error = exc
            continue
        except httpx.TimeoutException as exc:
            last_error = URLValidationError(
                f"DoH resolver {provider} timed out",
                taxonomy="timeout",
                retryable=True,
            )
            last_error.__cause__ = exc
            continue
        except httpx.RequestError as exc:
            taxonomy = (
                "dns_error"
                if any(
                    isinstance(item, socket.gaierror) for item in _exception_chain(exc)
                )
                else "network_error"
            )
            last_error = URLValidationError(
                f"Could not reach DoH resolver {provider}",
                taxonomy=taxonomy,
                retryable=True,
            )
            last_error.__cause__ = exc
            continue
    if last_error is not None:
        raise last_error
    raise URLValidationError(
        f"Could not securely resolve host through proxy: {hostname}",
        taxonomy="network_error",
        retryable=True,
    )


def _stable_answers(
    hostname: str,
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> tuple[str, ...]:
    if first != second:
        raise URLValidationError(
            f"DNS answers changed during validation for {hostname}",
            taxonomy="dns_rebinding",
        )
    return first


def _validate_address_set(hostname: str, addresses: set[str]) -> tuple[str, ...]:
    if not addresses:
        raise URLValidationError(
            f"Host resolved to no addresses: {hostname}",
            taxonomy="dns_rejected",
        )
    normalized: set[str] = set()
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise URLValidationError(
                f"Resolver returned an invalid address for {hostname}",
                taxonomy="dns_rejected",
            ) from exc
        if not _is_safe_public_unicast(ip):
            raise URLValidationError(
                f"Refusing non-public address for {hostname}: {address}"
            )
        normalized.add(str(ip))
    return tuple(sorted(normalized))


def _proxy_url_for(scheme: str, hostname: str) -> str | None:
    proxies = getproxies_environment()
    if proxy_bypass_environment(hostname, proxies):
        return None
    raw = proxies.get(scheme) or proxies.get("all")
    if not raw:
        return None
    candidate = str(raw)
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise URLValidationError(
            "Configured retrieval proxy is not a valid HTTP(S) proxy",
            taxonomy="proxy_configuration",
        )
    return candidate


def _reject_local_hostname(hostname: str) -> None:
    if (
        hostname == "localhost"
        or hostname.endswith(_LOCAL_HOST_SUFFIXES)
        or ("." not in hostname and _ip_literal(hostname) is None)
    ):
        raise URLValidationError(f"Refusing local hostname: {hostname}")


def _ip_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        return None


def _reject_non_public_literal(hostname: str) -> None:
    literal = _ip_literal(hostname)
    if literal is not None and not _is_safe_public_unicast(literal):
        raise URLValidationError(f"Refusing non-public address: {hostname}")


def _is_safe_public_unicast(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Reject every non-public, non-unicast, or transition-space target."""

    if (
        not address.is_global
        or address.is_private
        or address.is_reserved
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
    ):
        return False
    if not isinstance(address, ipaddress.IPv6Address):
        return True
    if any(address in network for network in _UNSAFE_IPV6_TRANSLATION_NETWORKS):
        return False
    # Transition formats can embed an otherwise unreachable private IPv4
    # destination. They are unnecessary for evidence retrieval, so reject the
    # whole mechanism rather than trying to predict host routing behavior.
    return (
        address.ipv4_mapped is None
        and address.sixtofour is None
        and address.teredo is None
    )


def _http_failure_taxonomy(status: int) -> tuple[str, bool]:
    if status in {401, 403, 407, 451}:
        return "access_blocked", False
    if status == 429:
        return "rate_limited", True
    return "network_error", status >= 500


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain = [error]
    while len(chain) < 8:
        next_error = chain[-1].__cause__ or chain[-1].__context__
        if next_error is None or next_error in chain:
            break
        chain.append(next_error)
    return chain


class _SNIPinningStream(NetworkStream):
    """Preserve the logical TLS hostname while CONNECT targets a pinned IP."""

    def __init__(
        self,
        wrapped: NetworkStream,
        *,
        selected_address: str,
        server_hostname: str,
    ) -> None:
        self._wrapped = wrapped
        self._selected_address = selected_address
        self._server_hostname = server_hostname

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._wrapped.read(max_bytes, timeout)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._wrapped.write(buffer, timeout)

    def close(self) -> None:
        self._wrapped.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> NetworkStream:
        logical_hostname = (
            self._server_hostname
            if server_hostname == self._selected_address
            else server_hostname
        )
        secured = self._wrapped.start_tls(
            ssl_context,
            logical_hostname,
            timeout,
        )
        return _SNIPinningStream(
            secured,
            selected_address=self._selected_address,
            server_hostname=self._server_hostname,
        )

    def get_extra_info(self, info: str) -> Any:
        return self._wrapped.get_extra_info(info)


class _SNIPinningBackend(NetworkBackend):
    """Wrap HTTP-core sockets so an IP-pinned proxy tunnel retains SNI."""

    def __init__(self, *, selected_address: str, server_hostname: str) -> None:
        self._wrapped = SyncBackend()
        self._selected_address = selected_address
        self._server_hostname = server_hostname

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        stream = self._wrapped.connect_tcp(
            host,
            port,
            timeout,
            local_address,
            socket_options,
        )
        return _SNIPinningStream(
            stream,
            selected_address=self._selected_address,
            server_hostname=self._server_hostname,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        stream = self._wrapped.connect_unix_socket(path, timeout, socket_options)
        return _SNIPinningStream(
            stream,
            selected_address=self._selected_address,
            server_hostname=self._server_hostname,
        )

    def sleep(self, seconds: float) -> None:
        self._wrapped.sleep(seconds)


class _PinnedProxyTransport(httpx.HTTPTransport):
    """HTTPX proxy transport with IP CONNECT and logical hostname TLS."""

    def __init__(
        self,
        proxy_url: str,
        *,
        selected_address: str,
        server_hostname: str,
    ) -> None:
        super().__init__(trust_env=False)
        self._pool.close()  # noqa: SLF001
        proxy = httpx.Proxy(proxy_url)
        proxy_ssl_context = (
            ssl.create_default_context() if proxy.url.scheme == "https" else None
        )
        self._pool = httpcore.HTTPProxy(  # noqa: SLF001
            proxy_url=httpcore.URL(
                scheme=proxy.url.raw_scheme,
                host=proxy.url.raw_host,
                port=proxy.url.port,
                target=proxy.url.raw_path,
            ),
            proxy_auth=proxy.raw_auth,
            proxy_headers=proxy.headers.raw,
            ssl_context=ssl.create_default_context(),
            proxy_ssl_context=proxy_ssl_context,
            network_backend=_SNIPinningBackend(
                selected_address=selected_address,
                server_hostname=server_hostname,
            ),
        )


__all__ = [
    "PinnedRequest",
    "URLValidationError",
    "ValidatedURL",
    "build_pinned_request",
    "canonicalize_http_url",
    "public_addresses",
    "revalidate_public_url",
    "validate_public_url",
]
