"""Small, thread-safe, aggressively sanitized evaluation traces."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, JsonValue

from .schema import TraceEvent


WallClock = Callable[[], datetime]
_REDACTED = "[REDACTED]"
_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|cookie|password|passwd|"
    r"secret|credential|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token)(?:$|[_-])",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret)=)[^&#\s]+"
)
_NAMED_INLINE_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|"
    r"authorization|cookie|password|passwd|secret|credential)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_PROVIDER_TOKEN = re.compile(
    r"\b(?:"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{30,}"
    r")\b"
)
_BODY_TEXT_KEYS = frozenset({"body", "content", "page_content", "response_body"})


class TraceCollector:
    """Collect ordered JSON events without retaining secrets or large bodies."""

    def __init__(
        self,
        *,
        max_text_chars: int = 2_000,
        max_collection_items: int = 100,
        clock: WallClock | None = None,
    ) -> None:
        if max_text_chars < 0:
            msg = "max_text_chars must be non-negative"
            raise ValueError(msg)
        if max_collection_items < 0:
            msg = "max_collection_items must be non-negative"
            raise ValueError(msg)
        self._max_text_chars = max_text_chars
        self._max_collection_items = max_collection_items
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._events: list[TraceEvent] = []

    def record(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        **details: Any,
    ) -> TraceEvent:
        """Sanitize and append one event, returning the immutable record."""
        if not event_type.strip():
            msg = "event_type must contain non-whitespace text"
            raise ValueError(msg)
        merged: dict[str, Any] = dict(payload or {})
        merged.update(details)
        sanitized = sanitize_trace_value(
            merged,
            max_text_chars=self._max_text_chars,
            max_collection_items=self._max_collection_items,
        )
        if not isinstance(sanitized, dict):
            msg = "trace payload sanitization must preserve the root object"
            raise TypeError(msg)
        with self._lock:
            event = TraceEvent(
                sequence=len(self._events) + 1,
                event_type=event_type,
                occurred_at=self._clock(),
                payload=sanitized,
            )
            self._events.append(event)
            return event

    def snapshot(self) -> tuple[TraceEvent, ...]:
        """Return an immutable point-in-time copy of collected events."""
        with self._lock:
            return tuple(self._events)

    def jsonl(self) -> str:
        """Serialize all events as newline-terminated JSONL."""
        return "".join(
            event.model_dump_json(exclude_none=False) + "\n"
            for event in self.snapshot()
        )

    def write_jsonl(self, path: Path) -> None:
        """Atomically materialize the sanitized trace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.jsonl(), encoding="utf-8")
        temporary.replace(path)


def sanitize_trace_value(
    value: Any,
    *,
    max_text_chars: int = 2_000,
    max_collection_items: int = 100,
    _key: str = "",
) -> JsonValue:
    """Recursively convert a value to bounded JSON while redacting secrets."""
    if _key and _is_secret_key(_key):
        return _REDACTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = _redact_inline_secrets(value)
        if _key.casefold() in _BODY_TEXT_KEYS:
            return {
                "_trace_value": "omitted_body_text",
                "chars": len(redacted),
                "sha256": hashlib.sha256(redacted.encode()).hexdigest(),
            }
        if _key.casefold() in {"url", "uri", "target_url", "source_url"}:
            if len(redacted) <= 4_096:
                return redacted
        if len(redacted) > max_text_chars:
            return {
                "_trace_value": "omitted_large_text",
                "chars": len(redacted),
                "sha256": hashlib.sha256(redacted.encode()).hexdigest(),
            }
        return redacted
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return _redact_inline_secrets(str(value))
    if isinstance(value, BaseModel):
        return sanitize_trace_value(
            value.model_dump(mode="json"),
            max_text_chars=max_text_chars,
            max_collection_items=max_collection_items,
            _key=_key,
        )
    if isinstance(value, Mapping):
        items = list(value.items())
        bounded = items[:max_collection_items]
        sanitized_map: dict[str, JsonValue] = {}
        for raw_key, item in bounded:
            key = str(raw_key)
            sanitized_map[key] = sanitize_trace_value(
                item,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
                _key=key,
            )
        if len(items) > max_collection_items:
            sanitized_map["_trace_omitted_items"] = len(items) - len(bounded)
        return sanitized_map
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        bounded_items = list(value[:max_collection_items])
        sanitized_items = [
            sanitize_trace_value(
                item,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
            )
            for item in bounded_items
        ]
        if len(value) > max_collection_items:
            sanitized_items.append(
                {"_trace_omitted_items": len(value) - len(bounded_items)}
            )
        return sanitized_items
    if isinstance(value, (bytes, bytearray)):
        return {
            "_trace_value": "omitted_binary",
            "bytes": len(value),
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
        }
    return sanitize_trace_value(
        repr(value),
        max_text_chars=max_text_chars,
        max_collection_items=max_collection_items,
        _key=_key,
    )


def _is_secret_key(key: str) -> bool:
    normalized = key.strip().replace(" ", "_")
    compact = re.sub(r"[^A-Z0-9]", "", key.upper())
    return (
        bool(_SECRET_KEY.search(normalized))
        or compact
        in {
            "TOKEN",
            "REFRESHTOKEN",
        }
        or any(
            marker in compact
            for marker in (
                "APIKEY",
                "ACCESSTOKEN",
                "AUTHTOKEN",
                "AUTHORIZATION",
                "PASSWORD",
                "SECRET",
                "COOKIE",
                "CREDENTIAL",
            )
        )
    )


def _redact_inline_secrets(value: str) -> str:
    redacted = _BEARER.sub("Bearer [REDACTED]", value)
    redacted = _OPENAI_STYLE_KEY.sub(_REDACTED, redacted)
    redacted = _PROVIDER_TOKEN.sub(_REDACTED, redacted)
    redacted = _NAMED_INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        redacted,
    )
    return _QUERY_SECRET.sub(r"\1[REDACTED]", redacted)
