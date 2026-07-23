"""Small, thread-safe, aggressively sanitized evaluation traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread
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
        persistent_trace_path: Path | None = None,
        partial_telemetry_path: Path | None = None,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        if max_text_chars < 0:
            msg = "max_text_chars must be non-negative"
            raise ValueError(msg)
        if max_collection_items < 0:
            msg = "max_collection_items must be non-negative"
            raise ValueError(msg)
        if heartbeat_interval_seconds <= 0:
            msg = "heartbeat_interval_seconds must be positive"
            raise ValueError(msg)
        self._max_text_chars = max_text_chars
        self._max_collection_items = max_collection_items
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._events: list[TraceEvent] = []
        self._persistent_trace_path = persistent_trace_path
        self._partial_telemetry_path = partial_telemetry_path
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stop_heartbeat = Event()
        self._heartbeat_thread: Thread | None = None
        self._closed = False
        self._request_start: datetime | None = None
        self._first_model_response: datetime | None = None
        self._first_tool_call: datetime | None = None
        self._last_progress_timestamp: datetime | None = None
        self._last_event_type: str | None = None
        self._started_tool_counts: dict[str, int] = {}
        self._completed_tool_counts: dict[str, int] = {}
        self._budget_snapshot: dict[str, JsonValue] | None = None
        self._reported_token_usage: dict[str, int] = {}
        self._responses_with_reported_usage = 0
        self._responses_without_usage = 0
        self._persistence_error: str | None = None
        if persistent_trace_path is not None or partial_telemetry_path is not None:
            with self._lock:
                self._persist_locked(heartbeat=True)
            self._heartbeat_thread = Thread(
                target=self._heartbeat_loop,
                name="evaluation-telemetry-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

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
            self._observe_event_locked(event)
            self._persist_locked(heartbeat=False)
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

    def close(self) -> None:
        """Flush durable telemetry and stop the best-effort heartbeat."""

        self._stop_heartbeat.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._heartbeat_interval_seconds + 0.5))
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._persist_locked(heartbeat=True)

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self._heartbeat_interval_seconds):
            with self._lock:
                if self._closed:
                    return
                self._persist_locked(heartbeat=True)

    def _observe_event_locked(self, event: TraceEvent) -> None:
        event_type = event.event_type
        payload = event.payload
        self._last_progress_timestamp = event.occurred_at
        self._last_event_type = event_type
        if event_type == "model_call_started" and self._request_start is None:
            self._request_start = event.occurred_at
        if (
            event_type
            in {
                "model_token_settled",
                "model_token_usage_unavailable",
                "model_token_budget_unverifiable",
                "model_call_finished",
            }
            and self._first_model_response is None
        ):
            self._first_model_response = event.occurred_at
        if event_type == "tool_call_started":
            if self._first_tool_call is None:
                self._first_tool_call = event.occurred_at
            tool_name = str(payload.get("tool_name") or "unknown")
            self._started_tool_counts[tool_name] = (
                self._started_tool_counts.get(tool_name, 0) + 1
            )
        if event_type in {
            "tool_call_finished",
            "tool_call_failed",
            "tool_call_budget_exceeded",
        }:
            tool_name = str(payload.get("tool_name") or "unknown")
            self._completed_tool_counts[tool_name] = (
                self._completed_tool_counts.get(tool_name, 0) + 1
            )
        budget = payload.get("budget")
        if isinstance(budget, dict):
            self._budget_snapshot = dict(budget)
        if event_type in {
            "model_token_settled",
            "model_token_budget_unverifiable",
        }:
            usage = payload.get("usage")
            if isinstance(usage, dict):
                self._responses_with_reported_usage += 1
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                ):
                    value = usage.get(key)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                    ):
                        self._reported_token_usage[key] = (
                            self._reported_token_usage.get(key, 0) + value
                        )
        if event_type in {
            "model_token_usage_unavailable",
            "model_token_budget_unverifiable",
        }:
            self._responses_without_usage += 1

    def _partial_payload_locked(self, *, heartbeat: bool) -> dict[str, JsonValue]:
        if self._responses_without_usage:
            token_usage_status = "usage_unavailable"
        elif self._responses_with_reported_usage:
            token_usage_status = "reported"
        else:
            token_usage_status = "pending"
        now = self._clock()
        return {
            "schema_version": 1,
            "process_id": os.getpid(),
            "request_start": _isoformat(self._request_start),
            "first_model_response": _isoformat(self._first_model_response),
            "first_tool_call": _isoformat(self._first_tool_call),
            "last_progress_timestamp": _isoformat(self._last_progress_timestamp),
            "last_flush_timestamp": now.isoformat(),
            "last_event_type": self._last_event_type,
            "trace_event_count": len(self._events),
            "heartbeat": heartbeat,
            "started_tool_counts": dict(sorted(self._started_tool_counts.items())),
            "completed_tool_counts": dict(sorted(self._completed_tool_counts.items())),
            "budget_snapshot": self._budget_snapshot,
            "token_usage_status": token_usage_status,
            "token_usage": (
                dict(sorted(self._reported_token_usage.items()))
                if self._reported_token_usage
                else None
            ),
            "responses_with_reported_usage": self._responses_with_reported_usage,
            "responses_without_usage": self._responses_without_usage,
            "persistence_error": self._persistence_error,
        }

    def _persist_locked(self, *, heartbeat: bool) -> None:
        try:
            if self._persistent_trace_path is not None:
                _atomic_replace_text(self._persistent_trace_path, self.jsonl())
            if self._partial_telemetry_path is not None:
                payload = self._partial_payload_locked(heartbeat=heartbeat)
                _atomic_replace_text(
                    self._partial_telemetry_path,
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            self._persistence_error = None
        except OSError as exc:
            # Telemetry must never alter the Agent's policy or terminal answer.
            self._persistence_error = f"{type(exc).__name__}: {exc}"


def _isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _atomic_replace_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
