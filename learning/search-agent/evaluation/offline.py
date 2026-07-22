"""Deterministic offline search, fetch, and chat-model fixtures.

This module is production code for TongAgent's evaluation harness.  It does
not import test helpers and deliberately contains no network client or live
fallback.  A fixture miss is returned as ``fixture_not_found`` so an incomplete
fixture cannot silently turn an offline smoke run into an online experiment.

Canonical fixture layout::

    fixture/
    ├── manifest.json
    ├── search.json
    └── pages.json

``manifest.json`` points at the other two files::

    {
      "schema_version": 1,
      "fixture_id": "one-hop",
      "search_file": "search.json",
      "page_file": "pages.json"
    }

Search and page files may be mappings keyed by query/URL or lists of entries.
A page value may be a list (or ``{"responses": [...]}``) to model a
deterministic retry sequence.  Once the sequence is exhausted, the final
response is repeated.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, cast
from urllib.parse import urlsplit, urlunsplit

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field


FIXTURE_SCHEMA_VERSION = 1
FIXTURE_FETCHED_AT = "1970-01-01T00:00:00+00:00"
MIN_RELEVANCE_SCORE = 20
_DEFAULT_ANSWER = "Fixture research completed."
_RESPONSE_KEYS = frozenset({"response", "responses"})
_SYSTEM_ALIASES = {
    "simple_react": ("simple_react", "b1", "B1"),
    "vanilla_deepagents": ("vanilla_deepagents", "b2", "B2"),
    "tongagent": ("tongagent", "b3", "B3"),
}
_T = TypeVar("_T")


class FixtureFormatError(ValueError):
    """Raised when fixture files are ambiguous, unsafe, or malformed."""


def normalize_query(query: str) -> str:
    """Return the exact-match key used for fixture search queries.

    Normalization is intentionally conservative: Unicode compatibility
    normalization, whitespace collapsing, and case folding.  It never applies
    fuzzy, substring, token-set, or semantic matching.
    """

    normalized = unicodedata.normalize("NFKC", str(query))
    return " ".join(normalized.split()).casefold()


def normalize_url(url: str) -> str:
    """Return a conservative exact-match key for a fixture URL.

    Scheme and hostname case, a trailing DNS dot, default ports, an empty root
    path, and fragments are normalized.  Path and query ordering remain exact;
    in particular, this function does not perform prefix or same-site matching.
    Malformed strings remain deterministic keys and therefore produce
    ``fixture_not_found`` rather than triggering any external lookup.
    """

    raw = unicodedata.normalize("NFKC", str(url)).strip()
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return raw
    if not parsed.scheme or hostname is None:
        return raw

    scheme = parsed.scheme.casefold()
    host = hostname.rstrip(".").casefold()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    port_suffix = "" if port is None or default_port else f":{port}"
    netloc = f"{userinfo}{host}{port_suffix}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FixtureFormatError(f"Fixture file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureFormatError(
            f"Fixture file is not valid JSON: {path}: {exc.msg}"
        ) from exc


def _safe_child(root: Path, reference: str) -> Path:
    candidate = (root / reference).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FixtureFormatError(
            f"Fixture file escapes its directory: {reference}"
        ) from exc
    return candidate


def _manifest_source(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    kind: str,
    default_names: Sequence[str],
) -> Any:
    singular = "page" if kind == "pages" else "search"
    file_keys = (f"{singular}_file", f"{kind}_file")
    for key in file_keys:
        value = manifest.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise FixtureFormatError(f"manifest.{key} must be a file name")
            return _json_load(_safe_child(root, value))

    files = manifest.get("files")
    if isinstance(files, Mapping):
        value = files.get(singular, files.get(kind))
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise FixtureFormatError(
                    f"manifest.files.{singular} must be a file name"
                )
            return _json_load(_safe_child(root, value))

    for key in (kind, singular):
        value = manifest.get(key)
        if isinstance(value, str):
            return _json_load(_safe_child(root, value))
        if value is not None:
            return value

    for name in default_names:
        candidate = root / name
        if candidate.is_file():
            return _json_load(candidate)
    return {}


def _as_response_sequence(value: Any, *, label: str) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping) and set(value).intersection(_RESPONSE_KEYS):
        value = value.get("responses", value.get("response"))
    if isinstance(value, Mapping):
        raw_responses = [value]
    elif isinstance(value, list):
        raw_responses = value
    else:
        raise FixtureFormatError(f"{label} response must be an object or list")
    if not raw_responses:
        raise FixtureFormatError(f"{label} response sequence must not be empty")
    responses: list[dict[str, Any]] = []
    for index, response in enumerate(raw_responses):
        if not isinstance(response, Mapping):
            raise FixtureFormatError(f"{label} response {index + 1} must be an object")
        responses.append(deepcopy(dict(response)))
    return tuple(responses)


def _unwrap_collection(raw: Any, *, kind: str) -> Any:
    if (
        isinstance(raw, Mapping)
        and kind in raw
        and len(raw) == 1
        and isinstance(raw[kind], (Mapping, list))
    ):
        return raw[kind]
    return raw


def _load_entries(
    raw: Any,
    *,
    kind: str,
    key_field: str,
    normalizer: Callable[[str], str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    raw = _unwrap_collection(raw, kind=kind)
    entries: list[tuple[str, Any]] = []
    if isinstance(raw, Mapping):
        entries = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise FixtureFormatError(f"{kind}[{index}] must be an object")
            key = item.get(key_field)
            if not isinstance(key, str) or not key.strip():
                raise FixtureFormatError(
                    f"{kind}[{index}].{key_field} must be a non-empty string"
                )
            if "responses" in item:
                value = item["responses"]
            elif "response" in item:
                value = item["response"]
            else:
                value = {
                    field_name: deepcopy(field_value)
                    for field_name, field_value in item.items()
                    if field_name != key_field
                }
            entries.append((key, value))
    else:
        raise FixtureFormatError(f"{kind} fixture must be an object or list")

    loaded: dict[str, tuple[dict[str, Any], ...]] = {}
    original_keys: dict[str, str] = {}
    for raw_key, value in entries:
        normalized = normalizer(raw_key)
        if not normalized:
            raise FixtureFormatError(f"{kind} contains an empty normalized key")
        if normalized in loaded:
            raise FixtureFormatError(
                f"{kind} keys normalize to the same exact-match key: "
                f"{original_keys[normalized]!r} and {raw_key!r}"
            )
        original_keys[normalized] = raw_key
        loaded[normalized] = _as_response_sequence(value, label=f"{kind}[{raw_key!r}]")
    return loaded


@dataclass
class _BackendRuntime:
    """Mutable counters separated from immutable fixture definitions."""

    search_offsets: dict[str, int] = field(default_factory=dict)
    fetch_offsets: dict[str, int] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default_factory=RLock, repr=False)


class FixtureBackend:
    """Strict, deterministic replacement for live search and page fetching."""

    def __init__(
        self,
        *,
        searches: Mapping[str, Any],
        pages: Mapping[str, Any],
        manifest: Mapping[str, Any] | None = None,
        root: Path | None = None,
    ) -> None:
        """Build a backend from already-loaded JSON-compatible data."""

        self.root = root.resolve() if root is not None else None
        self.manifest = deepcopy(dict(manifest or {}))
        self._searches = _load_entries(
            searches,
            kind="searches",
            key_field="query",
            normalizer=normalize_query,
        )
        self._pages = _load_entries(
            pages,
            kind="pages",
            key_field="url",
            normalizer=normalize_url,
        )
        self._runtime = _BackendRuntime()
        canonical = json.dumps(
            {
                "schema_version": FIXTURE_SCHEMA_VERSION,
                "searches": self._searches,
                "pages": self._pages,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.fixture_id = str(
            self.manifest.get("fixture_id", self.revision[:12])
        ).strip()

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        manifest_name: str = "manifest.json",
    ) -> FixtureBackend:
        """Load a manifest and its search/page JSON without any live fallback."""

        supplied = Path(path)
        manifest_path = (
            supplied.resolve()
            if supplied.is_file()
            else (supplied / manifest_name).resolve()
        )
        manifest_raw = _json_load(manifest_path)
        if not isinstance(manifest_raw, Mapping):
            raise FixtureFormatError("Fixture manifest must be a JSON object")
        manifest = dict(manifest_raw)
        schema_version = manifest.get("schema_version", FIXTURE_SCHEMA_VERSION)
        if schema_version != FIXTURE_SCHEMA_VERSION:
            raise FixtureFormatError(
                "Unsupported fixture schema_version: "
                f"{schema_version!r}; expected {FIXTURE_SCHEMA_VERSION}"
            )
        root = manifest_path.parent
        searches = _manifest_source(
            root,
            manifest,
            kind="searches",
            default_names=("search.json", "searches.json"),
        )
        pages = _manifest_source(
            root,
            manifest,
            kind="pages",
            default_names=("pages.json", "page.json"),
        )
        if not isinstance(searches, (Mapping, list)):
            raise FixtureFormatError("Search fixture must be a JSON object or list")
        if not isinstance(pages, (Mapping, list)):
            raise FixtureFormatError("Page fixture must be a JSON object or list")
        return cls(
            searches=cast("Mapping[str, Any]", searches)
            if isinstance(searches, Mapping)
            else {"searches": searches},
            pages=cast("Mapping[str, Any]", pages)
            if isinstance(pages, Mapping)
            else {"pages": pages},
            manifest=manifest,
            root=root,
        )

    @property
    def search_queries(self) -> tuple[str, ...]:
        """Return normalized search keys in fixture order."""

        return tuple(self._searches)

    @property
    def page_urls(self) -> tuple[str, ...]:
        """Return normalized page keys in fixture order."""

        return tuple(self._pages)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return a detached snapshot of all backend calls."""

        with self._runtime.lock:
            return deepcopy(self._runtime.calls)

    def reset(self) -> None:
        """Reset retry cursors and call history for an explicit fresh run."""

        with self._runtime.lock:
            self._runtime.search_offsets.clear()
            self._runtime.fetch_offsets.clear()
            self._runtime.calls.clear()

    def _next_response(
        self,
        *,
        kind: str,
        normalized_key: str,
        original_key: str,
    ) -> tuple[dict[str, Any] | None, int | None]:
        collection = self._searches if kind == "search" else self._pages
        offsets = (
            self._runtime.search_offsets
            if kind == "search"
            else self._runtime.fetch_offsets
        )
        with self._runtime.lock:
            sequence = collection.get(normalized_key)
            if sequence is None:
                self._runtime.calls.append(
                    {
                        "sequence": len(self._runtime.calls) + 1,
                        "tool": "web_search" if kind == "search" else "fetch_url",
                        "target": original_key,
                        "normalized_target": normalized_key,
                        "fixture_found": False,
                        "response_index": None,
                        "status": "fixture_not_found",
                    }
                )
                return None, None
            offset = offsets.get(normalized_key, 0)
            response_index = min(offset, len(sequence) - 1)
            offsets[normalized_key] = offset + 1
            response = deepcopy(sequence[response_index])
            self._runtime.calls.append(
                {
                    "sequence": len(self._runtime.calls) + 1,
                    "tool": "web_search" if kind == "search" else "fetch_url",
                    "target": original_key,
                    "normalized_target": normalized_key,
                    "fixture_found": True,
                    "response_index": response_index,
                    "status": str(response.get("status", "success")),
                }
            )
            return response, response_index

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Return one exact fixture search response as a detached dictionary."""

        normalized = normalize_query(query)
        payload, response_index = self._next_response(
            kind="search",
            normalized_key=normalized,
            original_key=str(query),
        )
        if payload is None:
            return {
                "status": "fixture_not_found",
                "query": query,
                "error": "fixture_not_found",
                "failure_class": "fixture_not_found",
                "retryable": False,
                "provider_success": False,
                "provider_outcome": "not_called",
                "nonempty_search": False,
                "relevant_search": False,
                "provider_failure": False,
                "relevant_results": 0,
                "results": [],
                "fixture_revision": self.revision,
            }

        payload.setdefault("status", "success")
        payload.setdefault("query", query)
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise FixtureFormatError(
                f"Search response for {query!r} has non-list results"
            )
        limit = max(0, int(max_results))
        payload["results"] = deepcopy(raw_results[:limit])
        status_success = payload["status"] == "success"
        relevant_results = sum(
            isinstance(item, Mapping)
            and int(item.get("relevance_score", 0)) >= MIN_RELEVANCE_SCORE
            for item in payload["results"]
        )
        payload.setdefault("provider_success", status_success)
        payload.setdefault(
            "provider_outcome", "success" if status_success else "failure"
        )
        payload.setdefault("nonempty_search", bool(payload["results"]))
        payload.setdefault("relevant_results", relevant_results)
        payload.setdefault("relevant_search", relevant_results > 0)
        payload.setdefault("provider_failure", not status_success)
        payload.setdefault("fixture_response_index", response_index)
        payload.setdefault("fixture_revision", self.revision)
        return payload

    def fetch(self, url: str, max_chars: int = 12_000) -> dict[str, Any]:
        """Return the next exact fixture response for one normalized URL."""

        normalized = normalize_url(url)
        payload, response_index = self._next_response(
            kind="page",
            normalized_key=normalized,
            original_key=str(url),
        )
        if payload is None:
            return {
                "status": "fixture_not_found",
                "url": url,
                "requested_url": url,
                "error": "fixture_not_found",
                "failure_class": "fixture_not_found",
                "retryable": False,
                "provider_outcome": "not_called",
                "fixture_revision": self.revision,
            }

        payload.setdefault("status", "success")
        payload.setdefault("url", url)
        payload.setdefault("requested_url", url)
        payload.setdefault("fixture_response_index", response_index)
        payload.setdefault("fixture_revision", self.revision)
        if payload["status"] != "success":
            return payload

        content = str(payload.get("content", ""))
        original_length = len(content)
        limit = max(0, int(max_chars))
        truncated_here = original_length > limit
        payload["content"] = content[:limit]
        payload["content_chars"] = len(payload["content"])
        payload.setdefault("content_length", original_length)
        payload.setdefault("observed_content_length", original_length)
        payload.setdefault("content_length_scope", "fixture_normalized_visible_text")
        payload.setdefault("fetched_at", FIXTURE_FETCHED_AT)
        existing_reasons = list(payload.get("truncation_reasons", []))
        if truncated_here and "returned_character_limit" not in existing_reasons:
            existing_reasons.append("returned_character_limit")
        payload["truncated"] = bool(payload.get("truncated", False) or truncated_here)
        payload["truncation_reasons"] = existing_reasons
        return payload

    def web_search(self, query: str, max_results: int = 5) -> str:
        """Tool-compatible JSON wrapper around :meth:`search`."""

        return json.dumps(
            self.search(query, max_results=max_results),
            ensure_ascii=False,
            indent=2,
        )

    def fetch_url(self, url: str, max_chars: int = 12_000) -> str:
        """Tool-compatible JSON wrapper around :meth:`fetch`."""

        return json.dumps(
            self.fetch(url, max_chars=max_chars),
            ensure_ascii=False,
            indent=2,
        )

    def as_tools(self) -> list[BaseTool]:
        """Build LangChain tools backed exclusively by this fixture instance."""

        backend = self

        @tool("web_search")
        def fixture_web_search(query: str, max_results: int = 5) -> str:
            """Search deterministic offline fixture results by exact query."""

            return backend.web_search(query, max_results=max_results)

        @tool("fetch_url")
        def fixture_fetch_url(url: str, max_chars: int = 12_000) -> str:
            """Fetch a deterministic offline fixture page by exact URL."""

            return backend.fetch_url(url, max_chars=max_chars)

        return [fixture_web_search, fixture_fetch_url]


def _tool_name(candidate: Any) -> str:
    if isinstance(candidate, BaseTool):
        return candidate.name
    if isinstance(candidate, Mapping):
        direct = candidate.get("name")
        if isinstance(direct, str):
            return direct
        function = candidate.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            return str(function["name"])
    name = getattr(candidate, "name", None) or getattr(candidate, "__name__", None)
    return str(name or "")


def _select_variant(value: Any, system_id: str) -> Any:
    if not isinstance(value, Mapping) or "tool" in value or "name" in value:
        return value
    aliases = (system_id, *_SYSTEM_ALIASES.get(system_id, ()))
    for alias in aliases:
        if alias in value:
            return value[alias]
    return value.get("default")


def _normalize_script(raw: Any, *, system_id: str) -> tuple[dict[str, Any], ...]:
    selected = _select_variant(raw, system_id)
    if selected is None:
        return ()
    if isinstance(selected, Mapping) and ("tool" in selected or "name" in selected):
        selected = [selected]
    if not isinstance(selected, list):
        raise FixtureFormatError(
            "metadata.research_script must be a list or a system-keyed mapping"
        )
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        if not isinstance(item, Mapping):
            raise FixtureFormatError(
                f"metadata.research_script[{index}] must be an object"
            )
        name = item.get("tool", item.get("name"))
        if not isinstance(name, str) or not name.strip():
            raise FixtureFormatError(
                f"metadata.research_script[{index}] needs tool/name"
            )
        args = item.get("args", {})
        if not isinstance(args, Mapping):
            raise FixtureFormatError(
                f"metadata.research_script[{index}].args must be an object"
            )
        action = deepcopy(dict(item))
        action["tool"] = name.strip()
        action["args"] = deepcopy(dict(args))
        actions.append(action)
    if len(actions) > 999:
        raise FixtureFormatError("research_script supports at most 999 actions")
    return tuple(actions)


def _string_answer(raw: Any, *, system_id: str) -> str:
    selected = _select_variant(raw, system_id)
    if selected is None:
        return _DEFAULT_ANSWER
    if not isinstance(selected, str):
        raise FixtureFormatError(
            "metadata.answer must be a string or a system-keyed string mapping"
        )
    return selected


def _substitute(value: _T, replacements: Mapping[str, str]) -> _T:
    if isinstance(value, str):
        rendered = value
        for key, replacement in replacements.items():
            rendered = rendered.replace(f"${{{key}}}", replacement)
        return cast("_T", rendered)
    if isinstance(value, list):
        return cast("_T", [_substitute(item, replacements) for item in value])
    if isinstance(value, Mapping):
        return cast(
            "_T",
            {str(key): _substitute(item, replacements) for key, item in value.items()},
        )
    return value


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        text += json.dumps(tool_calls, ensure_ascii=False, sort_keys=True, default=str)
    return text


def _estimated_tokens(text: str) -> int:
    """Use a deterministic approximation, explicitly not provider accounting."""

    return max(1, (len(text) + 3) // 4)


@dataclass
class FixtureModelRuntime:
    """Shared call allocator used by independently bound model clones."""

    next_message_sequence: int = 1
    next_tool_sequence: int = 1
    call_history: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default_factory=RLock, repr=False)

    def allocate(
        self,
        *,
        task_key: str,
        script_index: int | None,
    ) -> tuple[str, str | None]:
        """Atomically allocate deterministic message and optional tool IDs."""

        with self.lock:
            message_sequence = self.next_message_sequence
            self.next_message_sequence += 1
            message_id = f"fixture-message-{task_key}-{message_sequence:04d}"
            if script_index is None:
                return message_id, None
            tool_sequence = self.next_tool_sequence
            self.next_tool_sequence += 1
            tool_call_id = (
                f"fixture-{task_key}-s{script_index + 1:03d}-c{tool_sequence:04d}"
            )
            return message_id, tool_call_id

    def record(self, call: Mapping[str, Any]) -> None:
        """Append one detached model-call record."""

        with self.lock:
            self.call_history.append(deepcopy(dict(call)))

    def snapshot(self) -> list[dict[str, Any]]:
        """Return detached model call history."""

        with self.lock:
            return deepcopy(self.call_history)


class FixtureChatModel(BaseChatModel):
    """Deterministic tool-calling chat model driven by task metadata.

    ``metadata.research_script`` is a list of ``{"tool": ..., "args": ...}``
    actions, or a mapping keyed by ``simple_react``, ``vanilla_deepagents``,
    ``tongagent`` (also B1/B2/B3), and optionally ``default``.  The same script
    mechanism can call baseline tools (``web_search`` and ``fetch_url``) and
    TongAgent tools (``record_evidence``, ``get_evidence_graph``,
    ``update_subquestion``, and ``write_file``).

    Actions may optionally declare ``phase`` (``research`` or ``report``),
    one-based ``research_cycle``, and ``stop_cycle=true``.  These gates let a
    deterministic script exercise the real Stage 03D outer loop: future-cycle
    actions remain pending, and a completed stop action ends only the current
    inner-agent invocation.  Actions without those fields retain the baseline
    behavior and run as soon as their tool is bound.

    Tool-call completion is reconstructed from message history, so checkpoint
    replay does not depend on a process-local script cursor.  Tool call IDs
    encode their one-based script step and are globally unique within the
    shared :class:`FixtureModelRuntime`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    question: str
    task_metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)
    system_id: str = "default"
    answer: str = _DEFAULT_ANSWER
    research_script: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple, exclude=True, repr=False
    )
    planner_data: dict[str, Any] = Field(default_factory=dict, exclude=True)
    bound_tools: tuple[Any, ...] = Field(
        default_factory=tuple, exclude=True, repr=False
    )
    bound_tool_choice: str | None = Field(default=None, exclude=True)
    runtime: FixtureModelRuntime = Field(
        default_factory=FixtureModelRuntime, exclude=True, repr=False
    )
    model_name: str = "fixture-chat-model"

    @classmethod
    def from_task(
        cls,
        task: Any,
        *,
        system_id: str = "default",
        runtime: FixtureModelRuntime | None = None,
    ) -> FixtureChatModel:
        """Create a model from an ``EvalTask`` or equivalent mapping."""

        if isinstance(task, Mapping):
            task_id = task.get("id")
            question = task.get("question")
            metadata = task.get("metadata", {})
        else:
            task_id = getattr(task, "id", None)
            question = getattr(task, "question", None)
            metadata = getattr(task, "metadata", {})
        if not isinstance(task_id, str) or not task_id.strip():
            raise FixtureFormatError("Fixture task id must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise FixtureFormatError("Fixture task question must be a non-empty string")
        if not isinstance(metadata, Mapping):
            raise FixtureFormatError("Fixture task metadata must be an object")
        task_metadata = deepcopy(dict(metadata))
        script = _normalize_script(
            task_metadata.get("research_script", []),
            system_id=system_id,
        )
        answer = _string_answer(
            task_metadata.get("answer", _DEFAULT_ANSWER),
            system_id=system_id,
        )
        raw_planner = task_metadata.get(
            "planner",
            task_metadata.get("plan", task_metadata.get("research_plan", {})),
        )
        if raw_planner is not None and not isinstance(raw_planner, Mapping):
            raise FixtureFormatError("metadata.planner/plan must be an object")
        return cls(
            task_id=task_id,
            question=question,
            task_metadata=task_metadata,
            system_id=system_id,
            answer=answer,
            research_script=script,
            planner_data=deepcopy(dict(raw_planner or {})),
            runtime=runtime or FixtureModelRuntime(),
        )

    @property
    def _llm_type(self) -> str:
        return "fixture_chat_model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_id": self.task_id,
            "system_id": self.system_id,
        }

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        """Return tool names without exposing the mutable tool objects."""

        return tuple(name for item in self.bound_tools if (name := _tool_name(item)))

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Return call history shared by this model and its bound clones."""

        return self.runtime.snapshot()

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return an independent binding clone that shares only runtime state."""

        del kwargs
        return self.model_copy(
            update={
                "bound_tools": tuple(tools),
                "bound_tool_choice": tool_choice,
            },
            deep=False,
        )

    def _task_key(self) -> str:
        return hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:10]

    def _completed_script_steps(self, messages: Sequence[BaseMessage]) -> set[int]:
        task_key = re.escape(self._task_key())
        pattern = re.compile(
            rf"^fixture-{task_key}-s(?P<step>[0-9]{{3}})-c[0-9]{{4,}}$"
        )
        completed: set[int] = set()
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            match = pattern.fullmatch(str(message.tool_call_id))
            if match is not None:
                completed.add(int(match.group("step")) - 1)
        return completed

    @staticmethod
    def _workflow_position(
        messages: Sequence[BaseMessage],
    ) -> tuple[str, int, int]:
        """Return current phase, one-based research cycle, and phase marker.

        The outer TongAgent graph emits durable ``research-step-*`` and
        ``report-step-*`` human-message IDs.  Content checks retain compatibility
        with equivalent graph nodes that omit IDs.  Baseline agents have no
        marker and therefore remain in the neutral ``baseline`` phase.
        """

        research_cycle = 0
        phase = "baseline"
        marker_index = -1
        for index, message in enumerate(messages):
            if not isinstance(message, HumanMessage):
                continue
            message_id = str(message.id or "")
            content = _message_text(message)
            if message_id.startswith("research-step-") or "[RESEARCH STEP]" in content:
                research_cycle += 1
                phase = "research"
                marker_index = index
            elif (
                message_id.startswith("report-step-") or "[FINAL SYNTHESIS]" in content
            ):
                phase = "report"
                marker_index = index
        return phase, research_cycle, marker_index

    def _action_applies(
        self,
        action: Mapping[str, Any],
        *,
        phase: str,
        research_cycle: int,
    ) -> bool:
        systems = action.get("systems", action.get("system_ids"))
        if systems is not None:
            if isinstance(systems, str):
                allowed = {systems}
            elif isinstance(systems, list) and all(
                isinstance(item, str) for item in systems
            ):
                allowed = set(systems)
            else:
                raise FixtureFormatError(
                    "research_script action systems must be strings"
                )
            aliases = {self.system_id, *_SYSTEM_ALIASES.get(self.system_id, ())}
            if not aliases.intersection(allowed):
                return False

        configured_phase = action.get("phase")
        if configured_phase is not None:
            if configured_phase not in {"research", "report", "baseline"}:
                raise FixtureFormatError(
                    "research_script action phase must be research, report, or baseline"
                )
            if configured_phase != phase:
                return False

        configured_cycle = action.get("research_cycle")
        if configured_cycle is not None:
            if (
                isinstance(configured_cycle, bool)
                or not isinstance(configured_cycle, int)
                or configured_cycle < 1
            ):
                raise FixtureFormatError(
                    "research_script action research_cycle must be a positive integer"
                )
            if phase != "research" or configured_cycle != research_cycle:
                return False
        return True

    def _current_cycle_stopped(
        self,
        messages: Sequence[BaseMessage],
        *,
        marker_index: int,
    ) -> bool:
        """Return whether a stop action completed after the latest phase marker."""

        if marker_index < 0:
            return False
        completed_since_marker = self._completed_script_steps(
            messages[marker_index + 1 :]
        )
        for index in completed_since_marker:
            if index >= len(self.research_script):
                continue
            action = self.research_script[index]
            stop_cycle = action.get("stop_cycle", False)
            if not isinstance(stop_cycle, bool):
                raise FixtureFormatError(
                    "research_script action stop_cycle must be a boolean"
                )
            if stop_cycle:
                return True
        return False

    def _next_action(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[int, dict[str, Any]] | None:
        completed = self._completed_script_steps(messages)
        phase, research_cycle, marker_index = self._workflow_position(messages)
        if self._current_cycle_stopped(messages, marker_index=marker_index):
            return None
        available = set(self.bound_tool_names)
        strict_tools = bool(self.task_metadata.get("strict_fixture_tools", False))
        for index, action in enumerate(self.research_script):
            if index in completed or not self._action_applies(
                action,
                phase=phase,
                research_cycle=research_cycle,
            ):
                continue
            name = str(action["tool"])
            if name not in available:
                if strict_tools and not bool(action.get("optional", False)):
                    raise FixtureFormatError(
                        f"Scripted fixture tool is not bound: {name}"
                    )
                continue
            return index, action
        return None

    def _render_args(self, action: Mapping[str, Any]) -> dict[str, Any]:
        replacements = {
            "answer": self.answer,
            "question": self.question,
            "task_id": self.task_id,
            "system_id": self.system_id,
        }
        metadata_replacements = self.task_metadata.get("template_values", {})
        if isinstance(metadata_replacements, Mapping):
            replacements.update(
                {
                    str(key): str(value)
                    for key, value in metadata_replacements.items()
                    if isinstance(value, (str, int, float, bool))
                }
            )
        return cast(
            "dict[str, Any]",
            _substitute(deepcopy(dict(action.get("args", {}))), replacements),
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        next_action = self._next_action(messages)
        script_index = next_action[0] if next_action is not None else None
        message_id, tool_call_id = self.runtime.allocate(
            task_key=self._task_key(),
            script_index=script_index,
        )
        input_tokens = _estimated_tokens(
            "\n".join(_message_text(message) for message in messages)
        )
        if next_action is None:
            content = self.answer
            tool_calls: list[ToolCall] = []
            output_text = content
            action_name = None
        else:
            index, action = next_action
            args = self._render_args(action)
            action_name = str(action["tool"])
            tool_calls = [
                ToolCall(
                    name=action_name,
                    args=args,
                    id=cast("str", tool_call_id),
                )
            ]
            content = str(action.get("content", ""))
            output_text = json.dumps(
                {"tool": action_name, "args": args},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            script_index = index
        output_tokens = _estimated_tokens(output_text)
        message = AIMessage(
            id=message_id,
            content=content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata={
                "fixture": True,
                "task_id": self.task_id,
                "system_id": self.system_id,
                "script_index": script_index,
            },
        )
        self.runtime.record(
            {
                "message_id": message_id,
                "tool_call_id": tool_call_id,
                "tool": action_name,
                "script_index": script_index,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "bound_tools": list(self.bound_tool_names),
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def planner_payload(
        self,
        topic: str | None = None,
        max_subquestions: int = 3,
    ) -> dict[str, Any]:
        """Return a deterministic ``PlanDraft``-compatible dictionary."""

        requested_topic = " ".join((topic or self.question).split())
        raw = deepcopy(self.planner_data)
        raw.setdefault("objective", requested_topic)
        raw.setdefault(
            "subquestions",
            [
                {
                    "question": requested_topic,
                    "rationale": "Deterministic fixture subquestion.",
                    "depends_on": [],
                }
            ],
        )
        raw.setdefault(
            "completion_criteria",
            ["Answer the fixture question from registered fixture evidence."],
        )
        subquestions = raw.get("subquestions")
        if not isinstance(subquestions, list) or not subquestions:
            raise FixtureFormatError("Fixture planner needs non-empty subquestions")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(subquestions[: max(1, max_subquestions)]):
            if isinstance(item, str):
                normalized.append({"question": item, "rationale": "", "depends_on": []})
                continue
            if not isinstance(item, Mapping):
                raise FixtureFormatError(
                    f"Fixture planner subquestion {index + 1} must be an object"
                )
            normalized.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "rationale": str(item.get("rationale", "")).strip(),
                    "depends_on": list(item.get("depends_on", [])),
                }
            )
        raw["subquestions"] = normalized
        return raw

    def deterministic_planner(
        self, topic: str, max_subquestions: int
    ) -> dict[str, Any]:
        """Return a durable TongAgent plan through the production planner seam."""

        # Imported lazily to keep the generic B1/B2 fixture runtime independent
        # of TongAgent's graph module.
        from research_graph import create_research_plan  # noqa: PLC0415

        payload = self.planner_payload(topic, max_subquestions)
        plan_key = hashlib.sha256(
            f"{self.task_id}\0{topic}".encode("utf-8")
        ).hexdigest()[:16]
        return cast(
            "dict[str, Any]",
            create_research_plan(
                topic,
                payload["subquestions"],
                objective=str(payload["objective"]),
                completion_criteria=payload["completion_criteria"],
                planner="fixture",
                max_subquestions=max_subquestions,
                plan_id_factory=lambda: f"fixture-plan-{plan_key}",
            ),
        )

    def _structured_payload(self, schema: dict[str, Any] | type) -> dict[str, Any]:
        schema_name = getattr(schema, "__name__", "")
        configured = self.task_metadata.get("structured_outputs", {})
        if isinstance(configured, Mapping) and schema_name in configured:
            payload = configured[schema_name]
            if not isinstance(payload, Mapping):
                raise FixtureFormatError(
                    f"structured_outputs.{schema_name} must be an object"
                )
            return deepcopy(dict(payload))

        fields = getattr(schema, "model_fields", {})
        if isinstance(fields, Mapping) and {"objective", "subquestions"}.issubset(
            fields
        ):
            return self.planner_payload()
        if isinstance(schema, Mapping):
            properties = schema.get("properties", {})
            if isinstance(properties, Mapping) and {
                "objective",
                "subquestions",
            }.issubset(properties):
                return self.planner_payload()

        generic = self.task_metadata.get("structured_output")
        if isinstance(generic, Mapping):
            return deepcopy(dict(generic))
        raise FixtureFormatError(
            f"No deterministic structured output configured for {schema_name or schema!r}"
        )

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """Return a deterministic runnable compatible with model planners."""

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise FixtureFormatError(
                f"Unsupported fixture structured-output options: {names}"
            )

        def invoke_structured(
            model_input: LanguageModelInput,
        ) -> dict[str, Any] | BaseModel:
            payload = self._structured_payload(schema)
            if inspect.isclass(schema) and issubclass(schema, BaseModel):
                parsed: dict[str, Any] | BaseModel = schema.model_validate(payload)
            else:
                parsed = payload
            if not include_raw:
                return parsed
            input_text = str(model_input)
            output_text = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, default=str
            )
            message_id, _ = self.runtime.allocate(
                task_key=self._task_key(), script_index=None
            )
            raw = AIMessage(
                id=message_id,
                content=output_text,
                usage_metadata={
                    "input_tokens": _estimated_tokens(input_text),
                    "output_tokens": _estimated_tokens(output_text),
                    "total_tokens": _estimated_tokens(input_text)
                    + _estimated_tokens(output_text),
                },
                response_metadata={"fixture": True, "structured_output": True},
            )
            return cast(
                "dict[str, Any]",
                {"raw": raw, "parsed": parsed, "parsing_error": None},
            )

        return RunnableLambda(invoke_structured)


__all__ = [
    "FIXTURE_SCHEMA_VERSION",
    "FixtureBackend",
    "FixtureChatModel",
    "FixtureFormatError",
    "FixtureModelRuntime",
    "normalize_query",
    "normalize_url",
]
