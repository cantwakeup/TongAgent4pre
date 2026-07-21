"""Resolved, fingerprinted configuration shared by all evaluation systems."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
RecursionLimit = Annotated[int, Field(ge=2)]
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+\S+")
_OPENAI_STYLE_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_QUERY_CREDENTIAL = re.compile(
    r"(?i)[?&](?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|token|secret)=[^&#\s]+"
)
_NAMED_INLINE_CREDENTIAL = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|authorization|cookie|password|"
    r"passwd|private[_-]?key|secret|credential)\s*[:=]\s*\S+"
)
_PROVIDER_CREDENTIAL = re.compile(
    r"\b(?:"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{30,}"
    r")\b"
)
_URL_USERINFO_CREDENTIAL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@/\s]+@")


class FrozenStrictModel(BaseModel):
    """Immutable strict base for reproducible resolved configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class EvaluationModelConfig(FrozenStrictModel):
    """Model settings that must be shared across compared systems."""

    provider: NonEmptyString
    name: NonEmptyString
    temperature: float | None = None
    max_output_tokens: PositiveInt | None = None
    credential_env: list[NonEmptyString] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("credential_env")
    @classmethod
    def validate_credential_env(cls, value: list[str]) -> list[str]:
        """Persist canonical credential variable names, never their values."""
        if len(value) != len(set(value)):
            raise ValueError("credential_env names must be unique")
        for name in value:
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(
                    "credential_env names must use uppercase environment syntax"
                )
            if _is_implicit_runtime_env(name):
                raise ValueError(
                    f"credential_env cannot allow implicit runtime state: {name}"
                )
        return sorted(value)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Keep the shared output ceiling authoritative at provider invocation."""

        conflicting = {
            "max_tokens",
            "max_output_tokens",
            "max_completion_tokens",
        }.intersection(value)
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise ValueError(
                "model.parameters cannot override output-token ceilings "
                f"({names}); use model.max_output_tokens"
            )
        return value


class EvaluationJudgeConfig(FrozenStrictModel):
    """Identity and non-secret parameters for an optional external judge."""

    id: NonEmptyString
    version: NonEmptyString
    config: dict[str, JsonValue] = Field(default_factory=dict)


class SharedToolConfig(FrozenStrictModel):
    """Identity of the shared search and fetch implementations."""

    search_backend: NonEmptyString
    fetch_backend: NonEmptyString
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class BudgetLimits(FrozenStrictModel):
    """Global limits enforced uniformly for B1, B2, and B3."""

    max_search_calls: NonNegativeInt
    max_fetch_calls: NonNegativeInt
    max_total_tool_calls: NonNegativeInt
    max_model_calls: NonNegativeInt
    max_total_tokens: NonNegativeInt
    wall_time_seconds: NonNegativeFloat
    max_results_per_search: NonNegativeInt
    max_page_chars: NonNegativeInt
    recursion_limit: RecursionLimit = 125


class PermissiveWorkflowConfig(FrozenStrictModel):
    """Ablation switches for TongAgent's post-hoc verification workflow."""

    enable_posthoc_verifier: bool = True
    require_exact_quote_for_core_claims: bool = True
    allow_low_confidence_answer: bool = True
    enable_fact_gap_retrieval: bool = False
    repair_max_search_calls: NonNegativeInt = 6
    repair_max_fetch_calls: NonNegativeInt = 6
    repair_max_queries_per_slot: Annotated[int, Field(ge=1, le=2)] = 2


class ResolvedConfig(FrozenStrictModel):
    """Complete non-secret configuration for one evaluation attempt.

    The full fingerprint includes the system and its private options. The
    fairness fingerprint deliberately removes only those system-specific
    fields; it still covers the dataset, fixture/live backend, fixture
    revision, shared model/tools, seed, and every common budget.

    ``artifact_directory`` is runtime routing information needed because the
    runner receives only a task and this object. It is deliberately excluded
    from both fingerprints, so relocating or resuming an identical attempt
    does not alter its experimental identity.
    """

    schema_version: Literal[1] = 1
    system_id: NonEmptyString
    dataset_digest: NonEmptyString
    backend_kind: Literal["fixture", "live"]
    fixture_revision: str | None
    model: EvaluationModelConfig
    tools: SharedToolConfig
    budget: BudgetLimits
    runtime_mode: Literal["strict", "permissive"] = "strict"
    permissive_workflow: PermissiveWorkflowConfig = Field(
        default_factory=PermissiveWorkflowConfig
    )
    judge: EvaluationJudgeConfig | None = None
    seed: int
    system_options: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_directory: NonEmptyString
    config_fingerprint: str = ""
    fairness_fingerprint: str = ""

    @field_validator("fixture_revision")
    @classmethod
    def validate_fixture_revision(cls, value: str | None) -> str | None:
        """Reject an empty revision while retaining explicit null for live runs."""
        if value is not None and not value.strip():
            msg = "fixture_revision must be null or contain non-whitespace text"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def resolve_and_validate_fingerprints(self) -> ResolvedConfig:
        """Compute fingerprints and verify any persisted values on reload."""
        if self.backend_kind == "fixture" and self.fixture_revision is None:
            msg = "fixture backend requires fixture_revision"
            raise ValueError(msg)
        if self.backend_kind == "live" and self.fixture_revision is not None:
            msg = "live backend requires fixture_revision=null"
            raise ValueError(msg)
        if self.backend_kind == "fixture" and self.model.credential_env:
            msg = "fixture backend requires model.credential_env=[]"
            raise ValueError(msg)
        if "recursion_limit" in self.system_options:
            msg = (
                "system_options.recursion_limit is forbidden; use the shared "
                "budget.recursion_limit"
            )
            raise ValueError(msg)

        reject_persisted_config_secrets(
            self.model_dump(
                mode="json",
                exclude={"config_fingerprint", "fairness_fingerprint"},
            )
        )
        complete = _fingerprint(self._fingerprint_payload(fairness=False))
        fairness = _fingerprint(self._fingerprint_payload(fairness=True))
        # Stage F/strict artifacts predate the explicit runtime-mode fields.
        # They remain valid only for the exact old strict-default semantics;
        # all newly resolved configs include the mode and permissive switches
        # in both fingerprints.  This compatibility path never admits a
        # permissive config under an old strict fingerprint.
        legacy_complete = _fingerprint(
            self._fingerprint_payload(fairness=False, include_runtime_mode=False)
        )
        legacy_fairness = _fingerprint(
            self._fingerprint_payload(fairness=True, include_runtime_mode=False)
        )
        # Fact-Gap controls were added after the first permissive experiments.
        # Old frozen artifacts omit them, so accept their exact prior identity
        # only when the newly added controls retain the disabled defaults.
        legacy_fact_gap_complete = _fingerprint(
            self._fingerprint_payload(fairness=False, include_fact_gap=False)
        )
        legacy_fact_gap_fairness = _fingerprint(
            self._fingerprint_payload(fairness=True, include_fact_gap=False)
        )
        legacy_match = (
            self._is_legacy_strict_default()
            and self.config_fingerprint == legacy_complete
            and self.fairness_fingerprint == legacy_fairness
        )
        legacy_fact_gap_match = (
            self._is_fact_gap_default()
            and self.config_fingerprint == legacy_fact_gap_complete
            and self.fairness_fingerprint == legacy_fact_gap_fairness
        )
        if self.config_fingerprint and self.config_fingerprint != complete:
            if not legacy_match and not legacy_fact_gap_match:
                msg = "persisted config_fingerprint does not match resolved config"
                raise ValueError(msg)
            complete = legacy_complete if legacy_match else legacy_fact_gap_complete
        if self.fairness_fingerprint and self.fairness_fingerprint != fairness:
            if not legacy_match and not legacy_fact_gap_match:
                msg = "persisted fairness_fingerprint does not match resolved config"
                raise ValueError(msg)
            fairness = legacy_fairness if legacy_match else legacy_fact_gap_fairness
        object.__setattr__(self, "config_fingerprint", complete)
        object.__setattr__(self, "fairness_fingerprint", fairness)
        return self

    def _fingerprint_payload(
        self,
        *,
        fairness: bool,
        include_runtime_mode: bool = True,
        include_fact_gap: bool = True,
    ) -> dict[str, JsonValue]:
        """Return the canonical payload for one fingerprint scope."""
        excluded = {
            "artifact_directory",
            "config_fingerprint",
            "fairness_fingerprint",
        }
        if fairness:
            excluded.update({"system_id", "system_options"})
        if not include_runtime_mode:
            excluded.update({"runtime_mode", "permissive_workflow"})
        payload = self.model_dump(mode="json", exclude=excluded)
        if include_runtime_mode and not include_fact_gap:
            permissive = dict(payload.get("permissive_workflow", {}))
            for name in (
                "enable_fact_gap_retrieval",
                "repair_max_search_calls",
                "repair_max_fetch_calls",
                "repair_max_queries_per_slot",
            ):
                permissive.pop(name, None)
            payload["permissive_workflow"] = permissive
        return payload

    def _is_legacy_strict_default(self) -> bool:
        """Whether an old persisted fingerprint is semantically equivalent."""

        return (
            self.runtime_mode == "strict"
            and self.permissive_workflow == PermissiveWorkflowConfig()
        )

    def _is_fact_gap_default(self) -> bool:
        return (
            not self.permissive_workflow.enable_fact_gap_retrieval
            and self.permissive_workflow.repair_max_search_calls == 6
            and self.permissive_workflow.repair_max_fetch_calls == 6
            and self.permissive_workflow.repair_max_queries_per_slot == 2
        )

    def fairness_payload(self) -> dict[str, JsonValue]:
        """Expose the exact shared configuration covered by fairness checks."""
        return self._fingerprint_payload(fairness=True)


def canonical_json(payload: dict[str, JsonValue]) -> str:
    """Serialize a JSON object deterministically without ASCII-only rewriting."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(payload: dict[str, JsonValue]) -> str:
    """Return a versioned SHA-256 identity for canonical configuration JSON."""

    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return f"sha256:{digest}"


def _is_implicit_runtime_env(name: str) -> bool:
    """Reject endpoints, model overrides, tracing, and import paths."""

    if name in {
        "MODEL",
        "MODEL_NAME",
        "MODEL_PROVIDER",
        "PYTHONPATH",
        "WORKER_MODEL",
    }:
        return True
    if name.startswith(("LANGCHAIN_", "LANGSMITH_", "OTEL_", "TRACE_", "TRACING_")):
        return True
    return any(
        marker in name for marker in ("API_BASE", "API_VERSION", "BASE_URL", "ENDPOINT")
    ) or name.endswith(("_MODEL", "_MODEL_NAME", "_MODEL_PROVIDER"))


def reject_persisted_config_secrets(
    payload: object,
    *,
    path: str = "config",
) -> None:
    """Reject secret-shaped keys and bearer credentials in any config path.

    Validation lives on :class:`ResolvedConfig`, not only the default resolver,
    so a programmatic ``config_factory`` cannot bypass the no-secret persistence
    contract by constructing the model directly.
    """

    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            if _looks_like_secret_key(key):
                raise ValueError(
                    f"secrets are forbidden in persisted config: {path}.{key}"
                )
            reject_persisted_config_secrets(value, path=f"{path}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        for index, value in enumerate(payload):
            reject_persisted_config_secrets(value, path=f"{path}[{index}]")
        return
    if isinstance(payload, str) and _looks_like_secret_value(payload):
        raise ValueError(f"credential values are forbidden in persisted config: {path}")


def _looks_like_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]", "", key.upper())
    return normalized in {
        "TOKEN",
        "IDTOKEN",
        "REFRESHTOKEN",
        "PRIVATEKEY",
    } or any(
        marker in normalized
        for marker in (
            "APIKEY",
            "ACCESSTOKEN",
            "AUTHTOKEN",
            "REFRESHTOKEN",
            "IDTOKEN",
            "AUTHORIZATION",
            "PASSWORD",
            "PRIVATEKEY",
            "SECRET",
            "COOKIE",
        )
    )


def _looks_like_secret_value(value: str) -> bool:
    """Recognize common inline credentials before config fingerprinting."""

    return any(
        pattern.search(value)
        for pattern in (
            _BEARER_CREDENTIAL,
            _OPENAI_STYLE_CREDENTIAL,
            _QUERY_CREDENTIAL,
            _NAMED_INLINE_CREDENTIAL,
            _PROVIDER_CREDENTIAL,
            _URL_USERINFO_CREDENTIAL,
        )
    )
