"""Resource policies for TongAgent's selectable research intensity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal


EffortName = Literal["low", "medium", "high", "xhigh"]
ModeName = Literal["single", "multi", "auto"]
TopologyName = Literal["single", "multi"]
POLICY_FINGERPRINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EffortPolicy:
    """Concrete tool and quality budgets for one application-level effort tier."""

    name: EffortName
    max_searches: int
    max_fetches: int
    min_successful_sources: int
    max_results_per_search: int
    max_chars_per_page: int
    max_output_tokens: int
    max_subquestions: int
    require_reviewer: bool


EFFORT_POLICIES: dict[EffortName, EffortPolicy] = {
    "low": EffortPolicy(
        name="low",
        max_searches=2,
        max_fetches=3,
        min_successful_sources=2,
        max_results_per_search=4,
        max_chars_per_page=8_000,
        max_output_tokens=3_000,
        max_subquestions=1,
        require_reviewer=False,
    ),
    "medium": EffortPolicy(
        name="medium",
        max_searches=4,
        max_fetches=6,
        min_successful_sources=2,
        max_results_per_search=5,
        max_chars_per_page=12_000,
        max_output_tokens=5_000,
        max_subquestions=2,
        require_reviewer=False,
    ),
    "high": EffortPolicy(
        name="high",
        max_searches=8,
        max_fetches=9,
        min_successful_sources=3,
        max_results_per_search=6,
        max_chars_per_page=15_000,
        max_output_tokens=8_000,
        max_subquestions=4,
        require_reviewer=True,
    ),
    "xhigh": EffortPolicy(
        name="xhigh",
        max_searches=12,
        max_fetches=14,
        min_successful_sources=4,
        max_results_per_search=8,
        max_chars_per_page=20_000,
        max_output_tokens=12_000,
        max_subquestions=5,
        require_reviewer=True,
    ),
}


_COMPLEXITY_MARKERS = (
    "比较",
    "对比",
    "评估",
    "争议",
    "原因",
    "趋势",
    "方案",
    "综述",
    "compare",
    "versus",
    "evaluate",
    "trade-off",
    "review",
)


def resolve_topology(mode: ModeName, effort: EffortName, topic: str) -> TopologyName:
    """Resolve `auto` into a deterministic initial single- or multi-agent topology."""
    if mode != "auto":
        return mode
    if effort in {"high", "xhigh"}:
        return "multi"
    normalized = topic.lower()
    marker_hits = sum(marker in normalized for marker in _COMPLEXITY_MARKERS)
    if marker_hits >= 1 or len(topic) >= 80:
        return "multi"
    return "single"


def policy_fingerprint(
    *,
    strategy: str,
    effort: EffortName,
    requested_mode: ModeName,
    resolved_topology: TopologyName,
    model_name: str,
    worker_model_name: str,
    max_escalations: int,
) -> str:
    """Hash every pinned runtime choice that affects pending-plan semantics."""
    payload = {
        "schema_version": POLICY_FINGERPRINT_SCHEMA_VERSION,
        "strategy": strategy,
        "effort": effort,
        "requested_mode": requested_mode,
        "resolved_topology": resolved_topology,
        "model_name": model_name,
        "worker_model_name": worker_model_name,
        "max_escalations": max(0, max_escalations),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def policy_prompt(policy: EffortPolicy, topology: TopologyName) -> str:
    """Describe active hard budgets and quality gates to the model."""
    review_rule = (
        "Before finalizing, delegate a report review to the reviewer subagent and revise material issues."
        if topology == "multi" and policy.require_reviewer
        else "Do not spend calls on a separate review unless it is clearly necessary."
    )
    return f"""Active TongAgent policy:
- effort: {policy.name}
- topology: {topology}
- at most {policy.max_searches} searches and {policy.max_fetches} page fetches across all agents, partitioned into reserved per-subquestion slices
- at most {policy.max_subquestions} explicit research subquestions
- at least {policy.min_successful_sources} successfully fetched, relevant sources
- cite factual claims with source IDs such as [S1]
- list every cited source as `[S1] Page title — full URL` in the Sources section
- {review_rule}

Tool budgets are enforced in code. Use get_source_ledger to inspect the active SQ slice and canonical source mapping. If a budget is exhausted, finish with the best supported answer and disclose the limitation."""
