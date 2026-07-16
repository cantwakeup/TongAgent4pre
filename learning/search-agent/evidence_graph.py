"""Deterministic claim-to-excerpt provenance for TongAgent research runs."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterable

from research_state import ClaimRecord, ConflictRecord, EvidenceStance, EvidenceUnit


EVIDENCE_GRAPH_VERSION = 1
MIN_CLAIM_CHARS = 12
MAX_CLAIM_CHARS = 500
MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 800
CLAIM_ID_PATTERN = re.compile(r"^C[1-9][0-9]*$")
EVIDENCE_ID_PATTERN = re.compile(r"^E[1-9][0-9]*$")
CONFLICT_ID_PATTERN = re.compile(r"^X[1-9][0-9]*$")
SOURCE_ID_PATTERN = re.compile(r"^S[1-9][0-9]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REPORT_CLAIM_PATTERN = re.compile(r"\[(C[1-9][0-9]*)\]")
REPORT_SOURCE_PATTERN = re.compile(r"\[(S[1-9][0-9]*)\]")
CANONICAL_SOURCE_LINE_PATTERN = re.compile(
    r"^- \[(S[1-9][0-9]*)\] .* — https?://\S+$", re.IGNORECASE
)
SOURCES_HEADING_PATTERN = re.compile(r"^#{1,6}\s+Sources\s*$", re.IGNORECASE)
MAPPED_SECTIONS = {"short answer", "key findings"}
CONFLICT_SECTION = "conflicts and caveats"
REQUIRED_REPORT_SECTIONS = (
    "short answer",
    "key findings",
    CONFLICT_SECTION,
    "sources",
)


def normalize_evidence_text(value: str) -> str:
    """Collapse whitespace so copied excerpts survive HTML line boundaries."""
    return " ".join(value.split())


def text_sha256(value: str) -> str:
    """Return a stable digest of normalized visible text."""
    return sha256(normalize_evidence_text(value).encode()).hexdigest()


def _next_sequence(records: list[dict[str, Any]], key: str, pattern: re.Pattern) -> int:
    """Allocate after the greatest valid ID so restored catalog gaps stay safe."""
    sequences = [
        int(match.group(0)[1:])
        for item in records
        if (match := pattern.fullmatch(str(item.get(key, ""))))
    ]
    return max(sequences, default=0) + 1


def _source_revisions(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize new and legacy source metadata into immutable revisions."""
    revisions = [dict(item) for item in source.get("content_revisions", [])]
    known_hashes = {str(item.get("content_sha256", "")) for item in revisions if item}
    fallbacks = (
        (
            str(source.get("content_sha256", "")),
            {
                "title": source.get("title", ""),
                "content_chars": source.get("content_chars", 0),
                "evidence_quality": source.get("evidence_quality", "full"),
                "quality_reason": source.get("quality_reason", ""),
            },
        ),
        (
            str(source.get("latest_content_sha256", "")),
            {
                "title": source.get("latest_title", source.get("title", "")),
                "content_chars": source.get(
                    "latest_content_chars", source.get("content_chars", 0)
                ),
                "evidence_quality": source.get(
                    "latest_evidence_quality",
                    source.get("evidence_quality", "full"),
                ),
                "quality_reason": source.get(
                    "latest_quality_reason", source.get("quality_reason", "")
                ),
            },
        ),
    )
    for content_hash, metadata in fallbacks:
        if content_hash and content_hash not in known_hashes:
            revisions.append({"content_sha256": content_hash, **metadata})
            known_hashes.add(content_hash)
    return revisions


def independent_evidence_source_ids(
    *,
    source_ids: Iterable[str],
    sources: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
    claim_ids: set[str] | None = None,
) -> list[str]:
    """Collapse sources whose cited evidence uses only already-seen revisions.

    Source-level duplicate flags describe each URL's latest fetch and may change
    after content drift. Policy gates instead need the immutable revision hashes
    attached to the plan's evidence edges.
    """
    candidates = set(source_ids)
    hashes_by_source: dict[str, set[str]] = {}
    for unit in evidence_units:
        source_id = str(unit.get("source_id", ""))
        if source_id not in candidates:
            continue
        if claim_ids is not None and str(unit.get("claim_id", "")) not in claim_ids:
            continue
        content_hash = str(unit.get("source_content_sha256", ""))
        if SHA256_PATTERN.fullmatch(content_hash):
            hashes_by_source.setdefault(source_id, set()).add(content_hash)

    independent: list[str] = []
    seen_hashes: set[str] = set()
    for source in sources:
        source_id = str(source.get("source_id", ""))
        if source_id not in candidates:
            continue
        revision_hashes = hashes_by_source.get(source_id, set())
        if revision_hashes - seen_hashes:
            independent.append(source_id)
        seen_hashes.update(revision_hashes)
    return independent


@dataclass
class EvidenceGraphStore:
    """Process-local page cache plus checkpoint-safe claim provenance."""

    claims: list[ClaimRecord] = field(default_factory=list)
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    _page_contents: dict[str, str] = field(default_factory=dict, repr=False)
    _page_content_hashes: dict[str, str] = field(default_factory=dict, repr=False)
    _page_metadata: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def cache_page(
        self,
        source_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Cache normalized content for exact-quote validation in this process."""
        self._page_contents[source_id] = normalize_evidence_text(content)
        self._page_content_hashes[source_id] = text_sha256(content)
        self._page_metadata[source_id] = deepcopy(metadata or {})

    def record(
        self,
        *,
        source: dict[str, Any],
        subquestion_id: str | None,
        claim: str,
        quote: str,
        stance: EvidenceStance,
        claim_id: str = "",
    ) -> dict[str, Any]:
        """Create one validated claim-excerpt edge and any resulting conflict."""
        if subquestion_id is None:
            msg = "Evidence can only be recorded for an active subquestion"
            raise ValueError(msg)
        normalized_claim = normalize_evidence_text(claim)
        if not MIN_CLAIM_CHARS <= len(normalized_claim) <= MAX_CLAIM_CHARS:
            msg = (
                f"Claims must contain {MIN_CLAIM_CHARS}-{MAX_CLAIM_CHARS} "
                "normalized characters"
            )
            raise ValueError(msg)
        normalized_quote = normalize_evidence_text(quote)
        if not MIN_QUOTE_CHARS <= len(normalized_quote) <= MAX_QUOTE_CHARS:
            msg = (
                f"Evidence quotes must contain {MIN_QUOTE_CHARS}-{MAX_QUOTE_CHARS} "
                "normalized characters"
            )
            raise ValueError(msg)
        source_id = str(source.get("source_id", ""))
        page = self._page_contents.get(source_id)
        if page is None:
            msg = (
                f"Source {source_id} has no page body in this process; refetch its "
                "canonical URL before recording evidence"
            )
            raise ValueError(msg)
        if normalized_quote not in page:
            msg = f"Evidence quote is not an exact excerpt of canonical source {source_id}"
            raise ValueError(msg)
        if stance not in {"supports", "contradicts"}:
            msg = "Evidence stance must be supports or contradicts"
            raise ValueError(msg)

        target = self._resolve_claim(
            subquestion_id=subquestion_id,
            claim=normalized_claim,
            claim_id=claim_id,
        )
        opposite = next(
            (
                item
                for item in self.evidence_units
                if item["claim_id"] == target["claim_id"]
                and item["source_id"] == source_id
                and item["quote"] == normalized_quote
                and item["stance"] != stance
            ),
            None,
        )
        if opposite is not None:
            msg = (
                "The same canonical source excerpt cannot both support and "
                f"contradict claim {target['claim_id']}"
            )
            raise ValueError(msg)
        existing = next(
            (
                item
                for item in self.evidence_units
                if item["claim_id"] == target["claim_id"]
                and item["source_id"] == source_id
                and item["stance"] == stance
                and item["quote"] == normalized_quote
            ),
            None,
        )
        if existing is None:
            evidence_id = f"E{_next_sequence(self.evidence_units, 'evidence_id', EVIDENCE_ID_PATTERN)}"
            revision = self._page_metadata.get(source_id, {})
            existing = {
                "evidence_id": evidence_id,
                "claim_id": target["claim_id"],
                "subquestion_id": subquestion_id,
                "source_id": source_id,
                "stance": stance,
                "quote": normalized_quote,
                "quote_sha256": text_sha256(normalized_quote),
                "source_content_sha256": self._page_content_hashes[source_id],
                "url": str(source.get("url", "")),
                "title": str(revision.get("title", source.get("title", ""))),
                "evidence_quality": str(
                    revision.get(
                        "evidence_quality", source.get("evidence_quality", "full")
                    )
                ),
            }
            self.evidence_units.append(existing)
        self._refresh_claim(target)
        conflict = self._refresh_conflict(target)
        return {
            "claim": deepcopy(target),
            "evidence": deepcopy(existing),
            "conflict": deepcopy(conflict) if conflict is not None else None,
        }

    def _resolve_claim(
        self, *, subquestion_id: str, claim: str, claim_id: str
    ) -> ClaimRecord:
        if claim_id:
            if not CLAIM_ID_PATTERN.fullmatch(claim_id):
                msg = "Claim IDs must use the C# format"
                raise ValueError(msg)
            target = next(
                (item for item in self.claims if item["claim_id"] == claim_id), None
            )
            if target is None:
                msg = (
                    f"Unknown canonical claim ID: {claim_id}. Omit claim_id when "
                    "creating a new claim; only reuse an ID returned by this tool"
                )
                raise ValueError(msg)
            if target["subquestion_id"] != subquestion_id:
                msg = f"Claim {claim_id} belongs to another subquestion"
                raise ValueError(msg)
            if normalize_evidence_text(target["text"]).casefold() != claim.casefold():
                msg = f"Claim text does not match canonical claim {claim_id}"
                raise ValueError(msg)
            return target
        target = next(
            (
                item
                for item in self.claims
                if item["subquestion_id"] == subquestion_id
                and normalize_evidence_text(item["text"]).casefold() == claim.casefold()
            ),
            None,
        )
        if target is not None:
            return target
        target = {
            "claim_id": f"C{_next_sequence(self.claims, 'claim_id', CLAIM_ID_PATTERN)}",
            "subquestion_id": subquestion_id,
            "text": claim,
            "status": "contradicted",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "source_ids": [],
        }
        self.claims.append(target)
        return target

    def _refresh_claim(self, claim: ClaimRecord) -> None:
        related = [
            item
            for item in self.evidence_units
            if item["claim_id"] == claim["claim_id"]
        ]
        supporting = [
            item["evidence_id"] for item in related if item["stance"] == "supports"
        ]
        contradicting = [
            item["evidence_id"] for item in related if item["stance"] == "contradicts"
        ]
        claim["supporting_evidence_ids"] = supporting
        claim["contradicting_evidence_ids"] = contradicting
        claim["source_ids"] = list(dict.fromkeys(item["source_id"] for item in related))
        claim["status"] = (
            "contested"
            if supporting and contradicting
            else "supported"
            if supporting
            else "contradicted"
        )

    def _refresh_conflict(self, claim: ClaimRecord) -> ConflictRecord | None:
        if claim["status"] != "contested":
            return None
        conflict = next(
            (item for item in self.conflicts if item["claim_id"] == claim["claim_id"]),
            None,
        )
        if conflict is None:
            conflict = {
                "conflict_id": f"X{_next_sequence(self.conflicts, 'conflict_id', CONFLICT_ID_PATTERN)}",
                "claim_id": claim["claim_id"],
                "subquestion_id": claim["subquestion_id"],
                "status": "unresolved",
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "source_ids": [],
            }
            self.conflicts.append(conflict)
        conflict["supporting_evidence_ids"] = list(claim["supporting_evidence_ids"])
        conflict["contradicting_evidence_ids"] = list(
            claim["contradicting_evidence_ids"]
        )
        conflict["source_ids"] = list(claim["source_ids"])
        return conflict

    def snapshot(self) -> dict[str, Any]:
        """Return the checkpoint-safe graph without downloaded page bodies."""
        return {
            "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
            "claims": deepcopy(self.claims),
            "evidence_units": deepcopy(self.evidence_units),
            "conflicts": deepcopy(self.conflicts),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore graph records while requiring refetch before new excerpts."""
        self.claims = deepcopy(snapshot.get("claims", []))
        self.evidence_units = deepcopy(snapshot.get("evidence_units", []))
        self.conflicts = deepcopy(snapshot.get("conflicts", []))
        self._page_contents = {}
        self._page_content_hashes = {}
        self._page_metadata = {}

    def reset(self) -> None:
        """Start a new plan-local graph while source IDs remain thread-stable."""
        self.claims = []
        self.evidence_units = []
        self.conflicts = []
        self._page_contents = {}
        self._page_content_hashes = {}
        self._page_metadata = {}


def validate_evidence_graph(snapshot: dict[str, Any]) -> list[str]:
    """Return deterministic graph integrity errors without model judgment."""
    errors: list[str] = []
    sources = snapshot.get("successful_sources", [])
    claims = snapshot.get("claims", [])
    evidence_units = snapshot.get("evidence_units", [])
    conflicts = snapshot.get("conflicts", [])
    sources_by_id = {str(item.get("source_id", "")): item for item in sources}
    claims_by_id = {str(item.get("claim_id", "")): item for item in claims}
    evidence_by_id = {str(item.get("evidence_id", "")): item for item in evidence_units}
    conflicts_by_id = {str(item.get("conflict_id", "")): item for item in conflicts}
    if len(sources_by_id) != len(sources):
        errors.append("duplicate source IDs")
    if len(claims_by_id) != len(claims):
        errors.append("duplicate claim IDs")
    if len(evidence_by_id) != len(evidence_units):
        errors.append("duplicate evidence IDs")
    if len(conflicts_by_id) != len(conflicts):
        errors.append("duplicate conflict IDs")
    duplicate_edges: set[tuple[str, str, str, str]] = set()
    edge_stances: dict[tuple[str, str, str], set[str]] = {}

    for source_id, source in sources_by_id.items():
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            errors.append(f"invalid source ID {source_id}")
        revisions = _source_revisions(source)
        revision_hashes = [str(item.get("content_sha256", "")) for item in revisions]
        if len(revision_hashes) != len(set(revision_hashes)):
            errors.append(f"source {source_id} has duplicate content revisions")
        for content_hash in revision_hashes:
            if not SHA256_PATTERN.fullmatch(content_hash):
                errors.append(f"source {source_id} has an invalid revision hash")
        for key in ("content_sha256", "latest_content_sha256"):
            content_hash = str(source.get(key, ""))
            if content_hash and content_hash not in revision_hashes:
                errors.append(f"source {source_id} has an untracked {key}")

    for claim_id, claim in claims_by_id.items():
        if not CLAIM_ID_PATTERN.fullmatch(claim_id):
            errors.append(f"invalid claim ID {claim_id}")
        claim_text = normalize_evidence_text(str(claim.get("text", "")))
        if (
            claim_text != claim.get("text")
            or not MIN_CLAIM_CHARS <= len(claim_text) <= MAX_CLAIM_CHARS
        ):
            errors.append(f"claim {claim_id} has invalid canonical text")
        related = [
            item for item in evidence_units if str(item.get("claim_id", "")) == claim_id
        ]
        if not related:
            errors.append(f"claim {claim_id} has no evidence edges")
        supporting = [
            str(item.get("evidence_id", ""))
            for item in related
            if item.get("stance") == "supports"
        ]
        contradicting = [
            str(item.get("evidence_id", ""))
            for item in related
            if item.get("stance") == "contradicts"
        ]
        source_ids = list(
            dict.fromkeys(str(item.get("source_id", "")) for item in related)
        )
        expected_status = (
            "contested"
            if supporting and contradicting
            else "supported"
            if supporting
            else "contradicted"
        )
        if claim.get("supporting_evidence_ids") != supporting:
            errors.append(f"claim {claim_id} has inconsistent supporting edges")
        if claim.get("contradicting_evidence_ids") != contradicting:
            errors.append(f"claim {claim_id} has inconsistent contradicting edges")
        if claim.get("source_ids") != source_ids:
            errors.append(f"claim {claim_id} has inconsistent source IDs")
        if claim.get("status") != expected_status:
            errors.append(f"claim {claim_id} has inconsistent status")

    for evidence_id, evidence in evidence_by_id.items():
        if not EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
            errors.append(f"invalid evidence ID {evidence_id}")
        claim_id = str(evidence.get("claim_id", ""))
        source_id = str(evidence.get("source_id", ""))
        stance = str(evidence.get("stance", ""))
        claim = claims_by_id.get(claim_id)
        source = sources_by_id.get(source_id)
        if stance not in {"supports", "contradicts"}:
            errors.append(f"evidence {evidence_id} has an invalid stance")
        if claim is None:
            errors.append(f"evidence {evidence_id} references unknown claim {claim_id}")
        elif evidence.get("subquestion_id") != claim.get("subquestion_id"):
            errors.append(f"evidence {evidence_id} crosses subquestions")
        if source is None:
            errors.append(
                f"evidence {evidence_id} references unknown source {source_id}"
            )
        else:
            if evidence.get("url") != source.get("url"):
                errors.append(
                    f"evidence {evidence_id} mismatches source {source_id} url"
                )
            revision_hash = str(evidence.get("source_content_sha256", ""))
            revision = next(
                (
                    item
                    for item in _source_revisions(source)
                    if str(item.get("content_sha256", "")) == revision_hash
                ),
                None,
            )
            if revision is None:
                errors.append(
                    f"evidence {evidence_id} references an unknown source revision"
                )
            else:
                for key in ("title", "evidence_quality"):
                    if evidence.get(key) != revision.get(key):
                        errors.append(
                            f"evidence {evidence_id} mismatches source {source_id} "
                            f"revision {key}"
                        )
        quote = str(evidence.get("quote", ""))
        normalized_quote = normalize_evidence_text(quote)
        if (
            quote != normalized_quote
            or not MIN_QUOTE_CHARS <= len(quote) <= MAX_QUOTE_CHARS
        ):
            errors.append(f"evidence {evidence_id} has invalid canonical quote text")
        if evidence.get("quote_sha256") != text_sha256(quote):
            errors.append(f"evidence {evidence_id} has an invalid quote hash")
        edge_key = (
            claim_id,
            source_id,
            str(evidence.get("stance", "")),
            str(evidence.get("quote", "")),
        )
        if edge_key in duplicate_edges:
            errors.append(f"evidence {evidence_id} duplicates an existing edge")
        duplicate_edges.add(edge_key)
        stance_key = (claim_id, source_id, quote)
        edge_stances.setdefault(stance_key, set()).add(stance)

    for (claim_id, source_id, _quote), stances in edge_stances.items():
        if {"supports", "contradicts"}.issubset(stances):
            errors.append(
                f"claim {claim_id} assigns opposing stances to one {source_id} excerpt"
            )

    conflicts_by_claim: dict[str, list[dict[str, Any]]] = {}
    for conflict_id, conflict in conflicts_by_id.items():
        if not CONFLICT_ID_PATTERN.fullmatch(conflict_id):
            errors.append(f"invalid conflict ID {conflict_id}")
        claim_id = str(conflict.get("claim_id", ""))
        claim = claims_by_id.get(claim_id)
        conflicts_by_claim.setdefault(claim_id, []).append(conflict)
        if claim is None:
            errors.append(f"conflict {conflict_id} references unknown claim {claim_id}")
            continue
        if conflict.get("subquestion_id") != claim.get("subquestion_id"):
            errors.append(f"conflict {conflict_id} crosses subquestions")
        if claim.get("status") != "contested":
            errors.append(f"conflict {conflict_id} references a non-contested claim")
        for key in (
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "source_ids",
        ):
            if conflict.get(key) != claim.get(key):
                errors.append(f"conflict {conflict_id} has inconsistent {key}")
        if conflict.get("status") != "unresolved":
            errors.append(f"conflict {conflict_id} has an invalid status")
    for claim_id, claim in claims_by_id.items():
        related_conflicts = conflicts_by_claim.get(claim_id, [])
        if claim.get("status") == "contested" and len(related_conflicts) != 1:
            errors.append(f"contested claim {claim_id} must have one conflict")
        if claim.get("status") != "contested" and related_conflicts:
            errors.append(f"non-contested claim {claim_id} must not have conflicts")
    return errors


def report_claim_mapping_errors(
    report: str,
    *,
    plan_claim_ids: set[str],
    claims: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Validate constrained report lines against canonical claim-source edges."""
    lines = report.splitlines()
    claims_by_id = {str(item.get("claim_id", "")): item for item in claims}
    report_claim_ids: set[str] = set()
    source_without_claim: list[str] = []
    claim_without_source: list[str] = []
    mismatched_pairs: list[str] = []
    unmapped_findings: list[str] = []
    mismatched_claim_text: list[str] = []
    invalid_claim_status: list[str] = []
    misplaced_contested: list[str] = []
    misplaced_supported: list[str] = []
    incomplete_conflicts: list[str] = []
    invalid_section_structure: list[str] = []
    invalid_section_lines: list[str] = []
    multiple_claim_lines: list[str] = []
    invalid_sources_section_lines: list[str] = []
    current_section = ""
    section_order: list[str] = []

    def canonical_line_text(value: str) -> str:
        without_citations = REPORT_CLAIM_PATTERN.sub("", value)
        without_citations = REPORT_SOURCE_PATTERN.sub("", without_citations)
        without_bullet = re.sub(
            r"^\s*(?:(?:[-*+])|(?:[0-9]+[.)]))\s+", "", without_citations
        )
        return normalize_evidence_text(without_bullet)

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        h2_heading = re.fullmatch(r"##\s+(.+?)\s*", stripped)
        any_heading = re.fullmatch(r"(#{1,6})\s+(.+?)\s*", stripped)
        if h2_heading:
            current_section = h2_heading.group(1).casefold()
            section_order.append(current_section)
            if current_section not in REQUIRED_REPORT_SECTIONS:
                invalid_section_structure.append(f"line {line_number}:unexpected")
            continue
        if any_heading:
            invalid_section_structure.append(f"line {line_number}:unexpected")
            continue
        if not stripped:
            continue
        claim_ids = set(REPORT_CLAIM_PATTERN.findall(line))
        source_ids = set(REPORT_SOURCE_PATTERN.findall(line))
        if current_section == "sources":
            canonical_source = CANONICAL_SOURCE_LINE_PATTERN.fullmatch(stripped)
            if (
                canonical_source is None
                or len(source_ids) != 1
                or canonical_source.group(1) not in source_ids
                or claim_ids
            ):
                invalid_sources_section_lines.append(str(line_number))
            continue
        report_claim_ids.update(claim_ids)
        if current_section not in REQUIRED_REPORT_SECTIONS:
            invalid_section_lines.append(str(line_number))
            continue
        if current_section in MAPPED_SECTIONS and (not claim_ids or not source_ids):
            unmapped_findings.append(str(line_number))
        if source_ids and not claim_ids:
            source_without_claim.append(str(line_number))
            continue
        if claim_ids and not source_ids:
            claim_without_source.append(str(line_number))
            continue
        if not claim_ids and not source_ids:
            continue
        if len(claim_ids) != 1:
            multiple_claim_lines.append(str(line_number))
            continue
        known_claim_ids = claim_ids.intersection(claims_by_id)
        allowed_by_claim: dict[str, set[str]] = {}
        for canonical_claim_id in known_claim_ids:
            canonical_claim = claims_by_id[canonical_claim_id]
            if canonical_line_text(line) != normalize_evidence_text(
                str(canonical_claim.get("text", ""))
            ):
                mismatched_claim_text.append(f"line {line_number}:{canonical_claim_id}")
            claim_status = canonical_claim.get("status")
            if claim_status == "contradicted":
                invalid_claim_status.append(canonical_claim_id)
            if claim_status == "contested" and current_section != CONFLICT_SECTION:
                misplaced_contested.append(f"line {line_number}:{canonical_claim_id}")
            if claim_status == "supported" and current_section not in MAPPED_SECTIONS:
                misplaced_supported.append(f"line {line_number}:{canonical_claim_id}")
            allowed_by_claim[canonical_claim_id] = {
                str(item.get("source_id", ""))
                for item in evidence_units
                if item.get("claim_id") == canonical_claim_id
                and item.get("stance") in {"supports", "contradicts"}
            }
            if not source_ids or not source_ids.issubset(
                allowed_by_claim[canonical_claim_id]
            ):
                mismatched_pairs.append(f"line {line_number}:{canonical_claim_id}")
            if claim_status == "contested":
                supporting_sources = {
                    str(item.get("source_id", ""))
                    for item in evidence_units
                    if item.get("claim_id") == canonical_claim_id
                    and item.get("stance") == "supports"
                }
                contradicting_sources = {
                    str(item.get("source_id", ""))
                    for item in evidence_units
                    if item.get("claim_id") == canonical_claim_id
                    and item.get("stance") == "contradicts"
                }
                if not source_ids.intersection(
                    supporting_sources
                ) or not source_ids.intersection(contradicting_sources):
                    incomplete_conflicts.append(
                        f"line {line_number}:{canonical_claim_id}"
                    )
    if tuple(section_order) != REQUIRED_REPORT_SECTIONS:
        invalid_section_structure.append(
            "expected:" + ">".join(REQUIRED_REPORT_SECTIONS)
        )

    unknown = sorted(report_claim_ids - set(claims_by_id))
    non_plan = sorted(report_claim_ids - plan_claim_ids)
    missing = sorted(plan_claim_ids - report_claim_ids)

    return {
        "unknown_claim_ids": unknown,
        "non_plan_claim_ids": non_plan,
        "missing_plan_claim_ids": missing,
        "source_without_claim_lines": source_without_claim,
        "claim_without_source_lines": claim_without_source,
        "mismatched_claim_source_pairs": list(dict.fromkeys(mismatched_pairs)),
        "unmapped_finding_lines": unmapped_findings,
        "mismatched_claim_text_lines": list(dict.fromkeys(mismatched_claim_text)),
        "invalid_claim_status_ids": sorted(set(invalid_claim_status)),
        "misplaced_contested_claims": list(dict.fromkeys(misplaced_contested)),
        "misplaced_supported_claims": list(dict.fromkeys(misplaced_supported)),
        "incomplete_conflict_lines": list(dict.fromkeys(incomplete_conflicts)),
        "invalid_section_structure": list(dict.fromkeys(invalid_section_structure)),
        "invalid_section_lines": invalid_section_lines,
        "multiple_claim_lines": multiple_claim_lines,
        "invalid_sources_section_lines": invalid_sources_section_lines,
    }
