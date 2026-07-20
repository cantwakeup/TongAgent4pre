"""Deterministic query rewriting and entity-aware search-result scoring."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


MIN_SEARCH_RELEVANCE_SCORE = 40

_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9._'-]*")
_CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9.'-]*)(?:\s+(?:[A-Z][A-Za-z0-9.'-]*)){1,5}\b"
)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_QUOTED = re.compile(r'["“”]([^"“”]{2,})["“”]')
_SITE = re.compile(r"\bsite:([^\s]+)", flags=re.IGNORECASE)

# Keep this deliberately compact and deterministic. These words describe how
# to search, but do not identify what the result must be about.
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "between",
        "by",
        "date",
        "dates",
        "did",
        "do",
        "does",
        "during",
        "end",
        "find",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "latest",
        "many",
        "maximum",
        "minimum",
        "new",
        "of",
        "official",
        "on",
        "or",
        "page",
        "record",
        "released",
        "report",
        "research",
        "result",
        "results",
        "source",
        "start",
        "than",
        "that",
        "the",
        "their",
        "then",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "year",
        "years",
    }
)


def _host_matches_domain(host: str, domain: str) -> bool:
    normalized_host = host.casefold().rstrip(".").removeprefix("www.")
    normalized_domain = domain.casefold().rstrip(".").removeprefix("www.")
    return bool(
        normalized_host
        and normalized_domain
        and (
            normalized_host == normalized_domain
            or normalized_host.endswith(f".{normalized_domain}")
        )
    )


def _meaningful_terms(text: str) -> list[str]:
    return [
        token.casefold()
        for token in _ASCII_TOKEN.findall(text)
        if token.casefold() not in _ENGLISH_STOPWORDS
        and token.casefold() not in {"site", "http", "https", "www"}
    ]


def _entity_phrases(query: str) -> list[str]:
    candidates = [item.strip() for item in _QUOTED.findall(query) if item.strip()]
    candidates.extend(_CAPITALIZED_PHRASE.findall(query))
    entities: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        terms = _meaningful_terms(candidate)
        # A quoted identifier such as BIGAI is useful by itself. Unquoted
        # entities need at least two words so a sentence-initial "New" is not
        # mistaken for an entity.
        quoted = any(candidate == item.strip() for item in _QUOTED.findall(query))
        if not terms or (len(terms) < 2 and not quoted):
            continue
        normalized = " ".join(terms)
        if normalized not in seen:
            entities.append(candidate)
            seen.add(normalized)
    return entities


def _contains_token_phrase(needle: list[str], haystack: list[str]) -> bool:
    """Match one normalized phrase on token boundaries, never by substring."""

    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def assess_search_relevance(query: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable score whose relevance gate rejects one-word noise."""

    raw_haystack = " ".join(
        str(result.get(key, "")) for key in ("title", "snippet", "url")
    )
    haystack = raw_haystack.casefold()
    result_host = (urlparse(str(result.get("url", ""))).hostname or "").casefold()

    for domain in _SITE.findall(query):
        domain_host = domain.split("/", 1)[0].rstrip(".")
        if _host_matches_domain(result_host, domain_host):
            return {
                "score": 100,
                "gate": "site_match",
                "matched_terms": [],
                "matched_entity_terms": [],
                "matched_years": [],
                "matched_numbers": [],
            }

    query_without_site = _SITE.sub(" ", query)
    query_terms = list(dict.fromkeys(_meaningful_terms(query_without_site)))
    haystack_term_sequence = _meaningful_terms(raw_haystack)
    haystack_terms = set(haystack_term_sequence)
    matched_terms = [term for term in query_terms if term in haystack_terms]

    entities = _entity_phrases(query_without_site)
    entity_term_sets = [
        list(dict.fromkeys(_meaningful_terms(item))) for item in entities
    ]
    matched_entity_sets = [
        [term for term in terms if term in haystack_terms] for terms in entity_term_sets
    ]
    matched_entity_terms = list(
        dict.fromkeys(term for terms in matched_entity_sets for term in terms)
    )
    exact_entities = [
        entity
        for entity in entities
        if _contains_token_phrase(
            _meaningful_terms(entity),
            haystack_term_sequence,
        )
    ]

    quoted = [item.strip() for item in _QUOTED.findall(query) if item.strip()]
    matched_quoted = [
        item
        for item in quoted
        if _contains_token_phrase(
            _meaningful_terms(item),
            haystack_term_sequence,
        )
    ]
    query_years = set(_YEAR.findall(query))
    result_years = set(_YEAR.findall(raw_haystack))
    matched_years = sorted(query_years.intersection(result_years))
    query_numbers = set(_NUMBER.findall(query)).difference(query_years)
    result_numbers = set(_NUMBER.findall(raw_haystack)).difference(result_years)
    matched_numbers = sorted(query_numbers.intersection(result_numbers))

    if entity_term_sets:
        entity_gate = any(
            len(matched) >= min(2, len(terms))
            for terms, matched in zip(
                entity_term_sets, matched_entity_sets, strict=True
            )
        ) or bool(matched_quoted)
        gate = "entity_coverage" if entity_gate else "insufficient_entity_coverage"
    else:
        required_terms = min(2, len(query_terms))
        entity_gate = bool(query_terms) and len(matched_terms) >= required_terms
        gate = "term_coverage" if entity_gate else "insufficient_term_coverage"

    if not entity_gate:
        score = 0
    else:
        score = (
            50 * len(matched_quoted)
            + 35 * len(exact_entities)
            + 15 * len(matched_entity_terms)
            + 10 * len(set(matched_terms).difference(matched_entity_terms))
            + 25 * len(matched_years)
            + 20 * len(matched_numbers)
        )

    query_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", query))
    result_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", haystack))
    if len(query_cjk) >= 2 and result_cjk:
        query_pairs = {
            query_cjk[index : index + 2] for index in range(len(query_cjk) - 1)
        }
        result_pairs = {
            result_cjk[index : index + 2] for index in range(len(result_cjk) - 1)
        }
        if query_pairs:
            cjk_score = round(
                60 * len(query_pairs.intersection(result_pairs)) / len(query_pairs)
            )
            score = max(score, cjk_score)
            if cjk_score >= MIN_SEARCH_RELEVANCE_SCORE:
                gate = "cjk_pair_coverage"

    return {
        "score": min(score, 100),
        "gate": gate,
        "matched_terms": matched_terms,
        "matched_entity_terms": matched_entity_terms,
        "matched_years": matched_years,
        "matched_numbers": matched_numbers,
    }


def search_relevance_score(query: str, result: dict[str, Any]) -> int:
    """Return only the stable numeric portion of the relevance assessment."""

    return int(assess_search_relevance(query, result)["score"])


def deterministic_query_rewrite(query: str) -> str:
    """Quote detected entities and retain discriminative facts in stable order."""

    site_operators = [f"site:{item}" for item in _SITE.findall(query)]
    query_without_site = _SITE.sub(" ", query)
    entities = _entity_phrases(query_without_site)
    entity_terms = {term for entity in entities for term in _meaningful_terms(entity)}
    discriminative = [
        term
        for term in dict.fromkeys(_meaningful_terms(query_without_site))
        if term not in entity_terms
    ]
    numbers = list(dict.fromkeys(_NUMBER.findall(query_without_site)))

    pieces = list(site_operators)
    pieces.extend(f'"{" ".join(entity.split())}"' for entity in entities)
    pieces.extend(discriminative[:6])
    pieces.extend(number for number in numbers if number not in pieces)
    rewritten = " ".join(pieces).strip()
    return rewritten or " ".join(query.split())


__all__ = [
    "MIN_SEARCH_RELEVANCE_SCORE",
    "assess_search_relevance",
    "deterministic_query_rewrite",
    "search_relevance_score",
]
