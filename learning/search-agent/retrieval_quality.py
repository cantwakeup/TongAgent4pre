"""Deterministic query rewriting and entity-aware search-result scoring."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


MIN_SEARCH_RELEVANCE_SCORE = 40
MIN_UNCERTAIN_RELEVANCE_SCORE = 15

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

# A hit consisting only of one of these terms is not a useful candidate. They
# are common first-token failure modes in the Bing RSS results observed during
# the Stage F canary.
_GENERIC_SEARCH_TERMS = frozenset(
    {
        "best",
        "blue",
        "green",
        "james",
        "latest",
        "nelson",
        "new",
        "news",
        "official",
        "red",
        "san",
        "school",
        "site",
        "top",
        "white",
    }
)

_ATOMIC_ATTRIBUTE_CANONICAL = {
    "album": "discography",
    "albums": "discography",
    "apprehended": "imprisonment",
    "birth": "birthplace",
    "birthplace": "birthplace",
    "born": "birthplace",
    "building": "building",
    "deepest": "depth",
    "depth": "depth",
    "discography": "discography",
    "height": "height",
    "hometown": "hometown",
    "imprisoned": "imprisonment",
    "imprisonment": "imprisonment",
    "incarcerated": "imprisonment",
    "jail": "imprisonment",
    "admission": "statehood",
    "admitted": "statehood",
    "prison": "imprisonment",
    "ratified": "statehood",
    "release": "release",
    "released": "release",
    "statehood": "statehood",
    "tallest": "height",
    "union": "statehood",
}
_ENUMERATION_TERMS = frozenset(
    {
        "all",
        "albums",
        "chronology",
        "count",
        "discography",
        "enumerate",
        "list",
        "members",
        "overview",
        "table",
        "timeline",
    }
)
_COMPARISON_TERMS = frozenset(
    {"compare", "comparison", "difference", "higher", "lower", "ratio", "versus", "vs"}
)
_NUMERIC_LOOKUP_TERMS = frozenset(
    {
        "date",
        "dates",
        "depth",
        "height",
        "how",
        "many",
        "meters",
        "metres",
        "feet",
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
    """Return an auditable score and three-tier relevance assessment."""

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
                "tier": "relevant",
                "reason": "result_host_matches_site_operator",
                "rejection_reason": None,
                "matched_terms": [],
                "matched_entity_terms": [],
                "matched_years": [],
                "matched_numbers": [],
                "provider_rank_adjustment": 0,
            }

    query_without_site = _SITE.sub(" ", query)
    query_terms = list(dict.fromkeys(_meaningful_terms(query_without_site)))
    haystack_term_sequence = _meaningful_terms(raw_haystack)
    haystack_terms = set(haystack_term_sequence)
    title_url_text = " ".join(str(result.get(key, "")) for key in ("title", "url"))
    title_url_terms = _meaningful_terms(title_url_text)
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
    title_url_entities = [
        entity
        for entity in entities
        if _contains_token_phrase(
            _meaningful_terms(entity),
            title_url_terms,
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

    distinctive_entity_terms = [
        term
        for term in matched_entity_terms
        if term not in _GENERIC_SEARCH_TERMS and len(term) >= 3
    ]
    distinctive_terms = [
        term
        for term in matched_terms
        if term not in _GENERIC_SEARCH_TERMS and len(term) >= 3
    ]

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

    if entity_gate:
        score = (
            50 * len(matched_quoted)
            + 35 * len(exact_entities)
            + 15 * len(matched_entity_terms)
            + 10 * len(set(matched_terms).difference(matched_entity_terms))
            + 25 * len(matched_years)
            + 20 * len(matched_numbers)
        )
        distinctive_matches = set(distinctive_entity_terms).union(distinctive_terms)
        if distinctive_matches:
            score += 10 * len(distinctive_matches)
        elif not matched_years and not matched_numbers:
            score = 0
            gate = "generic_only"
    elif entity_term_sets and distinctive_entity_terms:
        # A single distinctive entity token such as "Mandela" is worth
        # verifying, but it is never enough to become relevant by itself.
        score = 18 * len(distinctive_entity_terms)
        gate = "partial_entity_coverage"
    elif not entity_term_sets and distinctive_terms:
        score = 12 * len(distinctive_terms)
        gate = "partial_term_coverage"
    else:
        score = 0
        if matched_terms:
            gate = "generic_only"

    provider_rank = result.get("provider_rank")
    provider_rank_adjustment = 0
    if (
        isinstance(provider_rank, int)
        and not isinstance(provider_rank, bool)
        and 1 <= provider_rank <= 5
        and score > 0
    ):
        provider_rank_adjustment = (6 - provider_rank) * 2
        score += provider_rank_adjustment

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

    bounded_score = min(score, 100)
    if (
        entities
        and entity_gate
        and not title_url_entities
        and gate != "cjk_pair_coverage"
    ):
        # A snippet can mention the queried person or organization while the
        # page itself is about a film, war, concert, or unrelated biography.
        # Keep such candidates available for verification, but never allow
        # snippet-only entity coverage to clear the relevant gate.
        bounded_score = min(bounded_score, MIN_SEARCH_RELEVANCE_SCORE - 1)
        gate = "snippet_only_entity_coverage"
        entity_gate = False
    if bounded_score >= MIN_SEARCH_RELEVANCE_SCORE and (
        entity_gate or gate == "cjk_pair_coverage"
    ):
        tier = "relevant"
        reason = gate
        rejection_reason = None
    elif bounded_score >= MIN_UNCERTAIN_RELEVANCE_SCORE:
        tier = "uncertain"
        reason = gate
        rejection_reason = "requires_page_verification"
    else:
        tier = "irrelevant"
        reason = gate
        rejection_reason = gate

    return {
        "score": bounded_score,
        "gate": gate,
        "tier": tier,
        "reason": reason,
        "rejection_reason": rejection_reason,
        "matched_terms": matched_terms,
        "matched_entity_terms": matched_entity_terms,
        "title_url_entities": title_url_entities,
        "matched_years": matched_years,
        "matched_numbers": matched_numbers,
        "provider_rank_adjustment": provider_rank_adjustment,
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


def classify_query_task_type(question: str) -> str:
    """Classify one SQ for deterministic query-shape guidance."""

    tokens = {item.casefold() for item in _ASCII_TOKEN.findall(question)}
    normalized = " ".join(question.casefold().split())
    if tokens.intersection(_ENUMERATION_TERMS) or (
        "how many" in normalized
        and bool(tokens.intersection({"album", "albums", "items", "members", "works"}))
    ):
        return "list_or_enumeration"
    if tokens.intersection(_COMPARISON_TERMS) or any(
        marker in normalized for marker in ("compared with", "compared to")
    ):
        return "comparison"
    if tokens.intersection(_NUMERIC_LOOKUP_TERMS) or _NUMBER.search(question):
        return "date_or_numeric_lookup"
    return "single_fact_lookup"


def normalize_atomic_search_query(query: str, *, max_words: int = 12) -> str:
    """Reduce one generated query to an entity-plus-attribute lookup.

    The transform is deterministic and deliberately does not infer an answer.
    It keeps at most one entity phrase, a small set of canonical attributes,
    and up to two year constraints. This prevents a whole multi-hop reasoning
    description from becoming one over-constrained provider query.
    """

    normalized = " ".join(query.split())
    if not normalized:
        return normalized
    if max_words < 4:
        raise ValueError("max_words must be at least four")

    task_type = classify_query_task_type(normalized)
    site_operators = [f"site:{item}" for item in _SITE.findall(normalized)[:1]]
    without_site = _SITE.sub(" ", normalized)
    entities = _entity_phrases(without_site)
    entity = entities[0] if entities else ""
    entity_terms = set(_meaningful_terms(entity))
    raw_tokens = [item.casefold() for item in _ASCII_TOKEN.findall(without_site)]

    attributes: list[str] = []
    for token in raw_tokens:
        canonical = _ATOMIC_ATTRIBUTE_CANONICAL.get(token)
        if canonical and canonical not in attributes:
            attributes.append(canonical)
    if not attributes:
        attributes = [
            term
            for term in dict.fromkeys(_meaningful_terms(without_site))
            if term not in entity_terms and term not in _GENERIC_SEARCH_TERMS
        ][:3]
    if task_type == "list_or_enumeration":
        if "discography" in raw_tokens or {"album", "albums"}.intersection(raw_tokens):
            attributes = ["discography", "overview"]
        else:
            attributes = ["list", "overview"]

    if not entity:
        fallback_terms = [
            term
            for term in dict.fromkeys(_meaningful_terms(without_site))
            if term not in set(attributes)
            and term not in _GENERIC_SEARCH_TERMS
            and term not in _ATOMIC_ATTRIBUTE_CANONICAL
        ]
        entity = " ".join(fallback_terms[:3])

    years = list(dict.fromkeys(_YEAR.findall(without_site)))[:2]
    pieces: list[str] = list(site_operators)
    if entity:
        pieces.append(f'"{" ".join(entity.split())}"')
    pieces.extend(attributes[:3])
    pieces.extend(years)

    output: list[str] = []
    used_words = 0
    for piece in pieces:
        piece_words = max(1, len(_ASCII_TOKEN.findall(piece)))
        if output and used_words + piece_words > max_words:
            continue
        output.append(piece)
        used_words += piece_words
    compact = " ".join(output).strip()
    return compact or normalized


__all__ = [
    "MIN_SEARCH_RELEVANCE_SCORE",
    "MIN_UNCERTAIN_RELEVANCE_SCORE",
    "assess_search_relevance",
    "classify_query_task_type",
    "deterministic_query_rewrite",
    "normalize_atomic_search_query",
    "search_relevance_score",
]
