"""Earlier-announcement cross-reference extraction and matching."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Sequence

from agent_memory.loader import Announcement


_REFERENCE_PATTERN = re.compile(
    r"\brefer to ASX announcement\s*[\"“]"
    r"(?P<title>[^\"”]+?)"
    r"[\"”]\s*(?:on|dated)\s*"
    r"(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
REFERENCE_DATE_WINDOW_DAYS = 3
FUZZY_MATCH_THRESHOLD = 0.90
AMBIGUITY_MARGIN = 0.02


@dataclass(frozen=True)
class CrossReference:
    cited_title: str
    cited_date: date


@dataclass(frozen=True)
class CrossReferenceMatch:
    cited_title: str
    cited_date: date
    target_doc_id: str
    match_score: float


@dataclass(frozen=True)
class CrossReferenceResolution:
    matches: tuple[CrossReferenceMatch, ...]
    unresolved: tuple[CrossReference, ...]


def _parse_cited_date(value: str) -> date:
    for format_string in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value.strip(), format_string).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported cross-reference date: {value!r}")


def extract_references(text: str) -> list[CrossReference]:
    references = []
    for match in _REFERENCE_PATTERN.finditer(text):
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        references.append(
            CrossReference(
                cited_title=title,
                cited_date=_parse_cited_date(match.group("date")),
            )
        )
    return references


def normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(_TITLE_TOKEN_PATTERN.findall(normalized))


def _title_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    sequence_score = SequenceMatcher(None, left, right).ratio()
    left_tokens = normalize_title(left).split()
    right_tokens = normalize_title(right).split()
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    minimum_size = min(len(left_set), len(right_set))
    if minimum_size < 5 or abs(len(left_set) - len(right_set)) > 4:
        return sequence_score
    overlap = len(left_set & right_set) / minimum_size
    if overlap >= FUZZY_MATCH_THRESHOLD:
        return max(sequence_score, overlap)
    return sequence_score


def _resolve_reference(
    reference: CrossReference,
    referring_document: Announcement,
    candidates: Sequence[Announcement],
) -> CrossReferenceMatch | None:
    normalized_title = normalize_title(reference.cited_title)
    if not normalized_title:
        return None

    eligible = [
        candidate
        for candidate in candidates
        if candidate.id != referring_document.id
        and candidate.published_at < referring_document.published_at
        and abs((candidate.published_at.date() - reference.cited_date).days)
        <= REFERENCE_DATE_WINDOW_DAYS
    ]
    scored = [
        (
            _title_similarity(normalized_title, normalize_title(candidate.headline)),
            candidate,
        )
        for candidate in eligible
    ]
    scored.sort(
        key=lambda item: (
            item[0],
            -abs((item[1].published_at.date() - reference.cited_date).days),
            item[1].id,
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < FUZZY_MATCH_THRESHOLD:
        return None
    if len(scored) > 1:
        same_score = scored[0][0] - scored[1][0] < AMBIGUITY_MARGIN
        same_date_distance = abs(
            (scored[0][1].published_at.date() - reference.cited_date).days
        ) == abs((scored[1][1].published_at.date() - reference.cited_date).days)
        if same_score and same_date_distance:
            return None
    return CrossReferenceMatch(
        cited_title=reference.cited_title,
        cited_date=reference.cited_date,
        target_doc_id=scored[0][1].id,
        match_score=scored[0][0],
    )


def resolve_references(
    referring_document: Announcement,
    text: str,
    candidates: Sequence[Announcement],
) -> CrossReferenceResolution:
    matches: list[CrossReferenceMatch] = []
    unresolved: list[CrossReference] = []
    for reference in extract_references(text):
        match = _resolve_reference(reference, referring_document, candidates)
        if match is None:
            unresolved.append(reference)
        else:
            matches.append(match)
    return CrossReferenceResolution(tuple(matches), tuple(unresolved))


def persist_cross_references(
    connection: sqlite3.Connection,
    source_doc_id: str,
    matches: Sequence[CrossReferenceMatch],
) -> None:
    connection.executemany(
        """
        INSERT INTO doc_xref (
            src_doc, dst_doc, cited_title, cited_date, match_score
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(src_doc, dst_doc, cited_title) DO UPDATE SET
            cited_date = excluded.cited_date,
            match_score = excluded.match_score
        """,
        [
            (
                source_doc_id,
                match.target_doc_id,
                match.cited_title,
                match.cited_date.isoformat(),
                match.match_score,
            )
            for match in matches
        ],
    )
