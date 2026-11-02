"""Point-in-time paragraph hashing and boilerplate detection."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence


TICKER_THRESHOLD = 3
GLOBAL_THRESHOLD = 20
GLOBAL_MIN_TICKERS = 2
MAX_HAMMING_DISTANCE = 3
_BAND_COUNT = MAX_HAMMING_DISTANCE + 1
_BAND_WIDTH = 64 // _BAND_COUNT
_TOKEN_PATTERN = re.compile(r"[\w$%]+", re.UNICODE)
_MATERIAL_TOPIC_PATTERN = re.compile(
    r"\b(?:cash|balance|revenue|sales|resource|reserve|grade|ore|production|"
    r"throughput|capacity|cost|capex|funding|price|drilling|lithium|"
    r"spodumene|shares|mineral|li2o|sc[456]|assay|interval|intercept)\b",
    re.IGNORECASE,
)
_MATERIAL_VALUE_PATTERN = re.compile(
    r"(?:\b(?:A\$|US\$)\s*[\d,]+(?:\.\d+)?|"
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|mt|kt|tpa|ktpa|mtpa|tonnes?|tonne|"
    r"million|billion|g/t|sc[456]|li\s*2\s*o|m)\b)",
    re.IGNORECASE,
)
_MATERIAL_TABLE_PATTERN = re.compile(
    r"<table\b|(?:\|[^|\r\n]*){3,}", re.IGNORECASE
)
_EXPLICIT_BOILERPLATE_PATTERNS = (
    re.compile(r"\babout core(?: lithium)?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:investor|media)\s+(?:enquiries|inquiries)\b|"
        r"\bfor further information\b|\bcontact details?\b|"
        r"\binfo@[\w.-]+\.\w+",
        re.IGNORECASE,
    ),
    re.compile(r"\bcompetent persons?\b", re.IGNORECASE),
    re.compile(
        r"\bforward[- ]looking statements?\b|\bimportant information\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class Paragraph:
    text: str
    simhash: str
    is_boilerplate: bool


@dataclass(frozen=True)
class HashObservation:
    scope: str
    ticker: str
    simhash: str
    doc_id: str
    published_at: datetime
    source_ticker: str
    sample_text: str


def split_paragraphs(text: str) -> list[str]:
    """Split markdown at blank lines while preserving paragraph wording."""
    normalized_line_endings = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n[ \t]*\n+", normalized_line_endings)
        if paragraph.strip()
    ]


def _normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return _TOKEN_PATTERN.findall(normalized)


def simhash64(text: str) -> str:
    """Return a deterministic 64-bit SimHash as a 16-character hex string."""
    tokens = _normalized_tokens(text)
    if not tokens:
        digest = hashlib.blake2b(
            unicodedata.normalize("NFKC", text).encode("utf-8"), digest_size=8
        )
        return digest.hexdigest()

    weights = [0] * 64
    for token in tokens:
        token_hash = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
        )
        for bit in range(64):
            weights[bit] += 1 if token_hash & (1 << bit) else -1

    result = sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0)
    return f"{result:016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def is_material_fact_paragraph(text: str) -> bool:
    """Protect numeric financial, operating, and resource details from filtering."""
    has_numeric_content = bool(
        _MATERIAL_VALUE_PATTERN.search(text)
        or (
            _MATERIAL_TABLE_PATTERN.search(text)
            and re.search(r"\d", text)
            and _MATERIAL_TOPIC_PATTERN.search(text)
        )
    )
    return has_numeric_content and bool(_MATERIAL_TOPIC_PATTERN.search(text))


def _is_explicit_boilerplate(text: str) -> bool:
    if is_material_fact_paragraph(text):
        return False
    return any(pattern.search(text) for pattern in _EXPLICIT_BOILERPLATE_PATTERNS)


def _bands(simhash: str) -> Iterable[tuple[int, int]]:
    value = int(simhash, 16)
    mask = (1 << _BAND_WIDTH) - 1
    for band in range(_BAND_COUNT):
        yield band, (value >> (band * _BAND_WIDTH)) & mask


class BoilerplateIndex:
    """In-memory candidate index over persisted, dated hash observations."""

    def __init__(
        self,
        *,
        ticker_threshold: int = TICKER_THRESHOLD,
        global_threshold: int = GLOBAL_THRESHOLD,
        global_min_tickers: int = GLOBAL_MIN_TICKERS,
    ) -> None:
        if min(ticker_threshold, global_threshold, global_min_tickers) <= 0:
            raise ValueError("Boilerplate thresholds must be positive")
        self.ticker_threshold = ticker_threshold
        self.global_threshold = global_threshold
        self.global_min_tickers = global_min_tickers
        self.max_hamming_distance = MAX_HAMMING_DISTANCE
        self._occurrences: dict[
            tuple[str, str, str], list[tuple[datetime, str, str]]
        ] = {}
        self._bands: dict[tuple[str, str, int, int], set[str]] = {}
        self._seen: set[tuple[str, str, str, str]] = set()

    @classmethod
    def from_connection(
        cls, connection: sqlite3.Connection, **thresholds: int
    ) -> BoilerplateIndex:
        index = cls(**thresholds)
        rows = connection.execute(
            """
            SELECT
                observation.scope,
                observation.ticker,
                observation.simhash,
                observation.doc_id,
                observation.published_at,
                document.ticker AS source_ticker,
                observation.sample_text
            FROM paragraph_hash_observation AS observation
            JOIN document ON document.id = observation.doc_id
            ORDER BY observation.published_at, observation.doc_id
            """
        ).fetchall()
        index.add_observations(
            HashObservation(
                scope=row["scope"],
                ticker=row["ticker"],
                simhash=row["simhash"],
                doc_id=row["doc_id"],
                published_at=datetime.fromisoformat(row["published_at"]),
                source_ticker=row["source_ticker"],
                sample_text=row["sample_text"],
            )
            for row in rows
        )
        return index

    def add_observations(self, observations: Iterable[HashObservation]) -> None:
        for observation in observations:
            identity = (
                observation.scope,
                observation.ticker,
                observation.simhash,
                observation.doc_id,
            )
            if identity in self._seen:
                continue
            self._seen.add(identity)
            key = (
                observation.scope,
                observation.ticker,
                observation.simhash,
            )
            self._occurrences.setdefault(key, []).append(
                (
                    observation.published_at,
                    observation.doc_id,
                    observation.source_ticker,
                )
            )
            for band, band_value in _bands(observation.simhash):
                bucket = (
                    observation.scope,
                    observation.ticker,
                    band,
                    band_value,
                )
                self._bands.setdefault(bucket, set()).add(observation.simhash)

    def _candidate_hashes(self, scope: str, ticker: str, simhash: str) -> set[str]:
        candidates: set[str] = set()
        for band, band_value in _bands(simhash):
            candidates.update(
                self._bands.get((scope, ticker, band, band_value), set())
            )
        return {
            candidate
            for candidate in candidates
            if hamming_distance(candidate, simhash) <= self.max_hamming_distance
        }

    def _prior_documents(
        self,
        *,
        scope: str,
        ticker: str,
        simhash: str,
        before: datetime,
    ) -> set[tuple[str, str]]:
        documents: set[tuple[str, str]] = set()
        for candidate in self._candidate_hashes(scope, ticker, simhash):
            for published_at, doc_id, source_ticker in self._occurrences[
                (scope, ticker, candidate)
            ]:
                if published_at < before:
                    documents.add((doc_id, source_ticker))
        return documents

    def is_boilerplate(self, simhash: str, ticker: str, published_at: datetime) -> bool:
        ticker_documents = self._prior_documents(
            scope="ticker",
            ticker=ticker,
            simhash=simhash,
            before=published_at,
        )
        if len(ticker_documents) >= self.ticker_threshold:
            return True

        global_documents = self._prior_documents(
            scope="global",
            ticker="",
            simhash=simhash,
            before=published_at,
        )
        global_tickers = {source_ticker for _, source_ticker in global_documents}
        return (
            len(global_documents) >= self.global_threshold
            and len(global_tickers) >= self.global_min_tickers
        )


def observations_for_document(
    *,
    doc_id: str,
    ticker: str,
    published_at: datetime,
    paragraphs: Sequence[str],
) -> list[HashObservation]:
    """Create one ticker and one global observation per unique document hash."""
    observations: dict[tuple[str, str, str], HashObservation] = {}
    for paragraph in paragraphs:
        signature = simhash64(paragraph)
        for scope, scope_ticker in (("ticker", ticker), ("global", "")):
            key = (scope, scope_ticker, signature)
            observations.setdefault(
                key,
                HashObservation(
                    scope=scope,
                    ticker=scope_ticker,
                    simhash=signature,
                    doc_id=doc_id,
                    published_at=published_at,
                    source_ticker=ticker,
                    sample_text=paragraph[:1000],
                ),
            )
    return list(observations.values())


def classify_paragraphs(
    paragraphs: Sequence[str],
    *,
    ticker: str,
    published_at: datetime,
    index: BoilerplateIndex,
) -> list[Paragraph]:
    """Classify all paragraphs against prior documents, not this document."""
    results = []
    for paragraph in paragraphs:
        signature = simhash64(paragraph)
        results.append(
            Paragraph(
                text=paragraph,
                simhash=signature,
                is_boilerplate=(
                    False
                    if is_material_fact_paragraph(paragraph)
                    else _is_explicit_boilerplate(paragraph)
                    or index.is_boilerplate(signature, ticker, published_at)
                ),
            )
        )
    return results


def persist_observations(
    connection: sqlite3.Connection,
    observations: Sequence[HashObservation],
) -> list[HashObservation]:
    """Persist observations and increment aggregate counts only for new rows."""
    inserted: list[HashObservation] = []
    for observation in observations:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO paragraph_hash_observation (
                scope, ticker, simhash, doc_id, published_at, sample_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                observation.scope,
                observation.ticker,
                observation.simhash,
                observation.doc_id,
                observation.published_at.isoformat(),
                observation.sample_text,
            ),
        )
        if cursor.rowcount == 0:
            continue
        connection.execute(
            """
            INSERT INTO paragraph_hash (
                scope, ticker, simhash, first_seen_at, doc_count, sample_text
            ) VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(scope, ticker, simhash) DO UPDATE SET
                first_seen_at = MIN(paragraph_hash.first_seen_at, excluded.first_seen_at),
                doc_count = paragraph_hash.doc_count + 1
            """,
            (
                observation.scope,
                observation.ticker,
                observation.simhash,
                observation.published_at.isoformat(),
                observation.sample_text,
            ),
        )
        inserted.append(observation)
    return inserted
