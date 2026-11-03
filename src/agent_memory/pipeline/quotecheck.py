"""Deterministic source-segment checks for extracted evidence quotes."""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


MIN_QUOTE_CHARACTERS = 20
FUZZY_THRESHOLD = 0.90
MAX_FUZZY_QUOTE_CHARACTERS = 500
MAX_FUZZY_WINDOWS = 2500
_REJECTION_KEYS_BY_PATH: dict[Path, set[tuple[str, ...]]] = {}


_REJECTION_FIELDS = (
    "document_id",
    "candidate_text",
    "quote",
    "segment_id",
    "reason",
)


@dataclass(frozen=True)
class QuoteCheckResult:
    passed: bool
    reason: str
    match_type: str | None = None
    similarity: float | None = None


def normalize_quote_text(value: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace for comparison."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized_characters = [
        character if character.isalnum() else " "
        for character in normalized
    ]
    return " ".join("".join(normalized_characters).split())


def contains_exact_quote(quote: str, source_segment: str) -> bool:
    normalized_quote = normalize_quote_text(quote)
    return bool(normalized_quote) and normalized_quote in normalize_quote_text(
        source_segment
    )


def _fuzzy_similarity(quote: str, source: str, threshold: float) -> float:
    minimum = max(1, len(quote) - max(2, round(len(quote) * 0.10)))
    maximum = min(len(source), len(quote) + max(2, round(len(quote) * 0.10)))
    maximum_start = max(0, len(source) - minimum)
    source_tokens: dict[str, list[int]] = {}
    for match in re.finditer(r"\w+", source):
        source_tokens.setdefault(match.group(), []).append(match.start())

    anchors: list[tuple[int, str, list[int]]] = []
    for token in set(re.findall(r"\w+", quote)):
        if len(token) < 3 or token not in source_tokens:
            continue
        anchors.append((len(source_tokens[token]), token, source_tokens[token]))
    candidate_starts: set[int] = set()
    tolerance = max(2, round(len(quote) * 0.10))
    offset_step = max(1, tolerance // 3)
    for _, token, source_positions in sorted(anchors)[:3]:
        quote_positions = [
            match.start()
            for match in re.finditer(rf"(?<!\w){re.escape(token)}(?!\w)", quote)
        ]
        for source_position in source_positions:
            for quote_position in quote_positions:
                estimated_start = source_position - quote_position
                for delta in range(-tolerance, tolerance + 1, offset_step):
                    candidate_starts.add(
                        min(maximum_start, max(0, estimated_start + delta))
                    )

    if not candidate_starts:
        stride = max(
            1,
            math.ceil((maximum_start + 1) / MAX_FUZZY_WINDOWS),
        )
        candidate_starts.update(range(0, maximum_start + 1, stride))
    starts = sorted(candidate_starts)
    if len(starts) > MAX_FUZZY_WINDOWS:
        stride = math.ceil(len(starts) / MAX_FUZZY_WINDOWS)
        starts = starts[::stride][:MAX_FUZZY_WINDOWS]

    best = 0.0
    windows_checked = 0
    for start in starts:
        for window_size in range(minimum, maximum + 1):
            if start + window_size > len(source):
                continue
            best = max(
                best,
                SequenceMatcher(
                    None,
                    quote,
                    source[start : start + window_size],
                    autojunk=False,
                ).ratio(),
            )
            windows_checked += 1
            if best >= threshold:
                return best
            if windows_checked >= MAX_FUZZY_WINDOWS:
                return best
    return best


def check_quote(
    quote: str,
    source_segment: str,
    *,
    threshold: float = FUZZY_THRESHOLD,
) -> QuoteCheckResult:
    """Check a quote only against its declared source segment."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    normalized_quote = normalize_quote_text(quote)
    normalized_source = normalize_quote_text(source_segment)
    if not normalized_quote:
        return QuoteCheckResult(False, "quote is blank")
    if not normalized_source:
        return QuoteCheckResult(False, "source segment is blank")

    occurrences = normalized_source.count(normalized_quote)
    if occurrences:
        if len(normalized_quote) < MIN_QUOTE_CHARACTERS and occurrences != 1:
            return QuoteCheckResult(
                False,
                "short quote is not unique in its source segment",
            )
        return QuoteCheckResult(True, "normalized exact match", "exact", 1.0)

    if len(normalized_quote) < MIN_QUOTE_CHARACTERS:
        return QuoteCheckResult(
            False,
            "quote is shorter than 20 characters and has no exact match",
        )
    if len(normalized_quote) > MAX_FUZZY_QUOTE_CHARACTERS:
        return QuoteCheckResult(False, "quote is too long for bounded fuzzy matching")

    similarity = _fuzzy_similarity(normalized_quote, normalized_source, threshold)
    if similarity >= threshold:
        return QuoteCheckResult(True, "bounded fuzzy match", "fuzzy", similarity)
    return QuoteCheckResult(
        False,
        "no source window reached the similarity threshold",
        similarity=similarity,
    )


def append_quote_rejections(
    path: Path,
    *,
    document_id: str,
    candidate_text: str,
    quote: str,
    segment_id: str,
    reason: str,
) -> None:
    """Append rejected evidence details to a local review CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = path.resolve()
    existing_keys = _REJECTION_KEYS_BY_PATH.get(resolved_path)
    if existing_keys is None:
        existing_keys = set()
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file)
                existing_keys.update(
                    tuple((row.get(field) or "") for field in _REJECTION_FIELDS)
                    for row in reader
                )
        _REJECTION_KEYS_BY_PATH[resolved_path] = existing_keys
    rejection = {
        "document_id": document_id,
        "candidate_text": candidate_text,
        "quote": quote,
        "segment_id": segment_id,
        "reason": reason,
    }
    key = tuple(rejection[field] for field in _REJECTION_FIELDS)
    if key in existing_keys:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=_REJECTION_FIELDS,
        )
        if write_header:
            writer.writeheader()
        writer.writerow(rejection)
    existing_keys.add(key)
