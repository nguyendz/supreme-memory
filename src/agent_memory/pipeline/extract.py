"""Source-grounded candidate-fact extraction with point-in-time cache replay."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import httpx
import ollama
from pydantic import BaseModel, ConfigDict, Field

from agent_memory.db import database_connection
from agent_memory.llm import LLMClient, LLMOutputError
from agent_memory.pipeline.quotecheck import (
    QuoteCheckResult,
    append_quote_rejections,
    check_quote,
)


MAX_INPUT_TOKENS = 22_000
PROMPT_AND_OUTPUT_RESERVE_TOKENS = 5_000
EXTRACTION_NUM_PREDICT = 8_192
CHARS_PER_TOKEN = 3
MAX_SEGMENT_SOURCE_CHARACTERS = (
    MAX_INPUT_TOKENS - PROMPT_AND_OUTPUT_RESERVE_TOKENS
) * CHARS_PER_TOKEN
FALLBACK_SEGMENT_SOURCE_CHARACTERS = (15_000, 7_500, 3_750, 1_875)
MAX_RECOVERY_FACTS_PER_SEGMENT = 15
_RECOVERY_PROMPT = (
    "This is a smaller recovery segment. Return no more than "
    f"{MAX_RECOVERY_FACTS_PER_SEGMENT} distinct facts. Prioritize material T1 "
    "facts, then the most useful T2 context. Do not enumerate every table row "
    "or repeat the same claim. Keep each emitted fact atomic and source-grounded."
)
MAX_PIT_CONTEXT_CHARACTERS = 7_000
MAX_PIT_CONTEXT_DOCUMENTS = 50
MAX_PIT_CONTEXT_FACTS = 40
SEGMENTER_VERSION = "paragraph-heading-page-v2"

FactType = Literal[
    "metric",
    "event",
    "plan_guidance",
    "relationship",
    "reference_data",
    "opinion_commentary",
]
FactTier = Literal["T0", "T1", "T2", "T3"]
SUPPORTED_FORM_FAMILIES = (
    "appendix_2a",
    "appendix_3b",
    "director_interest",
    "substantial_holder",
)
FORM_PROMPT_FILES = {
    "appendix_2a": "appendix_2a.md",
    "appendix_3b": "appendix_3b.md",
    "director_interest": "director_interest.md",
    "substantial_holder": "substantial_holder.md",
}


class CandidateEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class CandidateFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: CandidateEntity
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    value_num: float | None = None
    unit: str | None = None
    qualifier: str | None = None
    as_of: date | None = None
    period: str | None = None
    fact_type: FactType
    tier: FactTier
    salience: float = Field(ge=0, le=1)
    text: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[CandidateFact]


class RawSegmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    raw_content: str


class ExtractionUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 0
    wall_duration_ms: float = 0


class CachedExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_doc_id: str
    source_form_type: str | None
    form_family: str
    source_hash: str
    facts: list[CandidateFact]
    raw_segments: list[RawSegmentOutput]
    usage: ExtractionUsage
    segment_source_characters: int = Field(
        default=MAX_SEGMENT_SOURCE_CHARACTERS,
        gt=0,
    )
    truncated_fact_count: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class SourceSegment:
    segment_id: str
    text: str


def _next_fallback_segment_size(current: int | None) -> int | None:
    if current is None:
        return FALLBACK_SEGMENT_SOURCE_CHARACTERS[0]
    try:
        index = FALLBACK_SEGMENT_SOURCE_CHARACTERS.index(current)
    except ValueError:
        return None
    if index + 1 >= len(FALLBACK_SEGMENT_SOURCE_CHARACTERS):
        return None
    return FALLBACK_SEGMENT_SOURCE_CHARACTERS[index + 1]


@dataclass(frozen=True)
class CandidateReview:
    fact: CandidateFact
    quote_check: QuoteCheckResult


@dataclass(frozen=True)
class ExtractionResult:
    document_id: str
    status: Literal["extracted", "unsupported"]
    form_family: str | None
    cache_hit: bool
    facts: tuple[CandidateFact, ...]
    accepted_facts: tuple[CandidateFact, ...]
    rejected_facts: tuple[CandidateReview, ...]
    prompt_tokens: int
    output_tokens: int
    attempts: int
    wall_duration_ms: float
    truncated_fact_count: int = 0


@dataclass
class ExtractionSummary:
    discovered: int
    selected: int
    extracted: int = 0
    cache_hits: int = 0
    unsupported: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 0
    truncated_facts: int = 0
    elapsed_seconds: float = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _prompt_root(prompt_directory: Path | None) -> Path:
    return prompt_directory or Path(__file__).resolve().parents[3] / "prompts"


def normalize_form_family(
    form_type: str | None,
    headline: str,
    source_excerpt: str = "",
) -> str | None:
    """Map known ASX notice variants to the four supported extraction families."""
    if form_type in SUPPORTED_FORM_FAMILIES:
        return form_type

    text = unicodedata.normalize(
        "NFKC", f"{headline}\n{source_excerpt[:3000]}"
    ).casefold()
    if (
        re.search(r"\bappendix\s*3[yz]\b", text)
        or re.search(r"\b(?:change|initial)\s+of\s+director'?s?\s+interest\b", text)
    ):
        return "director_interest"
    if (
        re.search(r"\bform\s*(?:603|604|605)\b", text)
        or re.search(r"\bappendix\s*6(?:0[345])\b", text)
        or "substantial holder" in text
    ):
        return "substantial_holder"
    if (
        re.search(r"\bappendix\s*2a\b", text)
        or "application for quotation of securities" in text
    ):
        return "appendix_2a"
    if (
        re.search(r"\bappendix\s*3b\b", text)
        or "proposed issue of securities" in text
    ):
        return "appendix_3b"
    return None


def _split_long_text(text: str, max_characters: int) -> list[str]:
    remaining = text.strip()
    parts: list[str] = []
    while len(remaining) > max_characters:
        split_at = remaining.rfind(" ", 0, max_characters + 1)
        if split_at <= 0:
            split_at = max_characters
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def segment_chunks(
    chunks: Sequence[tuple[int, str]],
    *,
    max_source_characters: int = MAX_SEGMENT_SOURCE_CHARACTERS,
) -> list[SourceSegment]:
    """Group stable paragraph chunks at heading/page boundaries and token limits."""
    if max_source_characters <= 0:
        raise ValueError("max_source_characters must be positive")

    items: list[tuple[int, str]] = []
    for ordinal, text in chunks:
        pieces = _split_long_text(text, max_source_characters)
        items.extend((ordinal, piece) for piece in pieces)

    segments: list[SourceSegment] = []
    current: list[tuple[int, str]] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if not current:
            return
        segment_number = len(segments)
        segment_text = "\n\n".join(text for _, text in current)
        segments.append(
            SourceSegment(
                segment_id=(
                    f"segment-{segment_number:04d}-"
                    f"{current[0][0]:05d}-{current[-1][0]:05d}"
                ),
                text=segment_text,
            )
        )
        current = []
        current_length = 0

    boundary_pattern = re.compile(
        r"^(?:#{1,6}\s|<!--\s*page\b|[-=]{3,}\s*(?:page|end of page))",
        re.IGNORECASE,
    )
    for ordinal, text in items:
        heading_boundary = bool(boundary_pattern.search(text.strip()))
        if current and (
            (
                heading_boundary
                and current_length
                >= int(max_source_characters * 0.75)
            )
            or current_length + len(text) + 2 > max_source_characters
        ):
            flush()
        current.append((ordinal, text))
        current_length += len(text) + 2
    flush()
    return segments


def _segments_for_document(
    connection: sqlite3.Connection,
    doc_id: str,
    *,
    max_source_characters: int = MAX_SEGMENT_SOURCE_CHARACTERS,
) -> list[SourceSegment]:
    rows = connection.execute(
        """
        SELECT ord, text
        FROM chunk
        WHERE doc_id = ? AND is_boilerplate = 0
        ORDER BY ord
        """,
        (doc_id,),
    ).fetchall()
    segments = segment_chunks(
        [(int(row["ord"]), row["text"]) for row in rows],
        max_source_characters=max_source_characters,
    )
    if not segments:
        raise ValueError(f"Document has no non-boilerplate chunks: {doc_id}")
    return segments


def _source_hash(segments: Sequence[SourceSegment]) -> str:
    payload = json.dumps(
        [(segment.segment_id, segment.text) for segment in segments],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prioritize_recovery_facts(
    facts: Sequence[CandidateFact],
) -> tuple[list[CandidateFact], int]:
    tier_priority = {"T1": 0, "T2": 1, "T3": 2, "T0": 3}
    ranked = sorted(
        enumerate(facts),
        key=lambda item: (
            tier_priority[item[1].tier],
            -item[1].salience,
            item[0],
        ),
    )
    selected = sorted(
        ranked[:MAX_RECOVERY_FACTS_PER_SEGMENT],
        key=lambda item: item[0],
    )
    return [fact for _, fact in selected], len(facts) - len(selected)


def _prompt_versions(
    *,
    prompt_text: str,
    family_prompt: str,
    family: str,
    segment_source_characters: int = MAX_SEGMENT_SOURCE_CHARACTERS,
) -> tuple[str, str]:
    schema_json = json.dumps(
        ExtractionOutput.model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    mapping_json = json.dumps(FORM_PROMPT_FILES, sort_keys=True, separators=(",", ":"))
    common = "\n".join(
        (
            prompt_text,
            schema_json,
            mapping_json,
            SEGMENTER_VERSION,
            str(MAX_INPUT_TOKENS),
            str(PROMPT_AND_OUTPUT_RESERVE_TOKENS),
            str(EXTRACTION_NUM_PREDICT),
            str(CHARS_PER_TOKEN),
        )
    )
    base_version = hashlib.sha256(common.encode("utf-8")).hexdigest()[:16]
    family_payload = f"{family}\n{family_prompt}"
    if segment_source_characters != MAX_SEGMENT_SOURCE_CHARACTERS:
        family_payload += (
            f"\nsegment-source-characters={segment_source_characters}"
            f"\n{_RECOVERY_PROMPT}"
        )
    family_version = hashlib.sha256(family_payload.encode("utf-8")).hexdigest()[:12]
    return base_version, f"{base_version}-{family_version}"


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _verified_prior_context(
    connection: sqlite3.Connection,
    *,
    current_doc: sqlite3.Row,
    model: str,
    base_version: str,
) -> str:
    rows = connection.execute(
        """
        SELECT d.id, d.ticker, d.headline, d.published_at, c.output_json
        FROM extraction_cache AS c
        JOIN document AS d ON d.id = c.doc_id
        WHERE c.model = ?
          AND c.prompt_version LIKE ?
          AND d.ticker = ?
          AND date(d.published_at) <= date(?)
          AND d.triage_class IN ('full', 'form')
        ORDER BY d.published_at DESC, d.id DESC
        LIMIT ?
        """,
        (
            model,
            f"{base_version}-%",
            current_doc["ticker"],
            current_doc["published_at"],
            MAX_PIT_CONTEXT_DOCUMENTS,
        ),
    ).fetchall()
    current_published_at = _utc_datetime(current_doc["published_at"])
    context_lines: list[str] = []
    context_length = 0
    facts_added = 0
    for row in rows:
        if _utc_datetime(row["published_at"]) >= current_published_at:
            continue
        try:
            cached = CachedExtraction.model_validate_json(row["output_json"])
        except ValueError:
            continue
        if cached.source_doc_id != row["id"]:
            continue
        try:
            prior_segments = _segments_for_document(
                connection,
                row["id"],
                max_source_characters=cached.segment_source_characters,
            )
        except ValueError:
            continue
        if cached.source_hash != _source_hash(prior_segments):
            continue
        segment_map = {segment.segment_id: segment.text for segment in prior_segments}
        for fact in cached.facts:
            if fact.tier == "T0" or fact.segment_id not in segment_map:
                continue
            if not check_quote(fact.quote, segment_map[fact.segment_id]).passed:
                continue
            line = (
                f"[{row['published_at']} {row['ticker']} {row['headline']}] "
                f"{fact.tier} {fact.text} Evidence: {fact.quote}"
            )
            if context_length + len(line) + 1 > MAX_PIT_CONTEXT_CHARACTERS:
                return "\n".join(context_lines)
            context_lines.append(line)
            context_length += len(line) + 1
            facts_added += 1
            if facts_added >= MAX_PIT_CONTEXT_FACTS:
                return "\n".join(context_lines)
    return "\n".join(context_lines)


def _make_segment_prompt(
    *,
    document: sqlite3.Row,
    types_json: str,
    xref_titles: Sequence[str],
    prior_context: str,
    segment: SourceSegment,
) -> str:
    try:
        announcement_types = json.loads(types_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Document announcement type metadata is invalid: {document['id']}"
        ) from exc
    if not isinstance(announcement_types, list) or not all(
        isinstance(item, str) for item in announcement_types
    ):
        raise ValueError(f"Document announcement type metadata is invalid: {document['id']}")

    return "\n".join(
        (
            "Extract source-grounded candidate facts from this one source segment.",
            f"Document ID: {document['id']}",
            f"Ticker: {document['ticker']}",
            f"Headline: {document['headline']}",
            f"Published at: {document['published_at']}",
            f"Category: {document['category'] or 'unknown'}",
            f"Announcement types: {', '.join(announcement_types) or 'unknown'}",
            f"Price sensitive: {document['price_sensitive']}",
            f"Original form type: {document['form_type'] or 'not a form'}",
            "Referenced document titles:",
            *(f"- {title}" for title in xref_titles),
            "Earlier quote-verified context (context only; not evidence):",
            prior_context or "(none)",
            f"Source segment ID: {segment.segment_id}",
            "Only use the text between the source markers as evidence for quotes.",
            "<source-segment>",
            segment.text,
            "</source-segment>",
        )
    )


def _review_facts(
    *,
    document_id: str,
    facts: Sequence[CandidateFact],
    segments: Sequence[SourceSegment],
    rejection_path: Path | None,
) -> tuple[tuple[CandidateFact, ...], tuple[CandidateReview, ...]]:
    segment_map = {segment.segment_id: segment.text for segment in segments}
    accepted: list[CandidateFact] = []
    rejected: list[CandidateReview] = []
    for fact in facts:
        source_text = segment_map.get(fact.segment_id)
        if source_text is None:
            check = QuoteCheckResult(
                False, "candidate refers to an unknown source segment"
            )
        else:
            check = check_quote(fact.quote, source_text)
        if check.passed:
            accepted.append(fact)
        else:
            rejected.append(CandidateReview(fact=fact, quote_check=check))
            if rejection_path is not None:
                append_quote_rejections(
                    rejection_path,
                    document_id=document_id,
                    candidate_text=fact.text,
                    quote=fact.quote,
                    segment_id=fact.segment_id,
                    reason=check.reason,
                )
    return tuple(accepted), tuple(rejected)


def _cache_result(
    *,
    database_path: Path,
    document_id: str,
    model: str,
    prompt_version: str,
    cached: CachedExtraction,
) -> None:
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO extraction_cache (
                doc_id, model, prompt_version, output_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, model, prompt_version) DO UPDATE SET
                output_json = excluded.output_json,
                created_at = excluded.created_at
            """,
            (
                document_id,
                model,
                prompt_version,
                cached.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def extract_document(
    database_path: Path,
    client: LLMClient,
    document_id: str,
    *,
    prompt_directory: Path | None = None,
    model: str | None = None,
    force: bool = False,
    rejection_path: Path | None = None,
    max_segment_source_characters: int | None = None,
) -> ExtractionResult:
    """Extract and quote-check a document without writing canonical facts."""
    started = time.perf_counter()
    if (
        max_segment_source_characters is not None
        and max_segment_source_characters <= 0
    ):
        raise ValueError("max_segment_source_characters must be positive")
    segment_source_characters = (
        max_segment_source_characters or MAX_SEGMENT_SOURCE_CHARACTERS
    )
    selected_model = model or client.model_name
    with database_connection(database_path) as connection:
        document = connection.execute(
            "SELECT * FROM document WHERE id = ?", (document_id,)
        ).fetchone()
        if document is None:
            raise ValueError(f"Document is not in the database: {document_id}")
        if document["triage_class"] not in ("full", "form"):
            raise ValueError(
                f"Document is not an extraction candidate: {document_id}"
            )
        first_source_rows = connection.execute(
            """
            SELECT text FROM chunk
            WHERE doc_id = ? AND is_boilerplate = 0
            ORDER BY ord LIMIT 2
            """,
            (document_id,),
        ).fetchall()
        first_source_text = "\n".join(row["text"] for row in first_source_rows)
        family = (
            "full"
            if document["triage_class"] == "full"
            else normalize_form_family(
                document["form_type"], document["headline"], first_source_text
            )
        )
        if family is None:
            return ExtractionResult(
                document_id=document_id,
                status="unsupported",
                form_family=None,
                cache_hit=False,
                facts=(),
                accepted_facts=(),
                rejected_facts=(),
                prompt_tokens=0,
                output_tokens=0,
                attempts=0,
                wall_duration_ms=(time.perf_counter() - started) * 1000,
            )
        segments = _segments_for_document(
            connection,
            document_id,
            max_source_characters=segment_source_characters,
        )

        prompt_root = _prompt_root(prompt_directory)
        prompt_text = (prompt_root / "extract.md").read_text(encoding="utf-8")
        family_prompt = ""
        if family in FORM_PROMPT_FILES:
            family_prompt = (
                prompt_root / "forms" / FORM_PROMPT_FILES[family]
            ).read_text(encoding="utf-8")
        base_version, prompt_version = _prompt_versions(
            prompt_text=prompt_text,
            family_prompt=family_prompt,
            family=family,
            segment_source_characters=segment_source_characters,
        )
        source_hash = _source_hash(segments)
        cache_row = connection.execute(
            """
            SELECT output_json
            FROM extraction_cache
            WHERE doc_id = ? AND model = ? AND prompt_version = ?
            """,
            (document_id, selected_model, prompt_version),
        ).fetchone()
        if cache_row is None and max_segment_source_characters is None and not force:
            for fallback_size in FALLBACK_SEGMENT_SOURCE_CHARACTERS:
                _, fallback_prompt_version = _prompt_versions(
                    prompt_text=prompt_text,
                    family_prompt=family_prompt,
                    family=family,
                    segment_source_characters=fallback_size,
                )
                fallback_row = connection.execute(
                    """
                    SELECT output_json
                    FROM extraction_cache
                    WHERE doc_id = ? AND model = ? AND prompt_version = ?
                    """,
                    (document_id, selected_model, fallback_prompt_version),
                ).fetchone()
                if fallback_row is None:
                    continue
                segment_source_characters = fallback_size
                segments = _segments_for_document(
                    connection,
                    document_id,
                    max_source_characters=segment_source_characters,
                )
                source_hash = _source_hash(segments)
                prompt_version = fallback_prompt_version
                cache_row = fallback_row
                break
        cached: CachedExtraction | None = None
        if cache_row is not None and not force:
            cached = CachedExtraction.model_validate_json(cache_row["output_json"])
            if (
                cached.source_doc_id != document_id
                or cached.source_hash != source_hash
                or cached.form_family != family
                or cached.segment_source_characters != segment_source_characters
            ):
                cached = None

        xref_titles = [
            row["cited_title"]
            for row in connection.execute(
                """
                SELECT cited_title FROM doc_xref
                WHERE src_doc = ?
                ORDER BY cited_date, cited_title
                """,
                (document_id,),
            ).fetchall()
        ]
        prior_context = (
            ""
            if cached is not None
            else _verified_prior_context(
                connection,
                current_doc=document,
                model=selected_model,
                base_version=base_version,
            )
        )
        document_metadata = {
            "id": document["id"],
            "ticker": document["ticker"],
            "headline": document["headline"],
            "published_at": document["published_at"],
            "category": document["category"],
            "price_sensitive": document["price_sensitive"],
            "form_type": document["form_type"],
        }
        types_json = document["types_json"]

    if cached is None:
        raw_segments: list[RawSegmentOutput] = []
        extracted_facts: list[CandidateFact] = []
        prompt_tokens = 0
        output_tokens = 0
        attempts = 0
        wall_duration_ms = 0.0
        truncated_fact_count = 0
        system_prompt = prompt_text
        if family_prompt:
            system_prompt = f"{system_prompt}\n\n{family_prompt}"
        if segment_source_characters != MAX_SEGMENT_SOURCE_CHARACTERS:
            system_prompt = f"{system_prompt}\n\n{_RECOVERY_PROMPT}"
        for segment in segments:
            prompt = _make_segment_prompt(
                document=document_metadata,
                types_json=types_json,
                xref_titles=xref_titles,
                prior_context=prior_context,
                segment=segment,
            )
            try:
                result = client.chat_json(
                    prompt,
                    ExtractionOutput,
                    system_prompt=system_prompt,
                    doc_id=document_id,
                    model=selected_model,
                    num_predict=EXTRACTION_NUM_PREDICT,
                )
            except (LLMOutputError, ollama.ResponseError) as exc:
                fallback_worthy = isinstance(exc, LLMOutputError) or (
                    "token repeat limit reached" in str(exc).casefold()
                )
                next_segment_size = _next_fallback_segment_size(
                    max_segment_source_characters
                )
                if next_segment_size is not None and fallback_worthy:
                    return extract_document(
                        database_path,
                        client,
                        document_id,
                        prompt_directory=prompt_directory,
                        model=selected_model,
                        force=force,
                        rejection_path=rejection_path,
                        max_segment_source_characters=next_segment_size,
                    )
                raise
            facts_for_segment = list(result.value.facts)
            if segment_source_characters != MAX_SEGMENT_SOURCE_CHARACTERS:
                    facts_for_segment, discarded = _prioritize_recovery_facts(
                        facts_for_segment
                    )
                    truncated_fact_count += discarded
            segment_facts = [
                    fact.model_copy(update={"segment_id": segment.segment_id})
                    for fact in facts_for_segment
            ]
            raw_segments.append(
                RawSegmentOutput(
                    segment_id=segment.segment_id,
                    raw_content=result.raw_content,
                )
            )
            extracted_facts.extend(segment_facts)
            prompt_tokens += result.prompt_tokens
            output_tokens += result.output_tokens
            attempts += result.attempts
            wall_duration_ms += result.wall_duration_ms

        cached = CachedExtraction(
            source_doc_id=document_id,
            source_form_type=document["form_type"],
            form_family=family,
            source_hash=source_hash,
            facts=extracted_facts,
            raw_segments=raw_segments,
            usage=ExtractionUsage(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                attempts=attempts,
                wall_duration_ms=wall_duration_ms,
            ),
            segment_source_characters=segment_source_characters,
            truncated_fact_count=truncated_fact_count,
        )
        _cache_result(
            database_path=database_path,
            document_id=document_id,
            model=selected_model,
            prompt_version=prompt_version,
            cached=cached,
        )
        cache_hit = False
    else:
        cache_hit = True

    accepted, rejected = _review_facts(
        document_id=document_id,
        facts=cached.facts,
        segments=segments,
        rejection_path=rejection_path,
    )
    return ExtractionResult(
        document_id=document_id,
        status="extracted",
        form_family=family,
        cache_hit=cache_hit,
        facts=tuple(cached.facts),
        accepted_facts=accepted,
        rejected_facts=rejected,
        prompt_tokens=cached.usage.prompt_tokens,
        output_tokens=cached.usage.output_tokens,
        attempts=cached.usage.attempts,
        wall_duration_ms=cached.usage.wall_duration_ms,
        truncated_fact_count=cached.truncated_fact_count,
    )


def write_unsupported_forms(path: Path, rows: Sequence[tuple[str, str, str]]) -> None:
    """Write all unsupported form IDs, headlines, and source form types."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("document_id", "headline", "form_type"))
        writer.writerows(rows)


def run_extraction(
    database_path: Path,
    client: LLMClient,
    *,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    force: bool = False,
    prompt_directory: Path | None = None,
    model: str | None = None,
    rejection_path: Path | None = None,
    unsupported_path: Path | None = None,
) -> ExtractionSummary:
    """Process selected full/form documents and report per-document failures."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    started = time.perf_counter()
    conditions = ["triage_class IN ('full', 'form')"]
    parameters: list[object] = []
    if ticker is not None:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("ticker must not be blank")
        conditions.append("ticker = ?")
        parameters.append(normalized_ticker)
    if start_date is not None:
        conditions.append("date(published_at) >= date(?)")
        parameters.append(start_date.isoformat())
    if end_date is not None:
        conditions.append("date(published_at) <= date(?)")
        parameters.append(end_date.isoformat())
    where_clause = " AND ".join(conditions)
    with database_connection(database_path) as connection:
        discovered = int(
            connection.execute(
                f"SELECT COUNT(*) AS count FROM document WHERE {where_clause}",
                parameters,
            ).fetchone()["count"]
        )
        query = (
            "SELECT id, headline, form_type FROM document "
            f"WHERE {where_clause} ORDER BY published_at, ticker, id"
        )
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        rows = connection.execute(query, parameters).fetchall()
    summary = ExtractionSummary(discovered=discovered, selected=len(rows))
    unsupported: list[tuple[str, str, str]] = []
    for row in rows:
        try:
            result = extract_document(
                database_path,
                client,
                row["id"],
                prompt_directory=prompt_directory,
                model=model,
                force=force,
                rejection_path=rejection_path,
            )
        except (
            OSError,
            ValueError,
            sqlite3.Error,
            httpx.HTTPError,
            ollama.ResponseError,
        ) as exc:
            summary.failed += 1
            summary.errors.append((row["id"], str(exc)))
            continue
        if result.status == "unsupported":
            summary.unsupported += 1
            unsupported.append(
                (row["id"], row["headline"], row["form_type"] or "other")
            )
            continue
        summary.extracted += 1
        summary.cache_hits += int(result.cache_hit)
        summary.prompt_tokens += result.prompt_tokens
        summary.output_tokens += result.output_tokens
        summary.attempts += result.attempts
        summary.truncated_facts += result.truncated_fact_count
    if unsupported_path is not None:
        write_unsupported_forms(unsupported_path, unsupported)
    summary.elapsed_seconds = time.perf_counter() - started
    return summary
