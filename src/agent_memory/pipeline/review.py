"""Exports extraction results for focused manual tier review."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agent_memory.db import database_connection
from agent_memory.pipeline.extract import (
    CachedExtraction,
    SourceSegment,
    segment_chunks,
)
from agent_memory.pipeline.quotecheck import check_quote


FIXED_QUARTERLY_IDS = (
    "20250122-2145-Quarterly-Activities-and-Cashflow-Report-2A1574556",
    "20250428-2310-Quarterly-Activities-and-Cashflow-Report-2A1593142",
    "20250729-2314-Quarterly-Activities-and-Cashflow-Report-2A1610591",
    "20251012-2205-Quarterly-Activities-and-Cashflow-Report-2A1628714",
    "20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068",
)

_REVIEW_FIELDS = (
    "document_id",
    "headline",
    "truncated_fact_count",
    "candidate_text",
    "tier",
    "entity_name",
    "entity_type",
    "attribute",
    "as_of",
    "period",
    "quote",
    "segment_id",
    "review_status",
    "review_notes",
)


@dataclass(frozen=True)
class T1ReviewSummary:
    documents: int
    cached_documents: int
    candidates: int
    stale_or_missing: tuple[str, ...]


def _document_segments(
    connection: sqlite3.Connection,
    doc_id: str,
    *,
    max_source_characters: int,
) -> list[SourceSegment]:
    rows = connection.execute(
        """
        SELECT ord, text FROM chunk
        WHERE doc_id = ? AND is_boilerplate = 0
        ORDER BY ord
        """,
        (doc_id,),
    ).fetchall()
    return segment_chunks(
        [(int(row["ord"]), row["text"]) for row in rows],
        max_source_characters=max_source_characters,
    )


def _source_hash(segments: Sequence[SourceSegment]) -> str:
    serialized = json.dumps(
        [(segment.segment_id, segment.text) for segment in segments],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def export_t1_review(
    database_path: Path,
    model: str,
    output_path: Path,
) -> T1ReviewSummary:
    """Export current, quote-verified T1 candidates from five fixed quarterlies."""
    records: list[dict[str, str]] = []
    cached_documents = 0
    stale_or_missing: list[str] = []
    with database_connection(database_path) as connection:
        for doc_id in FIXED_QUARTERLY_IDS:
            document = connection.execute(
                "SELECT headline FROM document WHERE id = ?", (doc_id,)
            ).fetchone()
            cached_row = connection.execute(
                """
                SELECT output_json FROM extraction_cache
                WHERE doc_id = ? AND model = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (doc_id, model),
            ).fetchone()
            if document is None or cached_row is None:
                stale_or_missing.append(doc_id)
                continue
            cached = CachedExtraction.model_validate_json(cached_row["output_json"])
            segments = _document_segments(
                connection,
                doc_id,
                max_source_characters=cached.segment_source_characters,
            )
            if (
                cached.source_doc_id != doc_id
                or cached.source_hash != _source_hash(segments)
            ):
                stale_or_missing.append(doc_id)
                continue
            cached_documents += 1
            segment_map = {segment.segment_id: segment.text for segment in segments}
            for fact in cached.facts:
                if fact.tier != "T1":
                    continue
                source = segment_map.get(fact.segment_id)
                if source is None or not check_quote(fact.quote, source).passed:
                    continue
                records.append(
                    {
                        "document_id": doc_id,
                        "headline": document["headline"],
                        "truncated_fact_count": str(
                            cached.truncated_fact_count
                        ),
                        "candidate_text": fact.text,
                        "tier": fact.tier,
                        "entity_name": fact.entity.name,
                        "entity_type": fact.entity.type,
                        "attribute": fact.attribute,
                        "as_of": fact.as_of.isoformat() if fact.as_of else "",
                        "period": fact.period or "",
                        "quote": fact.quote,
                        "segment_id": fact.segment_id,
                        "review_status": "",
                        "review_notes": "",
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return T1ReviewSummary(
        documents=len(FIXED_QUARTERLY_IDS),
        cached_documents=cached_documents,
        candidates=len(records),
        stale_or_missing=tuple(stale_or_missing),
    )
