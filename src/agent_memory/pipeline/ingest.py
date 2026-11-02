"""Document catalog, warm-up, and per-announcement ingestion orchestration."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import ollama

from agent_memory.config import AppConfig
from agent_memory.db import (
    database_connection,
    initialize_database,
    rebuild_paragraph_hash_aggregates,
    upsert_document,
)
from agent_memory.llm import LLMClient
from agent_memory.loader import (
    Announcement,
    discover_announcements,
    read_announcement_text,
)
from agent_memory.pipeline.boilerplate import (
    BoilerplateIndex,
    classify_paragraphs,
    observations_for_document,
    persist_observations,
    split_paragraphs,
)
from agent_memory.pipeline.chunks import (
    build_chunks,
    create_embeddings,
    delete_document_chunks,
    persist_chunks,
)
from agent_memory.pipeline.triage import triage_announcement
from agent_memory.pipeline.xref import (
    resolve_references,
    persist_cross_references,
)


_DOCUMENT_ERRORS = (OSError, ValueError, sqlite3.Error, httpx.HTTPError, ollama.ResponseError)
_TRIAGE_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "triage.md"


@dataclass
class IngestSummary:
    discovered: int
    selected: int
    done: int = 0
    skipped: int = 0
    failed: int = 0
    no_op: int = 0
    warmed_documents: int = 0
    references: int = 0
    resolved_references: int = 0
    llm_calls: int = 0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _published_in_range(
    announcement: Announcement,
    *,
    start_date: date | None,
    end_date: date | None,
    ticker: str | None,
) -> bool:
    published_date = announcement.published_at.date()
    return (
        (ticker is None or announcement.ticker == ticker.upper())
        and (start_date is None or published_date >= start_date)
        and (end_date is None or published_date <= end_date)
    )


def _sync_document_catalog(
    database_path: Path, announcements: list[Announcement]
) -> None:
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        for announcement in announcements:
            upsert_document(
                connection,
                doc_id=announcement.id,
                ticker=announcement.ticker,
                folder=announcement.folder.name,
                headline=announcement.headline,
                published_at=announcement.published_at.isoformat(),
                category=announcement.category,
                announcement_types=announcement.announcement_types,
                is_price_sensitive=announcement.is_price_sensitive,
                text_path=(
                    str(announcement.text_path)
                    if announcement.text_path is not None
                    else None
                ),
                text_source=announcement.text_source,
            )


def _has_hash_observations(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT DISTINCT doc_id FROM paragraph_hash_observation"
    ).fetchall()
    return {str(row["doc_id"]) for row in rows}


def _warm_hash_history(
    database_path: Path,
    announcements: list[Announcement],
    *,
    through_date: date,
) -> int:
    with database_connection(database_path) as connection:
        indexed_ids = _has_hash_observations(connection)
        index = BoilerplateIndex.from_connection(connection)

    warmed_documents = 0
    for announcement in announcements:
        if announcement.published_at.date() > through_date:
            break
        if announcement.text_path is None or announcement.id in indexed_ids:
            continue

        text = read_announcement_text(announcement)
        paragraphs = split_paragraphs(text)
        observations = observations_for_document(
            doc_id=announcement.id,
            ticker=announcement.ticker,
            published_at=announcement.published_at,
            paragraphs=paragraphs,
        )
        if not observations:
            continue
        with database_connection(database_path) as connection:
            inserted = persist_observations(connection, observations)
        index.add_observations(inserted)
        indexed_ids.add(announcement.id)
        warmed_documents += 1
    return warmed_documents


def _last_llm_call_id(database_path: Path) -> int:
    with database_connection(database_path) as connection:
        row = connection.execute("SELECT COALESCE(MAX(id), 0) AS id FROM llm_call").fetchone()
    return int(row["id"])


def _clear_ingest_range(
    database_path: Path,
    *,
    ticker: str | None,
    start_date: date,
    end_date: date | None,
) -> None:
    conditions = ["date(published_at) >= ?"]
    parameters: list[object] = [start_date.isoformat()]
    if end_date is not None:
        conditions.append("date(published_at) <= ?")
        parameters.append(end_date.isoformat())
    if ticker is not None:
        conditions.append("ticker = ?")
        parameters.append(ticker.upper())
    where_clause = " AND ".join(conditions)
    document_ids = f"SELECT id FROM document WHERE {where_clause}"

    with database_connection(database_path) as connection:
        vector_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vec'"
        ).fetchone()
        if vector_table is not None:
            connection.execute(
                f"""
                DELETE FROM chunk_vec
                WHERE chunk_id IN (
                    SELECT id FROM chunk
                    WHERE doc_id IN ({document_ids})
                )
                """,
                parameters,
            )
        connection.execute(
            f"DELETE FROM chunk WHERE doc_id IN ({document_ids})", parameters
        )
        connection.execute(
            f"DELETE FROM doc_xref WHERE src_doc IN ({document_ids})",
            parameters,
        )
        connection.execute(
            f"DELETE FROM paragraph_hash_observation WHERE doc_id IN ({document_ids})",
            parameters,
        )
        connection.execute(
            f"""
            UPDATE document
            SET triage_class = NULL,
                form_type = NULL,
                triage_reason = NULL,
                ingest_status = 'pending',
                ingested_at = NULL,
                error = NULL
            WHERE {where_clause}
            """,
            parameters,
        )
        rebuild_paragraph_hash_aggregates(connection)


def _mark_missing_text(database_path: Path, announcement: Announcement) -> None:
    with database_connection(database_path) as connection:
        connection.execute(
            """
            UPDATE document
            SET triage_class = 'skip',
                triage_reason = 'No markdown text is available.',
                ingest_status = 'skipped',
                ingested_at = ?,
                error = NULL
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), announcement.id),
        )


def _mark_document_error(
    database_path: Path, announcement: Announcement, error: Exception
) -> None:
    with database_connection(database_path) as connection:
        connection.execute(
            """
            UPDATE document
            SET ingest_status = 'error',
                ingested_at = NULL,
                error = ?
            WHERE id = ?
            """,
            (str(error)[:2000], announcement.id),
        )


def _process_document(
    database_path: Path,
    config: AppConfig,
    announcement: Announcement,
    announcements: list[Announcement],
    index: BoilerplateIndex,
    client: LLMClient,
) -> tuple[str, int, int]:
    if announcement.text_path is None:
        _mark_missing_text(database_path, announcement)
        return "skipped", 0, 0

    text = read_announcement_text(announcement)
    triage = triage_announcement(
        announcement,
        text,
        client,
        skip_headline_patterns=config.skip_headline_patterns,
        prompt_path=_TRIAGE_PROMPT_PATH,
    )
    paragraph_texts = split_paragraphs(text)
    paragraphs = classify_paragraphs(
        paragraph_texts,
        ticker=announcement.ticker,
        published_at=announcement.published_at,
        index=index,
    )
    observations = observations_for_document(
        doc_id=announcement.id,
        ticker=announcement.ticker,
        published_at=announcement.published_at,
        paragraphs=paragraph_texts,
    )
    if triage.triage_class == "skip":
        chunks = []
        resolution = None
    else:
        chunks = build_chunks(
            doc_id=announcement.id,
            ticker=announcement.ticker,
            published_at=announcement.published_at.isoformat(),
            paragraphs=paragraphs,
        )
        resolution = resolve_references(announcement, text, announcements)
    embeddings = create_embeddings(chunks, client, database_path)

    with database_connection(database_path) as connection:
        delete_document_chunks(connection, announcement.id)
        connection.execute("DELETE FROM doc_xref WHERE src_doc = ?", (announcement.id,))
        connection.execute(
            """
            UPDATE document
            SET triage_class = ?,
                form_type = ?,
                triage_reason = ?,
                ingest_status = 'done',
                ingested_at = ?,
                error = NULL
            WHERE id = ?
            """,
            (
                triage.triage_class,
                triage.form_type,
                triage.reason,
                datetime.now(timezone.utc).isoformat(),
                announcement.id,
            ),
        )
        inserted_observations = persist_observations(connection, observations)
        persist_chunks(connection, chunks, embeddings)
        if resolution is not None:
            persist_cross_references(
                connection, announcement.id, resolution.matches
            )

    index.add_observations(inserted_observations)
    if resolution is None:
        return "done", 0, 0
    return (
        "done",
        len(resolution.matches) + len(resolution.unresolved),
        len(resolution.matches),
    )


def run_ingest(
    config: AppConfig,
    *,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    reingest: bool = False,
    warmup: bool = False,
    client: LLMClient | None = None,
) -> IngestSummary:
    started = time.perf_counter()
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("--from must be on or before --to")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if reingest and start_date is None:
        raise ValueError("--reingest requires --from")
    if reingest and limit is not None:
        raise ValueError("--reingest cannot be combined with --limit")
    if warmup and reingest:
        raise ValueError("--warmup and --reingest cannot be used together")
    if warmup and start_date is not None:
        raise ValueError("--warmup starts at the beginning of the archive; omit --from")
    if warmup and limit is not None:
        raise ValueError("--warmup cannot be limited to a partial document set")

    announcements = discover_announcements(config.data_dir)
    if ticker is not None and ticker.upper() not in {
        announcement.ticker for announcement in announcements
    }:
        raise FileNotFoundError(f"No announcements found for ticker {ticker}")
    _sync_document_catalog(config.database_path, announcements)

    selected = [
        announcement
        for announcement in announcements
        if _published_in_range(
            announcement,
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
        )
    ]
    if limit is not None:
        selected = selected[:limit]
    summary = IngestSummary(discovered=len(announcements), selected=len(selected))
    before_call_id = _last_llm_call_id(config.database_path)

    if warmup:
        through_date = end_date or max(
            (announcement.published_at.date() for announcement in announcements),
            default=date.min,
        )
        summary.warmed_documents = _warm_hash_history(
            config.database_path,
            announcements,
            through_date=through_date,
        )
    elif selected:
        if reingest:
            _clear_ingest_range(
                config.database_path,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
        through_date = max(announcement.published_at.date() for announcement in selected)
        summary.warmed_documents = _warm_hash_history(
            config.database_path,
            announcements,
            through_date=through_date,
        )
        with database_connection(config.database_path) as connection:
            index = BoilerplateIndex.from_connection(connection)
            statuses = {
                row["id"]: row["ingest_status"]
                for row in connection.execute(
                    "SELECT id, ingest_status FROM document"
                ).fetchall()
            }
        pipeline_client = client or LLMClient(config, config.database_path)
        for announcement in selected:
            if statuses.get(announcement.id) in {"done", "skipped"}:
                summary.no_op += 1
                continue
            try:
                status, references, resolved = _process_document(
                    config.database_path,
                    config,
                    announcement,
                    announcements,
                    index,
                    pipeline_client,
                )
            except _DOCUMENT_ERRORS as exc:
                _mark_document_error(config.database_path, announcement, exc)
                summary.failed += 1
                summary.errors.append((announcement.id, str(exc)))
                continue
            if status == "skipped":
                summary.skipped += 1
            else:
                summary.done += 1
            summary.references += references
            summary.resolved_references += resolved

    after_call_id = _last_llm_call_id(config.database_path)
    summary.llm_calls = max(0, after_call_id - before_call_id)
    summary.elapsed_seconds = time.perf_counter() - started
    return summary
