"""Paragraph chunking, local embeddings, and chunk search."""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

from agent_memory.db import initialize_database
from agent_memory.llm import LLMClient
from agent_memory.pipeline.boilerplate import Paragraph


MAX_CHUNK_CHARACTERS = 2000
EMBEDDING_BATCH_SIZE = 32


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    ticker: str
    published_at: str
    ord: int
    text: str
    is_boilerplate: bool


@dataclass(frozen=True)
class ChunkSearchResult:
    id: str
    doc_id: str
    ticker: str
    published_at: str
    ord: int
    text: str
    distance: float | None = None


def _pack_text(text: str, max_characters: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        pieces = [
            word[index : index + max_characters]
            for index in range(0, len(word), max_characters)
        ]
        for piece in pieces:
            added_length = len(piece) + (1 if current else 0)
            if current and current_length + added_length > max_characters:
                chunks.append(" ".join(current))
                current = []
                current_length = 0
                added_length = len(piece)
            current.append(piece)
            current_length += added_length
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_oversized_paragraph(
    text: str, *, max_characters: int = MAX_CHUNK_CHARACTERS
) -> list[str]:
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    paragraph = text.strip()
    if len(paragraph) <= max_characters:
        return [paragraph] if paragraph else []

    fragments: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if len(sentence) <= max_characters:
            fragments.append(sentence)
        else:
            fragments.extend(_pack_text(sentence, max_characters))

    result: list[str] = []
    current = ""
    for fragment in fragments:
        candidate = f"{current} {fragment}".strip() if current else fragment
        if current and len(candidate) > max_characters:
            result.append(current)
            current = fragment
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def build_chunks(
    *,
    doc_id: str,
    ticker: str,
    published_at: str,
    paragraphs: Sequence[Paragraph],
    max_characters: int = MAX_CHUNK_CHARACTERS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        for text in split_oversized_paragraph(
            paragraph.text, max_characters=max_characters
        ):
            ord_value = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{ord_value:05d}",
                    doc_id=doc_id,
                    ticker=ticker,
                    published_at=published_at,
                    ord=ord_value,
                    text=text,
                    is_boilerplate=paragraph.is_boilerplate,
                )
            )
    return chunks


def pack_vector(vector: Sequence[float]) -> sqlite3.Binary:
    return sqlite3.Binary(struct.pack(f"<{len(vector)}f", *vector))


def create_embeddings(
    chunks: Sequence[Chunk],
    client: LLMClient,
    database_path: Path,
    *,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    include_boilerplate: bool = False,
) -> dict[str, sqlite3.Binary]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidates = [
        chunk
        for chunk in chunks
        if include_boilerplate or not chunk.is_boilerplate
    ]
    embeddings: dict[str, sqlite3.Binary] = {}
    dimensions: int | None = None

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        result = client.embed(
            [chunk.text for chunk in batch],
            doc_id=batch[0].doc_id,
        )
        if dimensions is not None and result.dimensions != dimensions:
            raise ValueError(
                "Embedding dimensions changed within one document: "
                f"{dimensions} then {result.dimensions}"
            )
        dimensions = result.dimensions
        initialize_database(database_path, vector_dimensions=dimensions)
        embeddings.update(
            (chunk.id, pack_vector(vector))
            for chunk, vector in zip(batch, result.vectors, strict=True)
        )
    return embeddings


def delete_document_chunks(connection: sqlite3.Connection, doc_id: str) -> None:
    vector_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vec'"
    ).fetchone()
    if vector_table is not None:
        connection.execute(
            "DELETE FROM chunk_vec WHERE chunk_id IN (SELECT id FROM chunk WHERE doc_id = ?)",
            (doc_id,),
        )
    connection.execute("DELETE FROM chunk WHERE doc_id = ?", (doc_id,))


def persist_chunks(
    connection: sqlite3.Connection,
    chunks: Sequence[Chunk],
    embeddings: Mapping[str, sqlite3.Binary],
) -> None:
    for chunk in chunks:
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text, is_boilerplate
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id,
                chunk.doc_id,
                chunk.ticker,
                chunk.published_at,
                chunk.ord,
                chunk.text,
                int(chunk.is_boilerplate),
            ),
        )
        embedding = embeddings.get(chunk.id)
        if embedding is not None:
            connection.execute(
                "INSERT INTO chunk_vec (embedding, chunk_id) VALUES (?, ?)",
                (embedding, chunk.id),
            )


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w$%]+", query, flags=re.UNICODE)
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _search_filters(
    *,
    include_boilerplate: bool,
    ticker: str | None,
    knowledge_date: date | str | None,
) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    parameters: list[object] = []
    if not include_boilerplate:
        conditions.append("chunk.is_boilerplate = 0")
    if ticker is not None:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("ticker must not be blank")
        conditions.append("chunk.ticker = ?")
        parameters.append(normalized_ticker)
    if knowledge_date is not None:
        date_value = (
            knowledge_date.isoformat()
            if isinstance(knowledge_date, date)
            else date.fromisoformat(knowledge_date).isoformat()
        )
        conditions.append("date(chunk.published_at) <= date(?)")
        parameters.append(date_value)
    return conditions, parameters


def search_fts(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
    include_boilerplate: bool = False,
    ticker: str | None = None,
    knowledge_date: date | str | None = None,
) -> list[ChunkSearchResult]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    match_query = _fts_query(query)
    if not match_query:
        return []
    filters, parameters = _search_filters(
        include_boilerplate=include_boilerplate,
        ticker=ticker,
        knowledge_date=knowledge_date,
    )
    where_clause = " AND ".join(["chunk_fts MATCH ?", *filters])
    rows = connection.execute(
        f"""
        SELECT chunk.id, chunk.doc_id, chunk.ticker, chunk.published_at,
               chunk.ord, chunk.text
        FROM chunk_fts
        JOIN chunk ON chunk.rowid = chunk_fts.rowid
        WHERE {where_clause}
        ORDER BY bm25(chunk_fts), chunk.published_at, chunk.ord
        LIMIT ?
        """,
        [match_query, *parameters, limit],
    ).fetchall()
    return [
        ChunkSearchResult(
            id=row["id"],
            doc_id=row["doc_id"],
            ticker=row["ticker"],
            published_at=row["published_at"],
            ord=row["ord"],
            text=row["text"],
        )
        for row in rows
    ]


def search_vectors(
    connection: sqlite3.Connection,
    query_vector: Sequence[float],
    *,
    limit: int = 5,
    include_boilerplate: bool = False,
    ticker: str | None = None,
    knowledge_date: date | str | None = None,
) -> list[ChunkSearchResult]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    vector_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vec'"
    ).fetchone()
    if vector_table is None:
        raise RuntimeError("Vector tables are not initialized")
    vector_count = int(
        connection.execute("SELECT COUNT(*) AS count FROM chunk_vec").fetchone()["count"]
    )
    if not vector_count:
        return []
    filters, filter_parameters = _search_filters(
        include_boilerplate=include_boilerplate,
        ticker=ticker,
        knowledge_date=knowledge_date,
    )
    where_clause = " AND ".join(filters) if filters else "1 = 1"
    candidate_limit = min(vector_count, max(limit * 4, limit))
    packed_query = pack_vector(query_vector)

    while True:
        candidates = connection.execute(
            """
            SELECT chunk_id, distance
            FROM chunk_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (packed_query, candidate_limit),
        ).fetchall()
        if not candidates:
            return []
        candidate_ids = [candidate["chunk_id"] for candidate in candidates]
        rows: list[sqlite3.Row] = []
        for start in range(0, len(candidate_ids), 500):
            batch_ids = candidate_ids[start : start + 500]
            placeholders = ", ".join("?" for _ in batch_ids)
            rows.extend(
                connection.execute(
                    f"""
                    SELECT id, doc_id, ticker, published_at, ord, text
                    FROM chunk
                    WHERE {where_clause} AND id IN ({placeholders})
                    """,
                    [*filter_parameters, *batch_ids],
                ).fetchall()
            )
        chunks_by_id = {row["id"]: row for row in rows}
        results = [
            ChunkSearchResult(
                id=candidate["chunk_id"],
                doc_id=chunks_by_id[candidate["chunk_id"]]["doc_id"],
                ticker=chunks_by_id[candidate["chunk_id"]]["ticker"],
                published_at=chunks_by_id[candidate["chunk_id"]]["published_at"],
                ord=chunks_by_id[candidate["chunk_id"]]["ord"],
                text=chunks_by_id[candidate["chunk_id"]]["text"],
                distance=float(candidate["distance"]),
            )
            for candidate in candidates
            if candidate["chunk_id"] in chunks_by_id
        ]
        if len(results) >= limit or candidate_limit >= vector_count:
            return results[:limit]
        candidate_limit = min(vector_count, candidate_limit * 2)
