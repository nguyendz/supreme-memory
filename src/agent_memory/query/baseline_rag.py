"""Boilerplate-inclusive chunk-RAG baseline with citation and PIT checks."""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient
from agent_memory.pipeline.chunks import (
    Chunk,
    ChunkSearchResult,
    create_embeddings,
    EMBEDDING_BATCH_SIZE,
    search_fts,
    search_vectors,
)
from agent_memory.pipeline.quotecheck import contains_exact_quote


RRF_CONSTANT = 60
RETRIEVAL_LIMIT = 10
MAX_CONTEXT_CHARACTERS = 24_000
CHARS_PER_TOKEN = 3
_ABSTENTION_PATTERN = re.compile(
    r"\b(?:"
    r"cannot\s+(?:answer|determine|confirm|find)|"
    r"can't\s+(?:answer|determine|confirm|find)|"
    r"unable\s+to\s+(?:answer|determine|confirm|find)|"
    r"not\s+enough\s+(?:information|evidence)|"
    r"insufficient\s+(?:information|evidence)|"
    r"no\s+(?:relevant|supporting)\s+(?:information|evidence|source)|"
    r"does\s+not\s+answer|"
    r"(?:do|does)\s+not\s+(?:contain|provide|include)|"
    r"(?:don't|doesn't)\s+(?:contain|provide|include)|"
    r"not\s+(?:stated|reported|available|found)|"
    r"i\s+don't\s+know"
    r")\b",
    re.IGNORECASE,
)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation]


@dataclass(frozen=True)
class BaselineIndexStats:
    source_chunks: int
    embedded_chunks: int
    dimensions: int | None


@dataclass(frozen=True)
class BaselineAnswer:
    question: str
    knowledge_date: date
    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunks: tuple[ChunkSearchResult, ...]
    context_tokens: int
    prompt_tokens: int
    output_tokens: int
    attempts: int
    wall_duration_ms: float


def build_baseline_index(
    source_database_path: Path,
    baseline_database_path: Path,
    client: LLMClient,
) -> BaselineIndexStats:
    """Clone a corpus into an isolated database and embed every chunk."""
    source_path = source_database_path.expanduser().resolve()
    baseline_path = baseline_database_path.expanduser().resolve()
    if source_path == baseline_path:
        raise ValueError("The baseline database must be separate from the source database")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source database not found: {source_path}")

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with database_connection(source_path) as source_connection:
        with sqlite3.connect(baseline_path) as baseline_connection:
            source_connection.backup(baseline_connection)
    initialize_database(baseline_path)

    with database_connection(baseline_path) as connection:
        vector_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vec'"
        ).fetchone()
        if vector_table is not None:
            connection.execute("DELETE FROM chunk_vec")
        rows = connection.execute(
            """
            SELECT id, doc_id, ticker, published_at, ord, text, is_boilerplate
            FROM chunk
            ORDER BY doc_id, ord
            """
        ).fetchall()
        chunks = [
            Chunk(
                id=row["id"],
                doc_id=row["doc_id"],
                ticker=row["ticker"],
                published_at=row["published_at"],
                ord=int(row["ord"]),
                text=row["text"],
                is_boilerplate=bool(row["is_boilerplate"]),
            )
            for row in rows
        ]

    if not chunks:
        return BaselineIndexStats(source_chunks=0, embedded_chunks=0, dimensions=None)

    embedded_chunks = 0
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        embeddings = create_embeddings(
            batch,
            client,
            baseline_path,
            batch_size=EMBEDDING_BATCH_SIZE,
            include_boilerplate=True,
        )
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Baseline index embedded {len(embeddings)} of {len(batch)} chunks "
                f"in batch starting at {start}"
            )
        with database_connection(baseline_path) as connection:
            connection.executemany(
                "INSERT INTO chunk_vec (embedding, chunk_id) VALUES (?, ?)",
                [(embedding, chunk_id) for chunk_id, embedding in embeddings.items()],
            )
        embedded_chunks += len(embeddings)
    if embedded_chunks != len(chunks):
        raise RuntimeError(
            f"Baseline index embedded {embedded_chunks} of {len(chunks)} chunks"
        )
    with database_connection(baseline_path) as connection:
        dimensions_row = connection.execute(
            "SELECT dimensions FROM vector_metadata WHERE name = 'default'"
        ).fetchone()
        dimensions = (
            int(dimensions_row["dimensions"]) if dimensions_row is not None else None
        )
    return BaselineIndexStats(
        source_chunks=len(chunks),
        embedded_chunks=embedded_chunks,
        dimensions=dimensions,
    )


def _reciprocal_rank_fusion(
    lexical: Sequence[ChunkSearchResult],
    vector: Sequence[ChunkSearchResult],
) -> list[ChunkSearchResult]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, ChunkSearchResult] = {}
    for results in (lexical, vector):
        for rank, result in enumerate(results, start=1):
            scores[result.id] += 1 / (RRF_CONSTANT + rank)
            chunks[result.id] = result
    return sorted(
        chunks.values(),
        key=lambda item: (
            -scores[item.id],
            item.published_at,
            item.doc_id,
            item.ord,
        ),
    )


def _pack_context(
    chunks: Sequence[ChunkSearchResult],
) -> tuple[str, tuple[ChunkSearchResult, ...]]:
    blocks: list[str] = []
    selected: list[ChunkSearchResult] = []
    context_length = 0
    for chunk in chunks:
        block = (
            f"[document_id={chunk.doc_id}; published_at={chunk.published_at}; "
            f"ticker={chunk.ticker}]\n{chunk.text}"
        )
        added_length = len(block) + (2 if blocks else 0)
        if context_length + added_length > MAX_CONTEXT_CHARACTERS:
            continue
        blocks.append(block)
        selected.append(chunk)
        context_length += added_length
    return "\n\n".join(blocks), tuple(selected)


def retrieve_baseline_chunks(
    database_path: Path,
    query: str,
    query_vector: Sequence[float],
    *,
    knowledge_date: date,
    ticker: str,
    limit: int = RETRIEVAL_LIMIT,
) -> tuple[ChunkSearchResult, ...]:
    if not query.strip():
        raise ValueError("query must not be blank")
    if limit <= 0:
        raise ValueError("limit must be positive")
    with database_connection(database_path) as connection:
        vector_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vec'"
        ).fetchone()
        if vector_table is None:
            raise RuntimeError("Baseline vector index is not initialized")
        chunk_count = int(
            connection.execute("SELECT COUNT(*) AS count FROM chunk").fetchone()["count"]
        )
        vector_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM chunk_vec"
            ).fetchone()["count"]
        )
        if vector_count != chunk_count:
            raise RuntimeError(
                f"Baseline vector index is incomplete: "
                f"{vector_count}/{chunk_count} chunks embedded"
            )
        lexical = search_fts(
            connection,
            query,
            limit=limit,
            include_boilerplate=True,
            ticker=ticker,
            knowledge_date=knowledge_date,
        )
        vector = search_vectors(
            connection,
            query_vector,
            limit=limit,
            include_boilerplate=True,
            ticker=ticker,
            knowledge_date=knowledge_date,
        )
    return tuple(_reciprocal_rank_fusion(lexical, vector))


def answer_baseline(
    database_path: Path,
    client: LLMClient,
    question: str,
    *,
    knowledge_date: date,
    ticker: str,
    prompt_path: Path | None = None,
    model: str | None = None,
) -> BaselineAnswer:
    """Answer from a retrieved context and reject unsupported citations."""
    if not question.strip():
        raise ValueError("question must not be blank")
    started = time.perf_counter()
    embedding = client.embed([question])
    if len(embedding.vectors) != 1 or not embedding.vectors[0]:
        raise ValueError("Question embedding returned no vector")
    retrieved = retrieve_baseline_chunks(
        database_path,
        question,
        embedding.vectors[0],
        knowledge_date=knowledge_date,
        ticker=ticker,
    )
    context, context_chunks = _pack_context(retrieved)
    prompt_file = prompt_path or Path(__file__).resolve().parents[3] / "prompts" / "answer.md"
    system_prompt = prompt_file.read_text(encoding="utf-8")
    prompt = (
        f"Knowledge date (inclusive): {knowledge_date.isoformat()}\n"
        f"Ticker: {ticker.upper()}\n"
        f"Question: {question}\n\n"
        "Retrieved source chunks:\n"
        f"{context or '(no source chunks were retrieved)'}"
    )
    chunks_by_document: dict[str, list[ChunkSearchResult]] = defaultdict(list)
    for chunk in context_chunks:
        chunks_by_document[chunk.doc_id].append(chunk)
    prompt_tokens = 0
    output_tokens = 0
    attempts = 0
    result = None
    previous_raw_content: str | None = None
    validation_error = "Baseline answer failed citation validation"
    for semantic_attempt in (1, 2):
        current_prompt = prompt
        if semantic_attempt == 2:
            if previous_raw_content is None:
                raise AssertionError("A semantic retry requires a prior response")
            current_prompt += (
                "\n\nYour previous JSON failed citation validation: "
                f"{validation_error}\nPrevious JSON: {previous_raw_content}\n"
                "Correct the response. Use a document_id exactly as shown in "
                "the context, never a chunk identifier. Copy the quote exactly."
            )
        result = client.chat_json(
            current_prompt,
            AnswerResponse,
            system_prompt=system_prompt,
            model=model,
        )
        prompt_tokens += result.prompt_tokens
        output_tokens += result.output_tokens
        attempts += result.attempts
        previous_raw_content = result.raw_content
        validation_error = _citation_validation_error(
            result.value,
            chunks_by_document,
            knowledge_date=knowledge_date,
        )
        if validation_error is None:
            break
    if validation_error is not None or result is None:
        answer_excerpt = (
            " ".join(result.value.answer.split())[:300]
            if result is not None
            else ""
        )
        raise ValueError(
            f"{validation_error or 'Baseline answer was not produced'}; "
            f"last answer: {answer_excerpt!r}"
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return BaselineAnswer(
        question=question,
        knowledge_date=knowledge_date,
        answer=result.value.answer,
        citations=tuple(result.value.citations),
        retrieved_chunks=context_chunks,
        context_tokens=math.ceil(len(context) / CHARS_PER_TOKEN),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        attempts=attempts,
        wall_duration_ms=elapsed_ms,
    )


def _citation_validation_error(
    response: AnswerResponse,
    chunks_by_document: dict[str, list[ChunkSearchResult]],
    *,
    knowledge_date: date,
) -> str | None:
    if not response.citations and not _ABSTENTION_PATTERN.search(response.answer):
        return "answer has no citations and is not an abstention"
    for citation in response.citations:
        source_chunks = chunks_by_document.get(citation.document_id, [])
        if not source_chunks:
            return (
                "cited document is outside the retrieved context: "
                f"{citation.document_id}"
            )
        if not any(
            contains_exact_quote(citation.quote, chunk.text)
            for chunk in source_chunks
        ):
            return (
                "citation quote is not present in its retrieved source: "
                f"{citation.document_id}"
            )
        if any(
            date.fromisoformat(chunk.published_at[:10]) > knowledge_date
            for chunk in source_chunks
        ):
            return (
                "cited document is later than knowledge_date: "
                f"{citation.document_id}"
            )
    return None
