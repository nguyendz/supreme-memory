"""Point-in-time retrieval and source packing for memory queries."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient
from agent_memory.pipeline.chunks import (
    ChunkSearchResult,
    pack_vector,
    search_fts,
    search_vectors,
)
from agent_memory.pipeline.quotecheck import check_quote, normalize_quote_text
from agent_memory.query.pit import (
    current_facts_at,
    fact_sources_at,
    state_at,
    timeline_at,
    visible_facts_at,
)


RRF_CONSTANT = 60
DEFAULT_FACT_LIMIT = 10
DEFAULT_CHUNK_LIMIT = 5


@dataclass(frozen=True)
class MemorySource:
    citation_id: str
    fact_id: str | None
    document_id: str
    headline: str
    published_at: str
    quote: str
    source_path: str | None
    action: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class FactEvidence:
    fact_id: str
    ticker: str
    entity_id: str | None
    entity_name: str | None
    attribute: str | None
    value: str | None
    value_num: float | None
    unit: str | None
    qualifier: str | None
    as_of: str | None
    period: str | None
    fact_type: str
    tier: str
    salience: float | None
    text: str
    key_hash: str | None
    first_visible_at: str
    confirmation_count: int
    sources: tuple[MemorySource, ...]


@dataclass(frozen=True)
class ChunkEvidence:
    citation_id: str
    chunk_id: str
    document_id: str
    ticker: str
    published_at: str
    headline: str
    source_path: str | None
    text: str
    distance: float | None = None

    @property
    def quote(self) -> str:
        return self.text


@dataclass(frozen=True)
class MemoryRetrieval:
    facts: tuple[FactEvidence, ...]
    chunks: tuple[ChunkEvidence, ...] = ()
    fallback_used: bool = False


@dataclass(frozen=True)
class QueryEntityChoice:
    id: str
    name: str
    type: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class QueryAttributeChoice:
    name: str
    description: str


@dataclass(frozen=True)
class QueryChoices:
    entities: tuple[QueryEntityChoice, ...]
    attributes: tuple[QueryAttributeChoice, ...]


def _document_metadata(
    connection: sqlite3.Connection,
    document_id: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT id, ticker, headline, published_at, text_path
        FROM document WHERE id = ?
        """,
        (document_id,),
    ).fetchone()


def _source_document_text(
    connection: sqlite3.Connection,
    document_id: str,
    cache: dict[str, str],
) -> str:
    if document_id not in cache:
        rows = connection.execute(
            """
            SELECT text FROM chunk
            WHERE doc_id = ? AND is_boilerplate = 0
            ORDER BY ord
            """,
            (document_id,),
        ).fetchall()
        cache[document_id] = "\n\n".join(str(row["text"]) for row in rows)
    return cache[document_id]


def _fact_evidence(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
    knowledge_date: date,
) -> tuple[FactEvidence, ...]:
    result: list[FactEvidence] = []
    document_text_cache: dict[str, str] = {}
    citation_number = 0
    for row in rows:
        visible_sources = fact_sources_at(
            connection,
            str(row["id"]),
            knowledge_date,
        )
        sources: list[MemorySource] = []
        for source in visible_sources:
            metadata = _document_metadata(connection, str(source["doc_id"]))
            if metadata is None:
                continue
            if date.fromisoformat(str(metadata["published_at"])[:10]) > knowledge_date:
                continue
            quote = str(source["quote"])
            source_text = _source_document_text(
                connection,
                str(source["doc_id"]),
                document_text_cache,
            )
            if not check_quote(quote, source_text).passed:
                continue
            citation_number += 1
            sources.append(
                MemorySource(
                    citation_id=f"c{citation_number}",
                    fact_id=str(row["id"]),
                    document_id=str(source["doc_id"]),
                    headline=str(metadata["headline"]),
                    published_at=str(source["published_at"]),
                    quote=quote,
                    source_path=metadata["text_path"],
                    action=str(source["action"]),
                    role=str(source["role"]),
                )
            )
        if not sources:
            continue
        entity_name = None
        if row["entity_id"] is not None:
            entity = connection.execute(
                "SELECT canonical_name FROM entity WHERE id = ?",
                (row["entity_id"],),
            ).fetchone()
            if entity is not None:
                entity_name = str(entity["canonical_name"])
        confirmation_count = sum(
            source.action in {"CONFIRM", "SUPERSEDE"} for source in sources
        )
        result.append(
            FactEvidence(
                fact_id=str(row["id"]),
                ticker=str(row["ticker"]),
                entity_id=row["entity_id"],
                entity_name=entity_name,
                attribute=row["attribute"],
                value=row["value"],
                value_num=row["value_num"],
                unit=row["unit"],
                qualifier=row["qualifier"],
                as_of=row["as_of"],
                period=row["period"],
                fact_type=str(row["fact_type"]),
                tier=str(row["tier"]),
                salience=row["salience"],
                text=str(row["text"]),
                key_hash=row["key_hash"],
                first_visible_at=str(row["first_visible_at"]),
                confirmation_count=confirmation_count,
                sources=tuple(sources),
            )
        )
    return tuple(result)


def fact_matches_subject(
    fact: FactEvidence,
    *,
    subject_date: date | None,
    subject_period: str | None,
) -> bool:
    if subject_period is not None:
        if fact.period is None or fact.period.casefold() != subject_period.casefold():
            return False
    if subject_date is None:
        return True
    if fact.as_of is not None:
        return str(fact.as_of) == subject_date.isoformat()
    if fact.period is not None:
        period_match = re.fullmatch(r"(\d{4})-Q([1-4])", fact.period.strip(), re.I)
        if period_match:
            year = int(period_match.group(1))
            quarter = int(period_match.group(2))
            month_start = (quarter - 1) * 3 + 1
            month_end = month_start + 2
            return (
                subject_date.year == year
                and month_start <= subject_date.month <= month_end
            )
    date_forms = (
        subject_date.strftime("%d %B %Y"),
        subject_date.strftime("%d %b %Y"),
        subject_date.isoformat(),
    )
    haystack = normalize_quote_text(
        " ".join(
            [
                fact.text,
                *(source.quote for source in fact.sources),
            ]
        )
    )
    return any(normalize_quote_text(form) in haystack for form in date_forms)


def retrieve_state(
    database_path: Path,
    *,
    ticker: str,
    entity_id: str,
    attribute: str,
    knowledge_date: date,
    subject_date: date | None = None,
    subject_period: str | None = None,
) -> tuple[FactEvidence, ...]:
    """Retrieve current observations and source links for an entity attribute."""
    with database_connection(database_path) as connection:
        rows = state_at(
            connection,
            ticker=ticker,
            entity_id=entity_id,
            attribute=attribute,
            knowledge_time=knowledge_date,
        )
        facts = _fact_evidence(connection, rows, knowledge_date)
    return tuple(
        fact
        for fact in facts
        if fact_matches_subject(
            fact,
            subject_date=subject_date,
            subject_period=subject_period,
        )
    )


def retrieve_timeline(
    database_path: Path,
    *,
    ticker: str,
    entity_id: str,
    attribute: str,
    knowledge_date: date,
) -> tuple[FactEvidence, ...]:
    """Retrieve visible history, including facts later superseded by T."""
    with database_connection(database_path) as connection:
        rows = timeline_at(
            connection,
            ticker=ticker,
            entity_id=entity_id,
            attribute=attribute,
            knowledge_time=knowledge_date,
        )
        return _fact_evidence(connection, rows, knowledge_date)


def retrieve_summary_facts(
    database_path: Path,
    *,
    ticker: str,
    knowledge_date: date,
    start_date: date | None = None,
    end_date: date | None = None,
    include_t3: bool = False,
) -> tuple[FactEvidence, ...]:
    """Select PIT-visible facts for a summary window."""
    with database_connection(database_path) as connection:
        rows = visible_facts_at(
            connection,
            knowledge_date,
            ticker=ticker,
        )
        facts = _fact_evidence(connection, rows, knowledge_date)
    selected: list[FactEvidence] = []
    for fact in facts:
        if fact.tier == "T3" and not include_t3:
            continue
        subject_dates: list[date] = []
        if fact.as_of:
            try:
                subject_dates.append(date.fromisoformat(str(fact.as_of)))
            except ValueError:
                pass
        for source in fact.sources:
            subject_dates.append(date.fromisoformat(source.published_at[:10]))
        if start_date is not None and end_date is not None:
            if not any(start_date <= value <= end_date for value in subject_dates):
                continue
        elif start_date is not None and not any(
            value >= start_date for value in subject_dates
        ):
            continue
        elif end_date is not None and not any(
            value <= end_date for value in subject_dates
        ):
            continue
        selected.append(fact)
    return tuple(selected)


def available_query_choices(
    database_path: Path,
    *,
    ticker: str,
    knowledge_date: date,
) -> QueryChoices:
    """Return only entities and attributes supported by facts visible at T."""
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("ticker must not be blank")
    with database_connection(database_path) as connection:
        visible = visible_facts_at(
            connection,
            knowledge_date,
            ticker=normalized_ticker,
        )
        entity_ids = sorted(
            {str(row["entity_id"]) for row in visible if row["entity_id"] is not None}
        )
        entity_rows: list[sqlite3.Row] = []
        aliases_by_entity: dict[str, list[str]] = {}
        for start in range(0, len(entity_ids), 500):
            batch = entity_ids[start : start + 500]
            if not batch:
                continue
            placeholders = ", ".join("?" for _ in batch)
            entity_rows.extend(
                connection.execute(
                    f"""
                    SELECT id, canonical_name, type
                    FROM entity WHERE ticker = ? AND id IN ({placeholders})
                    ORDER BY canonical_name, id
                    """,
                    [normalized_ticker, *batch],
                ).fetchall()
            )
            aliases = connection.execute(
                f"""
                SELECT entity_id, alias
                FROM entity_alias
                WHERE date(first_seen_at) <= date(?)
                  AND entity_id IN ({placeholders})
                ORDER BY alias, entity_id
                """,
                [knowledge_date.isoformat(), *batch],
            ).fetchall()
            for alias in aliases:
                aliases_by_entity.setdefault(str(alias["entity_id"]), []).append(
                    str(alias["alias"])
                )

        attribute_names = sorted(
            {str(row["attribute"]) for row in visible if row["attribute"] is not None}
        )
        attributes: list[QueryAttributeChoice] = []
        for start in range(0, len(attribute_names), 500):
            batch = attribute_names[start : start + 500]
            placeholders = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT name, description, status, merged_into
                FROM attribute
                WHERE name IN ({placeholders})
                ORDER BY name
                """,
                batch,
            ).fetchall()
            for row in rows:
                if row["status"] == "active" or row["merged_into"] is not None:
                    attributes.append(
                        QueryAttributeChoice(
                            name=str(row["name"]),
                            description=str(row["description"]),
                        )
                    )

    return QueryChoices(
        entities=tuple(
            QueryEntityChoice(
                id=str(row["id"]),
                name=str(row["canonical_name"]),
                type=str(row["type"]),
                aliases=tuple(aliases_by_entity.get(str(row["id"]), ())),
            )
            for row in entity_rows
        ),
        attributes=tuple(attributes),
    )


def _fts_expression(query: str) -> str:
    terms = re.findall(r"[\w$%]+", query, flags=re.UNICODE)
    return " AND ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
    )


def _lexical_fact_ids(
    connection: sqlite3.Connection,
    query: str,
    *,
    ticker: str,
    limit: int,
) -> list[str]:
    expression = _fts_expression(query)
    if not expression:
        return []
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'fact_fts'"
    ).fetchone()
    if table_exists is None:
        return []
    rows = connection.execute(
        """
        SELECT fact_fts.fact_id
        FROM fact_fts
        JOIN fact AS f ON f.id = fact_fts.fact_id
        WHERE fact_fts MATCH ? AND f.ticker = ?
        ORDER BY bm25(fact_fts), fact_fts.fact_id
        LIMIT ?
        """,
        (expression, ticker.upper(), limit),
    ).fetchall()
    return [str(row["fact_id"]) for row in rows]


def _vector_fact_ids(
    database_path: Path,
    client: LLMClient,
    query: str,
    *,
    ticker: str,
    knowledge_date: date,
    limit: int,
    current_only: bool,
) -> list[str]:
    with database_connection(database_path) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'fact_vec'"
        ).fetchone()
        if table_exists is None:
            return []
        vector_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM fact_vec"
            ).fetchone()["count"]
        )
    if vector_count == 0:
        return []
    embedding = client.embed([query])
    if len(embedding.vectors) != 1 or not embedding.vectors[0]:
        raise ValueError("Fact search embedding returned no query vector")
    initialize_database(database_path, vector_dimensions=embedding.dimensions)
    packed_query = pack_vector(embedding.vectors[0])
    with database_connection(database_path) as connection:
        visible_facts = (
            current_facts_at(
                connection,
                knowledge_date,
                ticker=ticker,
            )
            if current_only
            else visible_facts_at(
                connection,
                knowledge_date,
                ticker=ticker,
            )
        )
        allowed_ids = {str(row["id"]) for row in visible_facts}
        candidate_limit = min(vector_count, max(limit * 4, limit))
        results: list[str] = []
        while True:
            candidates = connection.execute(
                """
                SELECT fact_id, distance
                FROM fact_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (packed_query, candidate_limit),
            ).fetchall()
            results = [
                str(row["fact_id"])
                for row in candidates
                if str(row["fact_id"]) in allowed_ids
            ]
            if len(results) >= limit or candidate_limit >= vector_count:
                return results[:limit]
            candidate_limit = min(vector_count, candidate_limit * 2)


def retrieve_facts(
    database_path: Path,
    client: LLMClient,
    query: str,
    *,
    ticker: str,
    knowledge_date: date,
    limit: int = DEFAULT_FACT_LIMIT,
    current_only: bool = True,
    include_t3: bool = False,
) -> tuple[FactEvidence, ...]:
    """Hybrid lexical/vector fact search restricted to PIT-visible sources."""
    if not query.strip():
        raise ValueError("Fact search query must not be blank")
    if limit <= 0:
        raise ValueError("limit must be positive")
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("ticker must not be blank")

    with database_connection(database_path) as connection:
        facts = (
            current_facts_at(
                connection,
                knowledge_date,
                ticker=normalized_ticker,
            )
            if current_only
            else visible_facts_at(
                connection,
                knowledge_date,
                ticker=normalized_ticker,
            )
        )
        facts_by_id = {str(row["id"]): row for row in facts}
        lexical_ids = _lexical_fact_ids(
            connection,
            query,
            ticker=normalized_ticker,
            limit=max(limit * 4, limit),
        )
    vector_ids = _vector_fact_ids(
        database_path,
        client,
        query,
        ticker=normalized_ticker,
        knowledge_date=knowledge_date,
        limit=max(limit * 4, limit),
        current_only=current_only,
    )
    ranks: dict[str, float] = {}
    for result_ids in (lexical_ids, vector_ids):
        for rank, fact_id in enumerate(result_ids, start=1):
            if fact_id in facts_by_id:
                ranks[fact_id] = ranks.get(fact_id, 0.0) + 1 / (
                    RRF_CONSTANT + rank
                )
    selected_ids = sorted(ranks, key=lambda fact_id: (-ranks[fact_id], fact_id))[
        :limit
    ]
    with database_connection(database_path) as connection:
        evidence = _fact_evidence(
            connection,
            [facts_by_id[fact_id] for fact_id in selected_ids],
            knowledge_date,
        )
    return tuple(
        fact
        for fact in evidence
        if include_t3 or fact.tier != "T3"
    )


def retrieve_chunks(
    database_path: Path,
    client: LLMClient,
    query: str,
    *,
    ticker: str,
    knowledge_date: date,
    limit: int = DEFAULT_CHUNK_LIMIT,
) -> tuple[ChunkEvidence, ...]:
    """Retrieve non-boilerplate chunks for the explicit memory fallback route."""
    if not query.strip():
        raise ValueError("Chunk search query must not be blank")
    if limit <= 0:
        raise ValueError("limit must be positive")
    embedding = client.embed([query])
    if len(embedding.vectors) != 1 or not embedding.vectors[0]:
        raise ValueError("Chunk search embedding returned no query vector")
    with database_connection(database_path) as connection:
        lexical = search_fts(
            connection,
            query,
            limit=limit * 3,
            include_boilerplate=False,
            ticker=ticker,
            knowledge_date=knowledge_date,
        )
        vector = search_vectors(
            connection,
            embedding.vectors[0],
            limit=limit * 3,
            include_boilerplate=False,
            ticker=ticker,
            knowledge_date=knowledge_date,
        )
        ranks: dict[str, float] = {}
        chunks_by_id: dict[str, ChunkSearchResult] = {}
        for result in lexical:
            chunks_by_id[result.id] = result
        for result in vector:
            chunks_by_id[result.id] = result
        for results in (lexical, vector):
            for rank, result in enumerate(results, start=1):
                ranks[result.id] = ranks.get(result.id, 0.0) + 1 / (
                    RRF_CONSTANT + rank
                )
        selected = sorted(ranks, key=lambda chunk_id: (-ranks[chunk_id], chunk_id))[
            :limit
        ]
        citations: list[ChunkEvidence] = []
        for index, chunk_id in enumerate(selected, start=1):
            chunk = chunks_by_id[chunk_id]
            metadata = _document_metadata(connection, chunk.doc_id)
            if metadata is None:
                continue
            citations.append(
                ChunkEvidence(
                    citation_id=f"c{index}",
                    chunk_id=chunk.id,
                    document_id=chunk.doc_id,
                    ticker=chunk.ticker,
                    published_at=chunk.published_at,
                    headline=str(metadata["headline"]),
                    source_path=metadata["text_path"],
                    text=chunk.text,
                    distance=chunk.distance,
                )
            )
        return tuple(citations)
