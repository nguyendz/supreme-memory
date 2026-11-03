"""Point-in-time fact visibility, state, and timeline queries."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime


KnowledgeTime = date | datetime | str


def _time_predicate(
    column: str,
    knowledge_time: KnowledgeTime,
    *,
    strictly_before: bool,
) -> tuple[str, str]:
    if isinstance(knowledge_time, datetime):
        return (
            f"julianday({column}) {'<' if strictly_before else '<='} "
            "julianday(?)",
            knowledge_time.isoformat(),
        )
    if isinstance(knowledge_time, date):
        if strictly_before:
            return f"date({column}) < date(?)", knowledge_time.isoformat()
        return f"date({column}) <= date(?)", knowledge_time.isoformat()
    value = knowledge_time.strip()
    if not value:
        raise ValueError("knowledge_time must not be blank")
    if "T" in value or " " in value:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("knowledge_time must be an ISO date or timestamp") from exc
        return (
            f"julianday({column}) {'<' if strictly_before else '<='} "
            "julianday(?)",
            value,
        )
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("knowledge_time must be an ISO date or timestamp") from exc
    if parsed_date.isoformat() != value:
        raise ValueError("knowledge_time must be an ISO date or timestamp")
    if strictly_before:
        return f"date({column}) < date(?)", value
    return f"date({column}) <= date(?)", value


def _fact_filters(
    *,
    ticker: str | None,
    entity_id: str | None,
    attribute: str | None,
    key_hash: str | None,
) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    parameters: list[object] = []
    if ticker is not None:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("ticker must not be blank")
        conditions.append("f.ticker = ?")
        parameters.append(normalized_ticker)
    if entity_id is not None:
        conditions.append("f.entity_id = ?")
        parameters.append(entity_id)
    if attribute is not None:
        conditions.append("f.attribute = ?")
        parameters.append(attribute)
    if key_hash is not None:
        conditions.append("f.key_hash = ?")
        parameters.append(key_hash)
    return conditions, parameters


def visible_facts_at(
    connection: sqlite3.Connection,
    knowledge_time: KnowledgeTime,
    *,
    ticker: str | None = None,
    entity_id: str | None = None,
    attribute: str | None = None,
    key_hash: str | None = None,
    strictly_before: bool = False,
) -> list[sqlite3.Row]:
    """Return facts with at least one source visible at the knowledge time."""
    filters, parameters = _fact_filters(
        ticker=ticker,
        entity_id=entity_id,
        attribute=attribute,
        key_hash=key_hash,
    )
    time_condition, time_value = _time_predicate(
        "fs.published_at",
        knowledge_time,
        strictly_before=strictly_before,
    )
    conditions = ["f.tier != 'T0'", time_condition, *filters]
    return connection.execute(
        f"""
        SELECT f.*, MIN(fs.published_at) AS first_visible_at
        FROM fact AS f
        JOIN fact_source AS fs ON fs.fact_id = f.id
        WHERE {" AND ".join(conditions)}
        GROUP BY f.id
        ORDER BY COALESCE(f.as_of, f.period, first_visible_at), first_visible_at, f.id
        """,
        [time_value, *parameters],
    ).fetchall()


def current_facts_at(
    connection: sqlite3.Connection,
    knowledge_time: KnowledgeTime,
    *,
    ticker: str | None = None,
    entity_id: str | None = None,
    attribute: str | None = None,
    key_hash: str | None = None,
    strictly_before: bool = False,
) -> list[sqlite3.Row]:
    """Return visible facts not superseded by the knowledge time."""
    filters, parameters = _fact_filters(
        ticker=ticker,
        entity_id=entity_id,
        attribute=attribute,
        key_hash=key_hash,
    )
    source_condition, source_value = _time_predicate(
        "fs.published_at",
        knowledge_time,
        strictly_before=strictly_before,
    )
    supersede_condition, supersede_value = _time_predicate(
        "supersede_source.published_at",
        knowledge_time,
        strictly_before=strictly_before,
    )
    conditions = ["f.tier != 'T0'", source_condition, *filters]
    return connection.execute(
        f"""
        SELECT f.*, MIN(fs.published_at) AS first_visible_at
        FROM fact AS f
        JOIN fact_source AS fs ON fs.fact_id = f.id
        WHERE {" AND ".join(conditions)}
          AND NOT EXISTS (
              SELECT 1
              FROM fact_source AS supersede_source
              WHERE supersede_source.supersedes_fact_id = f.id
                AND supersede_source.action = 'SUPERSEDE'
                AND {supersede_condition}
          )
        GROUP BY f.id
        ORDER BY COALESCE(f.as_of, f.period, first_visible_at), first_visible_at, f.id
        """,
        [source_value, *parameters, supersede_value],
    ).fetchall()


def state_at(
    connection: sqlite3.Connection,
    *,
    ticker: str,
    entity_id: str,
    attribute: str,
    knowledge_time: KnowledgeTime,
) -> list[sqlite3.Row]:
    """Return active observations for one entity/attribute as known at T."""
    return current_facts_at(
        connection,
        knowledge_time,
        ticker=ticker,
        entity_id=entity_id,
        attribute=attribute,
    )


def timeline_at(
    connection: sqlite3.Connection,
    *,
    ticker: str,
    entity_id: str,
    attribute: str,
    knowledge_time: KnowledgeTime,
) -> list[sqlite3.Row]:
    """Return the visible history for one entity/attribute at knowledge time T."""
    return visible_facts_at(
        connection,
        knowledge_time,
        ticker=ticker,
        entity_id=entity_id,
        attribute=attribute,
    )


def fact_sources_at(
    connection: sqlite3.Connection,
    fact_id: str,
    knowledge_time: KnowledgeTime,
) -> list[sqlite3.Row]:
    """Return source links for a fact that were published by knowledge time T."""
    predicate, value = _time_predicate(
        "published_at",
        knowledge_time,
        strictly_before=False,
    )
    return connection.execute(
        f"""
        SELECT fact_id, doc_id, published_at, quote, action,
               supersedes_fact_id, role
        FROM fact_source
        WHERE fact_id = ? AND {predicate}
        ORDER BY published_at, doc_id, action
        """,
        (fact_id, value),
    ).fetchall()
