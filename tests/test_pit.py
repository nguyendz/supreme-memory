from datetime import date

import pytest

from agent_memory.db import database_connection, initialize_database
from agent_memory.query.pit import (
    current_facts_at,
    fact_sources_at,
    state_at,
    timeline_at,
    visible_facts_at,
)


def _insert_document(connection, document_id, published_at):
    connection.execute(
        """
        INSERT INTO document (
            id, ticker, folder, headline, published_at, text_source
        ) VALUES (?, 'CXO', ?, 'Example', ?, 'v1')
        """,
        (document_id, document_id, published_at),
    )


def test_point_in_time_views_respect_source_publication_and_supersession(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        _insert_document(connection, "doc-1", "2025-01-01T09:00:00+00:00")
        _insert_document(connection, "doc-2", "2025-02-01T09:00:00+00:00")
        connection.execute(
            """
            INSERT INTO entity (id, ticker, type, canonical_name, first_seen_at)
            VALUES ('ent-1', 'CXO', 'company', 'Core Lithium',
                    '2025-01-01T09:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO attribute (name, description, first_seen_at)
            VALUES ('cash_balance', 'Cash balance', '2025-01-01T09:00:00+00:00')
            """
        )
        connection.executemany(
            """
            INSERT INTO fact (
                id, ticker, entity_id, attribute, value, value_num, unit,
                as_of, period, fact_type, tier, text, key_hash,
                first_seen_at, extractor_version
            ) VALUES (?, 'CXO', 'ent-1', 'cash_balance', ?, ?, 'AUD million',
                      '2024-12-31', '2024-Q4', 'metric', 'T1', ?, 'cash-q4',
                      ?, 'test-v1')
            """,
            [
                (
                    "fact-old",
                    "A$100 million",
                    100,
                    "Cash was A$100 million at 31 December 2024.",
                    "2025-01-01T09:00:00+00:00",
                ),
                (
                    "fact-new",
                    "A$90 million",
                    90,
                    "Cash was A$90 million at 31 December 2024.",
                    "2025-02-01T09:00:00+00:00",
                ),
            ],
        )
        connection.executemany(
            """
            INSERT INTO fact_source (
                fact_id, doc_id, published_at, quote, action,
                supersedes_fact_id, role
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "fact-old",
                    "doc-1",
                    "2025-01-01T09:00:00+00:00",
                    "Cash was A$100 million.",
                    "ADD",
                    None,
                    "origin",
                ),
                (
                    "fact-new",
                    "doc-2",
                    "2025-02-01T09:00:00+00:00",
                    "Cash was A$90 million.",
                    "SUPERSEDE",
                    "fact-old",
                    "origin",
                ),
            ],
        )

        before_update = current_facts_at(
            connection,
            date(2025, 1, 31),
            ticker="CXO",
            entity_id="ent-1",
            attribute="cash_balance",
        )
        after_update = state_at(
            connection,
            ticker="CXO",
            entity_id="ent-1",
            attribute="cash_balance",
            knowledge_time=date(2025, 2, 1),
        )
        history = timeline_at(
            connection,
            ticker="CXO",
            entity_id="ent-1",
            attribute="cash_balance",
            knowledge_time=date(2025, 2, 1),
        )
        sources = fact_sources_at(connection, "fact-new", date(2025, 2, 1))

    assert [row["id"] for row in before_update] == ["fact-old"]
    assert [row["id"] for row in after_update] == ["fact-new"]
    assert {row["id"] for row in history} == {"fact-old", "fact-new"}
    assert [row["action"] for row in sources] == ["SUPERSEDE"]


def test_backdated_as_of_does_not_make_later_source_visible_early(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        _insert_document(connection, "doc-late", "2026-03-01T09:00:00+00:00")
        connection.execute(
            """
            INSERT INTO fact (
                id, ticker, as_of, fact_type, tier, text, first_seen_at,
                extractor_version
            ) VALUES (
                'fact-late', 'CXO', '2025-01-01', 'event', 'T1',
                'A backdated fact', '2026-03-01T09:00:00+00:00', 'test-v1'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO fact_source (
                fact_id, doc_id, published_at, quote, action, role
            ) VALUES (
                'fact-late', 'doc-late', '2026-03-01T09:00:00+00:00',
                'A backdated fact quote.', 'ADD', 'origin'
            )
            """
        )

        early = visible_facts_at(connection, date(2025, 12, 31), ticker="CXO")
        known_later = visible_facts_at(connection, date(2026, 3, 1), ticker="CXO")
        same_instant_strict = visible_facts_at(
            connection,
            "2026-03-01T09:00:00+00:00",
            ticker="CXO",
            strictly_before=True,
        )

    assert early == []
    assert [row["id"] for row in known_later] == ["fact-late"]
    assert same_instant_strict == []


def test_point_in_time_helpers_reject_invalid_knowledge_time(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        with pytest.raises(ValueError, match="ISO date or timestamp"):
            visible_facts_at(connection, "not-a-date")
