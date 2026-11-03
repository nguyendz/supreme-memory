import sqlite3

from agent_memory.db import (
    database_connection,
    initialize_database,
    rebuild_paragraph_hash_aggregates,
)


def test_database_migrations_are_idempotent_and_fts_tracks_chunks(tmp_path):
    database_path = tmp_path / "var" / "memory.db"
    initialize_database(database_path)

    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source
            ) VALUES ('doc-1', 'CXO', 'doc-1', 'Example', '2026-01-01T00:00:00Z', 'v1')
            """
        )
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text
            ) VALUES (
                'chunk-1', 'doc-1', 'CXO', '2026-01-01T00:00:00Z',
                0, 'Finniss lithium production'
            )
            """
        )
        found = connection.execute(
            """
            SELECT chunk.id
            FROM chunk_fts
            JOIN chunk ON chunk.rowid = chunk_fts.rowid
            WHERE chunk_fts MATCH 'Finniss'
            """
        ).fetchall()
        assert [row["id"] for row in found] == ["chunk-1"]
        connection.execute(
            "UPDATE chunk SET text = 'BP33 underground development schedule' WHERE id = 'chunk-1'"
        )
        old_text = connection.execute(
            """
            SELECT chunk.id
            FROM chunk_fts
            JOIN chunk ON chunk.rowid = chunk_fts.rowid
            WHERE chunk_fts MATCH 'Finniss'
            """
        ).fetchall()
        updated_text = connection.execute(
            """
            SELECT chunk.id
            FROM chunk_fts
            JOIN chunk ON chunk.rowid = chunk_fts.rowid
            WHERE chunk_fts MATCH 'BP33'
            """
        ).fetchall()
        assert old_text == []
        assert [row["id"] for row in updated_text] == ["chunk-1"]

    initialize_database(database_path)
    with database_connection(database_path) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()
        assert [row["version"] for row in versions] == [1, 2, 3, 4, 5, 6]
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "reconciliation_cache",
            "reconciliation_review",
            "fact_fts",
            "memory_tool_call",
        } <= tables


def test_vector_tables_use_actual_embedding_width(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path, vector_dimensions=3)

    with database_connection(database_path) as connection:
        connection.execute(
            "INSERT INTO chunk_vec (embedding, chunk_id) VALUES (?, ?)",
            (sqlite3.Binary(bytes(12)), "chunk-1"),
        )
        connection.execute(
            "INSERT INTO entity_vec (embedding, entity_id) VALUES (?, ?)",
            (sqlite3.Binary(bytes(12)), "entity-1"),
        )
        dimensions = connection.execute(
            "SELECT dimensions FROM vector_metadata WHERE name = 'default'"
        ).fetchone()["dimensions"]
        assert dimensions == 3
        entity_vector_count = connection.execute(
            "SELECT COUNT(*) FROM entity_vec"
        ).fetchone()[0]
        assert entity_vector_count == 1

    try:
        initialize_database(database_path, vector_dimensions=4)
    except ValueError as exc:
        assert "use 3 dimensions" in str(exc)
    else:
        raise AssertionError("Changing embedding width must be explicit")


def test_initialize_database_restores_missing_entity_vector_table(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path, vector_dimensions=3)
    with database_connection(database_path) as connection:
        connection.execute("DROP TABLE entity_vec")

    initialize_database(database_path)

    with database_connection(database_path) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'entity_vec'"
        ).fetchone()
    assert table is not None


def test_fact_fts_tracks_fact_insert_update_and_delete(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO entity (id, ticker, type, canonical_name, first_seen_at)
            VALUES ('ent-1', 'CXO', 'project', 'BP33',
                    '2025-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO attribute (name, description, first_seen_at)
            VALUES ('production_volume', 'Production volume',
                    '2025-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO fact (
                id, ticker, entity_id, attribute, value, fact_type, tier,
                text, first_seen_at, extractor_version
            ) VALUES (
                'fact-1', 'CXO', 'ent-1', 'production_volume', '10,000 tonnes',
                'metric', 'T1', 'BP33 production reached 10,000 tonnes.',
                '2025-01-01T00:00:00+00:00', 'test-v1'
            )
            """
        )
        inserted = connection.execute(
            "SELECT fact_id FROM fact_fts WHERE fact_fts MATCH 'BP33'"
        ).fetchall()
        connection.execute(
            "UPDATE fact SET text = 'Finniss production reached 20,000 tonnes.' "
            "WHERE id = 'fact-1'"
        )
        old_after_update = connection.execute(
            "SELECT fact_id FROM fact_fts WHERE fact_fts MATCH 'BP33'"
        ).fetchall()
        new_after_update = connection.execute(
            "SELECT fact_id FROM fact_fts WHERE fact_fts MATCH 'Finniss'"
        ).fetchall()
        connection.execute("DELETE FROM fact WHERE id = 'fact-1'")
        after_delete = connection.execute(
            "SELECT fact_id FROM fact_fts WHERE fact_fts MATCH 'Finniss'"
        ).fetchall()

    assert [row["fact_id"] for row in inserted] == ["fact-1"]
    assert old_after_update == []
    assert [row["fact_id"] for row in new_after_update] == ["fact-1"]
    assert after_delete == []


def test_hash_observations_rebuild_aggregate_and_cascade_with_document(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    with database_connection(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source
            ) VALUES (?, 'CXO', ?, 'Example', ?, 'v1')
            """,
            [
                ("doc-1", "doc-1", "2025-01-01T00:00:00+00:00"),
                ("doc-2", "doc-2", "2025-01-02T00:00:00+00:00"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO paragraph_hash_observation (
                scope, ticker, simhash, doc_id, published_at, sample_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "ticker",
                    "CXO",
                    "0123456789abcdef",
                    "doc-1",
                    "2025-01-01T00:00:00+00:00",
                    "About Core",
                ),
                (
                    "ticker",
                    "CXO",
                    "0123456789abcdef",
                    "doc-2",
                    "2025-01-02T00:00:00+00:00",
                    "About Core",
                ),
            ],
        )
        rebuild_paragraph_hash_aggregates(connection)
        row = connection.execute(
            """
            SELECT first_seen_at, doc_count, sample_text
            FROM paragraph_hash
            WHERE scope = 'ticker' AND ticker = 'CXO'
            """
        ).fetchone()
        assert row["first_seen_at"] == "2025-01-01T00:00:00+00:00"
        assert row["doc_count"] == 2
        assert row["sample_text"] == "About Core"

        connection.execute("DELETE FROM document WHERE id = 'doc-2'")
        rebuild_paragraph_hash_aggregates(connection)
        row = connection.execute(
            "SELECT doc_count FROM paragraph_hash WHERE simhash = '0123456789abcdef'"
        ).fetchone()
        assert row["doc_count"] == 1


def test_schema_v2_database_migrates_to_v3(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute("ALTER TABLE document DROP COLUMN form_type")
        connection.execute("DELETE FROM schema_migrations WHERE version = 3")

    initialize_database(database_path)
    with database_connection(database_path) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        observation_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'paragraph_hash_observation'
            """
        ).fetchone()
        document_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(document)").fetchall()
        }
    assert [row["version"] for row in versions] == [1, 2, 3, 4, 5, 6]
    assert observation_table is not None
    assert "form_type" in document_columns


def test_schema_v1_database_runs_both_wave1_migrations(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute("DROP TABLE paragraph_hash_observation")
        connection.execute("DROP TABLE reconciliation_review")
        connection.execute("DROP TABLE reconciliation_cache")
        connection.execute("DROP TABLE memory_tool_call")
        connection.execute("DROP TRIGGER fact_fts_insert")
        connection.execute("DROP TRIGGER fact_fts_delete")
        connection.execute("DROP TRIGGER fact_fts_update")
        connection.execute("DROP TABLE fact_fts")
        connection.execute("ALTER TABLE document DROP COLUMN form_type")
        connection.execute("DELETE FROM schema_migrations WHERE version >= 2")

    initialize_database(database_path)
    with database_connection(database_path) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        document_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(document)").fetchall()
        }
        observation_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'paragraph_hash_observation'
            """
        ).fetchone()
    assert [row["version"] for row in versions] == [1, 2, 3, 4, 5, 6]
    assert observation_table is not None
    assert "form_type" in document_columns
