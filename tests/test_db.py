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
        assert [row["version"] for row in versions] == [1, 2, 3]


def test_vector_tables_use_actual_embedding_width(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path, vector_dimensions=3)

    with database_connection(database_path) as connection:
        connection.execute(
            "INSERT INTO chunk_vec (embedding, chunk_id) VALUES (?, ?)",
            (sqlite3.Binary(bytes(12)), "chunk-1"),
        )
        dimensions = connection.execute(
            "SELECT dimensions FROM vector_metadata WHERE name = 'default'"
        ).fetchone()["dimensions"]
        assert dimensions == 3

    try:
        initialize_database(database_path, vector_dimensions=4)
    except ValueError as exc:
        assert "use 3 dimensions" in str(exc)
    else:
        raise AssertionError("Changing embedding width must be explicit")


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
    assert [row["version"] for row in versions] == [1, 2, 3]
    assert observation_table is not None
    assert "form_type" in document_columns


def test_schema_v1_database_runs_both_wave1_migrations(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute("DROP TABLE paragraph_hash_observation")
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
    assert [row["version"] for row in versions] == [1, 2, 3]
    assert observation_table is not None
    assert "form_type" in document_columns
