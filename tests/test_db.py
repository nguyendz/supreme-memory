import sqlite3

from agent_memory.db import database_connection, initialize_database


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

    initialize_database(database_path)
    with database_connection(database_path) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()
        assert [row["version"] for row in versions] == [1]


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
