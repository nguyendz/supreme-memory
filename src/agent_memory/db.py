"""SQLite schema, migrations, and vector-table initialization."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import sqlite_vec


SCHEMA_VERSION = 4

_SCHEMA_SQL = """
CREATE TABLE document (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    folder TEXT NOT NULL,
    headline TEXT NOT NULL,
    published_at TEXT NOT NULL,
    category TEXT,
    types_json TEXT NOT NULL DEFAULT '[]',
    price_sensitive INTEGER,
    text_path TEXT,
    text_source TEXT NOT NULL CHECK (text_source IN ('v2', 'v1', 'missing')),
    triage_class TEXT,
    triage_reason TEXT,
    ingest_status TEXT NOT NULL DEFAULT 'pending',
    extractor_version TEXT,
    ingested_at TEXT,
    error TEXT,
    UNIQUE (ticker, folder)
);

CREATE INDEX document_ticker_published_idx
    ON document (ticker, published_at);

CREATE TABLE doc_xref (
    src_doc TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    dst_doc TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    cited_title TEXT NOT NULL,
    cited_date TEXT,
    match_score REAL NOT NULL,
    PRIMARY KEY (src_doc, dst_doc, cited_title)
);

CREATE TABLE paragraph_hash (
    scope TEXT NOT NULL CHECK (scope IN ('ticker', 'global')),
    ticker TEXT NOT NULL DEFAULT '',
    simhash TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    doc_count INTEGER NOT NULL DEFAULT 1,
    sample_text TEXT NOT NULL,
    PRIMARY KEY (scope, ticker, simhash)
);

CREATE TABLE chunk (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    published_at TEXT NOT NULL,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    is_boilerplate INTEGER NOT NULL DEFAULT 0 CHECK (is_boilerplate IN (0, 1)),
    UNIQUE (doc_id, ord)
);

CREATE VIRTUAL TABLE chunk_fts USING fts5(
    text,
    content='chunk',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER chunk_fts_insert AFTER INSERT ON chunk BEGIN
    INSERT INTO chunk_fts (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER chunk_fts_delete AFTER DELETE ON chunk BEGIN
    INSERT INTO chunk_fts (chunk_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER chunk_fts_update AFTER UPDATE OF text ON chunk BEGIN
    INSERT INTO chunk_fts (chunk_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunk_fts (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE INDEX chunk_ticker_published_idx ON chunk (ticker, published_at);

CREATE TABLE entity (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    UNIQUE (ticker, type, canonical_name, first_seen_at)
);

CREATE TABLE entity_alias (
    entity_id TEXT NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    source_doc TEXT REFERENCES document(id) ON DELETE SET NULL,
    PRIMARY KEY (entity_id, alias, first_seen_at)
);

CREATE INDEX entity_alias_name_idx ON entity_alias (alias);

CREATE TABLE attribute (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    merged_into TEXT REFERENCES attribute(name)
);

CREATE TABLE fact (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    entity_id TEXT REFERENCES entity(id) ON DELETE SET NULL,
    attribute TEXT REFERENCES attribute(name),
    value TEXT,
    value_num REAL,
    unit TEXT,
    qualifier TEXT,
    as_of TEXT,
    period TEXT,
    fact_type TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('T0', 'T1', 'T2', 'T3')),
    salience REAL,
    text TEXT NOT NULL,
    key_hash TEXT,
    first_seen_at TEXT NOT NULL,
    extractor_version TEXT NOT NULL
);

CREATE INDEX fact_key_idx ON fact (ticker, key_hash, first_seen_at);
CREATE INDEX fact_entity_attribute_idx ON fact (ticker, entity_id, attribute);

CREATE TABLE fact_source (
    fact_id TEXT NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    published_at TEXT NOT NULL,
    quote TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('ADD', 'CONFIRM', 'SUPERSEDE')),
    supersedes_fact_id TEXT REFERENCES fact(id),
    role TEXT NOT NULL CHECK (role IN ('origin', 'confirm', 'xref')),
    PRIMARY KEY (fact_id, doc_id, action)
);

CREATE INDEX fact_source_published_idx ON fact_source (published_at, fact_id);

CREATE TABLE extraction_cache (
    doc_id TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, model, prompt_version)
);

CREATE TABLE llm_call (
    id INTEGER PRIMARY KEY,
    step TEXT NOT NULL,
    doc_id TEXT REFERENCES document(id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    ms REAL NOT NULL,
    ok INTEGER NOT NULL CHECK (ok IN (0, 1)),
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE vector_metadata (
    name TEXT PRIMARY KEY,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0)
);

INSERT INTO chunk_fts (chunk_fts) VALUES ('rebuild');
"""

_MIGRATION_2_SQL = """
CREATE TABLE paragraph_hash_observation (
    scope TEXT NOT NULL CHECK (scope IN ('ticker', 'global')),
    ticker TEXT NOT NULL DEFAULT '',
    simhash TEXT NOT NULL,
    doc_id TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    published_at TEXT NOT NULL,
    sample_text TEXT NOT NULL,
    PRIMARY KEY (scope, ticker, simhash, doc_id)
);

CREATE INDEX paragraph_hash_observation_lookup_idx
    ON paragraph_hash_observation (scope, ticker, simhash, published_at);

CREATE INDEX paragraph_hash_observation_doc_idx
    ON paragraph_hash_observation (doc_id);
"""

_MIGRATION_3_SQL = """
ALTER TABLE document ADD COLUMN form_type TEXT;
"""

_MIGRATION_4_SQL = """
CREATE TABLE reconciliation_cache (
    step TEXT NOT NULL CHECK (step IN ('entity_resolution', 'narrative_dedup')),
    input_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (step, input_hash, model, prompt_version)
);

CREATE TABLE reconciliation_review (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    candidate_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    matches_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'ignored')),
    created_at TEXT NOT NULL,
    UNIQUE (doc_id, candidate_hash)
);

CREATE INDEX reconciliation_review_status_idx
    ON reconciliation_review (status, doc_id);
"""


def connect_database(path: Path) -> sqlite3.Connection:
    """Open a configured SQLite connection."""
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    _load_vector_extension(connection)
    return connection


@contextmanager
def database_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Provide a connection that commits on success and always closes."""
    connection = connect_database(path)
    try:
        yield connection
        connection.commit()
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _load_vector_extension(connection: sqlite3.Connection) -> None:
    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)


def _apply_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    rows = connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    applied_versions = {int(row["version"]) for row in rows}
    if applied_versions and max(applied_versions) > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema {max(applied_versions)} is newer than supported "
            f"schema {SCHEMA_VERSION}"
        )
    migrations = (
        (1, _SCHEMA_SQL),
        (2, _MIGRATION_2_SQL),
        (3, _MIGRATION_3_SQL),
        (4, _MIGRATION_4_SQL),
    )
    for version, migration_sql in migrations:
        if version in applied_versions:
            continue
        applied_at = datetime.now(timezone.utc).isoformat()
        script = (
            "BEGIN IMMEDIATE;\n"
            + migration_sql
            + f"\nINSERT INTO schema_migrations(version, applied_at) "
            f"VALUES ({version}, '{applied_at}');\nCOMMIT;"
        )
        try:
            connection.executescript(script)
        except sqlite3.Error:
            if connection.in_transaction:
                connection.rollback()
            raise


def rebuild_paragraph_hash_aggregates(connection: sqlite3.Connection) -> None:
    """Rebuild the current hash summary from its per-document observations."""
    connection.execute("DELETE FROM paragraph_hash")
    connection.execute(
        """
        INSERT INTO paragraph_hash (
            scope, ticker, simhash, first_seen_at, doc_count, sample_text
        )
        SELECT
            scope,
            ticker,
            simhash,
            MIN(published_at),
            COUNT(DISTINCT doc_id),
            MIN(sample_text)
        FROM paragraph_hash_observation
        GROUP BY scope, ticker, simhash
        """
    )


def ensure_vector_tables(connection: sqlite3.Connection, dimensions: int) -> None:
    """Create vec0 tables once the embedding model's actual width is known."""
    if dimensions <= 0:
        raise ValueError("Vector dimensions must be a positive integer")

    existing = connection.execute(
        "SELECT dimensions FROM vector_metadata WHERE name = 'default'"
    ).fetchone()
    if existing is not None and int(existing["dimensions"]) != dimensions:
        raise ValueError(
            f"Database vectors use {existing['dimensions']} dimensions, "
            f"not {dimensions}; create a new database to change embedding models"
        )
    vector_tables = (
        ("chunk_vec", "chunk_id"),
        ("fact_vec", "fact_id"),
        ("entity_vec", "entity_id"),
    )
    for table_name, id_column in vector_tables:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (table_name,)
        ).fetchone()
        if table_exists is None:
            connection.execute(
                f"CREATE VIRTUAL TABLE {table_name} USING vec0("
                f"embedding float[{dimensions}], +{id_column} TEXT)"
            )
    if existing is None:
        connection.execute(
            "INSERT INTO vector_metadata(name, dimensions) VALUES ('default', ?)",
            (dimensions,),
        )


def initialize_database(
    path: Path, vector_dimensions: int | None = None
) -> Path:
    """Create the database, apply migrations, and optionally create vec0 tables."""
    database_path = path.expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(database_path)
    try:
        _apply_migrations(connection)
        if vector_dimensions is not None:
            ensure_vector_tables(connection, vector_dimensions)
        else:
            vector_metadata = connection.execute(
                "SELECT dimensions FROM vector_metadata WHERE name = 'default'"
            ).fetchone()
            if vector_metadata is not None:
                ensure_vector_tables(connection, int(vector_metadata["dimensions"]))
        connection.commit()
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()
    return database_path


def record_llm_call(
    connection: sqlite3.Connection,
    *,
    step: str,
    doc_id: str | None,
    model: str,
    attempt: int,
    prompt_tokens: int | None,
    output_tokens: int | None,
    elapsed_ms: float,
    ok: bool,
    error: str | None = None,
) -> None:
    """Persist one successful or failed Ollama request."""
    connection.execute(
        """
        INSERT INTO llm_call (
            step, doc_id, model, attempt, prompt_tokens, output_tokens,
            ms, ok, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            step,
            doc_id,
            model,
            attempt,
            prompt_tokens,
            output_tokens,
            elapsed_ms,
            int(ok),
            error,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def upsert_document(
    connection: sqlite3.Connection,
    *,
    doc_id: str,
    ticker: str,
    folder: str,
    headline: str,
    published_at: str,
    category: str | None,
    announcement_types: tuple[str, ...],
    is_price_sensitive: bool | None,
    text_path: str | None,
    text_source: str,
) -> None:
    """Insert or refresh one discovered document's source metadata."""
    connection.execute(
        """
        INSERT INTO document (
            id, ticker, folder, headline, published_at, category, types_json,
            price_sensitive, text_path, text_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            ticker = excluded.ticker,
            folder = excluded.folder,
            headline = excluded.headline,
            published_at = excluded.published_at,
            category = excluded.category,
            types_json = excluded.types_json,
            price_sensitive = excluded.price_sensitive,
            text_path = excluded.text_path,
            text_source = excluded.text_source
        """,
        (
            doc_id,
            ticker,
            folder,
            headline,
            published_at,
            category,
            json.dumps(announcement_types),
            None if is_price_sensitive is None else int(is_price_sensitive),
            text_path,
            text_source,
        ),
    )
