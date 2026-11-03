from types import SimpleNamespace

from agent_memory.db import database_connection, initialize_database
from agent_memory.pipeline.extract import CandidateEntity
from agent_memory.pipeline.resolve import (
    normalize_attribute_name,
    resolve_attribute,
    resolve_entity,
    stable_entity_id,
)


class FakeClient:
    def __init__(self):
        self.embed_calls = 0

    def embed(self, texts, *, doc_id=None):
        del doc_id
        self.embed_calls += 1
        if isinstance(texts, str):
            texts = [texts]
        return SimpleNamespace(
            vectors=[[1.0, 0.0, 0.0] for _ in texts],
            dimensions=3,
            wall_duration_ms=1,
        )

    def chat_json(self, *_args, **_kwargs):
        raise AssertionError("This test should resolve without an LLM call")


def _insert_document(connection, document_id, published_at):
    connection.execute(
        """
        INSERT INTO document (
            id, ticker, folder, headline, published_at, text_source
        ) VALUES (?, 'CXO', ?, 'Example', ?, 'v1')
        """,
        (document_id, document_id, published_at),
    )


def test_entity_alias_is_visible_only_after_its_first_seen_time(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        _insert_document(connection, "doc-old", "2024-01-01T00:00:00+00:00")
        _insert_document(connection, "doc-alias", "2026-01-01T00:00:00+00:00")
        connection.execute(
            """
            INSERT INTO entity (id, ticker, type, canonical_name, first_seen_at)
            VALUES ('entity-old', 'CXO', 'company', 'Core Exploration',
                    '2024-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO entity_alias (
                entity_id, alias, first_seen_at, source_doc
            ) VALUES (
                'entity-old', 'Core Lithium', '2026-01-01T00:00:00+00:00',
                'doc-alias'
            )
            """
        )

    client = FakeClient()
    earlier = resolve_entity(
        database_path,
        client,
        ticker="CXO",
        source_entity=CandidateEntity(name="Core Lithium", type="company"),
        evidence="Core Lithium announced a project.",
        published_at="2025-01-01T00:00:00+00:00",
        document_id="doc-old",
        model="gemma4",
        same_document={},
    )
    later = resolve_entity(
        database_path,
        client,
        ticker="CXO",
        source_entity=CandidateEntity(name="Core Lithium", type="company"),
        evidence="Core Lithium announced a project.",
        published_at="2027-01-01T00:00:00+00:00",
        document_id="doc-alias",
        model="gemma4",
        same_document={},
    )

    assert earlier.entity is not None
    assert earlier.entity.id == stable_entity_id("CXO", "company", "Core Lithium")
    assert earlier.entity.id != "entity-old"
    assert later.entity is not None
    assert later.entity.id == "entity-old"
    assert client.embed_calls == 1


def test_attribute_resolution_normalizes_and_follows_explicit_merge(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO attribute (name, description, first_seen_at)
            VALUES ('cash_balance', 'Cash balance', '2025-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO attribute (
                name, description, first_seen_at, status, merged_into
            ) VALUES (
                'cash_position', 'Old cash key', '2025-01-01T00:00:00+00:00',
                'merged', 'cash_balance'
            )
            """
        )

    resolved = resolve_attribute(
        database_path,
        "Cash Position",
        same_document={},
    )

    assert normalize_attribute_name("Cash Position") == "cash_position"
    assert resolved.name == "cash_balance"
    assert not resolved.created
