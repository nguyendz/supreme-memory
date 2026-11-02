import sqlite3

import pytest

from agent_memory.db import database_connection, initialize_database
from agent_memory.pipeline.boilerplate import Paragraph
from agent_memory.pipeline.chunks import (
    build_chunks,
    create_embeddings,
    delete_document_chunks,
    persist_chunks,
    search_fts,
    search_vectors,
    split_oversized_paragraph,
)


class FakeEmbeddingClient:
    def __init__(self, dimensions=3):
        self.dimensions = dimensions
        self.calls = []

    def embed(self, texts, *, doc_id=None):
        self.calls.append((texts, doc_id))
        return type(
            "EmbeddingResult",
            (),
            {
                "dimensions": self.dimensions,
                "vectors": [
                    [1.0, 0.0, 0.0] if "cash" in text.lower() else [0.0, 1.0, 0.0]
                    for text in texts
                ],
            },
        )()


def _paragraph(text, is_boilerplate=False):
    return Paragraph(text=text, simhash="0000000000000000", is_boilerplate=is_boilerplate)


def _insert_document(connection, doc_id="doc-1"):
    connection.execute(
        """
        INSERT INTO document (
            id, ticker, folder, headline, published_at, text_source
        ) VALUES (?, 'CXO', ?, 'Example', '2026-01-01T00:00:00+00:00', 'v1')
        """,
        (doc_id, doc_id),
    )


def test_split_oversized_paragraph_keeps_all_text_with_bounded_chunks():
    text = "First sentence. " + ("long words " * 300) + " Final sentence."
    chunks = split_oversized_paragraph(text, max_characters=300)

    assert len(chunks) > 2
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert "First sentence." in chunks[0]
    assert "Final sentence." in chunks[-1]


def test_build_chunks_preserves_order_and_boilerplate_flag():
    chunks = build_chunks(
        doc_id="doc-1",
        ticker="CXO",
        published_at="2026-01-01T00:00:00+00:00",
        paragraphs=[
            _paragraph("Financial result: A$12 million."),
            _paragraph("About Core", is_boilerplate=True),
        ],
    )

    assert [chunk.ord for chunk in chunks] == [0, 1]
    assert [chunk.id for chunk in chunks] == ["doc-1:00000", "doc-1:00001"]
    assert [chunk.is_boilerplate for chunk in chunks] == [False, True]


def test_embedding_skips_boilerplate_and_persists_fts_and_vectors(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    client = FakeEmbeddingClient()
    chunks = build_chunks(
        doc_id="doc-1",
        ticker="CXO",
        published_at="2026-01-01T00:00:00+00:00",
        paragraphs=[
            _paragraph("Cash balance was A$200 million."),
            _paragraph("About Core", is_boilerplate=True),
        ],
    )
    embeddings = create_embeddings(chunks, client, database_path)

    with database_connection(database_path) as connection:
        _insert_document(connection)
        persist_chunks(connection, chunks, embeddings)
        fts_results = search_fts(connection, "cash balance")
        vector_results = search_vectors(connection, [1.0, 0.0, 0.0])
        assert [result.id for result in fts_results] == ["doc-1:00000"]
        assert [result.id for result in vector_results] == ["doc-1:00000"]
        assert len(client.calls) == 1
        assert client.calls[0][0] == ["Cash balance was A$200 million."]


def test_document_with_only_boilerplate_creates_no_vectors(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    client = FakeEmbeddingClient()
    chunks = build_chunks(
        doc_id="doc-1",
        ticker="CXO",
        published_at="2026-01-01T00:00:00+00:00",
        paragraphs=[_paragraph("About Core company information.", True)],
    )

    assert create_embeddings(chunks, client, database_path) == {}
    assert client.calls == []


def test_fts_and_vectors_exclude_boilerplate_and_delete_cleanly(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path, vector_dimensions=3)
    chunks = build_chunks(
        doc_id="doc-1",
        ticker="CXO",
        published_at="2026-01-01T00:00:00+00:00",
        paragraphs=[
            _paragraph("About Core lithium company contact information.", True),
            _paragraph("Finniss cash balance is A$200 million."),
        ],
    )
    embeddings = {
        chunks[1].id: sqlite3.Binary(bytes(12)),
    }

    with database_connection(database_path) as connection:
        _insert_document(connection)
        persist_chunks(connection, chunks, embeddings)
        assert [item.id for item in search_fts(connection, "About Core")] == []
        assert [item.id for item in search_vectors(connection, [0.0, 0.0, 0.0])] == [
            chunks[1].id
        ]
        delete_document_chunks(connection, "doc-1")
        assert search_fts(connection, "Finniss") == []
        assert search_vectors(connection, [0.0, 0.0, 0.0]) == []


def test_create_embeddings_rejects_dimensions_that_change_between_batches(tmp_path):
    class ChangingClient:
        def __init__(self):
            self.call_count = 0
            self.last_doc_id = None

        def embed(self, texts, *, doc_id=None):
            self.call_count += 1
            self.last_doc_id = doc_id
            dimensions = 3 if self.call_count == 1 else 4
            return type(
                "EmbeddingResult",
                (),
                {
                    "dimensions": dimensions,
                    "vectors": [[0.0] * dimensions for _ in texts],
                },
            )()

    chunks = [
        type(
            "Chunk",
            (),
            {
                "id": f"chunk-{index}",
                "doc_id": "doc-1",
                "is_boilerplate": False,
                "text": f"paragraph {index}",
            },
        )()
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="dimensions changed"):
        create_embeddings(
            chunks,
            ChangingClient(),
            tmp_path / "memory.db",
            batch_size=1,
        )


def test_search_rejects_invalid_limits():
    with pytest.raises(ValueError, match="limit must be positive"):
        search_fts(sqlite3.connect(":memory:"), "cash", limit=0)


def test_vector_search_reports_uninitialized_tables():
    with pytest.raises(RuntimeError, match="not initialized"):
        search_vectors(sqlite3.connect(":memory:"), [0.0, 0.0, 0.0])
