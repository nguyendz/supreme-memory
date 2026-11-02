import json
from datetime import date
from pathlib import Path
import sqlite3
import struct
from types import SimpleNamespace

import pytest

from agent_memory.db import database_connection, initialize_database
from agent_memory.pipeline.chunks import search_fts
from agent_memory.query.baseline_rag import (
    answer_baseline,
    build_baseline_index,
)


SOURCE_TEXT = "Cash balance was A$200 million at 30 September 2026."


class FakeBaselineClient:
    model_name = "qwen3.5:9b"

    def __init__(
        self,
        *,
        citation_document: str | None = "doc-1",
        citation_quote=SOURCE_TEXT,
        answer_text="The cash balance was A$200 million.",
    ):
        self.citation_document = citation_document
        self.citation_quote = citation_quote
        self.answer_text = answer_text
        self.prompts = []
        self.embed_calls = []

    def embed(self, texts, *, doc_id=None):
        self.embed_calls.append((texts, doc_id))
        return SimpleNamespace(
            vectors=[[1.0, 0.0, 0.0] for _ in texts],
            dimensions=3,
            wall_duration_ms=1,
        )

    def chat_json(self, prompt, response_model, **_kwargs):
        del _kwargs
        self.prompts.append(prompt)
        raw_content = json.dumps(
            {
                "answer": self.answer_text,
                "citations": (
                    [
                        {
                            "document_id": self.citation_document,
                            "quote": self.citation_quote,
                        }
                    ]
                    if self.citation_document is not None
                    else []
                ),
            }
        )
        return SimpleNamespace(
            value=response_model.model_validate_json(raw_content),
            raw_content=raw_content,
            prompt_tokens=500,
            output_tokens=50,
            attempts=1,
            wall_duration_ms=5,
        )


def _source_database(path: Path):
    initialize_database(path)
    with database_connection(path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source
            ) VALUES (
                'doc-1', 'CXO', 'doc-1', 'Cash report',
                '2026-10-13T21:39:00+00:00', 'v1'
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text, is_boilerplate
            ) VALUES (?, 'doc-1', 'CXO', '2026-10-13T21:39:00+00:00', ?, ?, ?)
            """,
            [
                ("doc-1:00000", 0, SOURCE_TEXT, 0),
                ("doc-1:00001", 1, "About Core company information.", 1),
            ],
        )
    return path


def test_baseline_index_embeds_all_chunks_without_mutating_source(tmp_path):
    source_path = _source_database(tmp_path / "source.db")
    baseline_path = tmp_path / "var" / "eval" / "baseline.db"
    client = FakeBaselineClient()

    stats = build_baseline_index(source_path, baseline_path, client)

    assert stats.source_chunks == 2
    assert stats.embedded_chunks == 2
    with database_connection(baseline_path) as connection:
        vector_count = connection.execute(
            "SELECT COUNT(*) FROM chunk_vec"
        ).fetchone()[0]
        boilerplate_results = search_fts(
            connection, "About Core", include_boilerplate=True
        )
        memory_results = search_fts(connection, "About Core")
    with database_connection(source_path) as connection:
        source_vector_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vec'"
        ).fetchone()

    assert vector_count == 2
    assert [row.id for row in boilerplate_results] == ["doc-1:00001"]
    assert memory_results == []
    assert source_vector_table is None


def test_baseline_answers_with_exact_retrieved_citations_and_cutoff(tmp_path):
    source_path = _source_database(tmp_path / "source.db")
    baseline_path = tmp_path / "baseline.db"
    client = FakeBaselineClient()
    build_baseline_index(source_path, baseline_path, client)

    result = answer_baseline(
        baseline_path,
        client,
        "What was cash at the end of September?",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
    )

    assert result.answer == "The cash balance was A$200 million."
    assert result.citations[0].document_id == "doc-1"
    assert result.context_tokens > 0
    assert "About Core" in client.prompts[0]
    assert all(
        row.published_at.startswith("2026-10-13")
        for row in result.retrieved_chunks
    )


@pytest.mark.parametrize(
    ("citation_document", "citation_quote", "message"),
    [
        ("not-retrieved", SOURCE_TEXT, "outside the retrieved context"),
        ("doc-1", "Cash balance was A$201 million.", "not present"),
    ],
)
def test_baseline_rejects_invalid_citations(
    tmp_path, citation_document, citation_quote, message
):
    source_path = _source_database(tmp_path / "source.db")
    baseline_path = tmp_path / "baseline.db"
    client = FakeBaselineClient(
        citation_document=citation_document,
        citation_quote=citation_quote,
    )
    build_baseline_index(source_path, baseline_path, client)

    with pytest.raises(ValueError, match=message):
        answer_baseline(
            baseline_path,
            client,
            "What was cash?",
            knowledge_date=date(2026, 10, 15),
            ticker="CXO",
        )


def test_baseline_rejects_same_database_path(tmp_path):
    source_path = _source_database(tmp_path / "source.db")

    with pytest.raises(ValueError, match="must be separate"):
        build_baseline_index(source_path, source_path, FakeBaselineClient())


def test_baseline_rejects_uncited_non_abstention(tmp_path):
    source_path = _source_database(tmp_path / "source.db")
    baseline_path = tmp_path / "baseline.db"
    client = FakeBaselineClient(citation_document=None)
    build_baseline_index(source_path, baseline_path, client)

    with pytest.raises(ValueError, match="no citations"):
        answer_baseline(
            baseline_path,
            client,
            "What was cash?",
            knowledge_date=date(2026, 10, 15),
            ticker="CXO",
        )


def test_baseline_retries_once_after_semantic_citation_error(tmp_path):
    class CorrectingClient(FakeBaselineClient):
        def __init__(self):
            super().__init__()
            self.chat_count = 0

        def chat_json(self, prompt, response_model, **kwargs):
            self.chat_count += 1
            self.citation_document = (
                "chunk-id-is-not-a-document-id"
                if self.chat_count == 1
                else "doc-1"
            )
            return super().chat_json(prompt, response_model, **kwargs)

    source_path = _source_database(tmp_path / "source.db")
    baseline_path = tmp_path / "baseline.db"
    client = CorrectingClient()
    build_baseline_index(source_path, baseline_path, client)

    result = answer_baseline(
        baseline_path,
        client,
        "What was cash?",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
    )

    assert client.chat_count == 2
    assert result.attempts == 2
    assert result.citations[0].document_id == "doc-1"


def test_baseline_accepts_an_explicit_abstention_without_citations(tmp_path):
    source_path = _source_database(tmp_path / "source.db")
    baseline_path = tmp_path / "baseline.db"
    client = FakeBaselineClient(
        citation_document=None,
        answer_text=(
            "The provided source chunks do not contain enough information "
            "to answer this question."
        ),
    )
    build_baseline_index(source_path, baseline_path, client)

    answer = answer_baseline(
        baseline_path,
        client,
        "What was reported?",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
    )

    assert answer.citations == ()


def test_baseline_rebuilds_existing_vector_table_without_changing_source(tmp_path):
    source_path = tmp_path / "source-with-vectors.db"
    initialize_database(source_path, vector_dimensions=3)
    with database_connection(source_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source
            ) VALUES (
                'doc-1', 'CXO', 'doc-1', 'Cash report',
                '2026-10-13T21:39:00+00:00', 'v1'
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text, is_boilerplate
            ) VALUES (?, 'doc-1', 'CXO', '2026-10-13T21:39:00+00:00', ?, ?, ?)
            """,
            [
                ("doc-1:00000", 0, SOURCE_TEXT, 0),
                ("doc-1:00001", 1, "About Core company information.", 1),
            ],
        )
        connection.execute(
            "INSERT INTO chunk_vec (embedding, chunk_id) VALUES (?, ?)",
            (sqlite3.Binary(struct.pack("<3f", 1.0, 0.0, 0.0)), "doc-1:00000"),
        )
    baseline_path = tmp_path / "baseline-with-vectors.db"

    stats = build_baseline_index(
        source_path, baseline_path, FakeBaselineClient()
    )

    with database_connection(source_path) as connection:
        source_vectors = connection.execute(
            "SELECT COUNT(*) FROM chunk_vec"
        ).fetchone()[0]
    with database_connection(baseline_path) as connection:
        baseline_vectors = connection.execute(
            "SELECT COUNT(*) FROM chunk_vec"
        ).fetchone()[0]
    assert stats.embedded_chunks == 2
    assert source_vectors == 1
    assert baseline_vectors == 2
