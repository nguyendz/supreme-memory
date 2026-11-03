from datetime import date
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.db import database_connection, initialize_database
from agent_memory.pipeline.chunks import pack_vector
from agent_memory.query.agent import (
    MemoryAnswerResponse,
    QueryPlan,
    answer_memory,
)
from agent_memory.query.retrieve import retrieve_facts


DOCUMENT_ID = "20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068"
PUBLISHED_AT = "2026-10-13T21:39:00+00:00"
SOURCE = "Cash balance of A$200 million at 30 September 2026."


class FakeClient:
    model_name = "gemma4"

    def __init__(
        self,
        *,
        plan: QueryPlan,
        answer_ids: list[list[str]] | None = None,
    ):
        self.plan = plan
        self.answer_ids = answer_ids or [["c1"]]
        self.plan_calls = 0
        self.answer_calls = 0

    def embed(self, texts, *, doc_id=None):
        del doc_id
        if isinstance(texts, str):
            texts = [texts]
        return SimpleNamespace(
            vectors=[[1.0, 0.0, 0.0] for _ in texts],
            dimensions=3,
            wall_duration_ms=1,
        )

    def chat_json(self, prompt, response_model, **_kwargs):
        del prompt, _kwargs
        if response_model is QueryPlan:
            self.plan_calls += 1
            value = self.plan
        elif response_model is MemoryAnswerResponse:
            citation_ids = self.answer_ids[
                min(self.answer_calls, len(self.answer_ids) - 1)
            ]
            self.answer_calls += 1
            value = MemoryAnswerResponse(
                answer="The cash balance was A$200 million.",
                citation_ids=citation_ids,
            )
        else:
            raise AssertionError(f"Unexpected response schema: {response_model}")
        return SimpleNamespace(
            value=value,
            raw_content=value.model_dump_json(),
            prompt_tokens=10,
            output_tokens=5,
            attempts=1,
            wall_duration_ms=1,
        )


def _pilot_database(path):
    initialize_database(path, vector_dimensions=3)
    with database_connection(path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                text_path, triage_class
            ) VALUES (?, 'CXO', ?, 'September quarterly', ?, 'v1',
                      'data/CXO/source.md', 'full')
            """,
            (DOCUMENT_ID, DOCUMENT_ID, PUBLISHED_AT),
        )
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text
            ) VALUES (?, ?, 'CXO', ?, 0, ?)
            """,
            (f"{DOCUMENT_ID}:00000", DOCUMENT_ID, PUBLISHED_AT, SOURCE),
        )
        connection.execute(
            """
            INSERT INTO entity (
                id, ticker, type, canonical_name, first_seen_at
            ) VALUES ('ent-cxo', 'CXO', 'company', 'Core Lithium', ?)
            """,
            (PUBLISHED_AT,),
        )
        connection.execute(
            """
            INSERT INTO attribute (name, description, first_seen_at)
            VALUES (
                'cash_balance_as_of', 'Cash balance as of a measurement date', ?
            )
            """,
            (PUBLISHED_AT,),
        )
        connection.execute(
            """
            INSERT INTO fact (
                id, ticker, entity_id, attribute, value, value_num, unit,
                fact_type, tier, salience, text, key_hash, first_seen_at,
                extractor_version
            ) VALUES (
                'fact-cash', 'CXO', 'ent-cxo', 'cash_balance_as_of',
                'A$200 million at 30 September 2026', NULL, NULL,
                'metric', 'T1', 0.9,
                'Cash balance of A$200 million at 30 September 2026.',
                NULL, ?, 'test-v1'
            )
            """,
            (PUBLISHED_AT,),
        )
        connection.execute(
            """
            INSERT INTO fact_source (
                fact_id, doc_id, published_at, quote, action,
                supersedes_fact_id, role
            ) VALUES (
                'fact-cash', ?, ?, ?, 'ADD', NULL, 'origin'
            )
            """,
            (DOCUMENT_ID, PUBLISHED_AT, SOURCE),
        )


def test_memory_answer_returns_gold_state_with_materialized_source(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    client = FakeClient(
        plan=QueryPlan(
            intent="state",
            entity_id="ent-cxo",
            attribute="cash_balance_as_of",
            subject_date=date(2026, 9, 30),
        )
    )

    result = answer_memory(
        database_path,
        client,
        "What was CXO's cash balance at 30 September 2026?",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
    )

    assert result.answer == "The cash balance was A$200 million."
    assert result.intent == "state"
    assert len(result.citations) == 1
    assert result.citations[0].document_id == DOCUMENT_ID
    assert result.citations[0].quote == SOURCE
    assert result.citations[0].headline == "September quarterly"
    assert result.citations[0].source_path == "data/CXO/source.md"
    with database_connection(database_path) as connection:
        tool_call = connection.execute(
            "SELECT query_id, tool_name, result_count, status, args_json "
            "FROM memory_tool_call"
        ).fetchone()
    assert tool_call["query_id"] == result.query_id
    assert tool_call["tool_name"] == "get_state"
    assert tool_call["result_count"] == 1
    assert tool_call["status"] == "ok"
    assert "cash balance at 30 September 2026" not in tool_call["args_json"]


def test_memory_answer_abstains_when_no_fact_is_visible_by_knowledge_date(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    client = FakeClient(
        plan=QueryPlan(intent="search", search_query="cash balance")
    )

    result = answer_memory(
        database_path,
        client,
        "What was the cash balance?",
        knowledge_date=date(2026, 10, 12),
        ticker="CXO",
        chunk_fallback=True,
    )

    assert "could not find a verified memory fact" in result.answer
    assert result.citations == ()
    assert client.answer_calls == 0
    with database_connection(database_path) as connection:
        tool_calls = connection.execute(
            "SELECT tool_name, result_count, status, elapsed_ms "
            "FROM memory_tool_call ORDER BY id"
        ).fetchall()
    assert [row["tool_name"] for row in tool_calls] == [
        "search_facts",
        "search_chunks",
    ]
    assert tool_calls[1]["result_count"] == 0
    assert tool_calls[1]["status"] == "ok"
    assert tool_calls[1]["elapsed_ms"] >= 0


def test_memory_answer_retries_then_rejects_unknown_citation_ids(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    client = FakeClient(
        plan=QueryPlan(
            intent="state",
            entity_id="ent-cxo",
            attribute="cash_balance_as_of",
            subject_date=date(2026, 9, 30),
        ),
        answer_ids=[["invented"], ["still-invented"]],
    )

    with pytest.raises(ValueError, match="failed citation validation"):
        answer_memory(
            database_path,
            client,
            "What was CXO's cash balance at 30 September 2026?",
            knowledge_date=date(2026, 10, 15),
            ticker="CXO",
        )
    assert client.answer_calls == 2


def test_memory_tool_call_records_retrieval_errors_without_question_text(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            "INSERT INTO fact_vec (embedding, fact_id) VALUES (?, ?)",
            (pack_vector([1.0, 0.0, 0.0]), "fact-cash"),
        )

    class EmptyEmbeddingClient(FakeClient):
        def embed(self, texts, *, doc_id=None):
            del texts, doc_id
            return SimpleNamespace(vectors=[], dimensions=3, wall_duration_ms=1)

    client = EmptyEmbeddingClient(
        plan=QueryPlan(intent="search", search_query="unmatched-vector-query")
    )
    with pytest.raises(ValueError, match="no query vector"):
        answer_memory(
            database_path,
            client,
            "Find the cash balance in the reports.",
            knowledge_date=date(2026, 10, 15),
            ticker="CXO",
        )

    with database_connection(database_path) as connection:
        tool_call = connection.execute(
            "SELECT status, error, result_count, elapsed_ms, args_json "
            "FROM memory_tool_call"
        ).fetchone()
    assert tool_call["status"] == "error"
    assert tool_call["error"] == "ValueError"
    assert tool_call["result_count"] == 0
    assert tool_call["elapsed_ms"] >= 0
    assert "Find the cash balance in the reports." not in tool_call["args_json"]


def test_timeline_and_summary_routes_use_visible_pilot_facts(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    timeline_client = FakeClient(
        plan=QueryPlan(
            intent="timeline",
            entity_id="ent-cxo",
            attribute="cash_balance_as_of",
        )
    )
    timeline = answer_memory(
        database_path,
        timeline_client,
        "Show the cash balance history.",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
    )
    summary_client = FakeClient(
        plan=QueryPlan(
            intent="summary",
            start_date=date(2026, 10, 1),
            end_date=date(2026, 10, 31),
        )
    )
    summary = answer_memory(
        database_path,
        summary_client,
        "Summarize the September 2026 quarter.",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
    )

    assert timeline.intent == "timeline"
    assert timeline.retrieved_facts[0].value.startswith("A$200 million")
    assert summary.intent == "summary"
    assert summary.retrieved_facts[0].fact_id == "fact-cash"


def test_fact_vector_search_respects_current_and_historical_modes(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    quote = "Cash balance increased to A$250 million on 15 October 2026."
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                text_path, triage_class
            ) VALUES (
                'doc-later', 'CXO', 'doc-later', 'Cash update',
                '2026-10-20T00:00:00+00:00', 'v1', 'data/CXO/later.md', 'full'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text
            ) VALUES (
                'doc-later:00000', 'doc-later', 'CXO',
                '2026-10-20T00:00:00+00:00', 0, ?
            )
            """,
            (quote,),
        )
        connection.execute(
            """
            INSERT INTO fact (
                id, ticker, entity_id, attribute, value, value_num, unit,
                as_of, fact_type, tier, salience, text, key_hash,
                first_seen_at, extractor_version
            ) VALUES (
                'fact-cash-later', 'CXO', 'ent-cxo', 'cash_balance_as_of',
                'A$250 million', 250, 'AUDm', '2026-10-15',
                'metric', 'T1', 0.9, ?, NULL,
                '2026-10-20T00:00:00+00:00', 'test-v1'
            )
            """,
            (quote,),
        )
        connection.execute(
            """
            INSERT INTO fact_source (
                fact_id, doc_id, published_at, quote, action,
                supersedes_fact_id, role
            ) VALUES (
                'fact-cash-later', 'doc-later', '2026-10-20T00:00:00+00:00',
                ?, 'SUPERSEDE', 'fact-cash', 'origin'
            )
            """,
            (quote,),
        )
        connection.execute(
            "INSERT INTO fact_vec (embedding, fact_id) VALUES (?, ?)",
            (pack_vector([1.0, 0.0, 0.0]), "fact-cash"),
        )
        connection.execute(
            "INSERT INTO fact_vec (embedding, fact_id) VALUES (?, ?)",
            (pack_vector([1.0, 0.0, 0.0]), "fact-cash-later"),
        )

    client = FakeClient(plan=QueryPlan(intent="search", search_query="unused"))
    current = retrieve_facts(
        database_path,
        client,
        "vector-only-token",
        ticker="CXO",
        knowledge_date=date(2026, 10, 22),
        current_only=True,
    )
    history = retrieve_facts(
        database_path,
        client,
        "vector-only-token",
        ticker="CXO",
        knowledge_date=date(2026, 10, 22),
        current_only=False,
    )

    assert [fact.fact_id for fact in current] == ["fact-cash-later"]
    assert {fact.fact_id for fact in history} == {
        "fact-cash",
        "fact-cash-later",
    }


def test_search_route_uses_fact_fts_before_chunk_fallback(tmp_path):
    database_path = tmp_path / "pilot.db"
    initialize_database(database_path, vector_dimensions=3)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                text_path, triage_class
            ) VALUES (
                'doc-1', 'CXO', 'doc-1', 'Quarterly report',
                '2026-10-13T21:39:00+00:00', 'v1', 'data/source.md', 'full'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text, is_boilerplate
            ) VALUES (
                'doc-1:00000', 'doc-1', 'CXO',
                '2026-10-13T21:39:00+00:00', 0,
                'Cash balance was A$200 million at 30 September 2026.', 0
            )
            """
        )
    client = FakeClient(
        plan=QueryPlan(intent="search", search_query="cash balance")
    )

    result = answer_memory(
        database_path,
        client,
        "Find the reported cash balance.",
        knowledge_date=date(2026, 10, 15),
        ticker="CXO",
        chunk_fallback=True,
    )

    assert result.intent == "search"
    assert result.used_chunk_fallback
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "doc-1"
    assert "A$200 million" in result.citations[0].quote
    with database_connection(database_path) as connection:
        tool_calls = connection.execute(
            "SELECT tool_name, result_count, elapsed_ms, args_json "
            "FROM memory_tool_call ORDER BY id"
        ).fetchall()
    assert [row["tool_name"] for row in tool_calls] == [
        "search_facts",
        "search_chunks",
    ]
    assert [row["result_count"] for row in tool_calls] == [0, 1]
    assert all(row["elapsed_ms"] >= 0 for row in tool_calls)
    assert "Find the reported cash balance." not in tool_calls[0]["args_json"]


def test_show_fact_entity_and_timeline_commands_display_source_quotes(tmp_path):
    database_path = tmp_path / "pilot.db"
    _pilot_database(database_path)
    runner = CliRunner()
    command_sets = (
        (
            [
                "show",
                "fact",
                "fact-cash",
                "--as-of",
                "2026-10-15",
                "--database",
                str(database_path),
            ],
            "Cash balance of A$200 million at 30 September 2026.",
        ),
        (
            [
                "show",
                "entity",
                "ent-cxo",
                "--as-of",
                "2026-10-15",
                "--database",
                str(database_path),
            ],
            "Core Lithium",
        ),
        (
            [
                "show",
                "timeline",
                "ent-cxo",
                "cash_balance_as_of",
                "--as-of",
                "2026-10-15",
                "--database",
                str(database_path),
            ],
            "Timeline: Core Lithium / cash_balance_as_of",
        ),
    )

    for arguments, expected_output in command_sets:
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert expected_output in result.output
        assert "2026-10-13" in result.output
