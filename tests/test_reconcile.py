import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import EmbeddingResult
from agent_memory.model_comparison import FIXED_DOCUMENT_IDS
from agent_memory.pipeline.extract import (
    CachedExtraction,
    CandidateEntity,
    CandidateFact,
    ExtractionUsage,
    _prompt_versions,
    RawSegmentOutput,
    _source_hash,
    segment_chunks,
)
from agent_memory.pipeline.reconcile import (
    NarrativeChoice,
    preflight_reconciliation,
    run_reconciliation,
)
from agent_memory.query.pit import current_facts_at


class FakeClient:
    model_name = "gemma4"

    def __init__(self, *, narrative_choice="same", narrative_index=0):
        self.narrative_choice = narrative_choice
        self.narrative_index = narrative_index
        self.chat_calls = 0
        self.embed_calls = 0

    def embed(self, texts, *, doc_id=None):
        del doc_id
        self.embed_calls += 1
        if isinstance(texts, str):
            texts = [texts]
        vectors = [[1.0, 0.0, 0.0] for _ in texts]
        return EmbeddingResult(
            vectors=vectors,
            dimensions=3,
            wall_duration_ms=1,
        )

    def chat_json(self, _prompt, response_model, **_kwargs):
        del _prompt, _kwargs
        self.chat_calls += 1
        if response_model is not NarrativeChoice:
            raise AssertionError(f"Unexpected decision model: {response_model}")
        value = NarrativeChoice(
            choice=self.narrative_choice,
            candidate_index=(
                self.narrative_index if self.narrative_choice == "same" else None
            ),
            reason="Synthetic test decision.",
        )
        return SimpleNamespace(
            value=value,
            raw_content=value.model_dump_json(),
            prompt_tokens=10,
            output_tokens=5,
            attempts=1,
            wall_duration_ms=1,
        )


def _insert_cached_document(
    connection,
    document_id,
    published_at,
    source,
    candidate,
    *,
    model="gemma4",
):
    connection.execute(
        """
        INSERT INTO document (
            id, ticker, folder, headline, published_at, text_source,
            triage_class
        ) VALUES (?, 'CXO', ?, 'Example', ?, 'v1', 'full')
        """,
        (document_id, document_id, published_at),
    )
    connection.execute(
        """
        INSERT INTO chunk (
            id, doc_id, ticker, published_at, ord, text
        ) VALUES (?, ?, 'CXO', ?, 0, ?)
        """,
        (f"{document_id}:00000", document_id, published_at, source),
    )
    segment = segment_chunks([(0, source)])[0]
    prompt_text = (
        Path(__file__).resolve().parents[1] / "prompts" / "extract.md"
    ).read_text(encoding="utf-8")
    _, prompt_version = _prompt_versions(
        prompt_text=prompt_text,
        family_prompt="",
        family="full",
    )
    cached = CachedExtraction(
        source_doc_id=document_id,
        source_form_type=None,
        form_family="full",
        source_hash=_source_hash([segment]),
        facts=[candidate],
        raw_segments=[
            RawSegmentOutput(
                segment_id=segment.segment_id,
                raw_content=json.dumps(
                    {"facts": [candidate.model_dump(mode="json")]}
                ),
            )
        ],
        usage=ExtractionUsage(),
    )
    connection.execute(
        """
        INSERT INTO extraction_cache (
            doc_id, model, prompt_version, output_json, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (document_id, model, prompt_version, cached.model_dump_json(), published_at),
    )


def _cash_fact(value, quote, segment_id):
    return CandidateFact(
        entity=CandidateEntity(name="Core Lithium", type="company"),
        attribute="cash_balance",
        value=f"A${value} million",
        value_num=value,
        unit="AUD million",
        qualifier=None,
        as_of=date(2024, 12, 31),
        period="2024-Q4",
        fact_type="metric",
        tier="T1",
        salience=0.9,
        text=f"Cash balance was A${value} million at 31 December 2024.",
        quote=quote,
        segment_id=segment_id,
    )


def test_key_reconciliation_and_full_ticker_replay_are_stable(tmp_path):
    database_path = tmp_path / "wave3.db"
    initialize_database(database_path)
    docs = [
        (
            "doc-1",
            "2025-01-15T00:00:00+00:00",
            "Cash balance was A$100 million at 31 December 2024.",
            100,
        ),
        (
            "doc-2",
            "2025-02-15T00:00:00+00:00",
            "Cash balance was A$100 million at 31 December 2024.",
            100,
        ),
        (
            "doc-3",
            "2025-03-15T00:00:00+00:00",
            "Cash balance was A$90 million at 31 December 2024.",
            90,
        ),
    ]
    with database_connection(database_path) as connection:
        for document_id, published_at, source, value in docs:
            segment = segment_chunks([(0, source)])[0]
            _insert_cached_document(
                connection,
                document_id,
                published_at,
                source,
                _cash_fact(value, source, segment.segment_id),
            )

    client = FakeClient()
    summary = run_reconciliation(
        database_path,
        client,
        model="gemma4",
        ticker="CXO",
        confirm_wave2_gate=True,
    )
    with database_connection(database_path) as connection:
        before_supersession = current_facts_at(
            connection,
            date(2025, 2, 28),
            ticker="CXO",
            attribute="cash_balance",
        )
        after_supersession = current_facts_at(
            connection,
            date(2025, 3, 15),
            ticker="CXO",
            attribute="cash_balance",
        )
        first_snapshot = (
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, ticker, type, canonical_name, first_seen_at "
                    "FROM entity ORDER BY id"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT name, description, first_seen_at FROM attribute ORDER BY name"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, entity_id, attribute, value, value_num, key_hash "
                    "FROM fact ORDER BY id"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT fact_id, doc_id, action, supersedes_fact_id "
                    "FROM fact_source ORDER BY doc_id, action"
                )
            ],
        )

    assert (summary.added, summary.confirmed, summary.superseded) == (1, 1, 1)
    assert len(before_supersession) == 1
    assert before_supersession[0]["value"] == "A$100 million"
    assert len(after_supersession) == 1
    assert after_supersession[0]["value"] == "A$90 million"

    replay = run_reconciliation(
        database_path,
        client,
        model="gemma4",
        ticker="CXO",
        replay=True,
        confirm_wave2_gate=True,
    )
    with database_connection(database_path) as connection:
        replay_snapshot = (
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, ticker, type, canonical_name, first_seen_at "
                    "FROM entity ORDER BY id"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT name, description, first_seen_at FROM attribute ORDER BY name"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, entity_id, attribute, value, value_num, key_hash "
                    "FROM fact ORDER BY id"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT fact_id, doc_id, action, supersedes_fact_id "
                    "FROM fact_source ORDER BY doc_id, action"
                )
            ],
        )

    assert (replay.added, replay.confirmed, replay.superseded) == (1, 1, 1)
    assert replay_snapshot == first_snapshot


def test_narrative_judgment_is_cached_and_reused_on_replay(tmp_path):
    database_path = tmp_path / "wave3.db"
    initialize_database(database_path)
    docs = [
        (
            "doc-1",
            "2025-01-15T00:00:00+00:00",
            "Finniss restart work commenced in March 2025.",
            "Finniss restart work commenced in March 2025.",
        ),
        (
            "doc-2",
            "2025-02-15T00:00:00+00:00",
            "Work on the Finniss restart began in March 2025.",
            "Work on the Finniss restart began in March 2025.",
        ),
    ]
    with database_connection(database_path) as connection:
        for document_id, published_at, source, quote in docs:
            segment = segment_chunks([(0, source)])[0]
            candidate = CandidateFact(
                entity=CandidateEntity(name="Core Lithium", type="company"),
                attribute="restart_milestone",
                value="Restart work commenced",
                value_num=None,
                unit=None,
                qualifier=None,
                as_of=None,
                period=None,
                fact_type="event",
                tier="T2",
                salience=0.7,
                text="Finniss restart work commenced in March 2025.",
                quote=quote,
                segment_id=segment.segment_id,
            )
            _insert_cached_document(
                connection,
                document_id,
                published_at,
                source,
                candidate,
            )

    client = FakeClient(narrative_choice="same")
    first = run_reconciliation(
        database_path,
        client,
        model="gemma4",
        ticker="CXO",
        confirm_wave2_gate=True,
    )
    assert (first.added, first.confirmed) == (1, 1)
    assert client.chat_calls == 1
    assert first.decision_cache_hits == 0

    second_client = FakeClient(narrative_choice="unresolved")
    replay = run_reconciliation(
        database_path,
        second_client,
        model="gemma4",
        ticker="CXO",
        replay=True,
        confirm_wave2_gate=True,
    )
    with database_connection(database_path) as connection:
        fact_count = connection.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
        review_count = connection.execute(
            "SELECT COUNT(*) FROM reconciliation_review WHERE status = 'pending'"
        ).fetchone()[0]
        decision_cache_count = connection.execute(
            "SELECT COUNT(*) FROM reconciliation_cache WHERE step = 'narrative_dedup'"
        ).fetchone()[0]

    assert (replay.added, replay.confirmed) == (1, 1)
    assert replay.decision_cache_hits == 1
    assert second_client.chat_calls == 0
    assert fact_count == 1
    assert review_count == 0
    assert decision_cache_count == 1


@pytest.mark.parametrize(
    ("narrative_choice", "narrative_index", "expected_reason"),
    [
        ("unresolved", 0, "Synthetic test decision"),
        ("same", 99, "unavailable candidate index"),
    ],
)
def test_uncertain_narrative_candidate_is_preserved_for_review(
    tmp_path,
    narrative_choice,
    narrative_index,
    expected_reason,
):
    database_path = tmp_path / "wave3.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        first_source = "Finniss restart work commenced in March 2025."
        first_segment = segment_chunks([(0, first_source)])[0]
        first = CandidateFact(
            entity=CandidateEntity(name="Core Lithium", type="company"),
            attribute="restart_milestone",
            value="Restart work commenced",
            value_num=None,
            unit=None,
            qualifier=None,
            as_of=None,
            period=None,
            fact_type="event",
            tier="T2",
            salience=0.7,
            text=first_source,
            quote=first_source,
            segment_id=first_segment.segment_id,
        )
        _insert_cached_document(
            connection,
            "doc-1",
            "2025-01-15T00:00:00+00:00",
            first_source,
            first,
        )
        second_source = "Finniss restart status remains under review."
        second_segment = segment_chunks([(0, second_source)])[0]
        second = CandidateFact(
            entity=CandidateEntity(name="Core Lithium", type="company"),
            attribute="restart_milestone",
            value="Restart status under review",
            value_num=None,
            unit=None,
            qualifier=None,
            as_of=None,
            period=None,
            fact_type="event",
            tier="T2",
            salience=0.7,
            text=second_source,
            quote=second_source,
            segment_id=second_segment.segment_id,
        )
        _insert_cached_document(
            connection,
            "doc-2",
            "2025-02-15T00:00:00+00:00",
            second_source,
            second,
        )

    summary = run_reconciliation(
        database_path,
        FakeClient(
            narrative_choice=narrative_choice,
            narrative_index=narrative_index,
        ),
        model="gemma4",
        ticker="CXO",
        confirm_wave2_gate=True,
    )
    with database_connection(database_path) as connection:
        facts = connection.execute("SELECT id, text FROM fact").fetchall()
        reviews = connection.execute(
            """
            SELECT reason, candidate_json FROM reconciliation_review
            WHERE status = 'pending'
            """
        ).fetchall()

    assert summary.added == 1
    assert summary.confirmed == 0
    assert summary.unresolved == 1
    assert len(facts) == 1
    assert len(reviews) == 1
    assert expected_reason in reviews[0]["reason"]


def test_same_time_key_conflicts_are_reviewed_not_arbitrarily_ordered(tmp_path):
    database_path = tmp_path / "wave3.db"
    initialize_database(database_path)
    published_at = "2025-01-15T00:00:00+00:00"
    docs = [
        ("doc-a", "Cash balance was A$100 million at 31 December 2024.", 100),
        ("doc-b", "Cash balance was A$90 million at 31 December 2024.", 90),
    ]
    with database_connection(database_path) as connection:
        for document_id, source, value in docs:
            segment = segment_chunks([(0, source)])[0]
            _insert_cached_document(
                connection,
                document_id,
                published_at,
                source,
                _cash_fact(value, source, segment.segment_id),
            )

    summary = run_reconciliation(
        database_path,
        FakeClient(),
        model="gemma4",
        ticker="CXO",
        confirm_wave2_gate=True,
    )
    with database_connection(database_path) as connection:
        fact_count = connection.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
        review_count = connection.execute(
            "SELECT COUNT(*) FROM reconciliation_review WHERE status = 'pending'"
        ).fetchone()[0]

    assert summary.added == summary.confirmed == summary.superseded == 0
    assert summary.unresolved == 2
    assert fact_count == 0
    assert review_count == 2


def test_reconciliation_requires_wave2_gate_and_rejects_nimble(tmp_path):
    database_path = tmp_path / "does-not-exist.db"
    client = FakeClient()

    with pytest.raises(ValueError, match="Gate 2"):
        run_reconciliation(database_path, client, model="gemma4")
    with pytest.raises(ValueError, match="not approved"):
        run_reconciliation(
            database_path,
            client,
            model="nimble",
            confirm_wave2_gate=True,
        )


def test_preflight_is_read_only_and_reports_current_quotes(tmp_path):
    database_path = tmp_path / "wave3.db"
    initialize_database(database_path)
    document_id = FIXED_DOCUMENT_IDS[0]
    source = "Cash balance was A$100 million at 31 December 2024."
    segment = segment_chunks([(0, source)])[0]
    with database_connection(database_path) as connection:
        _insert_cached_document(
            connection,
            document_id,
            "2025-01-15T00:00:00+00:00",
            source,
            _cash_fact(100, source, segment.segment_id),
        )

    before = preflight_reconciliation(database_path, model="gemma4", ticker="CXO")
    with database_connection(database_path) as connection:
        fact_count = connection.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
        migration_version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]

    assert before.requested_documents == before.current_caches == 1
    assert before.quote_verified == 1
    assert before.quote_rejected == 0
    assert not before.missing
    assert fact_count == 0
    assert migration_version == 6


def test_preflight_excludes_explicitly_unsupported_forms(tmp_path):
    database_path = tmp_path / "wave3.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                triage_class, form_type
            ) VALUES (
                'unsupported-form', 'CXO', 'unsupported-form',
                'Unrecognized disclosure', '2025-01-15T00:00:00+00:00',
                'v1', 'form', 'other'
            )
            """
        )

    result = preflight_reconciliation(
        database_path,
        model="gemma4",
        ticker="CXO",
    )

    assert result.requested_documents == 0
    assert result.current_caches == 0
    assert result.missing == ()
    assert result.unsupported == ("unsupported-form",)
