import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import EmbeddingResult
from agent_memory.pipeline.extract import (
    CachedExtraction,
    CandidateEntity,
    CandidateFact,
    ExtractionUsage,
    RawSegmentOutput,
    _prompt_versions,
    _source_hash,
    segment_chunks,
)
from agent_memory.pipeline.reconcile import (
    PILOT_DOCUMENT_ID,
    preflight_pilot_reconciliation,
    run_pilot_reconciliation,
)


SOURCE = (
    "Cash balance of A$200 million at 30 September 2026 "
    "(30 June 2026: A$182 million)."
)


class FakeClient:
    model_name = "gemma4"

    def __init__(self):
        self.embed_calls = 0

    def embed(self, texts, *, doc_id=None):
        del doc_id
        self.embed_calls += 1
        if isinstance(texts, str):
            texts = [texts]
        return EmbeddingResult(
            vectors=[[1.0, 0.0, 0.0] for _ in texts],
            dimensions=3,
            wall_duration_ms=1,
        )

def _insert_pilot_source(database_path: Path):
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                triage_class
            ) VALUES (
                ?, 'CXO', ?, 'September quarterly',
                '2026-10-13T21:39:00+00:00', 'v1', 'full'
            )
            """,
            (PILOT_DOCUMENT_ID, PILOT_DOCUMENT_ID),
        )
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text
            ) VALUES (
                ?, ?, 'CXO', '2026-10-13T21:39:00+00:00', 0, ?
            )
            """,
            (f"{PILOT_DOCUMENT_ID}:00000", PILOT_DOCUMENT_ID, SOURCE),
        )
        segment = segment_chunks([(0, SOURCE)])[0]
        candidate = CandidateFact(
            entity=CandidateEntity(name="Core Lithium", type="company"),
            attribute="cash_balance",
            value="A$200 million",
            value_num=200_000_000,
            unit="AUD",
            qualifier=None,
            as_of=date(2026, 9, 30),
            period="2026-Q3",
            fact_type="metric",
            tier="T1",
            salience=0.9,
            text="Cash balance was A$200 million at 30 September 2026.",
            quote="Cash balance of A$200 million at 30 September 2026",
            segment_id=segment.segment_id,
        )
        prompt_text = (
            Path(__file__).resolve().parents[1] / "prompts" / "extract.md"
        ).read_text(encoding="utf-8")
        _, prompt_version = _prompt_versions(
            prompt_text=prompt_text,
            family_prompt="",
            family="full",
        )
        cached = CachedExtraction(
            source_doc_id=PILOT_DOCUMENT_ID,
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
            ) VALUES (?, 'gemma4', ?, ?, '2026-10-13T21:40:00+00:00')
            """,
            (PILOT_DOCUMENT_ID, prompt_version, cached.model_dump_json()),
        )
    with database_connection(database_path) as connection:
        connection.execute("DROP TABLE reconciliation_review")
        connection.execute("DROP TABLE reconciliation_cache")
        connection.execute("DELETE FROM schema_migrations WHERE version = 4")


def _canonical_counts(database_path: Path) -> dict[str, int]:
    with database_connection(database_path) as connection:
        return {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("entity", "attribute", "fact", "fact_source")
        }


def test_pilot_preflight_is_read_only_and_reports_gold_coverage(tmp_path):
    source_path = tmp_path / "source.db"
    pilot_path = tmp_path / "pilot.db"
    _insert_pilot_source(source_path)

    result = preflight_pilot_reconciliation(
        source_path,
        document_id=PILOT_DOCUMENT_ID,
        model="gemma4",
    )
    with database_connection(source_path) as connection:
        schema_version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]

    assert result.eligible
    assert result.matching_candidates == 1
    assert result.prompt_version
    assert not pilot_path.exists()
    assert schema_version == 3
    assert set(_canonical_counts(source_path).values()) == {0}


def test_pilot_reconciles_one_document_only_in_a_new_copy(tmp_path):
    source_path = tmp_path / "source.db"
    pilot_path = tmp_path / "var" / "wave3" / "pilot.db"
    _insert_pilot_source(source_path)
    original_cache_count = 1
    client = FakeClient()

    summary = run_pilot_reconciliation(
        source_path,
        pilot_path,
        client,
        document_id=PILOT_DOCUMENT_ID,
        model="gemma4",
    )

    assert summary.documents == summary.processed == 1
    assert (summary.added, summary.confirmed, summary.superseded) == (1, 0, 0)
    assert client.embed_calls >= 2
    assert not any(_canonical_counts(source_path).values())
    with database_connection(source_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_cache"
        ).fetchone()[0] == original_cache_count
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0] == 3
    assert _canonical_counts(pilot_path)["fact"] == 1
    with database_connection(pilot_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM document WHERE triage_class IN ('full', 'form')"
        ).fetchone()[0] == 1


def test_pilot_refuses_existing_destination_and_non_gold_source(tmp_path):
    source_path = tmp_path / "source.db"
    pilot_path = tmp_path / "pilot.db"
    _insert_pilot_source(source_path)
    pilot_path.write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError, match="already exists"):
        run_pilot_reconciliation(
            source_path,
            pilot_path,
            FakeClient(),
            document_id=PILOT_DOCUMENT_ID,
            model="gemma4",
        )
    with pytest.raises(ValueError, match="only gold source"):
        preflight_pilot_reconciliation(
            source_path,
            document_id="other-document",
            model="gemma4",
        )


def test_pilot_preflight_rejects_source_after_knowledge_date(tmp_path):
    source_path = tmp_path / "source.db"
    _insert_pilot_source(source_path)
    with database_connection(source_path) as connection:
        connection.execute(
            "UPDATE document SET published_at = ? WHERE id = ?",
            ("2026-10-16T00:00:00+00:00", PILOT_DOCUMENT_ID),
        )

    result = preflight_pilot_reconciliation(
        source_path,
        document_id=PILOT_DOCUMENT_ID,
        model="gemma4",
    )

    assert not result.eligible
    assert "gold ticker/knowledge date" in result.reason


def test_cli_pilot_preflight_is_read_only(tmp_path):
    source_path = tmp_path / "source.db"
    _insert_pilot_source(source_path)

    result = CliRunner().invoke(
        app,
        [
            "reconcile",
            "--source-db",
            str(source_path),
            "--pilot",
            "--document-id",
            PILOT_DOCUMENT_ID,
            "--model",
            "gemma4",
            "--preflight",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pilot preflight: ready" in result.output
    assert "No database was changed." in result.output
    assert not any(_canonical_counts(source_path).values())
