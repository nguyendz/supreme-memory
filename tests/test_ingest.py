import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx

from agent_memory.config import AppConfig
from agent_memory.db import database_connection
from agent_memory.llm import LLMClient
from agent_memory.pipeline.ingest import run_ingest


class FakeOllama:
    def __init__(self, *, fail_chat=False, triage_response=None):
        self.fail_chat = fail_chat
        self.triage_response = triage_response or (
            '{"class":"full","form_type":null,"reason":"Contains company information."}'
        )
        self.chat_calls = 0
        self.chat_models = []
        self.embed_calls = 0

    def chat(self, **kwargs):
        self.chat_calls += 1
        self.chat_models.append(kwargs["model"])
        if self.fail_chat:
            raise httpx.ConnectError("local model unavailable")
        return SimpleNamespace(
            prompt_eval_count=20,
            eval_count=8,
            prompt_eval_duration=1_000_000,
            eval_duration=1_000_000,
            message=SimpleNamespace(content=self.triage_response),
        )

    def embed(self, **kwargs):
        self.embed_calls += 1
        texts = kwargs["input"]
        return SimpleNamespace(
            embeddings=[
                [1.0, 0.0, 0.0] if "cash" in text.lower() else [0.0, 1.0, 0.0]
                for text in texts
            ],
            prompt_eval_count=len(texts),
        )


def _config(tmp_path: Path, data_dir: Path) -> AppConfig:
    return AppConfig(
        config_path=tmp_path / "config.toml",
        project_root=Path(__file__).resolve().parents[1],
        data_dir=data_dir,
        database_path=tmp_path / "var" / "memory.db",
        ollama_host="http://localhost:11434",
        ollama_model="qwen3.5:9b",
        embedding_model="qwen3-embedding",
        embedding_dimensions=None,
        num_ctx=32768,
        num_predict=1024,
        keep_alive="30m",
        timeout_seconds=30,
        skip_headline_patterns=(r"cleansing",),
    )


def _write_announcement(
    data_dir: Path,
    folder_name: str,
    headline: str,
    text: str | None,
) -> None:
    folder = data_dir / "CXO" / "announcements" / folder_name
    version_dir = folder / "1.0.0"
    version_dir.mkdir(parents=True)
    metadata = {
        "ticker": "CXO",
        "company": "Core Lithium Ltd",
        "headline": headline,
        "category": "announcement",
        "announcement_types": [],
    }
    (version_dir / f"{folder_name}.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    if text is not None:
        (version_dir / f"{folder_name}.md").write_text(text, encoding="utf-8")


def _fixture_data(data_dir: Path) -> tuple[str, str, str, str]:
    earlier = "20241230-0100-Quarterly-Activities-Report-1"
    full = "20250121-0100-Quarterly-Operations-Report-2"
    skip = "20250122-0100-Cleansing-Notice-3"
    missing = "20250123-0100-Missing-Text-4"
    _write_announcement(
        data_dir,
        earlier,
        "Quarterly Activities Report",
        "About Core Lithium.\n\nFinniss cash balance was A$200 million.",
    )
    _write_announcement(
        data_dir,
        full,
        "Quarterly Operations Report",
        (
            "About Core Lithium.\n\n"
            'Refer to ASX announcement "Quarterly Activities Report" '
            "on 30 December 2024.\n\nFinniss cash balance was A$200 million."
        ),
    )
    _write_announcement(data_dir, skip, "Cleansing Notice", "Administrative notice.")
    _write_announcement(data_dir, missing, "Quarterly Missing Text", None)
    return earlier, full, skip, missing


def _llm(config: AppConfig, *, fail_chat=False, triage_response=None):
    fake = FakeOllama(fail_chat=fail_chat, triage_response=triage_response)
    return LLMClient(config, config.database_path, client=fake), fake


def _logical_snapshot(database_path: Path):
    with database_connection(database_path) as connection:
        documents = connection.execute(
            """
            SELECT id, triage_class, triage_reason, ingest_status, error
            FROM document ORDER BY id
            """
        ).fetchall()
        hashes = connection.execute(
            """
            SELECT scope, ticker, simhash, doc_id, published_at, sample_text
            FROM paragraph_hash_observation
            ORDER BY scope, ticker, simhash, doc_id
            """
        ).fetchall()
        aggregates = connection.execute(
            """
            SELECT scope, ticker, simhash, first_seen_at, doc_count, sample_text
            FROM paragraph_hash ORDER BY scope, ticker, simhash
            """
        ).fetchall()
        chunks = connection.execute(
            """
            SELECT id, doc_id, ticker, published_at, ord, text, is_boilerplate
            FROM chunk ORDER BY id
            """
        ).fetchall()
        vectors = connection.execute(
            "SELECT chunk_id, embedding FROM chunk_vec ORDER BY chunk_id"
        ).fetchall()
        xrefs = connection.execute(
            """
            SELECT src_doc, dst_doc, cited_title, cited_date, match_score
            FROM doc_xref ORDER BY src_doc, dst_doc, cited_title
            """
        ).fetchall()
        fts = connection.execute(
            "SELECT text FROM chunk_fts ORDER BY text"
        ).fetchall()
        return tuple(
            tuple(tuple(row) for row in rows)
            for rows in (documents, hashes, aggregates, chunks, vectors, xrefs, fts)
        )


def test_ingest_catalogs_all_docs_classifies_skips_missing_and_noops(tmp_path):
    data_dir = tmp_path / "data"
    _, full, skip, missing = _fixture_data(data_dir)
    config = _config(tmp_path, data_dir)
    client, fake = _llm(config)

    summary = run_ingest(
        config,
        ticker="CXO",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        client=client,
    )

    assert summary.discovered == 4
    assert summary.selected == 3
    assert summary.done == 2
    assert summary.skipped == 1
    assert summary.failed == 0
    assert summary.references == summary.resolved_references == 1
    assert fake.chat_calls == 1
    assert fake.embed_calls == 1
    with database_connection(config.database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, triage_class, ingest_status
            FROM document WHERE id IN (?, ?, ?) ORDER BY id
            """,
            (full, skip, missing),
        ).fetchall()
        assert [(row["triage_class"], row["ingest_status"]) for row in rows] == [
            ("full", "done"),
            ("skip", "done"),
            ("skip", "skipped"),
        ]
        assert connection.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_cache"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM doc_xref WHERE src_doc = ?", (full,)
        ).fetchone()[0] == 1
        chunks = connection.execute(
            "SELECT text, is_boilerplate FROM chunk WHERE doc_id = ? ORDER BY ord",
            (full,),
        ).fetchall()
        assert any(row["is_boilerplate"] == 1 for row in chunks if "About Core" in row["text"])
        assert any(
            row["is_boilerplate"] == 0
            for row in chunks
            if "cash balance" in row["text"].lower()
        )

    no_op_summary = run_ingest(
        config,
        ticker="CXO",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        client=client,
    )
    assert no_op_summary.no_op == 3
    assert no_op_summary.done == no_op_summary.skipped == 0
    assert fake.chat_calls == 1
    assert fake.embed_calls == 1


def test_reingest_rebuilds_the_same_logical_state(tmp_path):
    data_dir = tmp_path / "data"
    _fixture_data(data_dir)
    config = _config(tmp_path, data_dir)
    first_client, _ = _llm(config)
    run_ingest(
        config,
        ticker="CXO",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        client=first_client,
    )
    first_snapshot = _logical_snapshot(config.database_path)

    replay_client, _ = _llm(config)
    summary = run_ingest(
        config,
        ticker="CXO",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        reingest=True,
        client=replay_client,
    )

    assert summary.done == 2
    assert summary.skipped == 1
    assert summary.failed == 0
    assert _logical_snapshot(config.database_path) == first_snapshot


def test_warmup_makes_no_llm_calls_and_is_idempotent(tmp_path):
    data_dir = tmp_path / "data"
    _fixture_data(data_dir)
    config = _config(tmp_path, data_dir)
    client, fake = _llm(config)

    first = run_ingest(config, ticker="CXO", warmup=True, client=client)
    second = run_ingest(config, ticker="CXO", warmup=True, client=client)

    assert first.warmed_documents == 3
    assert second.warmed_documents == 0
    assert fake.chat_calls == fake.embed_calls == 0


def test_ingest_records_transport_errors_and_reports_failure(tmp_path):
    data_dir = tmp_path / "data"
    _, full, _, _ = _fixture_data(data_dir)
    config = _config(tmp_path, data_dir)
    client, fake = _llm(config, fail_chat=True)

    summary = run_ingest(
        config,
        ticker="CXO",
        start_date=date(2025, 1, 21),
        end_date=date(2025, 1, 21),
        client=client,
    )

    assert summary.failed == 1
    assert summary.errors[0][0] == full
    assert fake.chat_calls == 1
    with database_connection(config.database_path) as connection:
        status = connection.execute(
            "SELECT ingest_status, error FROM document WHERE id = ?", (full,)
        ).fetchone()
        failed_calls = connection.execute(
            "SELECT ok FROM llm_call WHERE doc_id = ? ORDER BY id", (full,)
        ).fetchall()
    assert status["ingest_status"] == "error"
    assert "local model unavailable" in status["error"]
    assert [row["ok"] for row in failed_calls] == [0]


def test_ingest_persists_form_type_for_wave_two(tmp_path):
    data_dir = tmp_path / "data"
    form_id = "20250121-0700-Appendix-3B-1"
    _write_announcement(data_dir, form_id, "Appendix 3B", "Standard form.")
    config = _config(tmp_path, data_dir)
    client, _ = _llm(
        config,
        triage_response=(
            '{"class":"form","form_type":"appendix_3b",'
            '"reason":"Prescribed securities filing."}'
        ),
    )

    summary = run_ingest(
        config,
        ticker="CXO",
        start_date=date(2025, 1, 21),
        end_date=date(2025, 1, 21),
        client=client,
    )

    assert summary.done == 1
    with database_connection(config.database_path) as connection:
        row = connection.execute(
            "SELECT triage_class, form_type FROM document WHERE id = ?", (form_id,)
        ).fetchone()
    assert tuple(row) == ("form", "appendix_3b")


def test_cli_date_validation_and_reingest_requirements(tmp_path):
    data_dir = tmp_path / "data"
    _fixture_data(data_dir)
    config = _config(tmp_path, data_dir)

    for kwargs, expected_message in (
        (
            {"start_date": date(2025, 2, 1), "end_date": date(2025, 1, 1)},
            "--from must be on or before --to",
        ),
        ({"reingest": True}, "--reingest requires --from"),
        ({"limit": 0}, "--limit must be positive"),
    ):
        try:
            run_ingest(config, **kwargs)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected validation error: {expected_message}")
