import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_memory.config import AppConfig
from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient, LLMOutputError
from agent_memory.pipeline.extract import (
    CandidateFact,
    ExtractionOutput,
    extract_document,
    segment_chunks,
)


SOURCE = "Cash balance was A$200 million at 30 September 2026."


def _config(tmp_path: Path) -> AppConfig:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    return AppConfig(
        config_path=config_path,
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "memory.db",
        ollama_host="http://localhost:11434",
        ollama_model="qwen3.5:9b",
        embedding_model="qwen3-embedding",
        embedding_dimensions=None,
        num_ctx=32768,
        num_predict=1024,
        keep_alive="30m",
        timeout_seconds=30,
        skip_headline_patterns=(),
    )


def _insert_document(
    connection,
    doc_id: str,
    *,
    published_at: str = "2026-10-13T21:39:00+00:00",
    headline: str = "Quarterly Activities and Cashflow Report",
    triage_class: str = "full",
    form_type: str | None = None,
    source: str = SOURCE,
):
    connection.execute(
        """
        INSERT INTO document (
            id, ticker, folder, headline, published_at, text_source, triage_class,
            form_type, types_json
        ) VALUES (?, 'CXO', ?, ?, ?, 'v1', ?, ?, '[]')
        """,
        (doc_id, doc_id, headline, published_at, triage_class, form_type),
    )
    connection.execute(
        """
        INSERT INTO chunk (
            id, doc_id, ticker, published_at, ord, text, is_boilerplate
        ) VALUES (?, ?, 'CXO', ?, 0, ?, 0)
        """,
        (f"{doc_id}:00000", doc_id, published_at, source),
    )


def _fact_payload(segment_id: str, *, text: str = "Cash balance at quarter end."):
    return {
        "facts": [
            {
                "entity": {"name": "Core Lithium", "type": "company"},
                "attribute": "cash_balance",
                "value": "A$200 million",
                "value_num": 200000000,
                "unit": "AUD",
                "qualifier": None,
                "as_of": "2026-09-30",
                "period": "2026-Q3",
                "fact_type": "metric",
                "tier": "T1",
                "salience": 0.95,
                "text": text,
                "quote": SOURCE,
                "segment_id": segment_id,
            }
        ]
    }


class FakeExtractionClient:
    model_name = "qwen3.5:9b"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat_json(self, prompt, response_model, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        raw_content = output if isinstance(output, str) else json.dumps(output)
        value = response_model.model_validate_json(raw_content)
        return SimpleNamespace(
            value=value,
            raw_content=raw_content,
            prompt_tokens=100,
            output_tokens=50,
            attempts=1,
            wall_duration_ms=25.0,
        )


def _database(tmp_path: Path, documents):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        for doc in documents:
            _insert_document(connection, **doc)
    return database_path


def test_candidate_fact_schema_requires_supported_types_and_tiers():
    segment_id = segment_chunks([(0, SOURCE)])[0].segment_id
    valid = _fact_payload(segment_id)["facts"][0]
    assert CandidateFact.model_validate(valid).tier == "T1"

    invalid_tier = {**valid, "tier": "T4"}
    with pytest.raises(ValidationError):
        CandidateFact.model_validate(invalid_tier)
    invalid_type = {**valid, "fact_type": "unknown"}
    with pytest.raises(ValidationError):
        CandidateFact.model_validate(invalid_type)
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate({"facts": [{"entity": {"name": "Core"}}]})


def test_empty_extraction_is_valid():
    assert ExtractionOutput.model_validate({"facts": []}).facts == []


def test_extraction_caches_raw_json_and_replays_without_model_call(tmp_path):
    segment_id = segment_chunks([(0, SOURCE)])[0].segment_id
    raw_content = json.dumps(_fact_payload(segment_id), separators=(",", ":"))
    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])
    client = FakeExtractionClient([raw_content])

    first = extract_document(database_path, client, "doc-1")
    second = extract_document(database_path, client, "doc-1")

    assert first.accepted_facts[0].value == "A$200 million"
    assert not first.cache_hit
    assert second.cache_hit
    assert len(client.calls) == 1
    assert client.calls[0]["num_predict"] == 8192
    with database_connection(database_path) as connection:
        cache = connection.execute(
            "SELECT output_json FROM extraction_cache"
        ).fetchone()
        stored = json.loads(cache["output_json"])
        persisted_fact_rows = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("fact", "fact_source", "entity", "attribute")
        ]
    assert stored["raw_segments"][0]["raw_content"] == raw_content
    assert stored["source_doc_id"] == "doc-1"
    assert persisted_fact_rows == [0, 0, 0, 0]


def test_extraction_binds_evidence_to_the_segment_used_for_the_call(tmp_path):
    wrong_segment_payload = _fact_payload("model-returned-wrong-segment")
    client = FakeExtractionClient([wrong_segment_payload])
    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])

    result = extract_document(database_path, client, "doc-1")

    assert result.accepted_facts[0].segment_id == segment_chunks([(0, SOURCE)])[0].segment_id


def test_recovery_caps_facts_and_preserves_raw_model_output(tmp_path):
    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])
    segment_id = segment_chunks([(0, SOURCE)])[0].segment_id
    facts = []
    for index in range(45):
        fact = _fact_payload(segment_id)["facts"][0]
        fact["attribute"] = f"attribute_{index}"
        fact["text"] = f"Candidate fact {index}."
        fact["tier"] = "T1" if index >= 40 else "T3"
        fact["salience"] = 0.9 if index >= 40 else 0.5
        facts.append(fact)
    raw_content = json.dumps({"facts": facts}, separators=(",", ":"))
    client = FakeExtractionClient([raw_content])

    result = extract_document(
        database_path,
        client,
        "doc-1",
        max_segment_source_characters=15_000,
    )

    assert len(result.facts) == 15
    assert sum(fact.tier == "T1" for fact in result.facts) == 5
    assert result.truncated_fact_count == 30
    with database_connection(database_path) as connection:
        cache = json.loads(
            connection.execute(
                "SELECT output_json FROM extraction_cache"
            ).fetchone()["output_json"]
        )
    assert cache["truncated_fact_count"] == 30
    assert len(json.loads(cache["raw_segments"][0]["raw_content"])["facts"]) == 45


def test_invalid_json_retries_with_smaller_segments_and_caches_fallback(tmp_path):
    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])
    client = FakeExtractionClient(
        [LLMOutputError("invalid JSON after one retry"), {"facts": []}]
    )

    first = extract_document(database_path, client, "doc-1")
    second = extract_document(database_path, client, "doc-1")

    assert first.status == "extracted"
    assert not first.cache_hit
    assert second.cache_hit
    assert len(client.calls) == 2
    assert "no more than 15 distinct facts" in client.calls[1]["system_prompt"]
    with database_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT output_json FROM extraction_cache"
        ).fetchall()
    cached = [json.loads(row["output_json"]) for row in rows]
    assert len(cached) == 1
    assert cached[0]["segment_source_characters"] == 15_000


def test_extraction_steps_down_again_when_first_fallback_is_truncated(tmp_path):
    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])
    client = FakeExtractionClient(
        [
            LLMOutputError("default segment was truncated"),
            LLMOutputError("first fallback segment was truncated"),
            {"facts": []},
        ]
    )

    result = extract_document(database_path, client, "doc-1")

    assert result.status == "extracted"
    assert len(client.calls) == 3
    with database_connection(database_path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT output_json FROM extraction_cache"
            ).fetchone()["output_json"]
        )
    assert payload["segment_source_characters"] == 7_500


def test_malformed_model_json_retries_and_caches_only_final_response(tmp_path):
    segment_id = segment_chunks([(0, SOURCE)])[0].segment_id
    raw_content = json.dumps(_fact_payload(segment_id))

    class FakeOllamaClient:
        def __init__(self):
            self.responses = ["not-json", raw_content]
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                prompt_eval_count=100,
                eval_count=50,
                prompt_eval_duration=1_000_000,
                eval_duration=2_000_000,
                message=SimpleNamespace(content=self.responses.pop(0)),
            )

    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])
    config = _config(tmp_path)
    client = LLMClient(config, database_path, client=FakeOllamaClient())

    result = extract_document(database_path, client, "doc-1")

    assert result.attempts == 2
    assert len(client._client.calls) == 2
    with database_connection(database_path) as connection:
        cached = json.loads(
            connection.execute(
                "SELECT output_json FROM extraction_cache"
            ).fetchone()["output_json"]
        )
        call_count = connection.execute(
            "SELECT COUNT(*) AS count FROM llm_call"
        ).fetchone()["count"]
    assert cached["raw_segments"][0]["raw_content"] == raw_content
    assert call_count == 2


def test_failed_later_segment_does_not_cache_partial_output(tmp_path, monkeypatch):
    import agent_memory.pipeline.extract as extract_module

    original_segment_chunks = extract_module.segment_chunks
    monkeypatch.setattr(
        extract_module,
        "segment_chunks",
        lambda chunks, *, max_source_characters=40: original_segment_chunks(
            chunks, max_source_characters=min(max_source_characters, 40)
        ),
    )
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        _insert_document(connection, "doc-1", source="First paragraph with evidence.")
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text, is_boilerplate
            ) VALUES ('doc-1:00001', 'doc-1', 'CXO',
                      '2026-10-13T21:39:00+00:00', 1,
                      'Second paragraph with more evidence.', 0)
            """
        )
    segments = segment_chunks(
        [(0, "First paragraph with evidence."), (1, "Second paragraph with more evidence.")],
        max_source_characters=40,
    )
    first_output = {"facts": []}
    client = FakeExtractionClient([first_output, ValueError("second segment failed")])

    with pytest.raises(ValueError, match="second segment failed"):
        extract_document(database_path, client, "doc-1")

    with database_connection(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM extraction_cache"
        ).fetchone()["count"]
    assert len(segments) == 2
    assert len(client.calls) == 2
    assert count == 0


def test_point_in_time_context_excludes_equal_time_and_future_sources(tmp_path):
    database_path = _database(
        tmp_path,
        [
            {
                "doc_id": "prior",
                "published_at": "2026-10-13T20:00:00+00:00",
                "source": SOURCE,
            },
            {
                "doc_id": "same-time",
                "published_at": "2026-10-13T21:39:00+00:00",
                "source": SOURCE,
            },
            {
                "doc_id": "future",
                "published_at": "2026-10-14T00:00:00+00:00",
                "source": SOURCE,
            },
            {
                "doc_id": "current",
                "published_at": "2026-10-13T21:39:00+00:00",
                "source": SOURCE,
            },
        ],
    )
    segment_id = segment_chunks([(0, SOURCE)])[0].segment_id
    outputs = [
        _fact_payload(segment_id, text="PRIOR_SENTINEL"),
        _fact_payload(segment_id, text="SAME_TIME_SENTINEL"),
        _fact_payload(segment_id, text="FUTURE_SENTINEL"),
        {"facts": []},
    ]
    client = FakeExtractionClient(outputs)
    for doc_id in ("prior", "same-time", "future"):
        extract_document(database_path, client, doc_id)

    extract_document(database_path, client, "current")
    current_prompt = client.calls[-1]["prompt"]

    assert "PRIOR_SENTINEL" in current_prompt
    assert "SAME_TIME_SENTINEL" not in current_prompt
    assert "FUTURE_SENTINEL" not in current_prompt


def test_segmenter_splits_on_heading_and_token_budget():
    segments = segment_chunks(
        [
            (0, "Opening facts."),
            (1, "# Production"),
            (2, "Production facts."),
            (3, "Later details."),
        ],
        max_source_characters=40,
    )

    assert len(segments) >= 2
    assert all(len(segment.text) <= 40 for segment in segments)
    assert len({segment.segment_id for segment in segments}) == len(segments)


def test_segmenter_groups_small_heading_sections_into_one_call():
    segments = segment_chunks(
        [(index, f"# Section {index}\nSmall section text.") for index in range(10)],
        max_source_characters=500,
    )

    assert len(segments) == 1


def test_extraction_client_failure_does_not_cache(tmp_path):
    database_path = _database(tmp_path, [{"doc_id": "doc-1"}])
    client = FakeExtractionClient(
        [
            LLMOutputError("invalid JSON after retry"),
            LLMOutputError("invalid JSON after 15k fallback"),
            LLMOutputError("invalid JSON after 7.5k fallback"),
            LLMOutputError("invalid JSON after 3.75k fallback"),
            LLMOutputError("invalid JSON after 1.875k fallback"),
        ]
    )

    with pytest.raises(LLMOutputError):
        extract_document(database_path, client, "doc-1")

    with database_connection(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_cache"
        ).fetchone()[0] == 0
    assert len(client.calls) == 5
