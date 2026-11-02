import csv
import json
from types import SimpleNamespace

from agent_memory.db import database_connection, initialize_database
from agent_memory.pipeline.extract import extract_document, segment_chunks
from agent_memory.pipeline.review import FIXED_QUARTERLY_IDS, export_t1_review


SOURCE = "Quarterly cash balance was A$200 million at 30 September 2026."


class FakeClient:
    model_name = "qwen3.5:9b"

    def __init__(self, response):
        self.response = response

    def chat_json(self, _prompt, response_model, **_kwargs):
        del _prompt, _kwargs
        raw_content = json.dumps(self.response)
        return SimpleNamespace(
            value=response_model.model_validate_json(raw_content),
            raw_content=raw_content,
            prompt_tokens=10,
            output_tokens=10,
            attempts=1,
            wall_duration_ms=1,
        )


def _fact(segment_id, quote, attribute):
    return {
        "entity": {"name": "Core Lithium", "type": "company"},
        "attribute": attribute,
        "value": "A$200 million",
        "value_num": 200000000,
        "unit": "AUD",
        "qualifier": None,
        "as_of": "2026-09-30",
        "period": "2026-Q3",
        "fact_type": "metric",
        "tier": "T1",
        "salience": 0.9,
        "text": "Cash balance was A$200 million.",
        "quote": quote,
        "segment_id": segment_id,
    }


def test_t1_review_export_includes_only_current_quote_verified_candidates(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    doc_id = FIXED_QUARTERLY_IDS[0]
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                triage_class
            ) VALUES (?, 'CXO', ?, 'Quarterly report',
                      '2025-01-22T21:45:00+00:00', 'v1', 'full')
            """,
            (doc_id, doc_id),
        )
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text
            ) VALUES (?, ?, 'CXO', '2025-01-22T21:45:00+00:00', 0, ?)
            """,
            (f"{doc_id}:00000", doc_id, SOURCE),
        )
    segment_id = segment_chunks([(0, SOURCE)])[0].segment_id
    response = {
        "facts": [
            _fact(segment_id, SOURCE, "cash_balance"),
            _fact(segment_id, "Invented evidence not in source.", "invented"),
        ]
    }
    extract_document(database_path, FakeClient(response), doc_id)
    output_path = tmp_path / "var" / "eval" / "t1-review.csv"

    summary = export_t1_review(database_path, "qwen3.5:9b", output_path)

    with output_path.open(encoding="utf-8", newline="") as review_file:
        rows = list(csv.DictReader(review_file))
    assert summary.documents == 5
    assert summary.cached_documents == 1
    assert summary.candidates == 1
    assert len(summary.stale_or_missing) == 4
    assert rows[0]["attribute"] == "cash_balance"
    assert rows[0]["quote"] == SOURCE
    assert rows[0]["review_status"] == ""
