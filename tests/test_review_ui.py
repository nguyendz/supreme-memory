import json

from agent_memory.db import database_connection, initialize_database
from agent_memory.model_comparison import FIXED_DOCUMENT_IDS, LOCAL_MODELS
from agent_memory.pipeline.extract import (
    CachedExtraction,
    CandidateEntity,
    CandidateFact,
    ExtractionUsage,
    RawSegmentOutput,
    _source_hash,
    segment_chunks,
)
from agent_memory.review_ui import generate_model_review_ui


SOURCE = "Quarterly cash balance was A$200 million at 30 September 2026."


def test_model_review_ui_embeds_comparison_and_escapes_source_html(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        for doc_id in FIXED_DOCUMENT_IDS:
            headline = "Quarterly report"
            if doc_id == FIXED_DOCUMENT_IDS[0]:
                headline = "Report </script><script>alert(1)</script>"
            connection.execute(
                """
                INSERT INTO document (
                    id, ticker, folder, headline, published_at, text_source,
                    triage_class
                ) VALUES (?, 'CXO', ?, ?, '2026-01-01T00:00:00+00:00', 'v1', 'full')
                """,
                (doc_id, doc_id, headline),
            )
            connection.execute(
                """
                INSERT INTO chunk (
                    id, doc_id, ticker, published_at, ord, text
                ) VALUES (?, ?, 'CXO', '2026-01-01T00:00:00+00:00', 0, ?)
                """,
                (f"{doc_id}:00000", doc_id, SOURCE),
            )

        doc_id = FIXED_DOCUMENT_IDS[0]
        segment = segment_chunks([(0, SOURCE)])[0]
        cached = CachedExtraction(
            source_doc_id=doc_id,
            source_form_type=None,
            form_family="full",
            source_hash=_source_hash([segment]),
            facts=[
                CandidateFact(
                    entity=CandidateEntity(name="Core Lithium", type="company"),
                    attribute="cash_balance",
                    value="A$200 million",
                    value_num=200_000_000,
                    unit="AUD",
                    qualifier=None,
                    as_of="2026-09-30",
                    period="2026-Q3",
                    fact_type="metric",
                    tier="T1",
                    salience=0.9,
                    text="Cash balance was A$200 million.",
                    quote=SOURCE,
                    segment_id=segment.segment_id,
                )
            ],
            raw_segments=[
                RawSegmentOutput(
                    segment_id=segment.segment_id,
                    raw_content=json.dumps({"facts": []}),
                )
            ],
            usage=ExtractionUsage(
                prompt_tokens=12,
                output_tokens=5,
                attempts=1,
                wall_duration_ms=1000,
            ),
        )
        connection.execute(
            """
            INSERT INTO extraction_cache (
                doc_id, model, prompt_version, output_json, created_at
            ) VALUES (?, ?, 'test-v1', ?, '2026-01-01T00:00:00+00:00')
            """,
            (doc_id, LOCAL_MODELS[0], cached.model_dump_json()),
        )

    output_path = generate_model_review_ui(
        database_path,
        tmp_path / "review" / "model-review.html",
    )
    page = output_path.read_text(encoding="utf-8")

    assert "cash_balance" in page
    assert SOURCE in page
    assert all(model in page for model in LOCAL_MODELS)
    assert all(doc_id in page for doc_id in FIXED_DOCUMENT_IDS)
    assert r"\u003c/script\u003e\u003cscript\u003ealert(1)" in page
    assert "</script><script>alert(1)" not in page
    assert ').join("\\n");' in page
    assert "Download review JSON" in page
