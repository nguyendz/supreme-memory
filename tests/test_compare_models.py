import json
from types import SimpleNamespace

from agent_memory.db import database_connection, initialize_database
from agent_memory.model_comparison import (
    FIXED_DOCUMENT_IDS,
    LOCAL_MODELS,
    run_model_comparison,
)


_FIXED_FORM_HEADLINES = {
    FIXED_DOCUMENT_IDS[5]: "Application for quotation of securities",
    FIXED_DOCUMENT_IDS[6]: "Proposed issue of securities",
    FIXED_DOCUMENT_IDS[7]: "Change of Director's Interest (Appendix 3Y)",
    FIXED_DOCUMENT_IDS[8]: "Becoming a substantial holder (Form 603)",
    FIXED_DOCUMENT_IDS[9]: "Change in substantial holding (Form 604)",
}


class FakeModelClient:
    model_name = "qwen3.5:9b"

    def __init__(self):
        self.models = []

    def chat_json(self, _prompt, response_model, **kwargs):
        del _prompt
        self.models.append(kwargs["model"])
        raw_content = json.dumps({"facts": []})
        return SimpleNamespace(
            value=response_model.model_validate_json(raw_content),
            raw_content=raw_content,
            prompt_tokens=10,
            output_tokens=5,
            attempts=1,
            wall_duration_ms=1,
        )


def test_model_comparison_uses_all_fixed_sources_and_local_models(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        for doc_id in FIXED_DOCUMENT_IDS:
            is_form = doc_id in _FIXED_FORM_HEADLINES
            headline = _FIXED_FORM_HEADLINES.get(doc_id, "Quarterly operational report")
            form_type = (
                "other"
                if doc_id
                in (FIXED_DOCUMENT_IDS[7], FIXED_DOCUMENT_IDS[8], FIXED_DOCUMENT_IDS[9])
                else (
                    "appendix_2a"
                    if doc_id == FIXED_DOCUMENT_IDS[5]
                    else "appendix_3b" if doc_id == FIXED_DOCUMENT_IDS[6] else None
                )
            )
            connection.execute(
                """
                INSERT INTO document (
                    id, ticker, folder, headline, published_at, text_source,
                    triage_class, form_type
                ) VALUES (?, 'CXO', ?, ?, '2026-01-01T00:00:00+00:00', 'v1',
                          ?, ?)
                """,
                (doc_id, doc_id, headline, "form" if is_form else "full", form_type),
            )
            connection.execute(
                """
                INSERT INTO chunk (
                    id, doc_id, ticker, published_at, ord, text
                ) VALUES (?, ?, 'CXO', '2026-01-01T00:00:00+00:00', 0,
                          'Source evidence for this fixed sample document.')
                """,
                (f"{doc_id}:00000", doc_id),
            )

    report_path = tmp_path / "models.md"
    client = FakeModelClient()
    rows = run_model_comparison(database_path, client, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert len(rows) == 30
    assert len(client.models) == 30
    assert set(client.models) == set(LOCAL_MODELS)
    assert all(row.status == "extracted" for row in rows)
    assert all(document_id in report for document_id in FIXED_DOCUMENT_IDS)
