import csv
from pathlib import Path

from agent_memory.db import database_connection, initialize_database
from agent_memory.pipeline.extract import (
    FORM_PROMPT_FILES,
    SUPPORTED_FORM_FAMILIES,
    normalize_form_family,
    run_extraction,
)


def test_supported_form_mapping_and_unsupported_notices():
    cases = (
        ("other", "Application for quotation of securities", "appendix_2a"),
        ("other", "Proposed issue of securities", "appendix_3b"),
        ("other", "Change of Director's Interest (Appendix 3Y)", "director_interest"),
        ("other", "Initial Director's Interest (Appendix 3Z)", "director_interest"),
        ("other", "Becoming a substantial holder (Form 603)", "substantial_holder"),
        ("other", "Change in substantial holding (Form 604)", "substantial_holder"),
        ("other", "Ceasing to be a substantial holder (Form 605)", "substantial_holder"),
        ("appendix_3b", "Unrelated headline", "appendix_3b"),
    )
    for form_type, headline, expected in cases:
        assert normalize_form_family(form_type, headline) == expected

    assert set(SUPPORTED_FORM_FAMILIES) == {
        "appendix_2a",
        "appendix_3b",
        "director_interest",
        "substantial_holder",
    }
    assert set(FORM_PROMPT_FILES) == set(SUPPORTED_FORM_FAMILIES)
    prompt_directory = Path(__file__).resolve().parents[1] / "prompts" / "forms"
    assert all(
        (prompt_directory / filename).is_file()
        for filename in FORM_PROMPT_FILES.values()
    )
    assert normalize_form_family("other", "Appendix 3H notice of cessation") is None


def test_unsupported_forms_are_reported_without_model_calls(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source,
                triage_class, form_type
            ) VALUES (
                'notice-1', 'CXO', 'notice-1',
                'Notification of cessation of securities',
                '2026-01-01T00:00:00+00:00', 'missing', 'form', 'other'
            )
            """
        )
    report_path = tmp_path / "var" / "eval" / "unsupported_forms.csv"

    class UnexpectedClient:
        model_name = "qwen3.5:9b"

        def chat_json(self, *_args, **_kwargs):
            del _args, _kwargs
            raise AssertionError("unsupported notices must not call the model")

    summary = run_extraction(
        database_path,
        UnexpectedClient(),
        unsupported_path=report_path,
    )

    with report_path.open(encoding="utf-8", newline="") as report_file:
        rows = list(csv.DictReader(report_file))
    assert summary.unsupported == 1
    assert summary.failed == 0
    assert rows == [
        {
            "document_id": "notice-1",
            "headline": "Notification of cessation of securities",
            "form_type": "other",
        }
    ]
