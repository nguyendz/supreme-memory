import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent_memory.db import database_connection, initialize_database
from agent_memory.evaluation import (
    GoldQuestion,
    GoldSource,
    _available_source_ids,
    load_gold_set,
    run_evaluation,
)
from agent_memory.query.baseline_rag import build_baseline_index


SOURCE_TEXT = "Cash balance was A$200 million at 30 September 2026."


class FakeEvaluationClient:
    model_name = "qwen3.5:9b"

    def embed(self, texts, *, doc_id=None):
        self.last_doc_id = doc_id
        return SimpleNamespace(
            vectors=[[1.0, 0.0, 0.0] for _ in texts],
            dimensions=3,
            wall_duration_ms=1,
        )

    def chat_json(self, _prompt, response_model, **_kwargs):
        del _prompt, _kwargs
        if response_model.__name__ == "AnswerResponse":
            payload = {
                "answer": "The cash balance was A$200 million.",
                "citations": [{"document_id": "doc-1", "quote": SOURCE_TEXT}],
            }
        else:
            payload = {"correct": True, "rationale": "The essential value matches."}
        raw_content = json.dumps(payload)
        return SimpleNamespace(
            value=response_model.model_validate_json(raw_content),
            raw_content=raw_content,
            prompt_tokens=100,
            output_tokens=20,
            attempts=1,
            wall_duration_ms=2,
        )


def _baseline(baseline_path: Path, client):
    source_path = baseline_path.with_name("source.db")
    initialize_database(source_path)
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
        connection.execute(
            """
            INSERT INTO chunk (
                id, doc_id, ticker, published_at, ord, text
            ) VALUES (
                'doc-1:00000', 'doc-1', 'CXO',
                '2026-10-13T21:39:00+00:00', 0, ?
            )
            """,
            (SOURCE_TEXT,),
        )
    stats = build_baseline_index(source_path, baseline_path, client)
    assert stats.embedded_chunks == 1


def test_gold_set_loads_the_sourced_yaml_questions():
    question_path = Path(__file__).resolve().parents[1] / "eval" / "questions.yaml"

    gold_set = load_gold_set(question_path)

    assert len(gold_set.questions) == 25
    assert gold_set.ticker == "CXO"


def test_source_in_catalog_without_chunks_is_out_of_corpus(tmp_path):
    database_path = tmp_path / "catalog.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source
            ) VALUES (
                'catalog-only', 'CXO', 'catalog-only', 'Old source',
                '2023-02-01T00:00:00+00:00', 'missing'
            )
            """
        )
    question = GoldQuestion(
        id="out-of-slice",
        category="point_in_time",
        knowledge_date=date(2023, 2, 1),
        question="What was reported?",
        expected_answer="A fact.",
        sources=[GoldSource(document_id="catalog-only", evidence="A fact.")],
    )

    assert _available_source_ids(database_path, [question]) == {}


def test_evaluation_reports_deferred_questions_and_validity_metrics(tmp_path):
    client = FakeEvaluationClient()
    baseline_path = tmp_path / "baseline.db"
    _baseline(baseline_path, client)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "ticker": "CXO",
                "questions": [
                    {
                        "id": "in-slice",
                        "category": "current_state",
                        "knowledge_date": "2026-10-15",
                        "question": "What was cash at 30 September?",
                        "expected_answer": "A$200 million.",
                        "sources": [
                            {"document_id": "doc-1", "evidence": SOURCE_TEXT}
                        ],
                    },
                    {
                        "id": "out-of-slice",
                        "category": "current_state",
                        "knowledge_date": "2023-02-01",
                        "question": "What was cash in 2023?",
                        "expected_answer": "Not available.",
                        "sources": [
                            {
                                "document_id": "20230201-old-source",
                                "evidence": "Historic source evidence.",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "var" / "eval" / "baseline.md"

    summary = run_evaluation(
        baseline_path,
        client,
        questions_path,
        report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert summary.loaded == 2
    assert summary.evaluated == 1
    assert summary.correct == 1
    assert summary.deferred == 1
    assert summary.citation_date_leaks == 0
    assert summary.invalid_quotes == 0
    assert "out-of-slice" in report
    assert "Returned citation-date validity: 1/1 valid; 0 out-of-date citation(s)" in report
