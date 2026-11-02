"""Gold-set evaluation for the chunk-RAG baseline."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
import ollama
import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_memory.db import database_connection
from agent_memory.llm import LLMClient
from agent_memory.pipeline.quotecheck import contains_exact_quote
from agent_memory.query.baseline_rag import BaselineAnswer, answer_baseline


class GoldSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    knowledge_date: date
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    sources: list[GoldSource]


class GoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    ticker: str = Field(min_length=1)
    questions: list[GoldQuestion]


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: bool
    rationale: str = Field(min_length=1)


@dataclass(frozen=True)
class EvaluationCase:
    question: GoldQuestion
    status: str
    correct: bool | None = None
    answer: str | None = None
    rationale: str | None = None
    citation_count: int = 0
    date_valid: bool = True
    quote_valid: bool = True
    context_tokens: int = 0
    error: str | None = None


@dataclass
class EvaluationSummary:
    loaded: int
    evaluated: int = 0
    correct: int = 0
    deferred: int = 0
    failed: int = 0
    citation_count: int = 0
    citation_date_leaks: int = 0
    invalid_quotes: int = 0
    context_tokens: int = 0
    cases: list[EvaluationCase] = field(default_factory=list)


def load_gold_set(path: Path) -> GoldSet:
    with path.open("r", encoding="utf-8") as question_file:
        value = yaml.safe_load(question_file)
    return GoldSet.model_validate(value)


def _available_source_ids(
    database_path: Path, questions: list[GoldQuestion]
) -> dict[str, date]:
    source_ids = sorted(
        {source.document_id for question in questions for source in question.sources}
    )
    if not source_ids:
        return set()
    placeholders = ", ".join("?" for _ in source_ids)
    with database_connection(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT document.id, document.published_at
            FROM document
            JOIN chunk ON chunk.doc_id = document.id
            WHERE document.id IN ({placeholders})
            """,
            source_ids,
        ).fetchall()
    return {
        row["id"]: date.fromisoformat(row["published_at"][:10])
        for row in rows
    }


def _judge_answer(
    client: LLMClient,
    question: GoldQuestion,
    answer: BaselineAnswer,
    *,
    prompt_path: Path,
    model: str | None,
) -> JudgeResponse:
    system_prompt = prompt_path.read_text(encoding="utf-8")
    prompt = (
        f"Question: {question.question}\n"
        f"Expected answer: {question.expected_answer}\n"
        f"Candidate answer: {answer.answer}"
    )
    return client.chat_json(
        prompt,
        JudgeResponse,
        system_prompt=system_prompt,
        model=model,
    ).value


def _citation_checks(
    database_path: Path,
    answer: BaselineAnswer,
) -> tuple[bool, bool]:
    if not answer.citations:
        return True, True
    dates_valid = True
    quotes_valid = True
    with database_connection(database_path) as connection:
        for citation in answer.citations:
            row = connection.execute(
                "SELECT published_at FROM document WHERE id = ?",
                (citation.document_id,),
            ).fetchone()
            if (
                row is None
                or date.fromisoformat(row["published_at"][:10])
                > answer.knowledge_date
            ):
                dates_valid = False
            if not any(
                chunk.doc_id == citation.document_id
                and contains_exact_quote(citation.quote, chunk.text)
                for chunk in answer.retrieved_chunks
            ):
                quotes_valid = False
    return dates_valid, quotes_valid


def _render_report(
    summary: EvaluationSummary,
    *,
    corpus_database_path: Path,
    question_path: Path,
) -> str:
    judged = [case for case in summary.cases if case.status == "evaluated"]
    eligible = [
        case for case in summary.cases if case.status in ("evaluated", "failed")
    ]
    accuracy = (
        f"{summary.correct}/{len(eligible)} "
        f"({summary.correct / len(eligible):.1%}; failed outputs count incorrect)"
        if eligible
        else "not measured (no eligible questions)"
    )
    date_valid_citations = summary.citation_count - summary.citation_date_leaks
    quote_valid_citations = summary.citation_count - summary.invalid_quotes
    average_context = (
        summary.context_tokens / len(judged) if judged else 0
    )
    lines = [
        "# Chunk-RAG baseline evaluation",
        "",
        f"- Corpus database: `{corpus_database_path}`",
        f"- Gold set: `{question_path}`",
        f"- Loaded questions: {summary.loaded}",
        f"- In-corpus eligible: {len(eligible)}",
        f"- Judge-completed: {summary.evaluated}",
        f"- Deferred/out of slice: {summary.deferred}",
        f"- Failed: {summary.failed}",
        f"- Answer accuracy: {accuracy}",
        (
            "- Returned citation-date validity: "
            f"{date_valid_citations}/{summary.citation_count} valid; "
            f"{summary.citation_date_leaks} out-of-date citation(s)"
        ),
        (
            "- Returned quote validity: "
            f"{quote_valid_citations}/{summary.citation_count} valid; "
            f"{summary.invalid_quotes} invalid quote(s)"
        ),
        f"- Average context tokens: {average_context:.0f}",
        "",
        "## Per-category results",
        "",
        "| Category | In corpus | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    by_category: dict[str, list[EvaluationCase]] = defaultdict(list)
    for case in summary.cases:
        if case.status in ("evaluated", "failed"):
            by_category[case.question.category].append(case)
    for category, cases in sorted(by_category.items()):
        correct = sum(case.correct is True for case in cases)
        lines.append(
            f"| {category} | {len(cases)} | {correct} | "
            f"{correct / len(cases):.1%} |"
        )

    skipped = [case for case in summary.cases if case.status == "deferred"]
    failures = [case for case in summary.cases if case.status == "failed"]
    if skipped:
        lines.extend(("", "## Deferred questions", ""))
        for case in skipped:
            lines.append(f"- `{case.question.id}`: {case.error}")
    if failures:
        lines.extend(("", "## Failed cases", ""))
        for case in failures:
            lines.append(f"- `{case.question.id}`: {case.error}")
            if case.answer:
                lines.append(f"  - Answer: {case.answer}")
            if case.rationale:
                lines.append(f"  - Judge: {case.rationale}")
    if not skipped and not failures:
        lines.extend(("", "## Failed and deferred cases", "", "None.", ""))
    return "\n".join(lines) + "\n"


def run_evaluation(
    database_path: Path,
    client: LLMClient,
    question_path: Path,
    report_path: Path,
    *,
    ticker: str | None = None,
    judge_prompt_path: Path | None = None,
    judge_model: str | None = None,
) -> EvaluationSummary:
    gold_set = load_gold_set(question_path)
    selected_ticker = (ticker or gold_set.ticker).upper()
    available_sources = _available_source_ids(database_path, gold_set.questions)
    evaluation_summary = EvaluationSummary(loaded=len(gold_set.questions))
    project_root = Path(__file__).resolve().parents[2]
    judge_path = judge_prompt_path or project_root / "prompts" / "judge.md"

    for question in gold_set.questions:
        eligible_sources = [
            source.document_id
            for source in question.sources
            if source.document_id in available_sources
            and available_sources[source.document_id] <= question.knowledge_date
        ]
        if not eligible_sources:
            evaluation_summary.deferred += 1
            evaluation_summary.cases.append(
                EvaluationCase(
                    question=question,
                    status="deferred",
                    error="no source document is in the selected corpus at knowledge_date",
                )
            )
            continue
        try:
            answer = answer_baseline(
                database_path,
                client,
                question.question,
                knowledge_date=question.knowledge_date,
                ticker=selected_ticker,
            )
            judge = _judge_answer(
                client,
                question,
                answer,
                prompt_path=judge_path,
                model=judge_model,
            )
            dates_valid, quotes_valid = _citation_checks(database_path, answer)
        except (
            OSError,
            ValueError,
            sqlite3.Error,
            httpx.HTTPError,
            ollama.ResponseError,
        ) as exc:
            evaluation_summary.failed += 1
            evaluation_summary.cases.append(
                EvaluationCase(
                    question=question,
                    status="failed",
                    error=str(exc),
                )
            )
            continue
        citation_count = len(answer.citations)
        evaluation_summary.evaluated += 1
        evaluation_summary.correct += int(judge.correct)
        evaluation_summary.citation_count += citation_count
        evaluation_summary.citation_date_leaks += (
            citation_count if not dates_valid else 0
        )
        evaluation_summary.invalid_quotes += citation_count if not quotes_valid else 0
        evaluation_summary.context_tokens += answer.context_tokens
        evaluation_summary.cases.append(
            EvaluationCase(
                question=question,
                status="evaluated",
                correct=judge.correct,
                answer=answer.answer,
                rationale=judge.rationale,
                citation_count=citation_count,
                date_valid=dates_valid,
                quote_valid=quotes_valid,
                context_tokens=answer.context_tokens,
            )
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            evaluation_summary,
            corpus_database_path=database_path,
            question_path=question_path,
        ),
        encoding="utf-8",
    )
    return evaluation_summary
