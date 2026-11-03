"""Memory-backed question planning, retrieval orchestration, and answering."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import ollama
from pydantic import BaseModel, ConfigDict, Field

from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient
from agent_memory.pipeline.reconcile import APPROVED_EXTRACTION_MODELS
from agent_memory.query.retrieve import (
    ChunkEvidence,
    FactEvidence,
    MemorySource,
    available_query_choices,
    fact_matches_subject,
    retrieve_chunks,
    retrieve_facts,
    retrieve_state,
    retrieve_summary_facts,
    retrieve_timeline,
)


_PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts"
MAX_MEMORY_CONTEXT_CHARACTERS = 24_000
_ABSTENTION_PATTERN = re.compile(
    r"\b(?:cannot answer|can't answer|unable to answer|"
    r"not enough (?:information|evidence)|insufficient (?:information|evidence)|"
    r"no verified (?:fact|evidence|source)|not found|i don't know)\b",
    re.IGNORECASE,
)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["state", "timeline", "event", "search", "summary"]
    entity_id: str | None = None
    attribute: str | None = None
    subject_date: date | None = None
    subject_period: str | None = None
    search_query: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    include_t3: bool = False


class MemoryAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class MemoryQueryResult:
    query_id: str
    question: str
    ticker: str
    knowledge_date: date
    intent: str
    answer: str
    citations: tuple[MemorySource | ChunkEvidence, ...]
    retrieved_facts: tuple[FactEvidence, ...]
    used_chunk_fallback: bool
    query_plan_tokens: int = 0
    answer_prompt_tokens: int = 0
    answer_output_tokens: int = 0
    answer_attempts: int = 0
    elapsed_ms: float = 0


def _validate_plan(
    plan: QueryPlan,
    *,
    entity_ids: set[str],
    attributes: set[str],
) -> None:
    if plan.entity_id is not None and plan.entity_id not in entity_ids:
        raise ValueError("Query planner selected an entity outside the allowed set")
    if plan.attribute is not None and plan.attribute not in attributes:
        raise ValueError("Query planner selected an attribute outside the allowed set")
    if plan.intent in ("state", "timeline") and (
        plan.entity_id is None or plan.attribute is None
    ):
        raise ValueError(
            f"The {plan.intent} query requires an allowed entity and attribute"
        )
    if plan.intent in ("event", "search") and not (
        plan.search_query and plan.search_query.strip()
    ):
        raise ValueError(f"The {plan.intent} query requires search terms")
    if plan.intent == "summary":
        if plan.start_date is None or plan.end_date is None:
            raise ValueError("A summary query requires a bounded date range")
        if plan.start_date > plan.end_date:
            raise ValueError("Summary start date must not be after end date")


def _format_fact_context(
    facts: tuple[FactEvidence, ...],
    *,
    context_limit: int,
) -> tuple[str, dict[str, MemorySource]]:
    blocks: list[str] = []
    citations: dict[str, MemorySource] = {}
    length = 0
    for fact in facts:
        source_lines: list[str] = []
        added_sources: list[MemorySource] = []
        for source in fact.sources:
            source_lines.append(
                f"[citation_id={source.citation_id}; document_id={source.document_id}; "
                f"published_at={source.published_at}]\n"
                f"Source quote: {source.quote}"
            )
            added_sources.append(source)
        block = (
            f"[fact_id={fact.fact_id}; ticker={fact.ticker}; "
            f"entity={fact.entity_name or fact.entity_id}; "
            f"attribute={fact.attribute}; tier={fact.tier}; "
            f"as_of={fact.as_of or 'unknown'}; period={fact.period or 'unknown'}]\n"
            f"Fact: {fact.text}\n"
            + "\n".join(source_lines)
        )
        addition = len(block) + (2 if blocks else 0)
        if length + addition > context_limit:
            continue
        blocks.append(block)
        length += addition
        citations.update((source.citation_id, source) for source in added_sources)
    return "\n\n".join(blocks), citations


def _format_chunk_context(
    chunks: tuple[ChunkEvidence, ...],
    *,
    context_limit: int,
) -> tuple[str, dict[str, ChunkEvidence]]:
    blocks: list[str] = []
    citations: dict[str, ChunkEvidence] = {}
    length = 0
    for chunk in chunks:
        block = (
            f"[citation_id={chunk.citation_id}; document_id={chunk.document_id}; "
            f"published_at={chunk.published_at}]\n{chunk.text}"
        )
        addition = len(block) + (2 if blocks else 0)
        if length + addition > context_limit:
            continue
        blocks.append(block)
        length += addition
        citations[chunk.citation_id] = chunk
    return "\n\n".join(blocks), citations


def _is_abstention(answer: str) -> bool:
    return bool(_ABSTENTION_PATTERN.search(answer))


def _record_tool_call(
    database_path: Path,
    *,
    query_id: str,
    tool_name: str,
    ticker: str,
    knowledge_date: date,
    arguments: dict[str, object],
    result_count: int,
    elapsed_ms: float,
    status: Literal["ok", "error"] = "ok",
    error: str | None = None,
) -> None:
    connection_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_tool_call (
                query_id, tool_name, ticker, knowledge_date, args_json,
                result_count, elapsed_ms, status, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                      strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                query_id,
                tool_name,
                ticker,
                knowledge_date.isoformat(),
                connection_arguments,
                result_count,
                elapsed_ms,
                status,
                error,
            ),
        )


def _dispatch_retrieval(
    database_path: Path,
    client: LLMClient,
    plan: QueryPlan,
    *,
    question: str,
    ticker: str,
    knowledge_date: date,
    include_t3: bool,
    chunk_fallback: bool,
) -> tuple[tuple[FactEvidence, ...], tuple[ChunkEvidence, ...], bool, bool, float]:
    if plan.intent == "state":
        if plan.entity_id is None or plan.attribute is None:
            raise AssertionError("Validated state plan is missing its selector")
        facts = retrieve_state(
            database_path,
            ticker=ticker,
            entity_id=plan.entity_id,
            attribute=plan.attribute,
            knowledge_date=knowledge_date,
            subject_date=plan.subject_date,
            subject_period=plan.subject_period,
        )
    elif plan.intent == "timeline":
        if plan.entity_id is None or plan.attribute is None:
            raise AssertionError("Validated timeline plan is missing its selector")
        facts = retrieve_timeline(
            database_path,
            ticker=ticker,
            entity_id=plan.entity_id,
            attribute=plan.attribute,
            knowledge_date=knowledge_date,
        )
        if plan.subject_date is not None or plan.subject_period is not None:
            facts = tuple(
                fact
                for fact in facts
                if fact_matches_subject(
                    fact,
                    subject_date=plan.subject_date,
                    subject_period=plan.subject_period,
                )
            )
    elif plan.intent == "summary":
        facts = retrieve_summary_facts(
            database_path,
            ticker=ticker,
            knowledge_date=knowledge_date,
            start_date=plan.start_date,
            end_date=plan.end_date,
            include_t3=include_t3 or plan.include_t3,
        )
    else:
        facts = retrieve_facts(
            database_path,
            client,
            plan.search_query or question,
            ticker=ticker,
            knowledge_date=knowledge_date,
            current_only=False,
            include_t3=include_t3 or plan.include_t3,
        )
    if not include_t3 and not plan.include_t3:
        facts = tuple(fact for fact in facts if fact.tier in {"T1", "T2"})
    chunks: tuple[ChunkEvidence, ...] = ()
    fallback_used = False
    fallback_attempted = False
    fallback_elapsed_ms = 0.0
    if not facts and chunk_fallback:
        fallback_attempted = True
        fallback_started = time.perf_counter()
        chunks = retrieve_chunks(
            database_path,
            client,
            plan.search_query or question,
            ticker=ticker,
            knowledge_date=knowledge_date,
        )
        fallback_elapsed_ms = (time.perf_counter() - fallback_started) * 1000
        fallback_used = bool(chunks)
    return facts, chunks, fallback_used, fallback_attempted, fallback_elapsed_ms


def answer_memory(
    database_path: Path,
    client: LLMClient,
    question: str,
    *,
    knowledge_date: date,
    ticker: str,
    model: str | None = None,
    chunk_fallback: bool = False,
    include_t3: bool = False,
) -> MemoryQueryResult:
    """Answer one memory question with PIT facts and validated source IDs."""
    if not question.strip():
        raise ValueError("question must not be blank")
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("ticker must not be blank")
    selected_model = model or client.model_name
    if selected_model not in APPROVED_EXTRACTION_MODELS:
        raise ValueError(
            f"Model is not approved for memory answering: {selected_model}"
        )
    started = time.perf_counter()
    query_id = uuid4().hex
    initialize_database(database_path)
    choices = available_query_choices(
        database_path,
        ticker=normalized_ticker,
        knowledge_date=knowledge_date,
    )
    entity_payload = [
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "aliases": list(entity.aliases),
        }
        for entity in choices.entities
    ]
    attribute_payload = [
        {"name": attribute.name, "description": attribute.description}
        for attribute in choices.attributes
    ]
    query_plan_prompt = (_PROMPT_ROOT / "query_plan.md").read_text(
        encoding="utf-8"
    )
    query_plan_input = {
        "question": question,
        "ticker": normalized_ticker,
        "knowledge_date": knowledge_date.isoformat(),
        "allowed_entities": entity_payload,
        "allowed_attributes": attribute_payload,
    }
    query_plan_result = client.chat_json(
        json.dumps(query_plan_input, ensure_ascii=False, sort_keys=True),
        QueryPlan,
        system_prompt=query_plan_prompt,
        model=selected_model,
        step="query_plan",
    )
    entity_ids = {entity.id for entity in choices.entities}
    attributes = {attribute.name for attribute in choices.attributes}
    _validate_plan(
        query_plan_result.value,
        entity_ids=entity_ids,
        attributes=attributes,
    )
    plan = query_plan_result.value
    tool_name = {
        "state": "get_state",
        "timeline": "get_timeline",
        "event": "search_facts",
        "search": "search_facts",
        "summary": "get_summary",
    }[plan.intent]
    tool_arguments: dict[str, object] = {
        "intent": plan.intent,
        "entity_id": plan.entity_id,
        "attribute": plan.attribute,
        "subject_date": plan.subject_date.isoformat()
        if plan.subject_date
        else None,
        "subject_period": plan.subject_period,
        "start_date": plan.start_date.isoformat() if plan.start_date else None,
        "end_date": plan.end_date.isoformat() if plan.end_date else None,
        "include_t3": include_t3 or plan.include_t3,
        "has_search_query": bool(plan.search_query),
    }
    tool_started = time.perf_counter()
    try:
        (
            facts,
            chunks,
            fallback_used,
            fallback_attempted,
            fallback_elapsed_ms,
        ) = _dispatch_retrieval(
            database_path,
            client,
            plan,
            question=question,
            ticker=normalized_ticker,
            knowledge_date=knowledge_date,
            include_t3=include_t3,
            chunk_fallback=chunk_fallback,
        )
    except (
        OSError,
        ValueError,
        sqlite3.Error,
        httpx.HTTPError,
        ollama.ResponseError,
    ) as exc:
        _record_tool_call(
            database_path,
            query_id=query_id,
            tool_name=tool_name,
            ticker=normalized_ticker,
            knowledge_date=knowledge_date,
            arguments=tool_arguments,
            result_count=0,
            elapsed_ms=(time.perf_counter() - tool_started) * 1000,
            status="error",
            error=type(exc).__name__,
        )
        raise
    _record_tool_call(
        database_path,
        query_id=query_id,
        tool_name=tool_name,
        ticker=normalized_ticker,
        knowledge_date=knowledge_date,
        arguments=tool_arguments,
        result_count=len(facts),
        elapsed_ms=max(
            0.0,
            (time.perf_counter() - tool_started) * 1000 - fallback_elapsed_ms,
        ),
    )
    if fallback_attempted:
        _record_tool_call(
            database_path,
            query_id=query_id,
            tool_name="search_chunks",
            ticker=normalized_ticker,
            knowledge_date=knowledge_date,
            arguments={"reason": "no_matching_facts"},
            result_count=len(chunks),
            elapsed_ms=fallback_elapsed_ms,
        )
    fact_context, fact_citations = _format_fact_context(
        facts,
        context_limit=MAX_MEMORY_CONTEXT_CHARACTERS,
    )
    chunk_context, chunk_citations = _format_chunk_context(
        chunks,
        context_limit=MAX_MEMORY_CONTEXT_CHARACTERS - len(fact_context),
    )
    context = "\n\n".join(
        block for block in (fact_context, chunk_context) if block
    )
    available_citations: dict[str, MemorySource | ChunkEvidence] = {
        **fact_citations,
        **chunk_citations,
    }
    if not available_citations:
        return MemoryQueryResult(
            query_id=query_id,
            question=question,
            ticker=normalized_ticker,
            knowledge_date=knowledge_date,
            intent=query_plan_result.value.intent,
            answer=(
                "I could not find a verified memory fact or source passage "
                f"for this question as of {knowledge_date.isoformat()}."
            ),
            citations=(),
            retrieved_facts=facts,
            used_chunk_fallback=fallback_used,
            query_plan_tokens=query_plan_result.prompt_tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    answer_prompt_path = (
        _PROMPT_ROOT / "summary.md"
        if query_plan_result.value.intent == "summary"
        else _PROMPT_ROOT / "memory_answer.md"
    )
    answer_system_prompt = answer_prompt_path.read_text(encoding="utf-8")
    base_prompt = (
        f"Ticker: {normalized_ticker}\n"
        f"Knowledge date (inclusive): {knowledge_date.isoformat()}\n"
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n{context}"
    )
    answer_prompt_tokens = 0
    answer_output_tokens = 0
    answer_attempts = 0
    previous_raw: str | None = None
    validation_error: str | None = None
    response: MemoryAnswerResponse | None = None
    for semantic_attempt in (1, 2):
        prompt = base_prompt
        if semantic_attempt == 2:
            if previous_raw is None or validation_error is None:
                raise AssertionError("A citation retry requires a prior failure")
            prompt += (
                "\n\nYour previous response failed citation validation: "
                f"{validation_error}\nPrevious JSON: {previous_raw}\n"
                "Return corrected JSON. Cite only supplied citation IDs."
            )
        result = client.chat_json(
            prompt,
            MemoryAnswerResponse,
            system_prompt=answer_system_prompt,
            model=selected_model,
            step="memory_answer",
        )
        answer_prompt_tokens += result.prompt_tokens
        answer_output_tokens += result.output_tokens
        answer_attempts += result.attempts
        previous_raw = result.raw_content
        response = result.value
        unknown_ids = [
            citation_id
            for citation_id in response.citation_ids
            if citation_id not in available_citations
        ]
        if unknown_ids:
            validation_error = (
                "unknown citation ID(s): " + ", ".join(sorted(set(unknown_ids)))
            )
            continue
        if not response.citation_ids and not _is_abstention(response.answer):
            validation_error = "a non-abstaining answer has no citations"
            continue
        validation_error = None
        break
    if response is None:
        raise AssertionError("The memory answer model did not return")
    if validation_error is not None:
        raise ValueError(
            f"Memory answer failed citation validation after one correction: "
            f"{validation_error}"
        )
    citations = tuple(
        available_citations[citation_id]
        for citation_id in dict.fromkeys(response.citation_ids)
    )
    return MemoryQueryResult(
        query_id=query_id,
        question=question,
        ticker=normalized_ticker,
        knowledge_date=knowledge_date,
        intent=query_plan_result.value.intent,
        answer=response.answer,
        citations=citations,
        retrieved_facts=facts,
        used_chunk_fallback=fallback_used,
        query_plan_tokens=query_plan_result.prompt_tokens,
        answer_prompt_tokens=answer_prompt_tokens,
        answer_output_tokens=answer_output_tokens,
        answer_attempts=answer_attempts,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
