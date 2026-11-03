"""Reconcile quote-verified extraction caches into canonical memory tables."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Literal, Sequence

import httpx
import ollama
from pydantic import BaseModel, ConfigDict, Field

from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient
from agent_memory.pipeline.chunks import pack_vector
from agent_memory.pipeline.extract import (
    FORM_PROMPT_FILES,
    CachedExtraction,
    CandidateFact,
    _prompt_versions,
    _segments_for_document,
    _source_hash,
)
from agent_memory.pipeline.extract import normalize_form_family
from agent_memory.pipeline.quotecheck import check_quote
from agent_memory.pipeline.resolve import (
    CachedDecision,
    ResolvedAttribute,
    ResolvedEntity,
    cached_structured_decision,
    resolve_attribute,
    resolve_entity,
)
from agent_memory.query.pit import current_facts_at


APPROVED_EXTRACTION_MODELS = frozenset({"gemma4", "gemma4:latest", "qwen3.5:9b"})
NARRATIVE_CANDIDATE_LIMIT = 5
_PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts"


class NarrativeChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: Literal["same", "distinct", "unresolved"]
    candidate_index: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class ReconciliationPreflight:
    requested_documents: int
    current_caches: int
    missing: tuple[str, ...]
    stale: tuple[str, ...]
    invalid: tuple[tuple[str, str], ...]
    unsupported: tuple[str, ...]
    quote_verified: int
    quote_rejected: int
    t0_candidates: int
    recovery_capped: int


@dataclass(frozen=True)
class ReviewCase:
    document_id: str
    candidate_hash: str
    reason: str
    candidate_json: str
    matches_json: str


@dataclass(frozen=True)
class PlannedFact:
    candidate: CandidateFact
    entity: ResolvedEntity
    attribute: ResolvedAttribute
    key_hash: str | None
    action: Literal["ADD", "CONFIRM", "SUPERSEDE"]
    fact_id: str
    supersedes_fact_id: str | None
    role: Literal["origin", "confirm", "xref"]
    embedding: tuple[float, ...] | None = None


@dataclass
class DocumentPlan:
    document_id: str
    ticker: str
    published_at: str
    prompt_version: str
    entities: dict[str, ResolvedEntity] = field(default_factory=dict)
    aliases: set[tuple[str, str]] = field(default_factory=set)
    attributes: dict[str, ResolvedAttribute] = field(default_factory=dict)
    facts: list[PlannedFact] = field(default_factory=list)
    reviews: list[ReviewCase] = field(default_factory=list)
    ambiguous_key_hashes: set[str] = field(default_factory=set)
    source_candidates: int = 0
    quote_rejected: int = 0
    ignored_t0: int = 0
    duplicate_candidates: int = 0
    decision_cache_hits: int = 0


@dataclass
class ReconciliationSummary:
    model: str
    documents: int
    processed: int = 0
    source_candidates: int = 0
    quote_rejected: int = 0
    ignored_t0: int = 0
    duplicate_candidates: int = 0
    added: int = 0
    confirmed: int = 0
    superseded: int = 0
    unresolved: int = 0
    unsupported: int = 0
    decision_cache_hits: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0
    stale_or_missing: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    document_results: list["DocumentReconciliationResult"] = field(
        default_factory=list
    )

    @property
    def add_ratio(self) -> float | None:
        reconciled = self.added + self.confirmed + self.superseded
        return self.added / reconciled if reconciled else None


@dataclass(frozen=True)
class DocumentReconciliationResult:
    document_id: str
    source_candidates: int
    quote_rejected: int
    ignored_t0: int
    duplicate_candidates: int
    added: int
    confirmed: int
    superseded: int
    unresolved: int


@dataclass(frozen=True)
class _CandidateMatch:
    fact_id: str
    entity_id: str
    text: str
    attribute: str | None
    value: str | None
    unit: str | None
    as_of: str | None
    period: str | None
    distance: float


def _stable_hash(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _published_order(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _selected_documents(
    database_path: Path,
    *,
    ticker: str | None,
    start_date: date | None,
    end_date: date | None,
    limit: int | None,
) -> tuple[list[sqlite3.Row], tuple[str, ...]]:
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("--from date must not be after --to date")
    conditions = ["triage_class IN ('full', 'form')"]
    parameters: list[object] = []
    if ticker is not None:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("ticker must not be blank")
        conditions.append("ticker = ?")
        parameters.append(normalized_ticker)
    if start_date is not None:
        conditions.append("date(published_at) >= date(?)")
        parameters.append(start_date.isoformat())
    if end_date is not None:
        conditions.append("date(published_at) <= date(?)")
        parameters.append(end_date.isoformat())
    query = (
        "SELECT id, ticker, published_at, triage_class, form_type, headline "
        "FROM document WHERE "
        + " AND ".join(conditions)
        + " ORDER BY published_at, id"
    )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    with database_connection(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
        eligible: list[sqlite3.Row] = []
        unsupported: list[str] = []
        for row in rows:
            if row["triage_class"] == "form":
                source_rows = connection.execute(
                    """
                    SELECT text FROM chunk
                    WHERE doc_id = ? AND is_boilerplate = 0
                    ORDER BY ord LIMIT 2
                    """,
                    (row["id"],),
                ).fetchall()
                excerpt = "\n".join(str(source["text"]) for source in source_rows)
                if (
                    normalize_form_family(
                        row["form_type"],
                        str(row["headline"]),
                        excerpt,
                    )
                    is None
                ):
                    unsupported.append(str(row["id"]))
                    continue
            eligible.append(row)
        eligible.sort(
            key=lambda row: (_published_order(str(row["published_at"])), str(row["id"]))
        )
        if limit is not None:
            eligible = eligible[:limit]
    return eligible, tuple(unsupported)


def _load_current_cache(
    database_path: Path,
    document_id: str,
    model: str,
) -> tuple[CachedExtraction, str, dict[str, str]] | None:
    with database_connection(database_path) as connection:
        cached_row = connection.execute(
            """
            SELECT output_json, prompt_version FROM extraction_cache
            WHERE doc_id = ? AND model = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (document_id, model),
        ).fetchone()
        if cached_row is None:
            return None
        document = connection.execute(
            """
            SELECT triage_class, form_type, headline
            FROM document WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if document is None:
            return None
        cached = CachedExtraction.model_validate_json(cached_row["output_json"])
        segments = _segments_for_document(
            connection,
            document_id,
            max_source_characters=cached.segment_source_characters,
        )
        first_source_rows = connection.execute(
            """
            SELECT text FROM chunk
            WHERE doc_id = ? AND is_boilerplate = 0
            ORDER BY ord LIMIT 2
            """,
            (document_id,),
        ).fetchall()
    source_excerpt = "\n".join(str(row["text"]) for row in first_source_rows)
    family = (
        "full"
        if document["triage_class"] == "full"
        else normalize_form_family(
            document["form_type"],
            str(document["headline"]),
            source_excerpt,
        )
    )
    if family is None:
        return cached, str(cached_row["prompt_version"]), {}
    family_prompt = ""
    if family in FORM_PROMPT_FILES:
        family_prompt = (
            (_PROMPT_ROOT / "forms" / FORM_PROMPT_FILES[family]).read_text(
                encoding="utf-8"
            )
        )
    prompt_text = (_PROMPT_ROOT / "extract.md").read_text(encoding="utf-8")
    _, expected_prompt_version = _prompt_versions(
        prompt_text=prompt_text,
        family_prompt=family_prompt,
        family=family,
        segment_source_characters=cached.segment_source_characters,
    )
    if (
        cached.source_doc_id != document_id
        or cached.source_form_type != document["form_type"]
        or cached.form_family != family
        or cached.source_hash != _source_hash(segments)
        or str(cached_row["prompt_version"]) != expected_prompt_version
    ):
        return cached, str(cached_row["prompt_version"]), {}
    return (
        cached,
        str(cached_row["prompt_version"]),
        {segment.segment_id: segment.text for segment in segments},
    )


def preflight_reconciliation(
    database_path: Path,
    *,
    model: str,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> ReconciliationPreflight:
    """Validate cache availability and evidence without writing canonical facts."""
    documents, unsupported = _selected_documents(
        database_path,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    missing: list[str] = []
    stale: list[str] = []
    invalid: list[tuple[str, str]] = []
    quote_verified = 0
    quote_rejected = 0
    t0_candidates = 0
    recovery_capped = 0
    current_caches = 0
    for row in documents:
        document_id = str(row["id"])
        try:
            loaded = _load_current_cache(database_path, document_id, model)
        except (ValueError, sqlite3.Error) as exc:
            invalid.append((document_id, str(exc)))
            continue
        if loaded is None:
            missing.append(document_id)
            continue
        cached, _, source_map = loaded
        if not source_map:
            stale.append(document_id)
            continue
        current_caches += 1
        recovery_capped += cached.truncated_fact_count
        for candidate in cached.facts:
            if candidate.tier == "T0":
                t0_candidates += 1
                continue
            result = check_quote(
                candidate.quote,
                source_map.get(candidate.segment_id, ""),
            )
            if result.passed:
                quote_verified += 1
            else:
                quote_rejected += 1
    return ReconciliationPreflight(
        requested_documents=len(documents),
        current_caches=current_caches,
        missing=tuple(missing),
        stale=tuple(stale),
        invalid=tuple(invalid),
        unsupported=unsupported,
        quote_verified=quote_verified,
        quote_rejected=quote_rejected,
        t0_candidates=t0_candidates,
        recovery_capped=recovery_capped,
    )


def candidate_key_hash(
    ticker: str,
    entity_id: str,
    attribute: str,
    candidate: CandidateFact,
) -> str | None:
    """Build a temporal identity key, excluding the reported value."""
    if candidate.as_of is None and not candidate.period:
        return None
    payload = {
        "ticker": ticker.strip().upper(),
        "entity_id": entity_id,
        "attribute": attribute,
        "as_of": candidate.as_of.isoformat() if candidate.as_of else None,
        "period": " ".join(candidate.period.casefold().split())
        if candidate.period
        else None,
        "qualifier": " ".join(candidate.qualifier.casefold().split())
        if candidate.qualifier
        else None,
    }
    return _stable_hash(payload)


def _normalized_value(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _compare_values(
    candidate: CandidateFact,
    existing: sqlite3.Row,
) -> Literal["same", "different", "incompatible"]:
    existing_unit = _normalized_value(existing["unit"])
    candidate_unit = _normalized_value(candidate.unit)
    if candidate.value_num is not None and existing["value_num"] is not None:
        if candidate_unit != existing_unit:
            return "incompatible"
        return (
            "same"
            if math.isclose(
                candidate.value_num,
                float(existing["value_num"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            else "different"
        )
    return (
        "same"
        if _normalized_value(candidate.value)
        == _normalized_value(existing["value"])
        and candidate_unit == existing_unit
        else "different"
    )


def _fact_identifier(
    document_id: str,
    candidate: CandidateFact,
    entity_id: str,
    attribute: str,
) -> str:
    return "fact_" + _stable_hash(
        {
            "document_id": document_id,
            "entity_id": entity_id,
            "attribute": attribute,
            "candidate": candidate.model_dump(mode="json"),
        }
    )[:32]


def _current_key_facts(
    database_path: Path,
    *,
    ticker: str,
    key_hash: str,
    published_at: str,
) -> list[sqlite3.Row]:
    with database_connection(database_path) as connection:
        return current_facts_at(
            connection,
            published_at,
            ticker=ticker,
            key_hash=key_hash,
            strictly_before=True,
        )


def _narrative_candidates(
    database_path: Path,
    client: LLMClient,
    *,
    candidate: CandidateFact,
    entity_id: str,
    ticker: str,
    published_at: str,
    document_id: str,
) -> list[_CandidateMatch]:
    with database_connection(database_path) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'fact_vec'"
        ).fetchone()
        if table_exists is None:
            return []
        vector_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM fact_vec"
            ).fetchone()["count"]
        )
        current_by_id = {
            str(row["id"]): row
            for row in current_facts_at(
                connection,
                published_at,
                ticker=ticker,
                entity_id=entity_id,
                strictly_before=True,
            )
        }
    if vector_count == 0 or not current_by_id:
        return []

    embedded = client.embed([candidate.text], doc_id=document_id)
    if len(embedded.vectors) != 1:
        raise ValueError("Narrative query embedding must return one vector")
    initialize_database(database_path, vector_dimensions=embedded.dimensions)
    packed_query = pack_vector(embedded.vectors[0])
    candidate_limit = min(vector_count, NARRATIVE_CANDIDATE_LIMIT * 4)
    while True:
        with database_connection(database_path) as connection:
            matches = connection.execute(
                """
                SELECT fact_id, distance
                FROM fact_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (packed_query, candidate_limit),
            ).fetchall()
        candidates = [
            _CandidateMatch(
                fact_id=str(row["fact_id"]),
                entity_id=str(current_by_id[str(row["fact_id"])]["entity_id"]),
                text=str(current_by_id[str(row["fact_id"])]["text"]),
                attribute=current_by_id[str(row["fact_id"])]["attribute"],
                value=current_by_id[str(row["fact_id"])]["value"],
                unit=current_by_id[str(row["fact_id"])]["unit"],
                as_of=current_by_id[str(row["fact_id"])]["as_of"],
                period=current_by_id[str(row["fact_id"])]["period"],
                distance=float(row["distance"]),
            )
            for row in matches
            if str(row["fact_id"]) in current_by_id
        ]
        candidates.sort(key=lambda item: (item.distance, item.fact_id))
        if len(candidates) >= NARRATIVE_CANDIDATE_LIMIT or candidate_limit >= vector_count:
            return candidates[:NARRATIVE_CANDIDATE_LIMIT]
        candidate_limit = min(vector_count, candidate_limit * 2)


def _review_case(
    document_id: str,
    candidate: CandidateFact,
    reason: str,
    matches: Sequence[dict[str, object]],
) -> ReviewCase:
    candidate_json = candidate.model_dump_json()
    candidate_hash = _stable_hash(candidate.model_dump(mode="json"))
    return ReviewCase(
        document_id=document_id,
        candidate_hash=candidate_hash,
        reason=reason,
        candidate_json=candidate_json,
        matches_json=json.dumps(
            matches,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _xref_role(
    database_path: Path,
    document_id: str,
    fact_id: str,
) -> Literal["confirm", "xref"]:
    with database_connection(database_path) as connection:
        targets = {
            str(row["dst_doc"])
            for row in connection.execute(
                "SELECT dst_doc FROM doc_xref WHERE src_doc = ?",
                (document_id,),
            ).fetchall()
        }
        if not targets:
            return "confirm"
        placeholders = ", ".join("?" for _ in targets)
        source = connection.execute(
            f"""
            SELECT 1 FROM fact_source
            WHERE fact_id = ? AND action = 'ADD'
              AND doc_id IN ({placeholders})
            LIMIT 1
            """,
            [fact_id, *sorted(targets)],
        ).fetchone()
    return "xref" if source is not None else "confirm"


def _plan_document(
    database_path: Path,
    client: LLMClient,
    *,
    document_id: str,
    ticker: str,
    published_at: str,
    prompt_version: str,
    cached: CachedExtraction,
    source_map: dict[str, str],
    model: str,
) -> DocumentPlan:
    plan = DocumentPlan(
        document_id=document_id,
        ticker=ticker,
        published_at=published_at,
        prompt_version=prompt_version,
    )
    same_document_entities: dict[
        tuple[str, str, str], ResolvedEntity
    ] = {}
    same_document_attributes: dict[str, ResolvedAttribute] = {}
    staged_by_key: dict[str, PlannedFact] = {}
    staged_by_text: dict[str, PlannedFact] = {}
    conflicted_keys: set[str] = set()

    for candidate in cached.facts:
        plan.source_candidates += 1
        if candidate.tier == "T0":
            plan.ignored_t0 += 1
            continue
        quote_result = check_quote(
            candidate.quote,
            source_map.get(candidate.segment_id, ""),
        )
        if not quote_result.passed:
            plan.quote_rejected += 1
            continue

        entity_result = resolve_entity(
            database_path,
            client,
            ticker=ticker,
            source_entity=candidate.entity,
            evidence=f"{candidate.text}\n{candidate.quote}",
            published_at=published_at,
            document_id=document_id,
            model=model,
            same_document=same_document_entities,
        )
        plan.decision_cache_hits += int(entity_result.cache_hit)
        if entity_result.entity is None:
            plan.reviews.append(
                _review_case(
                    document_id,
                    candidate,
                    entity_result.reason or "Entity identity is unresolved.",
                    [
                        {
                            "id": item.id,
                            "name": item.canonical_name,
                            "type": item.type,
                            "distance": item.distance,
                        }
                        for item in entity_result.candidates
                    ],
                )
            )
            continue

        entity = entity_result.entity
        attribute = resolve_attribute(
            database_path,
            candidate.attribute,
            same_document=same_document_attributes,
        )

        key_hash = candidate_key_hash(
            ticker,
            entity.id,
            attribute.name,
            candidate,
        )
        if key_hash is not None and key_hash in conflicted_keys:
            plan.reviews.append(
                _review_case(
                    document_id,
                    candidate,
                    "Another candidate with this key conflicts in the same publication.",
                    [],
                )
            )
            continue
        if key_hash is not None and key_hash in staged_by_key:
            prior = staged_by_key[key_hash]
            if _compare_planned_values(candidate, prior.candidate) == "same":
                plan.duplicate_candidates += 1
            else:
                plan.facts.remove(prior)
                staged_by_key.pop(key_hash)
                conflicted_keys.add(key_hash)
                plan.ambiguous_key_hashes.add(key_hash)
                conflict_reason = (
                    "Conflicting values for the same key in one publication."
                )
                plan.reviews.append(
                    _review_case(
                        document_id,
                        prior.candidate,
                        conflict_reason,
                        [{"fact_id": prior.fact_id, "text": candidate.text}],
                    )
                )
                plan.reviews.append(
                    _review_case(
                        document_id,
                        candidate,
                        conflict_reason,
                        [{"fact_id": prior.fact_id, "text": prior.candidate.text}],
                    )
                )
            continue
        normalized_text = _normalized_value(candidate.text)
        if key_hash is None and normalized_text in staged_by_text:
            plan.duplicate_candidates += 1
            continue

        action: Literal["ADD", "CONFIRM", "SUPERSEDE"] = "ADD"
        fact_id = ""
        supersedes_fact_id: str | None = None
        role: Literal["origin", "confirm", "xref"] = "origin"
        if key_hash is not None:
            existing_facts = _current_key_facts(
                database_path,
                ticker=ticker,
                key_hash=key_hash,
                published_at=published_at,
            )
            if len(existing_facts) > 1:
                conflicted_keys.add(key_hash)
                plan.ambiguous_key_hashes.add(key_hash)
                plan.reviews.append(
                    _review_case(
                        document_id,
                        candidate,
                        "Multiple current facts share the candidate key.",
                        [
                            {"fact_id": str(item["id"]), "text": str(item["text"])}
                            for item in existing_facts
                        ],
                    )
                )
                continue
            if existing_facts:
                existing = existing_facts[0]
                comparison = _compare_values(candidate, existing)
                if comparison == "incompatible":
                    conflicted_keys.add(key_hash)
                    plan.ambiguous_key_hashes.add(key_hash)
                    plan.reviews.append(
                        _review_case(
                            document_id,
                            candidate,
                            "Matching key has values with incompatible units.",
                            [
                                {
                                    "fact_id": str(existing["id"]),
                                    "value": existing["value"],
                                    "unit": existing["unit"],
                                }
                            ],
                        )
                    )
                    continue
                fact_id = str(existing["id"])
                if comparison == "same":
                    action = "CONFIRM"
                    role = _xref_role(database_path, document_id, fact_id)
                else:
                    action = "SUPERSEDE"
                    supersedes_fact_id = fact_id
                    fact_id = _fact_identifier(
                        document_id,
                        candidate,
                        entity.id,
                        attribute.name,
                    )
        else:
            narrative_candidates = _narrative_candidates(
                database_path,
                client,
                candidate=candidate,
                entity_id=entity.id,
                ticker=ticker,
                published_at=published_at,
                document_id=document_id,
            )
            if narrative_candidates:
                choice = _cached_narrative_choice(
                    database_path,
                    client,
                    candidate=candidate,
                    matches=narrative_candidates,
                    ticker=ticker,
                    published_at=published_at,
                    model=model,
                    document_id=document_id,
                )
                plan.decision_cache_hits += int(choice.cache_hit)
                decision = choice.value
                if decision.choice != "same" and decision.candidate_index is not None:
                    raise ValueError(
                        "Narrative judge returned an index for a non-same choice"
                    )
                if decision.choice == "same":
                    if (
                        decision.candidate_index is None
                        or decision.candidate_index >= len(narrative_candidates)
                    ):
                        raise ValueError(
                            "Narrative judge selected an unavailable candidate index"
                        )
                    fact_id = narrative_candidates[
                        decision.candidate_index
                    ].fact_id
                    action = "CONFIRM"
                    role = _xref_role(database_path, document_id, fact_id)
                elif decision.choice == "unresolved":
                    plan.reviews.append(
                        _review_case(
                            document_id,
                            candidate,
                            decision.reason,
                            [
                                {
                                    "fact_id": item.fact_id,
                                    "entity_id": item.entity_id,
                                    "text": item.text,
                                    "distance": item.distance,
                                }
                                for item in narrative_candidates
                            ],
                        )
                    )
                    continue
            if action == "ADD":
                fact_id = _fact_identifier(
                    document_id,
                    candidate,
                    entity.id,
                    attribute.name,
                )

        if action in ("ADD", "SUPERSEDE") and not fact_id:
            fact_id = _fact_identifier(
                document_id,
                candidate,
                entity.id,
                attribute.name,
            )
        planned = PlannedFact(
            candidate=candidate,
            entity=entity,
            attribute=attribute,
            key_hash=key_hash,
            action=action,
            fact_id=fact_id,
            supersedes_fact_id=supersedes_fact_id,
            role=role,
        )
        plan.facts.append(planned)
        if entity.created:
            plan.entities[entity.id] = entity
        if entity.alias_to_add:
            plan.aliases.add((entity.id, entity.alias_to_add))
        if attribute.created:
            plan.attributes[attribute.name] = attribute
        if key_hash is not None:
            staged_by_key[key_hash] = planned
        else:
            staged_by_text[normalized_text] = planned

    _prune_document_plan(plan)
    return plan


def _prune_document_plan(plan: DocumentPlan) -> None:
    used_entities = {item.entity.id for item in plan.facts}
    used_attributes = {item.attribute.name for item in plan.facts}
    plan.entities = {
        entity_id: entity
        for entity_id, entity in plan.entities.items()
        if entity_id in used_entities
    }
    plan.aliases = {
        (entity_id, alias)
        for entity_id, alias in plan.aliases
        if entity_id in used_entities
    }
    plan.attributes = {
        name: attribute
        for name, attribute in plan.attributes.items()
        if name in used_attributes
    }


def _mark_same_time_key_conflicts(plans: Sequence[DocumentPlan]) -> None:
    facts_by_key: dict[str, list[tuple[DocumentPlan, PlannedFact]]] = {}
    narratives_by_text: dict[
        tuple[str, str, str], list[tuple[DocumentPlan, PlannedFact]]
    ] = {}
    ambiguous_keys = set().union(*(plan.ambiguous_key_hashes for plan in plans))
    for plan in plans:
        for item in plan.facts:
            if item.key_hash is not None:
                facts_by_key.setdefault(item.key_hash, []).append((plan, item))
            else:
                identity = (
                    item.entity.id,
                    item.attribute.name,
                    _normalized_value(item.candidate.text),
                )
                narratives_by_text.setdefault(identity, []).append((plan, item))

    for key_hash, entries in facts_by_key.items():
        if len(entries) < 2 and key_hash not in ambiguous_keys:
            continue
        matches = [
            {
                "document_id": plan.document_id,
                "fact_id": item.fact_id,
                "action": item.action,
                "value": item.candidate.value,
                "text": item.candidate.text,
            }
            for plan, item in entries
        ]
        for plan, item in entries:
            plan.facts.remove(item)
            plan.ambiguous_key_hashes.add(key_hash)
            plan.reviews.append(
                _review_case(
                    plan.document_id,
                    item.candidate,
                    "Same-publication-time candidates share a key and cannot be ordered.",
                    matches,
                )
            )
    for entries in narratives_by_text.values():
        if len(entries) < 2:
            continue
        matches = [
            {
                "document_id": plan.document_id,
                "fact_id": item.fact_id,
                "action": item.action,
                "text": item.candidate.text,
            }
            for plan, item in entries
        ]
        for plan, item in entries:
            plan.facts.remove(item)
            plan.reviews.append(
                _review_case(
                    plan.document_id,
                    item.candidate,
                    "Same-publication-time duplicate narratives cannot be ordered.",
                    matches,
                )
            )
    for plan in plans:
        _prune_document_plan(plan)


def _compare_planned_values(
    candidate: CandidateFact,
    existing: CandidateFact,
) -> Literal["same", "different", "incompatible"]:
    if candidate.value_num is not None and existing.value_num is not None:
        if _normalized_value(candidate.unit) != _normalized_value(existing.unit):
            return "incompatible"
        return (
            "same"
            if math.isclose(
                candidate.value_num,
                existing.value_num,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            else "different"
        )
    return (
        "same"
        if _normalized_value(candidate.value) == _normalized_value(existing.value)
        and _normalized_value(candidate.unit) == _normalized_value(existing.unit)
        else "different"
    )


def _cached_narrative_choice(
    database_path: Path,
    client: LLMClient,
    *,
    candidate: CandidateFact,
    matches: Sequence[_CandidateMatch],
    ticker: str,
    published_at: str,
    model: str,
    document_id: str,
) -> CachedDecision[NarrativeChoice]:
    input_data: dict[str, object] = {
        "ticker": ticker,
        "published_at": published_at,
        "candidate": candidate.model_dump(mode="json"),
        "matches": [
            {
                "fact_id": item.fact_id,
                "entity_id": item.entity_id,
                "text": item.text,
                "attribute": item.attribute,
                "value": item.value,
                "unit": item.unit,
                "as_of": item.as_of,
                "period": item.period,
                "distance": item.distance,
            }
            for item in matches
        ],
    }
    return cached_structured_decision(
        database_path,
        client,
        step="narrative_dedup",
        prompt=(_PROMPT_ROOT / "judge_same.md").read_text(encoding="utf-8"),
        input_data=input_data,
        response_model=NarrativeChoice,
        model=model,
        document_id=document_id,
    )


def _attach_new_fact_embeddings(
    database_path: Path,
    client: LLMClient,
    plan: DocumentPlan,
) -> None:
    new_facts = [
        item
        for item in plan.facts
        if item.action in ("ADD", "SUPERSEDE")
    ]
    for start in range(0, len(new_facts), 32):
        batch = new_facts[start : start + 32]
        result = client.embed(
            [item.candidate.text for item in batch],
            doc_id=plan.document_id,
        )
        if len(result.vectors) != len(batch):
            raise ValueError("Fact embedding count does not match the fact batch")
        initialize_database(database_path, vector_dimensions=result.dimensions)
        embeddings = {
            item.fact_id: tuple(vector)
            for item, vector in zip(batch, result.vectors, strict=True)
        }
        plan.facts = [
            replace(item, embedding=embeddings[item.fact_id])
            if item.fact_id in embeddings
            else item
            for item in plan.facts
        ]


def _insert_review(connection: sqlite3.Connection, case: ReviewCase) -> None:
    review_id = "rev_" + _stable_hash(
        {"document_id": case.document_id, "candidate_hash": case.candidate_hash}
    )[:32]
    connection.execute(
        """
        INSERT INTO reconciliation_review (
            id, doc_id, candidate_hash, reason, candidate_json,
            matches_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        ON CONFLICT(doc_id, candidate_hash) DO UPDATE SET
            reason = excluded.reason,
            candidate_json = excluded.candidate_json,
            matches_json = excluded.matches_json
        WHERE reconciliation_review.status = 'pending'
        """,
        (
            review_id,
            case.document_id,
            case.candidate_hash,
            case.reason,
            case.candidate_json,
            case.matches_json,
        ),
    )


def _persist_plan(database_path: Path, plan: DocumentPlan) -> None:
    with database_connection(database_path) as connection:
        for entity in plan.entities.values():
            connection.execute(
                """
                INSERT INTO entity (
                    id, ticker, type, canonical_name, first_seen_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET first_seen_at =
                    CASE WHEN julianday(excluded.first_seen_at)
                              < julianday(entity.first_seen_at)
                         THEN excluded.first_seen_at
                         ELSE entity.first_seen_at END
                """,
                (
                    entity.id,
                    entity.ticker,
                    entity.type,
                    entity.canonical_name,
                    entity.first_seen_at,
                ),
            )
            if entity.embedding is not None:
                connection.execute(
                    "DELETE FROM entity_vec WHERE entity_id = ?",
                    (entity.id,),
                )
                connection.execute(
                    "INSERT INTO entity_vec (embedding, entity_id) VALUES (?, ?)",
                    (pack_vector(entity.embedding), entity.id),
                )
        for entity_id, alias in sorted(plan.aliases):
            entity = plan.entities.get(entity_id)
            if entity is None:
                resolved = next(
                    (
                        item.entity
                        for item in plan.facts
                        if item.entity.id == entity_id
                    ),
                    None,
                )
                if resolved is None:
                    raise ValueError(f"Alias references an unplanned entity: {entity_id}")
                ticker = resolved.ticker
            else:
                ticker = entity.ticker
            connection.execute(
                """
                INSERT OR IGNORE INTO entity_alias (
                    entity_id, alias, first_seen_at, source_doc
                ) VALUES (?, ?, ?, ?)
                """,
                (entity_id, alias, plan.published_at, plan.document_id),
            )
            if not ticker:
                raise ValueError("Alias ticker must not be blank")
        for attribute in plan.attributes.values():
            connection.execute(
                """
                INSERT OR IGNORE INTO attribute (
                    name, description, first_seen_at, status
                ) VALUES (?, ?, ?, 'active')
                """,
                (attribute.name, attribute.description, plan.published_at),
            )
        for planned in plan.facts:
            candidate = planned.candidate
            if planned.action in ("ADD", "SUPERSEDE"):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fact (
                        id, ticker, entity_id, attribute, value, value_num, unit,
                        qualifier, as_of, period, fact_type, tier, salience,
                        text, key_hash, first_seen_at, extractor_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        planned.fact_id,
                        plan.ticker,
                        planned.entity.id,
                        planned.attribute.name,
                        candidate.value,
                        candidate.value_num,
                        candidate.unit,
                        candidate.qualifier,
                        candidate.as_of.isoformat() if candidate.as_of else None,
                        candidate.period,
                        candidate.fact_type,
                        candidate.tier,
                        candidate.salience,
                        candidate.text,
                        planned.key_hash,
                        plan.published_at,
                        plan.prompt_version,
                    ),
                )
                if planned.embedding is None:
                    raise ValueError(
                        f"New fact has no embedding: {planned.fact_id}"
                    )
                connection.execute(
                    "DELETE FROM fact_vec WHERE fact_id = ?",
                    (planned.fact_id,),
                )
                connection.execute(
                    "INSERT INTO fact_vec (embedding, fact_id) VALUES (?, ?)",
                    (pack_vector(planned.embedding), planned.fact_id),
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO fact_source (
                    fact_id, doc_id, published_at, quote, action,
                    supersedes_fact_id, role
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    planned.fact_id,
                    plan.document_id,
                    plan.published_at,
                    candidate.quote,
                    planned.action,
                    planned.supersedes_fact_id,
                    planned.role,
                ),
            )
        for case in plan.reviews:
            _insert_review(connection, case)


def _clear_ticker_memory(database_path: Path, ticker: str) -> None:
    with database_connection(database_path) as connection:
        fact_ids = [
            str(row["id"])
            for row in connection.execute(
                "SELECT id FROM fact WHERE ticker = ?", (ticker,)
            ).fetchall()
        ]
        entity_ids = [
            str(row["id"])
            for row in connection.execute(
                "SELECT id FROM entity WHERE ticker = ?", (ticker,)
            ).fetchall()
        ]
        if fact_ids:
            placeholders = ", ".join("?" for _ in fact_ids)
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'fact_vec'"
            ).fetchone():
                connection.execute(
                    f"DELETE FROM fact_vec WHERE fact_id IN ({placeholders})",
                    fact_ids,
                )
            connection.execute(
                f"DELETE FROM fact_source WHERE fact_id IN ({placeholders})",
                fact_ids,
            )
        connection.execute("DELETE FROM fact WHERE ticker = ?", (ticker,))
        if entity_ids:
            placeholders = ", ".join("?" for _ in entity_ids)
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'entity_vec'"
            ).fetchone():
                connection.execute(
                    f"DELETE FROM entity_vec WHERE entity_id IN ({placeholders})",
                    entity_ids,
                )
        connection.execute("DELETE FROM entity WHERE ticker = ?", (ticker,))
        connection.execute(
            """
            DELETE FROM reconciliation_review
            WHERE doc_id IN (SELECT id FROM document WHERE ticker = ?)
              AND status = 'pending'
            """,
            (ticker,),
        )


def _llm_call_snapshot(database_path: Path) -> int:
    with database_connection(database_path) as connection:
        row = connection.execute(
            "SELECT COALESCE(MAX(id), 0) AS id FROM llm_call"
        ).fetchone()
        return int(row["id"])


def _llm_call_totals(database_path: Path, after_id: int) -> tuple[int, int, int]:
    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens
            FROM llm_call WHERE id > ?
            """,
            (after_id,),
        ).fetchone()
    return (
        int(row["calls"]),
        int(row["prompt_tokens"]),
        int(row["output_tokens"]),
    )


def run_reconciliation(
    database_path: Path,
    client: LLMClient,
    *,
    model: str,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    replay: bool = False,
    confirm_wave2_gate: bool = False,
) -> ReconciliationSummary:
    """Reconcile selected, current extraction caches without re-running extraction."""
    if not confirm_wave2_gate:
        raise ValueError(
            "Wave 2 Gate 2 must be accepted before promoting candidates to memory"
        )
    if model not in APPROVED_EXTRACTION_MODELS:
        raise ValueError(
            f"Model is not approved for extraction reconciliation: {model}"
        )
    if replay and (
        ticker is None
        or start_date is not None
        or end_date is not None
        or limit is not None
    ):
        raise ValueError(
            "A replay requires --ticker and cannot be combined with dates or --limit"
        )
    ticker = ticker.strip().upper() if ticker is not None else None
    if ticker == "":
        raise ValueError("ticker must not be blank")

    started = time.perf_counter()
    initialize_database(database_path)
    preflight = preflight_reconciliation(
        database_path,
        model=model,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if replay and (preflight.missing or preflight.stale or preflight.invalid):
        raise ValueError(
            "Replay preflight failed; missing, stale, or invalid caches must be "
            "resolved before clearing canonical memory"
        )
    documents, unsupported = _selected_documents(
        database_path,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if replay:
        if ticker is None:
            raise AssertionError("Replay validation must provide a ticker")
        _clear_ticker_memory(database_path, ticker)
    call_snapshot = _llm_call_snapshot(database_path)
    summary = ReconciliationSummary(
        model=model,
        documents=len(documents),
        unsupported=len(unsupported),
    )
    for _, grouped_rows in groupby(
        documents,
        key=lambda row: _published_order(str(row["published_at"])),
    ):
        plans: list[DocumentPlan] = []
        for row in grouped_rows:
            document_id = str(row["id"])
            try:
                loaded = _load_current_cache(database_path, document_id, model)
                if loaded is None:
                    summary.stale_or_missing.append(document_id)
                    continue
                cached, prompt_version, source_map = loaded
                if not source_map:
                    summary.stale_or_missing.append(document_id)
                    continue
                plans.append(
                    _plan_document(
                        database_path,
                        client,
                        document_id=document_id,
                        ticker=str(row["ticker"]),
                        published_at=str(row["published_at"]),
                        prompt_version=prompt_version,
                        cached=cached,
                        source_map=source_map,
                        model=model,
                    )
                )
            except (
                OSError,
                ValueError,
                sqlite3.Error,
                httpx.HTTPError,
                ollama.ResponseError,
            ) as exc:
                summary.errors.append((document_id, str(exc)))

        _mark_same_time_key_conflicts(plans)
        for plan in plans:
            try:
                _attach_new_fact_embeddings(database_path, client, plan)
                _persist_plan(database_path, plan)
            except (
                OSError,
                ValueError,
                sqlite3.Error,
                httpx.HTTPError,
                ollama.ResponseError,
            ) as exc:
                summary.errors.append((plan.document_id, str(exc)))
                continue

            summary.processed += 1
            summary.source_candidates += plan.source_candidates
            summary.quote_rejected += plan.quote_rejected
            summary.ignored_t0 += plan.ignored_t0
            summary.duplicate_candidates += plan.duplicate_candidates
            summary.unresolved += len(plan.reviews)
            summary.decision_cache_hits += plan.decision_cache_hits
            document_result = DocumentReconciliationResult(
                document_id=plan.document_id,
                source_candidates=plan.source_candidates,
                quote_rejected=plan.quote_rejected,
                ignored_t0=plan.ignored_t0,
                duplicate_candidates=plan.duplicate_candidates,
                added=sum(item.action == "ADD" for item in plan.facts),
                confirmed=sum(item.action == "CONFIRM" for item in plan.facts),
                superseded=sum(item.action == "SUPERSEDE" for item in plan.facts),
                unresolved=len(plan.reviews),
            )
            summary.document_results.append(document_result)
            for planned in plan.facts:
                if planned.action == "ADD":
                    summary.added += 1
                elif planned.action == "CONFIRM":
                    summary.confirmed += 1
                else:
                    summary.superseded += 1

    summary.llm_calls, summary.prompt_tokens, summary.output_tokens = (
        _llm_call_totals(database_path, call_snapshot)
    )
    summary.elapsed_seconds = time.perf_counter() - started
    return summary
