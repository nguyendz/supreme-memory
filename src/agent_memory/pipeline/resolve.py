"""Point-in-time entity and attribute resolution for cached candidate facts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient
from agent_memory.pipeline.chunks import pack_vector
from agent_memory.pipeline.extract import CandidateEntity


_ModelT = TypeVar("_ModelT", bound=BaseModel)
_PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts"


class EntityChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: Literal["existing", "new", "unresolved"]
    candidate_index: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class EntityCandidate:
    id: str
    ticker: str
    type: str
    canonical_name: str
    first_seen_at: str
    distance: float | None = None


@dataclass(frozen=True)
class ResolvedEntity:
    id: str
    ticker: str
    type: str
    canonical_name: str
    first_seen_at: str
    created: bool
    alias_to_add: str | None = None
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class EntityResolution:
    entity: ResolvedEntity | None
    candidates: tuple[EntityCandidate, ...] = ()
    reason: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True)
class ResolvedAttribute:
    name: str
    description: str
    created: bool


def normalize_entity_name(value: str) -> str:
    """Normalize only Unicode, case, and whitespace; preserve meaningful words."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


@dataclass(frozen=True)
class CachedDecision(Generic[_ModelT]):
    value: _ModelT
    cache_hit: bool


def normalize_attribute_name(value: str) -> str:
    """Map equivalent attribute punctuation and spacing to a snake-case key."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "_".join(re.findall(r"[a-z0-9]+", normalized))


def stable_entity_id(ticker: str, entity_type: str, canonical_name: str) -> str:
    identity = "\0".join(
        (
            ticker.strip().upper(),
            normalize_entity_name(entity_type),
            normalize_entity_name(canonical_name),
        )
    )
    return "ent_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _stable_json_hash(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cached_structured_decision(
    database_path: Path,
    client: LLMClient,
    *,
    step: Literal["entity_resolution", "narrative_dedup"],
    prompt: str,
    input_data: dict[str, object],
    response_model: type[_ModelT],
    model: str,
    document_id: str,
) -> CachedDecision[_ModelT]:
    """Replay or cache one schema-validated resolver/judge response."""
    input_hash = _stable_json_hash(input_data)
    prompt_version = _stable_json_hash(
        {
            "prompt": prompt,
            "schema": response_model.model_json_schema(),
        }
    )
    with database_connection(database_path) as connection:
        cached = connection.execute(
            """
            SELECT output_json FROM reconciliation_cache
            WHERE step = ? AND input_hash = ? AND model = ? AND prompt_version = ?
            """,
            (step, input_hash, model, prompt_version),
        ).fetchone()
    if cached is not None:
        return CachedDecision(
            response_model.model_validate_json(cached["output_json"]),
            cache_hit=True,
        )

    result = client.chat_json(
        json.dumps(input_data, ensure_ascii=False, sort_keys=True),
        response_model,
        system_prompt=prompt,
        doc_id=document_id,
        model=model,
        step=step,
    )
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO reconciliation_cache (
                step, input_hash, model, prompt_version, output_json,
                raw_content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(step, input_hash, model, prompt_version) DO NOTHING
            """,
            (
                step,
                input_hash,
                model,
                prompt_version,
                result.value.model_dump_json(),
                result.raw_content,
            ),
        )
    return CachedDecision(result.value, cache_hit=False)


def _entity_candidates(
    database_path: Path,
    client: LLMClient,
    *,
    ticker: str,
    entity_type: str,
    published_at: str,
    name: str,
    document_id: str,
    limit: int = 5,
) -> list[EntityCandidate]:
    with database_connection(database_path) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'entity_vec'"
        ).fetchone()
        if table_exists is None:
            return []
        vector_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM entity_vec"
            ).fetchone()["count"]
        )
    if vector_count == 0:
        return []

    embedded = client.embed([f"{entity_type}: {name}"], doc_id=document_id)
    if len(embedded.vectors) != 1:
        raise ValueError("Entity embedding must return exactly one vector")
    initialize_database(database_path, vector_dimensions=embedded.dimensions)

    packed_query = pack_vector(embedded.vectors[0])
    candidate_limit = min(vector_count, max(limit * 4, limit))
    while True:
        with database_connection(database_path) as connection:
            matches = connection.execute(
                """
                SELECT entity_id, distance
                FROM entity_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (packed_query, candidate_limit),
            ).fetchall()
            candidate_ids = [str(row["entity_id"]) for row in matches]
            eligible: dict[str, sqlite3.Row] = {}
            for start in range(0, len(candidate_ids), 500):
                batch = candidate_ids[start : start + 500]
                if not batch:
                    continue
                placeholders = ", ".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT id, ticker, type, canonical_name, first_seen_at
                    FROM entity
                    WHERE ticker = ? AND type = ?
                      AND julianday(first_seen_at) < julianday(?)
                      AND id IN ({placeholders})
                    """,
                    [ticker, entity_type, published_at, *batch],
                ).fetchall()
                eligible.update((str(row["id"]), row) for row in rows)

        candidates = [
            EntityCandidate(
                id=str(row["entity_id"]),
                ticker=str(eligible[str(row["entity_id"])]["ticker"]),
                type=str(eligible[str(row["entity_id"])]["type"]),
                canonical_name=str(eligible[str(row["entity_id"])]["canonical_name"]),
                first_seen_at=str(eligible[str(row["entity_id"])]["first_seen_at"]),
                distance=float(row["distance"]),
            )
            for row in matches
            if str(row["entity_id"]) in eligible
        ]
        candidates.sort(
            key=lambda item: (
                item.distance if item.distance is not None else float("inf"),
                item.id,
            )
        )
        if len(candidates) >= limit or candidate_limit >= vector_count:
            return candidates[:limit]
        candidate_limit = min(vector_count, candidate_limit * 2)


def resolve_entity(
    database_path: Path,
    client: LLMClient,
    *,
    ticker: str,
    source_entity: CandidateEntity,
    evidence: str,
    published_at: str,
    document_id: str,
    model: str,
    same_document: dict[tuple[str, str, str], ResolvedEntity],
) -> EntityResolution:
    """Resolve an entity using earlier exact names/aliases, then nearest candidates."""
    normalized_name = normalize_entity_name(source_entity.name)
    entity_type = normalize_entity_name(source_entity.type)
    if not entity_type:
        raise ValueError("Source entity type must not be blank")
    normalized_type = normalize_entity_name(entity_type)
    local_key = (ticker.upper(), normalized_type, normalized_name)
    if local_key in same_document:
        return EntityResolution(same_document[local_key])

    with database_connection(database_path) as connection:
        entity_rows = connection.execute(
            """
            SELECT id, ticker, type, canonical_name, first_seen_at
            FROM entity
            WHERE ticker = ? AND type = ?
              AND julianday(first_seen_at) < julianday(?)
            ORDER BY first_seen_at, id
            """,
            (ticker.upper(), entity_type, published_at),
        ).fetchall()
        alias_rows = connection.execute(
            """
            SELECT e.id, e.ticker, e.type, e.canonical_name, e.first_seen_at,
                   a.alias, a.first_seen_at AS alias_first_seen_at
            FROM entity_alias AS a
            JOIN entity AS e ON e.id = a.entity_id
            WHERE e.ticker = ? AND e.type = ?
              AND julianday(e.first_seen_at) < julianday(?)
              AND julianday(a.first_seen_at) < julianday(?)
            ORDER BY a.first_seen_at, e.id
            """,
            (ticker.upper(), entity_type, published_at, published_at),
        ).fetchall()

    exact_matches = [
        row
        for row in entity_rows
        if normalize_entity_name(str(row["canonical_name"])) == normalized_name
    ]
    alias_matches = [
        row
        for row in alias_rows
        if normalize_entity_name(str(row["alias"])) == normalized_name
    ]
    exact_by_id = {str(row["id"]): row for row in exact_matches + alias_matches}
    if len(exact_by_id) == 1:
        row = next(iter(exact_by_id.values()))
        resolved = ResolvedEntity(
            id=str(row["id"]),
            ticker=str(row["ticker"]),
            type=str(row["type"]),
            canonical_name=str(row["canonical_name"]),
            first_seen_at=str(row["first_seen_at"]),
            created=False,
            alias_to_add=(
                source_entity.name
                if normalize_entity_name(str(row["canonical_name"])) != normalized_name
                and not alias_matches
                else None
            ),
        )
        same_document[local_key] = resolved
        return EntityResolution(resolved)

    candidates: list[EntityCandidate] = [
        EntityCandidate(
            id=str(row["id"]),
            ticker=str(row["ticker"]),
            type=str(row["type"]),
            canonical_name=str(row["canonical_name"]),
            first_seen_at=str(row["first_seen_at"]),
            distance=0.0,
        )
        for row in exact_by_id.values()
    ]
    if not candidates:
        candidates = _entity_candidates(
            database_path,
            client,
            ticker=ticker.upper(),
            entity_type=entity_type,
            published_at=published_at,
            name=source_entity.name,
            document_id=document_id,
        )

    cache_hit = False
    if not candidates:
        decision = EntityChoice(choice="new", reason="No earlier entity candidates.")
    else:
        input_data: dict[str, object] = {
            "ticker": ticker.upper(),
            "published_at": published_at,
            "source_entity": {
                "name": source_entity.name,
                "type": source_entity.type,
            },
            "source_evidence": evidence,
            "candidates": [
                {
                    "id": item.id,
                    "name": item.canonical_name,
                    "type": item.type,
                    "distance": item.distance,
                }
                for item in candidates
            ],
        }
        cached_decision = cached_structured_decision(
            database_path,
            client,
            step="entity_resolution",
            prompt=(_PROMPT_ROOT / "resolve_entity.md").read_text(
                encoding="utf-8"
            ),
            input_data=input_data,
            response_model=EntityChoice,
            model=model,
            document_id=document_id,
        )
        decision = cached_decision.value
        cache_hit = cached_decision.cache_hit
        if decision.choice != "existing" and decision.candidate_index is not None:
            return EntityResolution(
                None,
                tuple(candidates),
                "Entity resolver returned a candidate index for a non-existing choice.",
                cache_hit=cache_hit,
            )
        if decision.choice == "existing":
            if (
                decision.candidate_index is None
                or decision.candidate_index >= len(candidates)
            ):
                return EntityResolution(
                    None,
                    tuple(candidates),
                    "Entity resolver selected an unavailable candidate index.",
                    cache_hit=cache_hit,
                )
            selected = candidates[decision.candidate_index]
            resolved = ResolvedEntity(
                id=selected.id,
                ticker=selected.ticker,
                type=selected.type,
                canonical_name=selected.canonical_name,
                first_seen_at=selected.first_seen_at,
                created=False,
                alias_to_add=(
                    source_entity.name
                    if normalize_entity_name(selected.canonical_name) != normalized_name
                    and not alias_matches
                    else None
                ),
            )
            same_document[local_key] = resolved
            return EntityResolution(
                resolved,
                tuple(candidates),
                decision.reason,
                cache_hit=cache_hit,
            )
        if decision.choice == "unresolved":
            return EntityResolution(
                None,
                tuple(candidates),
                decision.reason,
                cache_hit=cache_hit,
            )

    new_id = stable_entity_id(ticker, entity_type, source_entity.name)
    embedded = client.embed(
        [f"{ticker.upper()} {entity_type}: {source_entity.name}"],
        doc_id=document_id,
    )
    if len(embedded.vectors) != 1:
        raise ValueError("Entity embedding must return exactly one vector")
    initialize_database(database_path, vector_dimensions=embedded.dimensions)
    resolved = ResolvedEntity(
        id=new_id,
        ticker=ticker.upper(),
        type=entity_type,
        canonical_name=source_entity.name.strip(),
        first_seen_at=published_at,
        created=True,
        embedding=tuple(embedded.vectors[0]),
    )
    same_document[local_key] = resolved
    return EntityResolution(
        resolved,
        tuple(candidates),
        decision.reason,
        cache_hit=cache_hit,
    )


def _attribute_description(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _canonical_attribute(
    connection: sqlite3.Connection,
    name: str,
) -> tuple[str, str, bool]:
    normalized = normalize_attribute_name(name)
    if not normalized:
        raise ValueError(f"Attribute name is empty after normalization: {name!r}")

    rows = connection.execute(
        "SELECT name, description, status, merged_into FROM attribute ORDER BY name"
    ).fetchall()
    matching = [
        row
        for row in rows
        if normalize_attribute_name(str(row["name"])) == normalized
    ]
    if not matching:
        return normalized, _attribute_description(normalized), True
    if len(matching) > 1:
        raise ValueError(f"Attribute normalization is ambiguous: {name!r}")

    row = matching[0]
    visited: set[str] = set()
    while row["merged_into"] is not None:
        current_name = str(row["name"])
        if current_name in visited:
            raise ValueError(f"Attribute merge cycle contains {current_name!r}")
        visited.add(current_name)
        target = connection.execute(
            "SELECT name, description, status, merged_into FROM attribute WHERE name = ?",
            (row["merged_into"],),
        ).fetchone()
        if target is None:
            raise ValueError(
                f"Attribute {current_name!r} merges into a missing attribute"
            )
        row = target
    if row["status"] != "active":
        raise ValueError(
            f"Attribute {row['name']!r} is not active and has no merge target"
        )
    return str(row["name"]), str(row["description"]), False


def resolve_attribute(
    database_path: Path,
    name: str,
    *,
    same_document: dict[str, ResolvedAttribute],
) -> ResolvedAttribute:
    """Resolve normalized attributes and follow explicit registry merges."""
    normalized = normalize_attribute_name(name)
    if normalized in same_document:
        return same_document[normalized]
    with database_connection(database_path) as connection:
        resolved_name, description, created = _canonical_attribute(connection, name)
    resolved = ResolvedAttribute(
        name=resolved_name,
        description=description,
        created=created,
    )
    same_document[normalized] = resolved
    return resolved
