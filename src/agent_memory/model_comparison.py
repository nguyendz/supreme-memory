"""Reproducible local extraction-model comparison on ten fixed documents."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama

from agent_memory.db import database_connection
from agent_memory.llm import LLMClient
from agent_memory.pipeline.extract import (
    CandidateFact,
    ExtractionResult,
    extract_document,
)


LOCAL_MODELS = ("qwen3.5:9b", "gemma4", "nimble")
FIXED_DOCUMENT_IDS = (
    "20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068",
    "20250513-2303-Restart-Study-Repositions-Finniss-Operations-2A1596413",
    "20260317-2305-FID-Approved-and-Funding-Secured-for-Finniss-Restart-2A1661043",
    "20260810-2236-New-High-Grade-Lithium-Drill-Results-at-BP33-6A1338045",
    "20260429-2321-Sale-of-20,000-Tonnes-of-Lithium-Fines-6A1323092",
    "20250928-2328-Application-for-quotation-of-securities-CXO-2A1624827",
    "20250828-0016-Proposed-issue-of-securities-CXO-2A1617338",
    "20250928-2334-Change-of-Director's-Interest-M-McComas-2A1624839",
    "20251023-2126-Becoming-a-substantial-holder-2A1631214",
    "20260512-0659-Change-in-substantial-holding-6A1325174",
)


@dataclass(frozen=True)
class ModelComparisonRow:
    document_id: str
    model: str
    status: str
    cache_hit: bool
    facts: tuple[CandidateFact, ...]
    accepted_facts: tuple[CandidateFact, ...]
    prompt_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 0
    wall_duration_ms: float = 0
    truncated_fact_count: int = 0
    error: str | None = None

    @property
    def quote_pass_rate(self) -> float | None:
        candidates = [fact for fact in self.facts if fact.tier != "T0"]
        if not candidates:
            return None
        accepted = [
            fact for fact in self.accepted_facts if fact.tier != "T0"
        ]
        return len(accepted) / len(candidates)


def _row_from_result(
    document_id: str, model: str, result: ExtractionResult
) -> ModelComparisonRow:
    return ModelComparisonRow(
        document_id=document_id,
        model=model,
        status=result.status,
        cache_hit=result.cache_hit,
        facts=result.facts,
        accepted_facts=result.accepted_facts,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        attempts=result.attempts,
        wall_duration_ms=result.wall_duration_ms,
        truncated_fact_count=result.truncated_fact_count,
    )


def _render_report(rows: list[ModelComparisonRow]) -> str:
    lines = [
        "# Local extraction model comparison",
        "",
        "All runs use the same ten fixed source documents and local Ollama models.",
        "A model is not selected by latency alone; review candidate facts and quotes.",
        "",
        "| Document | Model | Status | Facts | Recovery-capped | Non-T0 quote pass | Prompt tokens | Output tokens | Attempts | Wall ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pass_rate = (
            "n/a"
            if row.quote_pass_rate is None
            else f"{row.quote_pass_rate:.1%}"
        )
        lines.append(
            f"| `{row.document_id}` | `{row.model}` | {row.status} | "
            f"{len(row.facts)} | {row.truncated_fact_count} | {pass_rate} | {row.prompt_tokens} | "
            f"{row.output_tokens} | {row.attempts} | {row.wall_duration_ms:.0f} |"
        )
        if row.error:
            lines.append(
                "| _Error: "
                + row.error.replace("|", "\\|")
                + "_ | | | | | | | | |"
            )

    lines.extend(("", "## Candidate facts and evidence", ""))
    for document_id in FIXED_DOCUMENT_IDS:
        lines.extend((f"### `{document_id}`", ""))
        document_rows = [row for row in rows if row.document_id == document_id]
        for row in document_rows:
            lines.append(f"#### `{row.model}`")
            if row.error:
                lines.append(f"- Error: {row.error}")
            elif row.status == "unsupported":
                lines.append("- Unsupported form; no extraction attempted.")
            elif not row.facts:
                lines.append("- No candidate facts.")
            else:
                for fact in row.facts:
                    accepted = fact in row.accepted_facts
                    marker = "quote pass" if accepted else "quote rejected"
                    lines.append(
                        f"- [{fact.tier}/{fact.fact_type}; {marker}] "
                        f"{fact.text} - evidence: `{fact.quote}`"
                    )
            lines.append("")
    return "\n".join(lines) + "\n"


def run_model_comparison(
    database_path: Path,
    client: LLMClient,
    report_path: Path,
    *,
    rejection_path: Path | None = None,
) -> list[ModelComparisonRow]:
    """Run the fixed 30 local model/document extraction comparisons."""
    with database_connection(database_path) as connection:
        present = {
            row["id"]
            for row in connection.execute(
                f"""
                SELECT id FROM document
                WHERE id IN ({", ".join("?" for _ in FIXED_DOCUMENT_IDS)})
                """,
                FIXED_DOCUMENT_IDS,
            ).fetchall()
        }
    rows: list[ModelComparisonRow] = []
    for document_id in FIXED_DOCUMENT_IDS:
        for model in LOCAL_MODELS:
            if document_id not in present:
                rows.append(
                    ModelComparisonRow(
                        document_id=document_id,
                        model=model,
                        status="missing",
                        cache_hit=False,
                        facts=(),
                        accepted_facts=(),
                        error="document is not in the selected database",
                    )
                )
                continue
            try:
                result = extract_document(
                    database_path,
                    client,
                    document_id,
                    model=model,
                    rejection_path=rejection_path,
                )
            except (
                OSError,
                ValueError,
                sqlite3.Error,
                httpx.HTTPError,
                ollama.ResponseError,
            ) as exc:
                rows.append(
                    ModelComparisonRow(
                        document_id=document_id,
                        model=model,
                        status="failed",
                        cache_hit=False,
                        facts=(),
                        accepted_facts=(),
                        error=str(exc),
                    )
                )
            else:
                rows.append(_row_from_result(document_id, model, result))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(rows), encoding="utf-8")
    return rows
