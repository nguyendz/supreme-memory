"""Command-line entry point for the announcement-memory MVP."""

from __future__ import annotations

import webbrowser
from collections import Counter
from datetime import date
from pathlib import Path
import sqlite3

import httpx
import ollama
import typer
from pydantic import BaseModel
from rich.console import Console

from agent_memory.config import AppConfig, load_config, validate_wave2_local_runtime
from agent_memory.db import database_connection, initialize_database
from agent_memory.evaluation import run_evaluation
from agent_memory.llm import LLMClient
from agent_memory.loader import discover_announcements
from agent_memory.pipeline.extract import run_extraction
from agent_memory.pipeline.ingest import run_ingest
from agent_memory.pipeline.review import export_t1_review
from agent_memory.pipeline.reconcile import (
    APPROVED_EXTRACTION_MODELS,
    preflight_reconciliation,
    preflight_pilot_reconciliation,
    run_reconciliation,
    run_pilot_reconciliation,
)
from agent_memory.query.baseline_rag import answer_baseline, build_baseline_index
from agent_memory.review_ui import generate_model_review_ui


app = typer.Typer(no_args_is_help=True, help="Local company-announcement memory.")
console = Console()


def _load_settings(config: Path | None) -> AppConfig:
    return load_config(config)


def _parse_cli_date(value: str | None, option_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must use YYYY-MM-DD") from exc


def _validate_wave2_runtime(
    settings: AppConfig,
    *,
    require_chat_model: bool = True,
    require_embedding_model: bool = True,
) -> None:
    try:
        validate_wave2_local_runtime(
            settings,
            require_chat_model=require_chat_model,
            require_embedding_model=require_embedding_model,
        )
    except (
        ValueError,
        OSError,
        sqlite3.Error,
        httpx.HTTPError,
        ollama.ResponseError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def stats(
    ticker: str | None = typer.Option(None, help="Limit the report to this ticker."),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Report source-document coverage without calling the LLM."""
    settings = _load_settings(config)
    announcements = discover_announcements(settings.data_dir, ticker=ticker)
    if not announcements:
        console.print(f"No announcements found under [path]{settings.data_dir}[/path]")
        return

    sources = Counter(item.text_source for item in announcements)
    missing_metadata = sum(item.metadata_path is None for item in announcements)
    first_date = min(item.published_at for item in announcements)
    last_date = max(item.published_at for item in announcements)

    console.print(f"Ticker: {ticker.upper() if ticker else 'All'}")
    console.print(f"Announcements: {len(announcements)}")
    console.print(
        "Markdown: "
        f"v2={sources['v2']}, v1={sources['v1']}, missing={sources['missing']}"
    )
    console.print(f"Missing metadata: {missing_metadata}")
    console.print(f"Published: {first_date.isoformat()} through {last_date.isoformat()}")
    console.print(f"Data directory: {settings.data_dir}")
    for announcement in announcements:
        if announcement.text_source == "missing" or announcement.metadata_path is None:
            missing = []
            if announcement.text_source == "missing":
                missing.append("markdown")
            if announcement.metadata_path is None:
                missing.append("metadata")
            console.print(
                f"Missing {' and '.join(missing)}: {announcement.id}"
            )


@app.command("db-init")
def db_init(
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Create the SQLite database and apply its migrations."""
    settings = _load_settings(config)
    database_path = initialize_database(
        settings.database_path, vector_dimensions=settings.embedding_dimensions
    )
    console.print(f"Database ready: {database_path}")


class _ProbeResponse(BaseModel):
    ok: bool
    message: str


@app.command()
def llm_check(
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Verify local Ollama structured output and embedding calls."""
    settings = _load_settings(config)
    initialize_database(settings.database_path)
    client = LLMClient(settings, settings.database_path)
    chat_result = client.chat_json(
        'Return {"ok": true, "message": "local Ollama is ready"}.',
        _ProbeResponse,
    )
    embedding_result = client.embed(["announcement memory smoke test"])
    initialize_database(
        settings.database_path, vector_dimensions=embedding_result.dimensions
    )
    console.print(
        f"Chat model: {settings.ollama_model} "
        f"({chat_result.wall_duration_ms:.0f} ms, "
        f"{chat_result.output_tokens} output tokens)"
    )
    console.print(
        f"Embedding model: {settings.embedding_model} "
        f"({embedding_result.dimensions} dimensions, "
        f"{embedding_result.wall_duration_ms:.0f} ms)"
    )
    console.print(f"Database ready: {settings.database_path}")


@app.command()
def ingest(
    ticker: str | None = typer.Option(None, help="Limit processing to this ticker."),
    from_date: str | None = typer.Option(
        None, "--from", help="Inclusive UTC publication date."
    ),
    to_date: str | None = typer.Option(
        None, "--to", help="Inclusive UTC publication date."
    ),
    limit: int | None = typer.Option(None, help="Limit the sorted document set."),
    reingest: bool = typer.Option(
        False, "--reingest", help="Rebuild stored outputs in the selected date range."
    ),
    warmup: bool = typer.Option(
        False, "--warmup", help="Build point-in-time hash history only; make no LLM calls."
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Classify announcements and build searchable chunks without fact extraction."""
    settings = _load_settings(config)
    summary = run_ingest(
        settings,
        ticker=ticker,
        start_date=_parse_cli_date(from_date, "--from"),
        end_date=_parse_cli_date(to_date, "--to"),
        limit=limit,
        reingest=reingest,
        warmup=warmup,
    )
    console.print(
        f"Discovered: {summary.discovered}; selected: {summary.selected}; "
        f"done: {summary.done}; skipped: {summary.skipped}; "
        f"failed: {summary.failed}; no-op: {summary.no_op}"
    )
    if warmup:
        console.print(
            f"Hash history warmed for {summary.warmed_documents} documents "
            "(no LLM calls)."
        )
    else:
        console.print(
            f"Cross-references: {summary.resolved_references}/"
            f"{summary.references} resolved; LLM calls: {summary.llm_calls}; "
            f"elapsed: {summary.elapsed_seconds:.2f}s"
        )
    for doc_id, error in summary.errors:
        console.print(f"[red]Failed {doc_id}: {error}[/red]")
    if summary.failed:
        raise typer.Exit(code=1)


def _baseline_ticker(database_path: Path, requested: str | None) -> str:
    if requested is not None:
        normalized = requested.strip().upper()
        if not normalized:
            raise typer.BadParameter("--ticker must not be blank")
        return normalized
    with database_connection(database_path) as connection:
        tickers = [
            row["ticker"]
            for row in connection.execute(
                "SELECT DISTINCT ticker FROM document ORDER BY ticker"
            ).fetchall()
        ]
    if len(tickers) != 1:
        raise typer.BadParameter(
            "Specify --ticker when the baseline corpus contains multiple tickers"
        )
    return tickers[0]


@app.command()
def extract(
    database: Path = typer.Option(
        ...,
        "--database",
        exists=True,
        dir_okay=False,
        help="Isolated source database to extract from; no canonical facts are written.",
    ),
    ticker: str | None = typer.Option(None, help="Limit extraction to this ticker."),
    from_date: str | None = typer.Option(
        None, "--from", help="Inclusive publication date."
    ),
    to_date: str | None = typer.Option(
        None, "--to", help="Inclusive publication date."
    ),
    limit: int | None = typer.Option(None, min=1, help="Limit sorted documents."),
    force: bool = typer.Option(
        False, "--force", help="Re-run extraction despite a matching cache entry."
    ),
    confirm_wave1_gate: bool = typer.Option(
        False,
        "--confirm-wave1-gate",
        help="Confirm triage, boilerplate, and cross-reference reviews are complete.",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Extract quote-checked candidate facts into the isolated extraction cache."""
    if not confirm_wave1_gate:
        raise typer.BadParameter(
            "Do not run extraction until the Wave 1 Gate 1 manual reviews are accepted"
        )
    settings = _load_settings(config)
    _validate_wave2_runtime(settings, require_embedding_model=False)
    client = LLMClient(settings, database)
    eval_directory = settings.project_root / "var" / "eval"
    summary = run_extraction(
        database,
        client,
        ticker=ticker,
        start_date=_parse_cli_date(from_date, "--from"),
        end_date=_parse_cli_date(to_date, "--to"),
        limit=limit,
        force=force,
        rejection_path=eval_directory / "quote-rejections.csv",
        unsupported_path=eval_directory / "unsupported_forms.csv",
    )
    console.print(
        f"Discovered: {summary.discovered}; selected: {summary.selected}; "
        f"extracted: {summary.extracted}; cache hits: {summary.cache_hits}; "
        f"unsupported: {summary.unsupported}; failed: {summary.failed}; "
        f"recovery-capped facts: {summary.truncated_facts}"
    )
    console.print(
        f"Prompt tokens: {summary.prompt_tokens}; output tokens: "
        f"{summary.output_tokens}; attempts: {summary.attempts}; "
        f"elapsed: {summary.elapsed_seconds:.2f}s"
    )
    for doc_id, error in summary.errors:
        console.print(f"[red]Failed {doc_id}: {error}[/red]")
    if summary.failed:
        raise typer.Exit(code=1)


@app.command("t1-review")
def t1_review(
    database: Path = typer.Option(
        ...,
        "--database",
        exists=True,
        dir_okay=False,
        help="Isolated source database containing extraction cache results.",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="CSV output path for the review file."
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Export quote-verified T1 candidates from the five fixed quarterlies."""
    settings = _load_settings(config)
    output_path = output or settings.project_root / "var" / "eval" / "t1-review.csv"
    summary = export_t1_review(database, settings.ollama_model, output_path)
    console.print(
        f"Quarterlies: {summary.cached_documents}/{summary.documents} cached; "
        f"T1 candidates: {summary.candidates}; review file: {output_path}"
    )
    for doc_id in summary.stale_or_missing:
        console.print(f"[yellow]No current extraction cache: {doc_id}[/yellow]")


@app.command("reconcile")
def reconcile(
    database: Path = typer.Option(
        ...,
        "--database",
        "--source-db",
        exists=True,
        dir_okay=False,
        help="Isolated source database with extraction caches.",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Approved extraction-cache model to reconcile."
    ),
    ticker: str | None = typer.Option(None, help="Limit to this ticker."),
    from_date: str | None = typer.Option(
        None, "--from", help="Inclusive UTC publication date."
    ),
    to_date: str | None = typer.Option(
        None, "--to", help="Inclusive UTC publication date."
    ),
    limit: int | None = typer.Option(None, min=1, help="Limit sorted documents."),
    replay: bool = typer.Option(
        False, "--replay", help="Clear and rebuild all canonical memory for --ticker."
    ),
    pilot: bool = typer.Option(
        False,
        "--pilot",
        help="Reconcile one gold-backed document into a new disposable pilot DB.",
    ),
    pilot_database: Path | None = typer.Option(
        None,
        "--pilot-db",
        dir_okay=False,
        help="New output database path under var/wave3 for --pilot.",
    ),
    document_id: str | None = typer.Option(
        None, "--document-id", help="One explicit source document for --pilot."
    ),
    preflight: bool = typer.Option(
        False, "--preflight", help="Validate caches and quotes without writing facts."
    ),
    confirm_wave2_gate: bool = typer.Option(
        False,
        "--confirm-wave2-gate",
        help="Confirm Wave 2 review and extraction-quality gates are accepted.",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Reconcile quote-verified extraction caches into point-in-time memory."""
    settings = _load_settings(config)
    selected_model = model or settings.ollama_model
    if selected_model not in APPROVED_EXTRACTION_MODELS:
        raise typer.BadParameter(
            "Reconciliation requires an approved extraction model "
            "(qwen3.5:9b or gemma4); decision-only models are not accepted"
        )
    try:
        start = _parse_cli_date(from_date, "--from")
        end = _parse_cli_date(to_date, "--to")
        if pilot:
            if (
                ticker is not None
                or start is not None
                or end is not None
                or limit is not None
                or replay
                or confirm_wave2_gate
            ):
                raise typer.BadParameter(
                    "--pilot requires only --document-id and cannot be combined "
                    "with ticker/date/limit/replay/full-gate options"
                )
            if document_id is None:
                raise typer.BadParameter("--document-id is required with --pilot")
            if database.expanduser().resolve() == settings.database_path.resolve():
                raise typer.BadParameter(
                    "The primary database cannot be used as a pilot source"
                )
            if preflight:
                if pilot_database is not None:
                    raise typer.BadParameter(
                        "--pilot-db is used only when executing a pilot"
                    )
                result = preflight_pilot_reconciliation(
                    database,
                    document_id=document_id,
                    model=selected_model,
                )
                console.print(
                    f"Pilot preflight: {'ready' if result.eligible else 'blocked'}; "
                    f"model: {result.model}; document: {result.document_id}; "
                    f"matching candidates: {result.matching_candidates}"
                )
                if not result.eligible:
                    console.print(f"[red]{result.reason}[/red]")
                    raise typer.Exit(code=1)
                console.print("[green]No database was changed.[/green]")
                return
            if pilot_database is None:
                raise typer.BadParameter("--pilot-db is required to execute a pilot")
            pilot_root = (settings.project_root / "var" / "wave3").resolve()
            pilot_path = pilot_database.expanduser().resolve()
            if not pilot_path.is_relative_to(pilot_root):
                raise typer.BadParameter(
                    f"--pilot-db must be under the disposable directory: {pilot_root}"
                )
            _validate_wave2_runtime(settings, require_chat_model=False)
            pilot_client = LLMClient(settings, pilot_path)
            summary = run_pilot_reconciliation(
                database,
                pilot_path,
                pilot_client,
                document_id=document_id,
                model=selected_model,
            )
            console.print(
                "[yellow]EXPLORATORY PILOT ONLY — Gate 2 remains open.[/yellow]"
            )
            console.print(
                f"Pilot DB: {pilot_path}; document: {document_id}; "
                f"model: {summary.model}; ADD {summary.added}, "
                f"CONFIRM {summary.confirmed}, SUPERSEDE {summary.superseded}; "
                f"quote rejected {summary.quote_rejected}; "
                f"unresolved {summary.unresolved}"
            )
            console.print(
                f"LLM calls: {summary.llm_calls}; prompt tokens: "
                f"{summary.prompt_tokens}; output tokens: {summary.output_tokens}; "
                f"elapsed: {summary.elapsed_seconds:.2f}s"
            )
            return
        if pilot_database is not None or document_id is not None:
            raise typer.BadParameter(
                "--pilot-db and --document-id require --pilot"
            )
        if preflight:
            if replay or confirm_wave2_gate:
                raise typer.BadParameter(
                    "--preflight cannot be combined with --replay or "
                    "--confirm-wave2-gate"
                )
            result = preflight_reconciliation(
                database,
                model=selected_model,
                ticker=ticker,
                start_date=start,
                end_date=end,
                limit=limit,
            )
            console.print(
                f"Preflight for {selected_model}: {result.current_caches}/"
                f"{result.requested_documents} current caches; "
                f"unsupported forms: {len(result.unsupported)}; "
                f"quote verified: {result.quote_verified}; "
                f"quote rejected: {result.quote_rejected}; T0: "
                f"{result.t0_candidates}; recovery-capped: {result.recovery_capped}"
            )
            for doc_id in result.missing:
                console.print(f"[yellow]Missing cache: {doc_id}[/yellow]")
            for doc_id in result.stale:
                console.print(f"[yellow]Stale cache: {doc_id}[/yellow]")
            for doc_id, error in result.invalid:
                console.print(f"[red]Invalid cache {doc_id}: {error}[/red]")
            if result.missing or result.stale or result.invalid:
                raise typer.Exit(code=1)
            return
        if not confirm_wave2_gate:
            raise typer.BadParameter(
                "Do not promote candidate facts until Wave 2 Gate 2 is accepted"
            )
        _validate_wave2_runtime(settings, require_chat_model=False)
        client = LLMClient(settings, database)
        summary = run_reconciliation(
            database,
            client,
            model=selected_model,
            ticker=ticker,
            start_date=start,
            end_date=end,
            limit=limit,
            replay=replay,
            confirm_wave2_gate=confirm_wave2_gate,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(
        f"Reconciled {summary.processed}/{summary.documents} documents with "
        f"{summary.model}: ADD {summary.added}, CONFIRM {summary.confirmed}, "
        f"SUPERSEDE {summary.superseded}; T0 ignored {summary.ignored_t0}; "
        f"quote rejected {summary.quote_rejected}; unresolved {summary.unresolved}; "
        f"unsupported forms {summary.unsupported}"
    )
    if summary.add_ratio is not None:
        console.print(f"ADD ratio: {summary.add_ratio:.1%}")
    console.print(
        f"LLM calls: {summary.llm_calls}; decision cache hits: "
        f"{summary.decision_cache_hits}; prompt tokens: {summary.prompt_tokens}; "
        f"output tokens: {summary.output_tokens}; "
        f"elapsed: {summary.elapsed_seconds:.2f}s"
    )
    for result in summary.document_results:
        console.print(
            f"{result.document_id}: ADD {result.added}, "
            f"CONFIRM {result.confirmed}, SUPERSEDE {result.superseded}; "
            f"unresolved {result.unresolved}; "
            f"quote rejected {result.quote_rejected}"
        )
    for doc_id in summary.stale_or_missing:
        console.print(f"[yellow]Missing or stale cache: {doc_id}[/yellow]")
    for doc_id, error in summary.errors:
        console.print(f"[red]Failed {doc_id}: {error}[/red]")
    if summary.stale_or_missing or summary.errors:
        raise typer.Exit(code=1)


@app.command("review-models")
def review_models(
    database: Path = typer.Option(
        ...,
        "--database",
        exists=True,
        dir_okay=False,
        help="Isolated source database containing model-comparison caches.",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Standalone HTML review UI output path."
    ),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the generated UI in the default browser."
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Generate a local browser UI for comparing cached extraction models."""
    settings = _load_settings(config)
    output_path = output or settings.project_root / "var" / "eval" / "model-review.html"
    generate_model_review_ui(database, output_path)
    console.print(f"Model review UI: {output_path.resolve()}")
    if open_browser and not webbrowser.open(output_path.resolve().as_uri()):
        console.print("[yellow]Could not open a browser automatically.[/yellow]")


@app.command("baseline-index")
def baseline_index(
    source_database: Path = typer.Option(
        ...,
        "--source-db",
        exists=True,
        dir_okay=False,
        help="Source corpus database to clone; the source remains unchanged.",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Build the isolated all-chunk vector index used by baseline RAG."""
    settings = _load_settings(config)
    _validate_wave2_runtime(settings, require_chat_model=False)
    baseline_path = settings.project_root / "var" / "eval" / "baseline.db"
    client = LLMClient(settings, baseline_path)
    stats = build_baseline_index(source_database, baseline_path, client)
    console.print(
        f"Baseline index ready: {baseline_path}; chunks: "
        f"{stats.embedded_chunks}/{stats.source_chunks}; "
        f"dimensions: {stats.dimensions or 'none'}"
    )


@app.command()
def ask(
    question: str,
    baseline: bool = typer.Option(
        False, "--baseline", help="Use the naive chunk-RAG baseline."
    ),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Inclusive knowledge date (YYYY-MM-DD)."
    ),
    ticker: str | None = typer.Option(None, help="Restrict retrieval to this ticker."),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Answer from a baseline corpus with validated source citations."""
    if not baseline:
        raise typer.BadParameter("Currently supported mode requires --baseline")
    knowledge_date = _parse_cli_date(as_of, "--as-of")
    if knowledge_date is None:
        raise typer.BadParameter("--as-of is required with --baseline")
    settings = _load_settings(config)
    baseline_path = settings.project_root / "var" / "eval" / "baseline.db"
    if not baseline_path.is_file():
        raise typer.BadParameter(
            f"Baseline database not found; run am baseline-index first: {baseline_path}"
        )
    _validate_wave2_runtime(settings)
    selected_ticker = _baseline_ticker(baseline_path, ticker)
    client = LLMClient(settings, baseline_path)
    answer = answer_baseline(
        baseline_path,
        client,
        question,
        knowledge_date=knowledge_date,
        ticker=selected_ticker,
    )
    console.print(answer.answer)
    for citation in answer.citations:
        console.print(f"[dim]{citation.document_id}: {citation.quote}[/dim]")


@app.command("eval")
def evaluate(
    baseline: bool = typer.Option(
        False, "--baseline", help="Evaluate the naive chunk-RAG baseline."
    ),
    questions: Path | None = typer.Option(
        None, "--questions", exists=True, dir_okay=False, help="Gold YAML file."
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Markdown report output path."
    ),
    ticker: str | None = typer.Option(None, help="Override the gold-set ticker."),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml."),
) -> None:
    """Evaluate baseline answers against the sourced gold question set."""
    if not baseline:
        raise typer.BadParameter("Currently supported evaluation requires --baseline")
    settings = _load_settings(config)
    baseline_path = settings.project_root / "var" / "eval" / "baseline.db"
    if not baseline_path.is_file():
        raise typer.BadParameter(
            f"Baseline database not found; run am baseline-index first: {baseline_path}"
        )
    _validate_wave2_runtime(settings)
    question_path = questions or settings.project_root / "eval" / "questions.yaml"
    report_path = report or settings.project_root / "var" / "eval" / "baseline.md"
    client = LLMClient(settings, baseline_path)
    summary = run_evaluation(
        baseline_path,
        client,
        question_path,
        report_path,
        ticker=ticker,
    )
    console.print(
        f"Loaded: {summary.loaded}; evaluated: {summary.evaluated}; "
        f"deferred: {summary.deferred}; failed: {summary.failed}; "
        f"correct: {summary.correct}; report: {report_path}"
    )
    if summary.failed:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
