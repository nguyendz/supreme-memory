"""Command-line entry point for the announcement-memory MVP."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer
from pydantic import BaseModel
from rich.console import Console

from agent_memory.config import AppConfig, load_config
from agent_memory.db import initialize_database
from agent_memory.llm import LLMClient
from agent_memory.loader import discover_announcements


app = typer.Typer(no_args_is_help=True, help="Local company-announcement memory.")
console = Console()


def _load_settings(config: Path | None) -> AppConfig:
    return load_config(config)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
