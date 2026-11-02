"""Benchmark one representative quarterly report through local Ollama."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from agent_memory.config import load_config
from agent_memory.db import database_connection, initialize_database, upsert_document
from agent_memory.llm import LLMClient
from agent_memory.loader import (
    Announcement,
    discover_announcements,
    read_announcement_text,
)


class QuarterlySummary(BaseModel):
    summary: str
    key_points: list[str] = Field(max_length=5)


def _target_announcement(announcements: list[Announcement]) -> Announcement:
    matches = [
        announcement
        for announcement in announcements
        if announcement.published_at.date() == datetime(2026, 10, 13).date()
        and "Quarterly Activities and Cashflow Report" in announcement.headline
        and announcement.text_path is not None
    ]
    if not matches:
        raise FileNotFoundError(
            "Could not find the 13 October 2026 quarterly activities and cashflow report"
        )
    return matches[0]


def _tokens_per_second(tokens: int, duration_ms: float) -> str:
    if duration_ms <= 0:
        return "n/a"
    return f"{tokens / (duration_ms / 1000):.1f}"


def main() -> None:
    config = load_config()
    announcements = discover_announcements(config.data_dir, ticker="CXO")
    announcement = _target_announcement(announcements)
    text = read_announcement_text(announcement)

    initialize_database(config.database_path)
    with database_connection(config.database_path) as connection:
        upsert_document(
            connection,
            doc_id=announcement.id,
            ticker=announcement.ticker,
            folder=announcement.folder.name,
            headline=announcement.headline,
            published_at=announcement.published_at.isoformat(),
            category=announcement.category,
            announcement_types=announcement.announcement_types,
            is_price_sensitive=announcement.is_price_sensitive,
            text_path=str(announcement.text_path) if announcement.text_path else None,
            text_source=announcement.text_source,
        )
    client = LLMClient(config, config.database_path)
    result = client.chat_json(
        (
            "Summarise this public-company quarterly report using only its text. "
            "Return a concise summary and no more than five key points.\n\n"
            f"Announcement: {announcement.headline}\n"
            f"Published: {announcement.published_at.isoformat()}\n\n{text}"
        ),
        QuarterlySummary,
        doc_id=announcement.id,
    )
    embedding_result = client.embed(
        [
            announcement.headline,
            "Finniss first spodumene concentrate production",
            "BP33 underground development schedule",
        ],
        doc_id=announcement.id,
    )
    initialize_database(
        config.database_path,
        vector_dimensions=embedding_result.dimensions,
    )

    report_path = config.project_root / "var" / "bench.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    report = f"""# Wave 0 Ollama benchmark

- Generated: {generated_at}
- Announcement: `{announcement.id}`
- Text source: {announcement.text_source}
- Input size: {len(text):,} characters
- Chat model: `{config.ollama_model}`
- Embedding model: `{config.embedding_model}`
- Context setting: {config.num_ctx:,} tokens
- End-to-end chat time: {result.wall_duration_ms / 1000:.2f} seconds
- Prompt evaluation: {result.prompt_tokens:,} tokens in {result.prompt_duration_ms / 1000:.2f} seconds ({_tokens_per_second(result.prompt_tokens, result.prompt_duration_ms)} tokens/s)
- Generation: {result.output_tokens:,} tokens in {result.generation_duration_ms / 1000:.2f} seconds ({_tokens_per_second(result.output_tokens, result.generation_duration_ms)} tokens/s)
- Chat attempts: {result.attempts}
- Embedding dimensions: {embedding_result.dimensions}
- Embedding time for 3 texts: {embedding_result.wall_duration_ms / 1000:.2f} seconds

## Model output

{result.value.summary}

"""
    report += "\n".join(f"- {point}" for point in result.value.key_points)
    report_path.write_text(report, encoding="utf-8")
    print(f"Benchmark report: {report_path}")
    print(f"Chat time: {result.wall_duration_ms / 1000:.2f} seconds")
    print(f"Embedding dimensions: {embedding_result.dimensions}")


if __name__ == "__main__":
    main()
