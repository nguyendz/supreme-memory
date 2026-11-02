"""Write a reproducible human-review sample of Wave 1 triage results."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date
from pathlib import Path

from agent_memory.config import AppConfig, load_config
from agent_memory.db import database_connection


def write_triage_review_sample(
    config: AppConfig,
    *,
    sample_size: int = 30,
    seed: int = 2026,
    start_date: date = date(2025, 1, 1),
    end_date: date = date(2026, 10, 31),
) -> Path:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    with database_connection(config.database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, ticker, published_at, headline, category,
                   price_sensitive, triage_class, form_type, triage_reason,
                   text_path
            FROM document
            WHERE date(published_at) >= ?
              AND date(published_at) <= ?
              AND ingest_status = 'done'
              AND triage_class IS NOT NULL
            ORDER BY id
            """,
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
    if len(rows) < sample_size:
        raise ValueError(
            f"Need at least {sample_size} completed triage results; found {len(rows)}"
        )

    selected = random.Random(seed).sample(rows, sample_size)
    selected.sort(key=lambda row: (row["published_at"], row["ticker"], row["id"]))
    output_path = config.project_root / "var" / "wave1" / "triage-review.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "id",
                "ticker",
                "published_at",
                "headline",
                "category",
                "price_sensitive",
                "triage_class",
                "form_type",
                "triage_reason",
                "text_excerpt",
            ),
        )
        writer.writeheader()
        for row in selected:
            excerpt = ""
            if row["text_path"] is not None:
                excerpt = Path(row["text_path"]).read_text(
                    encoding="utf-8-sig"
                )[:1000]
            writer.writerow(
                {
                    "id": row["id"],
                    "ticker": row["ticker"],
                    "published_at": row["published_at"],
                    "headline": row["headline"],
                    "category": row["category"],
                    "price_sensitive": row["price_sensitive"],
                    "triage_class": row["triage_class"],
                    "form_type": row["form_type"],
                    "triage_reason": row["triage_reason"],
                    "text_excerpt": excerpt,
                }
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.toml.")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_path = write_triage_review_sample(
        load_config(args.config),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Triage review sample: {output_path}")


if __name__ == "__main__":
    main()
