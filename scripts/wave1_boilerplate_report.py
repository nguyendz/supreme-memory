"""Report preliminary boilerplate recall and high-value false-positive candidates."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path

from agent_memory.config import AppConfig, load_config
from agent_memory.db import database_connection
from agent_memory.loader import discover_announcements, read_announcement_text
from agent_memory.pipeline.boilerplate import (
    BoilerplateIndex,
    classify_paragraphs,
    is_material_fact_paragraph,
    split_paragraphs,
)


_LABEL_PATTERNS = {
    "about_core": re.compile(r"^\s*about core(?: lithium)?\b", re.IGNORECASE),
    "contact": re.compile(
        r"\b(?:investor|media)\s+(?:enquiries|inquiries)\b|"
        r"\bfor further information\b|\bcontact details?\b|"
        r"\binfo@[\w.-]+\.\w+",
        re.IGNORECASE,
    ),
    "competent_person": re.compile(r"\bcompetent persons?\b", re.IGNORECASE),
    "legal_disclaimer": re.compile(
        r"\bforward[- ]looking statements?\b|\bimportant information\b",
        re.IGNORECASE,
    ),
}
_HIGH_VALUE_PATTERN = re.compile(
    r"\b(?:cash|balance|revenue|sales|resource|reserve|grade|ore|production|"
    r"throughput|capacity|cost|capex|funding|price|drilling|lithium|"
    r"spodumene|shares|mineral|li2o|sc[456]|assay|intercept)\b",
    re.IGNORECASE,
)


def build_report(
    config: AppConfig,
    *,
    start_date: date = date(2025, 1, 1),
    end_date: date = date(2026, 10, 31),
) -> tuple[Path, Path]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    with database_connection(config.database_path) as connection:
        index = BoilerplateIndex.from_connection(connection)

    counts = {name: [0, 0] for name in _LABEL_PATTERNS}
    review_rows: list[dict[str, str]] = []
    announcements = discover_announcements(config.data_dir, ticker="CXO")
    for announcement in announcements:
        if not start_date <= announcement.published_at.date() <= end_date:
            continue
        if announcement.text_path is None:
            continue
        paragraphs = split_paragraphs(read_announcement_text(announcement))
        classified = classify_paragraphs(
            paragraphs,
            ticker=announcement.ticker,
            published_at=announcement.published_at,
            index=index,
        )
        for paragraph in classified:
            material = is_material_fact_paragraph(paragraph.text)
            labels = [
                name
                for name, pattern in _LABEL_PATTERNS.items()
                if pattern.search(paragraph.text)
                and not (name == "competent_person" and material)
            ]
            for label in labels:
                counts[label][0] += 1
                counts[label][1] += int(paragraph.is_boilerplate)
            potential_false_positive = (
                paragraph.is_boilerplate
                and _HIGH_VALUE_PATTERN.search(paragraph.text) is not None
                and re.search(r"\d", paragraph.text) is not None
            )
            if labels and not paragraph.is_boilerplate or potential_false_positive:
                review_rows.append(
                    {
                        "doc_id": announcement.id,
                        "labels": ",".join(labels),
                        "is_boilerplate": str(paragraph.is_boilerplate).lower(),
                        "material_value_guard": str(material).lower(),
                        "text": paragraph.text[:2000],
                    }
                )

    output_dir = config.project_root / "var" / "wave1"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "boilerplate-review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as review_file:
        writer = csv.DictWriter(
            review_file,
            fieldnames=(
                "doc_id",
                "labels",
                "is_boilerplate",
                "material_value_guard",
                "text",
            ),
        )
        writer.writeheader()
        writer.writerows(review_rows)

    report_path = output_dir / "boilerplate-report.md"
    lines = [
        "# Wave 1 Boilerplate Report",
        "",
        f"- Date range: {start_date.isoformat()} through {end_date.isoformat()} (inclusive, UTC)",
        "- This is a preliminary pattern-labeled recall estimate, not the manual gate review.",
        "- High-value numeric paragraphs classified as boilerplate and label misses are listed in `boilerplate-review.csv`.",
        "",
        "| Label | Matches | Flagged | Recall |",
        "|---|---:|---:|---:|",
    ]
    for label, (total, flagged) in counts.items():
        recall = "n/a" if total == 0 else f"{flagged / total:.1%}"
        lines.append(f"| {label} | {total} | {flagged} | {recall} |")
    lines.extend(["", f"- Review rows: {len(review_rows)}", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, review_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.toml.")
    args = parser.parse_args()
    report_path, review_path = build_report(load_config(args.config))
    print(f"Boilerplate report: {report_path}")
    print(f"Review candidates: {review_path}")


if __name__ == "__main__":
    main()
