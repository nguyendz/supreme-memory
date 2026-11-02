"""Run the fixed local extraction-model comparison after Wave 1 review."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memory.config import load_config, validate_wave2_local_runtime
from agent_memory.llm import LLMClient
from agent_memory.model_comparison import run_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Isolated source database containing the fixed sample documents.",
    )
    parser.add_argument("--config", type=Path, help="Path to config.toml.")
    parser.add_argument(
        "--confirm-wave1-gate",
        action="store_true",
        help="Confirm triage, boilerplate, and cross-reference reviews are complete.",
    )
    args = parser.parse_args()
    if not args.confirm_wave1_gate:
        parser.error(
            "Do not run extraction comparisons until the Wave 1 Gate 1 reviews "
            "have been accepted."
        )
    settings = load_config(args.config)
    validate_wave2_local_runtime(
        settings,
        require_chat_model=False,
        require_embedding_model=False,
    )
    if not args.database.is_file():
        parser.error(f"Source database not found: {args.database}")

    report_path = settings.project_root / "var" / "eval" / "models.md"
    rejection_path = (
        settings.project_root / "var" / "eval" / "quote-rejections.csv"
    )
    client = LLMClient(settings, args.database)
    rows = run_model_comparison(
        args.database,
        client,
        report_path,
        rejection_path=rejection_path,
    )
    failures = sum(row.status in ("failed", "missing") for row in rows)
    print(
        f"Model runs: {len(rows)}; failures/missing documents: {failures}; "
        f"report: {report_path}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
