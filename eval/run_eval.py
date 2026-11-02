"""Standalone entry point for the baseline gold-set evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memory.config import load_config, validate_wave2_local_runtime
from agent_memory.evaluation import run_evaluation
from agent_memory.llm import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.toml.")
    parser.add_argument(
        "--database",
        type=Path,
        help="Baseline corpus database.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="Gold question YAML file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output Markdown report.",
    )
    args = parser.parse_args()
    settings = load_config(args.config)
    validate_wave2_local_runtime(settings)
    database_path = (
        args.database
        or settings.project_root / "var" / "eval" / "baseline.db"
    )
    question_path = args.questions or settings.project_root / "eval" / "questions.yaml"
    report_path = args.report or settings.project_root / "var" / "eval" / "baseline.md"
    if not database_path.is_file():
        raise FileNotFoundError(f"Baseline database not found: {database_path}")

    client = LLMClient(settings, database_path)
    summary = run_evaluation(
        database_path,
        client,
        question_path,
        report_path,
    )
    print(
        f"Loaded: {summary.loaded}; evaluated: {summary.evaluated}; "
        f"deferred: {summary.deferred}; failed: {summary.failed}; "
        f"correct: {summary.correct}"
    )


if __name__ == "__main__":
    main()
