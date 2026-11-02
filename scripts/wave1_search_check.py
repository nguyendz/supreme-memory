"""Check fixed in-slice FTS and vector queries against the gold set."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memory.config import AppConfig, load_config
from agent_memory.db import database_connection
from agent_memory.llm import LLMClient
from agent_memory.pipeline.chunks import search_fts, search_vectors


_QUERIES = (
    (
        "w0-01",
        "cash balance 2026 September 30 A$200 million",
        "cash balance 2026 September 30 A$200 million",
        {
            "20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068",
            "20261013-2142-September-Quarterly-Report-Presentation-6A1349070",
        },
    ),
    (
        "w0-04",
        "BP33 first development ore mid CY27",
        "BP33 first development ore mid CY27",
        {
            "20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068",
            "20261013-2142-September-Quarterly-Report-Presentation-6A1349070",
        },
    ),
    (
        "w0-07",
        "Glencore lithium fines stockpile",
        "Glencore final lithium fines stockpile",
        {"20260903-2316-Sale-of-Remaining-Lithium-Fines-Stockpile-6A1342177"},
    ),
    (
        "w0-14",
        "Meeting 14 November 2025 AWST",
        "2025 AGM date time venue Perth",
        {"20251014-2143-Notice-of-2025-AGM-and-Proxy-Form-2A1629281"},
    ),
    (
        "w1-25",
        "Grants 784kt 134kt SC5",
        "Grants mining commencement Finniss open pit May 2026",
        {"20260519-2253-Core-Commences-Mining-Operations-at-Finniss-6A1326209"},
    ),
)


def verify_searches(config: AppConfig, *, limit: int = 10) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    client = LLMClient(config, config.database_path)
    for question_id, fts_query, vector_query, expected_doc_ids in _QUERIES:
        query_embedding = client.embed([vector_query]).vectors[0]
        with database_connection(config.database_path) as connection:
            fts_ids = {
                result.doc_id
                for result in search_fts(connection, fts_query, limit=limit)
            }
            vector_ids = {
                result.doc_id
                for result in search_vectors(
                    connection, query_embedding, limit=limit
                )
            }
        if not (fts_ids & expected_doc_ids):
            raise RuntimeError(
                f"{question_id} FTS returned no expected source; got {sorted(fts_ids)}"
            )
        if not (vector_ids & expected_doc_ids):
            raise RuntimeError(
                f"{question_id} vector search returned no expected source; "
                f"got {sorted(vector_ids)}"
            )
        print(
            f"{question_id}: FTS {len(fts_ids & expected_doc_ids)} expected source(s); "
            f"vector {len(vector_ids & expected_doc_ids)} expected source(s)"
        )
    return len(_QUERIES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.toml.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    count = verify_searches(load_config(args.config), limit=args.limit)
    print(f"Verified FTS and vector retrieval for {count} in-slice gold questions.")


if __name__ == "__main__":
    main()
