from datetime import datetime
from pathlib import Path

from agent_memory.loader import Announcement
from agent_memory.pipeline.xref import (
    extract_references,
    normalize_title,
    resolve_references,
)


def _announcement(doc_id, headline, published_at):
    return Announcement(
        id=doc_id,
        ticker="CXO",
        company="Core Lithium Ltd",
        headline=headline,
        published_at=datetime.fromisoformat(published_at),
        category=None,
        announcement_types=(),
        is_price_sensitive=None,
        folder=Path(doc_id),
        metadata_path=None,
        text_path=None,
        text_source="missing",
    )


def test_extract_references_supports_curly_quotes_and_line_breaks():
    references = extract_references(
        "Refer to ASX announcement “Updated Finniss\n"
        "Resource Estimate” dated 14 May 2025."
    )
    assert len(references) == 1
    assert references[0].cited_title == "Updated Finniss Resource Estimate"
    assert references[0].cited_date.isoformat() == "2025-05-14"


def test_resolve_exact_and_fuzzy_titles_within_cited_date_window():
    referring = _announcement("source", "Quarterly report", "2025-06-01T00:00:00+00:00")
    exact = _announcement(
        "exact", "Updated Finniss Resource Estimate", "2025-05-14T01:00:00+00:00"
    )
    fuzzy = _announcement(
        "fuzzy", "BP33 Underground Resource Update", "2025-05-15T01:00:00+00:00"
    )
    resolution = resolve_references(
        referring,
        (
            'Refer to ASX announcement "Updated Finniss Resource Estimate" '
            "on 14 May 2025. "
            'Refer to ASX announcement "BP33 Underground Resource Updat" '
            "dated 15 May 2025."
        ),
        [referring, exact, fuzzy],
    )

    assert [match.target_doc_id for match in resolution.matches] == ["exact", "fuzzy"]
    assert resolution.unresolved == ()
    assert resolution.matches[0].match_score == 1.0
    assert resolution.matches[1].match_score >= 0.90


def test_resolve_reference_title_with_descriptive_suffix():
    referring = _announcement("source", "Quarterly report", "2024-11-07T00:00:00+00:00")
    target = _announcement(
        "target",
        "New High-grade Lithium Drill Results within 20km of Grants",
        "2024-11-05T22:04:00+00:00",
    )
    resolution = resolve_references(
        referring,
        (
            'Refer to ASX announcement "New high-grade Lithium Drill Results '
            'within 20km of the Grants processing facility" on 6 November 2024.'
        ),
        [referring, target],
    )
    assert [match.target_doc_id for match in resolution.matches] == ["target"]


def test_resolve_title_with_shared_terms_and_different_wording():
    referring = _announcement("source", "Quarterly report", "2026-01-21T00:00:00+00:00")
    target = _announcement(
        "target",
        "Core and Tesla execute Binding Term Sheet for Lithium supply",
        "2022-03-01T22:23:00+00:00",
    )
    resolution = resolve_references(
        referring,
        (
            'Refer to ASX announcement "Core Lithium and Tesla enter into binding '
            'Term Sheet for the supply of lithium" on 2 March 2022.'
        ),
        [referring, target],
    )
    assert [match.target_doc_id for match in resolution.matches] == ["target"]


def test_references_reject_future_and_out_of_window_candidates():
    referring = _announcement("source", "Quarterly report", "2025-06-01T00:00:00+00:00")
    future = _announcement(
        "future", "Finniss Restart Study", "2025-06-02T00:00:00+00:00"
    )
    old = _announcement(
        "old", "Finniss Restart Study", "2025-05-01T00:00:00+00:00"
    )
    resolution = resolve_references(
        referring,
        'Refer to ASX announcement "Finniss Restart Study" on 2 June 2025.',
        [referring, future, old],
    )
    assert resolution.matches == ()
    assert len(resolution.unresolved) == 1


def test_ambiguous_references_are_not_guessed():
    referring = _announcement("source", "Quarterly report", "2025-06-01T00:00:00+00:00")
    candidates = [
        _announcement("candidate-1", "Finniss Restart Study", "2025-05-13T00:00:00+00:00"),
        _announcement("candidate-2", "Finniss Restart Study", "2025-05-13T01:00:00+00:00"),
    ]
    resolution = resolve_references(
        referring,
        'Refer to ASX announcement "Finniss Restart Study" dated 14 May 2025.',
        [referring, *candidates],
    )
    assert resolution.matches == ()
    assert len(resolution.unresolved) == 1


def test_title_normalization_ignores_punctuation_and_case():
    assert normalize_title("BP33: Underground Resource (Update)") == (
        "bp33 underground resource update"
    )
