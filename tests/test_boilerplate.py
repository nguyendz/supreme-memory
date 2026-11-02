from datetime import datetime, timezone

import pytest

from agent_memory.pipeline.boilerplate import (
    BoilerplateIndex,
    HashObservation,
    classify_paragraphs,
    hamming_distance,
    is_material_fact_paragraph,
    observations_for_document,
    simhash64,
    split_paragraphs,
)


def _observation(
    doc_id: str,
    day: int,
    *,
    simhash: str = "0000000000000000",
    ticker: str = "CXO",
    scope: str = "ticker",
    scope_ticker: str | None = None,
) -> HashObservation:
    return HashObservation(
        scope=scope,
        ticker=scope_ticker if scope_ticker is not None else ticker,
        simhash=simhash,
        doc_id=doc_id,
        published_at=datetime(2025, 1, day, tzinfo=timezone.utc),
        source_ticker=ticker,
        sample_text="About Core",
    )


def test_simhash_is_deterministic_and_normalizes_case_and_whitespace():
    assert simhash64("About Core  Lithium") == simhash64(" about   core lithium ")
    assert simhash64("cash balance was 100 million") != simhash64(
        "cash balance was 200 million"
    )
    assert hamming_distance("0000000000000000", "0000000000000001") == 1


def test_split_paragraphs_handles_windows_line_endings():
    assert split_paragraphs("First\r\nline\r\n\r\nSecond\n\nThird") == [
        "First\nline",
        "Second",
        "Third",
    ]


def test_ticker_threshold_counts_distinct_strictly_earlier_documents():
    index = BoilerplateIndex(ticker_threshold=3, global_threshold=10)
    index.add_observations(
        [
            _observation("doc-1", 1),
            _observation("doc-1", 1),
            _observation("doc-2", 2),
        ]
    )

    assert not index.is_boilerplate(
        "0000000000000000",
        "CXO",
        datetime(2025, 1, 3, tzinfo=timezone.utc),
    )
    index.add_observations([_observation("doc-3", 3)])
    assert index.is_boilerplate(
        "0000000000000000",
        "CXO",
        datetime(2025, 1, 4, tzinfo=timezone.utc),
    )


def test_near_hashes_in_same_ticker_scope_are_counted():
    index = BoilerplateIndex(ticker_threshold=2, global_threshold=10)
    index.add_observations(
        [
            _observation("doc-1", 1, simhash="0000000000000000"),
            _observation("doc-2", 2, simhash="0000000000000001"),
        ]
    )

    assert index.is_boilerplate(
        "0000000000000002",
        "CXO",
        datetime(2025, 1, 3, tzinfo=timezone.utc),
    )


def test_global_scope_requires_multiple_tickers_and_threshold():
    index = BoilerplateIndex(
        ticker_threshold=10,
        global_threshold=3,
        global_min_tickers=2,
    )
    index.add_observations(
        [
            _observation("cxo-1", 1, scope="global", scope_ticker=""),
            _observation("cxo-2", 2, scope="global", scope_ticker=""),
            _observation("cxo-3", 3, scope="global", scope_ticker=""),
        ]
    )
    assert not index.is_boilerplate(
        "0000000000000000",
        "OTHER",
        datetime(2025, 1, 4, tzinfo=timezone.utc),
    )

    index.add_observations(
        [
            _observation(
                "other-1",
                1,
                ticker="OTHER",
                scope="global",
                scope_ticker="",
            )
        ]
    )
    assert index.is_boilerplate(
        "0000000000000000",
        "OTHER",
        datetime(2025, 1, 4, tzinfo=timezone.utc),
    )


def test_equal_timestamp_documents_are_not_prior():
    index = BoilerplateIndex(ticker_threshold=1, global_threshold=10)
    index.add_observations(
        [
            HashObservation(
                scope="ticker",
                ticker="CXO",
                simhash="0000000000000000",
                doc_id="same-time",
                published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                source_ticker="CXO",
                sample_text="About Core",
            )
        ]
    )

    assert not index.is_boilerplate(
        "0000000000000000",
        "CXO",
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def test_document_observations_deduplicate_repeated_paragraphs():
    observations = observations_for_document(
        doc_id="doc-1",
        ticker="CXO",
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        paragraphs=["About Core", "About Core"],
    )
    assert len(observations) == 2
    assert {observation.scope for observation in observations} == {"ticker", "global"}


def test_classification_is_computed_before_observing_the_current_document():
    index = BoilerplateIndex(ticker_threshold=1, global_threshold=10)
    paragraphs = ["A repeated legal paragraph.", "A repeated legal paragraph."]
    results = classify_paragraphs(
        paragraphs,
        ticker="CXO",
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        index=index,
    )
    assert [paragraph.is_boilerplate for paragraph in results] == [False, False]


def test_explicit_contact_and_competent_person_text_is_boilerplate():
    index = BoilerplateIndex()
    paragraphs = classify_paragraphs(
        [
            "Investor Enquiries: +61 8 8317 1700; info@corelithium.com.au",
            "The information relating to Mineral Resources was compiled by a Competent Person.",
            "The estimates were reviewed by the Competent Persons."
        ],
        ticker="CXO",
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        index=index,
    )
    assert [paragraph.is_boilerplate for paragraph in paragraphs] == [
        True,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Cash balance was A$200 million at quarter-end.",
        "BP33 returned 34.08 m at 2.09% Li2O.",
        "<table><tr><td>Mineral Resource</td><td>6.33</td><td>1.41</td></tr></table>",
    ],
)
def test_material_numeric_paragraphs_are_protected(text):
    assert is_material_fact_paragraph(text)
    index = BoilerplateIndex(ticker_threshold=1, global_threshold=1, global_min_tickers=1)
    index.add_observations(
        [
            _observation("doc-1", 1, simhash=simhash64(text)),
            _observation("doc-2", 2, simhash=simhash64(text)),
        ]
    )
    result = classify_paragraphs(
        [text],
        ticker="CXO",
        published_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
        index=index,
    )[0]
    assert result.is_boilerplate is False
