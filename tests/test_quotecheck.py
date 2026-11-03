import csv

from agent_memory.pipeline.quotecheck import (
    append_quote_rejections,
    check_quote,
    contains_exact_quote,
)


def test_quote_check_normalizes_unicode_case_punctuation_and_whitespace():
    result = check_quote(
        "LI₂O grade was 2.0% at 578.7 m",
        "Li2O grade was 2.0%\n\nat 578.7 m.",
    )

    assert result.passed
    assert result.match_type == "exact"


def test_quote_check_accepts_a_bounded_near_match():
    result = check_quote(
        "Cash balance was A$200 million at 30 September 2026",
        "Cash balance was A$200 million on 30 September 2026.",
    )

    assert result.passed
    assert result.match_type == "fuzzy"
    assert result.similarity is not None and result.similarity >= 0.90


def test_fuzzy_quote_matching_checks_bounded_windows_near_segment_end():
    quote = "Cash balance was A$200 million at 30 September 2026"
    source = ("Unrelated operating narrative without cash data. " * 120) + (
        "Cash balance was A$200 million on 30 September 2026."
    )

    result = check_quote(quote, source)

    assert result.passed
    assert result.match_type == "fuzzy"


def test_short_quotes_require_one_exact_occurrence():
    unique = check_quote("Cash", "Cash balance was A$200 million.")
    repeated = check_quote("cash", "Cash balance; later cash was spent.")
    absent = check_quote("cash", "Funds were spent.")

    assert unique.passed
    assert not repeated.passed
    assert not absent.passed


def test_quote_check_rejects_invented_and_wrong_segment_quotes():
    result = check_quote(
        "The company reached record production of 25,000 tonnes this quarter.",
        "The company reported production of 20,000 tonnes during the quarter.",
    )

    assert not result.passed
    assert not contains_exact_quote(
        "record production of 25,000 tonnes",
        "The company reported production of 20,000 tonnes.",
    )


def test_exact_citation_helper_does_not_accept_fuzzy_quotes():
    assert contains_exact_quote("Cash balance was A$200 million", "Cash balance was A$200 million.")
    assert not contains_exact_quote(
        "Cash balance was A$200 million",
        "Cash balance was A$201 million.",
    )


def test_quote_rejection_report_deduplicates_replayed_cache_results(tmp_path):
    report_path = tmp_path / "quote-rejections.csv"
    rejection = {
        "document_id": "doc-1",
        "candidate_text": "Cash balance was misstated.",
        "quote": "Invented quote that is not in this source.",
        "segment_id": "segment-0000",
        "reason": "quote is not present",
    }

    append_quote_rejections(report_path, **rejection)
    append_quote_rejections(report_path, **rejection)

    with report_path.open(encoding="utf-8", newline="") as report_file:
        rows = list(csv.DictReader(report_file))
    assert rows == [rejection]
