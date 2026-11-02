import json
from datetime import datetime, timezone

import pytest

from agent_memory.loader import discover_announcements, read_announcement_text


def _write_metadata(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loader_prefers_v2_and_uses_folder_time(tmp_path):
    folder = (
        tmp_path
        / "CXO"
        / "announcements"
        / "20261013-2139-Test-Announcement-6A1349068"
    )
    _write_metadata(
        folder / "2.0.0" / f"{folder.name}.json",
        {
            "source_metadata": {
                "ticker": "CXO",
                "company": "Core Lithium Ltd",
                "headline": "Test announcement",
                "published_at": "2026-10-13T21:39:17Z",
                "category": "quarterly_cashflow",
                "announcement_types": ["Quarterly Report"],
                "is_price_sensitive": True,
            }
        },
    )
    v2_text = folder / "2.0.0" / f"{folder.name}_v2.md"
    v2_text.write_text("preferred v2 content", encoding="utf-8")
    v1_text = folder / "1.0.0" / f"{folder.name}.md"
    v1_text.parent.mkdir(parents=True)
    v1_text.write_text("older v1 content", encoding="utf-8")

    [announcement] = discover_announcements(tmp_path)

    assert announcement.ticker == "CXO"
    assert announcement.published_at == datetime(
        2026, 10, 13, 21, 39, tzinfo=timezone.utc
    )
    assert announcement.text_source == "v2"
    assert announcement.is_price_sensitive is True
    assert read_announcement_text(announcement) == "preferred v2 content"


def test_loader_falls_back_to_v1_and_includes_missing_text(tmp_path):
    data_dir = tmp_path / "data"
    v1_folder = (
        data_dir
        / "CXO"
        / "announcements"
        / "20110210-0306-Old-Format-5A181763"
    )
    _write_metadata(
        v1_folder / "1.0.0" / f"{v1_folder.name}.json",
        {"headline": "Old format announcement", "ticker": "CXO"},
    )
    v1_text = v1_folder / "1.0.0" / f"{v1_folder.name}.md"
    v1_text.write_text("v1 content", encoding="utf-8")

    missing_folder = (
        data_dir
        / "CXO"
        / "announcements"
        / "20260714-2222-Missing-Text-6A1333881"
    )
    _write_metadata(
        missing_folder / "1.0.0" / f"{missing_folder.name}.json",
        {"headline": "Missing text"},
    )

    announcements = discover_announcements(data_dir, ticker="CXO")
    by_id = {item.id: item for item in announcements}

    assert by_id[v1_folder.name].text_source == "v1"
    assert read_announcement_text(by_id[v1_folder.name]) == "v1 content"
    assert by_id[missing_folder.name].text_source == "missing"
    with pytest.raises(FileNotFoundError, match="No markdown text"):
        read_announcement_text(by_id[missing_folder.name])


def test_loader_rejects_malformed_announcement_folder(tmp_path):
    malformed = tmp_path / "CXO" / "announcements" / "not-a-date"
    malformed.mkdir(parents=True)

    with pytest.raises(ValueError, match="YYYYMMDD-HHMM"):
        discover_announcements(tmp_path)


def test_loader_reports_invalid_metadata_json_path(tmp_path):
    folder = tmp_path / "CXO" / "announcements" / "20261013-2139-Test-Announcement-1"
    metadata_path = folder / "1.0.0" / f"{folder.name}.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid announcement metadata JSON") as error:
        discover_announcements(tmp_path)

    assert str(metadata_path) in str(error.value)
