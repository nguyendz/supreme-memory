"""Discovery and metadata loading for announcement source folders."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


_FOLDER_PREFIX = re.compile(r"^(?P<date>\d{8})-(?P<time>\d{4})-")
TextSource = Literal["v2", "v1", "missing"]


@dataclass(frozen=True)
class Announcement:
    id: str
    ticker: str
    company: str
    headline: str
    published_at: datetime
    category: str | None
    announcement_types: tuple[str, ...]
    is_price_sensitive: bool | None
    folder: Path
    metadata_path: Path | None
    text_path: Path | None
    text_source: TextSource


def _published_at(folder_name: str, folder: Path) -> datetime:
    match = _FOLDER_PREFIX.match(folder_name)
    if match is None:
        raise ValueError(
            f"Announcement folder does not start with YYYYMMDD-HHMM: {folder}"
        )
    try:
        timestamp = datetime.strptime(
            f"{match.group('date')}-{match.group('time')}", "%Y%m%d-%H%M"
        )
    except ValueError as exc:
        raise ValueError(f"Invalid announcement date in folder: {folder}") from exc
    return timestamp.replace(tzinfo=timezone.utc)


def _metadata_path(folder: Path) -> Path | None:
    for version in ("2.0.0", "1.0.0"):
        version_dir = folder / version
        if not version_dir.is_dir():
            continue
        candidates = sorted(
            path
            for path in version_dir.glob("*.json")
            if not path.name.endswith("_middle.json")
        )
        if candidates:
            return candidates[0]
    return None


def _text_file(folder: Path) -> tuple[Path | None, TextSource]:
    v2_dir = folder / "2.0.0"
    v2_candidates = sorted(v2_dir.glob("*_v2.md")) if v2_dir.is_dir() else []
    if v2_candidates:
        return v2_candidates[0], "v2"

    v1_dir = folder / "1.0.0"
    v1_candidates = sorted(v1_dir.glob("*.md")) if v1_dir.is_dir() else []
    if v1_candidates:
        return v1_candidates[0], "v1"
    return None, "missing"


def _read_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig") as metadata_file:
        try:
            value = json.load(metadata_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid announcement metadata JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Announcement metadata must be a JSON object: {path}")
    return value


def _field(
    metadata: dict[str, object], source_metadata: dict[str, object], name: str
) -> object | None:
    value = source_metadata.get(name)
    if value is not None:
        return value
    return metadata.get(name)


def _fallback_headline(folder_name: str) -> str:
    without_date = _FOLDER_PREFIX.sub("", folder_name, count=1)
    return re.sub(r"-[0-9A-Z]{8,}$", "", without_date).replace("-", " ")


def discover_announcements(
    data_dir: Path, ticker: str | None = None
) -> list[Announcement]:
    """Discover ticker announcement folders, preferring v2 text over v1."""
    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Announcement data directory not found: {root}")

    ticker_dirs = [root / ticker] if ticker else sorted(root.iterdir())
    announcements: list[Announcement] = []
    for ticker_dir in ticker_dirs:
        announcement_root = ticker_dir / "announcements"
        if not announcement_root.is_dir():
            if ticker and ticker_dir == root / ticker:
                raise FileNotFoundError(
                    f"Announcement directory not found for {ticker}: "
                    f"{announcement_root}"
                )
            continue

        for folder in sorted(announcement_root.iterdir()):
            if not folder.is_dir():
                continue
            timestamp = _published_at(folder.name, folder)
            metadata_path = _metadata_path(folder)
            metadata = _read_metadata(metadata_path)
            nested = metadata.get("source_metadata")
            source_metadata = nested if isinstance(nested, dict) else {}
            text_path, text_source = _text_file(folder)

            headline_value = _field(metadata, source_metadata, "headline")
            company_value = _field(metadata, source_metadata, "company")
            category_value = _field(metadata, source_metadata, "category")
            types_value = _field(metadata, source_metadata, "announcement_types")
            price_sensitive_value = _field(
                metadata, source_metadata, "is_price_sensitive"
            )
            announcement_types = (
                tuple(item for item in types_value if isinstance(item, str))
                if isinstance(types_value, list)
                else ()
            )

            announcements.append(
                Announcement(
                    id=folder.name,
                    ticker=ticker_dir.name.upper(),
                    company=(
                        company_value if isinstance(company_value, str) else ticker_dir.name
                    ),
                    headline=(
                        headline_value
                        if isinstance(headline_value, str)
                        else _fallback_headline(folder.name)
                    ),
                    published_at=timestamp,
                    category=(
                        category_value if isinstance(category_value, str) else None
                    ),
                    announcement_types=announcement_types,
                    is_price_sensitive=(
                        price_sensitive_value
                        if isinstance(price_sensitive_value, bool)
                        else None
                    ),
                    folder=folder,
                    metadata_path=metadata_path,
                    text_path=text_path,
                    text_source=text_source,
                )
            )

    announcements.sort(key=lambda item: (item.published_at, item.ticker, item.id))
    return announcements


def read_announcement_text(announcement: Announcement) -> str:
    """Read an announcement's markdown, raising if the source is missing or invalid."""
    if announcement.text_path is None:
        raise FileNotFoundError(
            f"No markdown text is available for announcement {announcement.id}"
        )
    return announcement.text_path.read_text(encoding="utf-8-sig")
