"""Local LLM triage for announcement documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from agent_memory.loader import Announcement
from agent_memory.llm import LLMClient


TRIAGE_EXCERPT_CHARACTERS = 3000


class TriageResult(BaseModel):
    triage_class: Literal["skip", "form", "full"] = Field(alias="class")
    form_type: (
        Literal[
            "appendix_2a",
            "appendix_3b",
            "director_interest",
            "substantial_holder",
            "other",
        ]
        | None
    ) = Field(...)
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def require_nonempty_reason(cls, value: str) -> str:
        reason = value.strip()
        if not reason:
            raise ValueError("reason must not be blank")
        return reason

    @model_validator(mode="after")
    def validate_form_type(self) -> TriageResult:
        if self.triage_class == "form" and not self.form_type:
            raise ValueError("form_type is required when class is form")
        if self.triage_class != "form" and self.form_type is not None:
            raise ValueError("form_type must be null unless class is form")
        return self


def _compile_skip_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid skip_headline_patterns regex {pattern!r}: {exc}") from exc
    return tuple(compiled)


def triage_announcement(
    announcement: Announcement,
    text: str,
    client: LLMClient,
    *,
    skip_headline_patterns: Sequence[str] = (),
    prompt_path: Path,
) -> TriageResult:
    """Classify a document, bypassing Ollama for explicit headline skips."""
    if not text.strip():
        raise ValueError(f"Announcement has no readable text: {announcement.id}")

    for pattern, compiled in zip(
        skip_headline_patterns, _compile_skip_patterns(skip_headline_patterns)
    ):
        if compiled.search(announcement.headline):
            return TriageResult.model_validate(
                {
                    "class": "skip",
                    "form_type": None,
                    "reason": f"Matched configured headline skip pattern: {pattern}",
                }
            )

    system_prompt = prompt_path.read_text(encoding="utf-8")
    excerpt = text[:TRIAGE_EXCERPT_CHARACTERS]
    prompt = (
        "Classify this ASX announcement using the supplied schema and rules.\n\n"
        f"Ticker: {announcement.ticker}\n"
        f"Headline: {announcement.headline}\n"
        f"Category: {announcement.category or 'unknown'}\n"
        "Announcement types: "
        f"{', '.join(announcement.announcement_types) or 'unknown'}\n"
        "Price sensitive: "
        f"{announcement.is_price_sensitive if announcement.is_price_sensitive is not None else 'unknown'}\n\n"
        "Opening text excerpt:\n"
        f"{excerpt}"
    )
    result = client.chat_json(
        prompt,
        TriageResult,
        system_prompt=system_prompt,
        doc_id=announcement.id,
    )
    return result.value
