from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_memory.loader import Announcement
from agent_memory.pipeline.triage import TriageResult, triage_announcement


class FakeLLM:
    def __init__(self, value: str):
        self.value = value
        self.calls = []

    def chat_json(self, prompt, response_model, *, system_prompt=None, doc_id=None):
        self.calls.append(
            {
                "prompt": prompt,
                "response_model": response_model,
                "system_prompt": system_prompt,
                "doc_id": doc_id,
            }
        )
        return SimpleNamespace(value=response_model.model_validate_json(self.value))


def _announcement(headline: str = "Quarterly Activities Report") -> Announcement:
    return Announcement(
        id="20261013-2139-Test-Announcement-1",
        ticker="CXO",
        company="Core Lithium Ltd",
        headline=headline,
        published_at=__import__("datetime").datetime.fromisoformat(
            "2026-10-13T21:39:00+00:00"
        ),
        category="quarterly_cashflow",
        announcement_types=("Quarterly Report",),
        is_price_sensitive=True,
        folder=Path("announcement"),
        metadata_path=None,
        text_path=Path("announcement.md"),
        text_source="v1",
    )


@pytest.mark.parametrize(
    ("payload", "expected_class", "expected_form_type"),
    [
        ('{"class":"skip","form_type":null,"reason":"Administrative notice"}', "skip", None),
        ('{"class":"form","form_type":"appendix_3b","reason":"Share issue form"}', "form", "appendix_3b"),
        ('{"class":"full","form_type":null,"reason":"Operational update"}', "full", None),
    ],
)
def test_triage_uses_typed_response_and_metadata_hints(
    tmp_path, payload, expected_class, expected_form_type
):
    prompt_path = tmp_path / "triage.md"
    prompt_path.write_text("Use the configured triage classes.", encoding="utf-8")
    client = FakeLLM(payload)

    result = triage_announcement(
        _announcement(),
        "Finniss production and funding update.",
        client,
        prompt_path=prompt_path,
    )

    assert result.triage_class == expected_class
    assert result.form_type == expected_form_type
    assert client.calls[0]["response_model"] is TriageResult
    assert client.calls[0]["doc_id"] == _announcement().id
    assert "quarterly_cashflow" in client.calls[0]["prompt"]
    assert "Quarterly Report" in client.calls[0]["prompt"]
    assert "Price sensitive: True" in client.calls[0]["prompt"]
    assert client.calls[0]["system_prompt"] == prompt_path.read_text(encoding="utf-8")


def test_triage_bypasses_llm_for_configured_skip_pattern(tmp_path):
    client = FakeLLM('{"class":"full","form_type":null,"reason":"unused"}')

    result = triage_announcement(
        _announcement("Notice of Cleansing"),
        "Administrative text.",
        client,
        skip_headline_patterns=(r"cleansing",),
        prompt_path=tmp_path / "unused.md",
    )

    assert result.triage_class == "skip"
    assert "cleansing" in result.reason
    assert client.calls == []


def test_triage_rejects_invalid_skip_pattern_and_blank_text(tmp_path):
    client = FakeLLM('{"class":"full","form_type":null,"reason":"unused"}')

    with pytest.raises(ValueError, match="Invalid skip_headline_patterns"):
        triage_announcement(
            _announcement(),
            "some text",
            client,
            skip_headline_patterns=("[",),
            prompt_path=tmp_path / "unused.md",
        )

    with pytest.raises(ValueError, match="no readable text"):
        triage_announcement(
            _announcement(),
            " \n ",
            client,
            prompt_path=tmp_path / "unused.md",
        )
    assert client.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        '{"class":"unknown","form_type":null,"reason":"No"}',
        '{"class":"form","form_type":null,"reason":"Missing type"}',
        '{"class":"full","form_type":"appendix_3b","reason":"Unexpected type"}',
        '{"class":"skip","form_type":null,"reason":"   "}',
    ],
)
def test_triage_schema_rejects_invalid_outputs(payload):
    with pytest.raises(ValidationError):
        TriageResult.model_validate_json(payload)


def test_triage_schema_requires_form_type_property():
    with pytest.raises(ValidationError):
        TriageResult.model_validate_json(
            '{"class":"form","reason":"Standardized filing"}'
        )
