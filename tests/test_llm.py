from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from agent_memory.config import AppConfig
from agent_memory.db import database_connection, initialize_database
from agent_memory.llm import LLMClient


class ProbeResponse(BaseModel):
    ok: bool
    message: str


class FakeOllamaClient:
    def __init__(self):
        self.chat_responses = [
            '{"ok": "not-a-bool", "message": "bad"}',
            '{"ok": true, "message": "ready"}',
        ]
        self.chat_calls = []
        self.embed_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        content = self.chat_responses.pop(0)
        return SimpleNamespace(
            prompt_eval_count=10,
            eval_count=8,
            prompt_eval_duration=2_000_000,
            eval_duration=3_000_000,
            message=SimpleNamespace(content=content),
        )

    def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]], prompt_eval_count=4)


def _config(tmp_path: Path) -> AppConfig:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    return AppConfig(
        config_path=config_path,
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "var" / "memory.db",
        ollama_host="http://localhost:11434",
        ollama_model="qwen3.5:9b",
        embedding_model="qwen3-embedding",
        embedding_dimensions=None,
        num_ctx=32768,
        num_predict=1024,
        keep_alive="30m",
        timeout_seconds=30,
        skip_headline_patterns=(),
    )


def test_llm_client_validates_retries_embeds_and_logs(tmp_path):
    config = _config(tmp_path)
    initialize_database(config.database_path)
    with database_connection(config.database_path) as connection:
        connection.execute(
            """
            INSERT INTO document (
                id, ticker, folder, headline, published_at, text_source
            ) VALUES (
                'doc-1', 'CXO', 'doc-1', 'Example',
                '2026-01-01T00:00:00Z', 'v1'
            )
            """
        )
    fake_client = FakeOllamaClient()
    llm = LLMClient(config, config.database_path, client=fake_client)

    result = llm.chat_json("Return a ready status.", ProbeResponse, doc_id="doc-1")
    embedding = llm.embed(["Finniss lithium"], doc_id="doc-1")
    initialize_database(config.database_path, vector_dimensions=embedding.dimensions)

    assert result.value == ProbeResponse(ok=True, message="ready")
    assert result.attempts == 2
    assert fake_client.chat_calls[0]["think"] is False
    assert "properties" in fake_client.chat_calls[0]["format"]
    assert embedding.dimensions == 3
    assert len(embedding.vectors) == 1
    assert fake_client.embed_calls[0]["model"] == "qwen3-embedding"

    with database_connection(config.database_path) as connection:
        calls = connection.execute(
            "SELECT step, attempt, ok FROM llm_call ORDER BY id"
        ).fetchall()
    assert [(row["step"], row["attempt"], row["ok"]) for row in calls] == [
        ("chat_json", 1, 0),
        ("chat_json", 2, 1),
        ("embed", 1, 1),
    ]
