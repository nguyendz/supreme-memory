from pathlib import Path

import pytest

from agent_memory.config import AppConfig, validate_wave2_local_runtime


def _config(
    tmp_path: Path,
    *,
    host="http://localhost:11434",
    chat_model="qwen3.5:9b",
    embedding_model="qwen3-embedding",
) -> AppConfig:
    return AppConfig(
        config_path=tmp_path / "config.toml",
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "memory.db",
        ollama_host=host,
        ollama_model=chat_model,
        embedding_model=embedding_model,
        embedding_dimensions=None,
        num_ctx=32768,
        num_predict=1024,
        keep_alive="30m",
        timeout_seconds=30,
        skip_headline_patterns=(),
    )


def test_wave2_local_runtime_accepts_only_local_host_and_model_set(tmp_path):
    validate_wave2_local_runtime(_config(tmp_path))
    validate_wave2_local_runtime(
        _config(
            tmp_path,
            chat_model="gemma4:latest",
            embedding_model="qwen3-embedding:latest",
        )
    )


@pytest.mark.parametrize(
    "config",
    [
        lambda path: _config(path, host="https://ollama.example.com"),
        lambda path: _config(path, chat_model="deepseek-v4.1-flash:cloud"),
        lambda path: _config(path, embedding_model="remote-embedding"),
    ],
)
def test_wave2_local_runtime_rejects_remote_endpoints_and_models(tmp_path, config):
    with pytest.raises(ValueError, match="local"):
        validate_wave2_local_runtime(config(tmp_path))
