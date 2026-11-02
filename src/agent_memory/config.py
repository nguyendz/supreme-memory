"""Typed configuration loading for the local application."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
_LOCAL_CHAT_MODELS = frozenset(
    {
        "qwen3.5:9b",
        "gemma4",
        "gemma4:latest",
        "nimble",
        "nimble:latest",
    }
)
_LOCAL_EMBEDDING_MODELS = frozenset(
    {"qwen3-embedding", "qwen3-embedding:latest"}
)


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    project_root: Path
    data_dir: Path
    database_path: Path
    ollama_host: str
    ollama_model: str
    embedding_model: str
    embedding_dimensions: int | None
    num_ctx: int
    num_predict: int
    keep_alive: str
    timeout_seconds: float
    skip_headline_patterns: tuple[str, ...]


def validate_wave2_local_runtime(
    config: AppConfig,
    *,
    require_chat_model: bool = True,
    require_embedding_model: bool = True,
) -> None:
    """Reject non-loopback hosts and model names outside the Wave 2 local set."""
    host = urlsplit(config.ollama_host)
    if host.scheme not in {"http", "https"} or host.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Wave 2 requires an Ollama host on the local machine")
    if require_chat_model and config.ollama_model not in _LOCAL_CHAT_MODELS:
        raise ValueError(
            f"Wave 2 chat model must be local and supported: {config.ollama_model}"
        )
    if (
        require_embedding_model
        and config.embedding_model not in _LOCAL_EMBEDDING_MODELS
    ):
        raise ValueError(
            "Wave 2 embedding model must be local and supported: "
            f"{config.embedding_model}"
        )


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section [{name}] must be a table")
    return value


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config.toml, resolving relative paths from its containing directory."""
    path = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as config_file:
        data = tomllib.load(config_file)

    paths = _section(data, "paths")
    ollama = _section(data, "ollama")
    ingest = _section(data, "ingest")
    base = path.parent

    patterns = ingest.get("skip_headline_patterns", [])
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        raise ValueError("[ingest].skip_headline_patterns must be an array of strings")

    dimensions = ollama.get("embedding_dimensions")
    if dimensions is not None and (
        not isinstance(dimensions, int) or dimensions <= 0
    ):
        raise ValueError("[ollama].embedding_dimensions must be a positive integer")

    return AppConfig(
        config_path=path,
        project_root=base,
        data_dir=_resolve_path(str(paths.get("data_dir", "data")), base=base),
        database_path=_resolve_path(
            str(paths.get("database", "var/memory.db")), base=base
        ),
        ollama_host=os.environ.get(
            "OLLAMA_HOST", str(ollama.get("host", "http://localhost:11434"))
        ),
        ollama_model=os.environ.get(
            "AM_LLM_MODEL", str(ollama.get("model", "qwen3.5:9b"))
        ),
        embedding_model=os.environ.get(
            "AM_EMBEDDING_MODEL",
            str(ollama.get("embedding_model", "qwen3-embedding")),
        ),
        embedding_dimensions=dimensions,
        num_ctx=int(ollama.get("num_ctx", 32768)),
        num_predict=int(ollama.get("num_predict", 1024)),
        keep_alive=str(ollama.get("keep_alive", "30m")),
        timeout_seconds=float(ollama.get("timeout_seconds", 300)),
        skip_headline_patterns=tuple(patterns),
    )
