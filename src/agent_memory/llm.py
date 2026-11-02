"""Structured chat and embedding calls to the local Ollama server."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from agent_memory.config import AppConfig
from agent_memory.db import database_connection, record_llm_call


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ChatResult(Generic[ModelT]):
    value: ModelT
    raw_content: str
    prompt_tokens: int
    output_tokens: int
    prompt_duration_ms: float
    generation_duration_ms: float
    wall_duration_ms: float
    attempts: int


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    dimensions: int
    wall_duration_ms: float


class LLMOutputError(ValueError):
    """Raised when Ollama returns invalid structured output after its retry."""


class LLMClient:
    def __init__(
        self,
        config: AppConfig,
        database_path: Path,
        client: ollama.Client | None = None,
    ) -> None:
        self._config = config
        self._database_path = database_path
        self._client = client or ollama.Client(
            host=config.ollama_host, timeout=config.timeout_seconds
        )

    @property
    def model_name(self) -> str:
        return self._config.ollama_model

    def _record(
        self,
        *,
        step: str,
        doc_id: str | None,
        model: str,
        attempt: int,
        prompt_tokens: int | None,
        output_tokens: int | None,
        elapsed_ms: float,
        ok: bool,
        error: str | None = None,
    ) -> None:
        with database_connection(self._database_path) as connection:
            record_llm_call(
                connection,
                step=step,
                doc_id=doc_id,
                model=model,
                attempt=attempt,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                elapsed_ms=elapsed_ms,
                ok=ok,
                error=error,
            )

    @staticmethod
    def _retryable_embedding_error(error: Exception) -> bool:
        if isinstance(error, httpx.TransportError):
            return True
        if isinstance(error, ollama.ResponseError):
            message = str(error).casefold()
            return any(
                marker in message
                for marker in (
                    "insufficient buffer space",
                    "queue was full",
                    "resource temporarily unavailable",
                )
            )
        return False

    def chat_json(
        self,
        prompt: str,
        response_model: type[ModelT],
        *,
        system_prompt: str | None = None,
        doc_id: str | None = None,
        model: str | None = None,
        num_predict: int | None = None,
    ) -> ChatResult[ModelT]:
        """Request schema-constrained JSON and retry once on validation failure."""
        selected_model = model or self._config.ollama_model
        if not selected_model.strip():
            raise ValueError("model must not be blank")
        selected_num_predict = (
            self._config.num_predict if num_predict is None else num_predict
        )
        if selected_num_predict <= 0:
            raise ValueError("num_predict must be positive")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                response = self._client.chat(
                    model=selected_model,
                    messages=messages,
                    format=response_model.model_json_schema(),
                    options={
                        "temperature": 0,
                        "num_ctx": self._config.num_ctx,
                        "num_predict": selected_num_predict,
                    },
                    think=False,
                    keep_alive=self._config.keep_alive,
                )
            except (ollama.ResponseError, httpx.HTTPError) as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                self._record(
                    step="chat_json",
                    doc_id=doc_id,
                    model=selected_model,
                    attempt=attempt,
                    prompt_tokens=None,
                    output_tokens=None,
                    elapsed_ms=elapsed_ms,
                    ok=False,
                    error=str(exc),
                )
                if (
                    attempt == 1
                    and isinstance(exc, ollama.ResponseError)
                    and "token repeat limit reached" in str(exc).casefold()
                ):
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": "The previous generation was interrupted.",
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Retry once. Avoid repeated text and return "
                                    "concise JSON matching the required schema."
                                ),
                            },
                        ]
                    )
                    continue
                raise

            elapsed_ms = (time.perf_counter() - started) * 1000
            prompt_tokens = response.prompt_eval_count or 0
            output_tokens = response.eval_count or 0
            prompt_duration_ms = (response.prompt_eval_duration or 0) / 1_000_000
            generation_duration_ms = (response.eval_duration or 0) / 1_000_000
            content = response.message.content or ""
            try:
                value = response_model.model_validate_json(content)
            except ValidationError as exc:
                self._record(
                    step="chat_json",
                    doc_id=doc_id,
                    model=selected_model,
                    attempt=attempt,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    elapsed_ms=elapsed_ms,
                    ok=False,
                    error=str(exc),
                )
                if attempt == 2:
                    raise LLMOutputError(
                        f"Ollama returned invalid {response_model.__name__} "
                        "JSON after one retry"
                    ) from exc
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response did not match the required "
                                f"JSON schema: {exc}. Return corrected JSON only."
                            ),
                        },
                    ]
                )
                continue

            self._record(
                step="chat_json",
                doc_id=doc_id,
                model=selected_model,
                attempt=attempt,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                elapsed_ms=elapsed_ms,
                ok=True,
            )
            return ChatResult(
                value=value,
                raw_content=content,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_duration_ms=prompt_duration_ms,
                generation_duration_ms=generation_duration_ms,
                wall_duration_ms=elapsed_ms,
                attempts=attempt,
            )

        raise AssertionError("The structured-output retry loop must return or raise")

    def embed(
        self, texts: str | Sequence[str], *, doc_id: str | None = None
    ) -> EmbeddingResult:
        """Create embeddings using the configured local embedding model."""
        model = self._config.embedding_model
        response = None
        elapsed_ms = 0.0
        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                response = self._client.embed(
                    model=model,
                    input=texts,
                    keep_alive=self._config.keep_alive,
                    dimensions=self._config.embedding_dimensions,
                )
            except (ollama.ResponseError, httpx.HTTPError) as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                self._record(
                    step="embed",
                    doc_id=doc_id,
                    model=model,
                    attempt=attempt,
                    prompt_tokens=None,
                    output_tokens=None,
                    elapsed_ms=elapsed_ms,
                    ok=False,
                    error=str(exc),
                )
                if attempt == 1 and self._retryable_embedding_error(exc):
                    continue
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000
            break
        if response is None:
            raise AssertionError("The embedding retry loop must return or raise")

        vectors = response.embeddings
        if not vectors or not vectors[0]:
            error = "Ollama returned no embedding vectors"
            self._record(
                step="embed",
                doc_id=doc_id,
                model=model,
                attempt=attempt,
                prompt_tokens=getattr(response, "prompt_eval_count", None),
                output_tokens=None,
                elapsed_ms=elapsed_ms,
                ok=False,
                error=error,
            )
            raise ValueError(error)

        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            error = "Ollama returned vectors with inconsistent dimensions"
            self._record(
                step="embed",
                doc_id=doc_id,
                model=model,
                attempt=attempt,
                prompt_tokens=getattr(response, "prompt_eval_count", None),
                output_tokens=None,
                elapsed_ms=elapsed_ms,
                ok=False,
                error=error,
            )
            raise ValueError(error)

        self._record(
            step="embed",
            doc_id=doc_id,
            model=model,
            attempt=attempt,
            prompt_tokens=getattr(response, "prompt_eval_count", None),
            output_tokens=None,
            elapsed_ms=elapsed_ms,
            ok=True,
        )
        return EmbeddingResult(
            vectors=vectors,
            dimensions=dimensions,
            wall_duration_ms=elapsed_ms,
        )
