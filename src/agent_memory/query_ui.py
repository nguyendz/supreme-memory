"""Local browser interface for point-in-time memory queries."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import ollama

from agent_memory.config import AppConfig
from agent_memory.llm import LLMClient
from agent_memory.query.agent import (
    MemoryCitationValidationError,
    MemoryQueryResult,
    answer_memory,
)
from agent_memory.query.retrieve import ChunkEvidence, MemorySource


_MAX_REQUEST_BYTES = 16_384
_MAX_QUESTION_LENGTH = 4_000
_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Announcement memory</title>
<style>
:root { color-scheme: light; font-family: Segoe UI, sans-serif; }
body { margin: 0; background: #f4f6f8; color: #17212b; }
header { padding: 1.15rem 1.5rem; background: #17324d; color: white; }
header h1 { margin: 0 0 .35rem; font-size: 1.4rem; }
header p { margin: 0; color: #dbe8f3; }
main { max-width: 1000px; margin: auto; padding: 1.25rem; }
.intro { margin: .25rem 0 1rem; }
.intro h2 { margin: 0 0 .35rem; font-size: 1.2rem; }
.intro p, .muted { color: #536574; }
.card { background: white; border: 1px solid #d5dde4; border-radius: 6px; }
form { padding: 1rem; }
.fields { display: grid; grid-template-columns: minmax(150px, .7fr) minmax(190px, 1fr); gap: .9rem; }
label { display: block; margin-bottom: .35rem; font-weight: 600; }
input, textarea, button { font: inherit; }
input, textarea { box-sizing: border-box; width: 100%; padding: .65rem; border: 1px solid #aebbc5; border-radius: 4px; color: inherit; }
textarea { min-height: 7rem; margin-top: .9rem; resize: vertical; }
.checkbox-row { display: flex; align-items: flex-start; gap: .55rem; margin: .85rem 0; }
.checkbox-row input { width: auto; margin: .2rem 0 0; }
.checkbox-row label { margin: 0; font-weight: 400; }
button { padding: .6rem .9rem; border: 1px solid #8494a2; border-radius: 4px; background: white; cursor: pointer; }
button.primary { border-color: #176b87; background: #176b87; color: white; font-weight: 600; }
button:disabled { cursor: wait; opacity: .65; }
#status { min-height: 1.4rem; margin: .7rem 0 0; color: #536574; }
#results { margin-top: 1rem; }
.answer-card { padding: 1rem; border-left: 4px solid #176b87; }
.answer-card.warning { border-left-color: #b66a12; }
.answer-card h2 { margin: 0 0 .5rem; font-size: 1.05rem; }
.answer { margin: 0; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.meta { margin: 0 0 .55rem; color: #536574; font-size: .9rem; }
.badge { display: inline-block; margin: 0 0 .5rem; padding: .18rem .55rem; border-radius: 10px; background: #edf1f4; font-size: .8rem; }
.citations { margin-top: 1rem; }
.citations h2 { font-size: 1.05rem; }
.citation { margin-top: .7rem; padding: .9rem 1rem; }
.citation h3 { margin: 0 0 .35rem; font-size: .98rem; }
.quote { margin: .65rem 0; padding: .7rem; border-left: 3px solid #97a8b4; background: #f7f9fa; white-space: pre-wrap; overflow-wrap: anywhere; }
.source-path { margin: .5rem 0 0; color: #536574; font-size: .88rem; overflow-wrap: anywhere; }
.error { padding: .9rem 1rem; border-left: 4px solid #9a2c20; color: #7d241b; }
.debug { margin-top: 1rem; padding: 1rem; }
.debug summary { cursor: pointer; font-weight: 600; }
.debug pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 32rem; overflow: auto; }
@media (max-width: 650px) {
  main { padding: .85rem; }
  .fields { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Announcement memory</h1>
  <p>Wave 4 · Point-in-time answers with source evidence</p>
</header>
<main>
  <section class="intro">
    <h2>Ask a question</h2>
    <p>Answers use information available by the inclusive knowledge date. Review the cited announcement and exact source quote.</p>
  </section>
  <form id="query-form" class="card">
    <div class="fields">
      <div>
        <label for="ticker">Ticker</label>
        <input id="ticker" name="ticker" value="CXO" maxlength="20" required>
      </div>
      <div>
        <label for="as-of">Knowledge date (inclusive)</label>
        <input id="as-of" name="as_of" type="date" value="__TODAY__" required>
      </div>
    </div>
    <label for="question" style="margin-top:.9rem">Question</label>
    <textarea id="question" name="question" maxlength="4000" placeholder="What was CXO's cash balance at 30 September 2026?" required></textarea>
    <div class="checkbox-row">
      <input id="chunk-fallback" name="chunk_fallback" type="checkbox">
      <label for="chunk-fallback">If no verified fact is found, also search source passages.</label>
    </div>
    <button id="submit" class="primary" type="submit">Ask memory</button>
    <p id="status" role="status" aria-live="polite"></p>
  </form>
  <section id="results" aria-live="polite"></section>
</main>
<script>
(() => {
  const form = document.getElementById("query-form");
  const button = document.getElementById("submit");
  const status = document.getElementById("status");
  const results = document.getElementById("results");

  function makeElement(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderDebug(debug) {
    if (!debug || !Object.keys(debug).length) return;
    const panel = makeElement("details", "card debug");
    panel.append(makeElement("summary", "", "Query debug details"));
    panel.append(makeElement(
      "p", "muted",
      "Diagnostic model output is unvalidated, not an accepted answer. Details include the question, prompts and retrieved evidence."
    ));
    panel.append(makeElement("pre", "", JSON.stringify(debug, null, 2)));
    results.append(panel);
  }

  function renderError(message, debug) {
    const error = makeElement("p", "card error", message);
    results.replaceChildren(error);
    renderDebug(debug);
  }

  function renderResult(data) {
    const answerCard = makeElement(
      "article",
      data.status === "no_evidence" ? "card answer-card warning" : "card answer-card"
    );
    answerCard.append(
      makeElement("p", "meta", `${data.ticker} · As known on ${data.knowledge_date} · ${data.intent}`)
    );
    if (data.used_chunk_fallback) {
      answerCard.append(makeElement("span", "badge", "Source-passage fallback used"));
    }
    answerCard.append(makeElement("p", "answer", data.answer));
    const content = [answerCard];

    if (data.citations.length) {
      const citationSection = makeElement("section", "citations");
      citationSection.append(makeElement("h2", "", "Sources"));
      for (const citation of data.citations) {
        const card = makeElement("article", "card citation");
        card.append(makeElement("h3", "", `${citation.citation_id} · ${citation.headline}`));
        card.append(makeElement(
          "p",
          "meta",
          `${citation.published_at.slice(0, 10)} · ${citation.document_id}`
        ));
        card.append(makeElement("blockquote", "quote", citation.quote));
        if (citation.source_path) {
          card.append(makeElement("p", "source-path", `Source file: ${citation.source_path}`));
        }
        citationSection.append(card);
      }
      content.push(citationSection);
    }
    results.replaceChildren(...content);
    results.append(makeElement("p", "muted", `Query ID: ${data.query_id}`));
    renderDebug(data.debug);
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    button.disabled = true;
    status.textContent = "Searching point-in-time memory…";
    results.replaceChildren();
    const payload = {
      ticker: form.elements.ticker.value,
      as_of: form.elements.as_of.value,
      question: form.elements.question.value,
      chunk_fallback: form.elements.chunk_fallback.checked
    };
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        renderError(data.error || `Request failed (${response.status})`, data.debug);
        status.textContent = "Query failed.";
        return;
      }
      renderResult(data);
      status.textContent = data.status === "no_evidence"
        ? "No verified evidence was available for that knowledge date."
        : "Answer ready.";
    } catch (error) {
      renderError(error.message || "The query failed.");
      status.textContent = "Query failed.";
    } finally {
      button.disabled = false;
    }
  });
})();
</script>
</body>
</html>
"""


def _validate_payload(payload: Any) -> tuple[str, str, date, bool]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    allowed_fields = {"question", "ticker", "as_of", "chunk_fallback"}
    if set(payload) - allowed_fields:
        raise ValueError("Request contains unsupported fields")

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be blank")
    if len(question) > _MAX_QUESTION_LENGTH:
        raise ValueError(f"Question must be at most {_MAX_QUESTION_LENGTH} characters")

    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip() or len(ticker) > 20:
        raise ValueError("Ticker must contain between 1 and 20 characters")

    as_of = payload.get("as_of")
    if not isinstance(as_of, str):
        raise ValueError("Knowledge date is required")
    try:
        knowledge_date = date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError("Knowledge date must use YYYY-MM-DD") from exc
    if knowledge_date.isoformat() != as_of:
        raise ValueError("Knowledge date must use YYYY-MM-DD")

    chunk_fallback = payload.get("chunk_fallback", False)
    if not isinstance(chunk_fallback, bool):
        raise ValueError("Chunk fallback must be true or false")
    return question.strip(), ticker.strip().upper(), knowledge_date, chunk_fallback


def _citation_payload(
    citation: MemorySource | ChunkEvidence,
) -> dict[str, str | None]:
    payload: dict[str, str | None] = {
        "citation_id": citation.citation_id,
        "document_id": citation.document_id,
        "headline": citation.headline,
        "published_at": citation.published_at,
        "quote": citation.quote,
        "source_path": citation.source_path,
    }
    if isinstance(citation, MemorySource):
        payload["fact_id"] = citation.fact_id
    else:
        payload["chunk_id"] = citation.chunk_id
    return payload


def _result_payload(result: MemoryQueryResult) -> dict[str, object]:
    return {
        "query_id": result.query_id,
        "ticker": result.ticker,
        "knowledge_date": result.knowledge_date.isoformat(),
        "intent": result.intent,
        "answer": result.answer,
        "status": "answered" if result.citations else "no_evidence",
        "used_chunk_fallback": result.used_chunk_fallback,
        "citations": [_citation_payload(item) for item in result.citations],
        "debug": result.debug,
    }


def create_query_server(
    settings: AppConfig,
    database_path: Path,
    *,
    model: str,
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create the loopback-only Wave 4 query UI server."""

    class QueryRequestHandler(BaseHTTPRequestHandler):
        server_version = "AnnouncementMemoryUI/1.0"

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
            )
            super().end_headers()

        def _expected_host(self) -> str:
            return f"127.0.0.1:{self.server.server_port}"

        def _is_local_request(self) -> bool:
            return (
                self.client_address[0] in {"127.0.0.1", "::1"}
                and self.headers.get("Host") == self._expected_host()
            )

        def _send_bytes(
            self,
            status: int,
            body: bytes,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def do_GET(self) -> None:
            if not self._is_local_request():
                self._send_error_json(403, "Use the local UI address.")
                return
            if self.path == "/":
                body = _HTML_PAGE.replace(
                    "__TODAY__",
                    date.today().isoformat(),
                ).encode("utf-8")
                self._send_bytes(200, body, "text/html; charset=utf-8")
                return
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_error_json(404, "Not found.")

        def do_POST(self) -> None:
            if not self._is_local_request():
                self._send_error_json(403, "Use the local UI address.")
                return
            expected_origin = f"http://{self._expected_host()}"
            origin = self.headers.get("Origin")
            if origin is not None and origin != expected_origin:
                self._send_error_json(403, "Cross-origin requests are not allowed.")
                return
            if self.path != "/api/ask":
                self._send_error_json(404, "Not found.")
                return
            if self.headers.get_content_type() != "application/json":
                self._send_error_json(415, "Content-Type must be application/json.")
                return

            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_error_json(400, "A valid Content-Length is required.")
                return
            if content_length <= 0:
                self._send_error_json(400, "Request body must not be empty.")
                return
            if content_length > _MAX_REQUEST_BYTES:
                self._send_error_json(413, "Request body is too large.")
                return

            try:
                raw_body = self.rfile.read(content_length)
                if len(raw_body) != content_length:
                    raise ValueError("Request body was incomplete")
                payload = json.loads(raw_body.decode("utf-8"))
                question, ticker, knowledge_date, chunk_fallback = (
                    _validate_payload(payload)
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send_error_json(400, str(exc))
                return

            try:
                client = LLMClient(settings, database_path)
                result = answer_memory(
                    database_path,
                    client,
                    question,
                    knowledge_date=knowledge_date,
                    ticker=ticker,
                    model=model,
                    chunk_fallback=chunk_fallback,
                )
            except MemoryCitationValidationError as exc:
                self._send_json(422, {"error": str(exc), "debug": exc.debug})
                return
            except ValueError as exc:
                self._send_error_json(422, str(exc))
                return
            except (httpx.HTTPError, ollama.ResponseError) as exc:
                self._send_error_json(502, str(exc))
                return
            except (OSError, sqlite3.Error) as exc:
                self._send_error_json(500, str(exc))
                return

            self._send_json(200, _result_payload(result))

    server = ThreadingHTTPServer(("127.0.0.1", port), QueryRequestHandler)
    server.daemon_threads = True
    return server
