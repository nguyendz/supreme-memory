"""Generate a self-contained local browser UI for extraction-model review."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memory.db import database_connection
from agent_memory.model_comparison import FIXED_DOCUMENT_IDS, LOCAL_MODELS
from agent_memory.pipeline.extract import (
    CachedExtraction,
    _source_hash,
    segment_chunks,
)
from agent_memory.pipeline.quotecheck import check_quote


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local extraction-model review</title>
<style>
:root { color-scheme: light; font-family: Segoe UI, sans-serif; }
body { margin: 0; background: #f4f6f8; color: #17212b; }
header { padding: 1.2rem 1.5rem; background: #17324d; color: white; }
header h1 { margin: 0 0 .35rem; font-size: 1.35rem; }
header p { margin: 0; color: #dbe8f3; }
main { max-width: 1500px; margin: auto; padding: 1rem; }
.toolbar, .document-meta, .review-controls { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; }
.toolbar, .document-meta { margin-bottom: 1rem; }
button, select, input, textarea { font: inherit; }
button, select { padding: .5rem .65rem; }
button { cursor: pointer; border: 1px solid #8494a2; border-radius: 4px; background: white; }
button.primary { background: #176b87; color: white; border-color: #176b87; }
.document-meta { padding: .75rem; background: white; border: 1px solid #d5dde4; border-radius: 5px; }
.model-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; align-items: start; }
.model-card { min-width: 0; background: white; border: 1px solid #d5dde4; border-radius: 5px; }
.model-card h2 { margin: 0; padding: .8rem; background: #e8eef3; font-size: 1rem; }
.stats { padding: .65rem .8rem; color: #435564; border-bottom: 1px solid #e3e8ec; font-size: .88rem; }
.fact { padding: .75rem .8rem; border-bottom: 1px solid #e3e8ec; overflow-wrap: anywhere; }
.fact:last-child { border-bottom: 0; }
.fact-title { display: flex; justify-content: space-between; gap: .5rem; align-items: center; }
.fact-title strong { font-size: .9rem; }
.fact p { margin: .55rem 0; line-height: 1.4; }
.quote { padding: .55rem; background: #f7f9fa; border-left: 3px solid #97a8b4; white-space: pre-wrap; font-size: .86rem; }
.passed { color: #12653b; font-weight: 600; }
.rejected { color: #9a2c20; font-weight: 600; }
.badge { border-radius: 9px; padding: .15rem .5rem; background: #edf1f4; font-size: .78rem; white-space: nowrap; }
.review-controls { margin-top: .6rem; font-size: .82rem; }
.review-controls select { padding: .25rem; }
textarea { display: block; width: 100%; min-height: 3.2rem; box-sizing: border-box; margin-top: .45rem; }
details { margin-top: .5rem; }
details pre { max-height: 18rem; overflow: auto; padding: .6rem; background: #f4f6f8; white-space: pre-wrap; overflow-wrap: anywhere; font-size: .78rem; }
.notes { width: min(460px, 100%); min-height: 2.4rem; }
.muted { color: #667784; }
.warning { color: #8a4b00; }
#progress { margin-left: auto; color: #dbe8f3; }
@media (max-width: 950px) { .model-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Local extraction-model review</h1>
  <p>Compare fixed-document facts and evidence. Review state stays in this browser until exported.</p>
</header>
<main>
  <div class="toolbar">
    <button id="previous">Previous</button>
    <label for="document-select">Document</label>
    <select id="document-select"></select>
    <button id="next">Next</button>
    <span id="progress"></span>
    <button id="export-json" class="primary">Download review JSON</button>
    <button id="export-csv">Download decision CSV</button>
  </div>
  <section id="document-meta" class="document-meta"></section>
  <section id="review-controls" class="review-controls"></section>
  <section id="model-grid" class="model-grid"></section>
</main>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const storageKey = "agent-memory-model-review-v1";
  const select = document.getElementById("document-select");
  const grid = document.getElementById("model-grid");
  const meta = document.getElementById("document-meta");
  const controls = document.getElementById("review-controls");
  const progress = document.getElementById("progress");
  let index = 0;
  let storageAvailable = true;
  let review = {};

  try {
    review = JSON.parse(localStorage.getItem(storageKey) || "{}");
  } catch (_) {
    storageAvailable = false;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function documentState(docId) {
    if (!review[docId]) review[docId] = { preferred_model: "", notes: "", facts: {} };
    if (!review[docId].facts) review[docId].facts = {};
    return review[docId];
  }

  function persist() {
    if (!storageAvailable) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(review));
    } catch (_) {
      storageAvailable = false;
    }
    updateProgress();
  }

  function updateProgress() {
    const reviewed = Object.values(review).filter(
      value => value.preferred_model || value.notes || Object.keys(value.facts || {}).length
    ).length;
    progress.textContent = `${reviewed} / ${data.documents.length} documents reviewed`;
  }

  function makeDownload(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  function renderFact(doc, modelName, fact, factIndex, modelState) {
    const card = element("article", "fact");
    const title = element("div", "fact-title");
    const heading = element(
      "strong",
      "",
      `${fact.tier} / ${fact.fact_type} / ${fact.attribute}`
    );
    const badge = element(
      "span",
      `badge ${fact.quote_pass ? "passed" : "rejected"}`,
      fact.quote_pass
        ? `Quote ${fact.quote_match_type || "pass"}`
        : `Quote rejected: ${fact.quote_reason}`
    );
    title.append(heading, badge);
    card.append(title);
    card.append(element(
      "div",
      "muted",
      `${fact.entity.name} (${fact.entity.type})`
    ));
    card.append(element("p", "", fact.text));
    card.append(element("pre", "quote", fact.quote));

    const reviewKey = JSON.stringify([
      modelName,
      fact.segment_id,
      fact.attribute,
      fact.quote,
    ]);
    const existing = modelState.facts[reviewKey] || {};
    const reviewControls = element("div", "review-controls");
    const decisionLabel = element("label", "", "Candidate");
    const decision = document.createElement("select");
    for (const [value, label] of [
      ["", "Not reviewed"],
      ["keep", "Keep"],
      ["flag", "Flag"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      decision.append(option);
    }
    decision.value = existing.decision || "";
    decision.addEventListener("change", () => {
      modelState.facts[reviewKey] = {
        ...modelState.facts[reviewKey],
        model: modelName,
        segment_id: fact.segment_id,
        attribute: fact.attribute,
        text: fact.text,
        quote: fact.quote,
        decision: decision.value,
        note: note.value,
      };
      persist();
    });
    decisionLabel.append(decision);
    reviewControls.append(decisionLabel);
    const note = element("textarea", "notes");
    note.placeholder = "Optional note for this candidate";
    note.value = existing.note || "";
    note.addEventListener("input", () => {
      modelState.facts[reviewKey] = {
        ...modelState.facts[reviewKey],
        model: modelName,
        segment_id: fact.segment_id,
        attribute: fact.attribute,
        text: fact.text,
        quote: fact.quote,
        decision: decision.value,
        note: note.value,
      };
      persist();
    });
    reviewControls.append(note);
    card.append(reviewControls);

    const sourceText = (modelState.source_segments || {})[fact.segment_id];
    if (sourceText) {
      const sourceDetails = element("details");
      sourceDetails.append(element("summary", "", `Source segment ${fact.segment_id}`));
      sourceDetails.append(element("pre", "", sourceText));
      card.append(sourceDetails);
    }
    return card;
  }

  function render() {
    const doc = data.documents[index];
    const docState = documentState(doc.document_id);
    select.value = doc.document_id;
    meta.replaceChildren(
      element("strong", "", doc.headline),
      element("span", "muted", `${doc.ticker} · ${doc.published_at} · ${doc.document_id}`)
    );
    controls.replaceChildren();

    const preferredLabel = element("label", "", "Preferred model");
    const preferred = document.createElement("select");
    for (const value of ["", ...data.models]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value || "Undecided";
      preferred.append(option);
    }
    preferred.value = docState.preferred_model || "";
    preferred.addEventListener("change", () => {
      docState.preferred_model = preferred.value;
      persist();
    });
    preferredLabel.append(preferred);
    controls.append(preferredLabel);
    const note = element("textarea", "notes");
    note.placeholder = "Document-level comparison notes";
    note.value = docState.notes || "";
    note.addEventListener("input", () => {
      docState.notes = note.value;
      persist();
    });
    controls.append(note);
    controls.append(element(
      "span",
      storageAvailable ? "muted" : "warning",
      storageAvailable
        ? "Review autosaves in this browser."
        : "Browser storage is unavailable; export review JSON before closing."
    ));

    grid.replaceChildren();
    for (const modelName of data.models) {
      const result = doc.models[modelName] || {
        status: "missing",
        facts: [],
        usage: { prompt_tokens: 0, output_tokens: 0, wall_duration_ms: 0 },
        truncated_fact_count: 0,
      };
      const card = element("section", "model-card");
      card.append(element("h2", "", modelName));
      const usage = result.usage || {
        prompt_tokens: 0,
        output_tokens: 0,
        wall_duration_ms: 0,
      };
      const facts = result.facts || [];
      const passRate = typeof result.quote_pass_rate !== "number"
        ? "n/a"
        : `${(result.quote_pass_rate * 100).toFixed(1)}%`;
      card.append(element(
        "div",
        "stats",
        `${result.status} · ${facts.length} facts · ${passRate} quote pass · ` +
        `${usage.prompt_tokens || 0} prompt tokens · ${usage.output_tokens || 0} output tokens · ` +
        `${((usage.wall_duration_ms || 0) / 1000).toFixed(1)}s · ` +
        `${result.truncated_fact_count || 0} recovery-capped`
      ));
      if (result.error) card.append(element("p", "warning", result.error));
      for (let factIndex = 0; factIndex < facts.length; factIndex++) {
        card.append(renderFact(doc, modelName, facts[factIndex], factIndex, docState));
      }
      for (const raw of result.raw_segments || []) {
        const details = element("details");
        details.append(element("summary", "", `Raw model JSON · ${raw.segment_id}`));
        details.append(element("pre", "", raw.raw_content));
        card.append(details);
      }
      grid.append(card);
    }
    updateProgress();
  }

  for (const doc of data.documents) {
    const option = document.createElement("option");
    option.value = doc.document_id;
    option.textContent = `${doc.published_at.slice(0, 10)} · ${doc.headline}`;
    select.append(option);
  }
  select.addEventListener("change", () => {
    index = data.documents.findIndex(doc => doc.document_id === select.value);
    render();
  });
  document.getElementById("previous").addEventListener("click", () => {
    index = (index + data.documents.length - 1) % data.documents.length;
    render();
  });
  document.getElementById("next").addEventListener("click", () => {
    index = (index + 1) % data.documents.length;
    render();
  });
  document.getElementById("export-json").addEventListener("click", () => {
    makeDownload(
      "model-review.json",
      JSON.stringify({ generated_at: data.generated_at, review }, null, 2),
      "application/json"
    );
  });
  document.getElementById("export-csv").addEventListener("click", () => {
    const rows = [[
      "document_id",
      "preferred_model",
      "document_notes",
      "model",
      "segment_id",
      "attribute",
      "candidate_text",
      "quote",
      "decision",
      "candidate_note",
    ]];
    for (const [docId, docState] of Object.entries(review)) {
      const facts = Object.values(docState.facts || {});
      if (!facts.length) {
        if (docState.preferred_model || docState.notes) {
          rows.push([
            docId,
            docState.preferred_model || "",
            docState.notes || "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
          ]);
        }
        continue;
      }
      for (const fact of facts) {
        rows.push([
          docId,
          docState.preferred_model || "",
          docState.notes || "",
          fact.model || "",
          fact.segment_id || "",
          fact.attribute || "",
          fact.text || "",
          fact.quote || "",
          fact.decision || "",
          fact.note || "",
        ]);
      }
    }
    const csv = rows.map(row =>
      row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(",")
    ).join("\\n");
    makeDownload("model-review.csv", csv, "text/csv");
  });
  render();
})();
</script>
</body>
</html>
"""


def _review_document_data(
    connection: sqlite3.Connection,
    document_id: str,
) -> dict[str, Any]:
    document = connection.execute(
        """
        SELECT id, ticker, headline, published_at, triage_class, form_type
        FROM document WHERE id = ?
        """,
        (document_id,),
    ).fetchone()
    if document is None:
        return {
            "document_id": document_id,
            "ticker": "",
            "headline": document_id,
            "published_at": "",
            "models": {},
        }

    model_data: dict[str, Any] = {}
    for model in LOCAL_MODELS:
        row = connection.execute(
            """
            SELECT output_json, prompt_version, created_at
            FROM extraction_cache
            WHERE doc_id = ? AND model = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (document_id, model),
        ).fetchone()
        if row is None:
            model_data[model] = {
                "status": "missing",
                "facts": [],
                "source_segments": {},
                "raw_segments": [],
                "usage": {},
                "truncated_fact_count": 0,
                "quote_pass_rate": None,
            }
            continue

        cached = CachedExtraction.model_validate_json(row["output_json"])
        chunk_rows = connection.execute(
            """
            SELECT ord, text FROM chunk
            WHERE doc_id = ? AND is_boilerplate = 0
            ORDER BY ord
            """,
            (document_id,),
        ).fetchall()
        segments = segment_chunks(
            [(int(chunk["ord"]), chunk["text"]) for chunk in chunk_rows],
            max_source_characters=cached.segment_source_characters,
        )
        segment_map = {segment.segment_id: segment.text for segment in segments}
        source_hash_matches = (
            cached.source_doc_id == document_id
            and cached.source_hash == _source_hash(segments)
        )
        facts: list[dict[str, Any]] = []
        source_segments: dict[str, str] = {}
        non_t0_count = 0
        non_t0_passed = 0
        for fact in cached.facts:
            source_segment = segment_map.get(fact.segment_id, "")
            quote_check = check_quote(fact.quote, source_segment)
            if fact.tier != "T0":
                non_t0_count += 1
                non_t0_passed += int(quote_check.passed)
            if source_segment:
                source_segments[fact.segment_id] = source_segment
            facts.append(
                {
                    **fact.model_dump(mode="json"),
                    "quote_pass": quote_check.passed,
                    "quote_match_type": quote_check.match_type,
                    "quote_reason": quote_check.reason,
                }
            )
        model_data[model] = {
            "status": "extracted" if source_hash_matches else "stale-source",
            "facts": facts,
            "source_segments": source_segments,
            "raw_segments": [
                segment.model_dump(mode="json") for segment in cached.raw_segments
            ],
            "usage": cached.usage.model_dump(mode="json"),
            "truncated_fact_count": cached.truncated_fact_count,
            "quote_pass_rate": (
                non_t0_passed / non_t0_count if non_t0_count else None
            ),
            "prompt_version": row["prompt_version"],
            "created_at": row["created_at"],
            "source_hash_matches": source_hash_matches,
        }

    return {
        "document_id": document["id"],
        "ticker": document["ticker"],
        "headline": document["headline"],
        "published_at": document["published_at"],
        "triage_class": document["triage_class"],
        "form_type": document["form_type"],
        "models": model_data,
    }


def build_model_review_data(database_path: Path) -> dict[str, Any]:
    """Load comparison outputs and verified source segments for the UI."""
    with database_connection(database_path) as connection:
        documents = [
            _review_document_data(connection, document_id)
            for document_id in FIXED_DOCUMENT_IDS
        ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_database": database_path.name,
        "models": list(LOCAL_MODELS),
        "documents": documents,
    }


def _safe_script_json(value: dict[str, Any]) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def generate_model_review_ui(database_path: Path, output_path: Path) -> Path:
    """Write a standalone HTML comparison and evidence-review interface."""
    data = build_model_review_data(database_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _HTML_TEMPLATE.replace("__REVIEW_DATA__", _safe_script_json(data)),
        encoding="utf-8",
    )
    return output_path
