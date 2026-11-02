# Wave 1 — Ingest Skeleton: Detailed Plan

This is the execution checklist for Wave 1 of the [MVP plan](./mvp-plan.md). Wave 1 turns the Wave 0 source inventory into a reproducible document pipeline. It classifies announcements, learns boilerplate from prior documents, stores searchable chunks and vectors, resolves references to earlier announcements, and supports safe retries and re-ingestion. It does **not** extract facts or write entities, attributes, or facts.

## Outcome and gate

At the end of Wave 1:

- `am ingest` processes a fixed, date-bounded CXO slice in deterministic order.
- Every selected folder is represented in `document`, including the known missing-text document.
- Text-bearing announcements have a validated `skip`, `form`, or `full` triage result.
- Boilerplate counts are point-in-time safe, auditable, and reversible on re-ingestion.
- Non-boilerplate passages are searchable with FTS5 and sqlite-vec.
- Cross-references resolve only to earlier announcements.
- Re-running a completed ingest is a no-op; re-ingestion rebuilds the same logical memory state.
- No facts or extraction-cache rows are created.

**Gate 1:** The fixed dev slice ingests end to end; a sample of 30 triage results selected with seed `2026` has no obvious classification errors; ticker-specific labeled boilerplate recall is at least 90%; at least 80% of audited references in quarterly reports resolve; the five fixed, in-slice seed queries `w0-01`, `w0-04`, `w0-07`, `w0-14`, and `w1-25` return sensible non-boilerplate FTS and vector results; and re-ingestion produces the same logical memory tables. Global-threshold behavior is unit-tested with multiple synthetic tickers but is not claimed as empirically calibrated by the CXO-only dev slice.

**Progress (2026-11-02):** Implementation is complete for the Wave 1 code and 25-question seed. The dev slice has 135 documents done and one known missing-text document skipped, with no outstanding errors after retry. It contains 19,484 chunks and 13,204 vectors at 4,096 dimensions, with no facts or extraction-cache rows. The no-op run made zero LLM calls; all five retrieval checks passed; the current resolver matches 45/45 parsed references in the 13 quarterlies. The preliminary pattern-labeled report estimates 100.0% About Core, 94.2% contact, 100.0% non-material Competent Person, and 97.5% legal-disclaimer recall. Gate 1 remains open for user triage review, manual boilerplate review, and manual reference audit; the global threshold remains provisional until cross-ticker data exists.

## Wave 0 baseline and dev slice

The MVP defines the dev slice as 1 January 2025 through 31 October 2026, inclusive by UTC `published_at` date. The current archive contains 136 CXO announcements in that window (latest date 22 October): 135 have markdown and one has missing text. Use the MVP's date range consistently; the full CXO history can be processed after the dev-slice gate passes.

Wave 0 provides `discover_announcements`, `read_announcement_text`, typed configuration including `skip_headline_patterns`, `LLMClient`, database connection helpers, `upsert_document`, FTS5 triggers, and vector tables. There is not yet a `pipeline` package or an `am ingest` command. Run tests with `uv run python -m pytest` on this Windows environment; the bare `uv run pytest` launcher currently fails with a uv trampoline error.

## Decisions and constraints

| Topic | Wave 1 default | Revisit if |
|---|---|---|
| Dev slice | `--from 2025-01-01 --to 2026-10-31`, inclusive by UTC `published_at` date | A later wave defines a different frozen evaluation slice |
| Triage classes | `skip`, `form`, `full`; metadata is a hint, not a hard classifier | Review shows the labels do not route documents usefully |
| Config skip patterns | Treat configured patterns as explicit hard skips, record the matched pattern and reason, and make no LLM call | The user wants a skip pattern to be advisory; that requires an LLM call and is not a cost guard |
| Boilerplate signature | Deterministic 64-bit SimHash; lowercase and collapse whitespace, but preserve numbers initially | A labeled sample shows date/number variation prevents matching without suppressing meaningful values |
| Ticker threshold | Start at 3 earlier documents, as proposed in the design brainstorm | False-positive review requires a higher value |
| Global threshold | Provisional `K=20` prior documents and at least two distinct tickers; synthetic tests only in this CXO-only wave | Cross-ticker data becomes available for empirical calibration |
| Chunking | Paragraph-first, deterministic ordering; split only oversized paragraphs at sentence boundaries | Five-query retrieval tests show poor passage boundaries |
| Embeddings | Existing local `qwen3-embedding`; initialize vector tables from measured response width | Retrieval testing gives a measured reason to change models |
| Form documents | Retain non-boilerplate chunks for baseline retrieval; Wave 2 may additionally apply fixed form schemas | Evaluation shows form chunks are not useful |
| Missing markdown | Keep the document row, set a clear skipped status/reason, and make no LLM call | Source repair changes the archive |

### Point-in-time counter history is a prerequisite

Schema version 1 stores only an aggregate `doc_count` in `paragraph_hash`. That aggregate cannot reconstruct the count as of an earlier document or roll back contributions from a re-ingest. Add a version 2 migration with a per-document hash-observation table keyed by document, scope, ticker, and SimHash. Keep `paragraph_hash` as the aggregate/summary table, but make boilerplate decisions from observations whose document `published_at` is **strictly earlier** than the current document. Equal-timestamp documents must not count as earlier than one another.

This history table supports both warm-up and re-ingest: warm-up scans all ticker history up to `--to` without exposing future counts to earlier documents, and re-ingest can delete observations for the selected date range and rebuild aggregates. Do not use the final aggregate `doc_count` to classify earlier documents. The global scope requires observations from multiple tickers; CXO-only counts must never qualify as cross-company boilerplate.

Schema version 3 adds `document.form_type`, so a validated form classification survives ingestion and can route to the correct Wave 2 schema.

## Work items

### 1.1 — Triage

**Implementation**
- Add a typed triage response with `class: skip | form | full`, optional `form_type`, and a non-empty reason.
- Persist `form_type` as well as `triage_class` and `triage_reason`; Wave 2 must not need to call the model again to recover the subtype.
- Add `prompts/triage.md`. Give the model the headline, category, announcement types, price-sensitivity metadata, and a bounded opening excerpt (start with 3,000 characters).
- Use `LLMClient.chat_json` and the configured local model. Do not add a cloud fallback or silently turn model errors into `skip`.
- Apply configured `skip_headline_patterns` before the LLM call and persist the reason. Keep `form_type` required for `form` results and null otherwise.
- Preserve the document in `document` for all classes. `skip` means no chunks or embeddings; `form` remains available to baseline retrieval.

**Files:** `src/agent_memory/pipeline/triage.py`, `prompts/triage.md`, `src/agent_memory/config.py`, `src/agent_memory/db.py`, `tests/test_triage.py`, `tests/test_db.py`, `scripts/wave1_review.py`.

**Acceptance**
- [x] Tests cover all classes, malformed/invalid model output, missing text, metadata hints, and skip-pattern bypass with no LLM call.
- [x] Schema version 3 migrates existing databases and persists `form_type`.
- [x] `uv run python scripts/wave1_review.py` writes a 30-result sample with seed `2026` to ignored `var/wave1/triage-review.csv`.
- [ ] User review of the sample finds no obvious classification errors.

### 1.2 — Boilerplate and warm-up

**Implementation**
- Split markdown into stable paragraphs; preserve original text for chunks and evidence. Normalize only the signature input.
- Implement deterministic 64-bit SimHash with the standard library. Use a provisional Hamming distance of at most 3 for near-duplicate matches; validate on labeled examples before accepting the threshold.
- Apply explicit patterns for clearly boilerplate About Core, contact, Competent Person, and legal-disclaimer passages. Numeric financial/resource values and data tables take precedence and must remain searchable even when a signature repeats.
- Record one observation per unique document/scope/hash. Count distinct earlier documents, not repeated occurrences within one document.
- Keep ticker-specific and global scopes separate. Tune the global threshold using observed frequencies and false-positive review.
- Add `am ingest --warmup`: scan all tickers' text-bearing documents through `--to`, record hash observations only, and make no LLM, chunk, or embedding calls. Reject `--from` and `--limit` in warm-up mode so the history cannot be accidentally truncated.
- Keep recurring but meaningful passages from being discarded; test changing numeric facts and resource statements explicitly.

**Files:** `src/agent_memory/pipeline/boilerplate.py`, `src/agent_memory/db.py`, `tests/test_boilerplate.py`, `tests/test_db.py`, `scripts/wave1_boilerplate_report.py`.

**Acceptance**
- [x] Migration from schema version 1 adds observation history without changing existing rows.
- [x] Migration from schema version 2 adds `document.form_type`.
- [x] Tests cover deterministic hashes, exact/near matches, threshold counts by distinct earlier documents, scope separation, equal timestamps, multi-ticker global behavior, and warm-up idempotence.
- [x] Rebuilding counters after deleting a date range gives the same state as a clean chronological warm-up.
- [ ] On a manually labeled dev-slice sample of "About Core", contact blocks, Competent Person statements, and standard disclaimers, boilerplate recall is at least 90%. The current pattern-labeled estimate is in `var/wave1/boilerplate-report.md`.
- [x] High-value numeric paragraphs and resource statements remain available; tests cover the material-value guard, and candidate misses/false positives are in `var/wave1/boilerplate-review.csv`.

### 1.3 — Chunk store

**Implementation**
- Create stable paragraph-first chunks with deterministic `ord` values. Preserve source wording; split only oversized paragraphs at sentence boundaries.
- Store chunk metadata and `is_boilerplate` in `chunk`. Search paths must filter boilerplate rows. Embed only non-boilerplate chunks.
- Use FTS5 through the existing external-content table and triggers. Store embedding rows in `chunk_vec`; initialize vector tables from the returned embedding dimension and rely on the existing width-mismatch guard.
- Batch calls through `LLMClient.embed`; keep LLM calls outside SQLite write transactions. If embedding fails, surface the error and do not mark the document complete.
- Before deleting or replacing chunks, delete corresponding `chunk_vec` rows explicitly; sqlite-vec rows do not cascade from the relational chunk table.

**Files:** `src/agent_memory/pipeline/chunks.py`, `tests/test_chunks.py`, `scripts/wave1_search_check.py`.

**Acceptance**
- [x] Tests cover paragraph ordering, oversized paragraph splitting, boilerplate filtering, FTS insert/update/delete behavior, embedding persistence, and dimension mismatch.
- [x] FTS and vector search return a sensible, non-boilerplate passage for the five fixed in-slice questions `w0-01`, `w0-04`, `w0-07`, `w0-14`, and `w1-25`. Their source documents all fall inside the dev slice.
- [x] A document with no non-boilerplate text completes without creating empty vectors.

### 1.4 — Cross-reference resolution

**Implementation**
- Extract references matching the design pattern `Refer to ASX announcement "<title>" on|dated <date>`, allowing straight/curly quotes and whitespace/newline variation.
- Normalize titles and dates; first try an exact normalized headline and cited date. Use `difflib.SequenceMatcher` plus a bounded token-overlap fallback for descriptive title expansions, with a provisional match threshold of 0.90.
- Accept only a candidate published before the referring document and within three days of the cited date. Never resolve a reference to a later document.
- Build the candidate catalog from all discovered announcement metadata, not just the documents whose content is processed in the dev slice. Earlier target documents can predate the 2025 processing window.
- Insert successful matches into `doc_xref` with the original title/date and a reproducible score. Report unresolved and ambiguous references; do not guess between candidates.

**Files:** `src/agent_memory/pipeline/xref.py`, `tests/test_xref.py`.

**Acceptance**
- [x] Tests cover exact matches, punctuation/quote variants, date parsing, fuzzy threshold, ambiguous candidates, unresolved references, and rejection of future documents.
- [x] The current resolver matches 45/45 parsed references in the frozen dev-slice quarterlies.
- [ ] Manually audit the matched quarterly references to confirm they point to the intended earlier announcements.
- [x] Every stored `doc_xref` target is earlier than its source and satisfies the cited-date window.

### 1.5 — Orchestrator and CLI

**Implementation**
- Add `src/agent_memory/pipeline/ingest.py` and an `am ingest` Typer command with `--ticker`, inclusive `--from`, inclusive `--to`, `--limit`, `--warmup`, and `--reingest`.
- Refresh the metadata-only document catalog for the full archive before processing the selected date range, so references to earlier out-of-slice documents satisfy `doc_xref` foreign keys.
- Sort by `(ticker, published_at, id)`. Apply the limit after filtering and sorting. Treat completed documents as no-ops; retry pending/error documents explicitly.
- Persist triage class/reason and `ingest_status`. Use `done` for successfully processed documents, `skipped` with a reason for missing source text, and `error` with a useful message for failed processing. Print per-document failures and exit non-zero if any selected document fails.
- Keep network calls outside long-running SQLite write transactions. Commit each document's durable pipeline outputs atomically; on failure, roll back that document and record its error in a short separate transaction. LLM call records remain in the existing append-only `llm_call` ledger.
- Require `--from` with `--reingest`; reject `--limit`. If `--to` is provided, rebuild that inclusive range; otherwise rebuild from `--from` onward. Preserve catalog rows, explicitly remove vector rows, remove hash observations, rebuild aggregate counters, then replay in order.
- Do not write to `fact`, `fact_source`, `entity`, `attribute`, or `extraction_cache` in Wave 1.

**Files:** `src/agent_memory/pipeline/ingest.py`, `src/agent_memory/pipeline/__init__.py`, `src/agent_memory/cli.py`, `tests/test_ingest.py`.

**Acceptance**
- [x] A fresh scratch database catalogs the full archive and processes the frozen slice, including the missing-text inventory row.
- [x] `am ingest --ticker CXO --from 2025-01-01 --to 2026-10-31` completes with explicit totals for done, skipped, failed, and no-op documents.
- [x] A second normal run is a no-op and makes no additional triage or embedding calls.
- [x] In a disposable database, `--reingest --from 2025-01-01` yields the same logical `document`, `paragraph_hash`, observation, `chunk`, `chunk_fts`, `chunk_vec`, and `doc_xref` contents as a clean run. Exclude append-only `llm_call` telemetry and wall-clock fields from equality checks.
- [x] Failed documents remain visible with an error; unrelated documents continue processing.
- [x] No facts, entities, attributes, or extraction-cache rows are created.

### 1.6 — Expand the gold set

**Implementation**
- Extend [`eval/questions.yaml`](../eval/questions.yaml) from 15 to 25 entries.
- Add at least three point-in-time questions and two negative/future-leak traps. Each entry must include an expected answer, inclusive `knowledge_date`, source document IDs, and evidence excerpts.
- Keep all cited sources at or before each question's `knowledge_date`; use the archive, not model-generated knowledge, as the answer authority.

**Acceptance**
- [x] The file contains 25 unique questions with the required fields and valid source IDs.
- [x] Every source document exists, has text, and is published on or before its question's `knowledge_date`.

## Execution order

1. [x] Apply schema version 2 hash history and version 3 `form_type` migrations.
2. [x] Implement and unit-test triage and prompt behavior.
3. [x] Implement boilerplate hashing and warm-up with strict point-in-time counters.
4. [x] Implement chunking, FTS, and vector persistence with mocked embedding tests.
5. [x] Implement cross-reference parsing and validate coverage on dev-slice quarterlies.
6. [x] Connect the modules through the per-document orchestrator and CLI.
7. [x] Run focused tests and `uv run python -m pytest`.
8. [x] Run the 10-document local pilot; the initial run took 108.92 seconds with 62 logged calls and one transient local Ollama failure, which succeeded on retry.
9. [x] Use a disposable database for full-slice warm-up, ingest, no-op rerun, and re-ingest checks.
10. [x] Generate and preliminarily inspect the 30-document triage sample with seed `2026` and the boilerplate recall/false-positive report.
11. [x] Run the five in-slice retrieval queries and expand the gold set to 25 entries.
12. [ ] Close Gate 1 after user triage review, manual boilerplate review, and manual xref audit.

## Risks to resolve during implementation

| Risk | Evidence to collect | Response |
|---|---|---|
| Aggregate paragraph counts leak future repeats or cannot be rolled back | Same-timestamp and re-ingest tests | Require per-document hash observations; use strict prior-date counts |
| Numeric normalization hides changing financial or resource facts | Labeled false-positive review | Preserve numeric tokens initially; only relax after tests demonstrate safety |
| Global boilerplate threshold cannot be calibrated from CXO alone | Synthetic multi-ticker tests; later real multi-ticker corpus | Keep the threshold provisional and require at least two tickers |
| Skip patterns suppress a high-value announcement | Fixed-seed review of 30 triage results | Keep the list short, record matches, and remove false positives |
| Re-ingest leaves orphaned vector or FTS rows | Disposable-database table comparison | Explicitly delete vec rows and verify FTS triggers/cascades |
| Ollama errors leave documents looking complete | Mocked transport/validation failures and interrupted-run tests | Roll back document outputs, record the error, and return a non-zero command result |
| The Wave 0 loader's invalid-JSON error omits the source path | Malformed-metadata regression test | Fix path-specific reporting before relying on full-archive failure diagnostics |
