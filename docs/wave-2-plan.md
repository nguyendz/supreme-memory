# Wave 2 — Extraction, Baseline RAG, and Evaluation: Detailed Plan

This is the execution checklist for Wave 2 of the [MVP plan](./mvp-plan.md). Wave 2 extracts source-grounded candidate facts, validates their evidence, and measures a chunk-RAG baseline. It does **not** resolve entities, reconcile facts, or write rows to `fact` or `fact_source`; those are Wave 3 responsibilities.

## Outcome and gate

At the end of Wave 2:

- Every supported `full` or `form` document in the dev slice has a validated candidate-fact result or an explicit error/unsupported-form outcome.
- Raw model output is cached by `(doc_id, model, prompt_version)` and can be revalidated without another model call.
- Candidate facts have self-contained text, a tier, and an evidence quote that passes deterministic source checking.
- The four supported form families have schema-specific prompts and tests.
- A local model comparison report records extraction quality and latency for ten fixed documents.
- A local browser UI compares cached model outputs and preserves user review decisions separately.
- `am ask --baseline --as-of ...` retrieves only chunks published on or before `knowledge_date` and returns citations.
- `am eval --baseline` produces a baseline report for all eligible gold questions.
- The five-quarterly T1 review file is ready for user review.

**Gate 2:** At least 98% of extraction responses are valid after one retry; at least 90% of non-T0 candidate facts pass quote checking; the user finds T1 coverage/tier labels acceptable on five quarterlies; the default local model is selected; and baseline evaluation numbers are recorded with citation/leakage checks passing.

## Implementation status

The Wave 2 code paths and tests are implemented. The user accepted the Wave 1 triage, boilerplate, and quarterly cross-reference reviews on 2026-11-02. Qwen extraction completed with current caches for all 108 supported documents; 16 forms are explicitly unsupported and no candidates remain unprocessed. Recovery extraction preserved full raw responses while recording 67 lower-priority facts capped from dense segments. Logged Qwen extraction/comparison calls were schema-valid on the first attempt or one retry in 664/750 cases (88.5%), below the 98% Gate 2 target; the Qwen non-T0 quote-pass rate is 88.9% (3,854/4,333), below the 90% target.

The user selected Gemma 4 as the provisional default and Qwen 3.5 as a manual alternate, then requested a full-slice Gemma refresh. Gemma extraction completed with caches for all 108 supported documents and no failures. Its full-slice non-T0 quote-pass rate is 82.4% (1,136/1,379). The Gemma T1 review has 28 candidates versus 142 in the preserved Qwen review; on the January, April, and July 2025 quarterlies Gemma produced one T1 candidate each. Keep Gemma provisional pending user review of `var/eval/t1-review.csv` and `var/eval/t1-review-qwen.csv`. The initial Qwen spot-check of the first eight January 2025 T1 rows found source-backed quotes, a rounded/precise duplicate cash balance, a stockpile fact split across two attributes, and capital-structure items worth considering for T1 versus T2.

The isolated baseline index contains 19,484/19,484 chunks at 4,096 dimensions. The Qwen baseline evaluation completed 24 in-corpus questions, deferred only `w0-09`, and recorded 12/24 judge-agreed answers (50.0%), zero failed runs, zero out-of-date citations, and zero invalid quotes; see ignored `var/eval/baseline.md`. The Gemma baseline evaluation recorded 13/24 judge-agreed answers (54.2%), one citation-validation failure, and zero out-of-date citations; see ignored `var/eval/baseline-gemma.md`. The fixed extraction-model comparison completed all 30 runs: aggregate non-T0 quote pass was 87.8% for `qwen3.5:9b`, 90.9% for `gemma4`, and 87.1% for `nimble`; see ignored `var/eval/models.md`. `config.toml` uses provisional `gemma4`, with Qwen available via `AM_LLM_MODEL=qwen3.5:9b`. Gate 2 remains open pending T1 review and quote/schema quality; the evaluation harness counts a source as in-corpus only when it has searchable chunks, not merely a catalog row.

The historical `nimble` comparison is not a like-for-like extraction result. Local Ollama metadata identifies it as a decision model with a one-choice output instruction, so its 87.1% quote-pass figure must not be used to select or evaluate a generative extraction model. Wave 3 excludes `nimble` from canonical fact promotion.

## Reviewing the model comparison

Generate and open the standalone local review UI with:

```powershell
uv run am review-models --database var/wave1/dev.db --open
```

The UI compares the cached Qwen, Gemma, and Nimble facts for each of the ten fixed documents, including quote checks, source segments, raw model JSON, usage metrics, and recovery-cap counts. Preferences, per-candidate decisions, and notes are stored in browser-local storage and can be downloaded as JSON or CSV. The command makes no model calls and does not modify extraction caches.

## Entry conditions and scope

Wave 1 code is implemented and its dev slice has been exercised. The user accepted the triage, boilerplate, and quarterly cross-reference manual reviews on 2026-11-02, closing Gate 1 for the CXO-only dev slice. The global boilerplate threshold remains provisional because the slice does not provide cross-ticker calibration data.

Use the isolated `var/wave1/dev.db` from Wave 1, not the default user database. The frozen slice is 1 January 2025 through 31 October 2026 and contains 136 documents: 71 `full`, 53 `form`, 11 text-bearing `skip`, and one missing-text document. Only `full` and supported `form` documents are extraction candidates. The current triage taxonomy has 24 forms labeled `other`; normalize supported form subtypes before routing and report unsupported forms rather than silently dropping them.

The current 25-question gold set has 24 questions whose source documents are in the dev slice. `w0-09` cites a February 2023 document outside the slice; report it as deferred for Wave 2 rather than counting it as a failed answer or silently omitting it.

Wave 0 measured a 35,594-character quarterly at 9,274 prompt tokens and 6.05 seconds end to end; this is a latency sample, not an extraction-accuracy benchmark. Wave 1's ten-document pilot took 108.92 seconds and logged 62 calls, with one transient local Ollama failure that succeeded on retry. Wave 2 should record its own call counts, token usage, retries, and elapsed time. Use only the local `qwen3.5:9b`, `gemma4`, and `nimble` models; do not use the installed cloud model or add a cloud fallback.

## Decisions and invariants

| Topic | Wave 2 default | Revisit if |
|---|---|---|
| Fact persistence | Keep candidate facts in `extraction_cache`; do not populate `fact`, `fact_source`, `entity`, or `attribute` | Wave 3 reconciliation begins |
| Candidate schema | Self-contained fact text plus entity name/type, attribute, string/numeric value, unit, qualifier, `as_of`, period, fact type, tier, salience, and source quote | The existing `fact` schema changes |
| Fact types | `metric`, `event`, `plan_guidance`, `relationship`, `reference_data`, `opinion_commentary` | Review identifies a missing type |
| Tiers | T1 core, T2 context, T3 short-lived, T0 noise; retain T0 only in the raw cache and do not promote it to a fact | User review changes the tier policy |
| PIT extraction context | Use only earlier, quote-verified cached candidates with `published_at < current published_at`; equal-time documents cannot see one another | Wave 3 introduces a validated PIT fact registry |
| Cache identity | `doc_id + model + prompt_version`; version includes prompt, schema, form mapping, and segmenter | Any of those components changes |
| Long documents | Split at page/heading boundaries before an estimated 24,000 input tokens, leaving room under the 32,768 context limit for prompt and output | Pilot token counts show a safer boundary |
| Unsupported forms | Normalize known forms to supported families; report remaining `form_type=other` rows in `var/eval/unsupported_forms.csv` | A new fixed form schema is added |
| Baseline retrieval | Separate ignored `var/eval/baseline.db` with embeddings for all chunks, including boilerplate; keep the Wave 1 `chunk_vec` restricted to non-boilerplate chunks | Storage or evaluation runtime becomes a real constraint |
| Evaluation scope | Evaluate the 24 gold questions with sources in the dev slice; explicitly report `w0-09` as out of slice | The dev corpus is expanded to include the 2023 source |

`LLMClient.chat_json` currently returns the validated Pydantic value but not the model's original response text. Extend `ChatResult` with the final successful `raw_content` so `extraction_cache.output_json` preserves the exact JSON used for replay. Keep the existing typed value, timing, token, and retry fields intact.

The current form classifier has only broad families, and some director/security forms are labeled `other` or mis-typed. Add deterministic normalization before extraction: Appendix 3Y/3Z to `director_interest`; Forms 603/604/605 to `substantial_holder`; retain Appendix 2A/3B as their own families. Keep Appendix 3H and other unsupported notices visible in the report until a schema is approved.

### Candidate fact schema

Each extracted candidate should validate against a fixed schema before it is cached or quote-checked. The entity remains a name/type pair in Wave 2; no entity IDs are assigned yet.

```json
{
  "facts": [
    {
      "entity": {"name": "Core Lithium", "type": "company"},
      "attribute": "cash_balance",
      "value": "A$200 million",
      "value_num": 200000000,
      "unit": "AUD",
      "qualifier": null,
      "as_of": "2026-09-30",
      "period": "2026-Q3-CY",
      "fact_type": "metric",
      "tier": "T1",
      "salience": 0.95,
      "text": "Core held A$200 million in cash at 30 September 2026.",
      "quote": "Cash balance of A$200 million at 30 September 2026.",
      "segment_id": "page-1"
    }
  ]
}
```

`key_hash`, `valid_from`, `valid_to`, and entity IDs are not model outputs in Wave 2. Wave 3 derives keys and resolves identity. Use a short quote (at least 20 characters where possible) that is copied from the exact source segment; do not permit paraphrase as evidence.

## Work items

### 2.1 — Candidate-fact extraction

**Implementation**
- Add `src/agent_memory/pipeline/extract.py`, a Pydantic candidate-fact schema, and `prompts/extract.md`.
- Extract from non-boilerplate text for `full` documents. Include the headline, source date, metadata hints, xref titles, and a point-in-time context built from earlier quote-verified cache entries only.
- Make facts atomic and self-contained. Normalize units/periods, separate `as_of` from `published_at`, and require a source quote and a supported tier/type.
- For large documents, segment on page markers or Markdown headings, keep stable `segment_id` values, and aggregate all segment outputs under one document/model/prompt cache key. Estimate tokens conservatively from the Wave 0 benchmark; keep each segment below 24,000 input tokens.
- Cache the final raw JSON after schema validation. A cache hit skips the LLM and feeds the same raw result to quote checking. If a segmented call fails, do not cache a partial document result.
- Do not insert facts, sources, entities, or attributes. Extraction errors leave Wave 1 `ingest_status` intact, produce an explicit command error, and remain retryable because no successful cache entry exists.

**Files:** `src/agent_memory/pipeline/extract.py`, `src/agent_memory/llm.py`, `prompts/extract.md`, `tests/test_extract.py`.

**Acceptance**
- [x] Tests cover required fields, type/tier enums, empty output, malformed JSON retry, raw-response caching, cache hits, and no-cache-on-failure.
- [x] Tests prove the PIT context excludes same-time and future source documents.
- [ ] The prompt produces one claim per fact and does not invent an entity, value, date, or quote absent from the source.
- [ ] At least 98% of extraction responses are schema-valid after one retry on the fixed dev slice.
- [x] No facts, entities, attributes, or fact sources are written.

### 2.2 — Standard form extraction

**Implementation**
- Add one prompt/schema per supported family: Appendix 2A, Appendix 3B, director-interest notices, and substantial-holder notices.
- Normalize Appendix 3Y/3Z and Forms 603/604/605 into the supported families before prompt selection. Preserve original `form_type` and source filing identity in the cached result.
- Convert form-specific fields into the same candidate-fact schema used by 2.1; do not add hand-written parsers.
- Write unsupported `form_type=other` documents to `var/eval/unsupported_forms.csv` with their ID/headline/type; do not pretend they were extracted.

**Fixed sample forms**

| Form family | Sample source documents |
|---|---|
| Appendix 2A | `20250928-2328-Application-for-quotation-of-securities-CXO-2A1624827`; `20260507-2320-Application-for-quotation-of-securities-CXO-6A1324676` |
| Appendix 3B | `20250828-0016-Proposed-issue-of-securities-CXO-2A1617338`; `20260318-2249-Proposed-issue-of-securities-CXO-2A1661229` |
| Director interest | `20250928-2334-Change-of-Director's-Interest-M-McComas-2A1624839`; `20260301-2238-Initial-Director's-Interest-M-Hine-2A1657210` |
| Substantial holder | `20251023-2126-Becoming-a-substantial-holder-2A1631214`; `20260512-0659-Change-in-substantial-holding-6A1325174`; `20260326-0440-Ceasing-to-be-a-substantial-holder-2A1662606`; `20260118-2128-Change-in-substantial-holding-2A1648680` |

**Files:** `prompts/forms/appendix_2a.md`, `prompts/forms/appendix_3b.md`, `prompts/forms/director_interest.md`, `prompts/forms/substantial_holder.md`, `src/agent_memory/pipeline/extract.py`, `tests/test_forms.py`.

**Acceptance**
- [ ] Ten fixed forms produce valid candidate facts with the expected security/holder/director values and evidence quotes.
- [x] Every supported `form_type` maps to exactly one prompt; unsupported forms are reported and not silently lost.

### 2.3 — Quote check

**Implementation**
- Add `src/agent_memory/pipeline/quotecheck.py`; run deterministically after schema validation and before a candidate can feed subsequent PIT extraction context.
- Normalize Unicode (including smart quotes and subscript digits), case, line breaks, whitespace, and punctuation while preserving numeric values and units.
- First check normalized exact substring within the candidate's source segment. For a near match, compare bounded source windows with a similarity threshold of 0.90; reject quotes shorter than 20 characters unless an exact match is unique.
- Keep the original quote unchanged for citations. Record pass/reject counts and rejected candidates/reasons in ignored `var/eval/quote-rejections.csv`; do not store rejected candidates as facts.

**Files:** `src/agent_memory/pipeline/quotecheck.py`, `tests/test_quotecheck.py`.

**Acceptance**
- [x] Tests cover Unicode/whitespace variants, exact quotes, bounded fuzzy matches, short/ambiguous quotes, wrong-source quotes, and invented text.
- [ ] At least 90% of non-T0 candidates pass quote checking; rejected examples are reviewed for hallucination, OCR noise, and poor segmentation. Current measured pass rate: 88.9%.

### 2.4 — Local model comparison

**Implementation**
- Run extraction on ten fixed documents: five `full` announcements (the September 2026 quarterly, May 2025 Restart Study, March 2026 FID, August 2026 BP33 drilling results, and April 2026 fines sale) plus five fixed forms from the 2.2 sample.
- Compare `qwen3.5:9b`, `gemma4`, and `nimble` using local Ollama only. Cache each output under its own model/prompt key.
- Report JSON validity, candidate facts/document, T1/T2/T3 distribution, quote-pass rate, retries, prompt/output tokens, and wall time. Compare facts and evidence side-by-side; do not choose a model on latency alone.

The five fixed `full` inputs are `20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068`, `20250513-2303-Restart-Study-Repositions-Finniss-Operations-2A1596413`, `20260317-2305-FID-Approved-and-Funding-Secured-for-Finniss-Restart-2A1661043`, `20260810-2236-New-High-Grade-Lithium-Drill-Results-at-BP33-6A1338045`, and `20260429-2321-Sale-of-20,000-Tonnes-of-Lithium-Fines-6A1323092`. The five fixed forms are the first five entries in the 2.2 sample table.

**Files:** `scripts/compare_models.py`, `var/eval/models.md` (generated, git-ignored), `tests/test_compare_models.py`.

**Acceptance**
- [x] The report contains all 30 model/document runs and links each result to the same fixed source document.
- [x] The user confirms the default model; no cloud model is used.

### 2.5 — Naive chunk-RAG baseline

**Implementation**
- Add `src/agent_memory/query/baseline_rag.py` and an `am ask --baseline --as-of YYYY-MM-DD` command.
- Enforce `document.published_at <= knowledge_date` for every retrieved chunk; use the ticker filter before ranking.
- Wave 1's production `chunk_vec` intentionally omits boilerplate. Build `var/eval/baseline.db` as a disposable copy and embed **all** chunks there, including boilerplate, so the baseline really has no boilerplate filter and does not contaminate the production vector index.
- Extend the search helpers with explicit `include_boilerplate`, ticker, and `knowledge_date` filters. Vector search must over-fetch and then filter relational metadata so future or wrong-ticker results cannot displace eligible matches. Use FTS and vector top-10 results, deduplicate chunk IDs, and combine rankings with reciprocal-rank fusion before packing a bounded context.
- Add `prompts/answer.md` and a typed response with an answer plus citations as `{document_id, quote}`. Answer from retrieved context only; every quote must be copied from a retrieved chunk. Do not use cached candidate facts for baseline answers.

**Files:** `src/agent_memory/query/baseline_rag.py`, `src/agent_memory/pipeline/chunks.py`, `src/agent_memory/cli.py`, `prompts/answer.md`, `tests/test_baseline_rag.py`.

**Acceptance**
- [x] Tests prove no chunk published after `knowledge_date` enters the answer context, including vector over-fetch/filtering cases.
- [x] Citations contain an existing document ID and a quote present in the retrieved source chunk.
- [x] Baseline vector search includes boilerplate in its disposable index; memory-mode chunk search remains boilerplate-filtered.
- [x] `am ask --baseline --as-of ...` answers an in-slice question with valid citations.

### 2.6 — Evaluation harness

**Implementation**
- Add YAML parsing as an evaluation/dev dependency and update `pyproject.toml`/`uv.lock`.
- Add `eval/run_eval.py` and `am eval --baseline`. Load questions, run baseline answers, then use the local judge to score expected-answer agreement.
- Explicitly report questions whose source documents are outside the selected corpus. For the current dev slice, evaluate 24 of 25 questions; defer `w0-09` (its source is from February 2023) until the full-history run.
- Measure answer accuracy, citation-date validity, quote validity, context tokens, and per-category results. Write `var/eval/baseline.md` with failed cases and skipped questions; never omit cases silently.

**Files:** `eval/run_eval.py`, `src/agent_memory/cli.py`, `pyproject.toml`, `uv.lock`, `tests/test_eval.py`, `var/eval/baseline.md` (generated, git-ignored).

**Implementation note:** Catalog-only source rows do not count as available evidence. The current dev corpus has 24 eligible gold questions; `w0-09` is deferred because its 2023 source has no searchable chunks in the selected slice.

**Acceptance**
- [x] The harness loads all 25 YAML entries and reports the 24 in-slice/one out-of-slice split.
- [x] Citation leakage is zero: every cited document is at or before `knowledge_date`.
- [x] Every cited quote is present in its source chunk and the answer context.
- [x] A baseline report is produced with answer-accuracy, citation, quote, token, and category metrics.

### 2.7 — Human review of T1 candidates

**Implementation**
- Export candidate facts for these five quarterlies to `var/eval/t1-review.csv`: `20250122-2145-Quarterly-Activities-and-Cashflow-Report-2A1574556`, `20250428-2310-Quarterly-Activities-and-Cashflow-Report-2A1593142`, `20250729-2314-Quarterly-Activities-and-Cashflow-Report-2A1610591`, `20251012-2205-Quarterly-Activities-and-Cashflow-Report-2A1628714`, and `20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068`.
- Include the candidate text, tier, entity/attribute, as-of/period, quote, and source segment. Review completeness and whether T1/T2/T3 labels fit the stated policy.
- Feed user corrections into `prompts/extract.md` before closing Gate 2; do not hand-edit model outputs in the cache.

**Acceptance**
- [x] The review file has all T1 candidates from the five fixed quarterlies.
- [ ] User review records missing/incorrect facts and tier corrections.

## Execution order

1. Confirm Wave 1 Gate 1 manual reviews before running extraction.
2. Audit and normalize the `form_type=other` rows for supported forms; create the ten-form fixture set.
3. Extend `ChatResult` with raw successful response text and test the cache boundary.
4. Implement extraction schema, prompt, context snapshot, segmenter, and cache replay.
5. Implement quote checking and verify it never searches outside the source segment.
6. Implement form adapters and run all ten fixed forms.
7. Run the 30 model/document comparison; record quality and latency before selecting a default.
8. In parallel, build the all-chunk baseline database and `am ask --baseline`.
9. Run `am eval --baseline` on the 24 in-slice gold questions and produce the report.
10. Export the five-quarterly T1 review file and wait for user review.
11. Close Gate 2 only after JSON, quote, baseline, model, and user-review criteria pass.

## Implemented command interfaces

- Build the isolated all-chunk baseline after Wave 1 ingestion: `uv run am baseline-index --source-db var/wave1/dev.db`.
- Ask a baseline question: `uv run am ask --baseline --as-of 2026-10-15 --ticker CXO "What was CXO's cash balance?"`.
- Evaluate the gold set: `uv run am eval --baseline`.
- After the manual Gate 1 reviews are accepted, run extraction with `uv run am extract --database var/wave1/dev.db --confirm-wave1-gate`.
- After Gate 1, compare extraction models with `uv run python scripts/compare_models.py --database var/wave1/dev.db --confirm-wave1-gate`.
- Export available T1 candidates for review with `uv run am t1-review --database var/wave1/dev.db`.
- To use the manual Qwen alternate for a PowerShell command, set `$env:AM_LLM_MODEL = 'qwen3.5:9b'` first. There is no automatic model failover.

Wave 2 commands reject non-loopback Ollama hosts and unsupported/cloud model names. Extraction commands require explicit Gate 1 confirmation and never write canonical `fact`, `fact_source`, `entity`, or `attribute` rows.

## Risks to resolve during implementation

| Risk | Evidence to collect | Response |
|---|---|---|
| Raw model response is lost before caching | `ChatResult` currently returns only the parsed value | Add `raw_content` and regression tests before extraction |
| Model uses future facts in extraction context | Same-date/future sentinel tests and source-date inspection | Build context only from earlier cache entries and quote-verified facts |
| Long reports exceed the local context | Pilot prompt token counts vs. `num_ctx=32768` | Split earlier at page/heading boundaries and keep a conservative 24k input budget |
| Unsupported forms are misrouted | Current 24 dev-slice `form_type=other` rows | Normalize known 3Y/3Z and 603/604/605 filings; report unsupported types |
| Baseline is not actually naive | Production `chunk_vec` contains only non-boilerplate embeddings | Keep an isolated all-chunk baseline index, including boilerplate |
| Search leaks post-cutoff chunks | PIT unit tests plus leakage sweep | Apply the date filter after vector over-fetch and before context packing |
| Quote checker rejects valid OCR text or accepts paraphrases | Labeled exact/variant/false-quote fixture and rejection review | Normalize conservatively; keep a strict threshold and expose rejections |
| One old gold source is not in the dev corpus | `w0-09` points to a 2023 source | Report it as deferred; include it in full-history evaluation later |
