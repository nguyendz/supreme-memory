# Wave 4 — Memory-Backed Query

This is the execution checklist for Wave 4 of the [MVP plan](./mvp-plan.md).
Wave 4 makes Wave 3 facts usable through point-in-time queries and a
citation-safe answer path. It does not rerun extraction or promote the full
dev-slice corpus.

The [pre-Wave-4 readiness plan](./pre-wave-4-gap-plan.md) covers pilot safety.
The [Wave 4.0 pilot plan](./wave-4-pilot-plan.md) specifies the first
answer-level query.

## Outcome and gate

At the end of Wave 4:

- The user can ask a natural-language question with a ticker and explicit
  `knowledge_date`.
- The query is routed to a supported memory operation (state, timeline, event,
  or summary), with a chunk search fallback only when the memory route has no
  suitable fact.
- Every answer claim is grounded in the facts or chunks supplied to the answer
  model. Citations are validated against source records and display the
  announcement ID, headline, publication date, and exact quote.
- Historical queries respect the inclusive point-in-time boundary; future
  sources cannot be retrieved or cited.
- The user can inspect answers and their evidence without reviewing raw
  extraction-cache rows.

**Gate 4:** At least 10 gold questions pass end to end with valid citations,
correct `knowledge_date` filtering, and no future-source leaks. The first
one-query pilot is a smoke test only; it does not pass Gate 4 or close Wave 2
Gate 2.

## Current status and entry conditions

- Wave 3 code and synthetic tests are implemented. The full dev-slice source
  DB `var/wave1/dev.db` remains schema v3 with zero entity, attribute, fact,
  and fact-source rows; it must remain unchanged.
- `var/wave3/pilot.db` is an isolated full copy of the Wave 1 source DB, not a
  pruned one-document database: it contains all 1,316 documents, 19,484 chunks,
  and 236 extraction-cache rows in the 1,316-document target corpus. Only the
  September 2026 quarterly was reconciled, producing 13 `ADD` facts, zero
  quote rejects, and zero unresolved decisions. Thus canonical document
  coverage is 1/1,316 (0.08%), even though raw-corpus coverage in the copy is
  100%. The cash-balance fact is PIT-visible on `2026-10-15`; it is not visible
  on `2026-10-12`.
- The pilot fact's stored attribute is `cash_balance_as_of`, illustrating that
  query routing must use the actual active attribute choices rather than
  assuming every model emits the exact string `cash_balance`.
- Wave 4 query code is implemented for state, timeline, event/search, and
  summary intents, with point-in-time fact retrieval, lexical/vector search,
  optional chunk fallback, citation validation, and source inspection commands.
  The loopback-only `am ui` interface shows answers alongside their source
  quotes; synthetic tests cover these routes and the `am show` commands.
- The real `w0-01` query was rerun on the isolated pilot DB after schema
  migration 6. It answered A$200 million and displayed the quarterly's source
  quote and path. The `get_state` tool call was recorded with one result and
  selector metadata; the raw question is not stored in that log.
- At the user's explicit acceptance of current quality for query testing,
  `var/wave3/dev.db` was promoted from the current Gemma caches in a separate
  copy. It contains 841 facts sourced from 100 documents, with 251 pending
  reconciliation reviews; the source DB remains untouched. The UI now queries
  this multi-document copy. A Finniss shipment-timing query returned four
  citations from multiple documents; this exploratory result is not a Gate 4
  pass.
- `am ask` currently defaults to memory mode; `--baseline` explicitly selects
  baseline RAG, and `--as-of` is required. This working pilot path is not a
  claim of full-corpus query quality or Gate 4 completion.
- Wave 2 Gate 2's numeric thresholds remain unmet. The user accepted the
  measured metrics for the isolated query experiment, not as a Gate 2 pass or
  production sign-off. Gemma and Qwen are explicitly allowed generation
  models; `nimble` is decision-only and is not eligible.
- `eval/questions.yaml` currently contains 25 gold questions. The 10-question
  Gate 4 run and expansion to 40 questions remain pending.
- Wave 4 should not ask the user to inspect T1 CSVs or internal action rows.
  User feedback is collected from the displayed answer and source evidence.

## Decisions and invariants

| Topic | Wave 4 default | Revisit if |
|---|---|---|
| Knowledge time | Require an inclusive `knowledge_date` for every memory request | A future UI adds a clearly documented default |
| Pilot database | Use `var/wave3/pilot.db` for the one-query smoke test; keep the source DB and primary DB unchanged | Gate 2 closes and a broader approved memory DB is built |
| Query model | Use an approved local generation model; select it explicitly for the pilot | Model evaluation selects a different approved default |
| Query plan | Return a typed intent and select only entity/attribute IDs supplied from the database | General query coverage needs another typed intent |
| Citation identity | The answer model returns turn-local source IDs; application code resolves them to source rows | Another citation protocol is proven equally safe |
| Quote rendering | Materialize the stored `fact_source.quote` or source-chunk quote after validating the source ID; do not trust generated quote text | A source API requires a different stable locator |
| Abstention | If no verified evidence is available, say so; do not silently fall back to baseline RAG | Product requirements specify an explicit fallback mode |
| Tier policy | T1–T2 by default; T3 only for a matching request; never retrieve T0 | Evaluation shows a justified change |
| User validation | Review answer usefulness and visible evidence, not raw candidate rows | A later, purpose-built data-review UI is requested |

## Work items

### 4.0 — One-query pilot

**Purpose:** finish the first user-facing vertical slice on the isolated
September 2026 quarterly, using gold question `w0-01`:

> What was CXO's cash balance at 30 September 2026?

**Implementation**

- Follow the isolated pilot procedure in
  [wave-4-pilot-plan.md](./wave-4-pilot-plan.md).
- Add the one supported query-planning intent (`state`) and a memory-backed
  `am ask --memory` branch.
- Use `state_at` and `fact_sources_at` against the pilot DB, with ticker CXO,
  subject date/period for the fact, and `knowledge_date=2026-10-15`.
- Show the answer, source quote, headline, publication date, document ID, and
  source path. Do not expose the T1 review spreadsheet as the acceptance UI.

**Acceptance**

- The answer is A$200 million and cites the September quarterly.
- The source and quote are present in the retrieved pilot evidence.
- The citation is visible by `2026-10-15` and not by `2026-10-12`.
- The user can open the source and judge the answer/evidence.

The pilot does not close Gate 2 or Gate 4.

### 4.1 — Intent planning and fact retrieval

**Implementation**

- Add a typed query-plan response. Start with `state`; add `timeline`, `event`,
  and `summary` only after the pilot works.
- Give the planner the ticker, `knowledge_date`, allowed entity/attribute
  choices, and question. Validate every selected ID and attribute against
  those choices.
- Implement `query/retrieve.py` helpers for:
  - state: current facts for an entity/attribute as known at T;
  - timeline: visible fact history ordered by subject time and publication;
  - event/fact search: ticker- and knowledge-date-filtered fact search;
  - summary: source facts selected for a date range or quarter.
- Keep `fact_source.published_at <= knowledge_date` as the knowledge boundary.
  A backdated `as_of` does not make a later-published fact visible earlier.
- Default to T1–T2. Include T3 only when the question asks for short-lived
  detail. Exclude T0.
- Deduplicate packed context by canonical fact ID/key; preserve each source
  link and its exact quote.
- Add hybrid `fact_vec`/lexical retrieval for intents that do not map to a
  direct state or timeline lookup. Any chunk fallback must be explicit in the
  trace and use the same point-in-time filter.

**Acceptance**

- Tests cover each intent, entity/attribute validation, no-match results,
  multiple periods, superseded facts, duplicate packing, and future-source
  exclusion.
- Packed context for 10 gold questions has no duplicate facts.

### 4.2 — Citation-safe answer loop

**Implementation**

- Use a separate typed memory-answer schema; retain the existing baseline
  answer path and its prompt behavior.
- Give the model retrieved fact text and turn-local citation IDs such as
  `c1`, `c2`. The model returns `answer` plus citation IDs, not free-form
  quotes.
- Validate returned IDs against the supplied set. Allow at most one correction
  attempt for invalid IDs; then fail explicitly.
- Resolve each ID to `fact_source`/document rows and render the stored quote,
  headline, publication date, and source path. Recheck publication time and
  quote validity before displaying the answer.
- Log the query plan, retrieved IDs, answer model, citations, retry, and
  abstention status without logging secrets or silently falling back to
  baseline mode.
- Keep the model/tool loop bounded. It may call the Wave 3 query helpers:
  `get_state`, `get_timeline`, `search_facts`, `get_summary`, `get_sources`,
  and `open_document`.

**Acceptance**

- A model cannot cite an ID it was not given or change a stored quote.
- An invalid citation, source published after T, or missing source causes a
  correction attempt or explicit failure/abstention, never a success-shaped
  response.
- Tests cover answer citation validation and the no-evidence abstention.

### 4.3 — Query-time summaries

**Implementation**

- Add `prompts/summary.md` and summarize only facts visible at the requested
  `knowledge_date`.
- For a quarter summary, include the requested reporting period and cite the
  source facts behind material bullets.
- Do not cache summaries in this wave; generate them from retrieved facts.

**Acceptance**

- “Summarise the September 2026 quarter” returns about 10 material bullets,
  omits boilerplate, and supplies valid citations.
- A later document cannot change an earlier `knowledge_date` summary.

### 4.4 — CLI and inspection commands

**Implementation**

- Add `am ui`, a loopback-only browser interface with ticker, knowledge date,
  question, and optional source-passage fallback controls. Display the answer,
  announcement metadata, exact citation quote, and source path.
- `am ask` currently defaults to memory mode; keep `--baseline` as an explicit
  comparison mode. The default does not imply that Gate 4 has passed.
- Require `--as-of` for memory mode and require or unambiguously infer ticker.
- Add `am show doc`, `fact`, `entity`, and `timeline` outputs sufficient to
  inspect a citation and its history.
- For the pilot, permit `--database var/wave3/pilot.db`; never silently use the
  pilot DB for normal queries.

**Acceptance**

- `am ask --memory ...` and `am ask --baseline ...` select distinct explicit
  paths; one mode never falls back to the other silently.
- `am ui --database var/wave3/pilot.db` can run the one-query pilot in a local
  browser, and displays the same validated answer/evidence as `am ask`.
- A user can follow every citation to a source document and inspect its
  underlying quote.

### 4.5 — Gold-set coverage

**Implementation**

- Grow `eval/questions.yaml` toward 40 questions covering all MVP categories:
  current state, point-in-time, history, event lookup, summary, trivial-on-
  request, and future-leak traps.
- Store expected answers, `knowledge_date`, source IDs, and evidence. Keep
  source facts and human-answer review separate.

**Acceptance**

- At least 40 unique questions have valid source IDs and evidence before the
  formal full Gate 4 run.
- A single pilot question does not count as the 10-question Gate 4 minimum.

## Test sequence and remaining gates

1. **Done:** unit-test the typed planner, retrieval filters, answer schema,
   citation materialization, vector retrieval, telemetry, browser UI, and
   inspection commands with synthetic databases.
2. **Done:** rerun `w0-01` on `var/wave3/pilot.db`; verify its A$200 million
   answer, source quote, and recorded memory tool call.
3. **Pending user review:** judge the displayed answer and source evidence; no
   raw extraction rows are needed.
4. **Pending:** run at least 10 gold questions end to end, including PIT and
   future-source checks.
5. **Pending:** grow the 25-question gold set to at least 40 and assess
   summary quality across an approved corpus.
6. Run Gate 4 only on an approved corpus with no leaked citations.

## Gates and deferred work

- **Pilot checkpoint:** one answer and source pass on the disposable DB. This
  is the first user value and does not close any full-corpus gate.
- **Gate 2:** remains the full-slice extraction-quality/model gate.
- **Gate 3:** remains the canonical reconciliation/PIT/replay mechanics gate.
- **Gate 4:** remains 10+ gold questions with valid citations and zero leakage.
- **Wave 5:** automates the broader evaluation, compares with baseline, adds
  leakage/duplicate/action metrics, and tunes from measured failures.

Do not run bulk reconciliation or claim production-quality query accuracy
until Gate 2 and Gate 3 are closed. No Wave 4 pilot task asks the user to
inspect raw T1 rows.
