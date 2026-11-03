# Wave 3 — Memory: Resolution, Reconciliation, and Point-in-Time Views

This is the execution checklist for Wave 3 of the [MVP plan](./mvp-plan.md).
Wave 3 converts validated extraction-cache candidates into canonical,
source-linked facts; resolves entities and attributes; and provides
point-in-time state and timeline helpers. It does not add the Wave 4 answer
agent or change the extraction prompt.

The immediate, user-facing continuation from Waves 0–3 is tracked in the
[pre-Wave-4 gap plan](./pre-wave-4-gap-plan.md).

## Outcome and gate

At the end of Wave 3:

- Quote-verified candidates from one explicitly selected extraction model can
  be resolved to stable, ticker-local entity and attribute identities.
- New facts, confirmations, and supersessions are represented by canonical
  `fact` rows and auditable `fact_source` rows.
- Narrative duplicates are conservatively confirmed against earlier
  point-in-time facts; uncertain decisions remain visible for review.
- Point-in-time helpers return only facts supported by documents published on
  or before the requested knowledge date.
- Reconciliation can be replayed from extraction and decision caches without
  rerunning extraction, and replay produces the same logical memory state.

**Gate 3 (mechanics):** Automated tests cover the worked example's
`ADD`/`CONFIRM`/`SUPERSEDE` behavior, cash and `shares_on_issue` timelines,
point-in-time boundaries, and deterministic replay. The first user-facing
validation is answer-level in the Wave 4 pilot below; the user is not asked to
inspect raw T1 candidate rows.

## Implementation status (2026-11-03)

The Wave 3 resolver, keyed and narrative reconciliation, PIT helpers, schema
migration, replay path, and guarded CLI are implemented. The read-only
preflight on `var/wave1/dev.db` found 108/108 current Gemma caches for supported
documents and explicitly excluded 16 unsupported forms. Among non-T0 Gemma
candidates, 1,136 quotes passed and 243 failed; four T0 and four recovery-capped
facts were reported. Gate 2 remains open, so no canonical facts were promoted
from the real dev database.

The full test suite passes (**113 passed**). Pylance reports no diagnostics in
the new resolver, reconciliation, PIT, or CLI code; the only diagnostic was a
pre-existing `contextmanager` annotation warning in `db.py`.

Run a read-only preflight with:

```powershell
uv run am reconcile --database var/wave1/dev.db --model gemma4 --preflight --ticker CXO
```

After Gate 2 is explicitly accepted, reconcile only on a separate copy:

```powershell
New-Item -ItemType Directory -Force var\wave3
Copy-Item var\wave1\dev.db var\wave3\dev.db
uv run am reconcile --database var/wave3/dev.db --model gemma4 --ticker CXO --replay --confirm-wave2-gate
```

`--replay` rebuilds canonical memory for that ticker from current caches and
cached resolver/judge decisions. It does not rerun fact extraction. Uncertain
decisions are retained in `reconciliation_review`; inspect them before Gate 3.

## Wave 4 handoff

Wave 3 supplies the canonical facts and PIT helpers. The one-query pilot and
answer/citation path belong to Wave 4; see
[wave-4-pilot-plan.md](./wave-4-pilot-plan.md). The pre-Wave-4 document covers
only the isolated one-document reconciliation prerequisite. This separation
keeps pilot safety work from duplicating the query implementation plan.

## Current state and entry conditions

- Wave 2 stores candidate facts and raw model output in
  `extraction_cache`; it does not populate `entity`, `attribute`, `fact`, or
  `fact_source`.
- Wave 2 Gate 2 is **still open**. Qwen's full-slice non-T0 quote pass is 88.9%
  and Gemma's is 82.4%, both below the 90% target. Logged Qwen calls were
  schema-valid on the first attempt or one retry in 88.5% of grouped calls,
  below the 98% target. These remain broad-quality concerns, not a reason to
  ask the user to inspect raw candidate rows before the isolated pilot.
- Gemma 4 remains provisional and Qwen 3.5 remains a manual alternate. The
  local `nimble` model is a decision/classification model with a one-choice
  output instruction; its earlier extraction comparison is not a comparable
  generative extraction result and must not be used for canonical promotion.
- The isolated source database is `var/wave1/dev.db`. Wave 3 work must use a
  separate copy at `var/wave3/dev.db`; preserve the Wave 1/Wave 2 database and
  all extraction caches.
- The database already defines `entity`, `entity_alias`, `attribute`, `fact`,
  `fact_source`, and `fact_vec`. Review their constraints and migration history
  before adding schema. Do not duplicate existing tables.
- Pre-gate implementation can cover schema checks, synthetic fixtures,
  resolver tests, and PIT helpers. A one-document exploratory pilot may write
  to its disposable database before Gate 2 closes; do not bulk-promote
  unreviewed candidates or change the default database/model.
- The family-specific, slot-guided extraction idea in the
  [design brainstorm](./memory-design-brainstorm.md) remains a proposal. Keep
  it separate from Wave 3 so reconciliation uses a frozen extraction cache.

## Decisions and invariants

| Topic | Wave 3 default | Revisit if |
|---|---|---|
| Input | One explicit, approved model and a current cache per document; validate `source_doc_id`, prompt version, and source hash | Gate 2 selects a different model or a cache is stale |
| Extraction | Consume the cached candidates; no extraction calls in reconciliation | A cache is missing or invalid; report it rather than silently extracting |
| Entity scope | Resolve entities within the document ticker; no cross-ticker merges | Cross-company entity linking is approved in a later wave |
| Entity matching | Exact normalized name, then source-backed alias, then bounded candidate matching; use a local LLM only for ambiguous choices | Reviews show false merges or too many unresolved cases |
| Attribute matching | Reuse an active canonical attribute when exact or explicitly aliased; require a reviewable proposal for new/ambiguous attributes | The attribute registry needs broader taxonomy governance |
| Knowledge time | `fact_source.published_at` controls when a fact is knowable; `as_of`/`period` describe the fact's subject time | The data model gains explicit bitemporal intervals |
| Key hash | Use canonical ticker/entity/attribute plus temporal grain and qualifier; exclude value so a changed value can match the old key | Tests expose a collision or a missing key dimension |
| Reconciliation | Same key and normalized same value → `CONFIRM`; same key and later, supported changed value → `SUPERSEDE`; otherwise `ADD` | Manual review finds a systematic exception |
| Narrative dedup | Compare only with earlier, same-ticker, PIT-visible facts; confirm only when the underlying claim and time scope match | False confirmations exceed the duplicate trade-off |
| Uncertainty | Never silently convert a failed/ambiguous judge response into `CONFIRM` or `SUPERSEDE`; retain the extraction cache and emit a review/error record | A review workflow establishes a safe automated threshold |
| Tiers | Do not persist T0 candidates. Persist T1–T3; exclude T3 from default state/retrieval later | User changes the tier policy |
| Replay | Cache model-assisted resolution/judge decisions and use stable IDs so replay does not depend on fresh LLM output | The cache format or prompt version changes |
| Timestamps | Equal-publication-time documents cannot influence each other's resolution or reconciliation. Conflicting same-time values are not arbitrarily ordered | A higher-resolution knowledge-time source becomes available |

`fact_source.action` currently allows only `ADD`, `CONFIRM`, and `SUPERSEDE`;
T0 `IGNORE` is a metric/report outcome, not a row to insert into that table.
Each stored fact must have a source record with its document publication time
and quote. Rejected quotes remain in the extraction cache/rejection report,
not in canonical facts.

## Work items

### 3.0 — Freeze and validate the Wave 3 input

**Implementation**

- Copy `var/wave1/dev.db` to `var/wave3/dev.db`; never test promotion against the
  user's primary database or the Wave 1 baseline.
- Add a preflight report for documents, selected model, cache prompt versions,
  stale/missing caches, quote-pass/reject counts, and recovery-capped facts.
- Require an explicit `--model` (or an already approved config value) for
  reconciliation. Do not silently mix Qwen and Gemma facts in one replay.
- Exclude `nimble` from extraction-model eligibility unless a future test
  establishes a suitable generative model/tag. Keep its historical cache for
  audit only.
- Validate the intended cache and source hash before any database writes.

**Acceptance**

- Preflight is read-only and names every skipped/stale/missing cache.
- A selected model is recorded in the run summary.
- No source data or extraction cache changes during preflight.
- Bulk promotion is blocked while Gate 2 remains open.

### 3.1 — Entity and attribute resolution

**Implementation**

- Add a resolver module with typed outcomes: existing identity, new identity,
  or unresolved/needs-review. Do not use a success-shaped fallback for invalid
  model output.
- Normalize names conservatively (Unicode/case/whitespace); do not remove
  meaningful tokens or infer that a rename is an alias.
- Resolve an entity by exact `(ticker, type, canonical_name)`, then an alias
  known by the current document's publication time. Preserve
  `entity_alias.source_doc` and `first_seen_at` when adding a source-backed
  alias.
- Keep Core Exploration/Core Lithium or project/company names distinct unless
  an earlier source or explicit user-approved mapping supports the alias.
- Use nearest-neighbour candidates only to shortlist identities. For
  ambiguous candidates, ask the approved local generative model to choose
  existing/new/unresolved with the choices and evidence supplied; cache the
  typed decision and raw response by input hash, model, and prompt version.
- Resolve attributes against the active registry. New attributes require a
  description and reviewable proposal; do not silently accept near-duplicate
  names.
- Use deterministic entity IDs or another tested stable-ID scheme so replay
  cannot create new identities for the same input.

**Acceptance**

- Tests cover exact match, approved alias, new entity, type mismatch,
  ticker isolation, ambiguous candidate, invalid LLM output, and a future alias
  that must not be visible to an earlier document.
- Spot checks resolve Finniss, BP33, Grants, Core Lithium, and Core
  Exploration consistently without collapsing distinct entities.
- Attribute count does not grow for simple capitalization/spelling variants.

### 3.2 — Key-based reconciliation

**Implementation**

- Implement and version a deterministic `key_hash` function. Use the resolved
  entity, canonical attribute, qualifier, and explicit `as_of`/`period` grain;
  do not include the value in the key.
- Normalize values conservatively. Compare numeric values only when units and
  temporal scope are compatible; otherwise emit a review case rather than
  superseding on a guessed conversion.
- For each accepted candidate, query only facts supported by sources earlier
  than the current document. Apply:
  - no matching key: `ADD` a canonical fact and its origin source;
  - same key and equivalent value: `CONFIRM` the existing fact with a new
    source;
  - same key and different value, later publication, same time grain:
    `SUPERSEDE` the prior fact and link `supersedes_fact_id`;
  - T0 or quote-rejected candidate: do not write a canonical fact.
- Insert or update `fact_vec` only for new canonical facts. A confirmation
  must not create another fact or vector.
- Make each document's reconciliation transaction atomic and idempotent.
  Store source quotes and role (`origin`, `confirm`, or `xref`) on every source
  link.

**Acceptance**

- Unit tests cover `ADD`, same-value `CONFIRM`, changed-value `SUPERSEDE`,
  incompatible units, different periods, rejected quotes, T0, duplicate input,
  and transaction rollback.
- The same cached candidate replayed twice produces no extra fact/source rows.
- A later-period measurement does not supersede a different period's fact.

### 3.3 — Narrative deduplication

**Implementation**

- Use `fact_vec` to retrieve a small candidate set (initially top 5) restricted
  to the same ticker and PIT-visible facts.
- Ask a local generative model only when no deterministic key resolves the
  candidate. The judge must distinguish same claim, related-but-distinct, and
  uncertain; return a short reason and validate against a schema.
- Cache every judge input/output by stable input hash, model, and prompt
  version. The cache is required for repeatable replay.
- `CONFIRM` only if the underlying claim and time scope are the same.
  Related developments, repeated boilerplate, and facts about different
  periods are not confirmations. An uncertain or failed judgment is surfaced
  for review; do not silently merge.

**Acceptance**

- On three quarterlies, reviewed restatements of earlier announcements become
  `CONFIRM`, not duplicate `ADD` facts.
- Similar but distinct events, different periods, and changed guidance remain
  separate.
- A replay uses cached judgments and makes no new model calls.

### 3.4 — Point-in-time views and helpers

**Implementation**

- Add `query/pit.py` helpers for:
  - facts visible at a knowledge date;
  - facts current at a knowledge date;
  - state for `(ticker, entity, attribute, knowledge_date)`;
  - timeline for an entity/attribute with source documents and actions.
- Follow the MVP rule: a fact is visible only if it has a `fact_source` row with
  `published_at <= knowledge_date`; it is current only if no visible
  `SUPERSEDE` row supersedes it.
- Keep knowledge time separate from subject time: a 2026 report about a 2025
  period is not visible to a 2025-06-30 query, even if its `as_of` date is
  earlier.
- Define deterministic ordering for timelines. Do not choose an arbitrary
  winner for conflicting same-time assertions; surface them as ambiguous.
- Keep these helpers database-focused; Wave 4 owns retrieval routing and the
  answer agent.

**Acceptance**

- Synthetic tests cover an add/confirm/supersede chain before, on, and after
  each publication date; back-dated `as_of`; same-day publication; and missing
  source links.
- No query can see a fact before its source publication date.
- Cash and `shares_on_issue` timelines match the reviewed source history.

### 3.5 — Replay, re-ingest, and command integration

**Implementation**

- Add an explicit reconciliation command that consumes caches and reports
  actions; extraction remains a separate, guarded command.
- Support a full dev-slice replay first. Add range-based replay only after
  tests prove that facts, source links, aliases, attributes, and vector rows
  created by the replay range are correctly removed/rebuilt without deleting
  earlier dependencies.
- Rebuild canonical tables in chronological `(published_at, document_id)`
  order using the frozen model/cache selection. Reuse cached resolver/judge
  decisions; no extraction LLM calls.
- Preserve `extraction_cache` across reconciliation replay. Never delete
  source documents or chunks as a shortcut for rebuilding facts.
- Report counts for source candidates, accepted/rejected quotes, ADD/CONFIRM/
  SUPERSEDE actions, ignored T0s, unresolved cases, LLM calls, cache hits, and
  elapsed time.

**Acceptance**

- A full dev-slice replay twice produces identical logical entity, attribute,
  fact, source, and vector contents.
- Injected failure in one document leaves its canonical writes rolled back
  while retaining extraction and decision caches.
- Range replay does not change facts or sources published before the range.
- Command guards reject an unapproved model, stale cache, invalid date range,
  or a request to promote while the Gate 2 prerequisite is open.

## Wave 4 handoff checks

1. [x] Keep schema, resolver, reconciliation, PIT, replay, and failure behavior
   covered by synthetic tests; the full suite passes.
2. [x] Keep `var/wave1/dev.db` unchanged; read-only preflight verifies cache
   freshness and quote coverage.
3. [x] Implement and test the bounded pilot-copy guard from the
   [pre-Wave-4 readiness plan](./pre-wave-4-gap-plan.md). It writes only the
   selected document to a new disposable DB and does not close Gate 2.
4. [x] Assert in tests that the pilot DB contains only the intended pilot
   reconciliation result. This is a structural check, not a request for the
   user to inspect internal fact rows.
5. Query planning, answer generation, citation rendering, and the user's
   answer-level “taste test” belong to the
   [Wave 4 plan](./wave-4-plan.md); see its
   [4.0 pilot details](./wave-4-pilot-plan.md).

Use the repository test command `uv run python -m pytest -q`; keep tests
focused during implementation, then run the full suite before Gate 3.

## Risks and proposed defaults

| Risk | Proposed mitigation |
|---|---|
| Entity alias false merge | Exact/approved aliases first; ticker/type scope; unresolved cases require review |
| Period mismatch mistaken for a revision | Include the period/as-of grain in keys; test quarterly and point-in-time values |
| Same-day conflicting reports get an arbitrary winner | Do not use a hidden tie-break as truth; retain and flag conflict |
| Narrative judge merges related events | Restrict candidates by ticker/PIT, cache decisions, and prefer uncertainty over false confirmation |
| Replay changes due fresh model choices | Cache resolution/judge outputs and use stable IDs |
| Extraction quality contaminates canonical facts | Keep Gate 2 open until user review and quality criteria pass; never promote Nimble's decision-model output |
| Full prompt redesign changes candidate distribution mid-wave | Freeze extraction caches; evaluate family-specific slot prompts separately |

**Suggested default:** implement the Wave 3 mechanics and tests now, but do not
run bulk canonical reconciliation until Gate 2 is closed. Review the same-time
conflict and unresolved-judgment behavior before implementing any automatic
merge policy.
