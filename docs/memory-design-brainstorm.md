# Agent Memory for Company Announcements — Brainstorm

Goal: give an agent a **memory** built from listed companies' announcements. The memory should:

1. Hold each piece of knowledge **once**, even when many announcements repeat it.
2. Leave out **trivial details** (AGM venue, webcast time, contact phone numbers) or keep them in a store that normal answers don't search.
3. Return **only what matters** for a user's question, with sources.
4. Answer **as of any past date** without using information published after that date (needed for backtesting).

Data used for this brainstorm: `data/CXO/announcements/` has about 1,316 Core Lithium (ASX: CXO) announcements from 2010 to 2026. Each has `*.json` metadata (headline, `category`, `announcement_types`, `is_price_sensitive`, `published_at`) and a `*_v2.md` text conversion.

---

## 0. Decisions (2026-11-02)

| Topic | Decision | Effect on the design |
|---|---|---|
| Scope | **Multiple tickers, about 2,000. Each ticker is handled separately; no cross-ticker entities** | Every entity belongs to its ticker. A counterparty (for example, Glencore) is just an entity within each ticker that mentions it, with no shared registry or merging across tickers. The attribute vocabulary must work for every sector. Scale is about 1–2M documents (§7) |
| Point in time | **Required, for backtesting** | Two time axes on every fact, and each fact's knowledge time is the **publish date**, not the date it was ingested. Summaries and the reconciliation context must also be point-in-time (§3.0, §6.1) |
| Processing style | **LLM-first: fast to build, flexible** | Use the LLM for sorting, extraction, importance and matching. Rules and metadata are only hints and cost filters, not hand-written parsers (§3.1) |
| Revisions / corrections | **Not in MVP** | No contradiction detection. A newer `as_of` (or a newer publish date for the same `as_of`) simply replaces the older value. Detailed financial statement extraction is deferred |
| User notes | **Query phase only** | Separate store. Never used as input to extraction or reconciliation (§6.2) |
| Refresh | **Incremental, one document at a time**. Re-ingest when quality is poor or new capability is added | Per-document pipeline that can safely be re-run, in publish order within each ticker. Every output records the extractor version, so a ticker can be rebuilt (§3.0, §9) |

---

## 1. What the data shows

### 1.1 Most announcements are administrative

Grouping headlines by type shows that the most common types are routine:

| Headline pattern | Count (approx.) | Value to an agent |
|---|---|---|
| Application for quotation of securities / Appendix 2A / 3B | 60+ | Low by itself. Useful only as a share-count time series |
| Cleansing notices (share issue, placement, SPP, options) | 80+ | Very low. Legal boilerplate |
| Trading halt / reinstatement | 38 | Low by itself, but it **points to** the material announcement that follows |
| Change of Director's Interest | 70+ | Medium as aggregated insider buy/sell behaviour, low per document |
| Substantial holder (become / change / cease) | 40 | Medium: tracks who holds the register |
| Quarterly activities & cashflow reports | ~56 | **High**, but each one repeats a lot |
| Notice of meeting / AGM results | ~20 | Resolutions and vote outcomes matter. Venue, time and proxy instructions do not |

So the first decision is **triage**: does the document produce long-term facts, structured data, or nothing?

### 1.2 Repetition within and across documents

From `20261013-...Quarterly-Activities-and-Cashflow-Report` (Sep-26 quarter):

- **Cross-references.** Each highlight ends with "Refer to ASX announcement 'X' on date". The fact was already stated in X. The quarterly repeats it.
- **Restated reference data.** "BP33 MRE of 10.5Mt @ 1.53% Li2O" is quoted from an **October 2023** announcement and repeated in every later quarterly.
- **Boilerplate.** "About Core", forward-looking-statement disclaimers, Competent Person statements, "approved for release by the Board", and investor/media contacts appear in nearly every document.
- **Large tables repeated each quarter.** The tenement schedule (about 40 rows) is almost the same each quarter. Only the "Changes during the Quarter" column matters.
- **Values that change over time.** "Cash A$200m at 30 Sep 2026 (30 Jun 2026: A$182m)". This is **not** a duplicate. It is a new point in a time series, and the bracketed value repeats the previous point.

### 1.3 Trivial or short-lived details in the same document

- Webcast date, time and registration link
- Office address, phone number, email
- Media adviser contact details
- Reminders like "a recording will be available on the website"

These are useful for a few days and then never again.

### 1.4 Entity drift

- The company renamed itself from **Core Exploration** to **Core Lithium**. Both names appear in headlines.
- Projects (Finniss, BP33, Grants, Yorke Peninsula) and people (directors, MD, CFO) come and go.
- Assets were spun out to **Axiant Resources (AXR)** and a tenement was acquired from **Charger Metals (CHR)**. Tenement ownership changes over time.

Without an alias table and time-bounded relationships, the dedup step will miss matches or merge the wrong things.

**Decision:** each ticker is handled separately. Entities that appear in several companies (counterparties, directors on several boards, spin-outs like CXO → AXR) are kept as separate entities **within each ticker**. They are not linked across tickers.

---

## 2. Core design idea: store facts, not documents

Raw documents already exist and act as long-term storage. The memory sits **on top of them** as several layers:

```
L0  Raw archive        PDFs / markdown / json        (already have; never delete)
L1  Clean chunks       boilerplate-stripped passages  (fallback retrieval, citations)
L2  Atomic facts       one claim each, with sources   (deduplicated)
L3  Entity state       current value + history        (time series / timeline)
L4  Summaries          per project / per quarter / company profile
```

- The agent answers from **L3/L4 first**, uses **L2** for detail, and falls back to **L1** only when needed.
- Dedup happens at L2 (facts) and L1 (chunks).
- Filtering trivial content happens at L1 (boilerplate removal) and L2 (importance tiers).

---

## 3. Ingestion pipeline (brainstorm)

### 3.0 Ordering and point-in-time rules for ingestion

Ingest **one document at a time**. Within a ticker, process documents strictly in `published_at` order. Different tickers can run in parallel.

Every step that reads existing memory (finding boilerplate, resolving cross-references, entity matching, reconciliation) must only see a **snapshot as of the document's `published_at`**:

```
context(doc) = memory WHERE first_seen_at < doc.published_at
```

This matters even when ingesting in order. When a ticker is re-ingested, or a missed older document is backfilled, later facts are already in the store. Without the filter, they would leak into decisions about earlier documents.

**Late or out-of-order documents.** In the MVP, don't try to merge them into the existing history. Re-ingest the ticker from that document's date onward. One ticker's history is small (about 1k documents), so this is affordable and keeps the logic simple.

**LLM knowledge leakage.** The model already knows how things turned out ("BP33 was later delayed"). Mitigations:
- The prompt contains only the document plus the point-in-time memory context.
- Instruct the model to extract **only what the text states**.
- Reject any extracted fact whose evidence quote can't be found in the document (a cheap substring or fuzzy check).

```
announcement  (one at a time, in publish order per ticker)
   │
   ├─► 1. Triage (LLM, metadata as hints)
   │        drop / structured-only / full-extract
   │
   ├─► 2. Boilerplate removal (running per-ticker + global paragraph counters)
   │
   ├─► 3. Cross-reference resolution ("Refer to ASX announcement …")
   │
   ├─► 4. Fact extraction + importance tier (LLM, JSON schema, evidence quote)
   │
   ├─► 5. Quote check (reject facts whose quote isn't in the document)
   │
   ├─► 6. Reconcile against point-in-time memory snapshot
   │        ADD | CONFIRM | SUPERSEDE | IGNORE      (CONTRADICT: deferred)
   │
   └─► 7. (nothing to refresh: state & summaries are point-in-time views)
```

### 3.1 Triage

Use the LLM, with the metadata as a hint rather than a hard rule:

- Input: the headline, `category`, `announcement_types`, `is_price_sensitive` and the first page or so of text. Use a cheap, fast model.
- Cost guard: a configurable list of headline patterns that are always skipped (cleansing notices and similar) can bypass the LLM. Keep it as config, not code, and let the LLM override it if the first page contradicts it.
- Three outputs:
  - **Skip.** Cleansing notices and pure ASX forms. Record that the document exists, extract nothing.
  - **Structured only.** Appendix 2A/3B, director interests, substantial holders. Use the LLM with a small fixed schema per form type (share count, holder %, director holdings), not hand-written parsers. These don't need free-text memory.
  - **Full extract.** Price-sensitive announcements, quarterlies, half/full-year reports, resource/reserve updates, drilling results, funding, offtake, M&A, guidance.
- Trading halts: store only "halt on date D → followed by announcement X". The halt explains the timing of X but carries no content of its own.

### 3.2 Boilerplate removal

Since documents arrive one at a time, boilerplate is detected with **running counts** rather than a full-corpus pass:

- Split each document into paragraphs. Normalise them (lowercase, collapse whitespace and numbers). Hash them with SimHash or MinHash.
- Keep two running counters of paragraph hashes, each limited to the point-in-time snapshot (§3.0):
  - **Per ticker**: company-specific boilerplate ("About Core", contact block, Competent Person text).
  - **Global**: cross-company boilerplate (ASX form wording, standard disclaimers, law-firm templates).
- A paragraph seen in **≥ K earlier documents** (for example, K = 3 for the same ticker, or much higher globally) is boilerplate → mark it and skip it in L1/L2.
- **Cold start** (a ticker's first documents): the counters are empty, so the LLM labels boilerplate during extraction ("ignore disclaimers, contact details and company descriptions"). The global counter takes over quickly because templates are shared across tickers.
- Keep a small allow-list for paragraphs that repeat but still matter (for example, the latest Mineral Resource statement). These are still CONFIRMed rather than added again, so nothing is lost.

### 3.3 Cross-reference resolution

- Regex: `Refer to ASX announcement “(.+?)” (?:on|dated) (\d{1,2} \w+ \d{4})`
- Match it to the original announcement by headline and date.
- Any fact extracted next to the reference is linked to the **original** source and treated as a **confirmation**, not a new fact.
- This builds a citation graph between announcements for free.

### 3.4 Fact extraction

Ask the LLM to output **atomic, self-contained facts** in a fixed schema (see §4). Guidelines:

- One claim per fact. Resolve pronouns ("the deposit" → "BP33").
- Normalise units and periods ("Sep quarter 2026" → `2026-Q3-CY`, "A$200 million" → `200000000 AUD`).
- Record `as_of` (the date the fact describes) separately from `published_at` (the date the document was released).
- Mark the type: `metric`, `event`, `plan/guidance`, `relationship`, `reference_data`, `opinion/commentary`.
- Every fact must include a short **evidence quote** from the document. It is used for the leakage check (§3.0) and for citations.
- **Attribute vocabulary across 2,000 tickers.** Pass the LLM the existing vocabulary: generic attributes plus those used for this ticker or sector (`cash_balance`, `revenue`, `production_volume`, `resource_estimate`, `guidance_*` …). The LLM either reuses one or **proposes a new one**, which goes into a registry. A periodic job merges near-duplicate attributes. This avoids designing a full schema up front while still allowing facts to be matched by key.

### 3.5 Importance tiers (how "trivial" gets decided)

Combine rules with an LLM score from 0 to 1. Store a tier label rather than deleting anything.

| Tier | Examples | Stored? | Searched by default? |
|---|---|---|---|
| **T1 Core** | Resource/reserve, production, cash, funding, debt, offtake, guidance, FID, M&A, strategy, key management changes | Yes (L2+L3+L4) | Yes |
| **T2 Context** | Drill intercepts, construction progress, capital raises, substantial holders, director trades (aggregated) | Yes (L2+L3) | Yes, ranked lower |
| **T3 Short-lived** | AGM venue/time, webcast details, conference attendance, office address, contacts | Optional cold store with an expiry | **No.** Only when the question asks for it directly (for example, "where was the 2024 AGM?") |
| **T0 Noise** | Boilerplate, legal disclaimers, forms | No (remains in L0 only) | No |

Signals that feed the score:
- `is_price_sensitive` flag
- Category weight (quarterly > ASX form)
- Is it a number tied to a known metric?
- Does it change a tracked entity's state?
- Will it still matter in 30 or 365 days? This can be a question to the LLM: "Would an investor care about this 1 year from now?" The model must judge from the document only, **not from knowing what happened afterwards** (§3.0).
- How often it appears in the corpus. Very frequent text is likely boilerplate. A value repeated with no change adds confidence but no new information.

Recommendation: **keep T3 in a cold store instead of dropping it**. Storing it is cheap. The goal is to keep it out of normal retrieval, not to lose it.

### 3.6 Reconciliation and dedup (the core of the problem)

For each new candidate fact, look for existing facts with the **same key**, searching only the point-in-time snapshot as of the document's publish date (§3.0):

```
key = (entity_id, attribute, period/as_of, qualifier)
       e.g. (BP33, mineral_resource_total, 2023-10-16, "JORC 2012")
```

Then decide:

| Situation | Action |
|---|---|
| Same key, same value | **CONFIRM**: add the source and update `last_confirmed`. No new row |
| Same key, different value, newer `as_of` | **SUPERSEDE**: add the new fact and record a supersede link to the old one. When queried at a date on or after the new document's publish date, the old fact is treated as closed (see §4 notes) |
| Same key, different value, same `as_of` | MVP: **SUPERSEDE** by the newer publish date. Keep the old row, closed, so it can still be audited. Proper revision and contradiction handling is deferred |
| New key for an existing time-series attribute | **ADD** a point (for example, the cash balance for a new quarter) |
| Same meaning, but no structured key (narrative events) | Embedding similarity above a threshold → LLM judge decides "same event?" → CONFIRM or ADD |
| Matching fact found in a cross-referenced document | CONFIRM (see §3.3) |

Dedup checks, from cheapest to most expensive:
1. Exact hash of normalised text
2. Near-duplicate (MinHash/SimHash)
3. Structured key match
4. Embedding similarity plus an LLM judge, used only for unclear cases

This is similar to the ADD/UPDATE/DELETE/NOOP step in Mem0 and the bi-temporal edges in Zep/Graphiti. Their papers are useful references.

### 3.7 Entity state and summaries

- **State**: latest value per `(entity, attribute)` as of the query date. Answers "what is X now?" quickly.
- **Timeline**: all versions with `valid_from` and `valid_to`. Answers "how has X changed?"
- **Rolling summaries**:
  - Company profile (about one page, rebuilt after each T1 change)
  - Per project (Finniss, BP33, Grants, …)
  - Per period (each quarter, about 10 bullets)
- Summaries are rebuilt **from facts**, not from documents, so repeated content in the documents doesn't carry over.
- **Point in time:** a summary written today mentions things that happened later, which would leak into a backtest. Options:
  - (a) Keep a version of each summary keyed by the publish date of the last document it covers. Look up the latest version on or before the query date.
  - (b) Don't store summaries. Build them at query time from point-in-time-filtered facts.
  
  MVP: **(b)** for correctness. Add (a) as a cache later if query cost becomes a problem.
- The state table has the same problem. Treat it as a view over facts, filtered by `first_seen_at` and `valid_from`/`valid_to`, rather than a separate table that is overwritten.

---

## 4. Draft data model

```text
Document(id, ticker, headline, published_at, category, types[], price_sensitive,
         triage_class, cites[] -> Document,
         ingest_status, extractor_version, ingested_at)

Entity(id, ticker,                                   -- always ticker-scoped
       type[company|project|deposit|tenement|person|counterparty|security],
       canonical_name, valid_from, valid_to)

EntityAlias(entity_id, alias, valid_from, valid_to, source_document_id)
       e.g. "Core Exploration Ltd" -> Core Lithium Ltd (renamed)

Attribute(name, description, sector?, status[active|merged], merged_into?)   -- growing vocabulary

Fact(id, ticker,
     entity_id, attribute, value, unit, qualifier,
     as_of, period, valid_from, valid_to,          -- event time
     first_seen_at,                                 -- knowledge time = MIN(source.published_at)
     fact_type, tier, salience, confidence,
     status[current|superseded],                    -- MVP: no contradicted/retracted
     text,                                          -- self-contained sentence
     embedding,
     extractor_version)

FactSource(fact_id, document_id, published_at, page, quote,
           role[origin|confirm|xref], action[ADD|CONFIRM|SUPERSEDE])

Relation(src_entity, rel_type, dst_entity, valid_from, valid_to, first_seen_at, source_fact_id)
       e.g. (Glencore) -offtake_buyer-> (Finniss), (EL 31146) -owned_by-> (AXR)

Chunk(id, document_id, published_at, text, is_boilerplate, simhash, embedding)   -- L1 fallback

ParagraphHash(scope[ticker|global], ticker?, simhash, first_seen_at, doc_count)   -- running boilerplate counters

UserNote(id, user_id, ticker?, entity_id?, text, created_at)    -- query phase only, never fed to extraction
```

Notes:
- `FactSource` is how dedup keeps information: one fact can have many sources. "Mentioned in 14 announcements" also works as a measure of how important the fact is.
- **Point-in-time rule:** a fact, and each of its sources, becomes visible at a query date `T` only if its `published_at <= T`. The confirmation count, `last_confirmed` and the superseded status are all **calculated as of `T`** from `FactSource` rows. They are not stored values that get overwritten. Example: a fact superseded in 2027 still shows as `current` for a backtest at 2026.
- Superseding is recorded as an event (`FactSource.action = SUPERSEDE` on the new fact, pointing to the old fact) rather than by editing the old fact's `valid_to`. This avoids look-ahead.
- `attribute` comes from the `Attribute` registry, which grows over time (§3.4). Examples include generic attributes (`cash_balance`, `revenue`, `shares_on_issue`, `director`, `ceo`, `guidance_*`) and sector-specific ones (`resource_tonnes`, `resource_grade`, `ore_reserve`, `aisc` for mining). Free-text attributes make key matching unreliable.
- `extractor_version` on documents and facts allows a ticker to be re-ingested selectively when prompts or models improve: delete that ticker's facts with version < N, then replay its documents in order.

---

## 5. Worked example: Sep-2026 quarterly

| Text in the announcement | Result |
|---|---|
| "First spodumene concentrate was produced… (Refer 8 Sep 2026)" | CONFIRM existing event fact from the 8-Sep announcement; add source |
| "Throughput capacity +20% to 1.2 Mtpa" | CONFIRM (originally in the 18-Mar-2026 FID announcement) |
| "First development ore at BP33 mid-CY27" | Guidance fact. CONFIRM if unchanged. SUPERSEDE if the date has moved (and flag it as a schedule change) |
| "34.08 m @ 2.09% Li2O from 578.7 m in NMRD100" | T2 drill result. CONFIRM (11-Aug announcement) |
| "Sale of ~25,000 t fines to Glencore; ~A$38.5m CY26 stockpile revenue" | CONFIRM the sale event. SUPERSEDE the `stockpile_revenue_cy26` metric |
| "Tranche 2 CN US$44m received" | CONFIRM (4-Aug) |
| "Cash A$200m at 30 Sep 2026 (30 Jun: A$182m)" | **ADD** a new cash time-series point. Bracketed value → CONFIRM the existing June point |
| "BP33 MRE 10.5Mt @ 1.53% Li2O" (from Oct 2023) | CONFIRM the existing reference fact. No new information |
| Capital structure (3.24bn shares, 32.9m perf rights, …) | ADD a `shares_on_issue` time-series point |
| Director fees A$184k (Appendix 5B item 6.1) | T2/T3 metric. Store it but rank it low |
| Tenement table | Diff against the previous quarter. Store only the **changes** (acquired a tenement from CHR; tenements transferred to AXR) |
| Webcast 14 Oct 8:30am AWST + link | T3 → cold store with `valid_to` = event date + a few days. The expiry is in **event time**, so a backtest at 13 Oct 2026 still sees it |
| About Core, disclaimers, CP statement, contacts | T0 boilerplate, not extracted |

Expected result: a 19-page document produces about **3–5 new facts**, about **10 confirmations**, and about **2 relationship changes**. That much reduction is the goal.

---

## 6. Retrieval for the agent

### 6.1 Point-in-time is a required parameter

Every retrieval call takes a `knowledge_date` (default: now). Every layer is filtered by it:

| Layer | Filter |
|---|---|
| L1 chunks | `published_at <= knowledge_date` |
| L2 facts | at least one `FactSource.published_at <= knowledge_date`. Status, confirmation count and earliest source are calculated using only those sources |
| L3 state / timeline | Latest fact per key, ignoring supersede links from documents published after `knowledge_date` |
| L4 summaries | Built at query time from the filtered facts (§3.7) |
| Entities / aliases / relations | `first_seen_at <= knowledge_date` (for example, don't use the name "Core Lithium" before the rename was announced) |

To check for leakage, add a test that runs the same question at `T` before and after ingesting documents published after `T`. The answers must be identical.

### 6.2 Query flow

1. **Intent routing**
   - "What is current X?" → L3 state
   - "How has X changed / when did X happen?" → L3 timeline + L2 facts
   - "Summarise the company / project / quarter" → L4 summary
   - Detailed, rare or exact-wording questions → L1 chunks (boilerplate excluded)
   - Questions that compare several tickers or a sector → filter by ticker list or sector, then L3 across tickers
2. **Default filters**: current status as of `knowledge_date`, `tier ∈ {T1, T2}`. Include T3 only if the question names that kind of detail (AGM, webcast, address, contact).
3. **Hybrid search**: ticker/entity/attribute filter + BM25 + vector, then rerank.
4. **User notes (query phase only)**: retrieve the user's notes for the tickers or entities in the question. Use them to **re-rank and frame** the answer (for example, "user tracks cash runway and BP33 timing"). Notes are never written into facts and never shown as company-sourced information. For backtests, decide whether notes written after `T` are allowed. Default: allowed, since they express user preferences rather than market information.
5. **Answer packing**: one row per fact with its **earliest source** and "confirmed N times, last on DATE" (both calculated as of `knowledge_date`). Never return the same fact several times from different documents.

Agent tools to expose (all accept `knowledge_date`):
- `get_state(ticker, entity, attribute?)`
- `get_timeline(ticker, entity, attribute, from?, to?)`
- `search_facts(query, tickers?, tier_min?)`
- `get_summary(scope)`
- `get_sources(fact_id)` / `open_document(doc_id, page?)`
- `get_user_notes(tickers?, entities?)`

---

## 7. Technology options

Scale: about 2,000 tickers × about 500–1,500 documents ≈ **1–2M documents**. At an estimated 5–20 facts and sources per extracted document, that is roughly **10–30M fact/source rows** plus chunks.

| Concern | MVP (a few tickers) | Full scale (2,000 tickers) |
|---|---|---|
| Storage | SQLite + `sqlite-vec` / DuckDB + FTS (one file, easy to inspect) | Postgres + pgvector, partitioned by ticker |
| Graph / relations | Relational tables | Still relational. Add Kùzu / Neo4j only if cross-company graph queries become central |
| Near-duplicate detection | `datasketch` (MinHash LSH), simhash | Same, with the global hash table in Postgres or Redis |
| Extraction | LLM with JSON-schema / structured output | Cheap model for triage and forms, stronger model for price-sensitive documents. Batch APIs for backfill |
| Orchestration | Simple loop: one doc at a time, ordered per ticker | Queue with **one worker per ticker at a time** (order preserved), parallel across tickers |
| Reference frameworks | Mem0 (ADD/UPDATE/NOOP), Zep/Graphiti (bi-temporal KG), Letta/MemGPT (tiered memory) | — |

Almost all reconciliation happens **within a ticker**. Partitioning by ticker keeps the context small and lets the full backfill run in parallel. Only the global boilerplate counters are shared.

**LLM cost check:** triage (cheap) runs on all documents. Full extraction runs on about 20–40% of them (price-sensitive documents and periodic reports). Estimate the cost on the CXO set first, then scale up.

---

## 8. Evaluation

- **Question set**: 50–100 realistic questions. Current state, history, point-in-time, "why", and some trivial ones such as "where was the AGM?"
- Metrics:
  - Answer accuracy and citation correctness
  - **Duplicate rate** in retrieved context (target ≈ 0)
  - **Tokens per answer** compared with naive RAG over chunks
  - **Facts per document** (shows how much is being compressed)
  - Proportion of T3/T0 content in default retrieval (target ≈ 0)
  - **Leakage rate**: proportion of point-in-time answers that cite a source published after `knowledge_date` (target = 0, must pass)
  - **Order-independence check**: ingest a ticker, then re-ingest it from scratch. The point-in-time answers should match
- Baseline: plain chunk RAG over `*_v2.md`. The memory system should match or beat it on accuracy while using far fewer tokens.

---

## 9. Suggested MVP path

> Concrete plan: see [mvp-plan.md](./mvp-plan.md).

Start with **CXO plus 2–4 tickers from different sectors** (for example, a bank, a tech company and a REIT) so the attribute vocabulary and entity model aren't tuned only for mining.

1. **Document registry**: load metadata into SQLite and order documents by `(ticker, published_at)`.
2. **Per-document pipeline** (one doc at a time, safe to re-run, stamped with `extractor_version`):
   1. LLM triage (with the skip list from config as a cost guard)
   2. Paragraph hashing + running boilerplate counters (per ticker + global), with the LLM handling cold start
   3. Cross-reference resolution against earlier documents
   4. LLM extraction: facts with an evidence quote, attribute (from the registry or a proposed new one), tier. Form types use small fixed schemas
   5. Quote check (rejects facts the model added from its own knowledge)
   6. Reconciliation against the point-in-time snapshot: ADD / CONFIRM / SUPERSEDE (no contradiction handling)
3. **Query layer**: point-in-time views for state and timeline, summaries built at query time, agent tools with `knowledge_date`, user notes used for re-ranking.
4. **Evaluation**: question set + leakage test + re-ingest consistency test.
5. **Re-ingest tooling**: `reingest(ticker, from_date?, min_version?)`.

**Deferred (not MVP):** detecting revisions and contradictions, detailed financial statement extraction (P&L, balance sheet, segment data), cached point-in-time summaries, graph database, attribute-merge job (handle manually at first).

---

## 10. Open questions

Resolved: scope (2,000 tickers), point-in-time (required), refresh (incremental, one doc at a time), corrections (deferred), user notes (query phase only).

Still open:
- **T3 handling**: drop it entirely or keep it in a cold store? (Recommendation: cold store with an event-time expiry.)
- **Who decides importance**: LLM score only, or later adjusted by user feedback ("this was useful / not useful")?
- **Global entity matching**: not needed (each ticker is handled separately).
- **Attribute registry governance**: allow the LLM to create attributes freely and merge them later, or require approval for new ones beyond a small set per sector?
- **Backtest and user notes**: should notes written after `T` be allowed in point-in-time queries? (Default above: yes.)
- **Re-ingest trigger**: how is "poor quality" detected? Evaluation score dropping, sampled human review, or a new model or prompt version?
- **Exchange coverage**: are all 2,000 tickers on the ASX (same form types), or are other exchanges with different document formats included?
