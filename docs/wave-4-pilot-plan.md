# Wave 4.0 Pilot — First Memory-Backed Query

This is the 4.0 vertical slice of the full [Wave 4 plan](./wave-4-plan.md), not
a pre-Wave-4 prerequisite.
It proves that one provisional Wave 3 memory can answer one useful question
with inspectable evidence. It does not complete the general Wave 4 agent or
close the full-slice Wave 2 quality gate.

Preconditions and the isolated reconciliation command are tracked in the
[pre-Wave-4 readiness plan](./pre-wave-4-gap-plan.md). Do not duplicate those
pilot-database safeguards in this query implementation.

## Current status (2026-11-03)

The Gemma one-document pilot has created `var/wave3/pilot.db` with 13 added
facts, zero quote rejects, and zero unresolved decisions. The cash-balance
fact is visible at the `w0-01` knowledge date, cites the September quarterly,
and is not visible before that document's publication date. The source database
remains unchanged. The memory-backed `am ask` path below is still unimplemented.

## Scope

**In scope**

- One current-state query: gold question `w0-01`, “What was CXO's cash balance
  at 30 September 2026?”
- Ticker `CXO`, inclusive `knowledge_date=2026-10-15`.
- The isolated one-document pilot DB and the explicitly selected extraction
  model used to build it.
- A query planner for the `state` intent, retrieval through Wave 3 PIT
  functions, a constrained answer, and validated source display.

**Out of scope**

- Timeline, event, summary, hybrid chunk fallback, and general agent tools.
- Multiple tickers, full-slice reconciliation, default-model promotion, and
  the formal Gate 4 requirement of 10+ gold questions.
- Asking the user to inspect extraction caches, candidate rows, or internal
  `ADD`/`CONFIRM`/`SUPERSEDE` actions.

## Query and data contracts

### CLI

```powershell
uv run am ask --memory --database var/wave3/pilot.db --ticker CXO --as-of 2026-10-15 "What was CXO's cash balance at 30 September 2026?"
```

`--memory` is explicit during the pilot; it is mutually exclusive with
`--baseline`. `--as-of` and `--ticker` are required for memory mode. The
`--database` override is required for the pilot and must point to its
disposable DB, not `var/wave1/dev.db` or `var/memory.db`. Normal memory mode may
use the configured database.

### Query planning

Use a Pydantic response model with exactly these pilot fields:

```text
intent: "state"
entity_id: one of the IDs supplied from this pilot database
attribute: one of the active attributes supplied from this pilot database
subject_date: optional ISO date
subject_period: optional string
```

The planner receives the user's question, ticker, knowledge date, and the
allowed entity/attribute choices. Reject an unknown ID or attribute; do not
let the model invent them. For `w0-01`, the expected plan resolves to the
Core Lithium entity and the actual cash-balance attribute in the pilot DB. The
extractor has emitted variants such as `cash_balance_as_of` and
`cash_balance_at_30_september_2026`; do not assume the key is exactly
`cash_balance`. The plan should select the matching stored attribute and
September 2026 measurement date/period.

### Retrieval and answer

1. Call `state_at` for the resolved `(ticker, entity_id, attribute,
   knowledge_date)`.
2. Filter the returned observations by `subject_date`/`subject_period` when
   the query planner supplies them.
3. Join each fact to its visible `fact_source` records using
   `fact_sources_at`; exclude any source after the knowledge date.
4. If there is no verified fact, return a clear abstention and do not fall back
   to baseline RAG.
5. Give the answer model only the question, selected facts, and turn-local
   citation IDs (`c1`, `c2`, ...). Use a separate `MemoryAnswer` schema with
   `answer` and `citation_ids`; do not reuse the baseline schema's
   model-generated quote field.
6. Validate every cited ID against the supplied set. On an invalid ID, allow
   one correction attempt; if it is still invalid, fail explicitly. Render the
   exact quote, document ID, headline, publication date, and source path from
   SQLite, never from generated text.

## Implementation checklist

| ID | Work | Likely files | Acceptance |
|---|---|---|---|
| 4.0.1 | Define typed `QuerySpec` and `MemoryAnswer` schemas plus prompts | `query/agent.py`, `prompts/query_plan.md`, `prompts/memory_answer.md` | Invalid intent, entity ID, attribute, or citation ID is rejected |
| 4.0.2 | Implement the single `state` retrieval route | `query/retrieve.py`, `query/pit.py` | Returns only current facts and sources visible by `knowledge_date` |
| 4.0.3 | Add explicit memory mode and pilot DB override to `am ask` | `cli.py` | Accepts question, `--database`, `--ticker`, and required `--as-of`; baseline mode still works |
| 4.0.4 | Render user-verifiable evidence | `query/agent.py`, `cli.py` | Output includes answer, source quote, headline/date, and source path; generated text cannot alter the quote |
| 4.0.5 | Add synthetic and gold-case tests | `tests/test_memory_query.py`, query tests | Correct `w0-01` answer/citation; no-fact abstention; future-source and invalid-citation rejection |
| 4.0.6 | Run the isolated pilot and record user-level feedback | pilot DB only | User can judge answer and source without viewing raw candidates |

## User-level acceptance

The pilot succeeds when:

- The answer is A$200 million for the September 2026 cash-balance question.
- At least one citation is the September quarterly, published before
  `2026-10-15`, and its displayed quote supports the answer.
- No-fact, future-source, and invalid-citation tests pass automatically.
- The user can open the source and judge whether the answer is useful.

The user's feedback is about the answer and evidence only. If the result is
wrong or incomplete, investigate the extraction, reconciliation, retrieval, or
answer layer internally.

Passing this pilot proves one end-to-end path only. Gate 2 remains the
full-slice promotion gate, and Gate 4 still requires 10+ gold questions.
