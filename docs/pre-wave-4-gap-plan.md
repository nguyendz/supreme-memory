# Ad Hoc Plan — Readiness Gaps Before Wave 4

This document covers only the remaining prerequisites at the Wave 3 exit.
The user-facing query itself is Wave 4 work; see the [Wave 4.0 pilot
plan](./wave-4-pilot-plan.md).

## Current checkpoint

- Wave 0–1 foundations and the CXO dev-slice ingestion path are implemented;
  Gate 1 is closed.
- Wave 2 extraction caches and baseline RAG are available. Full-slice
  extraction metrics remain below Gate 2, so the default model and broad
  promotion are still provisional.
- Wave 3 resolver, reconciliation, PIT helpers, replay, and synthetic tests
  are implemented. The real source database still has zero canonical facts.
- A read-only preflight found 108/108 current Gemma caches for supported docs
  and 16 unsupported forms. The user is not expected to inspect the raw T1
  review files.
- The focused pilot preflight found one quote-verified `w0-01` cash-balance
  candidate in both Gemma and Qwen caches. The Gemma pilot was then run on
  `var/wave3/pilot.db`: 13 facts added, zero quote rejects, zero unresolved
  decisions. Its cash fact is visible at `2026-10-15` and not at
  `2026-10-12`. The source DB remains schema version 3 with zero entity,
  attribute, fact, or fact-source rows.
- The supported test command is `uv run python -m pytest -q`; the bare
  `uv run pytest` launcher issue is an environment cleanup, not a Wave 4
  blocker.
- The full suite passes (118 tests); Pylance reports no diagnostics in the
  pilot/reconciliation changes.

## Gaps to fill before starting the Wave 4 pilot

| ID | Gap | Required work | Exit condition | Status |
|---|---|---|---|---|
| P4.1 | No way to reconcile one real document while Gate 2 is open | Add a bounded `--pilot` mode to the Wave 3 reconciliation command | It writes one explicitly named document only to a new disposable database | Complete; command and guard are implemented |
| P4.2 | No verified pilot cache selection | Check current source hash, form family, prompt version, quote validity, and expected `w0-01` fact for the explicitly selected model | Pilot either reports one eligible cache or a precise blocking reason | Complete; Gemma and Qwen each report one matching candidate |
| P4.3 | Pilot isolation is not yet demonstrated | Use SQLite backup to copy `var/wave1/dev.db` to a new `var/wave3/pilot.db`; never overwrite the source or primary DB | Source database schema and canonical-table counts remain unchanged | Complete in tests and the Gemma pilot run |

### Pilot-mode contract

The implemented interface is:

```powershell
uv run am reconcile --source-db var/wave1/dev.db --pilot --document-id 20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068 --model gemma4 --preflight
```

This read-only preflight reports that both Gemma and Qwen have one eligible
`w0-01` candidate. To create the disposable DB and reconcile the one document:

```powershell
uv run am reconcile --source-db var/wave1/dev.db --pilot --pilot-db var/wave3/pilot.db --document-id 20261013-2139-Quarterly-Activities-and-Cashflow-Report-6A1349068 --model gemma4
```

Execution requires a new destination under `var/wave3`; it rejects the primary
DB, an existing destination, multi-document options, replay, and
`--confirm-wave2-gate`. It copies with SQLite backup, prints an exploratory
pilot warning, and leaves extraction caches in the copy unchanged. **This
execution used Gemma and added 13 facts to the disposable copy; no source DB
rows were changed. The Wave 4 query has not been implemented or run.

For `w0-01`, the internal cache check is specific and deterministic: resolve
the Core Lithium cash-balance candidate for 30 September 2026, accepting the
observed attribute spellings `cash_balance`, `cash_balance_as_of`, or
`cash_balance_at_30_september_2026`; check that its value supports A$200
million, and verify its quote against its declared segment. Model selection
remains explicit; the command never switches from Gemma to Qwen automatically.

## Deferred until after the first query

- Do not block Wave 4 on the full-slice quote/schema thresholds. Keep Gate 2
  open for bulk promotion and production-default decisions.
- Do not ask the user to inspect T1 rows, entity IDs, `ADD`/`CONFIRM` actions,
  or reconciliation review JSON. Those remain engineering diagnostics.
- Defer the Wave 0 loader error-message cleanup and cross-ticker boilerplate
  calibration; neither prevents a one-document CXO pilot.
- Full intent routing, timelines, summaries, 10+ gold questions, and Wave 5
  evals remain Wave 4/5 scope.

## Ready-to-start-Wave-4 checklist

- [x] Pilot-mode guard and database copy are implemented and tested.
- [x] Gemma and Qwen `w0-01` caches pass the automated cache/value/quote
      check; otherwise a different explicitly chosen gold case is recorded.
- [x] A test proves the source DB remains unchanged and only the named document
      is reconciled in the pilot DB.
- [x] The [Wave 4.0 pilot plan](./wave-4-pilot-plan.md) is the sole checklist
      for query, answer, citation, and user-level validation work.

The pre-Wave-4 checklist is complete, including the isolated pilot
reconciliation. **The user-facing Wave 4 query has not been implemented or
run.** The pilot does not close Gate 2 or authorize full-slice reconciliation.
