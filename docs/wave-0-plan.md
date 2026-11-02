# Wave 0 — Foundations: Detailed Plan and Progress

This is the execution checklist for Wave 0 of the [MVP plan](./mvp-plan.md). Wave 0 establishes a reproducible local Python project, validates the source archive, creates the persistent schema, and proves that Ollama can return structured output and embeddings on the local machine. It does **not** extract announcement facts yet.

## Outcome and gate

At the end of Wave 0:

- The project runs under Python 3.13 using `uv`.
- The source loader inventories all 1,316 CXO announcement folders, chooses v2 markdown where present, and falls back to v1.
- SQLite migrations are repeatable; FTS5 and sqlite-vec are available.
- The configured local Ollama chat and embedding models both respond.
- A benchmark of the 13 October 2026 quarterly report records real prompt and generation throughput.

**Gate 0:** `am stats` reports 1,316 announcements, the correct date range and markdown-source counts; `am llm-check` succeeds; and the quarterly benchmark completes in ≤ 60 seconds. If the benchmark is slower, pause before Wave 1 and tune model/context settings.

**Technical gate result: PASS.** Inventory, unit tests, repeatable migrations, local chat, local embeddings and the full-quarterly benchmark all passed. Item 0.6 (gold questions) remains a user-review task but does not block the technical gate.

## Decisions and assumptions for this wave

| Topic | Wave 0 default | Change later if |
|---|---|---|
| Python environment | Python 3.13, `uv`, project-local `.venv` | A later deployment requires a different runtime |
| Chat model | Local `qwen3.5:9b` | The measured benchmark or Wave 2 quality comparison misses its gate |
| Embeddings | Local `qwen3-embedding` (4.7 GB, 4,096 dimensions) | Retrieval evaluation shows poor quality or the vector footprint becomes too large; compare with `bge-m3` or `nomic-embed-text` |
| Storage | SQLite, FTS5, sqlite-vec | Only revisit after the single-ticker MVP demonstrates a real scaling limit |
| Paths | `data/`, `var/memory.db`, relative to the project root | Use a config override for another location |
| LLM calls | Local endpoint at `http://localhost:11434`; no cloud model | Do not use the installed `deepseek-v4.1-flash:cloud` model in this MVP |
| Vector width | Read from the embedding response, not guessed in advance | A model change requires a new vector database or explicit rebuild |

The local machine was checked before implementation: Ollama 0.40.13, Python 3.13.13, uv 0.11.15, RTX 5070 Ti with 16 GB, and `qwen3.5:9b` are available. `qwen3-embedding` was pulled for Wave 0 and returned 4,096 dimensions.

## Work items

### 0.1 — Project scaffold

**Implementation**
- Initialize a `uv` package with a `src/agent_memory` layout.
- Pin the minimum Python version and declare runtime/development dependencies.
- Register the `am` console command.
- Keep generated databases, logs and benchmark results under ignored `var/`.

**Files:** `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `src/agent_memory/__init__.py`.

**Acceptance**
- `uv run am --help` displays the CLI.
- `uv run pytest` works from the project root.
- `uv sync --locked` can reproduce the environment.

**Progress:** Implemented. Initial `uv sync` and package installation succeeded.

### 0.2 — Announcement loader and inventory

**Implementation**
- Scan `data/<ticker>/announcements/<date-headline-id>/`.
- Parse UTC `published_at` from the folder prefix `YYYYMMDD-HHMM` so v1 and v2 have one consistent date source.
- Prefer `2.0.0/*_v2.md`; use `1.0.0/*.md` if v2 text is absent.
- Read either metadata shape: nested `source_metadata` or top-level v1 fields.
- Retain documents with missing markdown in the inventory, marked `text_source=missing`; do not silently drop them.
- Reject malformed date folders and invalid JSON with a path-specific error.
- Report total docs, version counts, missing text/metadata and date range without calling the LLM.

**Files:** `src/agent_memory/loader.py`, `src/agent_memory/config.py`, `src/agent_memory/cli.py`, `tests/test_loader.py`.

**Acceptance**
- `uv run am stats --ticker CXO` reports 1,316 docs.
- Expected text counts from the archive: 168 v2, 1,146 v1, 2 missing.
- Date range: 8 December 2010 through 22 October 2026 (UTC, based on folder timestamps).
- Two folders have neither markdown nor metadata; keep them visible and report their IDs rather than omitting them.
- Tests cover v2 preference, v1 fallback, missing markdown and malformed folder names.

**Progress:** Implemented and verified: 1,316 docs; 168 v2, 1,146 v1, 2 missing; 2 without metadata; 2010-12-08 through 2026-10-22. The two missing-text folders are `20220930-0058-Equity-Raising-Presentation-6A1112439` and `20260714-2222-Quarterly-Activities-and-Cashflow-Report-6A1333881`.

### 0.3 — SQLite schema and migration runner

**Implementation**
- Create the document, xref, paragraph hash, chunk/FTS, entity/alias, attribute, fact/source, extraction-cache, LLM-call and vector-metadata tables in one versioned migration.
- Turn on foreign keys and a bounded busy timeout.
- Use FTS5 external-content triggers so chunk inserts, updates and deletes stay synchronized.
- Load sqlite-vec and create vector tables only after the embedding width is observed. Store the width and reject accidental model-width mismatches.
- Keep migrations idempotent and fail explicitly if the database is newer than the application.

**Files:** `src/agent_memory/db.py`, `tests/test_db.py`.

**Acceptance**
- `uv run am db-init` creates `var/memory.db`.
- Running it twice applies schema version 1 once.
- A test inserts a chunk and finds it through FTS5.
- A test creates vec0 tables with a test width and rejects a different width.

**Progress:** Implemented; automated tests pass for the local sqlite-vec extension and Windows build.

### 0.4 — Local Ollama wrapper

**Implementation**
- Use Ollama's local HTTP endpoint and Python client.
- Structured chat passes a Pydantic JSON schema using `format`, `temperature=0`, `think=false`, configured context and keep-alive.
- Validate the result locally; retry once with the validation error, then raise a clear error.
- Embedding requests return vectors and their measured width.
- Record every successful request, invalid-JSON attempt and transport/server error in `llm_call`, with token counts and duration when available.

**Files:** `src/agent_memory/llm.py`, `src/agent_memory/config.py`, `src/agent_memory/db.py`, `src/agent_memory/cli.py`, `tests/test_llm.py`.

**Acceptance**
- Mocked tests prove schema use, retry behavior, embedding dimensions and persisted call logs.
- `uv run am llm-check` succeeds against Ollama and initializes vector tables at the returned embedding width.

**Progress:** Implemented and live-verified: `qwen3.5:9b` returned valid structured JSON; `qwen3-embedding` returned a vector; the database initialized vector tables at the returned width.

### 0.5 — Model pull and benchmark

**Implementation**
- Pull the default local embedding model.
- Run one structured-output request over the complete 13 October 2026 quarterly report.
- Run embeddings on three short representative texts.
- Save the model names, input characters, prompt/output token counts, prompt/generation speeds, wall time and vector width to `var/bench.md`.
- Do not use cloud models or send announcement text outside the local machine.

**Files:** `scripts/bench.py`, `var/bench.md` (generated, git-ignored).

**Commands**

```powershell
ollama pull qwen3-embedding
uv run am llm-check
uv run python scripts/bench.py
```

**Acceptance**
- Both model requests succeed from the local Ollama server.
- Embedding width is recorded and the database vector tables match it.
- Quarterly chat call completes in ≤ 60 seconds. Update the throughput estimate in the MVP plan from the measured numbers.

**Progress:** Complete. The quarterly report (35,594 characters; 9,274 prompt tokens) took 6.05 seconds end to end. Prompt evaluation was 29,837 tokens/s; generation was 113.9 tokens/s for 418 output tokens. Three embeddings took 0.10 seconds and returned 4,096 dimensions. Full details are in ignored `var/bench.md`.

The generated summary is a **latency sample only**, not a checked answer. It is not added to announcement memory; fact accuracy is tested in later waves.

### 0.6 — Gold questions and final decisions

**User contribution**
- Choose whether to review 15 draft questions or provide your own. The first implementation does not invent evaluation answers because those become the accuracy baseline.
- Confirm the embedding choice after seeing the local smoke test; keep `qwen3-embedding` unless there is a measured reason to switch.
- Confirm whether CXO-only is sufficient for the MVP. No second-ticker data is currently present in the workspace.

**Acceptance**
- 15 questions have a question, expected answer, `knowledge_date` and source document(s) before Wave 2 evaluation begins.

**Progress:** Pending user review; not a blocker for the technical Wave 0 gate.

## Execution order

1. `uv sync --locked` and `uv run am --help`.
2. `uv run am stats --ticker CXO`; compare inventory counts and dates to 0.2 acceptance.
3. `uv run pytest`; fix any Windows-specific database/extension issues.
4. `uv run am db-init`; run a second time and confirm no duplicate migration.
5. `ollama pull qwen3-embedding`.
6. `uv run am llm-check`; inspect the model call log and vector width.
7. `uv run python scripts/bench.py`; record timing in `var/bench.md`.
8. Run `uv run pytest` again and close the gate.

## Risks to resolve before Wave 1

| Risk | Evidence to collect | Response |
|---|---|---|
| sqlite-vec Windows extension loading fails | `tests/test_db.py` | Fix packaging/runtime loading before building the chunk index |
| The default embedding model is unavailable or too slow locally | `am llm-check`, `var/bench.md` | Compare an available local alternative; do not silently fall back to a cloud model |
| Chat is slower than 60 seconds on a representative quarterly | `var/bench.md` | Tune `num_ctx`, model, output limit and prompt before starting ingestion |
| v1/v2 metadata or text selection is wrong | Loader inventory and unit tests | Fix the loader before any persistent document indexing |
