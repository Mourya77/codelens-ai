# CodeLens AI

Production-grade Agentic RAG GitHub Repository & Code Documentation Analyzer.

Built phase-by-phase as a unified monorepo:

- **Backend:** Python + FastAPI (this phase)
- **Frontend:** React + TypeScript + Vite + Tailwind CSS (later phase)
- **Agents / LLM:** later phase

## Phase 4 — Frontend SPA (React + TypeScript + Vite + Tailwind) ✅

`frontend/` is a dark, VS Code/Linear-style developer workspace:

- **Top bar** — CodeLens logo, active-repository dropdown (from
  `GET /api/repos`), status/commit pill, and an **Index New Repo**
  modal with a GitHub URL input plus one-click samples
  (`tiangolo/fastapi`, and tiny `octocat/Hello-World` for instant
  demos).
- **Live ingestion banner** — consumes `POST /api/repos/index` as a
  real SSE stream (fetch + ReadableStream, since EventSource can't
  POST) and renders a progress bar, chunk/file counters, and the
  canonical stage stepper: Fetching Tree → AST Chunking → Vector
  Embedding → Ready. On `complete` it switches the workspace to the
  freshly indexed repo.
- **Left sidebar** — interactive file tree (`GET
  /api/repos/{id}/tree`) with per-file AST-chunk badges; selecting a
  file lists its symbols (via `GET /api/repos/{id}/chunks`, an
  endpoint added in this phase) with type icons and line ranges.
- **Center chat** — multi-turn (history sent with each query),
  starter questions, and above every assistant answer a collapsible
  **Agent Reasoning Trace** accordion rendering the exact
  `agent_trace` steps — analyze, retrieve, grader verdict (with
  missing-symbol chips), hops, answer — with a header badge that
  calls out when self-reflection triggered a second hop. Citations
  render as clickable badges.
- **Right inspector drawer** — clicking a citation badge, a source
  card, or a sidebar symbol opens the exact chunk: syntax-highlighted
  code with real line numbers, `context_header`, symbol, RRF /
  cosine / lexical scores, imports, and a "Open these exact lines on
  GitHub" link (`/blob/{sha}/{path}#Lstart-Lend`).

All API calls are relative (`/api/...`). Production: FastAPI serves
`frontend/dist` from the same origin (static mount registered after
the API routes, only when the build exists). Dev: `npm run dev` with
Vite proxying `/api` → `localhost:8000`.

```bash
cd frontend && npm install && npm run build   # tsc + vite build
```

## Phase 3 — Hybrid RRF retrieval + agentic RAG ✅

- `backend/app/retriever.py` — `HybridCodeRetriever`: cosine semantic
  ranking over stored embeddings fused with lexical ranking via
  Reciprocal Rank Fusion (`1/(60+rank_sem) + 1/(60+rank_lex)`. The
  lexical ranker is identifier-aware BM25 (full identifiers like
  `verify_jwt_token` kept whole *and* split into subwords) with hard
  boosts for exact `symbol_name` matches and file-path hits, blended
  with native `ts_rank_cd` on PostgreSQL. Supports `file_path_filter`
  and `symbol_type_filter`. `find_symbol_definition(repo_id, symbol)`
  returns exact definition chunks, falling back to chunks that import
  the symbol; `get_file_chunks` serves file-level second hops.
- `backend/app/agent.py` — `CodebaseAgent`, an explicit state machine:
  (1) Query Analyzer & Planner, (2) hybrid retrieval of the top 6,
  (3) Context Sufficiency Grader emitting structured JSON
  (`is_sufficient`, `missing_symbols_or_files`, `refined_query`,
  `reasoning`), (4) autonomous second hops — max 2 — running
  `find_symbol_definition`/file lookups plus a refined hybrid search,
  merging and deduplicating, then re-grading, and (5) grounded
  Markdown answer generation where every claim cites
  `[file_path:start_line-end_line]` (a References block is appended if
  a model answer ever omits citations). The LLM sits behind an
  `LLMBackend` protocol: `GeminiLLM` (Gemini REST, model configurable
  via `GEMINI_MODEL`, default `gemini-3.5-flash`) or a deterministic
  `HeuristicLLM` offline backend for `GEMINI_API_KEY=mock`/empty.
- Endpoint: `POST /api/repos/{repo_id}/query` with
  `{"question", "chat_history"}` returns `{answer, agent_trace,
  sources, hops_used}` — the trace is a step list the UI can render as
  the agent's workflow.

## Phase 2 — GitHub ingestion, embeddings, SSE streaming ✅

- `backend/app/github_client.py` — in-memory GitHub REST client (no
  `git clone`, nothing written to disk): repo metadata + latest commit
  SHA, the whole recursive tree in one call, file filtering (binaries,
  images, lockfiles, `node_modules/`, `dist/`, `.min.js`, >150KB), and
  raw contents fetched concurrently under `asyncio.Semaphore(10)`.
  Every response updates the tracked rate-limit state from
  `X-RateLimit-*` headers; 403/429s retry with exponential backoff +
  jitter, honouring `Retry-After` / the reset window.
- `backend/app/indexer.py` — the pipeline as an async generator of
  progress events, plus `EmbeddingService`: Gemini `gemini-embedding-001`
  over REST (`batchEmbedContents`) when a real `GEMINI_API_KEY` is set,
  and a deterministic local hashing/TF embedding (768-dim, sha256-based,
  L2-normalized) when the key is `"mock"` or empty, so tests run fully
  offline. Chunks are embedded as `context_header + "\n" + content`;
  `search_vector` is populated natively on Postgres
  (`to_tsvector('english', ...)`) and as an equivalent tsvector text
  representation elsewhere. **Cache rule:** if `(owner, name,
  commit_sha)` already exists with `status=ready`, re-indexing is
  skipped and the cached repository is returned.
- Endpoints: `POST /api/repos/index` streams the pipeline as
  Server-Sent Events (`fetching_tree` → `chunking` → `embedding` →
  `complete` with `repo_id`); `GET /api/repos` lists indexed repos;
  `GET /api/repos/{repo_id}/tree` returns per-file chunk counts (flat
  list + nested tree). No background workers — the pipeline runs inside
  the streaming request, for hosts that scale to zero.

Verified live: SSE-indexed the real `octocat/Hello-World` repo
end-to-end (mock embeddings) — metadata → tree → chunks → embeddings →
`complete`, then served it from `GET /api/repos` + `/tree`. Note:
unauthenticated GitHub access is 60 req/hr per IP — set `GITHUB_TOKEN`
for the 5,000/hr quota; the rate limiter surfaces exhaustion as a
terminal `error` SSE event instead of hanging.

## Phase 1 — Backend foundation, Postgres layer, AST chunker ✅

```
.
├── requirements.txt          # backend dependencies (repo root)
├── docker-compose.yml        # local Postgres (pgvector image, optional)
├── .env.example              # copy to .env and fill in
├── conftest.py               # makes `backend.app.*` importable in tests
└── backend/
    ├── __init__.py
    ├── app/
    │   ├── main.py           # FastAPI app, CORS, GET /api/health
    │   ├── config.py         # Pydantic BaseSettings (.env: DATABASE_URL,
    │   │                     #   GEMINI_API_KEY, GITHUB_TOKEN, PORT) +
    │   │                     #   postgres:// → postgresql:// normalization
    │   ├── db.py             # SQLAlchemy 2.0 async + sync engines/sessions
    │   ├── models.py         # Repository, CodeChunk
    │   └── chunker.py        # CodeAwareChunker (AST / structural / docs)
    └── tests/
        ├── test_chunker.py   # Python + TS + Markdown + sub-chunking
        ├── test_api_and_db.py
        ├── test_indexer.py   # mocked-GitHub end-to-end pipeline, cache,
        │                     # rate-limit retry, filters, SSE endpoint
        └── test_agent.py     # RRF exact-symbol ranking, filters, symbol
                              # lookup, second-hop triggering, query endpoint
```

### Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # optional for local dev — defaults exist
docker compose up -d db       # optional — API boots without a DB

uvicorn backend.app.main:app --reload --port 8000
# → GET http://localhost:8000/api/health  {"status":"ok", ...}
# → http://localhost:8000/docs
```

### Test it

```bash
python -m pytest backend/tests/ -q    # 30 passed
```

## Design notes

**Portable hybrid-search schema.** Managed Postgres often denies
superuser extensions, so the schema never requires pgvector:
`CodeChunk.embedding` is a `JSONB` list of floats (migratable to a
native `vector` column later) and `CodeChunk.search_vector` is a core
Postgres `TSVECTOR` with a GIN index. Vector similarity (app-side)
+ `ts_rank` keyword search = hybrid search on any managed instance.

**CodeAwareChunker.** Character chunking breaks code; this chunks on
symbol boundaries:

- Python: real `ast` parsing — intact functions, class skeletons and
  methods, decorators + docstrings preserved, exact line numbers.
- TS/JS, C++, Go, Java: regex signature detection + brace balancing
  that ignores braces in strings/comments, so blocks never split
  mid-scope.
- Markdown/RST: split by heading hierarchy, heading path kept in the
  context header.
- Symbols over 1,500 tokens are sub-chunked on line/block boundaries,
  each part prepended with its `context_header`
  (e.g. `File: src/auth.py | Class: AuthManager | Method: verify_token`).
- Coverage is lossless: top-level code outside symbols is emitted as
  `module_header` chunks instead of being dropped.

## Status

All four planned phases are built: backend foundation + AST chunker
(1), in-memory GitHub ingestion + embeddings + SSE (2), hybrid RRF
retrieval + agentic RAG (3), and the frontend SPA (4). Possible next
steps: deployment packaging, auth, persisted chat sessions, and
streaming (token-level) agent answers.
