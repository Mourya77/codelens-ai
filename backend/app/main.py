"""CodeLens AI — FastAPI entry point.

Phase 1: app + CORS + ``GET /api/health``.
Phase 2: in-memory GitHub ingestion streamed over SSE
(``POST /api/repos/index``), ``GET /api/repos`` and
``GET /api/repos/{repo_id}/tree``. No background workers — the
pipeline runs inside the streaming request, which suits hosts that
scale to zero.

Run from the repo root:

    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .agent import CodebaseAgent, LLMBackend, make_llm_backend
from .config import settings
from .db import get_db, init_db
from .github_client import GitHubClient
from .indexer import EmbeddingService, index_repository
from .models import CodeChunk, Repository
from .retriever import HybridCodeRetriever

logger = logging.getLogger("codelens.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create the database schema on startup.

    A fresh managed Postgres (e.g. a new Render database) starts with no
    tables at all; without this, every DB-backed endpoint 500s while
    /api/health stays green. init_db() is idempotent (create_all), so it
    is safe on every boot. Failures are retried briefly (database still
    provisioning) and then logged loudly without taking the app down —
    DB endpoints will report errors until the database is reachable and
    the service is restarted.
    """
    for attempt in range(1, 4):
        try:
            await asyncio.wait_for(init_db(), timeout=15)
            logger.info("Database schema ready (attempt %d)", attempt)
            break
        except Exception:
            if attempt == 3:
                logger.exception(
                    "init_db failed after 3 attempts; DB-backed endpoints "
                    "will fail until the database is reachable"
                )
            else:
                logger.warning(
                    "init_db attempt %d failed; retrying", attempt, exc_info=True
                )
                await asyncio.sleep(2)
    yield


app = FastAPI(
    title="CodeLens AI",
    description=(
        "Agentic RAG GitHub Repository & Code Documentation Analyzer. "
        "Phase 3: hybrid RRF retrieval + multi-step agentic RAG "
        "(analyze → retrieve → grade → second hop → cited answer). "
        "Phase 4: serves the built React frontend from the same origin."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- dependencies (overridable in tests) ------------------------------------


def get_github_client() -> GitHubClient:
    return GitHubClient(token=settings.github_token)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_llm_backend() -> LLMBackend:
    # Gemini with a real key; deterministic offline backend for
    # GEMINI_API_KEY="mock"/empty (see agent.make_llm_backend).
    return make_llm_backend()


# --- schemas ------------------------------------------------------------------


class IndexRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="github.com URL or 'owner/repo'",
        examples=["https://github.com/octocat/Hello-World"],
    )


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)


def _repo_summary(repo: Repository) -> dict:
    return {
        "id": str(repo.id),
        "repo_url": repo.repo_url,
        "owner": repo.owner,
        "name": repo.name,
        "default_branch": repo.default_branch,
        "commit_sha": repo.commit_sha,
        "status": repo.status.value if hasattr(repo.status, "value") else repo.status,
        "total_files": repo.total_files,
        "total_chunks": repo.total_chunks,
        "created_at": repo.created_at.isoformat() if repo.created_at else None,
    }


async def _get_repo_or_404(session: AsyncSession, repo_id: str) -> Repository:
    try:
        parsed = uuid.UUID(repo_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid repo_id: {repo_id!r}")
    repo = (
        await session.execute(select(Repository).where(Repository.id == parsed))
    ).scalars().first()
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


# --- endpoints ------------------------------------------------------------------


@app.get("/api/health", tags=["health"])
async def health() -> dict:
    """Liveness probe. Deliberately does not touch the database."""
    return {"status": "ok", "service": "CodeLens AI", "version": app.version}


@app.post("/api/repos/index", tags=["repos"])
async def index_repo(
    payload: IndexRequest,
    session: AsyncSession = Depends(get_db),
    github_client: GitHubClient = Depends(get_github_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> StreamingResponse:
    """Index a GitHub repository, streaming progress as SSE.

    Each event is one ``data:`` line carrying a JSON object:
    ``{"stage": ..., "progress": 0-100, "message": ...}``, ending with
    ``{"stage": "complete", "progress": 100, "repo_id": "..."}`` (or a
    terminal ``"error"`` event). If the repo's current commit is
    already indexed, the stream completes immediately with the cached
    repository.
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        async for event in index_repository(
            payload.repo_url,
            session,
            github_client=github_client,
            embedding_service=embedding_service,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering so events arrive live.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/repos/{repo_id}/query", tags=["repos"])
async def query_repo(
    repo_id: str,
    payload: QueryRequest,
    session: AsyncSession = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    llm: LLMBackend = Depends(get_llm_backend),
) -> dict:
    """Ask a question about an indexed repository.

    Runs the full agentic loop (analyze → hybrid retrieval → grader →
    optional second hops → grounded answer) and returns the answer,
    the agent's step-by-step trace for the UI, and the source chunks
    with their RRF scores.
    """
    repo = await _get_repo_or_404(session, repo_id)
    retriever = HybridCodeRetriever(
        session, repo.id, embedding_service=embedding_service
    )
    agent = CodebaseAgent(retriever, llm)
    result = await agent.answer_question(
        payload.question,
        chat_history=[m.model_dump() for m in payload.chat_history],
    )
    return result.to_dict()


@app.get("/api/repos/{repo_id}/chunks", tags=["repos"])
async def get_repo_file_chunks(
    repo_id: str,
    file_path: str,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """All chunks of one file (sidebar preview / citation inspector)."""
    repo = await _get_repo_or_404(session, repo_id)
    retriever = HybridCodeRetriever(session, repo.id)
    entries = await retriever.get_file_chunks(file_path)
    chunks = []
    for entry in entries:
        c = entry.chunk
        chunks.append(
            {
                "chunk_id": str(c.id),
                "file_path": c.file_path,
                "language": c.language,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
                "context_header": c.context_header,
                "imports": c.imports,
            }
        )
    return {"repo_id": str(repo.id), "file_path": file_path, "chunks": chunks}


@app.get("/api/repos", tags=["repos"])
async def list_repos(session: AsyncSession = Depends(get_db)) -> list[dict]:
    """All indexed repositories, newest first."""
    stmt = select(Repository).order_by(Repository.created_at.desc())
    repos = (await session.execute(stmt)).scalars().all()
    return [_repo_summary(r) for r in repos]


@app.get("/api/repos/{repo_id}", tags=["repos"])
async def get_repo(
    repo_id: str, session: AsyncSession = Depends(get_db)
) -> dict:
    repo = await _get_repo_or_404(session, repo_id)
    return _repo_summary(repo)


@app.get("/api/repos/{repo_id}/tree", tags=["repos"])
async def get_repo_tree(
    repo_id: str, session: AsyncSession = Depends(get_db)
) -> dict:
    """The indexed file tree: chunk counts per file (flat list), plus
    the same data as a nested tree for UI rendering."""
    repo = await _get_repo_or_404(session, repo_id)
    stmt = (
        select(
            CodeChunk.file_path,
            CodeChunk.language,
            func.count(CodeChunk.id).label("chunks"),
        )
        .where(CodeChunk.repo_id == repo.id)
        .group_by(CodeChunk.file_path, CodeChunk.language)
        .order_by(CodeChunk.file_path)
    )
    rows = (await session.execute(stmt)).all()
    files = [
        {"file_path": path, "language": language, "chunks": count}
        for path, language, count in rows
    ]

    # Nested tree: {"src": {"auth.py": {file node}}, "README.md": {...}}
    tree: dict = {}
    for item in files:
        parts = item["file_path"].split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {
            "file_path": item["file_path"],
            "language": item["language"],
            "chunks": item["chunks"],
        }

    return {
        "repo_id": str(repo.id),
        "owner": repo.owner,
        "name": repo.name,
        "commit_sha": repo.commit_sha,
        "total_files": repo.total_files,
        "total_chunks": repo.total_chunks,
        "files": files,
        "tree": tree,
    }


# --- production: serve the built frontend from the same origin --------------
# Registered LAST so every /api route above wins. When frontend/dist
# doesn't exist (backend-only dev), this is skipped and Vite serves
# the UI with its /api proxy instead.
from pathlib import Path as _Path  # noqa: E402

from fastapi.staticfiles import StaticFiles as _StaticFiles  # noqa: E402

_FRONTEND_DIST = _Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        _StaticFiles(directory=str(_FRONTEND_DIST), html=True),
        name="frontend",
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=settings.port)
