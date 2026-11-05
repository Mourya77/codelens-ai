"""Embedding & hybrid indexing service + the ingestion pipeline.

Pipeline (all in-memory, no disk, no background workers — it runs
inside the request that streams it):

    metadata ─► cache check ─► file tree ─► raw files ─► AST chunking
             ─► embeddings (Gemini, or deterministic local fallback)
             ─► CodeChunk rows with `embedding` + `search_vector`

``index_repository`` is an async generator yielding progress events;
the SSE endpoint in ``main.py`` serializes them verbatim, and tests
consume them directly.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncGenerator, Awaitable, Callable

import httpx
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .chunker import Chunk, CodeAwareChunker
from .config import settings
from .github_client import GitHubClient, parse_repo_url
from .models import CodeChunk, Repository, RepositoryStatus

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # requested output dimensionality (matches local fallback)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

ProgressCallback = Callable[[int, int], Awaitable[None] | None]

# ---------------------------------------------------------------------------
# Local deterministic fallback embeddings (offline / GEMINI_API_KEY="mock")
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> list[str]:
    """Tokenize code/text for hashing: split identifiers on camelCase
    and snake_case so `verifyToken` and `verify_token` share tokens."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for piece in raw.split("_"):
            for sub in _CAMEL_RE.split(piece):
                token = sub.lower()
                if token:
                    tokens.append(token)
    return tokens


def local_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic hashing-vectorizer embedding.

    TF-weighted feature hashing (sha256 — stable across processes,
    unlike ``hash()``): each token hashes to a bucket and a sign, term
    frequencies are log-scaled, and the vector is L2-normalized. Same
    text in → same vector out, always, with no model download and no
    network. This is the offline stand-in for Gemini in tests/dev, not
    a semantic equal — production uses the real API.
    """
    vector = [0.0] * dim
    counts: dict[str, int] = {}
    for token in _tokens(text):
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        # A second bucket per token reduces collision damage.
        bucket2 = int.from_bytes(digest[5:9], "big") % dim
        weight = 1.0 + math.log(count)
        vector[bucket] += sign * weight
        vector[bucket2] += -sign * weight * 0.5
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Keyword (TSVECTOR) payload
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for",
    "on", "as", "are", "was", "with", "by", "at", "from", "that", "this",
}


def build_search_vector_text(text: str, max_terms: int = 512) -> str:
    """A tsvector *text representation* (``'token':1,4 'other':2``).

    PostgreSQL parses this string directly into a real TSVECTOR value
    on insert, so keyword search works natively; on SQLite the same
    string is stored as TEXT, which keeps tests able to assert on the
    tokens. Positions are 1-based, per tsvector convention.
    """
    positions: dict[str, list[int]] = {}
    for idx, token in enumerate(_tokens(text), start=1):
        if token in _STOPWORDS:
            continue
        positions.setdefault(token, []).append(idx)
        if len(positions) > max_terms * 2:  # safety valve for huge chunks
            break
    items = sorted(positions.items(), key=lambda kv: -len(kv[1]))[:max_terms]
    items.sort(key=lambda kv: kv[0])
    parts = []
    for token, pos in items:
        # Cap positions list; tsvector itself caps at 16k positions.
        pos_str = ",".join(str(p) for p in pos[:64])
        parts.append(f"'{token}':{pos_str}")
    return " ".join(parts)


async def _populate_search_vector(
    session: AsyncSession, chunk: CodeChunk, text: str
) -> None:
    """Fill ``chunk.search_vector`` for a just-flushed chunk row.

    On PostgreSQL, prefer the native ``to_tsvector('english', ...)``
    (real stemming + stopwords) via an UPDATE; elsewhere store the
    Python-built tsvector text representation.
    """
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        await session.execute(
            update(CodeChunk)
            .where(CodeChunk.id == chunk.id)
            .values(search_vector=func.to_tsvector("english", text))
        )
        await session.refresh(chunk, attribute_names=["search_vector"])
    else:
        chunk.search_vector = build_search_vector_text(text)


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    pass


class EmbeddingService:
    """Batch text embedding via Gemini, with a local offline fallback.

    The fallback is used when the API key is ``"mock"`` (tests) or
    empty (dev without a key). Any other key goes to the real
    Gemini embedding REST endpoint (``batchEmbedContents``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        batch_size: int = 32,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.model = model or getattr(
            settings, "gemini_embedding_model", EMBEDDING_MODEL
        )
        self.batch_size = max(1, batch_size)
        self._http = http_client
        self.use_local_fallback = (self.api_key or "").strip().lower() in ("", "mock")

    @property
    def provider_name(self) -> str:
        return "local-hash" if self.use_local_fallback else f"gemini:{self.model}"

    async def embed_texts(
        self,
        texts: list[str],
        *,
        on_batch: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving order. ``on_batch``
        (batch_index_1_based, total_batches) fires per completed batch."""
        if not texts:
            return []
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        out: list[list[float]] = []
        for idx, batch in enumerate(batches, start=1):
            if self.use_local_fallback:
                out.extend(local_embedding(t) for t in batch)
            else:
                out.extend(await self._embed_batch_gemini(batch))
            if on_batch is not None:
                result = on_batch(idx, len(batches))
                if result is not None:
                    await result
        return out

    async def _embed_batch_gemini(self, batch: list[str]) -> list[list[float]]:
        url = f"{GEMINI_API_BASE}/models/{self.model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": EMBEDDING_DIM,
                }
                for text in batch
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        if self._http is not None:
            client = self._http
        else:
            import ssl

            from .config import env_proxy_url

            kwargs: dict = {
                "timeout": httpx.Timeout(60.0),
                "trust_env": False,
                # System CA context (SSL_CERT_FILE-aware), as in the
                # GitHub client — not httpx's bundled certifi store.
                "verify": ssl.create_default_context(),
            }
            proxy = env_proxy_url()
            if proxy:
                try:
                    client = httpx.AsyncClient(proxy=proxy, **kwargs)
                except Exception:
                    client = httpx.AsyncClient(**kwargs)
            else:
                client = httpx.AsyncClient(**kwargs)
        try:
            resp = await client.post(url, json=payload, headers=headers)
        finally:
            if self._http is None:
                await client.aclose()
        if resp.status_code >= 400:
            raise EmbeddingError(
                f"Gemini embedding API {resp.status_code}: {resp.text[:300]}"
            )
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Gemini returned {len(embeddings)} embeddings for a "
                f"batch of {len(batch)} texts"
            )
        return [list(map(float, e.get("values", []))) for e in embeddings]


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------


async def find_cached_repository(
    session: AsyncSession, owner: str, name: str, commit_sha: str
) -> Repository | None:
    """Return the Repository row if (owner, name, commit_sha) is
    already indexed and ready — callers then skip re-indexing."""
    if not commit_sha:
        return None
    stmt = select(Repository).where(
        Repository.owner == owner,
        Repository.name == name,
        Repository.commit_sha == commit_sha,
        Repository.status == RepositoryStatus.ready,
    )
    return (await session.execute(stmt)).scalars().first()


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _event(stage: str, progress: int, message: str, **extra) -> dict:
    return {"stage": stage, "progress": progress, "message": message, **extra}


async def index_repository(
    repo_url: str,
    session: AsyncSession,
    *,
    github_client: GitHubClient | None = None,
    embedding_service: EmbeddingService | None = None,
    chunker: CodeAwareChunker | None = None,
) -> AsyncGenerator[dict, None]:
    """Run the full ingestion pipeline, yielding progress events.

    Final success event: ``{"stage": "complete", "progress": 100,
    "repo_id": ...}`` (plus ``"cached": True`` on a cache hit).
    Failures yield a terminal ``{"stage": "error", ...}`` event and
    mark the Repository row ``failed`` when one exists.
    """
    gh = github_client or GitHubClient(token=settings.github_token)
    embedder = embedding_service or EmbeddingService()
    chunker = chunker or CodeAwareChunker()
    repo_row: Repository | None = None

    try:
        owner, name = parse_repo_url(repo_url)
        canonical_url = f"https://github.com/{owner}/{name}"

        # -- metadata ---------------------------------------------------
        metadata = await gh.get_repo_metadata(owner, name)
        yield _event(
            "fetching_metadata",
            5,
            f"Fetched metadata for {metadata.full_name} "
            f"({metadata.default_branch} @ {metadata.commit_sha[:8]})...",
        )

        # -- cache check --------------------------------------------------
        cached = await find_cached_repository(
            session, metadata.owner, metadata.name, metadata.commit_sha
        )
        if cached is not None:
            yield _event(
                "complete",
                100,
                f"{metadata.full_name} @ {metadata.commit_sha[:8]} is already "
                "indexed — returning the cached repository.",
                repo_id=str(cached.id),
                cached=True,
                total_files=cached.total_files,
                total_chunks=cached.total_chunks,
            )
            return

        # -- repo row: create or reuse (re-index of a known URL) ---------
        existing = (
            await session.execute(
                select(Repository).where(Repository.repo_url == canonical_url)
            )
        ).scalars().first()
        if existing is not None:
            repo_row = existing
            await session.execute(
                delete(CodeChunk).where(CodeChunk.repo_id == repo_row.id)
            )
            repo_row.commit_sha = metadata.commit_sha
            repo_row.default_branch = metadata.default_branch
            repo_row.total_files = 0
            repo_row.total_chunks = 0
        else:
            repo_row = Repository(
                repo_url=canonical_url,
                owner=metadata.owner,
                name=metadata.name,
                default_branch=metadata.default_branch,
                commit_sha=metadata.commit_sha,
            )
            session.add(repo_row)
        repo_row.status = RepositoryStatus.indexing
        await session.commit()
        await session.refresh(repo_row)

        # -- file tree -----------------------------------------------------
        entries = await gh.get_file_tree(
            metadata.owner, metadata.name, metadata.commit_sha
        )
        yield _event(
            "fetching_tree",
            10,
            f"Found {len(entries)} source files in {metadata.full_name}...",
            total_files=len(entries),
        )

        # -- raw file contents (concurrent, in memory) ---------------------
        files = await gh.fetch_files(
            metadata.owner, metadata.name, entries, metadata.commit_sha
        )
        yield _event(
            "fetching_files",
            25,
            f"Fetched {len(files)} files into memory...",
            fetched_files=len(files),
        )

        # -- chunking --------------------------------------------------------
        all_chunks: list[Chunk] = []
        for file in files:
            all_chunks.extend(chunker.chunk_file(file.path, file.content))
        yield _event(
            "chunking",
            40,
            f"Extracted {len(all_chunks)} AST code chunks from "
            f"{len(files)} files...",
            total_chunks=len(all_chunks),
        )

        # -- embedding -------------------------------------------------------
        texts = [f"{c.context_header}\n{c.content}" for c in all_chunks]
        batch_events: list[dict] = []

        def _on_batch(batch_idx: int, total_batches: int) -> None:
            progress = 55 + int(35 * batch_idx / max(1, total_batches))
            batch_events.append(
                _event(
                    "embedding",
                    progress,
                    f"Generating embeddings (batch {batch_idx}/{total_batches})...",
                    batch=batch_idx,
                    total_batches=total_batches,
                )
            )

        embeddings = await embedder.embed_texts(texts, on_batch=_on_batch)
        for event in batch_events:
            yield event
        if len(embeddings) != len(all_chunks):
            raise EmbeddingError(
                f"Embedding count mismatch: {len(embeddings)} vectors for "
                f"{len(all_chunks)} chunks"
            )

        # -- storing ---------------------------------------------------------
        yield _event(
            "storing", 95, f"Storing {len(all_chunks)} chunks with embeddings..."
        )
        rows: list[CodeChunk] = []
        for chunk, vector, text in zip(all_chunks, embeddings, texts):
            row = CodeChunk(
                repo_id=repo_row.id,
                file_path=chunk.file_path,
                language=chunk.language,
                symbol_name=chunk.symbol_name,
                symbol_type=chunk.symbol_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                context_header=chunk.context_header,
                imports=chunk.imports,
                embedding=[float(v) for v in vector],
            )
            rows.append(row)
        session.add_all(rows)
        await session.flush()  # assign ids for the search_vector pass
        for row, text in zip(rows, texts):
            await _populate_search_vector(session, row, text)

        repo_row.total_files = len(files)
        repo_row.total_chunks = len(all_chunks)
        repo_row.status = RepositoryStatus.ready
        await session.commit()

        yield _event(
            "complete",
            100,
            f"Indexed {metadata.full_name}: {len(files)} files, "
            f"{len(all_chunks)} chunks.",
            repo_id=str(repo_row.id),
            cached=False,
            total_files=len(files),
            total_chunks=len(all_chunks),
        )

    except Exception as exc:  # the stream must report, not crash
        await session.rollback()
        if repo_row is not None:
            try:
                repo_row.status = RepositoryStatus.failed
                await session.commit()
            except Exception:
                await session.rollback()
        yield _event("error", 0, f"Indexing failed: {exc}")
