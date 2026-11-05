"""Hybrid code retrieval with Reciprocal Rank Fusion (Phase 3).

Two complementary rankers run over a repository's chunks:

* **Semantic** — cosine similarity between the query embedding and
  each chunk's stored embedding. Good for "how is auth handled" style
  questions where the wording differs from the code.
* **Lexical** — an in-Python Okapi BM25 over identifier-aware tokens
  (full identifiers like ``verify_jwt_token`` are kept whole *and*
  split into subwords), plus hard boosts for exact ``symbol_name``
  matches and file-path matches. On PostgreSQL the native
  ``ts_rank_cd`` score over ``search_vector`` is blended in when
  available. This is what makes exact function names and error codes
  findable — pure vector search is notoriously weak at those.

The two ranked lists are fused with Reciprocal Rank Fusion:

    RRF(d) = 1 / (60 + rank_semantic(d)) + 1 / (60 + rank_lexical(d))

(k = 60, the standard constant from the original RRF paper.) A list
only contributes for documents it actually ranked: lexical rank
requires a positive lexical score, semantic rank requires a stored
embedding.

Everything is computed in application code over the repo's chunk rows
(the schema stores embeddings as JSONB for managed-Postgres
portability), which keeps behaviour identical on PostgreSQL and on
SQLite in tests.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .indexer import EmbeddingService
from .models import CodeChunk

RRF_K = 60

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_identifier(identifier: str) -> list[str]:
    parts: list[str] = []
    for piece in identifier.split("_"):
        parts.extend(p.lower() for p in _CAMEL_RE.split(piece) if p)
    return parts


def tokenize_for_index(text_value: str) -> list[str]:
    """Identifier-aware tokens: each identifier contributes its full
    lowercased form plus its snake/camel subwords, so both
    ``verify_jwt_token`` and ``jwt`` match the same chunk."""
    tokens: list[str] = []
    for match in _IDENT_RE.findall(text_value or ""):
        lowered = match.lower()
        tokens.append(lowered)
        if "_" in match or any(c.isupper() for c in match):
            tokens.extend(p for p in _split_identifier(match) if p != lowered)
    return tokens


def query_identifiers(query: str) -> list[str]:
    """Full identifiers mentioned in a query (for exact-symbol boosts)."""
    seen: list[str] = []
    for match in _IDENT_RE.findall(query or ""):
        lowered = match.lower()
        if lowered not in seen:
            seen.append(lowered)
    return seen


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(sum(x * x for x in a[:n]))
    norm_b = math.sqrt(sum(x * x for x in b[:n]))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedChunk:
    """A CodeChunk plus its retrieval scores (API-ready)."""

    chunk: CodeChunk
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    match_type: str | None = None  # set by find_symbol_definition

    # Convenience pass-throughs ------------------------------------------------
    @property
    def id(self) -> uuid.UUID:
        return self.chunk.id

    @property
    def file_path(self) -> str:
        return self.chunk.file_path

    @property
    def symbol_name(self) -> str | None:
        return self.chunk.symbol_name

    def to_source_dict(self) -> dict:
        c = self.chunk
        return {
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
            "semantic_score": round(self.semantic_score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "rrf_score": round(self.rrf_score, 6),
        }


def _chunk_document_text(chunk: CodeChunk) -> str:
    return "\n".join(
        [
            chunk.file_path,
            chunk.symbol_name or "",
            chunk.context_header,
            chunk.content,
            " ".join(chunk.imports or []),
        ]
    )


class HybridCodeRetriever:
    """Hybrid semantic + lexical retrieval for one repository."""

    def __init__(
        self,
        session: AsyncSession,
        repo_id: str | uuid.UUID,
        *,
        embedding_service: EmbeddingService | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self.session = session
        self.repo_id = uuid.UUID(str(repo_id))
        self.embedder = embedding_service or EmbeddingService()
        self.rrf_k = rrf_k

    # -- corpus ------------------------------------------------------------

    async def _load_chunks(
        self,
        *,
        file_path_filter: str | None = None,
        symbol_type_filter: str | None = None,
    ) -> list[CodeChunk]:
        stmt = select(CodeChunk).where(CodeChunk.repo_id == self.repo_id)
        if file_path_filter:
            stmt = stmt.where(CodeChunk.file_path.ilike(f"%{file_path_filter}%"))
        if symbol_type_filter:
            stmt = stmt.where(CodeChunk.symbol_type == symbol_type_filter)
        rows = (await self.session.execute(stmt)).scalars().all()
        return sorted(rows, key=lambda c: (c.file_path, c.start_line))

    # -- lexical (BM25 + boosts, + PG ts_rank when available) ---------------

    def _lexical_scores(
        self, query: str, chunks: list[CodeChunk]
    ) -> dict[uuid.UUID, float]:
        if not chunks:
            return {}
        query_tokens = tokenize_for_index(query)
        if not query_tokens:
            return {c.id: 0.0 for c in chunks}
        query_idents = set(query_identifiers(query))

        docs = {c.id: tokenize_for_index(_chunk_document_text(c)) for c in chunks}
        n_docs = len(docs)
        avg_len = sum(len(d) for d in docs.values()) / max(1, n_docs)
        doc_freq: dict[str, int] = {}
        for tokens in docs.values():
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        k1, b = 1.5, 0.75
        scores: dict[uuid.UUID, float] = {}
        for chunk in chunks:
            tokens = docs[chunk.id]
            if not tokens:
                scores[chunk.id] = 0.0
                continue
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            score = 0.0
            for token in set(query_tokens):
                df = doc_freq.get(token, 0)
                if df == 0 or token not in freq:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = freq[token]
                denom = tf + k1 * (1 - b + b * len(tokens) / max(1.0, avg_len))
                score += idf * (tf * (k1 + 1) / denom)

            # Hard boosts: exact symbols and paths are what lexical
            # search exists for — BM25 alone can bury them.
            symbol = (chunk.symbol_name or "").lower()
            if symbol:
                if symbol in query_idents:
                    score += 12.0  # exact symbol mention in the query
                else:
                    symbol_parts = set(_split_identifier(chunk.symbol_name or ""))
                    if symbol_parts & set(query_tokens):
                        score += 2.0
            path_lower = chunk.file_path.lower()
            for ident in query_idents:
                if len(ident) >= 3 and ident in path_lower:
                    score += 1.5
            scores[chunk.id] = score
        return scores

    async def _postgres_ts_scores(
        self, query: str, chunks: list[CodeChunk]
    ) -> dict[uuid.UUID, float]:
        """Native full-text scores on PostgreSQL; empty elsewhere.
        Blend-in only — any failure (no vectors populated, dialect
        differences) simply skips this signal."""
        try:
            if self.session.get_bind().dialect.name != "postgresql":
                return {}
            ids = [c.id for c in chunks]
            if not ids:
                return {}
            stmt = text(
                "SELECT id, ts_rank_cd(search_vector, "
                "plainto_tsquery('english', :q)) AS rank "
                "FROM code_chunks WHERE id = ANY(:ids) "
                "AND search_vector IS NOT NULL"
            )
            rows = await self.session.execute(
                stmt, {"q": query, "ids": ids}
            )
            return {row.id: float(row.rank or 0.0) for row in rows}
        except Exception:
            return {}

    # -- main search -----------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        file_path_filter: str | None = None,
        symbol_type_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        chunks = await self._load_chunks(
            file_path_filter=file_path_filter,
            symbol_type_filter=symbol_type_filter,
        )
        if not chunks:
            return []
        results = {c.id: RetrievedChunk(chunk=c) for c in chunks}

        # Semantic ranking.
        query_vector = (await self.embedder.embed_texts([query]))[0]
        semantic_sorted = sorted(
            (c for c in chunks if c.embedding),
            key=lambda c: (
                -cosine_similarity(query_vector, c.embedding or []),
                c.file_path,
                c.start_line,
            ),
        )
        for rank, chunk in enumerate(semantic_sorted, start=1):
            entry = results[chunk.id]
            entry.semantic_rank = rank
            entry.semantic_score = cosine_similarity(
                query_vector, chunk.embedding or []
            )

        # Lexical ranking (BM25 + boosts, blended with PG ts_rank).
        lexical = self._lexical_scores(query, chunks)
        ts_scores = await self._postgres_ts_scores(query, chunks)
        if ts_scores:
            for chunk_id, ts_rank in ts_scores.items():
                if chunk_id in lexical:
                    lexical[chunk_id] += 5.0 * ts_rank
        lexical_sorted = sorted(
            (c for c in chunks if lexical.get(c.id, 0.0) > 0),
            key=lambda c: (-lexical[c.id], c.file_path, c.start_line),
        )
        for rank, chunk in enumerate(lexical_sorted, start=1):
            entry = results[chunk.id]
            entry.lexical_rank = rank
            entry.lexical_score = lexical[chunk.id]
        for chunk in chunks:  # keep raw scores visible even when unranked
            if results[chunk.id].lexical_score == 0.0:
                results[chunk.id].lexical_score = lexical.get(chunk.id, 0.0)

        # Reciprocal Rank Fusion.
        for entry in results.values():
            score = 0.0
            if entry.semantic_rank is not None:
                score += 1.0 / (self.rrf_k + entry.semantic_rank)
            if entry.lexical_rank is not None:
                score += 1.0 / (self.rrf_k + entry.lexical_rank)
            entry.rrf_score = score

        fused = sorted(
            results.values(),
            key=lambda r: (
                -r.rrf_score,
                -r.lexical_score,
                -r.semantic_score,
                r.chunk.file_path,
                r.chunk.start_line,
            ),
        )
        return fused[:top_k]

    # -- symbol / file lookups (the agent's second-hop tools) ----------------

    async def find_symbol_definition(
        self, symbol_name: str
    ) -> list[RetrievedChunk]:
        """Exact definition lookup for a symbol, repo-wide.

        Returns definition chunks (function/class/method whose
        ``symbol_name`` matches, including qualified ``Outer.inner``
        forms) first; if nothing defines the symbol, falls back to
        chunks that *import* it — which is often exactly the context a
        second hop needs.
        """
        name = (symbol_name or "").strip()
        if not name:
            return []
        chunks = await self._load_chunks()
        lowered = name.lower()
        short = lowered.split(".")[-1]

        def is_definition(chunk: CodeChunk) -> bool:
            symbol = (chunk.symbol_name or "").lower()
            if not symbol:
                return False
            return (
                symbol == lowered
                or symbol == short
                or symbol.endswith("." + short)
                or lowered.endswith("." + symbol)
            )

        type_priority = {"class": 0, "function": 1, "method": 2}
        definitions = [c for c in chunks if is_definition(c)]
        definitions.sort(
            key=lambda c: (
                type_priority.get(c.symbol_type, 3),
                c.file_path,
                c.start_line,
            )
        )
        if definitions:
            return [
                RetrievedChunk(chunk=c, match_type="definition")
                for c in definitions
            ]

        import_hits = [
            c
            for c in chunks
            if any(
                # Qualified-module matching: exact, parent package, or
                # submodule — deliberately NOT bare last-segment, so
                # looking up "auth.jwt" doesn't match a plain `jwt`
                # library import.
                imp.lower() == lowered
                or lowered.startswith(imp.lower() + ".")
                or imp.lower().startswith(lowered + ".")
                or ("." not in lowered and imp.lower().split(".")[-1] == lowered)
                for imp in (c.imports or [])
            )
        ]
        import_hits.sort(key=lambda c: (c.file_path, c.start_line))
        return [
            RetrievedChunk(chunk=c, match_type="import") for c in import_hits
        ]

    async def get_file_chunks(self, file_path: str) -> list[RetrievedChunk]:
        """All chunks of one file (second-hop tool for file hints)."""
        chunks = await self._load_chunks(file_path_filter=file_path)
        exact = [c for c in chunks if c.file_path == file_path]
        chosen = exact or chunks
        return [RetrievedChunk(chunk=c, match_type="file") for c in chosen]
