"""SQLAlchemy models for CodeLens AI.

Portability note (the reason for the column choices below): the schema
must create cleanly on *standard managed PostgreSQL*, where the app
role usually has no superuser rights and ``CREATE EXTENSION vector``
can fail. So:

* ``embedding`` is a ``JSONB`` column holding a plain list of floats.
  That stores exactly the same data a ``pgvector`` column would for
  Phase 1, works everywhere, and can be migrated to a native
  ``vector`` column later without changing any application code that
  reads/writes lists. (``ARRAY(Float)`` would work too; JSONB was
  chosen because it round-trips identically on SQLite for tests.)
* ``search_vector`` is a native PostgreSQL ``TSVECTOR`` column — core
  Postgres, no extension — powering keyword search, with a GIN index.
  Hybrid search = vector similarity in app code over ``embedding``
  fused with ``ts_rank`` over ``search_vector``.

Both types fall back to portable variants on SQLite so the schema can
be created in tests.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# JSONB on Postgres, plain JSON elsewhere (SQLite tests).
JSONBType = JSONB().with_variant(JSON(), "sqlite")
# TSVECTOR on Postgres, TEXT elsewhere (SQLite has no tsvector).
TSVectorType = TSVECTOR().with_variant(Text(), "sqlite")
# Generic Uuid: native PG UUID on Postgres, CHAR-backed on SQLite,
# converting uuid objects both ways on every dialect.
UUIDType = Uuid()


class RepositoryStatus(str, enum.Enum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"


class SymbolType(str, enum.Enum):
    function = "function"
    class_ = "class"
    method = "method"
    module_header = "module_header"
    doc = "doc"


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, primary_key=True, default=uuid.uuid4
    )
    repo_url: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    owner: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), nullable=False, default="main")
    commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[RepositoryStatus] = mapped_column(
        Enum(RepositoryStatus, name="repository_status"),
        nullable=False,
        default=RepositoryStatus.pending,
        index=True,
    )
    total_files: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    chunks: Mapped[list["CodeChunk"]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Repository {self.owner}/{self.name} status={self.status}>"


class CodeChunk(Base):
    __tablename__ = "code_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, primary_key=True, default=uuid.uuid4
    )
    repo_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Stored as a plain string (values: function, class, method,
    # module_header, doc) to keep the column tolerant of new chunk
    # kinds the chunker may emit in later phases.
    symbol_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_header: Mapped[str] = mapped_column(String(1024), nullable=False)
    # JSON list of module names imported by the file this chunk came from.
    imports: Mapped[list] = mapped_column(JSONBType, nullable=False, default=list)
    # Embedding vector as a JSON list of floats — see module docstring.
    # (ARRAY(Float) is the documented alternative on Postgres-only
    # deployments; JSONB keeps one code path for every provider.)
    embedding: Mapped[list[float] | None] = mapped_column(JSONBType, nullable=True)
    # Keyword-search vector. On Postgres this is populated by the
    # ingestion pipeline with:
    #   func.to_tsvector("english", chunk.content)
    search_vector: Mapped[str | None] = mapped_column(TSVectorType, nullable=True)

    repository: Mapped[Repository] = relationship(back_populates="chunks")

    __table_args__ = (
        # GIN index for fast @@ / ts_rank queries on Postgres. SQLite
        # ignores index dialects it cannot use via the postgresql_using
        # kwarg being Postgres-scoped in practice; kept explicit here.
        Index(
            "ix_code_chunks_search_vector_gin",
            "search_vector",
            postgresql_using="gin",
        ),
        Index("ix_code_chunks_repo_file", "repo_id", "file_path"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<CodeChunk {self.file_path}:{self.start_line}-{self.end_line} "
            f"{self.symbol_type}:{self.symbol_name}>"
        )


# Re-exported so ingestion code can type embedding payloads plainly.
Embedding = list[float]
_ = Float  # kept imported: ARRAY(Float) alternative documented above
