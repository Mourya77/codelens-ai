"""SQLAlchemy 2.0 database setup for CodeLens AI.

Both an async engine (used by FastAPI request handlers) and a sync
engine (used by scripts, Alembic-style migrations and one-off jobs) are
provided, both built from the same ``DATABASE_URL``. URL scheme
normalization (``postgres://`` vs ``postgresql://``, driver selection)
lives in :mod:`app.config` so engines never fail on a provider URL.

Engines are lazy: creating one does not open a connection, so importing
this module (and booting the API / running the chunker tests) works
with no database running.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    """Declarative base shared by every model in :mod:`app.models`."""


def _engine_kwargs(url: str) -> dict:
    # SQLite (tests / throwaway local runs) neither needs nor accepts
    # pool sizing arguments.
    if url.startswith("sqlite"):
        return {}
    return {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 10}


# --- Async engine (FastAPI path) -------------------------------------------
async_engine = create_async_engine(
    settings.async_database_url, **_engine_kwargs(settings.async_database_url)
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

# --- Sync engine (scripts / migrations / jobs) ------------------------------
sync_engine = create_engine(
    settings.sync_database_url, **_engine_kwargs(settings.sync_database_url)
)
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a scoped async session."""
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_db() -> Generator[Session, None, None]:
    """Sync counterpart of :func:`get_db` for non-async callers."""
    with SyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables (dev convenience; use migrations in prod).

    Safe on any managed Postgres: the schema in :mod:`app.models` uses
    only JSONB + TSVECTOR, so it needs no superuser extensions.
    """
    # Imported here so model registration happens before create_all.
    from . import models  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
