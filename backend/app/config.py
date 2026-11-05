"""Application settings for CodeLens AI.

Reads configuration from environment variables / a local ``.env`` file
(see ``.env.example`` at the repo root). Everything has a local-dev
default so the app and the test suite boot with no ``.env`` present;
production deployments are expected to set real values.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def env_proxy_url() -> str | None:
    """The egress proxy for outbound HTTPS calls, if the environment
    defines one (``HTTPS_PROXY`` / ``ALL_PROXY``).

    Read explicitly instead of using httpx ``trust_env``: httpx's
    env parsing crashes on bracketed IPv6 ``NO_PROXY`` entries such as
    ``[::1]``, which are common in container runtimes. Never logged —
    proxy URLs may embed credentials.
    """
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_postgres_url(url: str, *, async_driver: bool) -> str:
    """Normalize a Postgres URL to a SQLAlchemy 2.0 driver URL.

    Managed providers (Heroku, Railway, Render, Neon dashboard snippets,
    ...) variously hand out ``postgres://`` and ``postgresql://`` URLs,
    and some append ``+asyncpg`` / ``+psycopg2`` themselves. SQLAlchemy
    rejects the bare ``postgres://`` scheme, so rewrite it here — once —
    instead of failing at engine creation.

    Non-Postgres URLs (e.g. ``sqlite+aiosqlite://`` in tests) pass
    through untouched.
    """
    if url.startswith("sqlite"):
        return url

    # Bare legacy scheme -> canonical scheme.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    if url.startswith("postgresql://"):
        driver = "asyncpg" if async_driver else "psycopg2"
        return "postgresql+" + driver + "://" + url[len("postgresql://") :]

    # Already carries an explicit driver (postgresql+asyncpg://, ...).
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local-dev default only; set DATABASE_URL in .env / the environment
    # for anything real.
    database_url: str = "postgresql://postgres:postgres@localhost:5432/codelens"
    gemini_api_key: str = ""
    github_token: str = ""
    port: int = 8000
    # Phase 3: model for the agentic RAG loop (analyzer/grader/answer).
    # Pinned after live key tests on 2026-11-05: gemini-2.5-flash is
    # closed to new API projects (404) and the "gemini-flash-latest"
    # alias returned persistent 503 high-demand errors, while
    # gemini-3.5-flash answered reliably. Override via GEMINI_MODEL.
    # (Updated again the same day: the free-tier daily quota for
    # gemini-3.5-flash / 3.6-flash was exhausted by live traffic —
    # 429s on the grader call — while gemini-3.1-flash-lite, also
    # live-verified for generation and JSON mode, still had quota.
    # Default therefore pins the lite model; set GEMINI_MODEL to a
    # full Flash model when the project has billing / higher quota.)
    gemini_model: str = "gemini-3.1-flash-lite"
    # Embedding model. text-embedding-004 is likewise closed to new
    # projects; gemini-embedding-001 is requested at 768 dimensions
    # (outputDimensionality) to match the local fallback's shape.
    gemini_embedding_model: str = "gemini-embedding-001"

    # Comma-friendly in a follow-up phase; Phase 1 keeps dev origins open.
    cors_origins: list[str] = [
        "http://localhost:5173",  # Vite dev server (Phase 2 frontend)
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @property
    def async_database_url(self) -> str:
        return normalize_postgres_url(self.database_url, async_driver=True)

    @property
    def sync_database_url(self) -> str:
        return normalize_postgres_url(self.database_url, async_driver=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
