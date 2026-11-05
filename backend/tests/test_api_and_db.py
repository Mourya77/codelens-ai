"""Phase 1 smoke tests: health endpoint, URL normalization, DB schema."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.config import normalize_postgres_url
from backend.app.db import Base
from backend.app.main import app
from backend.app.models import CodeChunk, Repository, RepositoryStatus


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "CodeLens AI"


def test_postgres_url_normalization():
    assert (
        normalize_postgres_url("postgres://u:p@host/db", async_driver=True)
        == "postgresql+asyncpg://u:p@host/db"
    )
    assert (
        normalize_postgres_url("postgres://u:p@host/db", async_driver=False)
        == "postgresql+psycopg2://u:p@host/db"
    )
    assert (
        normalize_postgres_url("postgresql://u:p@host/db", async_driver=True)
        == "postgresql+asyncpg://u:p@host/db"
    )
    # Already-driver-qualified URLs pass through untouched.
    assert (
        normalize_postgres_url("postgresql+asyncpg://u:p@host/db", async_driver=True)
        == "postgresql+asyncpg://u:p@host/db"
    )


def test_models_round_trip_on_sqlite():
    """Schema must create and round-trip without any Postgres extension
    (the JSONB/TSVECTOR-with-variants portability contract)."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        repo = Repository(
            repo_url="https://github.com/example/demo",
            owner="example",
            name="demo",
            default_branch="main",
            status=RepositoryStatus.indexing,
        )
        session.add(repo)
        session.flush()
        chunk = CodeChunk(
            repo_id=repo.id,
            file_path="src/auth.py",
            language="python",
            symbol_name="verify_token",
            symbol_type="method",
            start_line=22,
            end_line=24,
            content="async def verify_token(...): ...",
            context_header="File: src/auth.py | Class: AuthManager | Method: verify_token",
            imports=["os", "hashlib"],
            embedding=[0.1, 0.2, 0.3],
        )
        session.add(chunk)
        session.commit()

    with Session() as session:
        saved = session.query(CodeChunk).one()
        assert saved.embedding == [0.1, 0.2, 0.3]
        assert saved.imports == ["os", "hashlib"]
        assert saved.repository.name == "demo"
        assert saved.repository.status is RepositoryStatus.indexing
