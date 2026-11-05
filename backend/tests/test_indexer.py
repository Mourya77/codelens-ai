"""Phase 2 tests: GitHub ingestion pipeline end-to-end, offline.

The GitHub REST API is mocked with ``httpx.MockTransport`` (metadata,
latest-commit, recursive tree, and raw contents endpoints), the
embedding service runs in its deterministic local "mock" mode, and the
database is a throwaway SQLite file driven through the real async
SQLAlchemy layer — so the whole pipeline (fetch → filter → chunk →
embed → store) is exercised exactly as production runs it.
"""

import asyncio
import json
import math
import urllib.parse

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db import Base
from backend.app.github_client import (
    GitHubAPIError,
    GitHubClient,
    TreeEntry,
    is_indexable_file,
    parse_repo_url,
)
from backend.app.indexer import (
    EMBEDDING_DIM,
    EmbeddingService,
    build_search_vector_text,
    index_repository,
    local_embedding,
)
from backend.app.models import CodeChunk, Repository, RepositoryStatus

SHA = "abc123def4567890"

PY_SOURCE = "\n".join(
    [
        "import hashlib",                                            # 1
        "",                                                          # 2
        "def hash_password(password: str) -> str:",                  # 3
        "    return hashlib.sha256(password.encode()).hexdigest()",  # 4
        "",                                                          # 5
        "class TokenService:",                                       # 6
        "    def verify_token(self, token: str) -> bool:",           # 7
        "        return token == 'secret'",                          # 8
    ]
)

TS_SOURCE = "\n".join(
    [
        "export function greet(name: string): string {",  # 1
        "  return `Hello, ${name}!`;",                    # 2
        "}",                                              # 3
    ]
)

MD_SOURCE = "# Demo Guide\n\nSome documentation text.\n"

CONTENTS = {
    "src/auth.py": PY_SOURCE,
    "src/app.ts": TS_SOURCE,
    "README.md": MD_SOURCE,
}

TREE = [
    {"path": "src", "type": "tree"},
    {"path": "src/auth.py", "type": "blob", "size": 300, "sha": "a1"},
    {"path": "src/app.ts", "type": "blob", "size": 120, "sha": "a2"},
    {"path": "README.md", "type": "blob", "size": 80, "sha": "a3"},
    # Everything below must be filtered out by the pipeline.
    {"path": "package-lock.json", "type": "blob", "size": 5000, "sha": "b1"},
    {"path": "yarn.lock", "type": "blob", "size": 4000, "sha": "b2"},
    {"path": "assets/logo.png", "type": "blob", "size": 2000, "sha": "b3"},
    {"path": "assets/icon.svg", "type": "blob", "size": 900, "sha": "b4"},
    {"path": "node_modules/lib/index.js", "type": "blob", "size": 100, "sha": "b5"},
    {"path": "dist/bundle.js", "type": "blob", "size": 100, "sha": "b6"},
    {"path": "src/huge.py", "type": "blob", "size": 200_000, "sha": "b7"},
    {"path": "src/app.min.js", "type": "blob", "size": 100, "sha": "b8"},
]

RATE_HEADERS = {
    "X-RateLimit-Limit": "5000",
    "X-RateLimit-Remaining": "4999",
    "X-RateLimit-Reset": "1899999999",
}


class CallCounter:
    def __init__(self):
        self.tree = 0
        self.contents = 0
        self.metadata = 0


def make_handler(counter: CallCounter):
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/octocat/demo":
            counter.metadata += 1
            return httpx.Response(
                200,
                json={
                    "name": "demo",
                    "full_name": "octocat/demo",
                    "html_url": "https://github.com/octocat/demo",
                    "default_branch": "main",
                    "owner": {"login": "octocat"},
                    "description": "Demo repo",
                },
                headers=RATE_HEADERS,
            )
        if path == "/repos/octocat/demo/commits/main":
            return httpx.Response(200, json={"sha": SHA}, headers=RATE_HEADERS)
        if path == f"/repos/octocat/demo/git/trees/{SHA}":
            counter.tree += 1
            assert request.url.params.get("recursive") == "1"
            return httpx.Response(
                200, json={"sha": SHA, "tree": TREE}, headers=RATE_HEADERS
            )
        prefix = "/repos/octocat/demo/contents/"
        if path.startswith(prefix):
            counter.contents += 1
            file_path = urllib.parse.unquote(path[len(prefix):])
            if file_path not in CONTENTS:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(
                200, text=CONTENTS[file_path], headers=RATE_HEADERS
            )
        return httpx.Response(404, json={"message": f"unmocked: {path}"})

    return handler


def make_github_client(counter: CallCounter) -> GitHubClient:
    transport = httpx.MockTransport(make_handler(counter))
    return GitHubClient(
        token="test-token",
        client=httpx.AsyncClient(transport=transport),
        sleep=lambda delay: asyncio.sleep(0),
    )


async def _setup_engine(db_url: str):
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def _run_pipeline(db_url: str, counter: CallCounter) -> tuple[list[dict], dict]:
    engine = await _setup_engine(db_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    gh = make_github_client(counter)
    embedder = EmbeddingService(api_key="mock", batch_size=2)
    async with Session() as session:
        events = [
            event
            async for event in index_repository(
                "https://github.com/octocat/demo",
                session,
                github_client=gh,
                embedding_service=embedder,
            )
        ]
    # Snapshot the stored state as plain data.
    async with Session() as session:
        repo = (await session.execute(select(Repository))).scalars().one()
        chunks = (
            await session.execute(
                select(CodeChunk).order_by(CodeChunk.file_path, CodeChunk.start_line)
            )
        ).scalars().all()
        snapshot = {
            "repo": {
                "id": str(repo.id),
                "owner": repo.owner,
                "name": repo.name,
                "commit_sha": repo.commit_sha,
                "status": repo.status,
                "total_files": repo.total_files,
                "total_chunks": repo.total_chunks,
            },
            "chunks": [
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "symbol_name": c.symbol_name,
                    "symbol_type": c.symbol_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content,
                    "context_header": c.context_header,
                    "imports": c.imports,
                    "embedding": c.embedding,
                    "search_vector": c.search_vector,
                }
                for c in chunks
            ],
        }
    await engine.dispose()
    return events, snapshot


# --- end-to-end ---------------------------------------------------------------

def test_pipeline_end_to_end(tmp_path):
    counter = CallCounter()
    db_url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    events, snap = asyncio.run(_run_pipeline(db_url, counter))

    stages = [e["stage"] for e in events]
    assert stages[0] == "fetching_metadata"
    assert "fetching_tree" in stages
    assert "chunking" in stages
    assert "embedding" in stages
    final = events[-1]
    assert final["stage"] == "complete"
    assert final["progress"] == 100
    assert final["repo_id"] == snap["repo"]["id"]
    assert final["cached"] is False

    tree_event = next(e for e in events if e["stage"] == "fetching_tree")
    assert tree_event["message"] == "Found 3 source files in octocat/demo..."

    repo = snap["repo"]
    assert repo["status"] is RepositoryStatus.ready
    assert repo["commit_sha"] == SHA
    # Only the 3 indexable files made it — every filtered entry stayed out.
    assert repo["total_files"] == 3
    assert counter.tree == 1
    assert counter.contents == 3

    chunks = snap["chunks"]
    assert repo["total_chunks"] == len(chunks) > 0
    files_indexed = {c["file_path"] for c in chunks}
    assert files_indexed == {"src/auth.py", "src/app.ts", "README.md"}

    # Line numbers + context headers survived the whole pipeline.
    verify = next(c for c in chunks if c["symbol_name"] == "verify_token")
    assert verify["symbol_type"] == "method"
    assert (verify["start_line"], verify["end_line"]) == (7, 8)
    assert (
        verify["context_header"]
        == "File: src/auth.py | Class: TokenService | Method: verify_token"
    )
    assert verify["imports"] == ["hashlib"]

    greet = next(c for c in chunks if c["symbol_name"] == "greet")
    assert greet["language"] == "typescript"
    assert (greet["start_line"], greet["end_line"]) == (1, 3)

    # Every chunk carries a stored embedding + keyword tokens.
    for chunk in chunks:
        emb = chunk["embedding"]
        assert emb is not None and len(emb) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in emb)
        assert chunk["search_vector"]
    assert "verify" in verify["search_vector"]
    assert "token" in verify["search_vector"]


def test_pipeline_cache_hit_skips_reindexing(tmp_path):
    counter = CallCounter()
    db_url = f"sqlite+aiosqlite:///{tmp_path}/cache.db"
    events1, snap1 = asyncio.run(_run_pipeline(db_url, counter))
    assert events1[-1]["stage"] == "complete"
    tree_calls_after_first = counter.tree
    content_calls_after_first = counter.contents

    events2, snap2 = asyncio.run(_run_pipeline(db_url, counter))
    final = events2[-1]
    assert final["stage"] == "complete"
    assert final["cached"] is True
    assert final["repo_id"] == snap1["repo"]["id"]
    # The tree and file contents were NOT fetched again, and no
    # duplicate chunks were written.
    assert counter.tree == tree_calls_after_first
    assert counter.contents == content_calls_after_first
    assert len(snap2["chunks"]) == len(snap1["chunks"])


# --- GitHub client units -----------------------------------------------------

def test_parse_repo_url_forms():
    assert parse_repo_url("https://github.com/octocat/demo") == ("octocat", "demo")
    assert parse_repo_url("https://github.com/octocat/demo.git") == ("octocat", "demo")
    assert parse_repo_url("git@github.com:octocat/demo.git") == ("octocat", "demo")
    assert parse_repo_url("octocat/demo") == ("octocat", "demo")
    with pytest.raises(ValueError):
        parse_repo_url("https://github.com/octocat")


def test_file_filter():
    def entry(path, size=100):
        return TreeEntry(path=path, size=size)

    assert is_indexable_file(entry("src/main.py"))
    assert is_indexable_file(entry("README.md"))
    assert not is_indexable_file(entry("assets/logo.png"))
    assert not is_indexable_file(entry("package-lock.json"))
    assert not is_indexable_file(entry("yarn.lock"))
    assert not is_indexable_file(entry("poetry.lock"))
    assert not is_indexable_file(entry("node_modules/pkg/index.js"))
    assert not is_indexable_file(entry("dist/bundle.js"))
    assert not is_indexable_file(entry("src/vendor.min.js"))
    assert not is_indexable_file(entry("src/big.py", size=151 * 1024))
    assert is_indexable_file(entry("src/big.py", size=149 * 1024))


def test_rate_limit_retry_with_backoff():
    delays: list[float] = []
    attempts = {"metadata": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/octocat/demo":
            attempts["metadata"] += 1
            if attempts["metadata"] == 1:
                return httpx.Response(
                    429,
                    json={"message": "API rate limit exceeded"},
                    headers={
                        "X-RateLimit-Limit": "5000",
                        "X-RateLimit-Remaining": "10",
                        "X-RateLimit-Reset": "1899999999",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "name": "demo",
                    "full_name": "octocat/demo",
                    "html_url": "https://github.com/octocat/demo",
                    "default_branch": "main",
                    "owner": {"login": "octocat"},
                },
                headers=RATE_HEADERS,
            )
        if path == "/repos/octocat/demo/commits/main":
            return httpx.Response(200, json={"sha": SHA}, headers=RATE_HEADERS)
        return httpx.Response(404, json={"message": "nope"})

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def run():
        gh = GitHubClient(
            token="t",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            backoff_base_seconds=2.0,
            jitter=lambda ceiling: ceiling,  # deterministic backoff
            sleep=fake_sleep,
        )
        return await gh.get_repo_metadata("octocat", "demo"), gh

    metadata, gh = asyncio.run(run())
    assert metadata.commit_sha == SHA
    assert metadata.default_branch == "main"
    assert attempts["metadata"] == 2  # 429 then success
    assert delays == [2.0]  # base * 2^0, jitter = identity
    # Rate-limit state was tracked from response headers.
    assert gh.rate_limit.remaining == 4999
    assert gh.rate_limit.limit == 5000


def test_github_404_raises_api_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    async def run():
        gh = GitHubClient(
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            sleep=lambda delay: asyncio.sleep(0),
        )
        with pytest.raises(GitHubAPIError) as excinfo:
            await gh.get_repo_metadata("octocat", "missing")
        assert excinfo.value.status_code == 404

    asyncio.run(run())


def test_default_client_construction_survives_bracketed_no_proxy(monkeypatch):
    """Regression: httpx trust_env parsing crashes on NO_PROXY entries
    like "[::1]" (common in container runtimes). The default client
    must construct cleanly regardless."""
    monkeypatch.setenv("NO_PROXY", "localhost,[::1]")
    monkeypatch.setenv("no_proxy", "localhost,[::1]")

    async def run():
        gh = GitHubClient(token="t")
        await gh.aclose()

    asyncio.run(run())


# --- embedding fallback units --------------------------------------------------

def test_local_embedding_is_deterministic_and_normalized():
    text = "File: src/auth.py | Class: TokenService | Method: verify_token\ncode"
    v1 = local_embedding(text)
    v2 = local_embedding(text)
    assert v1 == v2  # deterministic across calls/processes
    assert len(v1) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-6)
    assert local_embedding("completely different content here") != v1


def test_mock_embedding_service_batches():
    async def run():
        service = EmbeddingService(api_key="mock", batch_size=2)
        seen: list[tuple[int, int]] = []
        out = await service.embed_texts(
            ["one", "two", "three"],
            on_batch=lambda i, total: seen.append((i, total)),
        )
        return out, seen, service

    out, seen, service = asyncio.run(run())
    assert service.use_local_fallback is True
    assert service.provider_name == "local-hash"
    assert len(out) == 3 and all(len(v) == EMBEDDING_DIM for v in out)
    assert seen == [(1, 2), (2, 2)]  # batch progress callbacks


def test_search_vector_text_format():
    sv = build_search_vector_text("verify_token checks the token twice")
    assert "'verify'" in sv
    assert "'token'" in sv
    assert "'checks'" in sv
    # 'the' is a stopword and must not appear as a term.
    assert "'the'" not in sv


# --- SSE endpoint ----------------------------------------------------------------

def test_sse_index_endpoint_streams_events(tmp_path):
    from fastapi.testclient import TestClient

    from backend.app import main as main_module

    db_url = f"sqlite+aiosqlite:///{tmp_path}/sse.db"
    counter = CallCounter()

    async def override_get_db():
        engine = await _setup_engine(db_url)
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with Session() as session:
            yield session
        await engine.dispose()

    def override_github_client():
        return make_github_client(counter)

    def override_embedding_service():
        return EmbeddingService(api_key="mock", batch_size=2)

    main_module.app.dependency_overrides[main_module.get_db] = override_get_db
    main_module.app.dependency_overrides[
        main_module.get_github_client
    ] = override_github_client
    main_module.app.dependency_overrides[
        main_module.get_embedding_service
    ] = override_embedding_service
    try:
        with TestClient(main_module.app) as client:
            resp = client.post(
                "/api/repos/index", json={"repo_url": "octocat/demo"}
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            events = [
                json.loads(line[len("data: "):])
                for line in resp.text.splitlines()
                if line.startswith("data: ")
            ]
            assert events[-1]["stage"] == "complete"
            assert events[-1]["progress"] == 100
            repo_id = events[-1]["repo_id"]

            listing = client.get("/api/repos")
            assert listing.status_code == 200
            repos = listing.json()
            assert len(repos) == 1
            assert repos[0]["id"] == repo_id
            assert repos[0]["status"] == "ready"
            assert repos[0]["total_files"] == 3

            tree = client.get(f"/api/repos/{repo_id}/tree")
            assert tree.status_code == 200
            body = tree.json()
            assert body["total_chunks"] == repos[0]["total_chunks"]
            per_file = {f["file_path"]: f["chunks"] for f in body["files"]}
            assert set(per_file) == {"src/auth.py", "src/app.ts", "README.md"}
            assert per_file["src/auth.py"] >= 3
            assert body["tree"]["src"]["auth.py"]["chunks"] == per_file["src/auth.py"]

            missing = client.get("/api/repos/00000000-0000-0000-0000-000000000000/tree")
            assert missing.status_code == 404
    finally:
        main_module.app.dependency_overrides.clear()
