"""Phase 3 tests: Hybrid RRF retrieval + the agentic RAG loop.

Corpora are seeded directly as CodeChunk rows (embeddings produced by
the same deterministic local embedder the indexer uses in mock mode),
so retrieval runs against the real async DB layer offline. The agent
tests use a scripted LLM backend implementing the LLMBackend protocol,
which is exactly how the Gemini backend is swapped out in production.
"""

import asyncio
import math

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.agent import (
    AnalysisResult,
    CodebaseAgent,
    GradeResult,
    HeuristicLLM,
)
from backend.app.db import Base
from backend.app.indexer import EmbeddingService, local_embedding
from backend.app.models import CodeChunk, Repository, RepositoryStatus
from backend.app.retriever import HybridCodeRetriever

# ---------------------------------------------------------------------------
# Corpus A — retrieval
# ---------------------------------------------------------------------------

def _chunk(file_path, symbol, symbol_type, start, end, content, imports=None, language="python"):
    header = f"File: {file_path}"
    if symbol_type == "class":
        header += f" | Class: {symbol}"
    elif symbol_type == "method":
        header += f" | Method: {symbol}"
    elif symbol:
        header += f" | Function: {symbol}"
    return {
        "file_path": file_path,
        "language": language,
        "symbol_name": symbol,
        "symbol_type": symbol_type,
        "start_line": start,
        "end_line": end,
        "content": content,
        "context_header": header,
        "imports": imports or [],
    }


CORPUS_A = [
    _chunk(
        "src/auth/jwt.py", "verify_jwt_token", "function", 10, 25,
        "def verify_jwt_token(token: str) -> dict:\n"
        '    """Verify a JWT and return its payload."""\n'
        "    header, payload, signature = split_token(token)\n"
        "    if not check_signature(header, payload, signature):\n"
        "        raise InvalidTokenError('bad signature')\n"
        "    if payload['exp'] < now():\n"
        "        raise InvalidTokenError('token expired')\n"
        "    return payload",
        imports=["jwt", "time"],
    ),
    _chunk(
        "src/auth/session.py", "verify_session_token", "function", 5, 18,
        "def verify_session_token(session_id: str):\n"
        '    """Verify a session token and return the user id."""\n'
        "    session = load_session(session_id)\n"
        "    if session is None or session.revoked:\n"
        "        raise InvalidSessionError('session revoked')\n"
        "    return session.user_id",
        imports=["db"],
    ),
    _chunk(
        "src/users.py", "get_user", "function", 3, 9,
        "def get_user(user_id: int):\n"
        "    return db.query(User).get(user_id)",
        imports=["db"],
    ),
    _chunk(
        "src/auth/middleware.py", "AuthMiddleware", "class", 1, 20,
        "class AuthMiddleware:\n"
        '    """Enforces authentication on incoming requests."""\n'
        "    def authenticate(self, request):\n"
        "        token = request.headers.get('Authorization')\n"
        "        return verify_jwt_token(token)",
        imports=["auth.jwt"],
    ),
    _chunk(
        "src/api/routes.py", "get_profile", "function", 12, 20,
        "def get_profile(request):\n"
        "    user = AuthMiddleware().authenticate(request)\n"
        "    return render_profile(user)",
        imports=["auth.middleware"],
    ),
]

# Corpus B — agent hops. Six filler chunks deliberately dominate the
# initial query's vocabulary so the AuthMiddleware definition starts
# OUTSIDE the iteration-1 top-6 and only a second hop can bring it in.
CORPUS_B = [
    _chunk(
        f"src/security/guard{i}.py", f"guard_{i}", "function", 1, 4,
        f"def guard_{i}(request):\n"
        "    # profile request protection unauthorized access check\n"
        "    return check_profile_access(request)",
    )
    for i in range(1, 7)
] + [
    _chunk(
        "src/api/routes.py", "get_profile", "function", 12, 15,
        "def get_profile(request):\n"
        "    user = AuthMiddleware().authenticate(request)\n"
        "    return render_profile(user)",
        imports=["auth.middleware"],
    ),
    _chunk(
        "src/auth/middleware.py", "AuthMiddleware", "class", 1, 3,
        "class AuthMiddleware:\n"
        "    def authenticate(self, request):\n"
        "        return verify_jwt_token(request.token)",
        imports=["auth.jwt"],
    ),
]


async def _seed(db_url: str, corpus: list[dict]) -> str:
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        repo = Repository(
            repo_url="https://github.com/octocat/demo",
            owner="octocat",
            name="demo",
            default_branch="main",
            commit_sha="abc123",
            status=RepositoryStatus.ready,
            total_files=len({c["file_path"] for c in corpus}),
            total_chunks=len(corpus),
        )
        session.add(repo)
        await session.flush()
        for spec in corpus:
            session.add(
                CodeChunk(
                    repo_id=repo.id,
                    embedding=local_embedding(
                        f"{spec['context_header']}\n{spec['content']}"
                    ),
                    **spec,
                )
            )
        await session.commit()
        repo_id = str(repo.id)
    await engine.dispose()
    return repo_id


def _retriever_for(db_url: str, repo_id: str):
    engine = create_async_engine(db_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    return engine, Session


async def _with_retriever(db_url: str, repo_id: str, fn):
    engine, Session = _retriever_for(db_url, repo_id)
    async with Session() as session:
        retriever = HybridCodeRetriever(
            session, repo_id, embedding_service=EmbeddingService(api_key="mock")
        )
        result = await fn(retriever)
    await engine.dispose()
    return result


# --- Hybrid RRF retrieval -------------------------------------------------

def test_rrf_ranks_exact_symbol_match_first(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/rrf.db"

    async def run():
        repo_id = await _seed(db_url, CORPUS_A)
        return await _with_retriever(
            db_url,
            repo_id,
            lambda r: r.search(
                "How does verify_jwt_token validate a JWT?", top_k=4
            ),
        )

    results = asyncio.run(run())
    assert results, "expected retrieval results"
    top = results[0]
    # The exact symbol beats the semantically-close session verifier.
    assert top.symbol_name == "verify_jwt_token"
    assert top.lexical_rank == 1
    assert top.rrf_score == max(r.rrf_score for r in results)
    # RRF formula, k=60, applied exactly as specified.
    for entry in results:
        expected = 0.0
        if entry.semantic_rank is not None:
            expected += 1.0 / (60 + entry.semantic_rank)
        if entry.lexical_rank is not None:
            expected += 1.0 / (60 + entry.lexical_rank)
        assert math.isclose(entry.rrf_score, expected, rel_tol=1e-9)


def test_retriever_metadata_filters(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/filters.db"

    async def run():
        repo_id = await _seed(db_url, CORPUS_A)
        by_path = await _with_retriever(
            db_url, repo_id,
            lambda r: r.search("token", top_k=6, file_path_filter="session"),
        )
        by_type = await _with_retriever(
            db_url, repo_id,
            lambda r: r.search("auth", top_k=6, symbol_type_filter="class"),
        )
        return by_path, by_type

    by_path, by_type = asyncio.run(run())
    assert by_path and all("session" in r.file_path for r in by_path)
    assert by_type and all(
        r.chunk.symbol_type == "class" for r in by_type
    )
    assert {r.symbol_name for r in by_type} == {"AuthMiddleware"}


def test_find_symbol_definition_and_imports(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/symbols.db"

    async def run():
        repo_id = await _seed(db_url, CORPUS_A)

        async def lookups(r: HybridCodeRetriever):
            return (
                await r.find_symbol_definition("verify_jwt_token"),
                await r.find_symbol_definition("AuthMiddleware"),
                await r.find_symbol_definition("auth.jwt"),
                await r.find_symbol_definition("nonexistent_symbol_xyz"),
            )

        return await _with_retriever(db_url, repo_id, lookups)

    definitions, class_defs, import_hits, none_found = asyncio.run(run())
    assert [d.symbol_name for d in definitions] == ["verify_jwt_token"]
    assert definitions[0].match_type == "definition"
    assert [d.symbol_name for d in class_defs] == ["AuthMiddleware"]
    # No chunk *defines* auth.jwt — the middleware chunk imports it.
    assert import_hits
    assert all(h.match_type == "import" for h in import_hits)
    assert {h.symbol_name for h in import_hits} == {"AuthMiddleware"}
    assert none_found == []


# --- Agentic loop ------------------------------------------------------------

class ScriptedLLM:
    """LLMBackend double: analyzer returns a fixed plan, the grader
    plays back a scripted sequence of verdicts."""

    def __init__(self, grades: list[GradeResult]):
        self._grades = list(grades)
        self.grade_calls = 0

    async def analyze_query(self, question, chat_history):
        return AnalysisResult(
            target_symbols=[],
            file_hints=[],
            search_query="profile request protection unauthorized access",
            reasoning="scripted",
        )

    async def grade_context(self, question, chunks):
        self.grade_calls += 1
        if self._grades:
            return self._grades.pop(0)
        return GradeResult(is_sufficient=True, reasoning="scripted-default")

    async def generate_answer(self, question, chunks, chat_history):
        return (
            "Profile requests are enforced by `AuthMiddleware` "
            "[src/auth/middleware.py:1-3]."
        )


def test_agent_triggers_second_hop_when_grader_insufficient(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/hop.db"

    async def run():
        repo_id = await _seed(db_url, CORPUS_B)
        engine, Session = _retriever_for(db_url, repo_id)
        llm = ScriptedLLM(
            grades=[
                GradeResult(
                    is_sufficient=False,
                    missing_symbols_or_files=["AuthMiddleware"],
                    refined_query="AuthMiddleware class definition",
                    reasoning="routes call AuthMiddleware but its "
                    "definition is not in context",
                ),
                GradeResult(is_sufficient=True, reasoning="now grounded"),
            ]
        )
        async with Session() as session:
            retriever = HybridCodeRetriever(
                session,
                repo_id,
                embedding_service=EmbeddingService(api_key="mock"),
            )
            agent = CodebaseAgent(retriever, llm)
            result = await agent.answer_question(
                "How are profile requests protected?"
            )
        await engine.dispose()
        return result, llm

    result, llm = asyncio.run(run())
    assert llm.grade_calls == 2  # initial grade + re-grade after the hop
    assert result.hops_used == 1
    stages = [t["stage"] for t in result.agent_trace]
    assert stages[0] == "analyze"
    assert "retrieve" in stages and "grade" in stages and "hop" in stages
    hop_step = next(t for t in result.agent_trace if t["stage"] == "hop")
    assert "AuthMiddleware" in hop_step["description"]
    assert hop_step["added_chunks"] >= 1
    # The hop actually delivered the missing definition into the sources.
    assert "AuthMiddleware" in {s.symbol_name for s in result.sources}
    assert "[src/auth/middleware.py:1-3]" in result.answer
    assert stages[-1] == "answer"


def test_agent_no_hop_when_grader_sufficient(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/nohop.db"

    async def run():
        repo_id = await _seed(db_url, CORPUS_B)
        engine, Session = _retriever_for(db_url, repo_id)
        llm = ScriptedLLM(
            grades=[GradeResult(is_sufficient=True, reasoning="fine")]
        )
        async with Session() as session:
            retriever = HybridCodeRetriever(
                session,
                repo_id,
                embedding_service=EmbeddingService(api_key="mock"),
            )
            agent = CodebaseAgent(retriever, llm)
            result = await agent.answer_question(
                "How are profile requests protected?"
            )
        await engine.dispose()
        return result, llm

    result, llm = asyncio.run(run())
    assert result.hops_used == 0
    assert llm.grade_calls == 1
    assert "hop" not in [t["stage"] for t in result.agent_trace]


def test_agent_respects_max_two_hops(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/maxhops.db"

    async def run():
        repo_id = await _seed(db_url, CORPUS_B)
        engine, Session = _retriever_for(db_url, repo_id)
        never_satisfied = GradeResult(
            is_sufficient=False,
            missing_symbols_or_files=["AuthMiddleware"],
            refined_query="AuthMiddleware",
            reasoning="never satisfied (scripted)",
        )
        llm = ScriptedLLM(grades=[never_satisfied] * 10)
        async with Session() as session:
            retriever = HybridCodeRetriever(
                session,
                repo_id,
                embedding_service=EmbeddingService(api_key="mock"),
            )
            agent = CodebaseAgent(retriever, llm)
            result = await agent.answer_question("anything?")
        await engine.dispose()
        return result

    result = asyncio.run(run())
    assert result.hops_used == 2  # hard cap, even if the grader never yields


# --- Query endpoint -----------------------------------------------------------

def test_query_endpoint_shape_and_citations(tmp_path):
    from fastapi.testclient import TestClient

    from backend.app import main as main_module

    db_url = f"sqlite+aiosqlite:///{tmp_path}/endpoint.db"
    repo_id = asyncio.run(_seed(db_url, CORPUS_A))

    async def override_get_db():
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with Session() as session:
            yield session
        await engine.dispose()

    main_module.app.dependency_overrides[main_module.get_db] = override_get_db
    main_module.app.dependency_overrides[
        main_module.get_embedding_service
    ] = lambda: EmbeddingService(api_key="mock")
    main_module.app.dependency_overrides[
        main_module.get_llm_backend
    ] = lambda: HeuristicLLM()
    try:
        with TestClient(main_module.app) as client:
            resp = client.post(
                f"/api/repos/{repo_id}/query",
                json={
                    "question": "How does verify_jwt_token work?",
                    "chat_history": [],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "[src/auth/jwt.py:10-25]" in body["answer"]
            stages = [t["stage"] for t in body["agent_trace"]]
            assert stages[0] == "analyze"
            assert "retrieve" in stages and "grade" in stages
            assert stages[-1] == "answer"
            assert body["sources"]
            top_files = {s["file_path"] for s in body["sources"]}
            assert "src/auth/jwt.py" in top_files
            assert all("rrf_score" in s for s in body["sources"])

            missing = client.post(
                "/api/repos/00000000-0000-0000-0000-000000000000/query",
                json={"question": "anything", "chat_history": []},
            )
            assert missing.status_code == 404

            # File-chunks endpoint (drives the frontend's AST explorer).
            chunks_resp = client.get(
                f"/api/repos/{repo_id}/chunks",
                params={"file_path": "src/auth/jwt.py"},
            )
            assert chunks_resp.status_code == 200
            file_chunks = chunks_resp.json()["chunks"]
            assert [c["symbol_name"] for c in file_chunks] == ["verify_jwt_token"]
            assert file_chunks[0]["start_line"] == 10
    finally:
        main_module.app.dependency_overrides.clear()
