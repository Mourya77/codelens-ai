"""Agentic RAG orchestrator for CodeLens AI (Phase 3).

Not a single-pass retrieve-and-answer pipeline. ``CodebaseAgent`` runs
an explicit state machine:

    1. Query Analyzer & Planner   — symbols, file hints, search query
    2. Hybrid Retrieval (iter 1)  — top-6 chunks via HybridCodeRetriever
    3. Context Sufficiency Grader — structured JSON self-reflection:
       {is_sufficient, missing_symbols_or_files, refined_query, reasoning}
    4. Autonomous second hop      — up to 2 hops: find_symbol_definition
       for each missing symbol/file + a fresh hybrid search on the
       refined query; results merged and deduplicated, then re-graded
    5. Grounded Answer            — Markdown where every technical
       claim cites [file_path:start_line-end_line]

The LLM work is behind the ``LLMBackend`` protocol:

* ``GeminiLLM`` — Gemini (``gemini-3.1-flash-lite`` by default) over REST.
* ``HeuristicLLM`` — a deterministic offline backend used when the API
  key is ``"mock"``/empty: identifier extraction for analysis, a
  referenced-but-undefined-symbol check for grading, and an extractive
  cited answer. It keeps the full loop runnable in tests and offline
  demos; it is a stand-in, not a reasoning equal of the real model.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .config import env_proxy_url, settings
from .retriever import HybridCodeRetriever, RetrievedChunk, query_identifiers

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
INITIAL_TOP_K = 6
MAX_HOPS = 2

_CITATION_RE = re.compile(r"\[[^\]\n]+:\d+-\d+\]")
_IDENT_CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_DEF_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_BUILTINS = {
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict",
    "set", "tuple", "open", "super", "isinstance", "enumerate", "zip",
    "min", "max", "sum", "abs", "sorted", "reversed", "type", "id",
    "if", "for", "while", "return", "require", "parseint", "parsefloat",
    "string", "number", "boolean", "object", "array", "json", "math",
    "console", "error", "exception", "valueerror", "typeerror",
}


# ---------------------------------------------------------------------------
# Structured step results
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    target_symbols: list[str] = field(default_factory=list)
    file_hints: list[str] = field(default_factory=list)
    search_query: str = ""
    reasoning: str = ""


@dataclass
class GradeResult:
    is_sufficient: bool = False
    missing_symbols_or_files: list[str] = field(default_factory=list)
    refined_query: str = ""
    reasoning: str = ""


@dataclass
class AgentResult:
    answer: str
    agent_trace: list[dict]
    sources: list[RetrievedChunk]
    hops_used: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "agent_trace": self.agent_trace,
            "sources": [s.to_source_dict() for s in self.sources],
            "hops_used": self.hops_used,
        }


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """The three LLM calls the agent makes. Implementations: GeminiLLM
    (production), HeuristicLLM (offline), and test doubles."""

    async def analyze_query(
        self, question: str, chat_history: list[dict]
    ) -> AnalysisResult: ...

    async def grade_context(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GradeResult: ...

    async def generate_answer(
        self, question: str, chunks: list[RetrievedChunk], chat_history: list[dict]
    ) -> str: ...


def _format_history(chat_history: list[dict]) -> str:
    lines = []
    for item in (chat_history or [])[-8:]:
        role = str(item.get("role", "user"))
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(no prior conversation)"


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for entry in chunks:
        c = entry.chunk
        blocks.append(
            f"--- {c.context_header} [lines {c.start_line}-{c.end_line}] ---\n"
            f"{c.content}"
        )
    return "\n\n".join(blocks) or "(no chunks retrieved)"


def _parse_json_object(raw: str) -> dict:
    """Tolerant JSON extraction: models wrap JSON in prose/fences."""
    text_value = (raw or "").strip()
    if text_value.startswith("```"):
        text_value = re.sub(r"^```(?:json)?", "", text_value).strip()
        text_value = re.sub(r"```$", "", text_value).strip()
    try:
        return json.loads(text_value)
    except json.JSONDecodeError:
        start, end = text_value.find("{"), text_value.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text_value[start : end + 1])
        raise


def _analysis_from_dict(data: dict, question: str) -> AnalysisResult:
    return AnalysisResult(
        target_symbols=[str(s) for s in data.get("target_symbols", []) if s],
        file_hints=[str(s) for s in data.get("file_hints", []) if s],
        search_query=str(data.get("search_query") or question),
        reasoning=str(data.get("reasoning", "")),
    )


def _grade_from_dict(data: dict) -> GradeResult:
    return GradeResult(
        is_sufficient=bool(data.get("is_sufficient", False)),
        missing_symbols_or_files=[
            str(s) for s in data.get("missing_symbols_or_files", []) if s
        ],
        refined_query=str(data.get("refined_query", "")),
        reasoning=str(data.get("reasoning", "")),
    )


class GeminiLLM:
    """Gemini-backed LLMBackend (REST generateContent)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.model = model or getattr(settings, "gemini_model", DEFAULT_MODEL)
        self._http = http_client

    async def _generate(self, prompt: str, *, json_mode: bool) -> str:
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        client = self._http
        owns = False
        if client is None:
            import ssl

            kwargs: dict = {
                "timeout": httpx.Timeout(90.0),
                "trust_env": False,
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
            owns = True
        # Retry transient failures (429 quota spikes, 5xx, transport
        # errors) with backoff: a single agent answer makes several
        # sequential LLM calls, so one momentary 429 used to fail the
        # whole request with a 500. Non-transient 4xx raise at once.
        retryable = {429, 500, 502, 503}
        try:
            resp: httpx.Response | None = None
            for attempt in range(3):
                try:
                    resp = await client.post(url, json=body, headers=headers)
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                if resp.status_code not in retryable or attempt == 2:
                    break
                delay = 2.0 * (attempt + 1)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = min(float(retry_after), 20.0)
                    except ValueError:
                        pass
                await asyncio.sleep(delay)
        finally:
            if owns:
                await client.aclose()
        assert resp is not None
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gemini API {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        return "".join(p.get("text", "") for p in parts)

    async def analyze_query(
        self, question: str, chat_history: list[dict]
    ) -> AnalysisResult:
        prompt = (
            "You are the Query Analyzer for a code-search agent.\n\n"
            f"CHAT HISTORY:\n{_format_history(chat_history)}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Extract the code symbols (function/class/method names) the "
            "question targets, any file path hints, and one optimized "
            "search query for hybrid (semantic + keyword) code search.\n"
            'Respond with ONLY this JSON: {"target_symbols": [str], '
            '"file_hints": [str], "search_query": str, "reasoning": str}'
        )
        raw = await self._generate(prompt, json_mode=True)
        return _analysis_from_dict(_parse_json_object(raw), question)

    async def grade_context(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GradeResult:
        prompt = (
            "You are the Context Sufficiency Grader for a code Q&A agent.\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"RETRIEVED CODE CHUNKS:\n{_format_chunks(chunks)}\n\n"
            "Do these code chunks contain the complete implementation "
            "details needed to answer the user's question accurately, or "
            "do they call/import another function/class in this repo "
            "whose definition is missing from the chunks? List every "
            "such missing symbol or file exactly by name, and propose a "
            "refined search query that would find them.\n"
            'Respond with ONLY this JSON: {"is_sufficient": bool, '
            '"missing_symbols_or_files": [str], "refined_query": str, '
            '"reasoning": str}'
        )
        raw = await self._generate(prompt, json_mode=True)
        return _grade_from_dict(_parse_json_object(raw))

    async def generate_answer(
        self, question: str, chunks: list[RetrievedChunk], chat_history: list[dict]
    ) -> str:
        prompt = (
            "You are CodeLens AI, answering questions about a codebase "
            "STRICTLY from the retrieved code below.\n\n"
            f"CHAT HISTORY:\n{_format_history(chat_history)}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"RETRIEVED CODE:\n{_format_chunks(chunks)}\n\n"
            "Write a Markdown answer. Rules:\n"
            "- Every technical claim MUST cite the exact file path and "
            "line range like [src/auth.py:10-25] using the chunk headers.\n"
            "- Use only facts present in the retrieved code; if the code "
            "does not show something, say so explicitly.\n"
            "- Be specific: name functions, classes, parameters and "
            "control flow as they appear in the code."
        )
        return await self._generate(prompt, json_mode=False)


class HeuristicLLM:
    """Deterministic offline backend (mock key / no key).

    Analyzer: extracts symbol-looking identifiers and file hints from
    the question. Grader: finds identifiers *called* inside the
    retrieved chunks that are neither defined in the context nor
    builtins — those are the missing symbols a second hop should fetch.
    Answer: an extractive, fully-cited summary of the top chunks.
    """

    async def analyze_query(
        self, question: str, chat_history: list[dict]
    ) -> AnalysisResult:
        # Sentence-initial / question words are capitalized but are
        # never code symbols.
        non_symbols = {
            "how", "what", "why", "when", "where", "which", "who",
            "does", "do", "is", "are", "the", "this", "that", "can",
            "could", "would", "should", "in", "on", "for",
        }
        symbols: list[str] = []
        for ident in query_identifiers(question):
            if ident in non_symbols:
                continue
            raw_forms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
            for raw in raw_forms:
                looks_like_symbol = "_" in raw or any(
                    c.isupper() for c in raw[1:]
                )
                if (
                    raw.lower() == ident
                    and looks_like_symbol
                    and raw not in symbols
                ):
                    symbols.append(raw)
        file_hints = [
            token
            for token in re.findall(r"[\w./-]+\.[A-Za-z]{1,5}\b", question)
            if "/" in token or token.rsplit(".", 1)[-1] in
            {"py", "ts", "tsx", "js", "jsx", "go", "java", "cpp", "h", "md"}
        ]
        history_note = ""
        if chat_history:
            history_note = " (with conversation history considered)"
        return AnalysisResult(
            target_symbols=symbols,
            file_hints=file_hints,
            search_query=question,
            reasoning=f"Heuristic analysis{history_note}: extracted "
            f"{len(symbols)} target symbol(s) from the question.",
        )

    async def grade_context(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GradeResult:
        defined: set[str] = set()
        for entry in chunks:
            c = entry.chunk
            if c.symbol_name:
                defined.add(c.symbol_name.lower())
                defined.add(c.symbol_name.split(".")[-1].lower())
            for name in _DEF_RE.findall(c.content):
                defined.add(name.lower())
            for imp in c.imports or []:
                defined.add(imp.split(".")[-1].lower())
        missing: list[str] = []
        for entry in chunks:
            for name in _IDENT_CALL_RE.findall(entry.chunk.content):
                lowered = name.lower()
                if (
                    lowered not in defined
                    and lowered not in _BUILTINS
                    and name not in missing
                    and len(name) > 2
                ):
                    missing.append(name)
        if not chunks:
            return GradeResult(
                is_sufficient=False,
                missing_symbols_or_files=[],
                refined_query=question,
                reasoning="No context was retrieved at all.",
            )
        if missing:
            return GradeResult(
                is_sufficient=False,
                missing_symbols_or_files=missing[:5],
                refined_query=" ".join(missing[:3]) + " definition",
                reasoning=(
                    "Retrieved code calls symbols whose definitions are "
                    f"not in context: {', '.join(missing[:5])}."
                ),
            )
        return GradeResult(
            is_sufficient=True,
            reasoning="All called symbols are defined within the "
            "retrieved context.",
        )

    async def generate_answer(
        self, question: str, chunks: list[RetrievedChunk], chat_history: list[dict]
    ) -> str:
        if not chunks:
            return (
                "I couldn't find any code in this repository that is "
                "relevant to that question. Try naming the function, "
                "class, or file you mean."
            )
        lines = [
            f"Based on the retrieved code, here is what answers "
            f"\"{question}\":",
            "",
        ]
        for entry in chunks[:6]:
            c = entry.chunk
            citation = f"[{c.file_path}:{c.start_line}-{c.end_line}]"
            label = (
                f"`{c.symbol_name}` ({c.symbol_type})"
                if c.symbol_name
                else f"`{c.file_path}`"
            )
            snippet_lines = [
                ln.strip() for ln in c.content.split("\n") if ln.strip()
            ]
            preview = snippet_lines[0][:140] if snippet_lines else ""
            lines.append(f"- {label} in {citation} — `{preview}`")
        lines += [
            "",
            "These excerpts are the exact definitions and code paths "
            "the retriever grounded this answer on; open the cited "
            "ranges for the full implementations.",
        ]
        return "\n".join(lines)


def make_llm_backend(api_key: str | None = None) -> LLMBackend:
    """Gemini when a real key is configured, heuristic offline backend
    for `GEMINI_API_KEY="mock"` / empty (tests, offline demos)."""
    key = settings.gemini_api_key if api_key is None else api_key
    if (key or "").strip().lower() in ("", "mock"):
        return HeuristicLLM()
    return GeminiLLM(api_key=key)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class CodebaseAgent:
    """The multi-step agentic RAG loop described in the module docstring."""

    def __init__(
        self,
        retriever: HybridCodeRetriever,
        llm: LLMBackend | None = None,
        *,
        max_hops: int = MAX_HOPS,
        initial_top_k: int = INITIAL_TOP_K,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or make_llm_backend()
        self.max_hops = max_hops
        self.initial_top_k = initial_top_k

    async def answer_question(
        self, question: str, chat_history: list[dict] | None = None
    ) -> AgentResult:
        history = chat_history or []
        trace: list[dict] = []

        def step(stage: str, description: str, **extra) -> None:
            trace.append(
                {
                    "step": len(trace) + 1,
                    "stage": stage,
                    "description": description,
                    **extra,
                }
            )

        # -- Step 1: analyze & plan -------------------------------------
        analysis = await self.llm.analyze_query(question, history)
        symbols_note = (
            f" Target symbols: {', '.join(analysis.target_symbols)}."
            if analysis.target_symbols
            else ""
        )
        step(
            "analyze",
            f"Step 1: Initial Query — planned search "
            f"\"{analysis.search_query}\".{symbols_note}",
            search_query=analysis.search_query,
            target_symbols=analysis.target_symbols,
            file_hints=analysis.file_hints,
        )

        # -- Step 2: hybrid retrieval, iteration 1 -----------------------
        context = await self.retriever.search(
            analysis.search_query, top_k=self.initial_top_k
        )
        step(
            "retrieve",
            f"Step 2: Retrieved {len(context)} chunks via hybrid "
            "search (RRF).",
            chunk_count=len(context),
            files=sorted({c.file_path for c in context}),
        )

        # -- Steps 3+4: grade, then hop while insufficient (max 2) -------
        hops_used = 0
        grade = await self.llm.grade_context(question, context)
        step(
            "grade",
            (
                "Step 3: Grader confirmed the context is sufficient."
                if grade.is_sufficient
                else "Step 3: Grader detected missing context: "
                + ", ".join(grade.missing_symbols_or_files or ["(unspecified)"])
                + "."
            ),
            is_sufficient=grade.is_sufficient,
            missing=grade.missing_symbols_or_files,
            reasoning=grade.reasoning,
        )

        while not grade.is_sufficient and hops_used < self.max_hops:
            hops_used += 1
            new_chunks = await self._second_hop(grade, analysis)
            added = self._merge(context, new_chunks)
            context = context + added
            step(
                "hop",
                f"Step 4: Executed hop-{hops_used} lookup "
                f"({', '.join(grade.missing_symbols_or_files) or 'refined query'} "
                f"→ {len(added)} new chunks merged, {len(context)} total).",
                hop=hops_used,
                added_chunks=len(added),
                missing=grade.missing_symbols_or_files,
                refined_query=grade.refined_query,
            )
            if hops_used < self.max_hops or added:
                grade = await self.llm.grade_context(question, context)
                step(
                    "grade",
                    (
                        f"Step 4b: Re-grade after hop-{hops_used} — "
                        "context is now sufficient."
                        if grade.is_sufficient
                        else f"Step 4b: Re-grade after hop-{hops_used} — "
                        "still missing: "
                        + ", ".join(
                            grade.missing_symbols_or_files or ["(unspecified)"]
                        )
                        + "."
                    ),
                    is_sufficient=grade.is_sufficient,
                    missing=grade.missing_symbols_or_files,
                    reasoning=grade.reasoning,
                )
                if grade.is_sufficient:
                    break

        # -- Step 5: grounded answer --------------------------------------
        answer = await self.llm.generate_answer(question, context, history)
        if context and not _CITATION_RE.search(answer):
            refs = "\n".join(
                f"- [{c.file_path}:{c.chunk.start_line}-{c.chunk.end_line}] "
                f"{c.chunk.context_header}"
                for c in context[:6]
            )
            answer = f"{answer}\n\n### References\n{refs}"
        step(
            "answer",
            f"Step 5: Synthesized answer from {len(context)} chunks "
            f"({hops_used} extra hop(s) used).",
            hops_used=hops_used,
            source_count=len(context),
        )
        return AgentResult(
            answer=answer, agent_trace=trace, sources=context, hops_used=hops_used
        )

    # -- helpers ------------------------------------------------------------

    async def _second_hop(
        self, grade: GradeResult, analysis: AnalysisResult
    ) -> list[RetrievedChunk]:
        found: list[RetrievedChunk] = []
        for item in grade.missing_symbols_or_files:
            if "/" in item or re.search(
                r"\.(py|ts|tsx|js|jsx|go|java|cpp|h|hpp|md)$", item
            ):
                found.extend(await self.retriever.get_file_chunks(item))
            else:
                found.extend(
                    await self.retriever.find_symbol_definition(item)
                )
        query = grade.refined_query or " ".join(
            grade.missing_symbols_or_files
        )
        if query:
            found.extend(
                await self.retriever.search(query, top_k=self.initial_top_k)
            )
        for symbol in analysis.target_symbols:
            if symbol in grade.missing_symbols_or_files:
                continue  # already handled above
        return found

    @staticmethod
    def _merge(
        existing: list[RetrievedChunk], new: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Chunks from `new` whose ids aren't already in `existing`,
        in arrival order, internally deduplicated."""
        seen = {entry.chunk.id for entry in existing}
        added: list[RetrievedChunk] = []
        for entry in new:
            if entry.chunk.id not in seen:
                seen.add(entry.chunk.id)
                added.append(entry)
        return added
