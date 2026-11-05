"""AST & syntax-aware code chunking for CodeLens AI.

Standard character/recursive chunking breaks code in the middle of
functions and strips the context a retriever needs. ``CodeAwareChunker``
instead chunks on *symbol boundaries*:

* **Python** — real AST parsing (``ast``): intact top-level functions,
  classes (as a skeleton chunk: signature + docstring + class-level
  body) and each class method as its own chunk, all with decorators,
  docstrings and exact 1-based line numbers.
* **TypeScript / JavaScript / C++ / Go / Java** — there is no stdlib
  AST for these, so a structural parser is used: regex signature
  detection + brace/scope balancing that ignores braces inside strings,
  character literals and comments, so a block is never split mid-scope.
* **Markdown / reStructuredText** — split by heading hierarchy, with
  the heading path preserved in the context header.
* **Anything else** — a lossless fallback chunk (see below).

Every chunk carries a ``context_header`` such as
``"File: src/auth.py | Class: AuthManager | Method: verify_token"``
and the file's import list — the metadata hybrid search and the
(Phase 2+) agents need to ground an answer.

Coverage is lossless: code that lives outside any symbol (module-level
constants, statements between functions, class fields between methods)
is emitted as ``module_header`` / class-skeleton chunks rather than
being dropped.

Oversized symbols (> ``max_tokens``, default 1,500) are sub-chunked on
line / logical-block boundaries, and every sub-chunk's content is
prepended with its ``context_header`` so a retrieved fragment always
knows which symbol it came from.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_TOKENS_DEFAULT = 1500

EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".go": "go",
    ".java": "java",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
}

_CONTROL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "else", "do",
    "try", "finally", "synchronized", "sizeof", "case",
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for code).

    Phase 1 deliberately avoids a tokenizer dependency; this heuristic
    is only used to decide when a symbol needs sub-chunking, where an
    approximation is fine.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Chunk:
    """One retrievable unit — mirrors the ``CodeChunk`` DB model
    minus persistence fields (id, repo_id, embedding, search_vector)."""

    file_path: str
    language: str
    symbol_name: str | None
    symbol_type: str  # function | class | method | module_header | doc
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    content: str
    context_header: str
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Brace-scanning helpers (brace languages)
# ---------------------------------------------------------------------------

def _sanitized_brace_delta(line: str, in_block_comment: bool) -> tuple[int, bool]:
    """Net ``{`` minus ``}`` on a line, ignoring braces inside string /
    char / template literals and comments. Returns (delta, still_in_block_comment)."""
    delta = 0
    i, n = 0, len(line)
    quote: str | None = None
    while i < n:
        ch = line[i]
        nxt = line[i + 1] if i + 1 < n else ""
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            break  # rest of line is a comment
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue
        if ch == "{":
            delta += 1
        elif ch == "}":
            delta -= 1
        i += 1
    return delta, in_block_comment


def _find_block_end(lines: list[str], start_idx: int) -> int | None:
    """Given the 0-based line a symbol starts on, return the 0-based
    line its opening brace block closes on (or a ``;``-terminated
    declaration line). None if no block opens within 6 lines."""
    in_block = False
    depth = 0
    opened = False
    for j in range(start_idx, min(len(lines), start_idx + 5000)):
        line = lines[j]
        delta, in_block = _sanitized_brace_delta(line, in_block)
        if not opened:
            if "{" in _strip_line_comment(line):
                opened = True
                depth += delta
                if depth <= 0:
                    return j
            elif line.rstrip().endswith(";") and j > start_idx:
                return j  # declaration without a body (e.g. .h prototype)
            elif j - start_idx > 6:
                return None
            continue
        depth += delta
        if depth <= 0:
            return j
    return None


def _strip_line_comment(line: str) -> str:
    # Good enough for brace *detection*; full sanitization happens in
    # _sanitized_brace_delta.
    return line.split("//", 1)[0]


# ---------------------------------------------------------------------------
# Signature regexes (brace languages)
# ---------------------------------------------------------------------------

_CLASS_RE = re.compile(
    r"^\s*(?:(?:export|default|declare|abstract|public|private|protected|"
    r"static|final|sealed|partial|open|internal)\s+)*"
    r"(class|interface|enum|struct|record)\s+([A-Za-z_$][\w$]*)"
)
_GO_TYPE_RE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(struct|interface)\s*\{?")
_TS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
    r"function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("
)
_TS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?"
    r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>"
)
_GO_FUNC_RE = re.compile(
    r"^\s*func\s+(?:\(\s*[^)]*\)\s*)?([A-Za-z_]\w*)\s*\("
)
# Java / C++ style: <modifiers> <return-type> name(args) [const] [throws ...] {
_GENERIC_FUNC_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|virtual|async|extern|"
    r"inline|constexpr|override|synchronized|native|abstract)\s+)*"
    r"(?:[A-Za-z_][\w<>\[\],.?&*:]*\s+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?(?:throws\s+[\w.,\s]+)?(?:\{|$)"
)


class CodeAwareChunker:
    """Split source code into intact, context-carrying chunks."""

    def __init__(self, max_tokens: int = MAX_TOKENS_DEFAULT) -> None:
        self.max_tokens = max_tokens

    # -- public API ------------------------------------------------------

    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        """Chunk one file's content. ``file_path`` is the repo-relative
        path (used in context headers and for language detection)."""
        language = EXTENSION_LANGUAGES.get(Path(file_path).suffix.lower(), "text")
        if not content.strip():
            return []
        if language == "python":
            chunks = self._chunk_python(file_path, content)
        elif language in ("markdown", "rst"):
            chunks = self._chunk_docs(file_path, content, language)
        elif language in ("typescript", "javascript", "cpp", "go", "java"):
            chunks = self._chunk_braces(file_path, content, language)
        else:
            chunks = self._chunk_fallback(file_path, content, language)
        # Sub-chunk oversized symbols, then present in file order.
        out: list[Chunk] = []
        for chunk in chunks:
            out.extend(self._maybe_subchunk(chunk))
        out.sort(key=lambda c: (c.start_line, c.end_line))
        return out

    # Backwards/alternate-name convenience.
    chunk_code = chunk_file

    # -- sub-chunking ------------------------------------------------------

    def _maybe_subchunk(self, chunk: Chunk) -> list[Chunk]:
        if estimate_tokens(chunk.content) <= self.max_tokens:
            return [chunk]
        lines = chunk.content.split("\n")
        header = chunk.context_header
        budget_chars = max(200, (self.max_tokens - estimate_tokens(header) - 8) * 4)
        parts: list[Chunk] = []
        seg_lines: list[str] = []
        seg_start = chunk.start_line

        def flush(end_line: int) -> None:
            if not seg_lines:
                return
            body = "\n".join(seg_lines)
            parts.append(
                Chunk(
                    file_path=chunk.file_path,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    start_line=seg_start,
                    end_line=end_line,
                    # Context header prepended to *every* sub-chunk.
                    content=f"{header}\n{body}",
                    context_header=header,
                    imports=list(chunk.imports),
                )
            )

        for offset, line in enumerate(lines):
            line_no = chunk.start_line + offset
            # Prefer breaking on a blank line (a logical-block edge)
            # once the segment is at least half full; otherwise break
            # when the budget is exhausted. Never split a single line.
            would_exceed = sum(len(x) + 1 for x in seg_lines) + len(line) > budget_chars
            at_block_edge = not line.strip()
            if seg_lines and would_exceed and (at_block_edge or len(seg_lines) > 1):
                flush(line_no - 1)
                seg_lines = []
                seg_start = line_no
            seg_lines.append(line)
        flush(chunk.start_line + len(lines) - 1)
        return parts or [chunk]

    # -- shared chunk builder ---------------------------------------------

    @staticmethod
    def _make(
        file_path: str,
        language: str,
        lines: list[str],
        start: int,  # 1-based
        end: int,  # 1-based inclusive
        symbol_name: str | None,
        symbol_type: str,
        context_header: str,
        imports: list[str],
    ) -> Chunk:
        return Chunk(
            file_path=file_path,
            language=language,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            start_line=start,
            end_line=end,
            content="\n".join(lines[start - 1 : end]),
            context_header=context_header,
            imports=list(imports),
        )

    @staticmethod
    def _gap_chunks(
        file_path: str,
        language: str,
        lines: list[str],
        covered: list[tuple[int, int]],  # 1-based inclusive ranges
        symbol_name: str | None,
        context_header: str,
        imports: list[str],
        symbol_type: str = "module_header",
    ) -> list[Chunk]:
        """Emit chunks for every non-blank line range not in ``covered``."""
        covered_set: set[int] = set()
        for s, e in covered:
            covered_set.update(range(s, e + 1))
        gaps: list[Chunk] = []
        run_start: int | None = None
        for idx, raw in enumerate(lines, start=1):
            uncovered = idx not in covered_set and raw.strip()
            if uncovered and run_start is None:
                run_start = idx
            if run_start is not None and (idx in covered_set or not raw.strip() and idx not in covered_set and False):
                # A covered line ends the run. Blank lines inside a run
                # are kept (they are part of the block); only a covered
                # line or EOF closes it.
                if idx in covered_set:
                    gaps.append(
                        CodeAwareChunker._make(
                            file_path, language, lines, run_start, idx - 1,
                            symbol_name, symbol_type, context_header, imports,
                        )
                    )
                    run_start = None
        if run_start is not None:
            gaps.append(
                CodeAwareChunker._make(
                    file_path, language, lines, run_start, len(lines),
                    symbol_name, symbol_type, context_header, imports,
                )
            )
        # Trim trailing blank lines inside gap chunks.
        for g in gaps:
            while g.end_line > g.start_line and not lines[g.end_line - 1].strip():
                g.end_line -= 1
                g.content = "\n".join(lines[g.start_line - 1 : g.end_line])
        return [g for g in gaps if g.content.strip()]

    # ------------------------------------------------------------------
    # Python (AST)
    # ------------------------------------------------------------------

    @staticmethod
    def _node_start(node: ast.AST) -> int:
        starts = [node.lineno]  # type: ignore[attr-defined]
        for dec in getattr(node, "decorator_list", []):
            starts.append(dec.lineno)
        return min(starts)

    @staticmethod
    def _python_imports(tree: ast.Module) -> list[str]:
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in imports:
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = "." * node.level + (node.module or "")
                if mod and mod not in imports:
                    imports.append(mod)
        return imports

    def _chunk_python(self, file_path: str, content: str) -> list[Chunk]:
        lines = content.split("\n")
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._chunk_fallback(file_path, content, "python")
        imports = self._python_imports(tree)
        module_name = Path(file_path).stem
        module_ctx = f"File: {file_path} | Module: {module_name}"
        chunks: list[Chunk] = []
        covered: list[tuple[int, int]] = []

        def emit_function(node, class_name: str | None) -> None:
            start, end = self._node_start(node), node.end_lineno
            if class_name:
                ctx = f"File: {file_path} | Class: {class_name} | Method: {node.name}"
                symbol_type = "method"
            else:
                ctx = f"File: {file_path} | Function: {node.name}"
                symbol_type = "function"
            chunks.append(
                self._make(file_path, "python", lines, start, end,
                           node.name, symbol_type, ctx, imports)
            )
            covered.append((start, end))

        def emit_class(node: ast.ClassDef, prefix: str = "") -> None:
            qual = f"{prefix}{node.name}"
            start, end = self._node_start(node), node.end_lineno
            methods = [
                n for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            nested = [n for n in node.body if isinstance(n, ast.ClassDef)]
            if methods or nested:
                first_child = min(
                    self._node_start(n) for n in (*methods, *nested)
                )
                skeleton_end = first_child - 1
                while skeleton_end > start and not lines[skeleton_end - 1].strip():
                    skeleton_end -= 1
            else:
                skeleton_end = end
            ctx = f"File: {file_path} | Class: {qual}"
            chunks.append(
                self._make(file_path, "python", lines, start, skeleton_end,
                           qual, "class", ctx, imports)
            )
            covered.append((start, skeleton_end))
            for method in methods:
                emit_function(method, qual)
            for child in nested:
                emit_class(child, prefix=f"{qual}.")

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit_function(node, None)
            elif isinstance(node, ast.ClassDef):
                emit_class(node)

        # Module header (docstring + imports) and any top-level code
        # between symbols are the uncovered ranges.
        chunks.extend(
            self._gap_chunks(file_path, "python", lines, covered,
                             module_name, module_ctx, imports)
        )
        return chunks

    # ------------------------------------------------------------------
    # Brace languages (TS/JS, C++, Go, Java)
    # ------------------------------------------------------------------

    @staticmethod
    def _brace_imports(content: str, language: str) -> list[str]:
        imports: list[str] = []
        patterns = [
            r"""^\s*import\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]""",  # TS/JS
            r"^\s*#include\s*[<\"]([^>\"]+)[>\"]",  # C++
            r"^\s*import\s+(?:\w+\s+)?\"([^\"]+)\"",  # Go
            r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;",  # Java
            r"^\s*using\s+([\w.]+)\s*;",  # C++ using
        ]
        for line in content.split("\n"):
            for pat in patterns:
                m = re.match(pat, line)
                if m and m.group(1) not in imports:
                    imports.append(m.group(1))
        return imports

    @staticmethod
    def _match_top_symbol(line: str, language: str):
        """Return (kind, name, receiver) for a symbol starting on this
        line, where kind is 'class' or 'function', else None."""
        m = _CLASS_RE.match(line)
        if m:
            return ("class", m.group(2), None)
        if language == "go":
            m = _GO_TYPE_RE.match(line)
            if m:
                return ("class", m.group(1), None)
            m = _GO_FUNC_RE.match(line)
            if m:
                receiver = "(" in line.split(m.group(1))[0].split("func", 1)[-1]
                return ("function", m.group(1), receiver)
        if language in ("typescript", "javascript"):
            m = _TS_FUNCTION_RE.match(line) or _TS_ARROW_RE.match(line)
            if m:
                return ("function", m.group(1), None)
        m = _GENERIC_FUNC_RE.match(line)
        if m and m.group(1) not in _CONTROL_KEYWORDS:
            # C++ destructors / operators etc. are intentionally skipped.
            return ("function", m.group(1), None)
        return None

    @staticmethod
    def _match_method(line: str, class_name: str, language: str):
        """Detect a method signature at class-body depth. Returns the
        method name or None."""
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*", "#")):
            return None
        if language in ("typescript", "javascript"):
            m = re.match(
                r"^(?:(?:public|private|protected|static|async|readonly|"
                r"override|abstract|declare)\s+)*"
                r"(?:get\s+|set\s+)?([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(",
                stripped,
            )
            if m and m.group(1) not in _CONTROL_KEYWORDS and m.group(1) != class_name.split(".")[-1]:
                # Exclude property-arrow fields handled at any depth and
                # plain calls: require a following brace/colon-paren shape
                # or a modifier — plain `name(...)` calls end with ';' or
                # are expressions, handled by the caller via brace check.
                return m.group(1)
            # constructor is a real method chunk.
            m = re.match(r"^constructor\s*\(", stripped)
            if m:
                return "constructor"
            return None
        # Java / C++ / Go-style methods inside a class body.
        m = _GENERIC_FUNC_RE.match(line)
        if m and m.group(1) not in _CONTROL_KEYWORDS:
            return m.group(1)
        m = re.match(
            r"^\s*(?:(?:public|private|protected|static|final|virtual|override)\s+)*"
            r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:\{|$)",
            line,
        )
        if m and m.group(1) not in _CONTROL_KEYWORDS:
            return m.group(1)  # constructor-like (no return type)
        return None

    def _chunk_braces(self, file_path: str, content: str, language: str) -> list[Chunk]:
        lines = content.split("\n")
        imports = self._brace_imports(content, language)
        module_name = Path(file_path).stem
        module_ctx = f"File: {file_path} | Module: {module_name}"
        chunks: list[Chunk] = []
        covered: list[tuple[int, int]] = []

        i = 0
        while i < len(lines):
            match = self._match_top_symbol(lines[i], language)
            if not match:
                i += 1
                continue
            kind, name, receiver = match
            end_idx = _find_block_end(lines, i)
            if end_idx is None:
                i += 1
                continue
            start, end = i + 1, end_idx + 1  # to 1-based
            if kind == "function":
                if language == "go" and receiver:
                    ctx = f"File: {file_path} | Method: {name}"
                    symbol_type = "method"
                else:
                    ctx = f"File: {file_path} | Function: {name}"
                    symbol_type = "function"
                chunks.append(
                    self._make(file_path, language, lines, start, end,
                               name, symbol_type, ctx, imports)
                )
                covered.append((start, end))
            else:  # class / interface / struct / enum
                self._emit_brace_class(
                    file_path, language, lines, start, end, name,
                    imports, chunks, covered,
                )
            i = end_idx + 1

        chunks.extend(
            self._gap_chunks(file_path, language, lines, covered,
                             module_name, module_ctx, imports)
        )
        return chunks

    def _emit_brace_class(
        self, file_path, language, lines, start, end, name,
        imports, chunks, covered,
    ) -> None:
        # Local coverage (skeleton + methods) drives the in-class gap
        # recovery below; the caller-facing `covered` gets the class's
        # *whole* range at the end, so the class's closing brace never
        # leaks out as a stray global module_header chunk.
        local_covered: list[tuple[int, int]] = []
        ctx = f"File: {file_path} | Class: {name}"
        # Walk the class body tracking relative brace depth; methods
        # live exactly at depth 1.
        methods: list[tuple[int, int, str]] = []  # 1-based start, end, name
        in_block = False
        depth = 0
        idx = start - 1
        # Advance to the class's opening brace line.
        while idx < end - 1:
            delta, in_block = _sanitized_brace_delta(lines[idx], in_block)
            depth += delta
            idx += 1
            if depth >= 1:
                break
        while idx < end - 1 and depth >= 1:
            line = lines[idx]
            if depth == 1:
                method_name = self._match_method(line, name, language)
                if method_name:
                    mend = _find_block_end(lines, idx)
                    if mend is not None and mend <= end - 1:
                        methods.append((idx + 1, mend + 1, method_name))
                        # Skip past the method body.
                        while idx <= mend:
                            delta, in_block = _sanitized_brace_delta(lines[idx], in_block)
                            depth += delta
                            idx += 1
                        continue
            delta, in_block = _sanitized_brace_delta(line, in_block)
            depth += delta
            idx += 1

        if not methods:
            chunks.append(
                self._make(file_path, language, lines, start, end,
                           name, "class", ctx, imports)
            )
            covered.append((start, end))
            return

        skeleton_end = methods[0][0] - 1
        while skeleton_end > start and not lines[skeleton_end - 1].strip():
            skeleton_end -= 1
        chunks.append(
            self._make(file_path, language, lines, start, skeleton_end,
                       name, "class", ctx, imports)
        )
        local_covered.append((start, skeleton_end))
        for m_start, m_end, m_name in methods:
            m_ctx = f"File: {file_path} | Class: {name} | Method: {m_name}"
            chunks.append(
                self._make(file_path, language, lines, m_start, m_end,
                           m_name, "method", m_ctx, imports)
            )
            local_covered.append((m_start, m_end))
        # Class-level members sitting between/after methods (fields,
        # nested types) are recovered as class-context gap chunks.
        inner_gaps = self._gap_chunks(
            file_path, language,
            lines[start - 1 : end],
            [(s - start + 1, e - start + 1) for (s, e) in local_covered
             if start <= s and e <= end],
            name, ctx, imports, symbol_type="class",
        )
        for g in inner_gaps:
            # _gap_chunks worked on the sliced line list — rebase it.
            g.start_line += start - 1
            g.end_line += start - 1
            # The class closing brace alone is not a useful chunk.
            if g.content.strip() in ("}", "};"):
                continue
            chunks.append(g)
            local_covered.append((g.start_line, g.end_line))
        # Claim the whole class range globally (see local_covered note).
        covered.append((start, end))

    # ------------------------------------------------------------------
    # Markdown / RST
    # ------------------------------------------------------------------

    def _chunk_docs(self, file_path: str, content: str, language: str) -> list[Chunk]:
        lines = content.split("\n")
        # (line_idx 0-based, level, title)
        headings: list[tuple[int, int, str]] = []
        for idx, line in enumerate(lines):
            m = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if m:
                headings.append((idx, len(m.group(1)), m.group(2).strip()))
                continue
            if language == "rst" and idx + 1 < len(lines):
                under = lines[idx + 1].strip()
                if (
                    line.strip()
                    and len(under) >= 3
                    and len(set(under)) == 1
                    and under[0] in "=-~^"
                    and len(under) >= len(line.strip())
                ):
                    level = {"=": 1, "-": 2, "~": 3, "^": 4}[under[0]]
                    headings.append((idx, level, line.strip()))
        chunks: list[Chunk] = []
        if not headings:
            body = content.strip()
            if body:
                chunks.append(
                    Chunk(file_path, language, Path(file_path).stem, "doc",
                          1, len(lines), content,
                          f"File: {file_path} | Document", [])
                )
            return chunks

        # Preamble before the first heading.
        if headings[0][0] > 0 and "\n".join(lines[: headings[0][0]]).strip():
            chunks.append(
                self._make(file_path, language, lines, 1, headings[0][0],
                           Path(file_path).stem, "doc",
                           f"File: {file_path} | Document", [])
            )
        stack: list[tuple[int, str]] = []
        for pos, (idx, level, title) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            section_path = " > ".join(t for _, t in stack)
            end_idx = (headings[pos + 1][0] - 1) if pos + 1 < len(headings) else len(lines) - 1
            ctx = f"File: {file_path} | Section: {section_path}"
            chunks.append(
                self._make(file_path, language, lines, idx + 1, end_idx + 1,
                           title, "doc", ctx, [])
            )
        return chunks

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _chunk_fallback(self, file_path: str, content: str, language: str) -> list[Chunk]:
        lines = content.split("\n")
        module_name = Path(file_path).stem
        ctx = f"File: {file_path} | Module: {module_name}"
        if not content.strip():
            return []
        return [
            self._make(file_path, language, lines, 1, len(lines),
                       module_name, "module_header", ctx, [])
        ]
