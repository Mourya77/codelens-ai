"""Tests for CodeAwareChunker.

The samples are built line-by-line (with the 1-based line number noted
in a trailing comment) so the line-number assertions are auditable at
a glance.
"""

from backend.app.chunker import CodeAwareChunker

# ---------------------------------------------------------------------------
# Sample: multi-class / multi-function Python file
# ---------------------------------------------------------------------------
PYTHON_SOURCE = "\n".join(
    [
        '"""Auth module."""',                                        # 1
        "",                                                          # 2
        "import os",                                                 # 3
        "import hashlib",                                            # 4
        "from typing import Optional",                              # 5
        "",                                                          # 6
        "",                                                          # 7
        "def hash_password(password: str) -> str:",                  # 8
        '    """Hash a password."""',                                # 9
        "    return hashlib.sha256(password.encode()).hexdigest()",  # 10
        "",                                                          # 11
        "",                                                          # 12
        "@register",                                                 # 13
        "class AuthManager:",                                        # 14
        '    """Manages authentication."""',                         # 15
        "",                                                          # 16
        "    TOKEN_TTL = 3600",                                      # 17
        "",                                                          # 18
        "    def __init__(self, secret: str):",                      # 19
        "        self.secret = secret",                              # 20
        "",                                                          # 21
        "    async def verify_token(self, token: str) -> Optional[bool]:",  # 22
        '        """Verify a token."""',                             # 23
        "        return bool(token) and token.startswith(self.secret)",     # 24
        "",                                                          # 25
        "",                                                          # 26
        "def module_helper() -> bool:",                              # 27
        "    return True",                                           # 28
    ]
)

# ---------------------------------------------------------------------------
# Sample: TypeScript file with an interface, a function and a class
# ---------------------------------------------------------------------------
TS_SOURCE = "\n".join(
    [
        'import { Injectable } from "./types";',                     # 1
        "",                                                          # 2
        "export interface User {",                                   # 3
        "  id: string;",                                             # 4
        "  name: string;",                                           # 5
        "}",                                                         # 6
        "",                                                          # 7
        "export function formatName(user: User): string {",          # 8
        "  return `${user.name} <${user.id}>`;",                     # 9
        "}",                                                         # 10
        "",                                                          # 11
        "export class UserService {",                                # 12
        "  private users: User[] = [];",                             # 13
        "",                                                          # 14
        "  constructor(private prefix: string) {}",                  # 15
        "",                                                          # 16
        "  async getUser(id: string): Promise<User | undefined> {",  # 17
        "    return this.users.find((u) => u.id === id);",           # 18
        "  }",                                                       # 19
        "",                                                          # 20
        "  addUser(user: User): void {",                             # 21
        "    this.users.push(user);",                               # 22
        "  }",                                                       # 23
        "}",                                                         # 24
    ]
)


def _by_symbol(chunks, name):
    matches = [c for c in chunks if c.symbol_name == name]
    assert matches, f"no chunk for symbol {name!r}; got {[c.symbol_name for c in chunks]}"
    return matches[0]


# --- Python ---------------------------------------------------------------

def test_python_top_level_function_is_intact():
    chunks = CodeAwareChunker().chunk_file("src/auth.py", PYTHON_SOURCE)
    fn = _by_symbol(chunks, "hash_password")
    assert fn.symbol_type == "function"
    assert (fn.start_line, fn.end_line) == (8, 10)
    assert fn.context_header == "File: src/auth.py | Function: hash_password"
    # Docstring and body survive intact, nothing from neighbours leaks in.
    assert '"""Hash a password."""' in fn.content
    assert "hexdigest()" in fn.content
    assert "class AuthManager" not in fn.content


def test_python_class_and_methods():
    chunks = CodeAwareChunker().chunk_file("src/auth.py", PYTHON_SOURCE)

    cls = _by_symbol(chunks, "AuthManager")
    assert cls.symbol_type == "class"
    # Decorator line counts as the class start; skeleton runs to the
    # last class-level line before the first method.
    assert (cls.start_line, cls.end_line) == (13, 17)
    assert cls.content.startswith("@register")
    assert '"""Manages authentication."""' in cls.content
    assert "TOKEN_TTL = 3600" in cls.content
    assert cls.context_header == "File: src/auth.py | Class: AuthManager"

    verify = _by_symbol(chunks, "verify_token")
    assert verify.symbol_type == "method"
    assert (verify.start_line, verify.end_line) == (22, 24)
    assert (
        verify.context_header
        == "File: src/auth.py | Class: AuthManager | Method: verify_token"
    )
    assert "async def verify_token" in verify.content
    assert '"""Verify a token."""' in verify.content

    init = _by_symbol(chunks, "__init__")
    assert init.symbol_type == "method"
    assert (init.start_line, init.end_line) == (19, 20)


def test_python_imports_and_module_header():
    chunks = CodeAwareChunker().chunk_file("src/auth.py", PYTHON_SOURCE)
    fn = _by_symbol(chunks, "hash_password")
    assert fn.imports == ["os", "hashlib", "typing"]

    headers = [c for c in chunks if c.symbol_type == "module_header"]
    assert headers, "expected a module_header chunk for docstring/imports"
    header = headers[0]
    assert (header.start_line, header.end_line) == (1, 5)
    assert '"""Auth module."""' in header.content
    assert "import hashlib" in header.content


def test_python_trailing_function_after_class():
    chunks = CodeAwareChunker().chunk_file("src/auth.py", PYTHON_SOURCE)
    helper = _by_symbol(chunks, "module_helper")
    assert helper.symbol_type == "function"
    assert (helper.start_line, helper.end_line) == (27, 28)


# --- TypeScript -------------------------------------------------------------

def test_typescript_function_is_intact():
    chunks = CodeAwareChunker().chunk_file("src/users.ts", TS_SOURCE)
    fn = _by_symbol(chunks, "formatName")
    assert fn.symbol_type == "function"
    assert (fn.start_line, fn.end_line) == (8, 10)
    assert fn.context_header == "File: src/users.ts | Function: formatName"
    # The chunk closes the block — brace balancing handled the ${...}
    # braces inside the template literal.
    assert fn.content.rstrip().endswith("}")
    assert "export class" not in fn.content


def test_typescript_interface_is_one_chunk():
    chunks = CodeAwareChunker().chunk_file("src/users.ts", TS_SOURCE)
    iface = _by_symbol(chunks, "User")
    assert iface.symbol_type == "class"
    assert (iface.start_line, iface.end_line) == (3, 6)
    assert "id: string;" in iface.content


def test_typescript_class_and_methods():
    chunks = CodeAwareChunker().chunk_file("src/users.ts", TS_SOURCE)

    cls = _by_symbol(chunks, "UserService")
    assert cls.symbol_type == "class"
    assert cls.start_line == 12
    assert "private users: User[] = [];" in cls.content

    get_user = _by_symbol(chunks, "getUser")
    assert get_user.symbol_type == "method"
    assert (get_user.start_line, get_user.end_line) == (17, 19)
    assert (
        get_user.context_header
        == "File: src/users.ts | Class: UserService | Method: getUser"
    )
    assert get_user.content.rstrip().endswith("}")
    assert "addUser" not in get_user.content

    add_user = _by_symbol(chunks, "addUser")
    assert add_user.symbol_type == "method"
    assert (add_user.start_line, add_user.end_line) == (21, 23)

    ctor = _by_symbol(chunks, "constructor")
    assert ctor.symbol_type == "method"
    assert (ctor.start_line, ctor.end_line) == (15, 15)

    assert get_user.imports == ["./types"]


# --- Markdown ---------------------------------------------------------------

def test_markdown_splits_by_heading_hierarchy():
    md = "\n".join(
        [
            "# Guide",          # 1
            "Intro text.",      # 2
            "## Install",       # 3
            "Run pip install.",  # 4
            "## Usage",         # 5
            "Use it.",          # 6
        ]
    )
    chunks = CodeAwareChunker().chunk_file("README.md", md)
    by_name = {c.symbol_name: c for c in chunks}
    assert by_name["Guide"].symbol_type == "doc"
    assert (by_name["Guide"].start_line, by_name["Guide"].end_line) == (1, 2)
    assert by_name["Install"].context_header == "File: README.md | Section: Guide > Install"
    assert (by_name["Install"].start_line, by_name["Install"].end_line) == (3, 4)
    assert by_name["Usage"].context_header == "File: README.md | Section: Guide > Usage"


# --- Oversized symbols --------------------------------------------------------

def test_oversized_function_is_subchunked_with_context_header():
    body = ["def big_function() -> int:"]
    body += [f'    value_{i} = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' for i in range(300)]
    body += ["    return 0"]
    source = "\n".join(body)

    chunks = CodeAwareChunker().chunk_file("src/big.py", source)
    parts = [c for c in chunks if c.symbol_name == "big_function"]
    assert len(parts) > 1, "a >1500-token function must be sub-chunked"
    for part in parts:
        # Every sub-chunk is prepended with its context header.
        assert part.content.startswith(part.context_header)
        assert part.context_header == "File: src/big.py | Function: big_function"
    # Sub-chunks are ordered and cover the whole function.
    assert parts[0].start_line == 1
    assert parts[-1].end_line == len(body)
    starts = [p.start_line for p in parts]
    assert starts == sorted(starts)
