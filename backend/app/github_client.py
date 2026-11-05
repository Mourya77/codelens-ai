"""In-memory GitHub REST API client for CodeLens AI.

Nothing here touches disk: no ``git clone``, no temp files. Repository
metadata, the recursive git tree and raw file contents are all fetched
over HTTP with ``httpx.AsyncClient`` and held in memory only — the
deployment target has an ephemeral filesystem.

Resilience:
* ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset`` (and ``Retry-After``)
  are inspected on *every* response and tracked on the rate limiter.
* HTTP 403 / 429 responses are retried with exponential backoff +
  jitter; a rate-limit reset time, when sooner, wins over the backoff.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import httpx

GITHUB_API_BASE = "https://api.github.com"

# Files larger than this are never fetched.
MAX_FILE_BYTES = 150 * 1024

# Raw-content fetches run at most this many at a time.
FETCH_CONCURRENCY = 10

IGNORED_DIR_NAMES = {"node_modules", "dist", ".git", "__pycache__", ".venv", "venv"}

LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "gemfile.lock",
    "cargo.lock",
    "composer.lock",
    "pipfile.lock",
}

# Extensions that are never source/documentation text.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg",
    ".tif", ".tiff", ".avif",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".o", ".a",
    ".class", ".jar", ".war", ".pyc", ".pyo", ".wasm",
    ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav", ".ogg",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".db", ".sqlite", ".sqlite3", ".parquet", ".pickle", ".pkl",
    ".npy", ".npz", ".onnx", ".pt", ".pth",
}


class GitHubAPIError(RuntimeError):
    """A non-retryable GitHub API failure (or retries exhausted)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubRateLimitError(GitHubAPIError):
    """Rate limit still in force after all retries."""


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Parse the common GitHub repo reference forms into (owner, name).

    Accepts ``https://github.com/owner/repo(.git)``, ``github.com/owner/repo``,
    ``git@github.com:owner/repo.git`` and plain ``owner/repo``.
    """
    value = repo_url.strip()
    value = re.sub(r"^git@github\.com:", "github.com/", value)
    value = re.sub(r"^(https?://|ssh://git@)?", "", value)
    value = value.removeprefix("github.com/")
    value = value.strip("/")
    if value.endswith(".git"):
        value = value[: -len(".git")]
    parts = [p for p in value.split("/") if p]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Cannot parse GitHub repository from {repo_url!r}; "
            "expected 'owner/repo' or a github.com URL."
        )
    return parts[0], parts[1]


@dataclass
class RepoMetadata:
    owner: str
    name: str
    full_name: str
    html_url: str
    default_branch: str
    commit_sha: str  # latest commit on the default branch
    description: str | None = None


@dataclass
class TreeEntry:
    """One blob from the recursive git tree."""

    path: str
    size: int | None = None
    sha: str | None = None

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name


@dataclass
class FileContent:
    path: str
    content: str

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8", errors="ignore"))


@dataclass
class RateLimitState:
    """Last-seen GitHub rate-limit budget, updated from response headers."""

    limit: int | None = None
    remaining: int | None = None
    reset_epoch: int | None = None  # X-RateLimit-Reset (seconds)
    used: int | None = None

    def update_from_headers(self, headers: httpx.Headers) -> None:
        def _int(name: str) -> int | None:
            raw = headers.get(name)
            try:
                return int(raw) if raw is not None else None
            except ValueError:
                return None

        for attr, header in (
            ("limit", "x-ratelimit-limit"),
            ("remaining", "x-ratelimit-remaining"),
            ("reset_epoch", "x-ratelimit-reset"),
            ("used", "x-ratelimit-used"),
        ):
            value = _int(header)
            if value is not None:
                setattr(self, attr, value)

    def seconds_until_reset(self, now: float | None = None) -> float:
        if self.reset_epoch is None:
            return 0.0
        return max(0.0, self.reset_epoch - (now if now is not None else time.time()))


def is_indexable_file(entry: TreeEntry, max_bytes: int = MAX_FILE_BYTES) -> bool:
    """The Phase 2 file filter: no binaries/images, no lockfiles, no
    vendored/build output, no minified bundles, nothing over 150KB."""
    path = PurePosixPath(entry.path)
    parts = set(path.parts[:-1])
    if parts & IGNORED_DIR_NAMES:
        return False
    name = path.name.lower()
    if name in LOCKFILE_NAMES:
        return False
    if name.endswith((".min.js", ".min.css", ".min.mjs")):
        return False
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False
    if entry.size is not None and entry.size > max_bytes:
        return False
    return True


class GitHubClient:
    """Async GitHub REST client with rate-limit-aware retries.

    ``client`` and ``sleep`` are injectable so tests can run the whole
    client against ``httpx.MockTransport`` with no real waiting.
    """

    def __init__(
        self,
        token: str = "",
        *,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
        max_wait_seconds: float = 60.0,
        concurrency: int = FETCH_CONCURRENCY,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        self._token = token or ""
        self._owns_client = client is None
        self._client = client or self._default_client()
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.max_wait_seconds = max_wait_seconds
        self.concurrency = concurrency
        self._sleep = sleep or asyncio.sleep
        # Full jitter in [0, ceiling]; injectable for deterministic tests.
        self._jitter = jitter or (lambda ceiling: random.uniform(0, ceiling))
        self.rate_limit = RateLimitState()

    @staticmethod
    def _default_client() -> httpx.AsyncClient:
        # No trust_env (see config.env_proxy_url for why); the egress
        # proxy, when the environment defines one, is passed explicitly.
        # verify uses the *system* CA context (honouring SSL_CERT_FILE)
        # rather than httpx's bundled certifi store, so TLS-intercepting
        # corporate/sandbox proxies with a locally installed CA work.
        import ssl

        from .config import env_proxy_url

        kwargs: dict = {
            "base_url": GITHUB_API_BASE,
            "timeout": httpx.Timeout(30.0),
            "follow_redirects": True,
            "trust_env": False,
            "verify": ssl.create_default_context(),
        }
        proxy = env_proxy_url()
        if proxy:
            try:
                return httpx.AsyncClient(proxy=proxy, **kwargs)
            except Exception:
                pass  # unusable proxy value — fall back to direct
        return httpx.AsyncClient(**kwargs)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- low-level request with rate-limit handling -----------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "CodeLensAI/0.2",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """How long to wait before the next attempt after a 403/429."""
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), self.max_wait_seconds)
            except ValueError:
                pass
        # Primary rate limit exhausted: wait for the reset window if it
        # is sooner than the (capped) exponential backoff.
        if self.rate_limit.remaining == 0:
            until_reset = self.rate_limit.seconds_until_reset()
            if 0 < until_reset <= self.max_wait_seconds:
                return until_reset
        ceiling = self.backoff_base_seconds * (2**attempt)
        return min(self._jitter(ceiling) if ceiling else 0.0, self.max_wait_seconds)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        url = path if path.startswith("http") else f"{GITHUB_API_BASE}{path}"
        last_response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method, url, headers=self._headers(headers), params=params
                )
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise GitHubAPIError(f"GitHub request failed: {exc}") from exc
                await self._sleep(self._retry_delay_for_exception(attempt))
                continue
            # Rate-limit bookkeeping on EVERY response.
            self.rate_limit.update_from_headers(response.headers)
            if response.status_code in (403, 429):
                last_response = response
                if attempt >= self.max_retries:
                    break
                await self._sleep(self._retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                detail = ""
                try:
                    detail = str(response.json().get("message", ""))
                except Exception:
                    detail = response.text[:200]
                raise GitHubAPIError(
                    f"GitHub API {response.status_code} for {method} {path}: {detail}",
                    status_code=response.status_code,
                )
            return response
        raise GitHubRateLimitError(
            f"GitHub rate limit still in force after {self.max_retries} retries "
            f"(remaining={self.rate_limit.remaining}, "
            f"reset in {self.rate_limit.seconds_until_reset():.0f}s)",
            status_code=last_response.status_code if last_response else 429,
        )

    def _retry_delay_for_exception(self, attempt: int) -> float:
        ceiling = self.backoff_base_seconds * (2**attempt)
        return min(self._jitter(ceiling) if ceiling else 0.0, self.max_wait_seconds)

    # -- API operations -----------------------------------------------------

    async def get_repo_metadata(self, owner: str, repo: str) -> RepoMetadata:
        """Repo metadata + latest commit SHA on the default branch."""
        resp = await self._request("GET", f"/repos/{owner}/{repo}")
        data = resp.json()
        default_branch = data.get("default_branch") or "main"
        commit_resp = await self._request(
            "GET", f"/repos/{owner}/{repo}/commits/{default_branch}"
        )
        commit_sha = commit_resp.json().get("sha", "")
        return RepoMetadata(
            owner=data.get("owner", {}).get("login", owner),
            name=data.get("name", repo),
            full_name=data.get("full_name", f"{owner}/{repo}"),
            html_url=data.get("html_url", f"https://github.com/{owner}/{repo}"),
            default_branch=default_branch,
            commit_sha=commit_sha,
            description=data.get("description"),
        )

    async def get_file_tree(
        self, owner: str, repo: str, sha: str
    ) -> list[TreeEntry]:
        """The entire recursive tree in one call, filtered to indexable blobs."""
        resp = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/git/trees/{sha}",
            params={"recursive": "1"},
        )
        data = resp.json()
        entries = [
            TreeEntry(path=item["path"], size=item.get("size"), sha=item.get("sha"))
            for item in data.get("tree", [])
            if item.get("type") == "blob" and item.get("path")
        ]
        return [e for e in entries if is_indexable_file(e)]

    async def fetch_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> FileContent | None:
        """One file's raw content, in memory. None when the file turns
        out to be binary, oversized, or unfetchable."""
        try:
            resp = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{path}",
                headers={"Accept": "application/vnd.github.raw+json"},
                params={"ref": ref},
            )
        except GitHubAPIError:
            return None
        content = resp.text
        if "\x00" in content:  # binary payload that slipped the filters
            return None
        if len(content.encode("utf-8", errors="ignore")) > MAX_FILE_BYTES:
            return None
        return FileContent(path=path, content=content)

    async def fetch_files(
        self,
        owner: str,
        repo: str,
        entries: list[TreeEntry],
        ref: str,
    ) -> list[FileContent]:
        """Fetch many files concurrently (semaphore-capped), in memory,
        preserving tree order and skipping failed/binary files."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _one(entry: TreeEntry) -> FileContent | None:
            async with semaphore:
                return await self.fetch_file_content(owner, repo, entry.path, ref)

        results = await asyncio.gather(*(_one(e) for e in entries))
        return [r for r in results if r is not None]
