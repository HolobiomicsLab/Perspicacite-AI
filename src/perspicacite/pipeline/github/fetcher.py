"""GitHub repository fetcher — URL parsing, tarball download, on-disk
SHA cache, and ``git clone`` fallback for rate-limited intakes.

This module is the lowest-level building block of the 2026-05-15
GitHub-repo / skill-bundle ingest pipeline. Higher layers
(``bundle.py`` for manifest parsing, ``chunk_producer.py`` for file
chunking, ``github_kb.py`` for KB orchestration) build on top of the
``(root_path, sha)`` tuple that :meth:`GitHubFetcher.fetch` returns.

Design references:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md``

Why both tarball *and* clone?
  GitHub's REST tarball endpoint is fast, single-request, and works
  for any commit. But unauthenticated callers get 60 req/hr and even
  authenticated callers can be throttled on 5xx storms. The
  ``git clone --depth=1`` fallback uses the public HTTPS git endpoint
  (separate rate-limit pool) so an ingest still completes when the
  REST API is throttling us.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

__all__ = [
    "FetcherError",
    "GitHubFetcher",
    "RateLimitedError",
    "RepoRef",
    "parse_repo_url",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FetcherError(RuntimeError):
    """Generic fetcher failure (network error, 404, malformed response).

    Callers should treat this as a hard fail for *this* ref; retrying
    the same ref typically won't help. Distinct from
    :class:`RateLimitedError`, which the orchestrator catches and
    routes to the clone fallback.
    """


class RateLimitedError(FetcherError):
    """The GitHub REST API throttled us.

    ``reset_at`` is the Unix epoch second when the rate-limit window
    resets, parsed from the ``X-RateLimit-Reset`` response header. The
    orchestrator (:meth:`GitHubFetcher.fetch`) catches this and falls
    through to ``git clone``, which uses a separate quota pool.
    """

    def __init__(self, message: str, *, reset_at: int) -> None:
        super().__init__(message)
        self.reset_at = reset_at


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoRef:
    """Reference to a GitHub repository (and optionally a sub-tree).

    Attributes
    ----------
    org : str
        GitHub user or organisation that owns the repository.
    repo : str
        Repository name. The ``.git`` suffix is stripped at parse time.
    ref : str | None
        A branch name or commit SHA. ``None`` means "use the default
        branch" — the fetcher resolves that via ``HEAD`` on the GitHub
        commits endpoint.
    subpath : str | None
        Path within the repository tree (set only for ``/tree/<branch>/<dir>``
        URLs). Bundle parsers use this to scope ingest to a sub-directory.
    """

    org: str
    repo: str
    ref: str | None
    subpath: str | None


def parse_repo_url(url: str) -> RepoRef:
    """Parse a GitHub repository URL into a :class:`RepoRef`.

    Accepted forms::

        https://github.com/<org>/<repo>
        https://github.com/<org>/<repo>.git
        https://github.com/<org>/<repo>@<branch-or-sha>
        https://github.com/<org>/<repo>/tree/<branch>
        https://github.com/<org>/<repo>/tree/<branch>/<subpath...>

    ``/blob/<branch>/<file>`` URLs are rejected with :class:`ValueError`
    because they target a single file, not a directory tree.

    Raises
    ------
    ValueError
        If the host is not ``github.com``, if the org/repo segments
        are missing, or if a ``/blob/`` URL is supplied.
    """

    if not isinstance(url, str) or not url:
        raise ValueError(f"empty or non-string URL: {url!r}")

    # Pull out a possible "@<ref>" suffix BEFORE urlparse — the "@" can
    # be ambiguous in netlocs (``user@host``) and we want it to apply to
    # the path tail.
    at_ref: str | None = None
    if "@" in url and "://" in url:
        # Only allow ``@`` after the host part of the URL.
        scheme_end = url.index("://") + 3
        host_end_candidates = [
            url.find("/", scheme_end),
            url.find("?", scheme_end),
        ]
        host_end = min((c for c in host_end_candidates if c != -1), default=-1)
        if host_end != -1 and "@" in url[host_end:]:
            head, _, at_ref = url.rpartition("@")
            url = head

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"unsupported URL scheme {parsed.scheme!r} (expected http/https)"
        )
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError(
            f"not a github.com URL (got host {parsed.netloc!r})"
        )

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"URL must include /<org>/<repo>, got path {parsed.path!r}"
        )

    org, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    ref: str | None = at_ref
    subpath: str | None = None

    if len(parts) >= 3:
        kind = parts[2]
        if kind == "blob":
            raise ValueError(
                f"/blob/ URLs point at a single file, not a directory: {url!r}"
            )
        if kind == "tree":
            if at_ref is not None:
                raise ValueError(
                    "cannot combine '@<ref>' with '/tree/' in the same URL"
                )
            if len(parts) >= 4:
                ref = parts[3]
            if len(parts) >= 5:
                subpath = "/".join(parts[4:])

    return RepoRef(org=org, repo=repo, ref=ref, subpath=subpath)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class GitHubFetcher:
    """Async client that materialises a repository commit on disk.

    Uses the GitHub REST API for SHA resolution and tarball download,
    falling back to ``git clone --depth=1`` when the REST API is
    rate-limited. The on-disk cache is keyed by commit SHA, so calling
    :meth:`fetch` repeatedly for the same ref is a single network
    round-trip (the ``commits/<ref>`` lookup) plus a directory lookup.

    Parameters
    ----------
    token : str, optional
        Personal-access token. When supplied, the fetcher sends
        ``Authorization: Bearer <token>`` on REST calls and uses
        ``https://<token>@github.com/...`` for the clone fallback so
        private repos work end-to-end.
    cache_dir : pathlib.Path
        Root of the on-disk cache. Each commit lives under
        ``cache_dir/<sha>/`` (the directory contents are the repo
        root). Created on first use.
    cache_max_mb : int
        Soft ceiling on cache size in MB (advisory; not enforced in v1
        — listed here so the constructor signature matches the config
        knob and so a future eviction policy has a place to land).
    user_agent : str
        Sent on every REST request. GitHub requires a non-empty
        ``User-Agent``.
    api_base : str
        Override for the REST API root. Always ``https://api.github.com``
        in production; tests can swap it.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        cache_dir: Path,
        cache_max_mb: int = 2048,
        user_agent: str = "Perspicacite/2.0",
        api_base: str = "https://api.github.com",
    ) -> None:
        self._token = token
        self._cache_dir = Path(cache_dir)
        self._cache_max_mb = cache_max_mb
        self._user_agent = user_agent
        self._api_base = api_base.rstrip("/")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {
            "User-Agent": self._user_agent,
            "Accept": "application/vnd.github+json",
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    @staticmethod
    def _check_rate_limit(response: httpx.Response) -> None:
        """Raise :class:`RateLimitedError` if the response is a
        throttle (403 or 429 with ``X-RateLimit-Reset``)."""

        if response.status_code in (403, 429):
            reset = response.headers.get("X-RateLimit-Reset")
            if reset is not None:
                try:
                    reset_at = int(reset)
                except ValueError as exc:
                    raise FetcherError(
                        f"malformed X-RateLimit-Reset header: {reset!r}"
                    ) from exc
                raise RateLimitedError(
                    f"GitHub rate-limited (status {response.status_code})",
                    reset_at=reset_at,
                )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path_for(self, sha: str) -> Path:
        return self._cache_dir / sha

    # Filename written into ``cache_dir/<sha>/`` once the tarball
    # extract loop (or the clone+checkout) has finished successfully.
    # ``_is_valid_cache_entry`` requires its presence — without it a
    # mid-extract cancellation would leave a partial directory that the
    # next fetch would happily serve as a "cache hit".
    _CACHE_SENTINEL = ".complete"

    @classmethod
    def _is_valid_cache_entry(cls, path: Path) -> bool:
        """A cache entry is valid iff it exists, is non-empty, AND
        contains the ``.complete`` sentinel.

        The sentinel is written ONCE the extract loop (tarball) or
        clone+checkout (git fallback) has finished successfully. A
        cache entry without the sentinel is treated as corrupt — for
        example, an extraction cancelled after writing 5 of 50 files
        leaves a non-empty dir that the old ``any(path.iterdir())``
        check would have accepted as valid.
        """

        if not path.is_dir():
            return False
        if not (path / cls._CACHE_SENTINEL).is_file():
            return False
        # Sanity: an entry with ONLY the sentinel and nothing else is
        # not actually a usable repo.
        return any(child.name != cls._CACHE_SENTINEL for child in path.iterdir())

    @classmethod
    def _purge_cache_entry(cls, path: Path) -> None:
        """Remove an invalid cache entry (partial extract, missing
        sentinel, etc). Safe to call on a path that does not exist."""

        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve_commit_sha(self, ref: RepoRef) -> str:
        """Resolve ``ref`` (branch name, tag, or SHA) to a full commit SHA.

        Implementation: ``GET /repos/{org}/{repo}/commits/{ref}`` —
        GitHub returns the commit object for the ref tip. When
        ``ref.ref is None`` we use the literal ``HEAD`` so the API
        picks the repo's default branch (handles forks renamed
        ``main → default`` etc.).
        """

        rev = ref.ref or "HEAD"
        url = f"{self._api_base}/repos/{ref.org}/{ref.repo}/commits/{rev}"
        async with httpx.AsyncClient(headers=self._headers(), timeout=30.0) as client:
            try:
                response = await client.get(url)
            except httpx.RequestError as exc:
                raise FetcherError(f"network error fetching {url}: {exc}") from exc
        self._check_rate_limit(response)
        if response.status_code == 404:
            raise FetcherError(f"repo or ref not found: {ref.org}/{ref.repo}@{rev}")
        if response.status_code >= 400:
            raise FetcherError(
                f"GitHub returned {response.status_code} for {url}: "
                f"{response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise FetcherError(f"non-JSON commits response from {url}") from exc
        sha = payload.get("sha")
        if not isinstance(sha, str) or not sha:
            raise FetcherError(f"commits response missing 'sha' field: {payload!r}")
        return sha

    async def fetch_tarball(self, ref: RepoRef, *, sha: str) -> Path:
        """Download the repo tarball for ``sha``, extract under
        ``cache_dir/<sha>/``, and return that path.

        Cache hits (a directory under ``cache_dir/<sha>/`` carrying the
        :attr:`_CACHE_SENTINEL` marker) return immediately without
        re-downloading. Stale partial extracts (no sentinel) are cleaned
        up and re-fetched.
        """

        target = self._cache_path_for(sha)
        if self._is_valid_cache_entry(target):
            return target
        # Either no entry, an empty partial-extract dir, or a non-empty
        # partial-extract dir missing the sentinel. All are invalid.
        self._purge_cache_entry(target)

        url = f"{self._api_base}/repos/{ref.org}/{ref.repo}/tarball/{sha}"
        async with httpx.AsyncClient(
            headers=self._headers(),
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            try:
                response = await client.get(url)
            except httpx.RequestError as exc:
                raise FetcherError(f"network error fetching tarball: {exc}") from exc
        self._check_rate_limit(response)
        if response.status_code == 404:
            raise FetcherError(
                f"tarball not found: {ref.org}/{ref.repo}@{sha}"
            )
        if response.status_code >= 400:
            raise FetcherError(
                f"GitHub returned {response.status_code} for {url}"
            )

        target.mkdir(parents=True, exist_ok=True)
        try:
            self._extract_tarball(response.content, target)
        except (tarfile.TarError, OSError) as exc:
            self._purge_cache_entry(target)
            raise FetcherError(f"failed to extract tarball: {exc}") from exc

        # Mark the extract complete BEFORE running the validity check;
        # the check requires the sentinel to be present.
        (target / self._CACHE_SENTINEL).touch()

        if not self._is_valid_cache_entry(target):
            self._purge_cache_entry(target)
            raise FetcherError("tarball extracted to an empty directory")

        return target

    @staticmethod
    def _extract_tarball(tar_bytes: bytes, target: Path) -> None:
        """Extract a GitHub-style tarball (one top-level directory) into
        ``target``, stripping the wrapper directory so the repo root
        lives directly under ``target/``.

        GitHub tarballs always have shape ``<org>-<repo>-<short_sha>/``
        at the top level; we use ``Path.parts`` to walk past it rather
        than parsing the directory name.

        Path-traversal hardening
        ------------------------
        Two layers of defence:

        1. Cheap *string* check: skip members whose name starts with
           ``/`` or contains a ``..`` component. Catches the
           overwhelming majority of malicious tarballs without
           resolving anything on disk.
        2. Resolved-path check: even after stripping the wrapper, the
           final destination must ``resolve()`` to a path that is
           ``is_relative_to(target.resolve())``. Catches edge cases
           that slip past the string filter (e.g. clever component
           interleavings, future loosening of the string check).
        """

        target_resolved = target.resolve()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            for member in tf.getmembers():
                # Layer 1: cheap string-level rejection.
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    continue
                parts = Path(member.name).parts
                if len(parts) <= 1:
                    # Top-level wrapper directory itself — skip.
                    continue
                rel = Path(*parts[1:])
                dest = target / rel
                # Layer 2: resolved-path check. Reject any member whose
                # *resolved* destination is not inside the target tree.
                try:
                    resolved_dest = dest.resolve()
                except (OSError, RuntimeError):
                    # Symlink loops, missing parents, etc.
                    continue
                if not resolved_dest.is_relative_to(target_resolved):
                    continue
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    # Skip symlinks/devices for safety.
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

    async def fetch_clone(self, ref: RepoRef, *, sha: str) -> Path:
        """Shallow ``git clone`` fallback when the REST API is throttled.

        Uses HTTPS so unauthenticated clones still work. When a token
        is configured it is passed via ``-c http.extraHeader=...``
        rather than baked into the clone URL — the URL would otherwise
        appear in ``git``'s stderr on failure and leak the token.
        Checks out the explicit ``sha`` to keep the cache key stable
        across re-runs.
        """

        target = self._cache_path_for(sha)
        if self._is_valid_cache_entry(target):
            return target
        # Any existing entry without the sentinel is a partial extract;
        # purge it before retrying.
        self._purge_cache_entry(target)

        target.parent.mkdir(parents=True, exist_ok=True)

        clone_url = f"https://github.com/{ref.org}/{ref.repo}.git"
        argv: list[str] = ["git"]
        if self._token:
            # Pass auth as an HTTP header rather than in the URL so
            # git never echoes the token in error messages. The header
            # value lives in argv (visible in ``ps``), which is still
            # a risk, but a far smaller one than URL-leaking.
            argv += [
                "-c",
                f"http.extraHeader=Authorization: Bearer {self._token}",
            ]
        argv += ["clone", "--depth=1", clone_url, str(target)]

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            self._purge_cache_entry(target)
            raise FetcherError(
                f"git clone failed (rc={proc.returncode}): "
                f"{self._scrub_secrets(stderr.decode('utf-8', 'replace'))[:500]}"
            )

        # If the clone landed on a different SHA (the default branch
        # has moved since ``resolve_commit_sha``), fetch & check out the
        # explicit SHA to keep cache keys stable.
        head_sha = await self._git_head_sha(target)
        if head_sha and head_sha != sha:
            await self._git_run(target, "fetch", "--depth=1", "origin", sha)
            await self._git_run(target, "checkout", sha)

        # Mark the cloned working tree as a complete extract.
        (target / self._CACHE_SENTINEL).touch()

        if not self._is_valid_cache_entry(target):
            self._purge_cache_entry(target)
            raise FetcherError("git clone produced an empty working tree")

        return target

    def _scrub_secrets(self, text: str) -> str:
        """Strip any literal occurrences of ``self._token`` from
        ``text`` before it lands in an error message or log line.

        Defense-in-depth: even with the ``-c http.extraHeader`` change,
        a misconfigured upstream layer (e.g. a stored credential
        helper, an old git config) could still echo the token. We
        replace it with a redaction marker so the rest of the error
        stays useful for debugging.
        """

        if not self._token:
            return text
        return text.replace(self._token, "<redacted-token>")

    @staticmethod
    async def _git_head_sha(repo: Path) -> str | None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return out.decode("utf-8", "replace").strip() or None

    async def _git_run(self, repo: Path, *args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise FetcherError(
                f"git {' '.join(args)} failed (rc={proc.returncode}): "
                f"{self._scrub_secrets(stderr.decode('utf-8', 'replace'))[:500]}"
            )

    async def fetch(self, ref: RepoRef) -> tuple[Path, str]:
        """High-level orchestration: SHA resolution → cache hit → tarball → clone.

        Returns the cache root path and the resolved commit SHA. The
        orchestrator is the only entry point higher layers need.
        """

        sha = await self.resolve_commit_sha(ref)
        cached = self._cache_path_for(sha)
        if self._is_valid_cache_entry(cached):
            return cached, sha
        try:
            path = await self.fetch_tarball(ref, sha=sha)
        except RateLimitedError:
            path = await self.fetch_clone(ref, sha=sha)
        return path, sha
