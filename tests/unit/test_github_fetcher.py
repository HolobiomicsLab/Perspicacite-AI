"""Tests for ``perspicacite.pipeline.github.fetcher``.

Covers URL parsing, ``GitHubFetcher`` HTTP path (tarball download +
on-disk SHA cache + token + rate-limit), and the ``git clone`` fallback
when the tarball endpoint is rate-limited.

The fetcher is the foundation of the 2026-05-15 GitHub / skill-bundle
ingest design — see
``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
for the wider picture.
"""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from perspicacite.pipeline.github.fetcher import (
    FetcherError,
    GitHubFetcher,
    RateLimitedError,
    RepoRef,
    parse_repo_url,
)

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class TestParseRepoUrl:
    """Lossless decomposition of GitHub repository URLs into ``RepoRef``."""

    def test_basic_url(self) -> None:
        assert parse_repo_url("https://github.com/org/repo") == RepoRef(
            org="org", repo="repo", ref=None, subpath=None
        )

    def test_with_branch_via_at(self) -> None:
        r = parse_repo_url("https://github.com/org/repo@main")
        assert r == RepoRef(org="org", repo="repo", ref="main", subpath=None)

    def test_with_commit_sha_via_at(self) -> None:
        r = parse_repo_url("https://github.com/org/repo@abc1234")
        assert r.ref == "abc1234"
        assert r.subpath is None

    def test_with_tree_path(self) -> None:
        r = parse_repo_url(
            "https://github.com/org/repo/tree/main/bundles/scrna-qc"
        )
        assert r == RepoRef(
            org="org", repo="repo", ref="main", subpath="bundles/scrna-qc"
        )

    def test_tree_branch_only_no_subpath(self) -> None:
        r = parse_repo_url("https://github.com/org/repo/tree/dev")
        assert r == RepoRef(org="org", repo="repo", ref="dev", subpath=None)

    def test_blob_url_rejected(self) -> None:
        """``/blob/<branch>/<file>`` targets a file, not a directory; the
        skill-bundle pipeline only ingests directories."""
        with pytest.raises(ValueError, match="blob"):
            parse_repo_url("https://github.com/org/repo/blob/main/README.md")

    def test_malformed_no_github_host(self) -> None:
        with pytest.raises(ValueError, match=r"github\.com"):
            parse_repo_url("https://example.com/org/repo")

    def test_trailing_slash_stripped(self) -> None:
        r = parse_repo_url("https://github.com/org/repo/")
        assert r == RepoRef(org="org", repo="repo", ref=None, subpath=None)

    def test_dot_git_suffix_stripped(self) -> None:
        r = parse_repo_url("https://github.com/org/repo.git")
        assert r == RepoRef(org="org", repo="repo", ref=None, subpath=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tarball(files: dict[str, bytes], top_dir: str = "org-repo-abc1234") -> bytes:
    """Build a small in-memory tarball that mimics GitHub's
    ``codeload`` shape: a single top-level directory containing files."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=f"{top_dir}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GitHubFetcher.resolve_commit_sha
# ---------------------------------------------------------------------------


async def test_resolve_commit_sha_returns_sha(
    tmp_path: Path, respx_mock: Any
) -> None:
    respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/commits/main"
    ).mock(return_value=httpx.Response(200, json={"sha": "abc1234deadbeef"}))

    fetcher = GitHubFetcher(cache_dir=tmp_path / "cache")
    sha = await fetcher.resolve_commit_sha(
        RepoRef(org="org", repo="repo", ref="main", subpath=None)
    )
    assert sha == "abc1234deadbeef"


async def test_resolve_commit_sha_uses_head_when_ref_none(
    tmp_path: Path, respx_mock: Any
) -> None:
    """``ref=None`` resolves against the literal ``HEAD`` so the GitHub
    API picks the repo's default branch (works for forks renamed
    main→default etc.)."""

    respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/commits/HEAD"
    ).mock(return_value=httpx.Response(200, json={"sha": "feedface"}))

    fetcher = GitHubFetcher(cache_dir=tmp_path / "cache")
    sha = await fetcher.resolve_commit_sha(
        RepoRef(org="org", repo="repo", ref=None, subpath=None)
    )
    assert sha == "feedface"


async def test_resolve_commit_sha_404_raises_fetcher_error(
    tmp_path: Path, respx_mock: Any
) -> None:
    respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/missing/repo/commits/.*"
    ).mock(return_value=httpx.Response(404, json={"message": "Not Found"}))

    fetcher = GitHubFetcher(cache_dir=tmp_path / "cache")
    with pytest.raises(FetcherError):
        await fetcher.resolve_commit_sha(
            RepoRef(org="missing", repo="repo", ref=None, subpath=None)
        )


# ---------------------------------------------------------------------------
# GitHubFetcher token + rate-limit headers
# ---------------------------------------------------------------------------


async def test_token_passed_as_authorization_header(
    tmp_path: Path, respx_mock: Any
) -> None:
    route = respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/commits/main"
    ).mock(return_value=httpx.Response(200, json={"sha": "abc"}))

    fetcher = GitHubFetcher(token="ghp_secret", cache_dir=tmp_path / "cache")
    await fetcher.resolve_commit_sha(
        RepoRef(org="org", repo="repo", ref="main", subpath=None)
    )
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer ghp_secret"
    # Always-on hygiene headers:
    assert "accept" in {h.lower() for h in sent.headers}
    assert sent.headers["accept"] == "application/vnd.github+json"
    assert sent.headers["user-agent"].startswith("Perspicacite")


async def test_rate_limit_raises_with_reset_at(
    tmp_path: Path, respx_mock: Any
) -> None:
    future = int(time.time()) + 3600
    respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/commits/main"
    ).mock(
        return_value=httpx.Response(
            403,
            headers={"X-RateLimit-Reset": str(future)},
            json={"message": "API rate limit exceeded"},
        )
    )

    fetcher = GitHubFetcher(cache_dir=tmp_path / "cache")
    with pytest.raises(RateLimitedError) as exc:
        await fetcher.resolve_commit_sha(
            RepoRef(org="org", repo="repo", ref="main", subpath=None)
        )
    assert exc.value.reset_at == future


# ---------------------------------------------------------------------------
# fetch_tarball — extraction + on-disk SHA cache
# ---------------------------------------------------------------------------


async def test_fetch_tarball_extracts_and_caches(
    tmp_path: Path, respx_mock: Any
) -> None:
    sha = "abc1234"
    tar_bytes = _build_tarball(
        {"README.md": b"hello\n", "src/x.py": b"print(1)\n"},
        top_dir=f"org-repo-{sha}",
    )
    route = respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/tarball/abc1234"
    ).mock(return_value=httpx.Response(200, content=tar_bytes))

    fetcher = GitHubFetcher(cache_dir=tmp_path / "cache")
    ref = RepoRef(org="org", repo="repo", ref="main", subpath=None)

    path1 = await fetcher.fetch_tarball(ref, sha=sha)
    assert path1.is_dir()
    assert (path1 / "README.md").read_bytes() == b"hello\n"
    assert (path1 / "src" / "x.py").read_bytes() == b"print(1)\n"
    # Cache hit: second call must not hit the network.
    before = route.call_count
    path2 = await fetcher.fetch_tarball(ref, sha=sha)
    assert path2 == path1
    assert route.call_count == before, "second call should be a cache hit"


async def test_fetch_tarball_empty_cache_dir_is_refetched(
    tmp_path: Path, respx_mock: Any
) -> None:
    """An empty ``cache_dir/<sha>/`` (left by a previously-aborted
    extraction) is not a valid cache entry; the fetcher must clear it
    and re-download."""

    sha = "cafef00d"
    cache_root = tmp_path / "cache"
    (cache_root / sha).mkdir(parents=True)  # empty: stale partial extract

    tar_bytes = _build_tarball(
        {"README.md": b"ok\n"}, top_dir=f"org-repo-{sha}"
    )
    respx_mock.get(
        url__regex=rf"https://api\.github\.com/repos/org/repo/tarball/{sha}"
    ).mock(return_value=httpx.Response(200, content=tar_bytes))

    fetcher = GitHubFetcher(cache_dir=cache_root)
    path = await fetcher.fetch_tarball(
        RepoRef(org="org", repo="repo", ref="main", subpath=None), sha=sha
    )
    assert (path / "README.md").read_bytes() == b"ok\n"


async def test_fetch_tarball_partial_extract_without_sentinel_is_refetched(
    tmp_path: Path, respx_mock: Any
) -> None:
    """Cache entries without a ``.complete`` sentinel must be treated as
    corrupt and re-downloaded.

    Reproduces the mid-extract-cancellation window: an ingest cancelled
    after writing 5 of 50 files leaves a non-empty directory that the
    old ``any(path.iterdir())`` check happily accepted as valid.
    """

    sha = "partialsha"
    cache_root = tmp_path / "cache"
    partial = cache_root / sha
    partial.mkdir(parents=True)
    # Non-empty but no sentinel: simulates an aborted extract.
    (partial / "partial-file.py").write_text("# only 1 of 50 files made it\n")
    # NOTE: no .complete sentinel — must be rejected as invalid.

    tar_bytes = _build_tarball(
        {"README.md": b"full\n", "src/x.py": b"complete\n"},
        top_dir=f"org-repo-{sha}",
    )
    respx_mock.get(
        url__regex=rf"https://api\.github\.com/repos/org/repo/tarball/{sha}"
    ).mock(return_value=httpx.Response(200, content=tar_bytes))

    fetcher = GitHubFetcher(cache_dir=cache_root)
    path = await fetcher.fetch_tarball(
        RepoRef(org="org", repo="repo", ref="main", subpath=None), sha=sha
    )
    # New full contents are present.
    assert (path / "README.md").read_bytes() == b"full\n"
    assert (path / "src" / "x.py").read_bytes() == b"complete\n"
    # The stale partial file from before must be gone.
    assert not (path / "partial-file.py").exists()
    # And the sentinel must now be present.
    assert (path / ".complete").is_file()


async def test_extract_tarball_rejects_path_traversal(tmp_path: Path) -> None:
    """``_extract_tarball`` must refuse members whose names try to
    escape the target directory.

    Constructs a tarball with three suspicious member names:

    * ``../escape/file.txt`` — classic parent-traversal
    * ``/etc/passwd`` — absolute path
    * ``inner/../../escape2/file.txt`` — embedded traversal

    Asserts none of them land outside ``target``.
    """

    sha = "evil-sha"
    target = tmp_path / "cache" / sha
    target.mkdir(parents=True)
    # Sandbox guard: a sibling directory we'll watch for escape leaks.
    sibling = tmp_path / "escape"
    sibling.mkdir()
    sibling_etc = tmp_path / "etc"

    top = f"org-repo-{sha}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Top-level wrapper directory (mirrors GitHub tarball shape).
        info = tarfile.TarInfo(name=top)
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        tf.addfile(info)

        # 1. parent-traversal
        bad1 = tarfile.TarInfo(name=f"{top}/../escape/file.txt")
        bad1.size = len(b"PWN1")
        tf.addfile(bad1, io.BytesIO(b"PWN1"))

        # 2. absolute path
        bad2 = tarfile.TarInfo(name="/etc/passwd")
        bad2.size = len(b"PWN2")
        tf.addfile(bad2, io.BytesIO(b"PWN2"))

        # 3. embedded traversal mid-path
        bad3 = tarfile.TarInfo(name=f"{top}/inner/../../escape/file2.txt")
        bad3.size = len(b"PWN3")
        tf.addfile(bad3, io.BytesIO(b"PWN3"))

        # A benign file so the extract still produces a non-empty dir.
        good = tarfile.TarInfo(name=f"{top}/README.md")
        good.size = len(b"ok\n")
        tf.addfile(good, io.BytesIO(b"ok\n"))

    GitHubFetcher._extract_tarball(buf.getvalue(), target)

    # None of the suspicious payloads must have escaped the target dir.
    assert list(sibling.iterdir()) == [], (
        f"path-traversal leak: {list(sibling.iterdir())!r}"
    )
    assert not sibling_etc.exists()
    # The benign file must still be there.
    assert (target / "README.md").read_bytes() == b"ok\n"
    # And none of the suspect names landed under target either.
    assert not (target / ".." / "escape" / "file.txt").exists()
    assert not (target / "escape" / "file.txt").exists()
    assert not (target / "escape" / "file2.txt").exists()


# ---------------------------------------------------------------------------
# fetch_clone — git fallback
# ---------------------------------------------------------------------------


async def test_fetch_clone_invokes_git(tmp_path: Path) -> None:
    sha = "sha123"
    cache_root = tmp_path / "cache"
    fetcher = GitHubFetcher(cache_dir=cache_root)

    captured: dict[str, Any] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*args: str, **kwargs: Any) -> _FakeProc:
        captured.setdefault("calls", []).append(args)
        # Materialise the target directory so callers see it exist.
        # Last positional arg of ``git clone`` is the target path.
        if "clone" in args:
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
            (Path(args[-1]) / "README.md").write_text("hello")
            # Match the new sentinel-based cache validity check so the
            # clone "succeeds" from the fetcher's perspective.
            (Path(args[-1]) / ".complete").write_text("")
        return _FakeProc()

    with patch(
        "perspicacite.pipeline.github.fetcher.asyncio.create_subprocess_exec",
        side_effect=_fake_exec,
    ):
        path = await fetcher.fetch_clone(
            RepoRef(org="org", repo="repo", ref="main", subpath=None), sha=sha
        )

    assert path == cache_root / sha
    assert (path / "README.md").read_text() == "hello"
    clone_argv = captured["calls"][0]
    assert clone_argv[0] == "git"
    assert "clone" in clone_argv
    assert "--depth=1" in clone_argv
    assert "https://github.com/org/repo.git" in clone_argv
    assert str(cache_root / sha) in clone_argv


async def test_fetch_clone_token_passed_via_extra_header_not_url(
    tmp_path: Path,
) -> None:
    """Tokens MUST be passed via ``git -c http.extraHeader=Authorization: Bearer ...``
    rather than baked into the clone URL.

    Baking the token into the URL leaks it on failure: git echoes the
    URL in stderr (``fatal: unable to access 'https://TOKEN@github.com/...'``),
    which the fetcher then captures into a ``FetcherError`` message.
    """

    sha = "sha-tok"
    cache_root = tmp_path / "cache"
    fetcher = GitHubFetcher(token="ghp_supersecret", cache_dir=cache_root)

    captured: dict[str, Any] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*args: str, **kwargs: Any) -> _FakeProc:
        captured.setdefault("calls", []).append(args)
        if "clone" in args:
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
            (Path(args[-1]) / "README.md").write_text("hi")
            (Path(args[-1]) / ".complete").write_text("")
        return _FakeProc()

    with patch(
        "perspicacite.pipeline.github.fetcher.asyncio.create_subprocess_exec",
        side_effect=_fake_exec,
    ):
        await fetcher.fetch_clone(
            RepoRef(org="org", repo="repo", ref="main", subpath=None), sha=sha
        )

    clone_argv = captured["calls"][0]
    # Token must be in the extraHeader (-c) pair, not in any URL.
    assert "-c" in clone_argv
    extra_header_idx = clone_argv.index("-c")
    assert clone_argv[extra_header_idx + 1] == (
        "http.extraHeader=Authorization: Bearer ghp_supersecret"
    )
    # No clone-argv element may contain the token-in-URL pattern.
    for arg in clone_argv:
        assert "ghp_supersecret@github.com" not in arg, (
            f"token leaked into clone argv element: {arg!r}"
        )
    # The clone URL itself must be plain (no embedded credentials).
    assert "https://github.com/org/repo.git" in clone_argv


async def test_fetch_clone_no_token_omits_extra_header(tmp_path: Path) -> None:
    """When ``token`` is ``None``, the fetcher must NOT add any
    ``-c http.extraHeader=...`` flag (an empty Bearer value would be
    a no-op at best and confusing in process listings at worst)."""

    sha = "sha-no-tok"
    cache_root = tmp_path / "cache"
    fetcher = GitHubFetcher(cache_dir=cache_root)

    captured: dict[str, Any] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*args: str, **kwargs: Any) -> _FakeProc:
        captured.setdefault("calls", []).append(args)
        if "clone" in args:
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
            (Path(args[-1]) / "README.md").write_text("hi")
            (Path(args[-1]) / ".complete").write_text("")
        return _FakeProc()

    with patch(
        "perspicacite.pipeline.github.fetcher.asyncio.create_subprocess_exec",
        side_effect=_fake_exec,
    ):
        await fetcher.fetch_clone(
            RepoRef(org="org", repo="repo", ref="main", subpath=None), sha=sha
        )

    clone_argv = captured["calls"][0]
    # No -c flag at all means no extraHeader arg.
    for i, arg in enumerate(clone_argv):
        if arg == "-c":
            # If -c is present, it must not carry an extraHeader for Bearer.
            assert "extraHeader" not in clone_argv[i + 1], (
                f"unexpected extraHeader injected when token is None: "
                f"{clone_argv[i + 1]!r}"
            )


async def test_fetch_clone_failure_does_not_leak_token_in_error(
    tmp_path: Path,
) -> None:
    """A failing ``git clone`` whose stderr happens to contain the token
    substring must NOT propagate that substring into the
    ``FetcherError`` message.

    Defense-in-depth: even though the new argv shape no longer puts the
    token in the URL, an upstream git layer or git config could still
    echo a token from elsewhere. The fetcher MUST scrub it before
    raising.
    """

    sha = "sha-fail"
    cache_root = tmp_path / "cache"
    fetcher = GitHubFetcher(token="ghp_supersecret", cache_dir=cache_root)

    class _FailingProc:
        returncode = 128

        async def communicate(self) -> tuple[bytes, bytes]:
            # Mimic git's behaviour: stderr contains the token (e.g.
            # from a previous-version URL or config).
            return (
                b"",
                (
                    b"fatal: unable to access "
                    b"'https://ghp_supersecret@github.com/org/repo.git/': "
                    b"The requested URL returned error: 403"
                ),
            )

    async def _fake_exec(*args: str, **kwargs: Any) -> _FailingProc:
        return _FailingProc()

    with patch(
        "perspicacite.pipeline.github.fetcher.asyncio.create_subprocess_exec",
        side_effect=_fake_exec,
    ), pytest.raises(FetcherError) as excinfo:
        await fetcher.fetch_clone(
            RepoRef(org="org", repo="repo", ref="main", subpath=None),
            sha=sha,
        )

    msg = str(excinfo.value)
    assert "ghp_supersecret" not in msg, (
        f"token leaked into FetcherError message: {msg!r}"
    )


# ---------------------------------------------------------------------------
# fetch() — orchestrator: tarball → clone fallback on rate-limit
# ---------------------------------------------------------------------------


async def test_fetch_falls_back_to_clone_on_rate_limit(
    tmp_path: Path, respx_mock: Any
) -> None:
    sha = "abcdef0"
    respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/commits/main"
    ).mock(return_value=httpx.Response(200, json={"sha": sha}))

    fetcher = GitHubFetcher(cache_dir=tmp_path / "cache")

    async def _raise_rl(*args: Any, **kwargs: Any) -> Path:
        raise RateLimitedError("tarball over quota", reset_at=int(time.time()) + 60)

    async def _fake_clone(*args: Any, **kwargs: Any) -> Path:
        p = tmp_path / "cache" / sha
        p.mkdir(parents=True, exist_ok=True)
        (p / "README.md").write_text("cloned")
        return p

    with (
        patch.object(fetcher, "fetch_tarball", side_effect=_raise_rl),
        patch.object(fetcher, "fetch_clone", side_effect=_fake_clone) as mock_clone,
    ):
        path, returned_sha = await fetcher.fetch(
            RepoRef(org="org", repo="repo", ref="main", subpath=None)
        )

    assert returned_sha == sha
    assert path == tmp_path / "cache" / sha
    assert (path / "README.md").read_text() == "cloned"
    mock_clone.assert_called_once()


async def test_fetch_returns_cache_hit_without_clone(
    tmp_path: Path, respx_mock: Any
) -> None:
    sha = "cached1"
    respx_mock.get(
        url__regex=r"https://api\.github\.com/repos/org/repo/commits/main"
    ).mock(return_value=httpx.Response(200, json={"sha": sha}))

    cache_dir = tmp_path / "cache"
    (cache_dir / sha).mkdir(parents=True)
    (cache_dir / sha / "README.md").write_text("cached")
    # Sentinel marks the cache entry as a fully-completed extract;
    # without it the fetcher now correctly refuses to trust the dir.
    (cache_dir / sha / ".complete").write_text("")

    fetcher = GitHubFetcher(cache_dir=cache_dir)
    clone_mock = AsyncMock()
    tarball_mock = AsyncMock()
    with (
        patch.object(fetcher, "fetch_clone", clone_mock),
        patch.object(fetcher, "fetch_tarball", tarball_mock),
    ):
        path, returned_sha = await fetcher.fetch(
            RepoRef(org="org", repo="repo", ref="main", subpath=None)
        )

    assert returned_sha == sha
    assert path == cache_dir / sha
    assert (path / "README.md").read_text() == "cached"
    clone_mock.assert_not_called()
    tarball_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------


def test_package_reexports_public_api() -> None:
    """``perspicacite.pipeline.github`` should re-export the four
    public symbols callers want, so they can write
    ``from perspicacite.pipeline.github import GitHubFetcher`` instead
    of digging into ``.fetcher``.
    """

    from perspicacite.pipeline import github as pkg
    from perspicacite.pipeline.github.fetcher import (
        FetcherError as F_Error,
    )
    from perspicacite.pipeline.github.fetcher import (
        GitHubFetcher as F_Fetcher,
    )
    from perspicacite.pipeline.github.fetcher import (
        RateLimitedError as F_RL,
    )
    from perspicacite.pipeline.github.fetcher import (
        RepoRef as F_Ref,
    )
    from perspicacite.pipeline.github.fetcher import (
        parse_repo_url as f_parse,
    )

    assert pkg.GitHubFetcher is F_Fetcher
    assert pkg.RepoRef is F_Ref
    assert pkg.parse_repo_url is f_parse
    assert pkg.FetcherError is F_Error
    assert pkg.RateLimitedError is F_RL
