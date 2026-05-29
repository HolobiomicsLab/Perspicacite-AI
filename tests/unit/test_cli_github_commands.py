"""CLI commands for GitHub-repo / skill-bundle ingest (Task 6).

Mirrors the patch-the-orchestrator pattern in test_cli_ingest_asb_run.py.
The three commands under test are thin Click wrappers around the
``perspicacite.pipeline.github_kb`` orchestrators.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_summary(
    *,
    kb_name: str = "fake_kb",
    bundle_name: str | None = None,
    repo_org: str | None = None,
    repo_name: str | None = None,
    commit_sha: str | None = None,
    files_added: int = 3,
    chunks_added: int = 7,
    linked_papers_added: int = 0,
    linked_papers_skipped_non_doi: list[tuple[str, str]] | None = None,
    mode: str = "repo",
):
    """Build an :class:`IngestSummary` with sensible defaults so each test
    only specifies the fields it cares about."""
    from perspicacite.pipeline.github_kb import IngestSummary

    return IngestSummary(
        kb_name=kb_name,
        bundle_name=bundle_name,
        repo_org=repo_org,
        repo_name=repo_name,
        commit_sha=commit_sha,
        files_added=files_added,
        chunks_added=chunks_added,
        linked_papers_added=linked_papers_added,
        linked_papers_skipped_non_doi=linked_papers_skipped_non_doi or [],
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Help tests
# ---------------------------------------------------------------------------


def test_ingest_github_repo_help() -> None:
    from perspicacite.cli import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["ingest-github-repo", "--help"])
    assert r.exit_code == 0, r.output
    out = r.output
    assert "KB" in out or "kb" in out
    assert "include" in out
    assert "exclude" in out


def test_ingest_skill_bundle_help() -> None:
    from perspicacite.cli import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["ingest-skill-bundle", "--help"])
    assert r.exit_code == 0, r.output
    out = r.output
    assert "KB" in out or "kb" in out
    assert "linked-papers" in out


def test_ingest_skill_bundles_help() -> None:
    from perspicacite.cli import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["ingest-skill-bundles", "--help"])
    assert r.exit_code == 0, r.output
    out = r.output
    assert "into" in out  # composite KB flag


# ---------------------------------------------------------------------------
# ingest-github-repo: invocation
# ---------------------------------------------------------------------------


def test_ingest_github_repo_invokes_orchestrator() -> None:
    """Click args + include/exclude flags should be forwarded into
    `ingest_github_repo` as a ContentSpec."""
    from perspicacite.cli import cli

    captured: dict = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return _make_summary(
            kb_name="my_kb",
            repo_org="acme",
            repo_name="repo",
            commit_sha="abc123",
            mode="repo",
        )

    with patch(
        "perspicacite.cli._ingest_github_repo",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-github-repo",
                "https://github.com/acme/repo",
                "--kb-name", "my_kb",
                "--include", "*.py",
                "--include", "*.md",
                "--exclude", "tests/**",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("url") == "https://github.com/acme/repo"
    assert captured.get("kb_name") == "my_kb"

    content = captured.get("content")
    assert content is not None
    assert list(content.include) == ["*.py", "*.md"]
    assert list(content.exclude) == ["tests/**"]

    # Human-readable summary echoes the KB name + counts
    assert "my_kb" in r.output


def test_ingest_github_repo_default_content_is_none() -> None:
    """When neither --include nor --exclude is given, content=None so the
    orchestrator picks the bundle defaults."""
    from perspicacite.cli import cli

    captured: dict = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return _make_summary(kb_name="k", repo_org="o", repo_name="r", mode="repo")

    with patch(
        "perspicacite.cli._ingest_github_repo",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-github-repo",
                "https://github.com/acme/repo",
                "--kb-name", "k",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("content") is None


# ---------------------------------------------------------------------------
# ingest-skill-bundle: invocation
# ---------------------------------------------------------------------------


def test_ingest_skill_bundle_invokes_orchestrator_with_local_path(
    tmp_path: Path,
) -> None:
    """If `source` is an existing local path, we pass a Path; if not, a
    URL string. Also: --no-linked-papers => ingest_linked_papers=False
    AND app_state_for_doi_ingest=None."""
    from perspicacite.cli import cli

    bundle_dir = tmp_path / "my_bundle"
    bundle_dir.mkdir()

    captured: dict = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return _make_summary(
            kb_name="my_bundle_kb",
            bundle_name="my_bundle",
            mode="per-skill",
            chunks_added=42,
        )

    with patch(
        "perspicacite.cli._ingest_skill_bundle",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-skill-bundle",
                str(bundle_dir),
                "--kb-name", "my_bundle_kb",
                "--no-linked-papers",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    src = captured.get("source")
    assert isinstance(src, Path)
    assert src == bundle_dir
    assert captured.get("kb_name") == "my_bundle_kb"
    assert captured.get("ingest_linked_papers") is False
    # When --no-linked-papers, no app_state needed
    assert captured.get("app_state_for_doi_ingest") is None
    assert "42" in r.output
    assert "my_bundle_kb" in r.output


def test_ingest_skill_bundle_passes_url_when_source_is_not_a_path() -> None:
    """URL strings are passed through verbatim; the orchestrator handles
    fetching."""
    from perspicacite.cli import cli

    captured: dict = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return _make_summary(
            kb_name="auto_kb", bundle_name="auto_kb", mode="per-skill"
        )

    with patch(
        "perspicacite.cli._ingest_skill_bundle",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-skill-bundle",
                "https://github.com/acme/repo/tree/main/skills/bundle",
                "--no-linked-papers",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    src = captured.get("source")
    assert isinstance(src, str)
    assert src == "https://github.com/acme/repo/tree/main/skills/bundle"


def test_ingest_skill_bundle_linked_papers_default_passes_app_state(
    tmp_path: Path,
) -> None:
    """Without --no-linked-papers, ingest_linked_papers=True AND
    app_state_for_doi_ingest is the built app state."""
    from perspicacite.cli import cli

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    captured: dict = {}
    fake_app_state = MagicMock()

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return _make_summary(
            kb_name="b", bundle_name="b", mode="per-skill", linked_papers_added=5
        )

    with patch(
        "perspicacite.cli._ingest_skill_bundle",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=fake_app_state),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-skill-bundle",
                str(bundle_dir),
                "--kb-name", "b",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("ingest_linked_papers") is True
    assert captured.get("app_state_for_doi_ingest") is fake_app_state
    assert "5" in r.output  # linked papers count surfaced


# ---------------------------------------------------------------------------
# ingest-skill-bundles (batch): per-skill vs composite
# ---------------------------------------------------------------------------


def test_ingest_skill_bundles_per_skill_when_no_into_flag(
    tmp_path: Path,
) -> None:
    from perspicacite.cli import cli

    root = tmp_path / "bundles"
    root.mkdir()

    captured: dict = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return [
            _make_summary(kb_name="b1_kb", bundle_name="b1", mode="per-skill"),
            _make_summary(kb_name="b2_kb", bundle_name="b2", mode="per-skill"),
        ]

    with patch(
        "perspicacite.cli._ingest_skill_bundles_batch",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-skill-bundles",
                str(root),
                "--no-linked-papers",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("composite_kb") is None
    assert captured.get("ingest_linked_papers") is False
    # Both KB names in output
    assert "b1_kb" in r.output
    assert "b2_kb" in r.output


def test_ingest_skill_bundles_composite_with_into_flag(
    tmp_path: Path,
) -> None:
    from perspicacite.cli import cli

    root = tmp_path / "bundles"
    root.mkdir()

    captured: dict = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return [
            _make_summary(
                kb_name="big_kb", bundle_name="b1", mode="composite"
            ),
            _make_summary(
                kb_name="big_kb", bundle_name="b2", mode="composite"
            ),
        ]

    with patch(
        "perspicacite.cli._ingest_skill_bundles_batch",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            [
                "ingest-skill-bundles",
                str(root),
                "--into", "big_kb",
                "--no-linked-papers",
            ],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("composite_kb") == "big_kb"
    assert "big_kb" in r.output


def test_ingest_skill_bundles_missing_dir_errors(tmp_path: Path) -> None:
    """Click should reject a non-existent SOURCE_DIR (Path(exists=True))."""
    from perspicacite.cli import cli

    runner = CliRunner()
    r = runner.invoke(
        cli,
        ["ingest-skill-bundles", str(tmp_path / "nope")],
    )
    assert r.exit_code != 0
