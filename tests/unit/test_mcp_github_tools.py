"""MCP tools ingest_github_repo + ingest_skill_bundle — thin wrappers
around the ``perspicacite.pipeline.github_kb`` orchestrator.

Mirrors the pattern from
:mod:`tests.unit.test_mcp_ingest_asb_run`: monkeypatch the pipeline
function imported into ``server`` and assert call-shape + envelope.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(**overrides):
    """Build an :class:`IngestSummary` with sensible defaults for tests."""
    from perspicacite.pipeline.github_kb import IngestSummary

    base = dict(
        kb_name="kb_smoke",
        bundle_name=None,
        repo_org=None,
        repo_name=None,
        commit_sha=None,
        files_added=0,
        chunks_added=0,
        linked_papers_added=0,
        linked_papers_skipped_non_doi=[],
        mode="repo",
    )
    base.update(overrides)
    return IngestSummary(**base)


def _mock_state():
    state = MagicMock()
    state.initialized = True
    state.config = MagicMock()
    state.session_store = MagicMock()
    state.vector_store = MagicMock()
    state.embedding_provider = MagicMock()
    return state


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_ingest_github_repo_tool_registered():
    """The tool must be exported by the server module + listed in _TOOL_NAMES."""
    from perspicacite.mcp import server

    assert hasattr(server, "ingest_github_repo"), (
        "server.ingest_github_repo not defined"
    )
    assert "ingest_github_repo" in server._TOOL_NAMES, (
        "ingest_github_repo missing from _TOOL_NAMES"
    )


def test_ingest_skill_bundle_tool_registered():
    """The tool must be exported by the server module + listed in _TOOL_NAMES."""
    from perspicacite.mcp import server

    assert hasattr(server, "ingest_skill_bundle"), (
        "server.ingest_skill_bundle not defined"
    )
    assert "ingest_skill_bundle" in server._TOOL_NAMES, (
        "ingest_skill_bundle missing from _TOOL_NAMES"
    )


# ---------------------------------------------------------------------------
# ingest_github_repo: arg shape + envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_github_repo_calls_pipeline_with_correct_args():
    """Tool must forward url/kb_name + state seams to the orchestrator."""
    from perspicacite.mcp import server

    state = _mock_state()
    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary(
            kb_name="kb_smoke",
            repo_org="x",
            repo_name="y",
            commit_sha="abc",
            files_added=2,
            chunks_added=7,
        )

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_github_repo_pipeline", new=fake_pipeline
    ):
        out_json = await server.ingest_github_repo(
            url="https://github.com/x/y",
            kb_name="kb_smoke",
        )

    assert captured["url"] == "https://github.com/x/y"
    assert captured["kb_name"] == "kb_smoke"
    assert captured["config"] is state.config
    assert captured["vector_store"] is state.vector_store
    assert captured["session_store"] is state.session_store
    assert captured["embedding_service"] is state.embedding_provider
    # No include/exclude → no ContentSpec passed
    assert captured.get("content") is None

    out = json.loads(out_json)
    assert out["success"] is True
    assert out["ok"] is True
    assert out["kb_name"] == "kb_smoke"
    assert out["repo_org"] == "x"
    assert out["repo_name"] == "y"
    assert out["commit_sha"] == "abc"
    assert out["files_added"] == 2
    assert out["chunks_added"] == 7
    assert out["mode"] == "repo"


@pytest.mark.asyncio
async def test_ingest_github_repo_with_include_exclude_builds_content_spec():
    """When include/exclude provided, build a ContentSpec; mix-and-match
    with defaults for the unspecified side."""
    from perspicacite.mcp import server
    from perspicacite.pipeline.github.bundle import ContentSpec

    state = _mock_state()
    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary()

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_github_repo_pipeline", new=fake_pipeline
    ):
        await server.ingest_github_repo(
            url="https://github.com/x/y",
            kb_name="kb",
            include=["*.py"],
            exclude=["tests/**"],
        )

    content = captured.get("content")
    assert isinstance(content, ContentSpec)
    assert content.include == ["*.py"]
    assert content.exclude == ["tests/**"]


@pytest.mark.asyncio
async def test_ingest_github_repo_handles_pipeline_exception():
    """Orchestrator raising → tool returns _json_error, not unhandled."""
    from perspicacite.mcp import server

    state = _mock_state()
    with patch.object(server, "mcp_state", state), patch.object(
        server,
        "ingest_github_repo_pipeline",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        out_json = await server.ingest_github_repo(
            url="https://github.com/x/y",
            kb_name="kb",
        )
    out = json.loads(out_json)
    assert out["success"] is False
    assert "boom" in out["error"]


@pytest.mark.asyncio
async def test_ingest_github_repo_requires_state():
    """Tool must short-circuit with a JSON error when MCP is uninitialized."""
    from perspicacite.mcp import server

    state = MagicMock()
    state.initialized = False

    with patch.object(server, "mcp_state", state):
        out_json = await server.ingest_github_repo(
            url="https://github.com/x/y",
            kb_name="kb",
        )
    out = json.loads(out_json)
    assert out["success"] is False
    assert "not initialized" in out["error"].lower()


def test_ingest_github_repo_docstring_mentions_latency():
    """Per Phase B3 — all multi-second tools must surface latency."""
    import inspect

    from perspicacite.mcp import server

    doc = inspect.getdoc(server.ingest_github_repo) or ""
    assert "Latency" in doc


# ---------------------------------------------------------------------------
# ingest_skill_bundle: path vs URL detection + linked-papers toggle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_skill_bundle_local_path_becomes_path(tmp_path):
    """Local existing path → orchestrator gets a Path, not a string."""
    from perspicacite.mcp import server

    state = _mock_state()
    bundle_dir = tmp_path / "my_bundle"
    bundle_dir.mkdir()

    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary(bundle_name="my_bundle", mode="per-skill")

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_skill_bundle_pipeline", new=fake_pipeline
    ):
        await server.ingest_skill_bundle(
            source=str(bundle_dir),
            kb_name="kb",
            ingest_linked_papers=False,
        )

    src = captured["source"]
    assert isinstance(src, Path)
    assert src == bundle_dir


@pytest.mark.asyncio
async def test_ingest_skill_bundle_github_url_passed_as_string():
    """URL source → orchestrator gets the raw string."""
    from perspicacite.mcp import server

    state = _mock_state()
    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary(bundle_name="b", mode="per-skill")

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_skill_bundle_pipeline", new=fake_pipeline
    ):
        await server.ingest_skill_bundle(
            source="https://github.com/x/y",
            kb_name="kb",
            ingest_linked_papers=False,
        )

    src = captured["source"]
    assert isinstance(src, str)
    assert src == "https://github.com/x/y"


@pytest.mark.asyncio
async def test_ingest_skill_bundle_ingest_linked_papers_false_passes_no_state():
    """ingest_linked_papers=False → app_state_for_doi_ingest must be None."""
    from perspicacite.mcp import server

    state = _mock_state()
    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary(bundle_name="b", mode="per-skill")

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_skill_bundle_pipeline", new=fake_pipeline
    ):
        await server.ingest_skill_bundle(
            source="https://github.com/x/y",
            kb_name="kb",
            ingest_linked_papers=False,
        )

    assert captured["ingest_linked_papers"] is False
    assert captured["app_state_for_doi_ingest"] is None


@pytest.mark.asyncio
async def test_ingest_skill_bundle_ingest_linked_papers_true_passes_state():
    """ingest_linked_papers=True (default) → app_state passed through."""
    from perspicacite.mcp import server

    state = _mock_state()
    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary(bundle_name="b", mode="per-skill")

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_skill_bundle_pipeline", new=fake_pipeline
    ):
        await server.ingest_skill_bundle(
            source="https://github.com/x/y",
            kb_name="kb",
        )

    assert captured["ingest_linked_papers"] is True
    assert captured["app_state_for_doi_ingest"] is state


@pytest.mark.asyncio
async def test_ingest_skill_bundle_default_kb_name_is_none():
    """When kb_name is omitted, the orchestrator must receive None
    (so the bundle.yml template derives the name)."""
    from perspicacite.mcp import server

    state = _mock_state()
    captured: dict = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_summary(kb_name="kb_from_bundle", bundle_name="b", mode="per-skill")

    with patch.object(server, "mcp_state", state), patch.object(
        server, "ingest_skill_bundle_pipeline", new=fake_pipeline
    ):
        await server.ingest_skill_bundle(
            source="https://github.com/x/y",
            ingest_linked_papers=False,
        )

    assert captured["kb_name"] is None


@pytest.mark.asyncio
async def test_ingest_skill_bundle_handles_pipeline_exception():
    """Orchestrator raising → tool returns _json_error."""
    from perspicacite.mcp import server

    state = _mock_state()
    with patch.object(server, "mcp_state", state), patch.object(
        server,
        "ingest_skill_bundle_pipeline",
        new=AsyncMock(side_effect=RuntimeError("kaboom")),
    ):
        out_json = await server.ingest_skill_bundle(
            source="/nonexistent/path",
            ingest_linked_papers=False,
        )
    out = json.loads(out_json)
    assert out["success"] is False
    assert "kaboom" in out["error"]


@pytest.mark.asyncio
async def test_ingest_skill_bundle_requires_state():
    """Uninitialized state → JSON error."""
    from perspicacite.mcp import server

    state = MagicMock()
    state.initialized = False

    with patch.object(server, "mcp_state", state):
        out_json = await server.ingest_skill_bundle(
            source="/tmp/whatever",
        )
    out = json.loads(out_json)
    assert out["success"] is False
    assert "not initialized" in out["error"].lower()


def test_ingest_skill_bundle_docstring_mentions_latency():
    """Per Phase B3 — multi-second tools must surface latency."""
    import inspect

    from perspicacite.mcp import server

    doc = inspect.getdoc(server.ingest_skill_bundle) or ""
    assert "Latency" in doc


@pytest.mark.asyncio
async def test_ingest_skill_bundle_envelope_contains_linked_papers_keys():
    """Summary fields linked_papers_added + linked_papers_skipped_non_doi
    must surface in the JSON envelope so operators can inspect them."""
    from perspicacite.mcp import server

    state = _mock_state()
    summary = _make_summary(
        bundle_name="b",
        mode="per-skill",
        files_added=3,
        chunks_added=12,
        linked_papers_added=2,
        linked_papers_skipped_non_doi=[("arxiv", "1234.5678")],
    )

    with patch.object(server, "mcp_state", state), patch.object(
        server,
        "ingest_skill_bundle_pipeline",
        new=AsyncMock(return_value=summary),
    ):
        out_json = await server.ingest_skill_bundle(
            source="https://github.com/x/y",
            ingest_linked_papers=False,
        )

    out = json.loads(out_json)
    assert out["bundle_name"] == "b"
    assert out["files_added"] == 3
    assert out["chunks_added"] == 12
    assert out["linked_papers_added"] == 2
    assert out["linked_papers_skipped_non_doi"] == [["arxiv", "1234.5678"]]
