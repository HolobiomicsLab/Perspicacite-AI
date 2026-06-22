#!/usr/bin/env python3
"""Tests for MCP server tools.

Tests tool registration, JSON response formatting, and state management.
Uses direct module loading to avoid heavy import chains.

Run: PYTHONPATH=src pytest tests/test_mcp_server.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Direct module loading to avoid chromadb etc. import chains
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent.parent / "src" / "perspicacite"


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, str(_BASE / rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Check if fastmcp is available
_fastmcp_spec = importlib.util.find_spec("fastmcp")
if _fastmcp_spec is None:
    pytest.skip("fastmcp not installed", allow_module_level=True)

# Load the MCP server module
_mcp_mod = _load_module("perspicacite.mcp.server", "mcp/server.py")

mcp = _mcp_mod.mcp
mcp_state = _mcp_mod.mcp_state
_json_ok = _mcp_mod._json_ok
_json_error = _mcp_mod._json_error


# ---------------------------------------------------------------------------
# DRY helpers for fastmcp 3.x (removed _tool_manager)
# ---------------------------------------------------------------------------


def _tool_fn(name: str):
    """Return the underlying callable for a registered tool by module attribute."""
    return getattr(_mcp_mod, name)


async def _registered_tool_names() -> list[str]:
    """Return the list of tool names registered with the FastMCP instance."""
    tools = await mcp._list_tools()
    return [t.name for t in tools]


# ---------------------------------------------------------------------------
# Helper: build a mock MCPState with all required attributes
# ---------------------------------------------------------------------------


def _make_mock_state():
    """Create a fully mocked MCPState."""
    state = MagicMock()
    state.initialized = True
    state.config = MagicMock()
    state.config.knowledge_base.chunk_size = 1000
    state.config.knowledge_base.chunk_overlap = 200
    state.config.knowledge_base.chunking_method = "token"
    state.config.knowledge_base.embedding_model = "text-embedding-3-small"
    state.config.pdf_download = MagicMock()
    state.config.pdf_download.unpaywall_email = None
    state.config.pdf_download.alternative_endpoint = None
    state.config.pdf_download.wiley_tdm_token = None
    state.config.pdf_download.aaas_api_key = None
    state.config.pdf_download.rsc_api_key = None
    state.config.pdf_download.springer_api_key = None
    state.embedding_provider = MagicMock()
    state.embedding_provider.dimension = 1536
    state.embedding_provider.model_name = "text-embedding-3-small"
    state.session_store = AsyncMock()
    state.vector_store = AsyncMock()
    state.llm_client = AsyncMock()
    state.pdf_parser = AsyncMock()
    return state


# ---------------------------------------------------------------------------
# Tests: JSON helpers
# ---------------------------------------------------------------------------


class TestJsonHelpers:
    def test_json_ok(self):
        result = _json_ok({"key": "value", "count": 5})
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["key"] == "value"
        assert parsed["count"] == 5

    def test_json_error(self):
        result = _json_error("Something went wrong")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert parsed["error"] == "Something went wrong"

    def test_json_error_with_extra(self):
        result = _json_error("fail", code=404)
        parsed = json.loads(result)
        assert parsed["code"] == 404


# ---------------------------------------------------------------------------
# Tests: Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify all expected tools are registered with FastMCP."""

    EXPECTED_TOOLS = [
        "search_literature",
        "get_paper_content",
        "get_paper_references",
        "list_knowledge_bases",
        "search_knowledge_base",
        "create_knowledge_base",
        "add_papers_to_kb",
        "generate_report",
        "screen_papers",
        "add_dois_to_kb",
        "push_to_zotero",
    ]

    def test_mcp_object_exists(self):
        assert mcp is not None

    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        """Check that all expected tool names are registered."""
        registered = set(await _registered_tool_names())
        for name in self.EXPECTED_TOOLS:
            assert name in registered, f"Tool '{name}' not found in {registered}"

    @pytest.mark.asyncio
    async def test_tool_count(self):
        """Should have at least the expected number of tools (server may have more)."""
        registered = await _registered_tool_names()
        assert len(registered) >= len(self.EXPECTED_TOOLS), (
            f"Expected at least {len(self.EXPECTED_TOOLS)} tools, got {len(registered)}: {registered}"
        )


# ---------------------------------------------------------------------------
# Tests: MCPState
# ---------------------------------------------------------------------------


class TestMCPState:
    def test_initial_state(self):
        """Fresh state should not be initialized."""
        fresh_state = _mcp_mod.MCPState()
        assert fresh_state.initialized is False
        assert fresh_state.session_store is None

    def test_require_state_returns_error_when_not_initialized(self):
        """_require_state should return error string when not initialized."""
        # Save and restore mcp_state
        old = _mcp_mod.mcp_state
        fresh = _mcp_mod.MCPState()
        _mcp_mod.mcp_state = fresh

        result = _mcp_mod._require_state()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["success"] is False

        # Restore
        _mcp_mod.mcp_state = old

    def test_require_state_returns_state_when_initialized(self):
        """_require_state should return the MCPState object when initialized."""
        old = _mcp_mod.mcp_state
        mock_state = _make_mock_state()
        _mcp_mod.mcp_state = mock_state

        result = _mcp_mod._require_state()
        assert result is mock_state

        _mcp_mod.mcp_state = old


# ---------------------------------------------------------------------------
# Tests: Tool responses (with mocked state)
# ---------------------------------------------------------------------------


class TestListKnowledgeBases:
    @pytest.mark.asyncio
    async def test_returns_json_with_kbs(self):
        old = _mcp_mod.mcp_state
        state = _make_mock_state()

        # Mock KB metadata
        mock_kb = MagicMock()
        mock_kb.name = "test_kb"
        mock_kb.description = "A test KB"
        mock_kb.paper_count = 5
        mock_kb.chunk_count = 30
        mock_kb.created_at = "2026-04-07"
        state.session_store.list_kbs = AsyncMock(return_value=[mock_kb])

        _mcp_mod.mcp_state = state

        fn = _tool_fn("list_knowledge_bases")
        result = await fn()
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert len(parsed["knowledge_bases"]) == 1
        assert parsed["knowledge_bases"][0]["name"] == "test_kb"
        assert parsed["knowledge_bases"][0]["paper_count"] == 5

        _mcp_mod.mcp_state = old


class TestCreateKnowledgeBase:
    @pytest.mark.asyncio
    async def test_creates_new_kb(self):
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        state.session_store.get_kb_metadata = AsyncMock(return_value=None)
        state.session_store.save_kb_metadata = AsyncMock()
        state.vector_store.create_collection = AsyncMock()

        _mcp_mod.mcp_state = state

        fn = _tool_fn("create_knowledge_base")
        result = await fn(name="new_kb", description="Test")
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert parsed["name"] == "new_kb"
        assert parsed["paper_count"] == 0

        _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_rejects_duplicate(self):
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        state.session_store.get_kb_metadata = AsyncMock(return_value=MagicMock())

        _mcp_mod.mcp_state = state

        fn = _tool_fn("create_knowledge_base")
        result = await fn(name="existing_kb")
        parsed = json.loads(result)

        assert parsed["success"] is False
        assert "already exists" in parsed["error"]

        _mcp_mod.mcp_state = old


class TestSearchLiterature:
    @pytest.mark.asyncio
    async def test_returns_error_when_search_fails(self):
        old = _mcp_mod.mcp_state
        state = _make_mock_state()

        _mcp_mod.mcp_state = state

        fn = _tool_fn("search_literature")
        # Search may fail if scilex not installed — should return error JSON
        result = await fn(query="test", max_results=5)
        parsed = json.loads(result)
        assert "success" in parsed
        assert isinstance(parsed["success"], bool)

        _mcp_mod.mcp_state = old


@pytest.mark.asyncio
async def test_screen_papers_tool_uninitialized():
    from perspicacite.mcp import server as mcp_server

    saved = mcp_server.mcp_state.initialized
    mcp_server.mcp_state.initialized = False
    try:
        out = await mcp_server.screen_papers(candidates=["10.1/a"], query="x")
        data = json.loads(out)
        assert data["success"] is False
    finally:
        mcp_server.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_screen_papers_tool_bm25_titles(monkeypatch):
    from perspicacite.mcp import server as mcp_server

    saved = mcp_server.mcp_state.initialized
    mcp_server.mcp_state.initialized = True
    try:
        out = await mcp_server.screen_papers(
            candidates=[
                "neural networks for protein structure prediction",
                "renaissance oil painting in florence",
            ],
            query="deep learning protein folding",
            method="bm25",
            threshold=0.0,
            max_results=10,
        )
        data = json.loads(out)
        assert data["success"] is True
        assert "screened" in data and len(data["screened"]) == 2
        # entries have score/kept and either doi or title
        for e in data["screened"]:
            assert "score" in e and "kept" in e and ("title" in e or "doi" in e)
    finally:
        mcp_server.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_screen_papers_tool_doi_candidate(monkeypatch):
    from perspicacite.mcp import server as mcp_server

    saved = mcp_server.mcp_state.initialized
    mcp_server.mcp_state.initialized = True

    # avoid network: stub retrieve_paper_content where the server module references it
    async def _fake_retrieve(doi, **kw):
        from perspicacite.pipeline.download.base import PaperContent

        return PaperContent(
            success=True,
            doi=doi,
            content_type="abstract",
            content_source="x",
            abstract="neural network protein folding deep learning",
            metadata={"title": "Fake Paper"},
        )

    monkeypatch.setattr("perspicacite.pipeline.download.retrieve_paper_content", _fake_retrieve)
    # also patch the name as imported inside server.py if it does `from perspicacite.pipeline.download import retrieve_paper_content`
    monkeypatch.setattr(mcp_server, "retrieve_paper_content", _fake_retrieve, raising=False)
    try:
        out = await mcp_server.screen_papers(
            candidates=["10.1/abc"], query="protein folding", method="bm25", threshold=0.0
        )
        data = json.loads(out)
        assert data["success"] is True and data["screened"][0].get("doi") == "10.1/abc"
        assert data["screened"][0].get("title") == "Fake Paper"
    finally:
        mcp_server.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_add_dois_to_kb_uninitialized():
    import json

    from perspicacite.mcp import server as s

    saved = s.mcp_state.initialized
    s.mcp_state.initialized = False
    try:
        assert json.loads(await s.add_dois_to_kb("k", ["10.1/a"]))["success"] is False
    finally:
        s.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_generate_report_accepts_contradiction_mode(monkeypatch):
    from perspicacite.mcp import server as s
    from perspicacite.models.rag import StreamEvent

    saved = s.mcp_state.initialized
    s.mcp_state.initialized = True

    class _KB:
        collection_name = "c"

    class _SS:
        async def get_kb_metadata(self, name):
            return _KB()

    s.mcp_state.session_store = _SS()

    class _FakeEngine:
        def __init__(self, *a, **k):
            pass

        async def query_stream(self, req, **kwargs):
            yield StreamEvent(event="content", data=json.dumps({"delta": "hello"}))
            yield StreamEvent(event="done", data="{}")

    monkeypatch.setattr("perspicacite.rag.engine.RAGEngine", _FakeEngine)
    try:
        out = json.loads(
            await s.generate_report(
                query="Does X cause Y?", kb_name="default", mode="contradiction"
            )
        )
        assert out["success"] is True
        assert out.get("mode") == "contradiction"
    finally:
        s.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_add_dois_to_kb_oversize():
    import json

    from perspicacite.mcp import server as s

    saved = s.mcp_state.initialized
    s.mcp_state.initialized = True
    try:
        out = json.loads(await s.add_dois_to_kb("k", ["10.1/x"] * 1000))
        assert out["success"] is False and "200" in out["error"]
    finally:
        s.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_generate_report_kb_names_mismatch(monkeypatch):
    from perspicacite.mcp import server as s

    saved = s.mcp_state.initialized
    s.mcp_state.initialized = True

    class _KB:
        def __init__(self, n, m):
            self.name = n
            self.collection_name = f"c_{n}"
            self.embedding_model = m

    class _SS:
        async def get_kb_metadata(self, name):
            return {"a": _KB("a", "m1"), "b": _KB("b", "m2")}.get(name)

    s.mcp_state.session_store = _SS()
    try:
        out = json.loads(await s.generate_report(query="q", kb_names=["a", "b"]))
        assert out["success"] is False and "embedding" in out["error"].lower()
    finally:
        s.mcp_state.initialized = saved


@pytest.mark.asyncio
async def test_get_info_includes_push_to_zotero():
    """get_info() resource must list push_to_zotero, build_kbs_from_zotero, and report 15 tools."""
    from perspicacite.mcp.server import get_info

    raw = await get_info()
    info = json.loads(raw)
    assert "push_to_zotero" in info["tools"], (
        f"push_to_zotero missing from tools list: {info['tools']}"
    )
    assert "build_kbs_from_zotero" in info["tools"], (
        f"build_kbs_from_zotero missing from tools list: {info['tools']}"
    )
    # _TOOL_NAMES in server.py now lists 51 tools (grew from original 49 + ensure_kb + ground_paper).
    # Update this assertion whenever new tools are added to _TOOL_NAMES.
    assert len(info["tools"]) == 51, (
        f"Expected 51 tools in get_info(), got {len(info['tools'])}: {info['tools']}"
    )
    assert info["tool_count"] == 51


# ---------------------------------------------------------------------------
# Tests: _asb_kb_slug helper
# ---------------------------------------------------------------------------


class TestAsbKbSlug:
    """Verify _asb_kb_slug produces the correct KB name from a DOI."""

    def test_standard_doi(self):
        slug = _mcp_mod._asb_kb_slug("10.1021/acs.jnatprod.7b00737")
        assert slug == "asb-paper-10-1021-acs-jnatprod-7b00737"

    def test_doi_with_slashes(self):
        slug = _mcp_mod._asb_kb_slug("10.1038/nature12345")
        assert slug == "asb-paper-10-1038-nature12345"

    def test_doi_with_dots_and_hyphens(self):
        # consecutive non-alnum chars collapse to one hyphen
        slug = _mcp_mod._asb_kb_slug("10.1093/nar/gkad540")
        assert slug == "asb-paper-10-1093-nar-gkad540"

    def test_result_is_lowercase(self):
        slug = _mcp_mod._asb_kb_slug("10.1234/ABC.XYZ")
        assert slug == slug.lower()

    def test_no_leading_trailing_hyphens(self):
        slug = _mcp_mod._asb_kb_slug("10.1234/test")
        assert not slug.startswith("asb-paper--")
        assert not slug.endswith("-")


# ---------------------------------------------------------------------------
# Tests: ensure_kb — idempotent create+ingest
# ---------------------------------------------------------------------------


class TestEnsureKb:
    """ensure_kb(doi) — idempotent: existing KB with chunks returns 'exists'."""

    @pytest.mark.asyncio
    async def test_existing_kb_with_chunks_returns_exists(self, monkeypatch):
        """If KB already has chunks > 0, return status='exists' without calling create/add."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()

        # KB metadata exists and has chunks
        mock_kb_meta = MagicMock()
        mock_kb_meta.chunk_count = 42
        state.session_store.get_kb_metadata = AsyncMock(return_value=mock_kb_meta)

        create_calls = []
        add_calls = []

        async def _fake_create(**kw):
            create_calls.append(kw)
            return _json_ok({"name": kw["name"]})

        async def _fake_add(**kw):
            add_calls.append(kw)
            return _json_ok({"added_chunks": 5})

        monkeypatch.setattr(_mcp_mod, "create_knowledge_base", _fake_create)
        monkeypatch.setattr(_mcp_mod, "add_dois_to_kb", _fake_add)
        _mcp_mod.mcp_state = state

        try:
            fn = _tool_fn("ensure_kb")
            result = await fn(doi="10.1021/acs.jnatprod.7b00737")
            parsed = json.loads(result)

            assert parsed["success"] is True
            assert parsed["status"] == "exists"
            assert parsed["chunks"] == 42
            assert parsed["kb_slug"] == "asb-paper-10-1021-acs-jnatprod-7b00737"
            # Must NOT have called create or add
            assert len(create_calls) == 0
            assert len(add_calls) == 0
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_absent_kb_calls_create_and_add(self, monkeypatch):
        """If KB is absent, ensure_kb calls create then add_dois and returns 'created'."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()

        # First call (existence check) returns None → absent
        state.session_store.get_kb_metadata = AsyncMock(return_value=None)

        doi = "10.1021/acs.jnatprod.7b00737"
        expected_slug = _mcp_mod._asb_kb_slug(doi)

        create_calls = []
        add_calls = []

        async def _fake_create(name, description=""):
            create_calls.append({"name": name, "description": description})
            return _json_ok({"name": name, "chunk_count": 0})

        async def _fake_add(kb_name, dois):
            add_calls.append({"kb_name": kb_name, "dois": dois})
            return _json_ok({
                "kb_name": kb_name,
                "added_chunks": 17,
                "added_with_full_text": 1,
                "added_metadata_only": 0,
            })

        monkeypatch.setattr(_mcp_mod, "create_knowledge_base", _fake_create)
        monkeypatch.setattr(_mcp_mod, "add_dois_to_kb", _fake_add)
        _mcp_mod.mcp_state = state

        try:
            fn = _tool_fn("ensure_kb")
            result = await fn(doi=doi)
            parsed = json.loads(result)

            assert parsed["success"] is True
            assert parsed["status"] == "created"
            assert parsed["kb_slug"] == expected_slug
            assert parsed["chunks"] == 17
            assert parsed["added_with_full_text"] == 1

            # create was called with the correct slug
            assert len(create_calls) == 1
            assert create_calls[0]["name"] == expected_slug

            # add_dois_to_kb was called with the correct slug + doi
            assert len(add_calls) == 1
            assert add_calls[0]["kb_name"] == expected_slug
            assert doi in add_calls[0]["dois"]
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_kb_exists_zero_chunks_reingest(self, monkeypatch):
        """If KB exists but chunk_count == 0, treat as absent and re-ingest."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()

        mock_kb_meta = MagicMock()
        mock_kb_meta.chunk_count = 0
        state.session_store.get_kb_metadata = AsyncMock(return_value=mock_kb_meta)

        add_calls = []

        async def _fake_create(name, description=""):
            return _json_ok({"name": name})

        async def _fake_add(kb_name, dois):
            add_calls.append({"kb_name": kb_name})
            return _json_ok({"kb_name": kb_name, "added_chunks": 8,
                             "added_with_full_text": 1, "added_metadata_only": 0})

        monkeypatch.setattr(_mcp_mod, "create_knowledge_base", _fake_create)
        monkeypatch.setattr(_mcp_mod, "add_dois_to_kb", _fake_add)
        _mcp_mod.mcp_state = state

        try:
            fn = _tool_fn("ensure_kb")
            result = await fn(doi="10.1038/nature12345")
            parsed = json.loads(result)

            assert parsed["success"] is True
            assert parsed["status"] == "created"
            assert len(add_calls) == 1
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_graceful_on_add_failure(self, monkeypatch):
        """If add_dois_to_kb returns an error JSON, ensure_kb propagates _json_error."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        state.session_store.get_kb_metadata = AsyncMock(return_value=None)

        async def _fake_create(name, description=""):
            return _json_ok({"name": name})

        async def _fake_add(kb_name, dois):
            return _json_error("Simulated network failure")

        monkeypatch.setattr(_mcp_mod, "create_knowledge_base", _fake_create)
        monkeypatch.setattr(_mcp_mod, "add_dois_to_kb", _fake_add)
        _mcp_mod.mcp_state = state

        try:
            fn = _tool_fn("ensure_kb")
            result = await fn(doi="10.1038/test")
            parsed = json.loads(result)

            assert parsed["success"] is False
            assert "error" in parsed
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_uninitialized_state_returns_error(self):
        """ensure_kb must return error JSON when state not initialized."""
        old = _mcp_mod.mcp_state
        fresh = _mcp_mod.MCPState()
        _mcp_mod.mcp_state = fresh
        try:
            fn = _tool_fn("ensure_kb")
            result = await fn(doi="10.1/x")
            parsed = json.loads(result)
            assert parsed["success"] is False
        finally:
            _mcp_mod.mcp_state = old


# ---------------------------------------------------------------------------
# Tests: ground_paper — compose ensure_kb + generate_report
# ---------------------------------------------------------------------------


class TestGroundPaper:
    """ground_paper(doi, question, tier) — composes ensure_kb + generate_report."""

    @pytest.mark.asyncio
    async def test_basic_composition(self, monkeypatch):
        """ground_paper calls ensure_kb then generate_report with kb_names=[slug]."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        _mcp_mod.mcp_state = state

        doi = "10.1021/acs.jnatprod.7b00737"
        slug = _mcp_mod._asb_kb_slug(doi)

        ensure_calls = []
        report_calls = []

        async def _fake_ensure(doi, mode="paper"):
            ensure_calls.append({"doi": doi})
            return _json_ok({"kb_slug": slug, "status": "exists", "chunks": 10})

        async def _fake_report(query, kb_names=None, mode="advanced", **kw):
            report_calls.append({"query": query, "kb_names": kb_names, "mode": mode, **kw})
            return _json_ok({
                "report": "Answer about natural products.",
                "sources": [{"doi": doi, "title": "Test Paper"}],
            })

        monkeypatch.setattr(_mcp_mod, "ensure_kb", _fake_ensure)
        monkeypatch.setattr(_mcp_mod, "generate_report", _fake_report)

        try:
            fn = _tool_fn("ground_paper")
            result = await fn(doi=doi, question="What are the main compounds?")
            parsed = json.loads(result)

            assert parsed["success"] is True
            assert parsed["kb_slug"] == slug
            assert "answer" in parsed
            assert "sources" in parsed

            # ensure_kb was called with the doi
            assert len(ensure_calls) == 1
            assert ensure_calls[0]["doi"] == doi

            # generate_report got kb_names=[slug] and mode="basic"
            assert len(report_calls) == 1
            assert report_calls[0]["kb_names"] == [slug]
            assert report_calls[0]["mode"] == "basic"
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_tier_si_passes_context_hint(self, monkeypatch):
        """When tier='si', ground_paper prepends a supplementary context hint to the query."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        _mcp_mod.mcp_state = state

        doi = "10.1093/nar/gkad540"
        slug = _mcp_mod._asb_kb_slug(doi)

        report_calls = []

        async def _fake_ensure(doi, mode="paper"):
            return _json_ok({"kb_slug": slug, "status": "exists", "chunks": 5})

        async def _fake_report(query, kb_names=None, mode="advanced", **kw):
            report_calls.append({"query": query})
            return _json_ok({"report": "SI answer.", "sources": []})

        monkeypatch.setattr(_mcp_mod, "ensure_kb", _fake_ensure)
        monkeypatch.setattr(_mcp_mod, "generate_report", _fake_report)

        try:
            fn = _tool_fn("ground_paper")
            result = await fn(doi=doi, question="What do the SI tables show?", tier="si")
            parsed = json.loads(result)

            assert parsed["success"] is True
            # The SI hint should be prepended to the query
            effective_query = report_calls[0]["query"]
            assert "supplementary" in effective_query.lower()
            assert "What do the SI tables show?" in effective_query
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_tier_paper_passes_no_context(self, monkeypatch):
        """When tier='paper' (default), query is passed verbatim (no prepended hint)."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        _mcp_mod.mcp_state = state

        doi = "10.1038/nature12345"
        slug = _mcp_mod._asb_kb_slug(doi)

        report_calls = []

        async def _fake_ensure(doi, mode="paper"):
            return _json_ok({"kb_slug": slug, "status": "exists", "chunks": 5})

        async def _fake_report(query, kb_names=None, mode="advanced", **kw):
            report_calls.append({"query": query})
            return _json_ok({"report": "Paper answer.", "sources": []})

        monkeypatch.setattr(_mcp_mod, "ensure_kb", _fake_ensure)
        monkeypatch.setattr(_mcp_mod, "generate_report", _fake_report)

        try:
            fn = _tool_fn("ground_paper")
            result = await fn(doi=doi, question="What is the method?")
            parsed = json.loads(result)

            assert parsed["success"] is True
            # With tier="paper", query should be passed verbatim (no SI hint prepended)
            assert report_calls[0]["query"] == "What is the method?"
        finally:
            _mcp_mod.mcp_state = old

    @pytest.mark.asyncio
    async def test_propagates_ensure_kb_error(self, monkeypatch):
        """If ensure_kb errors, ground_paper returns that error."""
        old = _mcp_mod.mcp_state
        state = _make_mock_state()
        _mcp_mod.mcp_state = state

        doi = "10.1/bad"

        async def _fake_ensure(doi, mode="paper"):
            return _json_error("KB creation failed")

        report_calls = []

        async def _fake_report(*a, **kw):
            report_calls.append(kw)
            return _json_ok({"report": "", "sources": []})

        monkeypatch.setattr(_mcp_mod, "ensure_kb", _fake_ensure)
        monkeypatch.setattr(_mcp_mod, "generate_report", _fake_report)

        try:
            fn = _tool_fn("ground_paper")
            result = await fn(doi=doi, question="What is the answer?")
            parsed = json.loads(result)

            assert parsed["success"] is False
            assert len(report_calls) == 0
        finally:
            _mcp_mod.mcp_state = old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
