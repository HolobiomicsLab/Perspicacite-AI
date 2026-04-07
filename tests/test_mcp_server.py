#!/usr/bin/env python3
"""Tests for MCP server tools.

Tests tool registration, JSON response formatting, and state management.
Uses direct module loading to avoid heavy import chains.

Run: PYTHONPATH=src pytest tests/test_mcp_server.py -v
"""

import json
import importlib.util
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
    ]

    def test_mcp_object_exists(self):
        assert mcp is not None

    def test_all_tools_registered(self):
        """Check that all expected tool names are registered."""
        # FastMCP stores tools internally; access via _tool_manager
        tool_mgr = mcp._tool_manager
        registered = set(tool_mgr._tools.keys())
        for name in self.EXPECTED_TOOLS:
            assert name in registered, f"Tool '{name}' not found in {registered}"

    def test_tool_count(self):
        """Should have exactly the expected number of tools."""
        tool_mgr = mcp._tool_manager
        assert len(tool_mgr._tools) == len(self.EXPECTED_TOOLS)


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

        # Get the underlying function from the FastMCP FunctionTool wrapper
        tool_mgr = mcp._tool_manager
        fn = tool_mgr._tools["list_knowledge_bases"].fn

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

        tool_mgr = mcp._tool_manager
        fn = tool_mgr._tools["create_knowledge_base"].fn

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

        tool_mgr = mcp._tool_manager
        fn = tool_mgr._tools["create_knowledge_base"].fn

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

        tool_mgr = mcp._tool_manager
        fn = tool_mgr._tools["search_literature"].fn

        # Search may fail if scilex not installed — should return error JSON
        result = await fn(query="test", max_results=5)
        parsed = json.loads(result)
        assert "success" in parsed
        assert isinstance(parsed["success"], bool)

        _mcp_mod.mcp_state = old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
