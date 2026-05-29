"""Tests for the opt-in ``indicia`` payload on the generate_report MCP tool.

Verifies that:
- ``extract_claims=True`` causes the returned JSON to include an ``indicia`` key
  that is a list (the typed claim set produced by the pipeline claims extractor).
- Omitting ``extract_claims`` (or passing ``False``) leaves ``indicia`` absent from
  the payload so existing callers see no change.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import perspicacite.mcp.server as mcp_server
from perspicacite.mcp.server import MCPState, generate_report
from perspicacite.models.rag import StreamEvent
from perspicacite.rag.engine import RAGEngine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers — mirrors test_mcp_generate_report_knobs.py exactly
# ---------------------------------------------------------------------------


def _real_config() -> MagicMock:
    cfg = MagicMock()
    cfg.llm.default_provider = "deepseek"
    cfg.llm.default_model = "deepseek-chat"
    return cfg


def _kb_meta(name: str = "kb", model: str = "text-embedding-3-small") -> SimpleNamespace:
    return SimpleNamespace(name=name, embedding_model=model, collection_name=f"coll_{name}")


def _make_state() -> MCPState:
    state = MCPState()
    state.initialized = True
    state.config = _real_config()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None
    meta = _kb_meta("kb")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(return_value=meta)
    state.session_store = ss_mock
    return state


def _install_source_emitting_engine():
    """Swap RAGEngine for one that yields a content delta + a source event.

    Returns a cleanup callable.
    """

    class _SourceEmittingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(event="content", data='{"delta": "Some report text."}')
            yield StreamEvent(
                event="source",
                data=json.dumps({
                    "title": "Test Paper",
                    "authors": ["Author A"],
                    "year": 2024,
                    "doi": "10.1234/test",
                    "relevance_score": 0.9,
                    "section": "Abstract",
                    "kb_name": "kb",
                }),
            )
            yield StreamEvent(event="done", data="{}")

    import perspicacite.rag.engine as _engine_mod

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _SourceEmittingRAGEngine  # type: ignore[assignment]

    def _cleanup() -> None:
        _engine_mod.RAGEngine = original_cls

    return _cleanup


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_report_attaches_indicia_when_requested() -> None:
    """extract_claims=True must produce an ``indicia`` list in the payload."""
    pytest.importorskip("indicium", reason="indicia extra not installed")
    state = _make_state()

    # The claim-extraction call goes through state.llm_client.complete.
    # Return a valid {"claims": [...]} JSON string with one 5-slot claim.
    valid_claim = {
        "context": "microbial ecology",
        "subject": "Bacteroides",
        "qualifier": "increases",
        "relation": "abundance in",
        "object": "gut microbiome",
        "claim_type": "explicit",
        "evidence_type": "data",
        "source_type": "text",
        "quote": "Bacteroides increased.",
        "source_doi": "10.1234/test",
    }
    state.llm_client.complete = AsyncMock(
        return_value=json.dumps({"claims": [valid_claim]})
    )

    cleanup = _install_source_emitting_engine()
    try:
        with patch.object(mcp_server, "mcp_state", state):
            raw = await generate_report(query="q", kb_name="kb", extract_claims=True)
    finally:
        cleanup()

    payload = json.loads(raw)
    assert payload.get("success") is True, f"Expected success payload, got: {raw[:300]}"
    assert "indicia" in payload, f"'indicia' key missing from payload: {list(payload.keys())}"
    assert isinstance(payload["indicia"], list), (
        f"Expected indicia to be a list, got {type(payload['indicia'])}"
    )


@pytest.mark.asyncio
async def test_generate_report_omits_indicia_by_default() -> None:
    """Calling generate_report without extract_claims must NOT produce an ``indicia`` key."""
    state = _make_state()

    cleanup = _install_source_emitting_engine()
    try:
        with patch.object(mcp_server, "mcp_state", state):
            raw = await generate_report(query="q", kb_name="kb")
    finally:
        cleanup()

    payload = json.loads(raw)
    assert payload.get("success") is True, f"Expected success payload, got: {raw[:300]}"
    assert "indicia" not in payload, (
        f"'indicia' should be absent when extract_claims is not set; "
        f"got keys: {list(payload.keys())}"
    )
