"""MCP ``generate_report`` + ``search_knowledge_base`` include
``asb_metadata`` in their JSON envelope when retrieval surfaces ASB
chunks.

Three checks:
- White-box: the helper is imported + wired into both tool bodies.
- Integration: stub ``RAGEngine.query_stream`` to emit a synthetic
  source carrying ASB metadata; the ``generate_report`` return must
  contain the expected ``asb_metadata`` block.
- (Companion) the ``asb_metadata`` block in the integration return
  has the dedup + executable invariants enforced by the helper.
"""
from __future__ import annotations

import inspect
import json as _json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _real_config() -> MagicMock:
    """Config mock with the str fields RAGRequest validates as ``str``."""
    cfg = MagicMock()
    cfg.llm.default_provider = "deepseek"
    cfg.llm.default_model = "deepseek-chat"
    return cfg


def _kb_meta(name: str, model: str = "text-embedding-3-small") -> SimpleNamespace:
    """Minimal KB metadata stand-in with the attrs the MCP tools read."""
    return SimpleNamespace(name=name, embedding_model=model, collection_name=f"coll_{name}")


def test_mcp_server_imports_and_uses_helper():
    """White-box: confirm the helper is wired into both tool bodies."""
    from perspicacite.mcp import server as mcp_mod

    src = inspect.getsource(mcp_mod)
    assert "build_asb_response_metadata" in src, "helper must be imported"

    # generate_report path — between its def and the next @mcp.tool, the
    # helper must appear in the body AND the return must include
    # ``asb_metadata``.
    gr_idx = src.index("async def generate_report(")
    next_tool = src.index("@mcp.tool()", gr_idx + 1)
    gr_body = src[gr_idx:next_tool]
    assert "build_asb_response_metadata" in gr_body, (
        "generate_report missing helper wire-up"
    )
    assert "asb_metadata" in gr_body, (
        "generate_report return missing asb_metadata key"
    )

    # search_knowledge_base path — same checks against its body.
    skb_idx = src.index("async def search_knowledge_base(")
    next_tool = src.index("@mcp.tool()", skb_idx + 1)
    skb_body = src[skb_idx:next_tool]
    assert "build_asb_response_metadata" in skb_body, (
        "search_knowledge_base missing helper wire-up"
    )
    assert "asb_metadata" in skb_body, (
        "search_knowledge_base return missing asb_metadata key"
    )


@pytest.mark.asyncio
async def test_generate_report_returns_asb_metadata_block_for_asb_sources() -> None:
    """Stub ``query_stream`` to emit a source carrying ASB
    skill_body metadata; ``generate_report`` must surface that as an
    ``asb_metadata`` block in its return envelope."""
    import perspicacite.mcp.server as mcp_server
    from perspicacite.mcp.server import MCPState, generate_report
    from perspicacite.models.rag import StreamEvent
    from perspicacite.rag.engine import RAGEngine

    state = MCPState()
    state.initialized = True
    state.config = _real_config()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None

    meta_solo = _kb_meta("solo")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(return_value=meta_solo)
    state.session_store = ss_mock

    asb_source = {
        "title": "Skill One",
        "authors": ["A. Author"],
        "year": 2025,
        "doi": None,
        "relevance_score": 0.9,
        "section": None,
        "kb_name": "solo",
        "metadata": {
            "content_kind": "skill_body",
            "skill_id": "skill-one",
            "skill_name": "Skill One",
            "tools": [
                {
                    "name": "T1",
                    "canonical_url": "https://example/t1",
                    "install": "pip install t1",
                }
            ],
            "environment": [{"language": "python"}],
            "parameters": [],
        },
    }

    class _CapturingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(event="content", data=_json.dumps({"delta": "report body"}))
            # Emit a source whose model_dump shape carries the asb metadata.
            yield StreamEvent(event="source", data=_json.dumps(asb_source))
            yield StreamEvent(event="done", data="{}")

    import perspicacite.rag.engine as _engine_mod

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _CapturingRAGEngine  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            result_str = await generate_report(
                query="what does skill-one do?",
                kb_name="solo",
                mode="advanced",
            )
    finally:
        _engine_mod.RAGEngine = original_cls

    result = _json.loads(result_str)
    assert result.get("success") is True, f"tool returned error: {result}"

    # The asb_metadata block must surface the skill_id from the source.
    asb_md = result.get("asb_metadata")
    assert isinstance(asb_md, dict), f"asb_metadata missing from envelope: {result.keys()}"
    skill_ids = {s["skill_id"] for s in asb_md.get("skill_metadata", [])}
    assert skill_ids == {"skill-one"}, (
        f"expected skill_id 'skill-one' in asb_metadata.skill_metadata, got {skill_ids}"
    )
    # No workflow cards were emitted in this fixture.
    assert asb_md.get("workflow_metadata") == []

    # The single tool has both canonical_url + install → executable=True.
    sk = asb_md["skill_metadata"][0]
    assert sk.get("executable") is True
    assert sk.get("asb_mcp_hint") == "asb://skill/skill-one"

    # The source dict must also carry the underlying metadata so
    # downstream clients can re-derive the block themselves.
    sources = result.get("sources") or []
    assert sources, "no sources in envelope"
    assert sources[0].get("metadata") is not None, (
        "MCP source dict must carry per-source metadata"
    )


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_asb_metadata_block_for_asb_chunks() -> None:
    """Multi-KB ``search_knowledge_base`` path: when retrieval surfaces
    a chunk whose ``ChunkMetadata.paper_metadata_json`` carries an ASB
    skill_body payload, the tool must decode it and surface an
    ``asb_metadata`` block alongside ``results``."""
    import json as _json_local

    import perspicacite.mcp.server as mcp_server
    from perspicacite.mcp.server import MCPState, search_knowledge_base

    state = MCPState()
    state.initialized = True
    state.config = MagicMock()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None

    meta_alpha = _kb_meta("alpha")
    meta_beta = _kb_meta("beta")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(
        side_effect=lambda name: {"alpha": meta_alpha, "beta": meta_beta}.get(name)
    )
    state.session_store = ss_mock

    asb_payload = {
        "content_kind": "skill_body",
        "skill_id": "skill-multi",
        "skill_name": "Skill Multi",
        "tools": [
            {
                "name": "T1",
                "canonical_url": "https://example/t1",
                "install": "pip install t1",
            }
        ],
        "environment": [],
        "parameters": [],
    }
    asb_blob = _json_local.dumps(asb_payload)

    class _FakeMultiKBRetriever:
        def __init__(self, *, vector_store, embedding_service, kb_metas, **kw):
            self.kb_metas = list(kb_metas)

        async def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
            # The retrieval row's ``metadata`` is a ChunkMetadata-like
            # object exposing ``paper_metadata_json``; the MCP tool
            # decodes it via decode_paper_metadata_json.
            return [
                {
                    "text": "skill body chunk",
                    "score": 0.9,
                    "paper_id": "asb_skill:skill-multi",
                    "metadata": SimpleNamespace(
                        title="Skill Multi",
                        section="abstract",
                        doi=None,
                        paper_metadata_json=asb_blob,
                    ),
                    "kb_name": "alpha",
                },
            ]

    import perspicacite.retrieval.multi_kb as _multi_kb_mod

    original_retr_cls = _multi_kb_mod.MultiKBRetriever
    _multi_kb_mod.MultiKBRetriever = _FakeMultiKBRetriever  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            result_str = await search_knowledge_base(
                query="skill-multi",
                top_k=3,
                kb_names=["alpha", "beta"],
            )
    finally:
        _multi_kb_mod.MultiKBRetriever = original_retr_cls

    result = _json.loads(result_str)
    assert result.get("success") is True, f"tool returned error: {result}"

    # Per-chunk metadata is the decoded ASB payload.
    chunks = result.get("results") or []
    assert chunks, "no chunks in envelope"
    assert chunks[0].get("metadata") == asb_payload

    # The ``asb_metadata`` block surfaces the ASB skill.
    asb_md = result.get("asb_metadata")
    assert isinstance(asb_md, dict)
    skill_ids = {s["skill_id"] for s in asb_md.get("skill_metadata", [])}
    assert skill_ids == {"skill-multi"}, (
        f"expected 'skill-multi' in asb_metadata.skill_metadata, got {skill_ids}"
    )
