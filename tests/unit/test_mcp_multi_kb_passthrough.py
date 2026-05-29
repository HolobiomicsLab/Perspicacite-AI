"""Regression tests: MCP tools propagate kb_names through to retrieval.

Locks in the kb_names plumbing for `generate_report` and `search_knowledge_base`
described in CLAUDE.md:

  > "the `generate_report` and `search_knowledge_base` MCP tools also accept an
  > optional `kb_names` list with the same compat check."

These tests do not exercise live retrieval; they verify that:
- `generate_report` builds a `RAGRequest` whose `kb_names` matches the caller's
  argument and runs `check_embedding_compat` on the resolved KB metadatas.
- `search_knowledge_base` instantiates a `MultiKBRetriever` with the resolved
  KB metadatas and returns results tagged with `kb_name`.
"""

from __future__ import annotations

import json as _json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch


def _real_config() -> MagicMock:
    """Config mock with real string fields that pass RAGRequest validation.

    Uses MagicMock (not SimpleNamespace) so that mode-handler __init__ calls
    like ``config.knowledge_base.default_top_k`` auto-create attrs rather than
    raising AttributeError. Only the llm fields are set to real strings because
    those are the only ones RAGRequest validates as ``str``.
    """
    cfg = MagicMock()
    cfg.llm.default_provider = "deepseek"
    cfg.llm.default_model = "deepseek-chat"
    return cfg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest

import perspicacite.mcp.server as mcp_server
from perspicacite.mcp.server import MCPState, generate_report, search_knowledge_base
from perspicacite.models.rag import StreamEvent
from perspicacite.rag.engine import RAGEngine


def _kb_meta(name: str, model: str = "text-embedding-3-small") -> SimpleNamespace:
    """Build a minimal KB metadata stand-in with the attrs the MCP tools read."""
    return SimpleNamespace(name=name, embedding_model=model, collection_name=f"coll_{name}")


@pytest.mark.asyncio
async def test_generate_report_passes_kb_names_to_rag_request() -> None:
    """`generate_report(kb_names=[...])` must build a RAGRequest carrying that list."""
    captured_requests: list[Any] = []

    state = MCPState()
    state.initialized = True
    state.config = _real_config()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None

    # Provide compatible kb metadatas for both requested KBs.
    meta_alpha = _kb_meta("alpha")
    meta_beta = _kb_meta("beta")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(
        side_effect=lambda name: {"alpha": meta_alpha, "beta": meta_beta}.get(name)
    )
    state.session_store = ss_mock

    class _CapturingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            captured_requests.append(req)
            yield StreamEvent(event="content", data='{"delta": "ok"}')
            yield StreamEvent(event="done", data="{}")

    import perspicacite.rag.engine as _engine_mod

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _CapturingRAGEngine  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            result_str = await generate_report(
                query="multi kb question",
                mode="advanced",
                kb_names=["alpha", "beta"],
            )
    finally:
        _engine_mod.RAGEngine = original_cls

    result = _json.loads(result_str)
    assert result.get("success") is True, f"tool returned error: {result}"
    assert result.get("kb_names") == ["alpha", "beta"]

    assert captured_requests, "query_stream was never invoked"
    req = captured_requests[0]
    assert getattr(req, "kb_names", None) == ["alpha", "beta"], (
        f"RAGRequest.kb_names must propagate, got {getattr(req, 'kb_names', None)!r}"
    )


@pytest.mark.asyncio
async def test_generate_report_rejects_incompatible_kb_embeddings() -> None:
    """The compat check must fire when KBs use different embedding models."""
    state = MCPState()
    state.initialized = True
    state.config = MagicMock()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None

    meta_a = _kb_meta("alpha", model="model-A")
    meta_b = _kb_meta("beta", model="model-B")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(
        side_effect=lambda name: {"alpha": meta_a, "beta": meta_b}.get(name)
    )
    state.session_store = ss_mock

    with patch.object(mcp_server, "mcp_state", state):
        result_str = await generate_report(
            query="q", mode="advanced", kb_names=["alpha", "beta"]
        )

    result = _json.loads(result_str)
    assert result.get("success") is False
    assert "different embedding models" in result.get("error", "")


@pytest.mark.asyncio
async def test_generate_report_single_kb_names_collapses_to_kb_name() -> None:
    """When `kb_names` has exactly one entry, it must override kb_name."""
    captured_requests: list[Any] = []

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

    class _CapturingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            captured_requests.append(req)
            yield StreamEvent(event="content", data='{"delta": "ok"}')
            yield StreamEvent(event="done", data="{}")

    import perspicacite.rag.engine as _engine_mod

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _CapturingRAGEngine  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(
                query="q",
                kb_name="ignored_default",
                mode="advanced",
                kb_names=["solo"],
            )
    finally:
        _engine_mod.RAGEngine = original_cls

    assert captured_requests, "query_stream was never invoked"
    req = captured_requests[0]
    # Single-entry list collapses: kb_names not propagated, kb_name overridden.
    assert req.kb_name == "solo"
    assert getattr(req, "kb_names", None) is None


@pytest.mark.asyncio
async def test_search_knowledge_base_fans_out_via_multi_kb_retriever() -> None:
    """search_knowledge_base must instantiate MultiKBRetriever with the
    resolved KB metadatas and surface their kb_name in the response."""
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

    captured_init_kwargs: dict[str, Any] = {}

    class _FakeMultiKBRetriever:
        def __init__(self, *, vector_store, embedding_service, kb_metas, **kw):
            captured_init_kwargs["vector_store"] = vector_store
            captured_init_kwargs["embedding_service"] = embedding_service
            captured_init_kwargs["kb_metas"] = list(kb_metas)

        async def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
            return [
                {
                    "text": "hit from alpha",
                    "score": 0.9,
                    "paper_id": "p1",
                    "metadata": SimpleNamespace(
                        title="Paper One", section="abstract", doi="10.1/p1"
                    ),
                    "kb_name": "alpha",
                },
                {
                    "text": "hit from beta",
                    "score": 0.7,
                    "paper_id": "p2",
                    "metadata": SimpleNamespace(
                        title="Paper Two", section="results", doi="10.1/p2"
                    ),
                    "kb_name": "beta",
                },
            ]

    # search_knowledge_base does `from perspicacite.retrieval.multi_kb import
    # MultiKBRetriever, check_embedding_compat` inside the function body,
    # so we patch on the source module.
    import perspicacite.retrieval.multi_kb as _multi_kb_mod

    original_retr_cls = _multi_kb_mod.MultiKBRetriever
    _multi_kb_mod.MultiKBRetriever = _FakeMultiKBRetriever  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            result_str = await search_knowledge_base(
                query="multi kb search",
                top_k=3,
                kb_names=["alpha", "beta"],
            )
    finally:
        _multi_kb_mod.MultiKBRetriever = original_retr_cls

    result = _json.loads(result_str)
    assert result.get("success") is True, f"tool returned error: {result}"

    # Tool must echo the kb_names the caller asked for.
    assert result.get("kb_names") == ["alpha", "beta"]

    # MultiKBRetriever must have been constructed with the resolved metadatas
    # (one per requested kb_name, in order).
    metas = captured_init_kwargs.get("kb_metas")
    assert metas is not None, "MultiKBRetriever was never instantiated"
    assert [getattr(m, "name", None) for m in metas] == ["alpha", "beta"]
    assert captured_init_kwargs["vector_store"] is state.vector_store
    assert captured_init_kwargs["embedding_service"] is state.embedding_provider

    # Each chunk in the response must be tagged with its originating KB.
    chunks = result.get("results") or []
    kb_tags = {c.get("kb_name") for c in chunks}
    assert kb_tags == {"alpha", "beta"}, f"expected per-chunk kb_name tags, got {kb_tags}"


@pytest.mark.asyncio
async def test_search_knowledge_base_rejects_incompatible_kb_embeddings() -> None:
    """search_knowledge_base must reject mixed embedding models with a clear error."""
    state = MCPState()
    state.initialized = True
    state.config = MagicMock()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None

    meta_a = _kb_meta("alpha", model="model-A")
    meta_b = _kb_meta("beta", model="model-B")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(
        side_effect=lambda name: {"alpha": meta_a, "beta": meta_b}.get(name)
    )
    state.session_store = ss_mock

    with patch.object(mcp_server, "mcp_state", state):
        result_str = await search_knowledge_base(
            query="q", top_k=3, kb_names=["alpha", "beta"]
        )

    result = _json.loads(result_str)
    assert result.get("success") is False
    assert "different embedding models" in result.get("error", "")


@pytest.mark.asyncio
async def test_search_knowledge_base_single_kb_names_uses_single_kb_path() -> None:
    """A `kb_names` list of length 1 must collapse to the single-KB path."""
    state = MCPState()
    state.initialized = True
    state.config = MagicMock()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None
    state.embedding_provider.dimension = 8

    meta_solo = _kb_meta("solo")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(return_value=meta_solo)
    state.session_store = ss_mock

    # MultiKBRetriever should NOT be instantiated for a 1-entry list.
    multi_kb_called = {"flag": False}

    import perspicacite.retrieval.multi_kb as _multi_kb_mod

    class _Sentinel:
        def __init__(self, *a, **k):
            multi_kb_called["flag"] = True

    original_retr_cls = _multi_kb_mod.MultiKBRetriever
    _multi_kb_mod.MultiKBRetriever = _Sentinel  # type: ignore[assignment]

    # Stub DynamicKnowledgeBase.search to avoid touching ChromaDB.
    import perspicacite.rag.dynamic_kb as _dkb_mod

    captured_collection: dict[str, Any] = {}
    original_dkb_cls = _dkb_mod.DynamicKnowledgeBase

    class _FakeDKB:
        def __init__(self, vector_store, embedding_provider, config=None):
            self.collection_name = None
            self._initialized = False

        async def search(self, query: str, top_k: int = 5, filters=None):
            captured_collection["collection_name"] = self.collection_name
            return []

    _dkb_mod.DynamicKnowledgeBase = _FakeDKB  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            result_str = await search_knowledge_base(
                query="q",
                kb_name="ignored_default",
                top_k=3,
                kb_names=["solo"],
            )
    finally:
        _multi_kb_mod.MultiKBRetriever = original_retr_cls
        _dkb_mod.DynamicKnowledgeBase = original_dkb_cls

    result = _json.loads(result_str)
    assert result.get("success") is True, f"tool returned error: {result}"
    assert multi_kb_called["flag"] is False, (
        "MultiKBRetriever must not be used for a single-entry kb_names list"
    )
    assert result.get("kb_name") == "solo"
    # Single-KB path selects the collection name for the resolved kb_name.
    assert "solo" in (captured_collection.get("collection_name") or "")


def _route_kbs_state(metas: list[SimpleNamespace]) -> MCPState:
    state = MCPState()
    state.initialized = True
    state.config = MagicMock()
    state.llm_client = MagicMock()
    state.vector_store = MagicMock()
    ss_mock = MagicMock()
    ss_mock.list_kbs = AsyncMock(return_value=metas)
    state.session_store = ss_mock
    return state


@pytest.mark.asyncio
async def test_route_kbs_accepts_kb_names_alias(monkeypatch) -> None:
    """route_kbs must accept `kb_names` as an alias for `candidate_kbs` so agents
    that reach for the sibling-tool name don't error on the first call."""
    state = _route_kbs_state([_kb_meta("kept"), _kb_meta("dropped")])

    captured: dict[str, Any] = {}

    async def _fake_auto_route_kbs(*, kb_metas, **kw):
        captured["names"] = [m.name for m in kb_metas]
        return []

    import perspicacite.llm.client as _client_mod
    import perspicacite.rag.kb_router as _router_mod

    monkeypatch.setattr(_router_mod, "auto_route_kbs", _fake_auto_route_kbs)
    monkeypatch.setattr(
        _client_mod, "resolve_stage_model", lambda cfg, stage: ("p", "m")
    )

    with patch.object(mcp_server, "mcp_state", state):
        out = await mcp_server.route_kbs(query="q", kb_names=["kept"])

    assert "hits" in out, f"unexpected route_kbs payload: {out}"
    assert captured.get("names") == ["kept"], (
        "kb_names must filter candidate KBs exactly like candidate_kbs"
    )


@pytest.mark.asyncio
async def test_route_kbs_candidate_kbs_takes_precedence(monkeypatch) -> None:
    """When both candidate_kbs and kb_names are given, candidate_kbs wins."""
    state = _route_kbs_state([_kb_meta("a"), _kb_meta("b")])

    captured: dict[str, Any] = {}

    async def _fake_auto_route_kbs(*, kb_metas, **kw):
        captured["names"] = [m.name for m in kb_metas]
        return []

    import perspicacite.llm.client as _client_mod
    import perspicacite.rag.kb_router as _router_mod

    monkeypatch.setattr(_router_mod, "auto_route_kbs", _fake_auto_route_kbs)
    monkeypatch.setattr(
        _client_mod, "resolve_stage_model", lambda cfg, stage: ("p", "m")
    )

    with patch.object(mcp_server, "mcp_state", state):
        await mcp_server.route_kbs(
            query="q", candidate_kbs=["a"], kb_names=["b"]
        )

    assert captured.get("names") == ["a"], "candidate_kbs must win over kb_names"
