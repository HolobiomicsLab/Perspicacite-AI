# tests/integration/test_search_optimization_e2e.py
"""End-to-end: chat router -> grounding extractor -> basic mode ->
query optimizer -> aggregator. All LLM and aggregator calls are stubbed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_multi_turn_chat_search_uses_grounded_optimized_query():
    """Simulate: user has a multi-turn conversation, second turn triggers
    a search. We expect the aggregator to receive an optimized query that
    benefitted from the auto-extracted grounding context."""

    # Stub LLM: first call is grounding extractor, second is optimizer.
    llm_responses = [
        '{"context": "LSD1 inhibitors in AML"}',
        '{"searched_query": "LSD1 inhibitor mechanism acute myeloid leukemia"}',
    ]

    async def fake_complete(*_args, **kwargs):
        return llm_responses.pop(0)

    fake_agg = MagicMock()
    fake_agg._providers = []
    fake_agg.available = True
    fake_agg.search = AsyncMock(return_value=[])

    # Load the singleton before patching so we reference the real object.
    from perspicacite.config.schema import Config
    from perspicacite.models.rag import StreamEvent
    from perspicacite.web import state as _state_mod

    # Stash originals so we can restore after the test.
    _orig_initialized = _state_mod.app_state.initialized
    _orig_llm_client = _state_mod.app_state.llm_client
    _orig_rag_engine = _state_mod.app_state.rag_engine
    _orig_session_store = _state_mod.app_state.session_store
    _orig_provenance_store = _state_mod.app_state.provenance_store
    _orig_config = getattr(_state_mod.app_state, "config", None)

    try:
        # Mark as already initialized so the endpoint skips bootstrap.
        _state_mod.app_state.initialized = True
        _state_mod.app_state.session_store = None
        _state_mod.app_state.provenance_store = None

        # Wire up a minimal config so query_optimization settings are readable.
        _state_mod.app_state.config = Config()

        # Wire up a stub LLM client whose complete() consumes our fake responses.
        stub_llm = MagicMock()
        stub_llm.complete = AsyncMock(side_effect=fake_complete)
        _state_mod.app_state.llm_client = stub_llm

        # Build a stub RAG engine whose query_stream calls _web_fallback_papers
        # directly (simulating what BasicRAGMode.execute_stream does when the
        # KB returns nothing), threading app_state and _resolved_context through
        # from the rag_request attributes.
        async def _fake_query_stream(request, *, message_id=None, conversation_id=None):
            """Minimal stub: skip KB search, call web fallback directly."""
            from perspicacite.rag.modes.basic import _web_fallback_papers

            await _web_fallback_papers(
                query=request.query,
                databases=request.databases,
                max_docs=5,
                app_state=getattr(request, "app_state", None),
                context=getattr(request, "_resolved_context", None),
            )
            yield StreamEvent.content("stub answer")
            yield StreamEvent.done(
                conversation_id=conversation_id or "",
                tokens_used=0,
                mode="basic",
                iterations=1,
            )

        stub_engine = MagicMock()
        stub_engine.query_stream = _fake_query_stream
        _state_mod.app_state.rag_engine = stub_engine

        with patch(
            "perspicacite.search.domain_aggregator.build_aggregator",
            return_value=fake_agg,
        ):
            from perspicacite.web.routers.chat import (
                ChatMessage,
                ChatRequest,
                chat_endpoint,
            )
            request = ChatRequest(
                query="how does it work",
                messages=[
                    ChatMessage(role="user", content="tell me about LSD1 inhibitors"),
                    ChatMessage(
                        role="assistant",
                        content="LSD1 inhibitors target histone demethylase activity "
                                "and are studied in AML.",
                    ),
                ],
                mode="basic",
                stream=False,
            )
            await chat_endpoint(request, raw_request=MagicMock())

    finally:
        # Restore original app_state to avoid test pollution.
        _state_mod.app_state.initialized = _orig_initialized
        _state_mod.app_state.llm_client = _orig_llm_client
        _state_mod.app_state.rag_engine = _orig_rag_engine
        _state_mod.app_state.session_store = _orig_session_store
        _state_mod.app_state.provenance_store = _orig_provenance_store
        if _orig_config is not None:
            _state_mod.app_state.config = _orig_config

    # The aggregator should have received the optimized rewrite, not the
    # original "how does it work" or the keyword query.
    final_query = fake_agg.search.call_args.kwargs["query"]
    assert final_query == "LSD1 inhibitor mechanism acute myeloid leukemia"
    assert llm_responses == []  # both LLM calls were consumed
