"""Unit tests for HyDE (Hypothetical Document Embeddings) query helper.

Tests cover:
- Happy path: LLM returns a valid synthetic abstract.
- Fallback: LLM raises an exception — original claim is returned unchanged.
- Empty input: empty/whitespace claim is returned as-is without an LLM call.
- RAGRequest model field: ``use_hyde`` is present and defaults to False.
- basic.py integration smoke test: ``use_hyde`` flag is wired into the source.
- ChatRequest integration smoke test: ``use_hyde`` is forwarded to RAGRequest.
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Unit tests for generate_hyde_query()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hyde_returns_synthetic_abstract():
    """When the LLM succeeds, the synthetic abstract is returned."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(
        return_value=(
            "Prevalence studies using immunohistochemistry on tonsil biopsies "
            "detected PrP^Sc deposition in approximately 1 in 2000 UK individuals, "
            "suggesting subclinical variant CJD infection in the general population."
        )
    )

    from perspicacite.rag.modes.hyde_query import generate_hyde_query

    result = await generate_hyde_query(
        claim="1/2000 in UK have abnormal PrP positivity",
        llm_client=mock_llm,
        model="deepseek-chat",
        provider="deepseek",
    )

    assert "PrP" in result or "prion" in result.lower() or "prevalence" in result.lower()
    assert result != "1/2000 in UK have abnormal PrP positivity"
    mock_llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_hyde_falls_back_on_llm_error():
    """When the LLM raises, the original claim is returned unchanged."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    from perspicacite.rag.modes.hyde_query import generate_hyde_query

    claim = "patients with BRCA1 mutation have elevated cancer risk"
    result = await generate_hyde_query(
        claim=claim,
        llm_client=mock_llm,
        model="claude-haiku-4-5",
        provider="anthropic",
    )

    assert result == claim


@pytest.mark.asyncio
async def test_hyde_falls_back_on_empty_llm_response():
    """When the LLM returns an empty string, the original claim is returned."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="   ")

    from perspicacite.rag.modes.hyde_query import generate_hyde_query

    claim = "aspirin reduces cardiovascular risk"
    result = await generate_hyde_query(
        claim=claim,
        llm_client=mock_llm,
        model="deepseek-chat",
        provider="deepseek",
    )

    assert result == claim


@pytest.mark.asyncio
async def test_hyde_empty_claim_skips_llm():
    """An empty claim should be returned as-is without calling the LLM."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="should not be called")

    from perspicacite.rag.modes.hyde_query import generate_hyde_query

    for empty in ("", "   ", None):
        result = await generate_hyde_query(
            claim=empty or "",
            llm_client=mock_llm,
            model="deepseek-chat",
            provider="deepseek",
        )
        assert result == (empty or "")

    mock_llm.complete.assert_not_awaited()


# ---------------------------------------------------------------------------
# RAGRequest field test
# ---------------------------------------------------------------------------


def test_rag_request_use_hyde_field_defaults_false():
    """RAGRequest.use_hyde must exist and default to False."""
    from perspicacite.models.rag import RAGRequest

    req = RAGRequest(query="test claim")
    assert req.use_hyde is False


def test_rag_request_use_hyde_field_can_be_set_true():
    """RAGRequest.use_hyde can be set to True at construction time."""
    from perspicacite.models.rag import RAGRequest

    req = RAGRequest(query="test claim", use_hyde=True)
    assert req.use_hyde is True


# ---------------------------------------------------------------------------
# ChatRequest field test
# ---------------------------------------------------------------------------


def test_chat_request_use_hyde_field_defaults_false():
    """ChatRequest.use_hyde must exist and default to False."""
    from perspicacite.web.routers.chat import ChatRequest

    req = ChatRequest(query="test query")
    assert req.use_hyde is False


def test_chat_request_use_hyde_field_can_be_set_true():
    """ChatRequest.use_hyde=True is accepted and stored."""
    from perspicacite.web.routers.chat import ChatRequest

    req = ChatRequest(query="test query", use_hyde=True)
    assert req.use_hyde is True


# ---------------------------------------------------------------------------
# Source-file smoke tests (typo / wiring defense)
# ---------------------------------------------------------------------------

_BASIC_SRC = pathlib.Path(__file__).resolve().parents[2] / (
    "src/perspicacite/rag/modes/basic.py"
)
_HYDE_SRC = pathlib.Path(__file__).resolve().parents[2] / (
    "src/perspicacite/rag/modes/hyde_query.py"
)
_CHAT_SRC = pathlib.Path(__file__).resolve().parents[2] / (
    "src/perspicacite/web/routers/chat.py"
)


def test_basic_mode_imports_hyde_query():
    """basic.py must reference the hyde_query import."""
    text = _BASIC_SRC.read_text()
    assert "hyde_query" in text, "basic.py does not import hyde_query"
    assert "use_hyde" in text, "basic.py does not check use_hyde flag"


def test_hyde_query_module_has_generate_function():
    """hyde_query.py must define generate_hyde_query and the system prompt."""
    text = _HYDE_SRC.read_text()
    assert "async def generate_hyde_query" in text
    assert "_SYSTEM_PROMPT" in text


def test_chat_router_forwards_use_hyde():
    """chat.py must include use_hyde in the RAGReq constructor call."""
    text = _CHAT_SRC.read_text()
    assert "use_hyde" in text, "chat.py does not forward use_hyde to RAGRequest"


# ---------------------------------------------------------------------------
# _resolve_hyde_model tests
# ---------------------------------------------------------------------------


def test_resolve_hyde_model_no_config():
    """When config is None, returns the cheap fallback model."""
    from perspicacite.rag.modes.basic import (
        _HYDE_FALLBACK_MODEL,
        _HYDE_FALLBACK_PROVIDER,
        _resolve_hyde_model,
    )

    provider, model = _resolve_hyde_model(None)
    assert provider == _HYDE_FALLBACK_PROVIDER
    assert model == _HYDE_FALLBACK_MODEL


def test_resolve_hyde_model_unpinned_stage():
    """When the 'hyde' stage is not in config, returns fallback model."""
    from perspicacite.rag.modes.basic import (
        _HYDE_FALLBACK_MODEL,
        _HYDE_FALLBACK_PROVIDER,
        _resolve_hyde_model,
    )

    cfg = MagicMock()
    cfg.llm.models = {}          # 'hyde' stage not pinned
    cfg.llm.providers_per_stage = {}

    provider, model = _resolve_hyde_model(cfg)
    assert provider == _HYDE_FALLBACK_PROVIDER
    assert model == _HYDE_FALLBACK_MODEL


def test_resolve_hyde_model_pinned_stage():
    """When 'hyde' stage is pinned in config, uses that model."""
    from perspicacite.rag.modes.basic import _resolve_hyde_model

    cfg = MagicMock()
    cfg.llm.models = {"hyde": "claude-haiku-4-5"}
    cfg.llm.providers_per_stage = {"hyde": "anthropic"}

    provider, model = _resolve_hyde_model(cfg)
    assert provider == "anthropic"
    assert model == "claude-haiku-4-5"
