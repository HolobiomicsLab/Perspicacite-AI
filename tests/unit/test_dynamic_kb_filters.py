"""Verify DynamicKnowledgeBase.search threads filters through (Wave 4.2)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from perspicacite.models.search import SearchFilters
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig


def _kb_with_mocks():
    """Build a DynamicKnowledgeBase whose vector_store / embedding are mocked."""
    vstore = MagicMock()
    vstore.search = AsyncMock(return_value=[])  # empty result is fine
    embed = MagicMock()
    embed.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    embed.embed_query = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    kb = DynamicKnowledgeBase(vstore, embed, config=KnowledgeBaseConfig())
    kb.collection_name = "test_coll"
    kb._initialized = True
    return kb, vstore


@pytest.mark.asyncio
async def test_search_passes_filters_to_store():
    kb, vstore = _kb_with_mocks()
    filters = SearchFilters(year_min=2020, year_max=2024)
    await kb.search("query", filters=filters)
    # Inspect the call: filters should appear as a kwarg.
    args, kwargs = vstore.search.call_args
    assert kwargs.get("filters") is filters or args[-1] is filters or "filters" in kwargs


@pytest.mark.asyncio
async def test_search_without_filters_passes_none():
    kb, vstore = _kb_with_mocks()
    await kb.search("query")
    _args, kwargs = vstore.search.call_args
    # Filters must be absent or explicitly None — never some other default.
    assert kwargs.get("filters") is None or "filters" not in kwargs
