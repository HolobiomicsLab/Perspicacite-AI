"""Unit tests for SciLExAdapter.search_with_warnings dropped-APIs reporting."""
from unittest.mock import AsyncMock, patch

import pytest

from perspicacite.search.scilex_adapter import (
    SciLExAdapter,
    SciLExSearchResult,
)


@pytest.mark.asyncio
async def test_search_with_warnings_reports_unknown_apis():
    """google_scholar isn't SciLEx-backed; expect it in dropped_apis."""
    adapter = SciLExAdapter()

    async def fake_search(*args, **kwargs):
        adapter._last_dropped_apis = ["google_scholar"]
        return []

    with patch.object(adapter, "search", AsyncMock(side_effect=fake_search)):
        result = await adapter.search_with_warnings(
            query="x", apis=["semantic_scholar", "google_scholar"],
        )
    assert isinstance(result, SciLExSearchResult)
    assert result.dropped_apis == ["google_scholar"]
    assert result.papers == []


@pytest.mark.asyncio
async def test_search_with_warnings_empty_when_all_known():
    """All-known apis -> empty dropped_apis list."""
    adapter = SciLExAdapter()

    async def fake_search(*args, **kwargs):
        adapter._last_dropped_apis = []
        return []

    with patch.object(adapter, "search", AsyncMock(side_effect=fake_search)):
        result = await adapter.search_with_warnings(
            query="x", apis=["semantic_scholar", "openalex"],
        )
    assert result.dropped_apis == []


@pytest.mark.asyncio
async def test_search_with_warnings_clears_state_between_calls():
    """State from a previous call doesn't bleed into the next."""
    adapter = SciLExAdapter()
    adapter._last_dropped_apis = ["leftover"]

    async def fake_search(*args, **kwargs):
        adapter._last_dropped_apis = []
        return []

    with patch.object(adapter, "search", AsyncMock(side_effect=fake_search)):
        result = await adapter.search_with_warnings(query="x")
    assert result.dropped_apis == []
