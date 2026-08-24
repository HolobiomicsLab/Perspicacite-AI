"""The KB listing must read counts in bulk, not once per knowledge base.

Scanning every collection through the Chroma API blocked the event loop for
minutes on a corpus of thousands of KBs, stalling concurrent RAG traffic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from perspicacite.web.routers.kb import list_knowledge_bases
from perspicacite.web.state import app_state


def _kb(name: str, paper_count: int = 0) -> SimpleNamespace:
    """Build the minimal knowledge base shape the listing route reads."""
    return SimpleNamespace(
        name=name,
        description="",
        collection_name=f"kb_{name}",
        paper_count=paper_count,
        created_at=None,
    )


class _RecordingStore:
    """Vector store that counts how often the bulk stats call is made."""

    def __init__(self, stats: dict[str, dict[str, int]]):
        self.stats = stats
        self.bulk_calls = 0

    async def all_collection_stats(self) -> dict[str, dict[str, int]]:
        """Return canned stats and record the call."""
        self.bulk_calls += 1
        return self.stats

    async def get_collection_stats(self, collection: str) -> dict[str, int]:
        """Fail loudly: the listing must never fall back to per-collection reads."""
        raise AssertionError(f"per-collection read on {collection}")


@pytest.fixture
def listing_state(monkeypatch):
    """Point app_state at fake stores and hand the test the vector store."""

    def _install(kbs, stats):
        store = _RecordingStore(stats)
        session = SimpleNamespace(list_kbs=lambda: _returns(kbs))
        monkeypatch.setattr(app_state, "session_store", session, raising=False)
        monkeypatch.setattr(app_state, "vector_store", store, raising=False)
        return store

    return _install


async def _returns(value):
    """Await-able passthrough for the fake session store."""
    return value


@pytest.mark.asyncio
async def test_listing_reads_stats_once_for_many_kbs(listing_state):
    """One bulk read serves the whole listing, however many KBs there are."""
    kbs = [_kb(f"p{i}") for i in range(50)]
    stats = {f"kb_p{i}": {"count": i, "unique_papers": 1} for i in range(50)}
    store = listing_state(kbs, stats)

    result = await list_knowledge_bases()

    assert store.bulk_calls == 1
    assert len(result) == 50
    assert result[7]["chunk_count"] == 7
    assert result[7]["paper_count"] == 1


@pytest.mark.asyncio
async def test_listing_falls_back_to_stored_paper_count(listing_state):
    """A KB missing from the index keeps its stored paper count and zero chunks."""
    listing_state([_kb("orphan", paper_count=4)], {})

    result = await list_knowledge_bases()

    assert result[0]["paper_count"] == 4
    assert result[0]["chunk_count"] == 0


@pytest.mark.asyncio
async def test_listing_is_empty_without_a_session_store(monkeypatch):
    """No session store means no knowledge bases, not an error."""
    monkeypatch.setattr(app_state, "session_store", None, raising=False)

    assert await list_knowledge_bases() == []
