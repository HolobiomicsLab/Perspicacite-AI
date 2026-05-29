"""When a title-like query returns 0 hits, the adapter should retry
once with a normalised form (drop post-colon subtitle, drop parens)."""
from unittest.mock import patch

import pytest


def test_normalize_title_strips_subtitle_after_colon():
    from perspicacite.search.title_normalize import normalize_title
    assert normalize_title(
        "AgentSquare: Automatic LLM Agent Search in Modular Design Space"
    ) == "AgentSquare"


def test_normalize_title_strips_parentheticals():
    from perspicacite.search.title_normalize import normalize_title
    assert normalize_title(
        "Promptbreeder (v2): Self-Referential Self-Improvement"
    ) == "Promptbreeder"


def test_normalize_title_returns_input_when_already_short():
    from perspicacite.search.title_normalize import normalize_title
    assert normalize_title("Attention") == "Attention"


def test_normalize_title_empty_falls_back_to_original():
    from perspicacite.search.title_normalize import normalize_title
    # Pathological: ":subtitle" → would strip to empty; we keep original
    assert normalize_title(":subtitle only") == ":subtitle only"


@pytest.mark.asyncio
async def test_search_retries_once_on_zero_result_titlelike_query():
    """SciLExAdapter.search must call into the inner search path twice
    when the first call returns [] and the query is title-like."""
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.search.scilex_adapter import SciLExAdapter

    calls: list[str] = []

    async def fake_inner(self, **kw):
        calls.append(kw["query"])
        if len(calls) == 1:
            return []
        return [Paper(id="x", title="AgentSquare", source=PaperSource.SCILEX)]

    adapter = SciLExAdapter()
    # Patch the helper that does the actual work — name discovered via grep
    with patch.object(SciLExAdapter, "_search_once", new=fake_inner, create=False):
        out = await adapter.search(
            query="AgentSquare: Automatic LLM Agent Search in Modular Design Space",
            max_results=5,
            apis=["semantic_scholar"],
        )
    assert len(out) == 1
    assert len(calls) == 2
    assert calls[0].startswith("AgentSquare:")
    assert calls[1] == "AgentSquare"
    # Retry-derived papers carry an annotation
    assert (out[0].metadata or {}).get("search_normalized_from", "").startswith("AgentSquare:")


@pytest.mark.asyncio
async def test_search_no_retry_when_first_pass_succeeds():
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.search.scilex_adapter import SciLExAdapter

    calls: list[str] = []

    async def fake_inner(self, **kw):
        calls.append(kw["query"])
        return [Paper(id="x", title=kw["query"], source=PaperSource.SCILEX)]

    adapter = SciLExAdapter()
    with patch.object(SciLExAdapter, "_search_once", new=fake_inner, create=False):
        out = await adapter.search(query="Attention Is All You Need", max_results=5,
                                   apis=["semantic_scholar"])
    assert len(out) == 1
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_search_no_retry_when_query_is_not_titlelike():
    """A query without ':' or '(...)' shouldn't trigger normalize-retry."""
    from perspicacite.search.scilex_adapter import SciLExAdapter

    calls: list[str] = []

    async def fake_inner(self, **kw):
        calls.append(kw["query"])
        return []

    adapter = SciLExAdapter()
    with patch.object(SciLExAdapter, "_search_once", new=fake_inner, create=False):
        out = await adapter.search(query="diamond magnetometry", max_results=5,
                                   apis=["semantic_scholar"])
    assert out == []
    assert len(calls) == 1  # no retry
