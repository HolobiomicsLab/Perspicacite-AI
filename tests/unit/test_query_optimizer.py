# tests/unit/test_query_optimizer.py
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from perspicacite.search.query_optimizer import (
    _PROMPT,
    optimize_query,
)


def _state(enabled: bool = True, timeout_s: float = 5.0, max_ctx: int = 300) -> MagicMock:
    """Build a stub app_state with config.search.query_optimization and an llm_client."""
    state = MagicMock()
    state.config.search.query_optimization.enabled = enabled
    state.config.search.query_optimization.timeout_s = timeout_s
    state.config.search.query_optimization.max_context_chars = max_ctx
    state.config.llm.default_provider = "anthropic"
    state.config.llm.default_model = "claude-haiku-4-5"
    state.config.llm.models = {}
    state.config.llm.providers_per_stage = {}
    state.llm_client = MagicMock()
    state.llm_client.complete = AsyncMock(
        return_value='{"searched_query": "myocardial infarction biomarkers"}'
    )
    return state


@pytest.mark.asyncio
async def test_optimize_disabled_returns_verbatim():
    state = _state()
    result = await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
        optimize_enabled=False,
    )
    assert result.searched_query == "heart attack biomarkers"
    assert result.enabled is False
    assert result.applied is False
    assert result.context_used is False
    assert result.fallback_reason is None
    state.llm_client.complete.assert_not_called()


@pytest.mark.asyncio
async def test_optimize_success_rewrites_query():
    state = _state()
    result = await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
    )
    assert result.searched_query == "myocardial infarction biomarkers"
    assert result.enabled is True
    assert result.applied is True
    assert result.context_used is False
    assert result.fallback_reason is None
    state.llm_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_optimize_model_no_op_returns_unchanged():
    state = _state()
    state.llm_client.complete = AsyncMock(
        return_value='{"searched_query": "heart attack biomarkers"}'
    )
    result = await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
    )
    assert result.searched_query == "heart attack biomarkers"
    assert result.applied is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_optimize_llm_error_falls_back():
    state = _state()
    state.llm_client.complete = AsyncMock(side_effect=RuntimeError("boom"))
    result = await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
    )
    assert result.searched_query == "heart attack biomarkers"
    assert result.applied is False
    assert result.fallback_reason == "llm_error"


@pytest.mark.asyncio
async def test_optimize_unparseable_json_falls_back():
    state = _state()
    state.llm_client.complete = AsyncMock(return_value="not json at all")
    result = await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
    )
    assert result.searched_query == "heart attack biomarkers"
    assert result.applied is False
    assert result.fallback_reason == "unparseable"


@pytest.mark.asyncio
async def test_optimize_timeout_falls_back():
    state = _state(timeout_s=0.05)

    async def slow_complete(**_kwargs):
        await asyncio.sleep(0.5)
        return '{"searched_query": "..."}'

    state.llm_client.complete = AsyncMock(side_effect=slow_complete)
    result = await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
    )
    assert result.searched_query == "heart attack biomarkers"
    assert result.applied is False
    assert result.fallback_reason == "timeout"


@pytest.mark.asyncio
async def test_optimize_context_used_flag_set_when_truncation_keeps_content():
    state = _state(max_ctx=300)
    result = await optimize_query(
        query="LSD1 inhibitor mechanism",
        context="user has been discussing LSD1 inhibitors in AML",
        app_state=state,
    )
    assert result.context_used is True
    # The prompt actually sent should include the context substring.
    sent_msgs = state.llm_client.complete.call_args.kwargs["messages"]
    assert "LSD1 inhibitors in AML" in sent_msgs[0]["content"]


@pytest.mark.asyncio
async def test_optimize_context_truncated_when_too_long():
    state = _state(max_ctx=50)
    long_ctx = "x" * 500
    result = await optimize_query(
        query="some query",
        context=long_ctx,
        app_state=state,
    )
    sent_msgs = state.llm_client.complete.call_args.kwargs["messages"]
    body = sent_msgs[0]["content"]
    # Truncated context appears (50 chars max), full 500-char string does not.
    assert "x" * 50 in body
    assert "x" * 51 not in body
    assert result.context_used is True


@pytest.mark.asyncio
async def test_optimize_empty_context_does_not_set_context_used():
    state = _state()
    result = await optimize_query(
        query="some query",
        context="",
        app_state=state,
    )
    assert result.context_used is False


@pytest.mark.asyncio
async def test_optimize_forwards_sink_to_llm_complete():
    """The telemetry sink must be plumbed into the LLM call so token/cost
    events emitted by the client land in the response metadata collector."""
    state = _state()
    sink: list = []
    await optimize_query(
        query="heart attack biomarkers",
        context=None,
        app_state=state,
        sink=sink,
    )
    state.llm_client.complete.assert_called_once()
    assert state.llm_client.complete.call_args.kwargs.get("sink") is sink


def test_prompt_instructs_preserving_author_names():
    """The rewrite prompt must explicitly protect author/person surnames.

    Author searches (e.g. "Libis biosynthetic gene cluster") are a distinct
    intent: the surname is search-critical but is not a recognisable
    scientific term, so the recall-maximising rewrite would otherwise strip
    it. The prompt must name authors/persons in its keep-verbatim guidance.
    """
    lowered = _PROMPT.lower()
    assert "surname" in lowered or "author name" in lowered


def test_prompt_forbids_deleting_user_supplied_tokens():
    """Governing rule: the optimizer is additive/normalising, never subtractive
    of content words the user typed. This root-fixes the whole class of
    'dropped an unrecognised-but-critical token' failures (authors, strain IDs,
    cell lines, instrument names), not just author surnames.
    """
    lowered = _PROMPT.lower()
    assert "never delete" in lowered


@pytest.mark.asyncio
async def test_optimize_sends_author_preservation_guidance_in_prompt():
    state = _state()
    await optimize_query(
        query="Libis biosynthetic gene cluster",
        context=None,
        app_state=state,
    )
    sent = state.llm_client.complete.call_args.kwargs["messages"][0]["content"].lower()
    assert "surname" in sent or "author name" in sent


@pytest.mark.asyncio
async def test_optimize_strips_json_code_fence():
    state = _state()
    state.llm_client.complete = AsyncMock(
        return_value='```json\n{"searched_query": "fenced"}\n```'
    )
    result = await optimize_query(
        query="raw", context=None, app_state=state,
    )
    assert result.searched_query == "fenced"
    assert result.applied is True
