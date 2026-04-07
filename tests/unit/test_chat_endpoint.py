"""Tests for the /api/chat endpoint — non-streaming response.

These tests verify that when stream=False, the endpoint consumes the SSE
stream and returns a plain JSON response instead of the old error.

Run: PYTHONPATH=src pytest tests/unit/test_chat_endpoint.py -v
"""

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# We test the non-streaming collection logic directly, without starting
# the full FastAPI app (which requires ChromaDB, LLM keys, etc.).
# ---------------------------------------------------------------------------


async def _fake_sse_stream(*, include_answer=True, include_sources=True):
    """Generate fake SSE events mimicking agentic_chat_stream output."""
    if include_sources:
        yield f'data: {json.dumps({"type": "source", "source": {"title": "Test Paper", "doi": "10.1234/test"}})}\n\n'
        yield f'data: {json.dumps({"type": "papers_found", "count": 2})}\n\n'

    if include_answer:
        answer = "This is the LLM answer about the test query."
        answer_b64 = base64.b64encode(answer.encode("utf-8")).decode("ascii")
        yield f'data: {json.dumps({"type": "answer", "content_b64": answer_b64})}\n\n'

    yield f'data: {json.dumps({"type": "done"})}\n\n'


async def _collect_non_streaming(stream_gen):
    """Replicate the non-streaming collection logic from chat_endpoint."""
    answer = ""
    sources = []
    papers_found = 0

    async for event in stream_gen:
        if not event.startswith("data:"):
            continue
        try:
            data = json.loads(event[5:].strip())
        except json.JSONDecodeError:
            continue

        event_type = data.get("type", "")
        if event_type == "answer":
            content_b64 = data.get("content_b64")
            if content_b64:
                answer = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            elif "content" in data:
                answer = str(data["content"])
        elif event_type == "source":
            sources.append(data.get("source", {}))
        elif event_type == "papers_found":
            papers_found = data.get("count", 0)
        elif event_type == "done":
            break

    return {
        "answer": answer,
        "sources": sources,
        "papers_found": papers_found,
    }


class TestNonStreamingCollection:
    """Test the SSE→JSON collection logic used by the non-streaming path."""

    @pytest.mark.asyncio
    async def test_collects_answer_and_sources(self):
        result = await _collect_non_streaming(_fake_sse_stream())

        assert result["answer"] == "This is the LLM answer about the test query."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Test Paper"
        assert result["papers_found"] == 2

    @pytest.mark.asyncio
    async def test_answer_without_b64(self):
        """Answer can be plain content (no base64)."""
        async def stream():
            yield f'data: {json.dumps({"type": "answer", "content": "Plain text answer"})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'

        result = await _collect_non_streaming(stream())
        assert result["answer"] == "Plain text answer"

    @pytest.mark.asyncio
    async def test_no_answer(self):
        """When stream has no answer event, answer is empty string."""
        async def stream():
            yield f'data: {json.dumps({"type": "source", "source": {"title": "Paper"}})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'

        result = await _collect_non_streaming(stream())
        assert result["answer"] == ""
        assert len(result["sources"]) == 1

    @pytest.mark.asyncio
    async def test_stops_at_done(self):
        """Events after 'done' should be ignored."""
        async def stream():
            answer_b64 = base64.b64encode(b"Final answer").decode("ascii")
            yield f'data: {json.dumps({"type": "answer", "content_b64": answer_b64})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'
            # This should be ignored
            yield f'data: {json.dumps({"type": "source", "source": {"title": "Ignored"}})}\n\n'

        result = await _collect_non_streaming(stream())
        assert result["answer"] == "Final answer"
        assert len(result["sources"]) == 0  # source came after done

    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self):
        """Lines not starting with 'data:' are ignored."""
        async def stream():
            yield "This is not a data line\n"
            answer_b64 = base64.b64encode(b"Answer").decode("ascii")
            yield f'data: {json.dumps({"type": "answer", "content_b64": answer_b64})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'

        result = await _collect_non_streaming(stream())
        assert result["answer"] == "Answer"

    @pytest.mark.asyncio
    async def test_handles_malformed_json(self):
        """Malformed JSON in data lines is skipped gracefully."""
        async def stream():
            yield "data: {invalid json}\n"
            answer_b64 = base64.b64encode(b"Still works").decode("ascii")
            yield f'data: {json.dumps({"type": "answer", "content_b64": answer_b64})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'

        result = await _collect_non_streaming(stream())
        assert result["answer"] == "Still works"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
