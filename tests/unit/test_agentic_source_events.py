# tests/unit/test_agentic_source_events.py
"""Item 10 regression: _stream_agentic must emit individual `source` SSE events
for each paper in papers_found, so eval clients using the same SSE contract as
other RAG modes can collect sources without special-casing agentic mode.

The orchestrator emits events in this order:
  answer → papers_found (orchestrator order)

_stream_agentic buffers `answer` until `papers_found` arrives, then:
  source (paper 1) → source (paper 2) → ... → answer (with sources[]) → papers_found

See NEXT_STEPS_2026_05_25 Item 10 and PERSPICACITE_PATCHES.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from perspicacite.memory.session_store import SessionStore
from perspicacite.provenance.store import ProvenanceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAPER_A = {
    "title": "Paper Alpha",
    "authors": ["Smith, J.", "Jones, K."],
    "year": 2023,
    "doi": "https://doi.org/10.1000/alpha",
    "url": "https://example.com/alpha",
    "relevance_score": 0.91,
    "kb_name": "scifact_abstracts",
}
PAPER_B = {
    "title": "Paper Beta",
    "authors": "Brown, T.",  # string form (should pass through unchanged)
    "year": 2022,
    "doi": "10.1000/beta",  # no prefix — should pass through as-is
    "url": None,
    "relevance_score": 0.77,
    "kb_name": None,
}
# KB paper: has paper_id ("scifact:N") but no DOI → eval must use paper_id for matching.
PAPER_KB = {
    "title": "SciFact Paper",
    "authors": ["Author, A."],
    "year": 2021,
    "doi": None,
    "url": None,
    "relevance_score": 0.88,
    "kb_name": "scifact_abstracts",
    "paper_id": "scifact:4983",
}


def _build_fake_chat(events):
    """Return an async generator function that yields the given events."""
    async def fake_chat(query, session_id, kb_name, stream, **kw):
        for e in events:
            yield e

    return fake_chat


def _parse_frames(chunks: list[str]) -> list[dict]:
    """Parse a list of SSE `data: {...}\n\n` strings into dicts."""
    frames = []
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload:
                    frames.append(json.loads(payload))
    return frames


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_state(tmp_path: Path, monkeypatch):
    """Monkeypatch app_state with minimal stubs and return (monkeypatch, chat_router)."""
    from perspicacite.web import state as state_mod
    from perspicacite.web.routers import chat as chat_router

    ss = SessionStore(tmp_path / "p.db")
    ps = ProvenanceStore(db_path=tmp_path / "p.db", sidecar_dir=tmp_path / "provenance")
    monkeypatch.setattr(state_mod.app_state, "session_store", ss, raising=False)
    monkeypatch.setattr(state_mod.app_state, "provenance_store", ps, raising=False)
    return monkeypatch, state_mod, chat_router


def _make_req(kb_name: str = "default"):
    req = MagicMock()
    req.query = "test query"
    req.session_id = "sess-1"
    req.kb_name = kb_name
    req.kb_names = None
    req.max_papers_to_download = 0
    req.databases = None
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_papers_found_emits_individual_source_events(patched_state):
    """When orchestrator yields answer then papers_found, _stream_agentic must
    emit individual `type=source` SSE frames — one per paper — BEFORE the
    `type=answer` frame."""
    monkeypatch, state_mod, chat_router = patched_state

    orchestrator = MagicMock()
    orchestrator.chat = _build_fake_chat([
        {"type": "answer", "session_id": "sess-1", "content": "Here is what I found."},
        {"type": "papers_found", "papers": [PAPER_A, PAPER_B]},
    ])
    monkeypatch.setattr(state_mod.app_state, "orchestrator", orchestrator, raising=False)

    chunks: list[str] = []
    async for ev in chat_router._stream_agentic(_make_req(), conversation_id="conv-1"):
        chunks.append(ev)

    frames = _parse_frames(chunks)
    types = [f["type"] for f in frames]

    # Verify we got source events at all
    source_frames = [f for f in frames if f["type"] == "source"]
    assert len(source_frames) == 2, f"Expected 2 source events, got {len(source_frames)}; types={types}"

    # Source events must appear BEFORE the answer event
    first_source_idx = types.index("source")
    answer_idx = types.index("answer")
    assert first_source_idx < answer_idx, (
        f"source event at index {first_source_idx} must precede answer at {answer_idx}"
    )

    # Verify source event content for PAPER_A (list authors, doi prefix stripped)
    src_a = source_frames[0]["source"]
    assert src_a["title"] == "Paper Alpha"
    assert src_a["doi"] == "10.1000/alpha"      # prefix stripped
    assert src_a["authors"] == "Smith, J., Jones, K."  # list joined
    assert src_a["relevance_score"] == pytest.approx(0.91)

    # Verify source event content for PAPER_B (string authors, doi unchanged)
    src_b = source_frames[1]["source"]
    assert src_b["title"] == "Paper Beta"
    assert src_b["doi"] == "10.1000/beta"       # no prefix to strip
    assert src_b["authors"] == "Brown, T."

    # The answer event's `sources` array must contain both papers
    answer_frame = next(f for f in frames if f["type"] == "answer")
    assert len(answer_frame["sources"]) == 2

    # A `done` frame must close the stream
    assert any(f["type"] == "done" for f in frames)


@pytest.mark.asyncio
async def test_kb_paper_id_forwarded_in_source_event(patched_state):
    """B-10: paper_id (e.g. 'scifact:4983') must be included in the source event
    so eval clients can match KB papers that have no DOI via the scifact: prefix.

    Without this fix, agentic easy-bucket claims are false misses: the KB paper
    is retrieved but the eval can't identify it because paper_id is absent from
    the source SSE event.
    """
    monkeypatch, state_mod, chat_router = patched_state

    orchestrator = MagicMock()
    orchestrator.chat = _build_fake_chat([
        {"type": "answer", "session_id": "sess-1", "content": "Found in KB."},
        {"type": "papers_found", "papers": [PAPER_KB]},
    ])
    monkeypatch.setattr(state_mod.app_state, "orchestrator", orchestrator, raising=False)

    chunks: list[str] = []
    async for ev in chat_router._stream_agentic(_make_req(), conversation_id="conv-b10"):
        chunks.append(ev)

    frames = _parse_frames(chunks)
    source_frames = [f for f in frames if f["type"] == "source"]

    assert len(source_frames) == 1, f"Expected 1 source event; got {len(source_frames)}"
    src = source_frames[0]["source"]

    # B-10: paper_id must survive the pipeline
    assert src.get("paper_id") == "scifact:4983", (
        "B-10: paper_id must be forwarded in the source event so eval clients "
        f"can match KB papers without a DOI. Got: {src!r}"
    )


@pytest.mark.asyncio
async def test_no_papers_found_still_emits_answer(patched_state):
    """If orchestrator never emits papers_found, the buffered answer must still
    be flushed at end of stream (with empty sources)."""
    monkeypatch, state_mod, chat_router = patched_state

    orchestrator = MagicMock()
    orchestrator.chat = _build_fake_chat([
        {"type": "answer", "session_id": "sess-1", "content": "Nothing found."},
        # no papers_found
    ])
    monkeypatch.setattr(state_mod.app_state, "orchestrator", orchestrator, raising=False)

    chunks: list[str] = []
    async for ev in chat_router._stream_agentic(_make_req(), conversation_id="conv-2"):
        chunks.append(ev)

    frames = _parse_frames(chunks)
    types = [f["type"] for f in frames]

    # No source events
    assert not any(f["type"] == "source" for f in frames), f"Unexpected source frames; types={types}"

    # Answer still emitted with empty sources
    answer_frame = next((f for f in frames if f["type"] == "answer"), None)
    assert answer_frame is not None, "answer frame missing"
    assert answer_frame["sources"] == []

    # Done frame present
    assert "done" in types


@pytest.mark.asyncio
async def test_duplicate_papers_in_papers_found_are_deduplicated(patched_state):
    """Papers with the same doi must not generate duplicate source events."""
    monkeypatch, state_mod, chat_router = patched_state

    orchestrator = MagicMock()
    orchestrator.chat = _build_fake_chat([
        {"type": "answer", "session_id": "sess-1", "content": "Found."},
        {
            "type": "papers_found",
            "papers": [PAPER_A, PAPER_A],  # duplicate
        },
    ])
    monkeypatch.setattr(state_mod.app_state, "orchestrator", orchestrator, raising=False)

    chunks: list[str] = []
    async for ev in chat_router._stream_agentic(_make_req(), conversation_id="conv-3"):
        chunks.append(ev)

    frames = _parse_frames(chunks)
    source_frames = [f for f in frames if f["type"] == "source"]

    # Deduplication must collapse duplicates → only 1 unique paper
    assert len(source_frames) == 1, (
        f"Duplicate papers should be deduplicated; got {len(source_frames)} source frames"
    )


@pytest.mark.asyncio
async def test_papers_found_before_answer_emits_sources_then_answer(patched_state):
    """If orchestrator yields papers_found BEFORE answer (unusual but possible),
    sources are emitted and answer is not buffered — it arrives next and is emitted
    directly after sources."""
    monkeypatch, state_mod, chat_router = patched_state

    orchestrator = MagicMock()
    orchestrator.chat = _build_fake_chat([
        {"type": "papers_found", "papers": [PAPER_A]},
        {"type": "answer", "session_id": "sess-1", "content": "Early sources."},
    ])
    monkeypatch.setattr(state_mod.app_state, "orchestrator", orchestrator, raising=False)

    chunks: list[str] = []
    async for ev in chat_router._stream_agentic(_make_req(), conversation_id="conv-4"):
        chunks.append(ev)

    frames = _parse_frames(chunks)
    types = [f["type"] for f in frames]

    source_frames = [f for f in frames if f["type"] == "source"]
    # papers_found_received=True prevents the end-of-stream flush from
    # re-emitting source events, so each paper appears exactly once.
    assert len(source_frames) == 1, (
        f"Expected 1 source event (no duplicate from end-flush); got {len(source_frames)}; "
        f"types={types}"
    )

    # Ordering: source(from papers_found loop) → papers_found → answer(end-flush) → done
    first_source_idx = types.index("source")
    answer_idx = types.index("answer")
    assert first_source_idx < answer_idx, (
        f"source at {first_source_idx} must precede answer at {answer_idx}"
    )
    # The answer frame still has sources populated from accumulated_papers
    answer_frame = next(f for f in frames if f["type"] == "answer")
    assert len(answer_frame["sources"]) == 1
