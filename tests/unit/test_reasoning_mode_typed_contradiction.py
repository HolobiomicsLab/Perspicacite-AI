"""Unit tests for Phase 3 (typed_contradiction) strategy."""

import json

import pytest

from perspicacite.config.schema import Config
from perspicacite.models.rag import RAGMode, RAGRequest

# The reasoning modes reach perspicacite.indicium_layer.queries, which imports
# the private, maintainer-only indicium package at module scope.
pytest.importorskip("indicium", reason="indicium is a private, maintainer-only package")


class _FakeLLM:
    async def complete(self, *, messages, stage=None, **kw):
        if (stage or "").endswith("typed_contradiction.brief"):
            return json.dumps(
                {
                    "consensus": [
                        {
                            "summary": "X inhibits Y",
                            "claim_iris": ["__C0__"],
                            "papers": ["10.1/p1", "10.1/p2"],
                            "eco": "data",
                        }
                    ],
                    "dispute": [],
                    "open": [],
                }
            )
        if (stage or "").startswith("cito_classifier"):
            return json.dumps([{"pair_id": 0, "label": "supports", "confidence": 0.85}])
        # claim extraction (one claim per chunk)
        return json.dumps(
            {
                "claims": [
                    {
                        "context": "in vitro",
                        "subject": "X",
                        "qualifier": "inhibits",
                        "relation": "binds_to",
                        "object": "Y",
                        "evidence_type": "data",
                        "source_type": "text",
                        "source_doi": "10.1/p1",
                        "quote": "X binds Y",
                    }
                ]
            }
        )


class _FakeDKB:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kb_name = "kb"
        self.collection_name = "kb"
        self._initialized = True

    async def search(self, query, top_k=10, **kw):
        return self._chunks


async def _drain(stream):
    return [ev async for ev in stream]


async def test_typed_contradiction_emits_three_buckets(monkeypatch):
    from perspicacite.rag.modes.reasoning import typed_contradiction as tc

    chunks = [
        {
            "text": "X binds Y.",
            "score": 0.9,
            "metadata": {"paper_id": "10.1/p1", "doi": "10.1/p1", "title": "P1", "year": 2024},
            "chunk_idx": 0,
        },
        {
            "text": "X also binds Y.",
            "score": 0.88,
            "metadata": {"paper_id": "10.1/p2", "doi": "10.1/p2", "title": "P2", "year": 2024},
            "chunk_idx": 0,
        },
        {
            "text": "X has no effect on Y.",
            "score": 0.85,
            "metadata": {"paper_id": "10.1/p3", "doi": "10.1/p3", "title": "P3", "year": 2024},
            "chunk_idx": 0,
        },
    ]

    def _fake_build_retriever(self, request, vs, ep):
        return _FakeDKB(chunks)

    monkeypatch.setattr(
        "perspicacite.rag.modes.base.BaseRAGMode._build_kb_retriever",
        _fake_build_retriever,
    )

    req = RAGRequest(
        query="Does X inhibit Y?",
        mode=RAGMode.REASONING,
        reasoning_strategy="contradiction",
        kb_name="kb",
    )
    events = await _drain(
        tc.run_typed_contradiction_stream(
            request=req,
            llm=_FakeLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    content = "".join(json.loads(e.data).get("delta", "") for e in events if e.event == "content")
    assert "Consensus" in content
    assert "Dispute" in content
    assert "Open" in content


async def test_typed_contradiction_emits_typed_flag_and_edges(monkeypatch):
    from perspicacite.rag.modes.reasoning import typed_contradiction as tc

    chunks = [
        {
            "text": "X binds Y.",
            "score": 0.9,
            "metadata": {"paper_id": "10.1/p1", "doi": "10.1/p1", "title": "P1", "year": 2024},
            "chunk_idx": 0,
        }
    ]

    def _fake_build_retriever(self, request, vs, ep):
        return _FakeDKB(chunks)

    monkeypatch.setattr(
        "perspicacite.rag.modes.base.BaseRAGMode._build_kb_retriever",
        _fake_build_retriever,
    )

    req = RAGRequest(
        query="q", mode=RAGMode.REASONING, reasoning_strategy="contradiction", kb_name="kb"
    )
    events = await _drain(
        tc.run_typed_contradiction_stream(
            request=req,
            llm=_FakeLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    brief_status = [e for e in events if e.event == "status" and "brief" in (e.data or "").lower()]
    assert brief_status
    payload = json.loads(brief_status[-1].data)
    assert payload.get("typed") is True
    assert "cito_edges" in payload  # may be empty when only one paper
