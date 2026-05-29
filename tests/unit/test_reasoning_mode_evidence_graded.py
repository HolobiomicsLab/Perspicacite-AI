"""Unit tests for Phase 1 (evidence_graded) strategy."""

import json

import pytest

pytest.importorskip("indicium", reason="indicia extra not installed")

from perspicacite.config.schema import Config
from perspicacite.models.rag import RAGMode, RAGRequest


class _FakeLLM:
    def __init__(self):
        self._calls = 0

    async def complete(self, *, messages, stage=None, **kw):
        self._calls += 1
        if (stage or "").endswith("evidence_graded.compose"):
            return json.dumps(
                {
                    "tiers": [
                        {
                            "eco": "data",
                            "bullets": [{"text": "X binds Y in vitro.", "papers": ["10.1/p1"]}],
                        },
                        {
                            "eco": "citation",
                            "bullets": [
                                {
                                    "text": "Authors of P2 cite this finding.",
                                    "papers": ["10.1/p2"],
                                }
                            ],
                        },
                        {"eco": "inference", "bullets": []},
                    ]
                }
            )
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
                        "quote": "X inhibits Y",
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


async def test_evidence_graded_emits_three_tier_headers(monkeypatch):
    from perspicacite.rag.modes.reasoning import evidence_graded as eg

    chunks = [
        {
            "text": "X binds Y.",
            "score": 0.9,
            "metadata": {
                "paper_id": "10.1/p1",
                "doi": "10.1/p1",
                "title": "P1",
                "year": 2024,
            },
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
        query="What does X do?",
        mode=RAGMode.REASONING,
        reasoning_strategy="evidence_graded",
        kb_name="kb",
    )
    events = await _drain(
        eg.run_evidence_graded_stream(
            request=req,
            llm=_FakeLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    content = "".join(json.loads(e.data).get("delta", "") for e in events if e.event == "content")
    assert "What the data show" in content
    assert "What the literature claims" in content
    assert "What is inferred" in content
    assert "[ECO:0000006" in content  # data tier IRI label
    assert "10.1/p1" in content  # citation
