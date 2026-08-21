"""Unit tests for Phase 4 (provenance) strategy."""

import json

import pytest

from perspicacite.config.schema import Config
from perspicacite.models.rag import RAGMode, RAGRequest

# The reasoning modes reach perspicacite.indicium_layer.queries, which imports
# the private, maintainer-only indicium package at module scope.
pytest.importorskip("indicium", reason="indicium is a private, maintainer-only package")


class _FakeLLM:
    def __init__(self):
        self._calls = 0

    async def complete(self, *, messages, stage=None, **kw):
        self._calls += 1
        if (stage or "").endswith("provenance.compose"):
            return json.dumps(
                {
                    "narrative": [
                        {
                            "sentence": "Compound X inhibits enzyme Y in vitro.",
                            "supports": ["__CLAIM_0__"],
                        }
                    ],
                    "claims_used": ["__CLAIM_0__"],
                }
            )
        # claim extraction
        return json.dumps(
            {
                "claims": [
                    {
                        "context": "in vitro",
                        "subject": "compound X",
                        "qualifier": "inhibits",
                        "relation": "binds_to",
                        "object": "enzyme Y",
                        "evidence_type": "data",
                        "source_type": "text",
                        "source_doi": "10.1/p",
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


async def test_provenance_emits_bound_narrative(monkeypatch):
    from perspicacite.rag.modes.reasoning import provenance as prov

    chunks = [
        {
            "text": "X binds Y in vitro.",
            "score": 0.9,
            "metadata": {"paper_id": "10.1/p", "doi": "10.1/p", "title": "Paper", "year": 2024},
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
        reasoning_strategy="provenance",
        kb_name="kb",
    )
    events = await _drain(
        prov.run_provenance_stream(
            request=req,
            llm=_FakeLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    content = "".join(json.loads(e.data).get("delta", "") for e in events if e.event == "content")
    assert "Compound X inhibits enzyme Y" in content
    assert "[1]" in content  # footnote rendering

    sources = [e for e in events if e.event == "source"]
    assert sources  # paper surfaced

    done = next(e for e in events if e.event == "done")
    payload = json.loads(done.data)
    assert payload["mode"] == "reasoning"


async def test_provenance_emits_sidecar_jsonld(monkeypatch):
    from perspicacite.rag.modes.reasoning import provenance as prov

    chunks = [
        {
            "text": "X binds Y in vitro.",
            "score": 0.9,
            "metadata": {"paper_id": "10.1/p", "doi": "10.1/p", "title": "P", "year": 2024},
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
        query="q", mode=RAGMode.REASONING, reasoning_strategy="provenance", kb_name="kb"
    )
    events = await _drain(
        prov.run_provenance_stream(
            request=req,
            llm=_FakeLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    sidecar_status = [
        e for e in events if e.event == "status" and "sidecar" in (e.data or "").lower()
    ]
    assert sidecar_status
    payload = json.loads(sidecar_status[0].data)
    assert "claims" in payload  # status payload carries the JSON-LD claims
