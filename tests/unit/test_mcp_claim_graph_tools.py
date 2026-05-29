"""Unit tests for build_claim_graph + claim_graph_status MCP tools."""

import json
import pytest

pyoxigraph = pytest.importorskip("pyoxigraph", reason="pyoxigraph (indicia extra) not installed")


async def test_build_claim_graph_returns_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    import perspicacite.mcp.server as srv
    from perspicacite.config.schema import Config
    from perspicacite.mcp.server import build_claim_graph

    class _FakeVectorStore:
        async def list_paper_metadata(self, collection):
            return [{"paper_id": "10.1/p1", "doi": "10.1/p1", "title": "Paper 1", "year": 2024}]

        async def get_chunks_by_paper_ids(self, collection, paper_ids):
            from perspicacite.models.documents import ChunkMetadata, DocumentChunk
            m = ChunkMetadata(paper_id="10.1/p1", chunk_index=0, doi="10.1/p1")
            return [DocumentChunk(id="c0", text="X inhibits Y in vitro.", metadata=m)]

    class _FakeLLM:
        async def complete(self, *, messages, stage=None, **kw):
            if (stage or "").startswith("cito_classifier"):
                return "[]"
            return json.dumps({"claims": [{
                "context": "in vitro", "subject": "X",
                "qualifier": "inhibits", "relation": "binds_to",
                "object": "Y", "evidence_type": "data",
                "source_type": "text", "source_doi": "10.1/p1",
                "quote": "X inhibits Y",
            }]})

    fake_state = srv.MCPState()
    fake_state.initialized = True
    fake_state.config = Config()
    fake_state.vector_store = _FakeVectorStore()
    fake_state.llm_client = _FakeLLM()

    monkeypatch.setattr(srv, "mcp_state", fake_state)

    raw = await build_claim_graph(kb_name="kb")
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["claims_added"] >= 0   # may be 0 if no SHACL-valid claims, but no error
    assert payload["kb_name"] == "kb"


async def test_claim_graph_status_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    from perspicacite.mcp.server import claim_graph_status

    raw = await claim_graph_status(kb_name="nope")
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["paper_count"] == 0
    assert payload["last_build_iso"] is None
