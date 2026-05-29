"""license_id round-trips through ChunkMetadata <-> Chroma metadata,
is stamped by dynamic_kb, and is included in _metadata_from_discovery."""

import pytest

from perspicacite.models.documents import ChunkMetadata
from perspicacite.pipeline.download.base import PaperDiscovery
from perspicacite.pipeline.download.unified import _metadata_from_discovery
from perspicacite.retrieval.chroma_store import (
    _chunk_to_metadata,
    _metadata_to_chunk,
)

# ---------------------------------------------------------------------------
# (a) ChunkMetadata round-trip through Chroma serialisation
# ---------------------------------------------------------------------------

def test_license_id_round_trips_through_chroma_metadata():
    cm = ChunkMetadata(paper_id="10.1/x", chunk_index=0, license_id="cc-by")
    flat = _chunk_to_metadata(cm)
    assert flat["license_id"] == "cc-by"
    back = _metadata_to_chunk(flat)
    assert back.license_id == "cc-by"


def test_license_id_absent_is_omitted_not_none():
    cm = ChunkMetadata(paper_id="10.1/x", chunk_index=0)  # no license_id
    flat = _chunk_to_metadata(cm)
    assert "license_id" not in flat  # Chroma rejects None; absent is correct


# ---------------------------------------------------------------------------
# (b) dynamic_kb stamps license_id from paper.license onto chunk-0 metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dynamic_kb_stamps_license_id_on_metadata_chunk():
    """_add_paper should stamp paper.license as license_id on the metadata chunk."""
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

    # Build a minimal paper with a license
    paper = Paper(
        id="doi:10.1/test",
        title="Test Paper",
        doi="10.1/test",
        abstract="Short abstract.",
        source=PaperSource.OPENALEX,
        license="cc-by",
    )

    # Stub the vector store so we can capture what chunks are added
    captured_chunks: list = []

    class _StubVectorStore:
        async def create_collection(self, name, embedding_dim=None):
            pass

        async def add_documents(self, collection, chunks, **kwargs):
            captured_chunks.extend(chunks)
            return len(chunks)

    class _StubEmbeddingService:
        dimension = 768
        model_name = "stub-embed"

        async def embed(self, texts):
            return [[0.0] * 768 for _ in texts]

    cfg = KnowledgeBaseConfig()
    dkb = DynamicKnowledgeBase(
        vector_store=_StubVectorStore(),
        embedding_service=_StubEmbeddingService(),
        config=cfg,
    )
    dkb.collection_name = "test_collection"
    dkb._initialized = True

    await dkb.add_papers([paper])

    # chunk-0 is the metadata chunk
    assert len(captured_chunks) >= 1
    meta_chunk = next(c for c in captured_chunks if c.metadata.section == "metadata")
    assert meta_chunk.metadata.license_id == "cc-by"


@pytest.mark.asyncio
async def test_dynamic_kb_body_chunks_also_carry_license_id():
    """Body chunks (full text splits) must also carry license_id."""
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

    paper = Paper(
        id="doi:10.1/test2",
        title="Test Paper 2",
        doi="10.1/test2",
        abstract="Short abstract.",
        full_text="Body text " * 300,  # long enough to produce body chunks
        source=PaperSource.OPENALEX,
        license="cc-by-sa",
    )

    captured_chunks: list = []

    class _StubVectorStore:
        async def create_collection(self, name, embedding_dim=None):
            pass

        async def add_documents(self, collection, chunks, **kwargs):
            captured_chunks.extend(chunks)
            return len(chunks)

    class _StubEmbeddingService:
        dimension = 768
        model_name = "stub-embed"

        async def embed(self, texts):
            return [[0.0] * 768 for _ in texts]

    cfg = KnowledgeBaseConfig()
    dkb = DynamicKnowledgeBase(
        vector_store=_StubVectorStore(),
        embedding_service=_StubEmbeddingService(),
        config=cfg,
    )
    dkb.collection_name = "test_collection2"
    dkb._initialized = True

    await dkb.add_papers([paper])

    # All body chunks should carry the license_id
    body_chunks = [c for c in captured_chunks if c.metadata.section != "metadata"]
    assert len(body_chunks) >= 1
    for chunk in body_chunks:
        assert chunk.metadata.license_id == "cc-by-sa"


# ---------------------------------------------------------------------------
# (c) _metadata_from_discovery includes "license" key from PaperDiscovery
# ---------------------------------------------------------------------------

def test_metadata_from_discovery_includes_license():
    disc = PaperDiscovery(doi="10.1/x", license="cc-by")
    result = _metadata_from_discovery(disc, doi="10.1/x")
    assert result["license"] == "cc-by"


def test_metadata_from_discovery_license_none_when_not_set():
    disc = PaperDiscovery(doi="10.1/x")  # no license
    result = _metadata_from_discovery(disc, doi="10.1/x")
    assert result["license"] is None
