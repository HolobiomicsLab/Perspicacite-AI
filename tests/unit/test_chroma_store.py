"""Tests for Chroma vector store."""

import pytest

from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.papers import PaperSource


class TestChromaVectorStore:
    """Tests for ChromaVectorStore."""

    @pytest.fixture
    async def store(self, temp_dir, mock_embedding_provider):
        """Create test store."""
        from perspicacite.retrieval.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(temp_dir),
            embedding_provider=mock_embedding_provider,
        )
        return store

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            DocumentChunk(
                id="chunk-1",
                text="This is a test document about machine learning.",
                metadata=ChunkMetadata(
                    paper_id="paper-1",
                    chunk_index=0,
                    title="Test Paper",
                    year=2024,
                    source=PaperSource.BIBTEX,
                ),
            ),
            DocumentChunk(
                id="chunk-2",
                text="Deep learning is a subset of machine learning.",
                metadata=ChunkMetadata(
                    paper_id="paper-1",
                    chunk_index=1,
                    title="Test Paper",
                    year=2024,
                    source=PaperSource.BIBTEX,
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_create_collection(self, store):
        """Test creating a collection."""
        await store.create_collection("test_kb", embedding_dim=384)

        collections = await store.list_collections()
        assert "test_kb" in collections

    @pytest.mark.asyncio
    async def test_add_documents(self, store, sample_chunks):
        """Test adding documents."""
        await store.create_collection("test_kb", embedding_dim=384)

        count = await store.add_documents("test_kb", sample_chunks)
        assert count == 2

    @pytest.mark.asyncio
    async def test_add_documents_empty(self, store):
        """Test adding empty document list."""
        await store.create_collection("test_kb", embedding_dim=384)

        count = await store.add_documents("test_kb", [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_search(self, store, sample_chunks):
        """Test search."""
        await store.create_collection("test_kb", embedding_dim=384)
        await store.add_documents("test_kb", sample_chunks)

        # Mock query embedding
        query_embedding = [0.1] * 384

        results = await store.search(
            collection="test_kb",
            query_embedding=query_embedding,
            top_k=2,
        )

        assert len(results) <= 2
        if results:
            assert all(isinstance(r.score, float) for r in results)
            assert all(r.retrieval_method == "vector" for r in results)

    @pytest.mark.asyncio
    async def test_search_collection_not_found(self, store):
        """Test searching non-existent collection."""
        query_embedding = [0.1] * 384

        results = await store.search(
            collection="nonexistent",
            query_embedding=query_embedding,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_delete_collection(self, store):
        """Test deleting collection."""
        await store.create_collection("delete_me", embedding_dim=384)

        collections_before = await store.list_collections()
        assert "delete_me" in collections_before

        await store.delete_collection("delete_me")

        collections_after = await store.list_collections()
        assert "delete_me" not in collections_after

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, store, sample_chunks):
        """Test getting collection stats."""
        await store.create_collection("test_kb", embedding_dim=384)
        await store.add_documents("test_kb", sample_chunks)

        stats = await store.get_collection_stats("test_kb")
        assert stats["name"] == "test_kb"
        assert stats["count"] == 2

    @pytest.mark.asyncio
    async def test_all_collection_stats_agrees_with_per_collection(self, store, sample_chunks):
        """The bulk index read must report what the Chroma API reports."""
        await store.create_collection("bulk_a", embedding_dim=384)
        await store.add_documents("bulk_a", sample_chunks)
        await store.create_collection("bulk_b", embedding_dim=384)

        bulk = await store.all_collection_stats()

        for name in ("bulk_a", "bulk_b"):
            single = await store.get_collection_stats(name)
            assert bulk[name]["count"] == single["count"], name
            assert bulk[name]["unique_papers"] == single["unique_papers"], name

    @pytest.mark.asyncio
    async def test_all_collection_stats_counts_empty_collection_as_zero(self, store):
        """An empty collection must appear with zero counts, not be missing."""
        await store.create_collection("bulk_empty", embedding_dim=384)

        bulk = await store.all_collection_stats()

        assert bulk["bulk_empty"] == {"count": 0, "unique_papers": 0}

    @pytest.mark.asyncio
    async def test_all_collection_stats_returns_empty_when_index_missing(
        self, tmp_path, mock_embedding_provider
    ):
        """A missing index degrades to no stats, so callers fall back to stored counts."""
        from perspicacite.retrieval.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "never_created"),
            embedding_provider=mock_embedding_provider,
        )
        (tmp_path / "never_created" / "chroma.sqlite3").unlink(missing_ok=True)

        assert await store.all_collection_stats() == {}


class TestMetadataConversion:
    """Tests for metadata conversion functions."""

    def test_chunk_to_metadata(self):
        """Test converting chunk metadata to Chroma format."""
        from perspicacite.retrieval.chroma_store import _chunk_to_metadata

        metadata = ChunkMetadata(
            paper_id="paper-1",
            chunk_index=0,
            section="Abstract",
            year=2024,
            source=PaperSource.BIBTEX,
        )

        result = _chunk_to_metadata(metadata)

        assert result["paper_id"] == "paper-1"
        assert result["chunk_index"] == 0
        assert result["section"] == "Abstract"
        assert result["year"] == 2024
        assert result["source"] == "bibtex"
