"""Tests for Chroma vector store."""

from pathlib import Path

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


class TestHnswFdBudget:
    """Tests for the bounded HNSW segment cache."""

    @pytest.fixture(autouse=True)
    def _isolate_budget_registry(self):
        """Keep each test's budget claims out of the other tests."""
        from perspicacite.retrieval import chroma_store

        claimed = set(chroma_store._budgeted_persist_dirs)
        yield
        chroma_store._budgeted_persist_dirs.clear()
        chroma_store._budgeted_persist_dirs.update(claimed)

    def test_budget_derives_from_the_constants(self, monkeypatch):
        """The index budget is computed, not a hardcoded number."""
        from perspicacite.retrieval import chroma_store

        monkeypatch.setattr(chroma_store, "CHROMA_HNSW_FD_BUDGET", 100)
        monkeypatch.setattr(chroma_store, "CHROMA_FDS_PER_HNSW_INDEX", 5)

        assert chroma_store._hnsw_index_budget() == 20

    def test_budget_is_far_below_the_ambient_file_limit(self):
        """The whole point: chroma must not size its cache off ulimit -n."""
        import resource

        from perspicacite.retrieval import chroma_store

        ambient_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]

        assert chroma_store._hnsw_index_budget() < ambient_soft

    def test_soft_limit_is_lowered_during_construction(self, monkeypatch, tmp_path):
        """Chroma reads RLIMIT_NOFILE while it builds, so it must be low then."""
        import resource

        from perspicacite.retrieval import chroma_store

        seen = []

        def _record(path):
            seen.append(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
            return object()

        monkeypatch.setattr(chroma_store.chromadb, "PersistentClient", _record)
        ambient_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        chroma_store._bounded_chroma_client(str(tmp_path))

        assert seen == [chroma_store._bounded_soft_limit(ambient_soft)]
        assert seen[0] < ambient_soft

    def test_soft_limit_is_restored_when_construction_raises(self, monkeypatch, tmp_path):
        """A failed client must not leave the process on a lowered limit."""
        import resource

        from perspicacite.retrieval import chroma_store

        before = resource.getrlimit(resource.RLIMIT_NOFILE)

        def _boom(path):
            raise RuntimeError("chroma refused to start")

        monkeypatch.setattr(chroma_store.chromadb, "PersistentClient", _boom)
        with pytest.raises(RuntimeError):
            chroma_store._bounded_chroma_client(str(tmp_path))

        assert resource.getrlimit(resource.RLIMIT_NOFILE) == before

    def test_second_store_on_same_path_does_not_relower(self, monkeypatch, tmp_path):
        """Negative side: a repeat build cannot rebuild chroma's cached System."""
        import resource

        from perspicacite.retrieval import chroma_store

        ambient_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        seen = []

        def _record(path):
            seen.append(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
            return object()

        monkeypatch.setattr(chroma_store.chromadb, "PersistentClient", _record)
        chroma_store._bounded_chroma_client(str(tmp_path))
        chroma_store._bounded_chroma_client(str(tmp_path))

        assert seen == [chroma_store._bounded_soft_limit(ambient_soft), ambient_soft]

    def test_a_different_path_is_still_budgeted(self, monkeypatch, tmp_path):
        """The claim is per directory, not a global one-shot latch."""
        import resource

        from perspicacite.retrieval import chroma_store

        seen = []

        def _record(path):
            seen.append(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
            return object()

        monkeypatch.setattr(chroma_store.chromadb, "PersistentClient", _record)
        ambient_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        chroma_store._bounded_chroma_client(str(tmp_path / "one"))
        chroma_store._bounded_chroma_client(str(tmp_path / "two"))

        assert seen == [chroma_store._bounded_soft_limit(ambient_soft)] * 2

    def test_relative_and_absolute_paths_normalise_together(
        self, monkeypatch, tmp_path, mock_embedding_provider
    ):
        """Two spellings of one directory must not build two HNSW caches."""
        from perspicacite.retrieval.chroma_store import ChromaVectorStore

        (tmp_path / "kb").mkdir()
        absolute = ChromaVectorStore(
            persist_dir=str(tmp_path / "kb"),
            embedding_provider=mock_embedding_provider,
        )
        monkeypatch.chdir(tmp_path)
        relative = ChromaVectorStore(
            persist_dir="kb",
            embedding_provider=mock_embedding_provider,
        )

        assert relative.persist_dir == absolute.persist_dir

    def test_persist_dir_is_absolute(self, tmp_path, mock_embedding_provider):
        """Callers and chroma both key on the resolved path."""
        from perspicacite.retrieval.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "kb"),
            embedding_provider=mock_embedding_provider,
        )

        assert Path(store.persist_dir).is_absolute()


class TestOpenFdCount:
    """Tests for the descriptor counter."""

    def test_counts_descriptors_on_this_platform(self):
        """macOS exposes /dev/fd, Linux /proc/self/fd."""
        from perspicacite.retrieval.chroma_store import open_fd_count

        assert open_fd_count() > 0

    def test_reports_unavailable_rather_than_zero(self, monkeypatch):
        """Unknown must be distinguishable from 'checked and found none'."""
        from perspicacite.retrieval import chroma_store

        monkeypatch.setattr(chroma_store, "FD_DIRS", ("/no/such/fd/dir",))

        assert chroma_store.open_fd_count() == chroma_store.FD_COUNT_UNAVAILABLE


class TestBoundedSoftLimit:
    """Tests for choosing the construction-window descriptor limit."""

    def test_uses_the_budget_when_few_descriptors_are_open(self, monkeypatch):
        """The common case: the budget alone decides the cache size."""
        from perspicacite.retrieval import chroma_store

        monkeypatch.setattr(chroma_store, "open_fd_count", lambda: 10)

        assert chroma_store._bounded_soft_limit(1_000_000) == (
            chroma_store.CHROMA_HNSW_FD_BUDGET
        )

    def test_never_starves_a_process_already_holding_descriptors(self, monkeypatch):
        """Below open FDs + headroom chroma's rust runtime aborts with EMFILE."""
        from perspicacite.retrieval import chroma_store

        monkeypatch.setattr(chroma_store, "open_fd_count", lambda: 9_000)

        limit = chroma_store._bounded_soft_limit(1_000_000)

        assert limit == 9_000 + chroma_store.CHROMA_FD_HEADROOM

    def test_never_raises_the_operators_limit(self, monkeypatch):
        """A deliberately low ulimit must be respected, not overridden."""
        from perspicacite.retrieval import chroma_store

        monkeypatch.setattr(chroma_store, "open_fd_count", lambda: 10)

        assert chroma_store._bounded_soft_limit(64) == 64

    def test_unknown_descriptor_count_still_keeps_a_floor(self, monkeypatch):
        """An unavailable count must not be mistaken for zero open."""
        from perspicacite.retrieval import chroma_store

        monkeypatch.setattr(
            chroma_store, "open_fd_count", lambda: chroma_store.FD_COUNT_UNAVAILABLE
        )

        assert chroma_store._bounded_soft_limit(1_000_000) >= (
            chroma_store.CHROMA_MIN_SOFT_FD_LIMIT
        )
