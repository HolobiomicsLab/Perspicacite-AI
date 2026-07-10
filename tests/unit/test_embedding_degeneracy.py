"""Regression tests: a failing embedder must never produce a silent zero vector.

A zero vector is not a neutral placeholder. Chroma stores it happily and then
reports cosine distance 1.0 against every document, so `score = 1/(1+distance)`
collapses to a constant 0.5 and retrieval returns arbitrary passages while the
API still reports success. These tests pin the loud behaviour at both ends: the
write path (ingest) and the read path (search).
"""

import pytest

from perspicacite.llm.embeddings import EmbeddingFailedError, is_zero_vector
from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.papers import Paper, PaperSource
from perspicacite.retrieval.chroma_store import ChromaVectorStore

DIMENSION = 8


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    """Build a minimal chunk carrying the given text."""
    return DocumentChunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            paper_id="paper-1",
            chunk_index=0,
            title="Test Paper",
            year=2024,
            source=PaperSource.BIBTEX,
        ),
    )


class _StubProvider:
    """Embedding provider whose behaviour each test dictates."""

    def __init__(self, behaviour):
        self._behaviour = behaviour

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def model_name(self) -> str:
        return "stub-embeddings"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return self._behaviour(texts)

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self._behaviour(texts)


def _store(temp_dir, behaviour) -> ChromaVectorStore:
    """Chroma store backed by a provider with the given behaviour."""
    return ChromaVectorStore(persist_dir=str(temp_dir), embedding_provider=_StubProvider(behaviour))


def _raises_quota_error(_texts):
    raise RuntimeError("OpenAIException - Error code: 429 ... insufficient_quota")


def _returns_zero_vectors(texts):
    return [[0.0] * DIMENSION for _ in texts]


def _returns_unit_vectors(texts):
    return [[1.0] + [0.0] * (DIMENSION - 1) for _ in texts]


def _returns_too_few_vectors(texts):
    """Mimic a provider that silently drops an input, as LiteLLM does for empty text."""
    return [[1.0] + [0.0] * (DIMENSION - 1) for _ in texts[:-1]]


def test_is_zero_vector_distinguishes_degenerate_from_real():
    """The degeneracy check keys on norm, not on individual components."""
    assert is_zero_vector([0.0] * DIMENSION)
    assert not is_zero_vector([0.0] * (DIMENSION - 1) + [1e-6])


@pytest.mark.asyncio
async def test_provider_failure_surfaces_and_writes_nothing(temp_dir):
    """A provider that raises must abort the ingest, not zero-fill it."""
    store = _store(temp_dir, _raises_quota_error)
    with pytest.raises(EmbeddingFailedError, match="insufficient_quota"):
        await store.add_documents("test-kb", [_chunk("chunk-1", "real content")])

    collection = store.client.get_or_create_collection(name="test-kb")
    assert collection.count() == 0, "a failed embed must not persist any chunk"


@pytest.mark.asyncio
async def test_zero_vector_for_nonempty_text_is_rejected(temp_dir):
    """A zero vector for a text with content means the embedder failed."""
    store = _store(temp_dir, _returns_zero_vectors)
    with pytest.raises(EmbeddingFailedError, match="provider returned a zero vector"):
        await store.add_documents("test-kb", [_chunk("chunk-1", "real content")])

    collection = store.client.get_or_create_collection(name="test-kb")
    assert collection.count() == 0


@pytest.mark.asyncio
async def test_zero_vector_for_an_empty_chunk_is_also_rejected(temp_dir):
    """A cache-style provider zero-fills empty text; that must not reach Chroma.

    A chunk with no text and no title composes to an empty embedding text, so a
    provider preserving positional alignment returns a zero vector for it. Stored,
    it would answer every query at cosine distance 1.0.
    """

    def zero_for_empty(texts):
        unit = [1.0] + [0.0] * (DIMENSION - 1)
        return [[0.0] * DIMENSION if not t.strip() else unit for t in texts]

    store = _store(temp_dir, zero_for_empty)
    bare = DocumentChunk(
        id="chunk-empty",
        text="",
        metadata=ChunkMetadata(
            paper_id="paper-1", chunk_index=0, title=None, year=None, source=PaperSource.BIBTEX
        ),
    )
    with pytest.raises(EmbeddingFailedError, match="no text to embed"):
        await store.add_documents("test-kb", [bare])

    collection = store.client.get_or_create_collection(name="test-kb")
    assert collection.count() == 0


@pytest.mark.asyncio
async def test_pre_embedded_zero_vector_is_rejected(temp_dir):
    """capsule_builder, capsule_reader and local_docs embed before calling add_documents.

    Those chunks arrive with .embedding already set, so they never pass through the
    provider branch. Screen them too, or the collection is poisoned by the back door.
    """
    store = _store(temp_dir, _returns_unit_vectors)
    chunk = _chunk("chunk-1", "real content")
    chunk.embedding = [0.0] * DIMENSION

    with pytest.raises(EmbeddingFailedError, match="zero vector"):
        await store.add_documents("test-kb", [chunk])

    collection = store.client.get_or_create_collection(name="test-kb")
    assert collection.count() == 0


@pytest.mark.asyncio
async def test_pre_embedded_real_vector_still_stores(temp_dir):
    """A caller-supplied healthy vector must still be accepted."""
    store = _store(temp_dir, _returns_unit_vectors)
    chunk = _chunk("chunk-1", "real content")
    chunk.embedding = [1.0] + [0.0] * (DIMENSION - 1)

    await store.add_documents("test-kb", [chunk])
    assert store.client.get_collection(name="test-kb").count() == 1


@pytest.mark.asyncio
async def test_misaligned_embedding_count_is_rejected(temp_dir):
    """A short vector list would misassign every embedding after the dropped text."""
    store = _store(temp_dir, _returns_too_few_vectors)
    chunks = [_chunk("chunk-1", "first"), _chunk("chunk-2", "second")]
    with pytest.raises(EmbeddingFailedError, match="misaligned"):
        await store.add_documents("test-kb", chunks)

    collection = store.client.get_or_create_collection(name="test-kb")
    assert collection.count() == 0


@pytest.mark.asyncio
async def test_zero_norm_query_errors_instead_of_scoring_everything_equally(temp_dir):
    """Chroma answers a zero query with distance 1.0 for every doc -> constant 0.5."""
    store = _store(temp_dir, _returns_unit_vectors)
    await store.add_documents("test-kb", [_chunk("chunk-1", "real content")])

    with pytest.raises(EmbeddingFailedError, match="zero-norm query"):
        await store.search("test-kb", [0.0] * DIMENSION, top_k=5)


@pytest.mark.asyncio
async def test_real_query_still_searches(temp_dir):
    """The guard must not disturb a healthy query."""
    store = _store(temp_dir, _returns_unit_vectors)
    await store.add_documents("test-kb", [_chunk("chunk-1", "real content")])

    results = await store.search("test-kb", [1.0] + [0.0] * (DIMENSION - 1), top_k=5)
    assert len(results) == 1
    assert results[0].chunk.id == "chunk-1"


@pytest.mark.asyncio
async def test_empty_text_input_is_still_graceful():
    """An empty string is legitimately zero — only failures are errors."""
    from perspicacite.llm.embeddings import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider()
    result = await provider.embed(["", "   "])
    assert len(result) == 2
    assert all(is_zero_vector(vector) for vector in result)


def _knowledge_base_raising(error: Exception):
    """A DynamicKnowledgeBase whose per-paper ingest raises the given error."""
    from unittest.mock import AsyncMock, MagicMock

    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

    kb = DynamicKnowledgeBase(MagicMock(), MagicMock())
    kb._initialized = True
    kb._add_paper = AsyncMock(side_effect=error)
    return kb


@pytest.mark.asyncio
async def test_add_papers_propagates_embedding_failure():
    """Ingest must not report a successful run of zero chunks when the embedder is down."""
    kb = _knowledge_base_raising(EmbeddingFailedError("quota exhausted"))

    with pytest.raises(EmbeddingFailedError):
        await kb.add_papers([Paper(id="p1", title="T", source=PaperSource.BIBTEX)])


@pytest.mark.asyncio
async def test_add_papers_still_tolerates_ordinary_paper_errors():
    """A single unparseable paper must not abort the batch, as before."""
    kb = _knowledge_base_raising(ValueError("bad pdf"))

    added = await kb.add_papers([Paper(id="p1", title="T", source=PaperSource.BIBTEX)])
    assert added == 0
