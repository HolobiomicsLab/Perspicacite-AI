"""Tests for embedding providers."""

import numpy as np
import pytest

from perspicacite.llm.embeddings import (
    FallbackEmbeddingProvider,
    LiteLLMEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)


class TestLiteLLMEmbeddingProvider:
    """Tests for LiteLLM embedding provider."""

    def test_init(self):
        """Test initialization."""
        provider = LiteLLMEmbeddingProvider(model="text-embedding-3-small")
        assert provider.model == "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_dimension_large(self):
        """Test dimension for large model."""
        provider = LiteLLMEmbeddingProvider(model="text-embedding-3-large")
        assert provider.dimension == 3072

    @pytest.mark.asyncio
    async def test_embed_empty(self):
        """Test embedding empty list."""
        provider = LiteLLMEmbeddingProvider()
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_empty_strings(self):
        """Test embedding empty strings."""
        provider = LiteLLMEmbeddingProvider()
        result = await provider.embed(["", "   ", ""])
        assert len(result) == 3
        assert all(len(r) == 1536 for r in result)


class TestSentenceTransformerEmbeddingProvider:
    """Tests for sentence-transformer provider."""

    def test_init(self):
        """Test initialization."""
        provider = SentenceTransformerEmbeddingProvider(model="all-MiniLM-L6-v2")
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.device == "cpu"

    def test_dimension(self):
        """Test dimension."""
        provider = SentenceTransformerEmbeddingProvider()
        assert provider.dimension == 384

    @pytest.mark.asyncio
    async def test_embed(self):
        """Test embedding."""
        provider = SentenceTransformerEmbeddingProvider()

        # Skip if sentence-transformers not installed
        try:
            result = await provider.embed(["Hello world", "Test sentence"])
            assert len(result) == 2
            assert len(result[0]) == 384
            assert len(result[1]) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.asyncio
    async def test_embed_empty(self):
        """Test embedding empty list."""
        provider = SentenceTransformerEmbeddingProvider()
        result = await provider.embed([])
        assert result == []


class TestFallbackEmbeddingProvider:
    """Tests for fallback provider."""

    def test_init(self):
        """Test initialization."""
        primary = LiteLLMEmbeddingProvider()
        fallback = SentenceTransformerEmbeddingProvider()
        provider = FallbackEmbeddingProvider(primary, fallback)

        assert provider.primary == primary
        assert provider.fallback == fallback
        assert provider.dimension == primary.dimension

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, monkeypatch):
        """Test fallback when primary fails."""
        primary = LiteLLMEmbeddingProvider()
        fallback = SentenceTransformerEmbeddingProvider()
        provider = FallbackEmbeddingProvider(primary, fallback)

        # Mock primary to fail
        async def failing_embed(texts):
            raise Exception("API error")

        monkeypatch.setattr(primary, "embed", failing_embed)

        # Skip if sentence-transformers not installed
        try:
            result = await provider.embed(["Hello"])
            assert len(result) == 1
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestCreateEmbeddingProvider:
    """Tests for provider factory."""

    def test_create_openai_model(self):
        """Test creating provider for OpenAI model."""
        provider = create_embedding_provider("text-embedding-3-small")
        assert isinstance(provider, LiteLLMEmbeddingProvider)
        assert provider.model_name == "text-embedding-3-small"

    def test_create_local_model(self):
        """Test creating provider for local model."""
        provider = create_embedding_provider("all-MiniLM-L6-v2", use_local_fallback=False)
        assert isinstance(provider, SentenceTransformerEmbeddingProvider)

    def test_create_with_fallback(self):
        """Test creating provider with fallback."""
        provider = create_embedding_provider("text-embedding-3-small", use_local_fallback=True)
        assert isinstance(provider, FallbackEmbeddingProvider)
