"""Test configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from perspicacite.config.schema import Config


@pytest.fixture
def config() -> Config:
    """Test configuration with defaults."""
    return Config()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    class MockLLM:
        async def complete(self, messages: list[dict], **kwargs) -> str:
            return "This is a mock LLM response."

        async def stream(self, messages: list[dict], **kwargs):
            for word in "This is a mock LLM response.".split():
                yield word + " "

    return MockLLM()


@pytest.fixture
def sample_papers() -> list[dict]:
    """Sample paper data for tests."""
    return [
        {
            "id": "doi:10.1234/test1",
            "title": "Test Paper 1: Introduction to Testing",
            "authors": [{"name": "John Doe", "family": "Doe"}],
            "year": 2024,
            "doi": "10.1234/test1",
            "abstract": "This is a test paper.",
        },
        {
            "id": "doi:10.1234/test2",
            "title": "Test Paper 2: Advanced Testing Methods",
            "authors": [{"name": "Jane Smith", "family": "Smith"}],
            "year": 2023,
            "doi": "10.1234/test2",
            "abstract": "Another test paper.",
        },
    ]


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider that returns random vectors."""
    import numpy as np

    class MockEmbeddingProvider:
        def __init__(self, dimension: int = 384):
            self._dimension = dimension

        @property
        def dimension(self) -> int:
            return self._dimension

        @property
        def model_name(self) -> str:
            return "mock-embeddings"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            """Return random embeddings."""
            return [
                np.random.randn(self._dimension).tolist()
                for _ in texts
            ]

    return MockEmbeddingProvider()
