"""Tests for data models."""

import pytest
from pydantic import ValidationError

from perspicacite.models import (
    Author,
    ChunkConfig,
    DocumentChunk,
    KnowledgeBase,
    Message,
    Paper,
    PaperSource,
    RAGMode,
)


class TestAuthor:
    """Tests for Author model."""

    def test_create_basic(self):
        """Test creating a basic author."""
        author = Author(name="John Doe")
        assert author.name == "John Doe"
        assert author.given is None
        assert author.family is None

    def test_create_full(self):
        """Test creating author with all fields."""
        author = Author(
            name="John Doe",
            given="John",
            family="Doe",
            orcid="0000-0001-2345-6789",
        )
        assert author.name == "John Doe"
        assert author.given == "John"
        assert author.family == "Doe"
        assert author.orcid == "0000-0001-2345-6789"

    def test_str(self):
        """Test string representation."""
        author = Author(name="John Doe")
        assert str(author) == "John Doe"


class TestPaper:
    """Tests for Paper model."""

    def test_create_minimal(self):
        """Test creating minimal paper."""
        paper = Paper(id="test-1", title="Test Paper")
        assert paper.id == "test-1"
        assert paper.title == "Test Paper"
        assert paper.authors == []
        assert paper.source == PaperSource.BIBTEX

    def test_create_full(self):
        """Test creating paper with all fields."""
        paper = Paper(
            id="doi:10.1234/test",
            title="Test Paper",
            authors=[Author(name="John Doe", family="Doe")],
            year=2024,
            doi="10.1234/test",
            journal="Test Journal",
            abstract="An abstract",
        )
        assert paper.first_author == "John Doe"
        assert paper.citation_key == "Doe2024"

    def test_year_validation(self):
        """Test year validation."""
        # Valid year
        Paper(id="t1", title="T", year=2024)

        # Invalid year (too old)
        with pytest.raises(ValidationError):
            Paper(id="t1", title="T", year=1700)

        # Invalid year (future)
        with pytest.raises(ValidationError):
            Paper(id="t1", title="T", year=2100)

    def test_from_bibtex(self):
        """Test creating from BibTeX entry."""
        entry = {
            "title": "A Test Paper",
            "author": "Doe, John and Smith, Jane",
            "year": "2024",
            "journal": "Test Journal",
            "doi": "10.1234/test",
        }
        paper = Paper.from_bibtex(entry)

        assert paper.title == "A Test Paper"
        assert len(paper.authors) == 2
        assert paper.authors[0].family == "Doe"
        assert paper.year == 2024
        assert paper.doi == "10.1234/test"


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_create(self):
        """Test creating document chunk."""
        from perspicacite.models.documents import ChunkMetadata

        chunk = DocumentChunk(
            id="chunk-1",
            text="This is test content.",
            metadata=ChunkMetadata(
                paper_id="paper-1",
                chunk_index=0,
            ),
        )
        assert chunk.id == "chunk-1"
        assert chunk.text == "This is test content."
        assert chunk.metadata.paper_id == "paper-1"


class TestKnowledgeBase:
    """Tests for KnowledgeBase model."""

    def test_create(self):
        """Test creating KB."""
        kb = KnowledgeBase(
            name="test-kb",
            description="Test knowledge base",
            collection_name="test_kb",
        )
        assert kb.name == "test-kb"
        assert kb.paper_count == 0

    def test_name_validation(self):
        """Test KB name validation."""
        # Valid names
        KnowledgeBase(name="test-kb", collection_name="test_kb")
        KnowledgeBase(name="test_kb_123", collection_name="test_kb")

        # Invalid name (spaces)
        with pytest.raises(ValidationError):
            KnowledgeBase(name="test kb", collection_name="test_kb")

        # Invalid name (special chars)
        with pytest.raises(ValidationError):
            KnowledgeBase(name="test@kb", collection_name="test_kb")


class TestChunkConfig:
    """Tests for ChunkConfig model."""

    def test_defaults(self):
        """Test default values."""
        config = ChunkConfig()
        assert config.method == "token"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_validation(self):
        """Test field validation."""
        # Valid
        ChunkConfig(chunk_size=500, chunk_overlap=100)

        # Invalid chunk_size
        with pytest.raises(ValidationError):
            ChunkConfig(chunk_size=50)  # Too small

        with pytest.raises(ValidationError):
            ChunkConfig(chunk_size=20000)  # Too large


class TestMessage:
    """Tests for Message model."""

    def test_create(self):
        """Test creating message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.sources == []

    def test_to_dict(self):
        """Test serialization."""
        msg = Message(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"


class TestRAGMode:
    """Tests for RAGMode enum."""

    def test_modes(self):
        """Test all modes exist."""
        assert RAGMode.QUICK == "quick"
        assert RAGMode.STANDARD == "standard"
        assert RAGMode.ADVANCED == "advanced"
        assert RAGMode.DEEP == "deep"
        assert RAGMode.CITATION == "citation"
