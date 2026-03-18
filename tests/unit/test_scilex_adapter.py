"""Tests for SciLEx adapter."""

import pytest

from perspicacite.search.scilex_adapter import SciLExAdapter, SciLExSearchProvider


class TestSciLExAdapter:
    """Tests for SciLExAdapter."""

    def test_init(self):
        """Test adapter initialization."""
        adapter = SciLExAdapter()
        assert adapter.api_config == {}

    def test_check_scilex(self):
        """Test SciLEx availability check."""
        adapter = SciLExAdapter()
        # Should return False or True depending on environment
        assert isinstance(adapter._scilex_available, bool)

    @pytest.mark.asyncio
    async def test_search_fallback(self):
        """Test fallback when SciLEx unavailable."""
        adapter = SciLExAdapter()
        # Force unavailable
        adapter._scilex_available = False

        results = await adapter.search("machine learning", max_results=5)
        assert results == []

    def test_build_api_config(self):
        """Test API config building."""
        adapter = SciLExAdapter()
        config = adapter._build_api_config(["semantic_scholar", "ieee"])

        assert "SEMANTIC_SCHOLAR" in config
        assert "IEEE" in config

    def test_map_single_record(self):
        """Test mapping SciLEx record to Paper."""
        adapter = SciLExAdapter()

        # Mock row data (Zotero format)
        row = {
            "title": "Test Paper",
            "author": "Doe, John; Smith, Jane",
            "date": "2024-01-15",
            "DOI": "10.1234/test",
            "publicationTitle": "Test Journal",
            "abstractNote": "This is a test abstract.",
            "url": "https://example.com/paper",
            "citation_count": 42,
            "api_source": "semantic_scholar",
        }

        paper = adapter._map_single_record(row)

        assert paper.title == "Test Paper"
        assert paper.doi == "10.1234/test"
        assert paper.year == 2024
        assert paper.journal == "Test Journal"
        assert len(paper.authors) == 2
        assert paper.citation_count == 42

    def test_map_single_record_minimal(self):
        """Test mapping minimal record."""
        adapter = SciLExAdapter()

        row = {"title": "Minimal Paper"}

        paper = adapter._map_single_record(row)

        assert paper.title == "Minimal Paper"
        assert paper.id.startswith("generated:")


class TestSciLExSearchProvider:
    """Tests for SciLExSearchProvider."""

    def test_init(self):
        """Test provider initialization."""
        provider = SciLExSearchProvider()
        assert provider.adapter is not None

    def test_name(self):
        """Test provider name."""
        provider = SciLExSearchProvider()
        assert provider.name == "scilex"

    def test_description(self):
        """Test provider description."""
        provider = SciLExSearchProvider()
        assert "SciLEx" in provider.description
