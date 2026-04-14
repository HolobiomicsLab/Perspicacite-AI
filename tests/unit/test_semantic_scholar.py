"""Unit tests for Semantic Scholar paper lookup.

Tests:
- normalize_paper_id(): ID normalization for the S2 API
- lookup_paper(): paper retrieval with mocked httpx
- StepType.PAPER_LOOKUP: enum and coercion
- Orchestrator PAPER_LOOKUP handler

Run: PYTHONPATH=src pytest tests/unit/test_semantic_scholar.py -v
"""

import pytest

from perspicacite.search.semantic_scholar import normalize_paper_id, lookup_paper
from perspicacite.rag.agentic.planner import StepType, coerce_step_type


# ---------------------------------------------------------------------------
# normalize_paper_id
# ---------------------------------------------------------------------------


class TestNormalizePaperId:
    """Tests for paper ID normalization."""

    def test_doi_gets_prefix(self):
        assert normalize_paper_id("10.1038/s41586-023-12345") == "DOI:10.1038/s41586-023-12345"

    def test_doi_with_existing_prefix_unchanged(self):
        assert normalize_paper_id("DOI:10.1038/s41586-023-12345") == "DOI:10.1038/s41586-023-12345"

    def test_arxiv_modern_id(self):
        assert normalize_paper_id("2604.06788") == "ArXiv:2604.06788"

    def test_arxiv_with_version(self):
        assert normalize_paper_id("2604.06788v1") == "ArXiv:2604.06788v1"

    def test_arxiv_with_existing_prefix_unchanged(self):
        assert normalize_paper_id("ArXiv:2604.06788") == "ArXiv:2604.06788"

    def test_old_arxiv_id(self):
        assert normalize_paper_id("cs/0701001") == "ArXiv:cs/0701001"

    def test_pmid_with_prefix(self):
        assert normalize_paper_id("PMID:12345678") == "PMID:12345678"

    def test_corpus_id_with_prefix(self):
        assert normalize_paper_id("CorpusId:12345") == "CorpusId:12345"

    def test_url_passthrough(self):
        assert normalize_paper_id("https://arxiv.org/abs/2604.06788") == "https://arxiv.org/abs/2604.06788"

    def test_http_url_passthrough(self):
        assert normalize_paper_id("http://example.com/paper") == "http://example.com/paper"

    def test_s2_hex_id_passthrough(self):
        """40-char hex S2 paper ID passes through unchanged."""
        assert normalize_paper_id("649def34f2be55c941d709a2e3c1f3f0f3f3f3f3") == "649def34f2be55c941d709a2e3c1f3f0f3f3f3f3"

    def test_empty_string(self):
        assert normalize_paper_id("") == ""

    def test_whitespace_trimmed(self):
        assert normalize_paper_id("  10.1038/test  ") == "DOI:10.1038/test"


# ---------------------------------------------------------------------------
# lookup_paper (mocked httpx)
# ---------------------------------------------------------------------------


class TestLookupPaper:
    """Tests for Semantic Scholar paper lookup with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_successful_lookup(self, monkeypatch):
        """Successful paper lookup returns a Paper model."""
        mock_response_data = {
            "paperId": "abc123",
            "title": "Test Paper Title",
            "abstract": "A test abstract.",
            "authors": [{"authorId": "1", "name": "Alice Smith"}, {"authorId": "2", "name": "Bob Jones"}],
            "year": 2024,
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.12345"},
            "citationCount": 42,
            "venue": "Nature",
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "url": "https://semanticscholar.org/paper/abc123",
        }

        class MockResponse:
            status_code = 200
            def json(self):
                return mock_response_data
            def raise_for_status(self):
                pass

        class MockClient:
            def __init__(self, **kwargs):
                pass
            async def get(self, url, **kwargs):
                return MockResponse()
            async def aclose(self):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)

        paper = await lookup_paper("DOI:10.1234/test")
        assert paper is not None
        assert paper.title == "Test Paper Title"
        assert paper.doi == "10.1234/test"
        assert len(paper.authors) == 2
        assert paper.year == 2024
        assert paper.citation_count == 42
        assert paper.pdf_url == "https://example.com/paper.pdf"
        assert paper.metadata.get("s2_arxiv_id") == "2401.12345"

    @pytest.mark.asyncio
    async def test_404_returns_none(self, monkeypatch):
        """404 response returns None."""
        class MockResponse:
            status_code = 404

        class MockClient:
            def __init__(self, **kwargs):
                pass
            async def get(self, url, **kwargs):
                return MockResponse()
            async def aclose(self):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)

        paper = await lookup_paper("DOI:10.1234/nonexistent")
        assert paper is None

    @pytest.mark.asyncio
    async def test_429_returns_none(self, monkeypatch):
        """429 rate limit returns None."""
        class MockResponse:
            status_code = 429

        class MockClient:
            def __init__(self, **kwargs):
                pass
            async def get(self, url, **kwargs):
                return MockResponse()
            async def aclose(self):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)

        paper = await lookup_paper("DOI:10.1234/rate_limited")
        assert paper is None

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self, monkeypatch):
        """Network error returns None."""
        class MockClient:
            def __init__(self, **kwargs):
                pass
            async def get(self, url, **kwargs):
                raise ConnectionError("Network error")
            async def aclose(self):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)

        paper = await lookup_paper("DOI:10.1234/network_error")
        assert paper is None

    @pytest.mark.asyncio
    async def test_empty_paper_id_returns_none(self, monkeypatch):
        """Empty paper ID returns None without making a request."""
        paper = await lookup_paper("")
        assert paper is None

    @pytest.mark.asyncio
    async def test_missing_fields_handled(self, monkeypatch):
        """Response with missing optional fields still returns a Paper."""
        mock_response_data = {
            "paperId": "abc123",
            "title": "Minimal Paper",
        }

        class MockResponse:
            status_code = 200
            def json(self):
                return mock_response_data
            def raise_for_status(self):
                pass

        class MockClient:
            def __init__(self, **kwargs):
                pass
            async def get(self, url, **kwargs):
                return MockResponse()
            async def aclose(self):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)

        paper = await lookup_paper("abc123")
        assert paper is not None
        assert paper.title == "Minimal Paper"
        assert paper.abstract is None
        assert paper.doi is None
        assert paper.year is None
        assert paper.authors == []


# ---------------------------------------------------------------------------
# StepType.PAPER_LOOKUP coercion
# ---------------------------------------------------------------------------


class TestPaperLookupStepType:
    """Tests for PAPER_LOOKUP step type and coercion."""

    def test_enum_value(self):
        assert StepType.PAPER_LOOKUP.value == "paper_lookup"

    def test_direct_coercion(self):
        assert coerce_step_type("paper_lookup") == StepType.PAPER_LOOKUP

    def test_alias_semantic_scholar_lookup(self):
        assert coerce_step_type("semantic_scholar_lookup") == StepType.PAPER_LOOKUP

    def test_alias_lookup_paper(self):
        assert coerce_step_type("lookup_paper") == StepType.PAPER_LOOKUP

    def test_tool_field_coercion(self):
        assert coerce_step_type("paper_lookup", tool="paper_lookup") == StepType.PAPER_LOOKUP
