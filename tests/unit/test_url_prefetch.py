"""Unit tests for URL pre-processing in the agentic orchestrator.

Tests:
- get_arxiv_id_from_url(): regex-based arXiv ID extraction
- AgenticOrchestrator._try_resolve_url(): URL detection and paper fetching
- AgenticOrchestrator._generate_single_paper_answer(): single-paper answer path

Run: PYTHONPATH=src pytest tests/unit/test_url_prefetch.py -v
"""

import pytest

from perspicacite.models import PaperSource
from perspicacite.pipeline.download.arxiv import get_arxiv_id_from_url
from perspicacite.rag.agentic.orchestrator import _DOI_IN_URL_RE, _URL_RE, AgenticOrchestrator


class _MockRetrieveResult:
    """Minimal stand-in for PaperContent used in monkeypatched tests."""

    def __init__(
        self,
        success=True,
        full_text="Full paper text.",
        content_type="full_text",
        sections=None,
        abstract=None,
        metadata=None,
    ):
        self.success = success
        self.full_text = full_text
        self.content_type = content_type
        self.sections = sections
        self.abstract = abstract
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# get_arxiv_id_from_url
# ---------------------------------------------------------------------------


class TestGetArxivIdFromUrl:
    """Tests for arXiv ID extraction from various URL formats."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://arxiv.org/abs/2604.06788", "2604.06788"),
            ("https://arxiv.org/html/2604.06788v1", "2604.06788v1"),
            ("https://arxiv.org/pdf/2604.06788", "2604.06788"),
            ("https://arxiv.org/format/2604.06788", "2604.06788"),
            ("http://arxiv.org/abs/2101.12345", "2101.12345"),
            ("https://arxiv.org/abs/2101.12345v3", "2101.12345v3"),
            ("https://arxiv.org/abs/2301.00123", "2301.00123"),
        ],
    )
    def test_valid_arxiv_urls(self, url, expected):
        assert get_arxiv_id_from_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/paper",
            "https://doi.org/10.1038/s41586-023-12345",
            "not a url at all",
            "",
        ],
    )
    def test_non_arxiv_urls(self, url):
        assert get_arxiv_id_from_url(url) is None

    def test_none(self):
        assert get_arxiv_id_from_url(None) is None

    def test_url_in_context(self):
        """ID extracted even when URL is embedded in longer text."""
        url = "Please read https://arxiv.org/abs/2604.06788 and tell me"
        assert get_arxiv_id_from_url(url) == "2604.06788"

    def test_case_insensitive(self):
        assert get_arxiv_id_from_url("https://ARXIV.ORG/abs/2604.06788") == "2604.06788"


# ---------------------------------------------------------------------------
# _URL_RE and _DOI_IN_URL_RE module-level patterns
# ---------------------------------------------------------------------------


class TestUrlPatterns:
    """Tests for the module-level URL and DOI detection regexes."""

    def test_url_re_matches_http(self):
        m = _URL_RE.search("see https://arxiv.org/abs/2604.06788 for details")
        assert m is not None
        assert m.group(0) == "https://arxiv.org/abs/2604.06788"

    def test_url_re_matches_httpx(self):
        m = _URL_RE.search("check http://example.com")
        assert m is not None
        assert m.group(0) == "http://example.com"

    def test_url_re_no_match_no_scheme(self):
        m = _URL_RE.search("just some text without urls")
        assert m is None

    def test_doi_in_url_re_doi_org(self):
        m = _DOI_IN_URL_RE.search("https://doi.org/10.1038/s41586-023-12345-6")
        assert m is not None
        assert m.group(0) == "10.1038/s41586-023-12345-6"

    def test_doi_in_url_re_publisher_url(self):
        m = _DOI_IN_URL_RE.search("https://www.nature.com/articles/10.1038/s41586-023-12345-6")
        assert m is not None
        assert m.group(0).startswith("10.1038/")

    def test_doi_in_url_re_no_match(self):
        m = _DOI_IN_URL_RE.search("https://arxiv.org/abs/2604.06788")
        assert m is None


# ---------------------------------------------------------------------------
# _try_resolve_url (with mocks)
# ---------------------------------------------------------------------------


class TestTryResolveUrl:
    """Tests for AgenticOrchestrator._try_resolve_url.

    The function maps any URL (arxiv.org or doi.org/publisher) to a DOI and
    runs the unified retrieval pipeline once. These tests mock
    ``retrieve_paper_content`` (and, for the no-full-text path,
    ``lookup_paper``) and assert the contract of the returned normalized dict.
    """

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _patch_retrieve(self, monkeypatch, mock_func):
        """Patch retrieve_paper_content at both import sites."""
        monkeypatch.setattr(
            "perspicacite.pipeline.download.unified.retrieve_paper_content",
            mock_func,
        )

    @pytest.mark.asyncio
    async def test_no_url_returns_none(self):
        orch = self._make_orchestrator()
        result = await orch._try_resolve_url("what is CRISPR gene editing")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_resolvable_url_returns_none(self, monkeypatch):
        """A URL that isn't arXiv and contains no DOI doesn't trigger retrieval."""
        orch = self._make_orchestrator()

        async def mock_lookup(*args, **kwargs):
            return None

        monkeypatch.setattr(
            "perspicacite.search.semantic_scholar.lookup_paper", mock_lookup,
        )

        result = await orch._try_resolve_url("https://example.com/some-page")
        assert result is None

    @pytest.mark.asyncio
    async def test_arxiv_url_resolves_to_arxiv_doi(self, monkeypatch):
        """arXiv URLs are mapped to 10.48550/arXiv.<id> and routed through retrieve."""
        captured = {}

        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            captured["doi"] = doi
            return _MockRetrieveResult(
                full_text="Full paper text here with detailed content.",
                metadata={
                    "title": "Test Paper Title",
                    "authors": ["Alice Smith", "Bob Jones"],
                    "year": 2024,
                    "doi": doi,
                    "arxiv_id": "2604.06788",
                },
            )

        self._patch_retrieve(monkeypatch, mock_retrieve)
        orch = self._make_orchestrator()

        result = await orch._try_resolve_url("https://arxiv.org/html/2604.06788v1")
        assert captured["doi"] == "10.48550/arXiv.2604.06788"
        assert result is not None
        assert result["source"] == PaperSource.OPENALEX
        assert result["relevance_score"] == 5
        assert result["title"] == "Test Paper Title"
        assert result["doi"] == "10.48550/arXiv.2604.06788"
        assert result["full_text"].startswith("Full paper text")
        assert result["authors"] == ["Alice Smith", "Bob Jones"]
        assert result["year"] == 2024

    @pytest.mark.asyncio
    async def test_arxiv_url_strips_version(self, monkeypatch):
        """Version suffix is stripped before constructing the DOI."""
        captured = {}

        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            captured["doi"] = doi
            return _MockRetrieveResult(metadata={"title": "P", "doi": doi})

        self._patch_retrieve(monkeypatch, mock_retrieve)
        orch = self._make_orchestrator()

        result = await orch._try_resolve_url("https://arxiv.org/abs/2604.06788v2")
        assert captured["doi"] == "10.48550/arXiv.2604.06788"
        assert result is not None

    @pytest.mark.asyncio
    async def test_doi_url_resolves(self, monkeypatch):
        """A doi.org URL feeds the DOI directly into retrieve_paper_content."""
        captured = {}

        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            captured["doi"] = doi
            return _MockRetrieveResult(
                full_text="Full text from DOI paper.",
                content_type="full_text",
                metadata={
                    "title": "DOI Paper Title",
                    "authors": ["Carol White"],
                    "year": 2023,
                    "doi": doi,
                },
            )

        self._patch_retrieve(monkeypatch, mock_retrieve)
        orch = self._make_orchestrator()

        result = await orch._try_resolve_url("https://doi.org/10.1038/s41586-023-12345")
        assert captured["doi"] == "10.1038/s41586-023-12345"
        assert result is not None
        assert result["source"] == PaperSource.OPENALEX
        assert result["doi"] == "10.1038/s41586-023-12345"
        assert result["title"] == "DOI Paper Title"
        assert result["full_text"] == "Full text from DOI paper."
        assert result["authors"] == ["Carol White"]
        assert result["year"] == 2023

    @pytest.mark.asyncio
    async def test_arxiv_doi_url_populates_authors_year(self, monkeypatch):
        """Regression: doi.org URL with arXiv DOI must yield rich authors+year.

        Before unified.py started populating authors/year in metadata, this
        path produced ``authors=[], year=None`` and the references section
        rendered "Unknown, None".
        """
        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            return _MockRetrieveResult(
                full_text="Body of the paper.",
                metadata={
                    "title": "Heterogeneous Scientific Foundation Model Collaboration",
                    "authors": ["First Author", "Second Author", "Third Author"],
                    "year": 2026,
                    "arxiv_id": "2604.27351",
                    "doi": "10.48550/arXiv.2604.27351",
                },
            )

        self._patch_retrieve(monkeypatch, mock_retrieve)
        orch = self._make_orchestrator()

        result = await orch._try_resolve_url("https://doi.org/10.48550/arXiv.2604.27351")
        assert result is not None
        assert result["authors"] == ["First Author", "Second Author", "Third Author"]
        assert result["year"] == 2026
        assert result["title"] == "Heterogeneous Scientific Foundation Model Collaboration"

    @pytest.mark.asyncio
    async def test_url_with_trailing_punctuation_stripped(self, monkeypatch):
        """Trailing .,;:) is stripped from the detected URL before resolution."""
        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            return _MockRetrieveResult(metadata={"title": "T", "doi": doi})

        self._patch_retrieve(monkeypatch, mock_retrieve)
        orch = self._make_orchestrator()

        result = await orch._try_resolve_url("Check https://arxiv.org/abs/2604.06788.")
        assert result is not None

    @pytest.mark.asyncio
    async def test_mixed_query_with_url(self, monkeypatch):
        """URL in a longer query is still detected and resolved."""
        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            return _MockRetrieveResult(
                metadata={"title": "Paper Title", "doi": doi, "arxiv_id": "2604.06788"},
            )

        self._patch_retrieve(monkeypatch, mock_retrieve)
        orch = self._make_orchestrator()

        result = await orch._try_resolve_url(
            "summarize https://arxiv.org/abs/2604.06788 and compare with FBMN"
        )
        assert result is not None
        assert result["title"] == "Paper Title"

    @pytest.mark.asyncio
    async def test_falls_back_to_s2_when_no_full_text(self, monkeypatch):
        """When retrieve returns no full text, falls back to Semantic Scholar lookup."""
        from perspicacite.models.papers import Author, Paper

        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            return _MockRetrieveResult(success=False, full_text=None)

        async def mock_lookup(paper_id, http_client=None):
            return Paper(
                id="s2:abc",
                title="Paper From S2",
                authors=[Author(name="Eve Smith")],
                year=2025,
                doi="10.48550/arXiv.2604.06788",
                citation_count=3,
                source=PaperSource.WEB_SEARCH,
            )

        self._patch_retrieve(monkeypatch, mock_retrieve)
        monkeypatch.setattr(
            "perspicacite.search.semantic_scholar.lookup_paper", mock_lookup,
        )

        orch = self._make_orchestrator()
        result = await orch._try_resolve_url("https://arxiv.org/abs/2604.06788")
        assert result is not None
        assert result["title"] == "Paper From S2"
        assert result["authors"] == ["Eve Smith"]
        assert result["year"] == 2025
        assert result["_metadata_only"] is True
        assert result["relevance_score"] == 4


# ---------------------------------------------------------------------------
# _generate_single_paper_answer
# ---------------------------------------------------------------------------


class TestGenerateSinglePaperAnswer:
    """Tests for single-paper answer fast path."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _make_session(self):
        from perspicacite.rag.agentic.orchestrator import AgentSession
        session = AgentSession.__new__(AgentSession)
        session.messages = []
        session.knowledge_base = None
        session.research_findings = []
        session.evidence = None
        return session

    @pytest.mark.asyncio
    async def test_summary_request_uses_summary_prompt(self, monkeypatch):
        """is_summary_request=True triggers summary mode in the prompt."""
        orch = self._make_orchestrator()
        session = self._make_session()

        captured_prompt = {}

        async def mock_complete(prompt, **kwargs):
            captured_prompt["prompt"] = prompt
            return "This is a comprehensive summary of the paper."

        from unittest.mock import MagicMock
        orch.llm = MagicMock()
        orch.llm.complete = mock_complete
        orch._format_references_section = lambda papers: "[1] Test Paper"

        papers = [{"title": "Test Paper", "doi": "10.1/test", "full_text": "A" * 500}]

        _answer, citation_map = await orch._generate_single_paper_answer(
            "https://arxiv.org/abs/2604.06788", papers, session,
            is_summary_request=True,
        )

        assert "comprehensive overview" in captured_prompt["prompt"]
        assert "A" * 500 in captured_prompt["prompt"]
        assert citation_map["single_paper"] is True
        assert citation_map["cited_count"] == 1

    @pytest.mark.asyncio
    async def test_specific_question_uses_query(self, monkeypatch):
        """is_summary_request=False passes the query through as-is."""
        orch = self._make_orchestrator()
        session = self._make_session()

        captured_prompt = {}

        async def mock_complete(prompt, **kwargs):
            captured_prompt["prompt"] = prompt
            return "The paper uses a multi-agent approach."

        from unittest.mock import MagicMock
        orch.llm = MagicMock()
        orch.llm.complete = mock_complete
        orch._format_references_section = lambda papers: ""

        papers = [{"title": "Test Paper", "doi": "10.1/test", "full_text": "Methodology section here."}]

        _answer, _citation_map = await orch._generate_single_paper_answer(
            "what methodology does this paper use?",
            papers, session,
            is_summary_request=False,
        )

        # Should use the original query, not the summary prompt
        assert "what methodology" in captured_prompt["prompt"]
