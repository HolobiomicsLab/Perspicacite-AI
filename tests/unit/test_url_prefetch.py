"""Unit tests for URL pre-processing in the agentic orchestrator.

Tests:
- get_arxiv_id_from_url(): regex-based arXiv ID extraction
- AgenticOrchestrator._try_resolve_url(): URL detection and paper fetching
- AgenticOrchestrator._resolve_arxiv_metadata(): OpenAlex metadata lookup
- AgenticOrchestrator._generate_single_paper_answer(): single-paper answer path

Run: PYTHONPATH=src pytest tests/unit/test_url_prefetch.py -v
"""

import pytest

from perspicacite.pipeline.download.arxiv import get_arxiv_id_from_url
from perspicacite.rag.agentic.orchestrator import AgenticOrchestrator, _URL_RE, _DOI_IN_URL_RE


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
    """Tests for AgenticOrchestrator._try_resolve_url."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    @pytest.mark.asyncio
    async def test_no_url_returns_none(self):
        orch = self._make_orchestrator()
        result = await orch._try_resolve_url("what is CRISPR gene editing")
        assert result is None

    @pytest.mark.asyncio
    async def test_arxiv_url_fetches_html(self, monkeypatch):
        orch = self._make_orchestrator()

        async def mock_fetch_html(arxiv_id, http_client=None):
            return "Full paper text here with detailed content.", {"Intro": "Some intro"}, "HTML Page Title"

        async def mock_resolve_metadata(arxiv_id):
            return "Test Paper Title", "10.1234/test"

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch, "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url("https://arxiv.org/html/2604.06788v1")
        assert result is not None
        assert result["source"] == "url_fetch"
        assert result["relevance_score"] == 5
        assert result["title"] == "Test Paper Title"
        assert result["doi"] == "10.1234/test"
        assert result["full_text"] == "Full paper text here with detailed content."
        assert result["arxiv_id"] == "2604.06788"

    @pytest.mark.asyncio
    async def test_arxiv_url_strips_version(self, monkeypatch):
        """Version suffix is stripped for fetching."""
        captured_id = {}

        async def mock_fetch_html(arxiv_id, http_client=None):
            captured_id["id"] = arxiv_id
            return "Paper content.", None, None

        async def mock_resolve_metadata(arxiv_id):
            return None, None

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch := self._make_orchestrator(), "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url("https://arxiv.org/abs/2604.06788v2")
        assert captured_id["id"] == "2604.06788"
        assert result is not None
        assert result["arxiv_id"] == "2604.06788"

    @pytest.mark.asyncio
    async def test_arxiv_fetch_fails_returns_none(self, monkeypatch):
        orch = self._make_orchestrator()

        async def mock_fetch_html(arxiv_id, http_client=None):
            return None, None, None

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )

        result = await orch._try_resolve_url("https://arxiv.org/abs/9999.99999")
        # fetch_arxiv_html returns None → no arXiv result, and no DOI in URL
        assert result is None

    @pytest.mark.asyncio
    async def test_doi_url_resolves(self, monkeypatch):
        orch = self._make_orchestrator()

        class MockResult:
            success = True
            full_text = "Full text from DOI paper."
            content_type = "pdf"
            sections = None
            metadata = {"title": "DOI Paper Title"}

        async def mock_retrieve(doi, pdf_parser=None, unpaywall_email=None):
            return MockResult()

        monkeypatch.setattr(
            "perspicacite.pipeline.download.unified.retrieve_paper_content",
            mock_retrieve,
        )

        result = await orch._try_resolve_url("https://doi.org/10.1038/s41586-023-12345")
        assert result is not None
        assert result["source"] == "url_fetch"
        assert result["doi"] == "10.1038/s41586-023-12345"
        assert result["title"] == "DOI Paper Title"
        assert result["full_text"] == "Full text from DOI paper."

    @pytest.mark.asyncio
    async def test_non_resolvable_url_returns_none(self):
        orch = self._make_orchestrator()
        # A URL that's neither arXiv nor contains a DOI
        result = await orch._try_resolve_url("https://example.com/some-page")
        assert result is None

    @pytest.mark.asyncio
    async def test_url_with_trailing_punctuation_stripped(self, monkeypatch):
        """Trailing .,;:) is stripped from detected URL."""
        orch = self._make_orchestrator()

        captured_url = {}

        async def mock_fetch_html(arxiv_id, http_client=None):
            return "text", None, None

        async def mock_resolve_metadata(arxiv_id):
            return None, None

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch, "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url("Check https://arxiv.org/abs/2604.06788.")
        assert result is not None

    @pytest.mark.asyncio
    async def test_mixed_query_with_url(self, monkeypatch):
        """URL in a longer query is still detected and resolved."""
        orch = self._make_orchestrator()

        async def mock_fetch_html(arxiv_id, http_client=None):
            return "Paper content.", None, None

        async def mock_resolve_metadata(arxiv_id):
            return "Paper Title", None

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch, "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url(
            "summarize https://arxiv.org/abs/2604.06788 and compare with FBMN"
        )
        assert result is not None
        assert result["arxiv_id"] == "2604.06788"

    @pytest.mark.asyncio
    async def test_html_title_fallback_when_openalex_fails(self, monkeypatch):
        """When OpenAlex returns None, HTML title is used instead of arXiv ID fallback."""
        orch = self._make_orchestrator()

        async def mock_fetch_html(arxiv_id, http_client=None):
            return "Paper full text.", None, "From Perception to Autonomous Computational Modeling"

        async def mock_resolve_metadata(arxiv_id):
            return None, None  # OpenAlex hasn't indexed this paper

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch, "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url("https://arxiv.org/abs/2604.06788")
        assert result is not None
        assert result["title"] == "From Perception to Autonomous Computational Modeling"

    @pytest.mark.asyncio
    async def test_openalex_title_preferred_over_html_title(self, monkeypatch):
        """OpenAlex title takes priority when both sources return a title."""
        orch = self._make_orchestrator()

        async def mock_fetch_html(arxiv_id, http_client=None):
            return "Paper text.", None, "HTML Title"

        async def mock_resolve_metadata(arxiv_id):
            return "OpenAlex Canonical Title", "10.1234/test"

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch, "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url("https://arxiv.org/abs/2604.06788")
        assert result["title"] == "OpenAlex Canonical Title"

    @pytest.mark.asyncio
    async def test_no_title_uses_arxiv_id_fallback(self, monkeypatch):
        """When neither OpenAlex nor HTML provides a title, use arXiv ID."""
        orch = self._make_orchestrator()

        async def mock_fetch_html(arxiv_id, http_client=None):
            return "Paper text.", None, None  # No HTML title

        async def mock_resolve_metadata(arxiv_id):
            return None, None  # No OpenAlex title

        monkeypatch.setattr(
            "perspicacite.pipeline.download.arxiv.fetch_arxiv_html",
            mock_fetch_html,
        )
        monkeypatch.setattr(orch, "_resolve_arxiv_metadata", mock_resolve_metadata)

        result = await orch._try_resolve_url("https://arxiv.org/abs/2604.06788")
        assert result["title"] == "arXiv:2604.06788"


# ---------------------------------------------------------------------------
# _resolve_arxiv_metadata (with mocked httpx)
# ---------------------------------------------------------------------------


class TestResolveArxivMetadata:
    """Tests for OpenAlex metadata resolution from arXiv ID."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    @pytest.mark.asyncio
    async def test_successful_resolution(self, monkeypatch):
        orch = self._make_orchestrator()

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "title": "A Great Paper",
                    "ids": {"doi": "https://doi.org/10.1234/great"},
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def get(self, url, **kwargs):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)
        title, doi = await orch._resolve_arxiv_metadata("2604.06788")
        assert title == "A Great Paper"
        assert doi == "10.1234/great"

    @pytest.mark.asyncio
    async def test_404_returns_none(self, monkeypatch):
        orch = self._make_orchestrator()

        class MockResponse:
            status_code = 404

            def json(self):
                return {}

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def get(self, url, **kwargs):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)
        title, doi = await orch._resolve_arxiv_metadata("9999.99999")
        assert title is None
        assert doi is None

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self, monkeypatch):
        orch = self._make_orchestrator()

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def get(self, url, **kwargs):
                raise ConnectionError("Network unreachable")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)
        title, doi = await orch._resolve_arxiv_metadata("2604.06788")
        assert title is None
        assert doi is None

    @pytest.mark.asyncio
    async def test_missing_doi_field(self, monkeypatch):
        orch = self._make_orchestrator()

        class MockResponse:
            status_code = 200

            def json(self):
                return {"title": "Paper Without DOI", "ids": {}}

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def get(self, url, **kwargs):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr("httpx.AsyncClient", MockClient)
        title, doi = await orch._resolve_arxiv_metadata("2604.06788")
        assert title == "Paper Without DOI"
        assert doi is None


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

        answer, citation_map = await orch._generate_single_paper_answer(
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

        answer, citation_map = await orch._generate_single_paper_answer(
            "what methodology does this paper use?",
            papers, session,
            is_summary_request=False,
        )

        # Should use the original query, not the summary prompt
        assert "what methodology" in captured_prompt["prompt"]
