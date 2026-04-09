"""Tests for PDF download utilities."""

import pytest
import httpx
from unittest.mock import Mock, AsyncMock, patch

from perspicacite.pipeline.download import (
    PDFDownloader,
    get_open_access_url,
    get_pdf_from_alternative_endpoint,
    retrieve_paper_content,
)
from perspicacite.pipeline.download.base import PaperContent, PaperDiscovery


class TestPDFDownloader:
    """Tests for PDFDownloader."""

    @pytest.fixture
    def downloader(self):
        return PDFDownloader()

    @pytest.mark.asyncio
    async def test_download_success(self, downloader):
        """Test successful PDF download."""
        mock_response = Mock()
        mock_response.content = b"PDF content here"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await downloader.download(
            "https://example.com/paper.pdf",
            http_client=mock_client,
        )

        assert result == b"PDF content here"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_http_error(self, downloader):
        """Test download with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "404",
                request=Mock(),
                response=Mock(status_code=404),
            )
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await downloader.download(
            "https://example.com/notfound.pdf",
            http_client=mock_client,
        )

        assert result is None


class TestAlternativeEndpoint:
    """Tests for alternative endpoint PDF download."""

    @pytest.mark.asyncio
    async def test_get_pdf_from_alternative_endpoint_embed(self):
        """Test finding PDF in embed tag."""
        html_content = """
        <html>
        <body>
            <embed type="application/pdf" src="/download/paper.pdf">
        </body>
        </html>
        """

        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()

        mock_pdf_response = Mock()
        mock_pdf_response.content = b"PDF content from embed"
        mock_pdf_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_response,
            mock_pdf_response,
        ])

        result = await get_pdf_from_alternative_endpoint(
            "10.1234/test",
            "https://example.com/",
            http_client=mock_client,
        )

        assert result == b"PDF content from embed"
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_pdf_from_alternative_endpoint_iframe(self):
        """Test finding PDF in iframe tag."""
        html_content = """
        <html>
        <body>
            <iframe src="https://example.com/file.pdf"></iframe>
        </body>
        </html>
        """

        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()

        mock_pdf_response = Mock()
        mock_pdf_response.content = b"PDF content from iframe"
        mock_pdf_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_response,
            mock_pdf_response,
        ])

        result = await get_pdf_from_alternative_endpoint(
            "10.1234/test",
            "https://example.com/",
            http_client=mock_client,
        )

        assert result == b"PDF content from iframe"

    @pytest.mark.asyncio
    async def test_get_pdf_from_alternative_endpoint_link(self):
        """Test finding PDF in a link tag."""
        html_content = """
        <html>
        <body>
            <a href="/files/paper.pdf">Download PDF</a>
        </body>
        </html>
        """

        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()

        mock_pdf_response = Mock()
        mock_pdf_response.content = b"PDF content from link"
        mock_pdf_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_response,
            mock_pdf_response,
        ])

        result = await get_pdf_from_alternative_endpoint(
            "10.1234/test",
            "https://example.com/",
            http_client=mock_client,
        )

        assert result == b"PDF content from link"

    @pytest.mark.asyncio
    async def test_get_pdf_from_alternative_endpoint_not_found(self):
        """Test when no PDF is found in the page."""
        html_content = "<html><body>No PDF here</body></html>"

        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_pdf_from_alternative_endpoint(
            "10.1234/test",
            "https://example.com/",
            http_client=mock_client,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_pdf_from_alternative_endpoint_http_error(self):
        """Test handling HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "404",
                request=Mock(),
                response=Mock(status_code=404),
            )
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_pdf_from_alternative_endpoint(
            "10.1234/test",
            "https://example.com/",
            http_client=mock_client,
        )

        assert result is None


class TestRetrievePaperContent:
    """Tests for the unified retrieve_paper_content pipeline."""

    @pytest.mark.asyncio
    async def test_returns_abstract_when_no_full_text(self, monkeypatch, tmp_path):
        """When no full text sources work but abstract is available, return abstract."""
        monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.com")
        # Use unique DOI to avoid cache collisions
        doi = "10.1234/test-abstract-only-unique-001"

        # Mock OpenAlex returning an abstract
        mock_oa_response = Mock()
        mock_oa_response.status_code = 200
        mock_oa_response.json = Mock(return_value={
            "title": "Test Paper",
            "type": "article",
            "open_access": {"is_oa": False, "oa_url": None},
            "ids": {},
            "abstract_inverted_index": {"Test": [0], "abstract": [1], "text": [2], "for": [3], "testing": [4], "purposes": [5]},
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_oa_response)

        with patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path):
            result = await retrieve_paper_content(
                doi,
                http_client=mock_client,
            )

        assert result.success is True
        assert result.content_type == "abstract"
        assert result.abstract == "Test abstract text for testing purposes"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_content(self, monkeypatch, tmp_path):
        """When no content at all is available, return none type."""
        monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
        monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
        doi = "10.1234/nonexistent-unique-002"

        # OpenAlex fails
        mock_oa_response = Mock()
        mock_oa_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_oa_response)

        with patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path):
            result = await retrieve_paper_content(
                doi,
                http_client=mock_client,
            )

        assert result.success is False
        assert result.content_type == "none"

    @pytest.mark.asyncio
    async def test_structured_from_pmc(self, monkeypatch, tmp_path):
        """When PMCID is found and PMC has content, return structured."""
        monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.com")
        doi = "10.1234/pmc-paper-unique-003"

        # Mock OpenAlex returning PMCID
        mock_oa_response = Mock()
        mock_oa_response.status_code = 200
        mock_oa_response.json = Mock(return_value={
            "title": "PMC Paper",
            "type": "article",
            "open_access": {"is_oa": True, "oa_url": None},
            "ids": {"pmcid": "PMC12345"},
            "abstract_inverted_index": None,
        })

        # Mock Unpaywall
        mock_up_response = Mock()
        mock_up_response.status_code = 200
        mock_up_response.json = Mock(return_value={
            "is_oa": True,
            "best_oa_location": None,
            "oa_locations": [],
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_oa_response, mock_up_response])

        # Mock PMC to return structured text
        with patch(
            "perspicacite.pipeline.download.unified.get_fulltext_from_pmc",
            new_callable=AsyncMock,
            return_value=("This is a long full text from PMC " * 20, {"Introduction": "Some intro text"}),
        ), patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path):
            result = await retrieve_paper_content(
                doi,
                http_client=mock_client,
            )

        assert result.success is True
        assert result.content_type == "structured"
        assert result.content_source == "pmc"
        assert result.sections == {"Introduction": "Some intro text"}

    @pytest.mark.asyncio
    async def test_skips_pdf_when_no_parser(self, monkeypatch, tmp_path):
        """When pdf_parser is None, PDF sources are skipped entirely."""
        monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.com")
        doi = "10.1234/oa-paper-unique-004"

        # OpenAlex returns OA URL but no parser
        mock_oa_response = Mock()
        mock_oa_response.status_code = 200
        mock_oa_response.json = Mock(return_value={
            "title": "OA Paper",
            "type": "article",
            "open_access": {"is_oa": True, "oa_url": "https://publisher.com/paper.pdf"},
            "ids": {},
            "abstract_inverted_index": {"This": [0], "is": [1], "a": [2], "longer": [3], "abstract": [4], "for": [5], "testing": [6]},
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_oa_response)

        with patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path):
            result = await retrieve_paper_content(
                doi,
                http_client=mock_client,
                pdf_parser=None,  # No parser
            )

        # Should fall through to abstract (no PDF parsing possible)
        assert result.success is True
        assert result.content_type == "abstract"


class TestUnpaywall:
    """Tests for Unpaywall integration."""

    @pytest.fixture
    def test_email(self):
        """Test email for Unpaywall."""
        return "test@example.com"

    @pytest.mark.asyncio
    async def test_get_open_access_url_found(self, test_email):
        """Test finding OA URL via Unpaywall."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={
            "is_oa": True,
            "best_oa_location": {
                "pdf_url": "https://oa.example.com/paper.pdf"
            }
        })
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_open_access_url(
            "10.1234/test",
            http_client=mock_client,
            email=test_email,
        )

        assert result == "https://oa.example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_get_open_access_url_not_found(self, test_email):
        """Test when no OA version available."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={
            "is_oa": False,
            "best_oa_location": None
        })
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await get_open_access_url(
            "10.1234/paywalled",
            http_client=mock_client,
            email=test_email,
        )

        assert result is None
