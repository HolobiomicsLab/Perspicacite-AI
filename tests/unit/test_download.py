"""Tests for PDF download utilities."""

import pytest
import httpx
from unittest.mock import Mock, AsyncMock

from perspicacite.pipeline.download import (
    PDFDownloader,
    get_open_access_url,
    get_pdf_from_alternative_endpoint,
    get_pdf_with_fallback,
)


class TestPDFDownloader:
    """Tests for PDFDownloader."""

    @pytest.fixture
    def downloader(self):
        return PDFDownloader()

    @pytest.mark.asyncio
    async def test_download_success(self, downloader):
        """Test successful PDF download."""
        # Mock response
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
        # HTML with embed tag
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
        
        # Mock PDF download response
        mock_pdf_response = Mock()
        mock_pdf_response.content = b"PDF content from embed"
        mock_pdf_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_response,      # First call for HTML page
            mock_pdf_response,  # Second call for PDF
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


class TestPDFWithFallback:
    """Tests for PDF download with fallback to alternative endpoint."""

    @pytest.fixture
    def test_email(self):
        """Test email for Unpaywall."""
        return "test@example.com"

    @pytest.mark.asyncio
    async def test_get_pdf_with_fallback_unpaywall_success(self, test_email, monkeypatch):
        """Test when Unpaywall finds the PDF."""
        # Set email in environment for the test
        monkeypatch.setenv("UNPAYWALL_EMAIL", test_email)
        
        # Mock Unpaywall response
        mock_unpaywall_response = Mock()
        mock_unpaywall_response.json = Mock(return_value={
            "is_oa": True,
            "best_oa_location": {
                "pdf_url": "https://oa.example.com/paper.pdf"
            }
        })
        mock_unpaywall_response.raise_for_status = Mock()
        
        # Mock PDF download response
        mock_pdf_response = Mock()
        mock_pdf_response.content = b"PDF from Unpaywall"
        mock_pdf_response.headers = {"content-type": "application/pdf"}
        mock_pdf_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_unpaywall_response,   # First call: Unpaywall API
            mock_pdf_response,         # Second call: PDF download
        ])
        
        result = await get_pdf_with_fallback(
            "10.1234/test",
            alternative_endpoint="https://alternative.com/",
            http_client=mock_client,
        )
        
        assert result == b"PDF from Unpaywall"
        # Should only make 2 calls (Unpaywall + PDF download), not try alternative
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_pdf_with_fallback_uses_alternative(self, test_email, monkeypatch):
        """Test fallback to alternative endpoint when Unpaywall fails."""
        # Set email in environment for the test
        monkeypatch.setenv("UNPAYWALL_EMAIL", test_email)
        
        # Mock Unpaywall response - no OA available
        mock_unpaywall_response = Mock()
        mock_unpaywall_response.json = Mock(return_value={
            "is_oa": False,
            "best_oa_location": None
        })
        mock_unpaywall_response.raise_for_status = Mock()
        
        # Mock alternative endpoint HTML
        html_content = """
        <html>
        <body>
            <embed type="application/pdf" src="/download/paper.pdf">
        </body>
        </html>
        """
        mock_html_response = Mock()
        mock_html_response.text = html_content
        mock_html_response.raise_for_status = Mock()
        
        # Mock PDF download from alternative
        mock_pdf_response = Mock()
        mock_pdf_response.content = b"PDF from alternative"
        mock_pdf_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_unpaywall_response,  # Unpaywall
            mock_html_response,       # Alternative HTML
            mock_pdf_response,        # PDF from alternative
        ])
        
        result = await get_pdf_with_fallback(
            "10.1234/test",
            alternative_endpoint="https://alternative.com/",
            http_client=mock_client,
        )
        
        assert result == b"PDF from alternative"
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_get_pdf_with_fallback_no_alternative(self, test_email, monkeypatch):
        """Test when Unpaywall fails and no alternative is provided."""
        # Set email in environment for the test
        monkeypatch.setenv("UNPAYWALL_EMAIL", test_email)
        
        mock_unpaywall_response = Mock()
        mock_unpaywall_response.json = Mock(return_value={
            "is_oa": False,
            "best_oa_location": None
        })
        mock_unpaywall_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_unpaywall_response)
        
        result = await get_pdf_with_fallback(
            "10.1234/test",
            alternative_endpoint=None,  # No alternative
            http_client=mock_client,
        )
        
        assert result is None
        # Should only call Unpaywall
        assert mock_client.get.call_count == 1


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
