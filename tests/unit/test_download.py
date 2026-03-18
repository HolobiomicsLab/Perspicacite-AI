"""Tests for PDF download utilities."""

import pytest
import httpx
from unittest.mock import Mock, AsyncMock

from perspicacite.pipeline.download import PDFDownloader, get_open_access_url


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

    @pytest.mark.asyncio
    async def test_download_and_parse_success(self, downloader):
        """Test download and parse flow."""
        # Mock PDF bytes
        pdf_bytes = b"%PDF-1.4 test content"

        mock_response = Mock()
        mock_response.content = pdf_bytes
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        # Mock parser
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=Mock(text="Parsed text content"))

        result = await downloader.download_and_parse(
            "https://example.com/paper.pdf",
            pdf_parser=mock_parser,
            http_client=mock_client,
        )

        assert result == "Parsed text content"

    @pytest.mark.asyncio
    async def test_download_and_parse_download_fails(self, downloader):
        """Test download and parse when download fails."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.NetworkError("Connection failed"))

        mock_parser = Mock()

        result = await downloader.download_and_parse(
            "https://example.com/paper.pdf",
            pdf_parser=mock_parser,
            http_client=mock_client,
        )

        assert result is None


class TestUnpaywall:
    """Tests for Unpaywall integration."""

    @pytest.mark.asyncio
    async def test_get_open_access_url_found(self):
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
        )

        assert result == "https://oa.example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_get_open_access_url_not_found(self):
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
        )

        assert result is None
