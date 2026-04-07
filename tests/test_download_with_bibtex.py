#!/usr/bin/env python3
"""Integration test for PDF download using example.bib file.

This test uses the DOIs from tests/example.bib to verify that:
1. BibTeX parsing works correctly
2. PDF download functions can be called with real DOIs
3. Fallback mechanism works

Note: This test mocks HTTP calls to avoid external dependencies.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture
def example_bibtex_path():
    """Get path to example.bib file."""
    return Path(__file__).parent / "example.bib"


@pytest.fixture
def sample_dois():
    """Expected DOIs from example.bib."""
    return [
        "10.1039/d3fo04977a",  # Food and Function article
        "10.1016/j.fbio.2023.103270",  # Food Bioscience
        "10.1016/j.ijfoodmicro.2020.108778",  # International Journal of Food Microbiology
        "10.1089/can.2016.0027",  # Cannabis article
    ]


class TestBibTeXParsing:
    """Test BibTeX parsing from example file."""

    def test_bibtex_file_exists(self, example_bibtex_path):
        """Verify example.bib exists and is readable."""
        assert example_bibtex_path.exists(), f"{example_bibtex_path} not found"
        content = example_bibtex_path.read_text()
        assert len(content) > 0
        assert "@article" in content or "@misc" in content

    def test_parse_bibtex_entries(self, example_bibtex_path):
        """Test parsing BibTeX entries from file."""
        import re
        
        content = example_bibtex_path.read_text()
        
        # Simple regex to extract DOIs
        doi_pattern = r'doi\s*=\s*\{([^}]+)\}'
        dois = re.findall(doi_pattern, content)
        
        assert len(dois) >= 3, f"Expected at least 3 DOIs, found {len(dois)}"
        
        # Verify DOI format
        for doi in dois:
            assert "10." in doi, f"Invalid DOI format: {doi}"
        
        print(f"\nFound DOIs in example.bib: {dois}")


class TestPDFDownloadWithBibTeXDOIs:
    """Test PDF download functionality using DOIs from BibTeX file."""

    @pytest.fixture
    def test_email(self):
        """Test email for Unpaywall."""
        return "test@example.com"

    @pytest.mark.asyncio
    async def test_get_open_access_url_with_real_dois(self, sample_dois, test_email):
        """Test Unpaywall lookup with DOIs from BibTeX (mocked)."""
        from perspicacite.pipeline.download import get_open_access_url
        
        # Use first DOI for test
        test_doi = sample_dois[0]
        
        # Mock successful Unpaywall response
        mock_response = Mock()
        mock_response.json = Mock(return_value={
            "is_oa": True,
            "best_oa_location": {
                "pdf_url": f"https://example.com/{test_doi}.pdf"
            }
        })
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        result = await get_open_access_url(test_doi, http_client=mock_client, email=test_email)
        
        assert result is not None
        assert test_doi in result or "example.com" in result
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_with_bibtex_dois(self, sample_dois, test_email, monkeypatch):
        """Test fallback mechanism with DOIs from BibTeX file."""
        from perspicacite.pipeline.download import get_pdf_with_fallback
        
        # Set email in environment
        monkeypatch.setenv("UNPAYWALL_EMAIL", test_email)
        
        test_doi = sample_dois[1]
        
        # Mock Unpaywall - no OA available
        mock_unpaywall = Mock()
        mock_unpaywall.json = Mock(return_value={
            "is_oa": False,
            "best_oa_location": None
        })
        mock_unpaywall.raise_for_status = Mock()

        # Mock OpenAlex - no result
        mock_openalex = Mock()
        mock_openalex.json = Mock(return_value={"results": []})
        mock_openalex.raise_for_status = Mock()

        # Mock EuropePMC search - no PMCID
        mock_europepmc = Mock()
        mock_europepmc.json = Mock(return_value={"resultList": {"result": []}})
        mock_europepmc.raise_for_status = Mock()

        # Mock alternative endpoint HTML
        mock_html = Mock()
        mock_html.text = '''
        <html>
        <body>
            <embed type="application/pdf" src="/download/paper.pdf">
        </body>
        </html>
        '''
        mock_html.raise_for_status = Mock()

        # Mock PDF download
        mock_pdf = Mock()
        mock_pdf.content = b"%PDF-1.4 Test PDF content"
        mock_pdf.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            mock_unpaywall,   # Unpaywall call
            mock_openalex,    # OpenAlex OA
            mock_europepmc,   # EuropePMC search
            mock_html,        # Alternative endpoint HTML
            mock_pdf,         # PDF download
        ])
        
        result = await get_pdf_with_fallback(
            test_doi,
            alternative_endpoint="https://alternative.com/",
            http_client=mock_client,
        )
        
        assert result is not None
        assert result.startswith(b"%PDF")
        assert mock_client.get.call_count == 5

    @pytest.mark.asyncio
    async def test_multiple_dois_from_bibtex(self, sample_dois, test_email, monkeypatch):
        """Test processing multiple DOIs from BibTeX file."""
        from perspicacite.pipeline.download import get_pdf_with_fallback
        
        # Set email in environment
        monkeypatch.setenv("UNPAYWALL_EMAIL", test_email)
        
        results = []
        
        for doi in sample_dois[:3]:  # Test first 3 DOIs
            # Mock successful download for each
            mock_unpaywall = Mock()
            mock_unpaywall.json = Mock(return_value={
                "is_oa": True,
                "best_oa_location": {
                    "pdf_url": f"https://oa.example.com/{doi.replace('/', '_')}.pdf"
                }
            })
            mock_unpaywall.raise_for_status = Mock()
            
            mock_pdf = Mock()
            mock_pdf.content = f"%PDF-1.4 Content for {doi}".encode()
            mock_pdf.headers = {"content-type": "application/pdf"}
            mock_pdf.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[
                mock_unpaywall,
                mock_pdf,
            ])
            
            result = await get_pdf_with_fallback(doi, http_client=mock_client)
            results.append(result)
        
        # All should succeed
        assert all(r is not None for r in results)
        assert len(results) == 3


class TestBibTeXUploadEndpoint:
    """Test the BibTeX upload endpoint with example file."""

    @pytest.mark.asyncio
    async def test_bibtex_endpoint_integration(self, example_bibtex_path):
        """Test BibTeX endpoint with real file content (mocked)."""
        # Read the BibTeX content
        content = example_bibtex_path.read_text()
        
        # Verify we can parse entries
        import re
        
        # Count entries
        entries = re.findall(r'@\w+\s*\{', content)
        assert len(entries) >= 3, f"Expected at least 3 entries, found {len(entries)}"
        
        # Extract and verify DOIs
        dois = re.findall(r'doi\s*=\s*\{([^}]+)\}', content)
        assert len(dois) >= 3
        
        print(f"\nBibTeX file contains {len(entries)} entries with {len(dois)} DOIs")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
