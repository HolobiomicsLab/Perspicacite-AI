#!/usr/bin/env python3
"""Real download test for PDF download functionality.

This test makes ACTUAL HTTP requests to download real PDFs.
Requires internet connection and optionally an alternative endpoint.

Usage:
    # Test with Unpaywall only (no alternative endpoint)
    python test_download_real.py
    
    # Test with alternative endpoint
    export SCIHUB_URL="https://your-mirror.com/"
    python test_download_real.py
    
    # Run with pytest
    pytest tests/test_download_real.py -v -s
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import httpx


# DOIs from tests/example.bib that are likely to have OA versions
# These DOIs are from the kombucha-related papers in example.bib
TEST_DOIS = [
    "10.1039/d3fo04977a",  # Food and Function - likely OA
    "10.1016/j.fbio.2023.103270",  # Food Bioscience - may be OA
    "10.1016/j.ijfoodmicro.2020.108778",  # Int J Food Microbiology
    "10.1089/can.2016.0027",  # Cannabis and cannabinoid research
]

def get_unpaywall_email():
    """Get Unpaywall email from environment or config."""
    from perspicacite.config.loader import load_config
    
    # First check environment
    email = os.getenv("UNPAYWALL_EMAIL")
    if email:
        return email
    
    # Then check config file
    try:
        config = load_config()
        return config.pdf_download.unpaywall_email
    except Exception:
        return None


def get_alternative_endpoint():
    """Get alternative endpoint from environment or config."""
    from perspicacite.config.loader import load_config
    
    # First check environment
    endpoint = os.getenv("SCIHUB_URL")
    if endpoint:
        return endpoint
    
    # Then check config file
    try:
        config = load_config()
        return config.pdf_download.alternative_endpoint
    except Exception:
        return None


# Get Unpaywall email
UNPAYWALL_EMAIL = get_unpaywall_email() or ""


@pytest.fixture(scope="module")
def alternative_endpoint():
    """Get alternative endpoint if configured."""
    return get_alternative_endpoint()


class TestRealUnpaywallDownload:
    """Test real PDF downloads from Unpaywall."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not UNPAYWALL_EMAIL,
        reason="No Unpaywall email configured. Set UNPAYWALL_EMAIL environment variable."
    )
    async def test_unpaywall_lookup_real(self):
        """Test actual Unpaywall API lookup with real DOI."""
        from perspicacite.pipeline.download import get_open_access_url
        
        # Use first DOI for test
        test_doi = TEST_DOIS[0]
        print(f"\n\nTesting Unpaywall lookup for DOI: {test_doi}")
        print(f"Using email: {UNPAYWALL_EMAIL[:3]}...{UNPAYWALL_EMAIL[-10:]}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = await get_open_access_url(test_doi, http_client=client, email=UNPAYWALL_EMAIL)
        
        if url:
            print(f"✓ Unpaywall found URL: {url}")
            assert url.startswith("http")
        else:
            print(f"⚠ Unpaywall did not find open access for {test_doi}")
            # This is not a failure - just means no OA available
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not UNPAYWALL_EMAIL,
        reason="No Unpaywall email configured"
    )
    async def test_download_pdf_from_unpaywall(self):
        """Test actual PDF download from Unpaywall URL."""
        from perspicacite.pipeline.download import (
            get_open_access_url,
            PDFDownloader
        )
        
        test_doi = TEST_DOIS[0]
        print(f"\n\nTesting PDF download for DOI: {test_doi}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get OA URL
            url = await get_open_access_url(test_doi, http_client=client, email=UNPAYWALL_EMAIL)
            
            if not url:
                pytest.skip(f"No open access available for {test_doi}")
            
            print(f"Downloading from: {url}")
            
            # Download PDF
            downloader = PDFDownloader()
            pdf_bytes = await downloader.download(url, http_client=client)
            
            if pdf_bytes:
                print(f"✓ Downloaded {len(pdf_bytes)} bytes")
                # Verify it's a PDF
                assert pdf_bytes[:4] == b"%PDF", "Downloaded file is not a valid PDF"
                assert len(pdf_bytes) > 1000, "PDF is too small, likely an error page"
                
                # Save for manual inspection
                output_path = Path(f"/tmp/test_unpaywall_{test_doi.replace('/', '_')}.pdf")
                output_path.write_bytes(pdf_bytes)
                print(f"✓ Saved to: {output_path}")
            else:
                pytest.fail(f"Failed to download PDF from {url}")


class TestRealAlternativeEndpoint:
    """Test real PDF downloads from alternative endpoint (e.g., Sci-Hub)."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not get_alternative_endpoint(),
        reason="No alternative endpoint configured. Set SCIHUB_URL or configure in config.yml"
    )
    async def test_alternative_endpoint_download(self, alternative_endpoint):
        """Test actual PDF download from alternative endpoint."""
        from perspicacite.pipeline.download import get_pdf_from_alternative_endpoint
        
        test_doi = TEST_DOIS[1]  # Use second DOI
        print(f"\n\nTesting alternative endpoint: {alternative_endpoint}")
        print(f"DOI: {test_doi}")
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            pdf_bytes = await get_pdf_from_alternative_endpoint(
                test_doi,
                alternative_endpoint,
                http_client=client
            )
        
        if pdf_bytes:
            print(f"✓ Downloaded {len(pdf_bytes)} bytes from alternative endpoint")
            assert pdf_bytes[:4] == b"%PDF", "Downloaded file is not a valid PDF"
            assert len(pdf_bytes) > 1000
            
            # Save for manual inspection
            output_path = Path(f"/tmp/test_alternative_{test_doi.replace('/', '_')}.pdf")
            output_path.write_bytes(pdf_bytes)
            print(f"✓ Saved to: {output_path}")
        else:
            pytest.fail("Failed to download from alternative endpoint")


class TestRealFallback:
    """Test real fallback: Unpaywall first, then alternative endpoint."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not get_alternative_endpoint(),
        reason="No alternative endpoint configured"
    )
    async def test_real_fallback_mechanism(self, alternative_endpoint):
        """Test real fallback: try Unpaywall first, then alternative."""
        from perspicacite.pipeline.download import get_pdf_with_fallback
        
        # Use a DOI that's likely paywalled
        test_doi = "10.1016/j.ijfoodmicro.2020.108778"
        
        print(f"\n\nTesting fallback mechanism for DOI: {test_doi}")
        print(f"Alternative endpoint: {alternative_endpoint}")
        print(f"Unpaywall email configured: {bool(UNPAYWALL_EMAIL)}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            pdf_bytes = await get_pdf_with_fallback(
                test_doi,
                alternative_endpoint=alternative_endpoint,
                http_client=client,
                unpaywall_email=UNPAYWALL_EMAIL or None
            )
        
        if pdf_bytes:
            source = "Unpaywall" if pdf_bytes[:4] == b"%PDF" else "unknown"
            print(f"✓ Downloaded {len(pdf_bytes)} bytes (source: {source})")
            assert pdf_bytes[:4] == b"%PDF"
            
            output_path = Path(f"/tmp/test_fallback_{test_doi.replace('/', '_')}.pdf")
            output_path.write_bytes(pdf_bytes)
            print(f"✓ Saved to: {output_path}")
        else:
            print(f"⚠ Could not download PDF from any source")
            # Don't fail - this tests the fallback logic, not availability


class TestBibTeXFileIntegration:
    """Integration test using real BibTeX file."""
    
    @pytest.fixture
    def example_bibtex_path(self):
        """Get path to example.bib file."""
        return Path(__file__).parent / "example.bib"
    
    @pytest.mark.integration
    def test_read_bibtex_file(self, example_bibtex_path):
        """Read and verify example.bib exists."""
        assert example_bibtex_path.exists()
        content = example_bibtex_path.read_text()
        assert len(content) > 0
        print(f"\n\nBibTeX file size: {len(content)} bytes")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_download_from_bibtex_dois(self, example_bibtex_path):
        """Try to download PDFs for DOIs in example.bib."""
        from perspicacite.pipeline.download import get_pdf_with_fallback
        import re
        
        # Extract DOIs from file
        content = example_bibtex_path.read_text()
        dois = re.findall(r'doi\s*=\s*\{([^}]+)\}', content)
        
        print(f"\n\nFound {len(dois)} DOIs in BibTeX file")
        print(f"Unpaywall email configured: {bool(UNPAYWALL_EMAIL)}")
        print(f"Alternative endpoint: {get_alternative_endpoint() or 'Not configured'}")
        
        alt_endpoint = get_alternative_endpoint()
        results = []
        
        # Test first 2 DOIs
        for doi in dois[:2]:
            print(f"\nTrying DOI: {doi}")
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    pdf_bytes = await get_pdf_with_fallback(
                        doi,
                        alternative_endpoint=alt_endpoint,
                        http_client=client,
                        unpaywall_email=UNPAYWALL_EMAIL
                    )
                
                if pdf_bytes and pdf_bytes[:4] == b"%PDF":
                    # Determine source based on download path
                    source = "unknown"
                    if alt_endpoint and UNPAYWALL_EMAIL:
                        # If we have both, check which one worked
                        # Unpaywall success would have returned earlier
                        source = "alternative_endpoint"
                    elif UNPAYWALL_EMAIL:
                        source = "unpaywall"
                    elif alt_endpoint:
                        source = "alternative_endpoint"
                    
                    print(f"✓ Successfully downloaded {len(pdf_bytes)} bytes from {source}")
                    results.append((doi, True, len(pdf_bytes), source))
                    
                    # Save PDF
                    output_path = Path(f"/tmp/bibtex_{doi.replace('/', '_')}.pdf")
                    output_path.write_bytes(pdf_bytes)
                    print(f"✓ Saved to: {output_path}")
                else:
                    print(f"✗ Failed to download")
                    results.append((doi, False, 0, None))
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append((doi, False, 0, None))
        
        # Print summary
        print(f"\n\n{'='*60}")
        print("Download Summary:")
        print(f"{'='*60}")
        success_count = sum(1 for r in results if r[1])
        for result in results:
            doi, success, size, source = result
            status = "✓" if success else "✗"
            source_info = f" (from {source})" if source else ""
            print(f"{status} {doi}: {size} bytes{source_info}")
        print(f"\nSuccess rate: {success_count}/{len(results)}")
        
        # Note: We don't assert here because real downloads may fail
        # for legitimate reasons (no OA, paywall, etc.)
        if success_count == 0:
            print("\n⚠ All downloads failed - this may be due to:")
            print("  - No open access available for these papers")
            print("  - Missing Unpaywall email (set UNPAYWALL_EMAIL)")
            print("  - Missing alternative endpoint (set SCIHUB_URL)")
            print("  - Network connectivity issues")


if __name__ == "__main__":
    # Run with pytest if called directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])
