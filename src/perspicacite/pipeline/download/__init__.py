"""PDF and content download from multiple sources.

This package provides download functionality from various sources:
- Unpaywall (open access; requires contact email)
- arXiv (open access, PDF + HTML)
- Publisher routes (ACS, RSC, AAAS, Springer, Wiley, Elsevier)
- OpenAlex OA PDF URLs
- Europe PMC JATS XML (structured sections + references)
- Alternative endpoints (user-configured private/institutional repositories)

Main entry point:

    retrieve_paper_content() - Unified pipeline with quality-based priority:
        structured > full_text > abstract > discard

    Usage:
        from perspicacite.pipeline.download import retrieve_paper_content, PaperContent

        result = await retrieve_paper_content(
            doi="10.1038/s41586-024-12345-6",
            pdf_parser=my_parser,
            unpaywall_email="user@example.com",
        )
        if result.success:
            print(result.content_type)   # "structured", "full_text", "abstract"
            print(result.full_text)
            print(result.sections)       # None unless from PMC/arXiv HTML
"""

# Publisher-specific modules (for direct access)
from . import (
    aaas,
    acs,
    alternative,
    arxiv,
    elsevier,
    openalex_oa,
    pmc,
    rsc,
    springer,
    unpaywall,
    wiley,
)
from .alternative import download_from_alternative_endpoint as get_pdf_from_alternative_endpoint
from .base import ContentResult, DownloadResult, PaperContent, PaperDiscovery, PDFDownloader
from .fallback import get_content_with_fallback, get_pdf_with_fallback
from .unified import retrieve_paper_content
from .unpaywall import get_open_access_url

__all__ = [
    "ContentResult",
    "DownloadResult",
    "PDFDownloader",
    "PaperContent",
    "PaperDiscovery",
    "aaas",
    "acs",
    "alternative",
    "arxiv",
    "elsevier",
    "get_content_with_fallback",
    # Common utilities
    "get_open_access_url",
    "get_pdf_from_alternative_endpoint",
    # Legacy (will be removed after full migration)
    "get_pdf_with_fallback",
    "openalex_oa",
    "pmc",
    # Unified pipeline (preferred)
    "retrieve_paper_content",
    "rsc",
    "springer",
    # Publisher modules
    "unpaywall",
    "wiley",
]
