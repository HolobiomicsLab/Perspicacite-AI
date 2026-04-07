"""PDF and content download from multiple sources.

This package provides download functionality from various sources:
- Unpaywall (open access; requires contact email)
- arXiv (open access, PDF + HTML)
- Publisher routes (ACS, RSC, AAAS, Springer, Wiley, Elsevier)
- OpenAlex OA PDF URLs
- Europe PMC JATS XML (structured sections + references)
- Alternative endpoints (Sci-Hub mirrors)

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

from .unified import retrieve_paper_content
from .base import DownloadResult, ContentResult, PDFDownloader, PaperContent, PaperDiscovery
from .fallback import get_pdf_with_fallback, get_content_with_fallback
from .unpaywall import get_open_access_url
from .alternative import download_from_alternative_endpoint as get_pdf_from_alternative_endpoint

# Publisher-specific modules (for direct access)
from . import unpaywall
from . import arxiv
from . import wiley
from . import elsevier
from . import aaas
from . import acs
from . import rsc
from . import springer
from . import alternative
from . import openalex_oa
from . import europepmc

__all__ = [
    # Unified pipeline (preferred)
    "retrieve_paper_content",
    "PaperContent",
    "PaperDiscovery",
    # Legacy (will be removed after full migration)
    "get_pdf_with_fallback",
    "get_content_with_fallback",
    # Common utilities
    "get_open_access_url",
    "get_pdf_from_alternative_endpoint",
    "DownloadResult",
    "ContentResult",
    "PDFDownloader",
    # Publisher modules
    "unpaywall",
    "arxiv",
    "wiley",
    "elsevier",
    "aaas",
    "acs",
    "rsc",
    "springer",
    "alternative",
    "openalex_oa",
    "europepmc",
]
