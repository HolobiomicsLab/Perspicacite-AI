"""PDF and content download from multiple sources.

This package provides download functionality from various sources:
- Unpaywall (open access; requires contact email)
- arXiv (open access)
- Publisher routes (ACS, RSC, AAAS, Springer, Wiley TDM, …)
- OpenAlex OA PDF URLs and Europe PMC PDFs (no API keys)
- Wiley ``/doi/pdf/`` for typical ``10.1002/`` DOIs without a TDM token
- Alternative endpoints (Sci-Hub mirrors)

Usage:
    from perspicacite.pipeline.download import get_pdf_with_fallback
    
    pdf_bytes = await get_pdf_with_fallback(
        doi="10.1002/xxx",
        unpaywall_email="user@example.com",
        wiley_tdm_token="...",
    )
"""

from .fallback import get_pdf_with_fallback, get_content_with_fallback
from .base import DownloadResult, ContentResult, PDFDownloader
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
    "get_pdf_with_fallback",
    "get_content_with_fallback",
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
