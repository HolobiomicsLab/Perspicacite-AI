"""PDF and content download from multiple sources.

This package provides download functionality from various sources:
- Unpaywall (open access)
- arXiv (open access)
- Publisher APIs (Wiley, Elsevier, AAAS, ACS, RSC, etc.)
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

__all__ = [
    "get_pdf_with_fallback",
    "get_content_with_fallback",
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
]
