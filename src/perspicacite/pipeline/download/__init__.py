"""PDF and content download from multiple sources.

This package provides download functionality from various sources:
- Unpaywall (open access; requires contact email)
- arXiv (open access)
- Publisher routes (ACS, RSC, AAAS, Springer, Wiley TDM, …)
- OpenAlex OA PDF URLs and Europe PMC PDFs (no API keys)
- Wiley ``/doi/pdf/`` for typical ``10.1002/`` DOIs without a TDM token
- Elsevier API (structured XML/text, requires API key)
- Alternative endpoints (Sci-Hub mirrors)

Two main functions are provided:

1. get_pdf_with_fallback() - Returns raw PDF bytes
   Best for: Simple PDF downloads
   
   Usage:
       from perspicacite.pipeline.download import get_pdf_with_fallback
       
       pdf_bytes = await get_pdf_with_fallback(
           doi="10.1002/xxx",
           unpaywall_email="user@example.com",
           wiley_tdm_token="...",
       )

2. get_content_with_fallback() - Returns ContentResult (PDF or structured text)
   Best for: Structure-aware chunking, when PDF parsing fails, or when you need
   document structure (sections, headings) preserved.
   
   This function can return structured XML from Elsevier API which is better
   for chunking than raw PDF text extraction.
   
   Usage:
       from perspicacite.pipeline.download import get_content_with_fallback
       
       result = await get_content_with_fallback(
           doi="10.1016/xxx",
           elsevier_api_key="your-key",
       )
       if result.content_type == "text":
           # Structured text from Elsevier XML
           structured_text = result.content
       elif result.content_type == "pdf":
           # PDF bytes from other sources
           pdf_bytes = result.content
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
