"""Document parsers for various formats."""

from perspicacite.pipeline.parsers.pdf import PDFParser
from perspicacite.pipeline.parsers.html import HTMLParser

__all__ = ["PDFParser", "HTMLParser"]
