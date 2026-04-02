"""PDF text extraction parser.

Prefers PyMuPDF (fitz) for better two-column handling via sorted
text-block extraction.  Falls back to pdfplumber when fitz is
unavailable.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.parsers.pdf")


def _clean_text(text: str, threshold: float = 0.05) -> str:
    """Collapse excess newlines when they dominate the text.

    Scientific PDFs often have many spurious line breaks from column extraction.
    If the ratio of newlines exceeds *threshold*, collapse runs of whitespace
    into single spaces.
    """
    if not text:
        return text
    newline_ratio = text.count("\n") / len(text)
    if newline_ratio > threshold:
        return " ".join(text.replace("\n", " ").split())
    return text


def _fix_fused_words(text: str) -> str:
    """Insert a space where a lowercase letter is immediately followed
    by an uppercase letter and another lowercase — a common artifact
    when PDF columns are merged without proper layout analysis.
    """
    if not text:
        return text
    return re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)


@dataclass
class ParsedContent:
    """Result of parsing a document."""

    text: str
    title: str | None = None
    sections: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


class PDFParser:
    """Parser for PDF documents.

    Uses PyMuPDF (fitz) by default for superior two-column layout
    handling.  Falls back to pdfplumber if fitz is not installed.
    """

    def __init__(self):
        self._fitz = None
        self._pdfplumber = None

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------

    def _get_fitz(self) -> Any:
        if self._fitz is None:
            try:
                import fitz

                self._fitz = fitz
            except ImportError:
                pass
        return self._fitz

    def _get_pdfplumber(self) -> Any:
        if self._pdfplumber is None:
            try:
                import pdfplumber

                self._pdfplumber = pdfplumber
            except ImportError:
                raise ImportError(
                    "Neither PyMuPDF nor pdfplumber is installed. "
                    "Install one with: pip install pymupdf  (recommended) "
                    "or: pip install pdfplumber"
                )
        return self._pdfplumber

    # ------------------------------------------------------------------
    # Extraction backends
    # ------------------------------------------------------------------

    def _extract_with_fitz(self, source: str | Path | bytes) -> tuple[str, dict[str, str], int] | None:
        """Extract text using PyMuPDF sorted-block mode.

        Returns (full_text, sections_dict, page_count) or None on failure.
        """
        fitz = self._get_fitz()
        if fitz is None:
            return None

        try:
            if isinstance(source, (str, Path)):
                doc = fitz.open(str(source))
            else:
                doc = fitz.open(stream=source, filetype="pdf")

            all_text = []
            sections = {}

            for i, page in enumerate(doc):
                # get_text("blocks", sort=True) groups text into spatial
                # blocks and sorts them top-to-bottom, left-to-right,
                # which naturally respects column boundaries.
                blocks = page.get_text("blocks", sort=True)
                page_texts = [
                    b[4].strip()
                    for b in blocks
                    if b[6] == 0  # text blocks only (skip images)
                ]
                text = "\n".join(page_texts)
                if text:
                    all_text.append(text)
                    sections[f"page_{i + 1}"] = text

            page_count = len(doc)
            doc.close()

            return "\n\n".join(all_text), sections, page_count

        except Exception as e:
            logger.warning("fitz_extraction_failed", error=str(e))
            return None

    def _extract_with_pdfplumber(self, source: str | Path | bytes) -> tuple[str, dict[str, str], int]:
        """Extract text using pdfplumber (fallback).

        Returns (full_text, sections_dict, page_count).
        """
        import io

        pdfplumber = self._get_pdfplumber()

        if isinstance(source, (str, Path)):
            pdf = pdfplumber.open(str(source))
        else:
            pdf = pdfplumber.open(io.BytesIO(source))

        all_text = []
        sections = {}

        for i, page in enumerate(pdf.pages):
            text = page.extract_text(
                x_tolerance=1.5,
                y_tolerance=1.5,
                use_text_flow=True,
            )
            if text:
                all_text.append(text)
                sections[f"page_{i + 1}"] = text

        page_count = len(pdf.pages)
        pdf.close()

        return "\n\n".join(all_text), sections, page_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(self, source: str | Path | bytes) -> ParsedContent:
        """
        Parse PDF and extract text.

        Args:
            source: Path to PDF file, or PDF bytes

        Returns:
            Parsed content with text and metadata
        """
        # Try PyMuPDF first (better column handling)
        result = self._extract_with_fitz(source)

        if result is None:
            # Fallback to pdfplumber
            raw_text, sections, page_count = self._extract_with_pdfplumber(source)
            backend = "pdfplumber"
        else:
            raw_text, sections, page_count = result
            backend = "fitz"

        full_text = _fix_fused_words(_clean_text(raw_text))

        logger.info(
            "pdf_parsed",
            pages=page_count,
            text_length=len(full_text),
            backend=backend,
        )

        return ParsedContent(
            text=full_text,
            sections=sections,
            metadata={"pages": page_count, "backend": backend},
        )

    async def parse_file(self, path: Path) -> ParsedContent:
        """Parse PDF from file path."""
        return await self.parse(path)

    async def parse_bytes(self, data: bytes) -> ParsedContent:
        """Parse PDF from bytes."""
        return await self.parse(data)
