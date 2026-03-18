"""PDF text extraction parser."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.parsers.pdf")


@dataclass
class ParsedContent:
    """Result of parsing a document."""

    text: str
    title: str | None = None
    sections: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


class PDFParser:
    """Parser for PDF documents."""

    def __init__(self):
        self._pdfplumber = None

    def _get_pdfplumber(self) -> Any:
        """Lazy import pdfplumber."""
        if self._pdfplumber is None:
            try:
                import pdfplumber

                self._pdfplumber = pdfplumber
            except ImportError:
                raise ImportError(
                    "pdfplumber not installed. "
                    "Install with: pip install pdfplumber"
                )
        return self._pdfplumber

    async def parse(self, source: str | Path | bytes) -> ParsedContent:
        """
        Parse PDF and extract text.

        Args:
            source: Path to PDF file, or PDF bytes

        Returns:
            Parsed content with text and metadata
        """
        pdfplumber = self._get_pdfplumber()

        try:
            if isinstance(source, (str, Path)):
                pdf = pdfplumber.open(str(source))
            else:
                import io

                pdf = pdfplumber.open(io.BytesIO(source))

            all_text = []
            sections = {}

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text.append(text)
                    sections[f"page_{i+1}"] = text

            pdf.close()

            full_text = "\n\n".join(all_text)

            return ParsedContent(
                text=full_text,
                sections=sections,
                metadata={"pages": len(pdf.pages)},
            )

        except Exception as e:
            logger.error("pdf_parse_error", error=str(e))
            raise

    async def parse_file(self, path: Path) -> ParsedContent:
        """Parse PDF from file path."""
        return await self.parse(path)

    async def parse_bytes(self, data: bytes) -> ParsedContent:
        """Parse PDF from bytes."""
        return await self.parse(data)
