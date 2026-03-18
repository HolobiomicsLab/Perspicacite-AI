"""HTML text extraction parser."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.parsers.html")


@dataclass
class ParsedContent:
    """Result of parsing a document."""

    text: str
    title: str | None = None
    sections: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


class HTMLParser:
    """Parser for HTML documents."""

    def __init__(self):
        self._bs4 = None

    def _get_bs4(self) -> Any:
        """Lazy import BeautifulSoup."""
        if self._bs4 is None:
            try:
                from bs4 import BeautifulSoup

                self._bs4 = BeautifulSoup
            except ImportError:
                raise ImportError(
                    "beautifulsoup4 not installed. "
                    "Install with: pip install beautifulsoup4"
                )
        return self._bs4

    async def parse(self, source: str | Path) -> ParsedContent:
        """
        Parse HTML and extract text.

        Args:
            source: Path to HTML file, or HTML string

        Returns:
            Parsed content with text and metadata
        """
        BeautifulSoup = self._get_bs4()

        try:
            if isinstance(source, Path) or (
                isinstance(source, str) and Path(source).exists()
            ):
                with open(source, "r", encoding="utf-8") as f:
                    html = f.read()
            else:
                html = source

            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title = None
            if soup.title:
                title = soup.title.get_text(strip=True)
            elif soup.h1:
                title = soup.h1.get_text(strip=True)

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract main content (prefer article or main tags)
            main_content = soup.find("article") or soup.find("main")
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return ParsedContent(
                text=text,
                title=title,
                metadata={"url": None},  # Could extract from meta tags
            )

        except Exception as e:
            logger.error("html_parse_error", error=str(e))
            raise

    async def parse_url(self, url: str, http_client: Any) -> ParsedContent:
        """Fetch and parse HTML from URL."""
        response = await http_client.get(url)
        response.raise_for_status()
        return await self.parse(response.text)
