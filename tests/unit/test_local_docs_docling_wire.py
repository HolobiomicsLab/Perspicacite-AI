import asyncio
import unittest
from pathlib import Path


class TestReadTextIsFitzTextOnly(unittest.TestCase):
    def test_pdf_returns_text_string(self):
        from perspicacite.integrations.local_docs import _read_text
        from perspicacite.pipeline.parsers.pdf import ParsedContent

        class _FakeParser:
            async def parse(self, source):
                return ParsedContent(text="body text")

        out = asyncio.run(_read_text(Path("/x.pdf"), "pdf", _FakeParser()))
        assert out == "body text"

    def test_pdf_empty_returns_none(self):
        from perspicacite.integrations.local_docs import _read_text
        from perspicacite.pipeline.parsers.pdf import ParsedContent

        class _FakeParser:
            async def parse(self, source):
                return ParsedContent(text="")

        assert asyncio.run(_read_text(Path("/x.pdf"), "pdf", _FakeParser())) is None

    def test_non_pdf_returns_text(self):
        import os
        import tempfile

        from perspicacite.integrations.local_docs import _read_text
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            p = Path(f.name)
        try:
            out = asyncio.run(_read_text(p, "text", None))
            assert "hello world" in out
        finally:
            os.unlink(p)
