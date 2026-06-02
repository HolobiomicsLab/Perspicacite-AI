import asyncio
import unittest
from pathlib import Path


class _FakeParsed:
    pass


class TestReadTextThreadsConfigAndTables(unittest.TestCase):
    def test_pdf_returns_parsedcontent_with_tables_and_passes_config(self):
        from perspicacite.integrations.local_docs import _read_text
        from perspicacite.pipeline.parsers.docling_pdf import DoclingTable
        from perspicacite.pipeline.parsers.pdf import ParsedContent

        seen = {}

        class _FakeParser:
            async def parse(self, source, config=None):
                seen["config"] = config
                return ParsedContent(
                    text="body text",
                    tables=[DoclingTable(page=1, caption="Table 1.",
                                         markdown="| a |", headers=["a"], rows=[["1"]])],
                )

        sentinel = object()
        out = asyncio.run(_read_text(Path("/x.pdf"), "pdf", _FakeParser(), sentinel))
        assert isinstance(out, ParsedContent)
        assert out.text == "body text"
        assert len(out.tables) == 1
        assert seen["config"] is sentinel  # config threaded to parse()

    def test_pdf_empty_text_returns_none(self):
        from perspicacite.integrations.local_docs import _read_text
        from perspicacite.pipeline.parsers.pdf import ParsedContent

        class _FakeParser:
            async def parse(self, source, config=None):
                return ParsedContent(text="")

        out = asyncio.run(_read_text(Path("/x.pdf"), "pdf", _FakeParser(), None))
        assert out is None

    def test_non_pdf_wraps_text_in_parsedcontent(self):
        import os
        import tempfile

        from perspicacite.integrations.local_docs import _read_text
        from perspicacite.pipeline.parsers.pdf import ParsedContent
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            p = Path(f.name)
        try:
            out = asyncio.run(_read_text(p, "text", None, None))
            assert isinstance(out, ParsedContent)
            assert "hello world" in out.text
        finally:
            os.unlink(p)
