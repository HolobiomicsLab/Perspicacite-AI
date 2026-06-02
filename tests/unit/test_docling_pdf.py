import unittest


class TestRecordsAndParsedContent(unittest.TestCase):
    def test_parsed_content_defaults_empty_tables_figures(self):
        from perspicacite.pipeline.parsers.pdf import ParsedContent
        pc = ParsedContent(text="hi")
        assert pc.tables == []
        assert pc.figures == []

    def test_record_dataclasses_construct(self):
        from perspicacite.pipeline.parsers.docling_pdf import DoclingTable, DoclingFigure
        t = DoclingTable(page=2, caption="Table 1.", markdown="| a |", headers=["a"], rows=[["1"]])
        assert t.n_rows == 1 and t.n_cols == 1
        f = DoclingFigure(page=1, caption="Figure 1.", width_px=300, height_px=300, image_bytes=b"x")
        assert f.width_px == 300
