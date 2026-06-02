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


class _FakeProv:
    def __init__(self, page_no): self.page_no = page_no

class _FakeImg:
    def __init__(self, png): self._png = png; self.width = 300; self.height = 300
    def save(self, buf, fmt): buf.write(self._png)

class _FakePicture:
    def __init__(self, page, caption, png):
        self.prov = [_FakeProv(page)]; self._caption = caption; self._png = png
    def caption_text(self, doc): return self._caption
    def get_image(self, doc): return _FakeImg(self._png)

class _FakeTable:
    def __init__(self, page, caption, headers, rows):
        self.prov = [_FakeProv(page)]; self._caption = caption
        self._headers = headers; self._rows = rows
    def caption_text(self, doc): return self._caption
    def export_to_markdown(self, doc=None): return "| " + " | ".join(self._headers) + " |"
    def export_to_dataframe(self, doc=None):
        import pandas as pd
        return pd.DataFrame(self._rows, columns=self._headers)

class _FakeDoc:
    def __init__(self, pictures, tables): self.pictures = pictures; self.tables = tables

class _FakeResult:
    def __init__(self, doc): self.document = doc

class _FakeConverter:
    def __init__(self, doc): self._doc = doc
    def convert(self, source): return _FakeResult(self._doc)


class TestDoclingExtraction(unittest.TestCase):
    def test_maps_pictures_and_tables_dims_populated(self):
        import importlib.util
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("pandas required")
        from perspicacite.pipeline.parsers import docling_pdf as d
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
        doc = _FakeDoc(
            pictures=[_FakePicture(1, "Figure 1.", png)],
            tables=[_FakeTable(2, "Table 1.", ["k", "v"], [["a", "1"]])],
        )
        parser = d.DoclingPDFParser(converter_factory=lambda: _FakeConverter(doc))
        res = parser.extract("/x.pdf")
        assert len(res.figures) == 1
        assert res.figures[0].width_px == 300 and res.figures[0].height_px == 300
        assert len(res.tables) == 1
        assert res.tables[0].headers == ["k", "v"] and res.tables[0].rows == [["a", "1"]]
        assert "k" in res.tables[0].markdown


class TestDoclingConverterConfig(unittest.TestCase):
    def test_converter_enables_picture_images(self):
        import importlib.util
        if importlib.util.find_spec("docling") is None:
            self.skipTest("docling extra required")
        from perspicacite.pipeline.parsers.docling_pdf import _make_docling_converter
        from docling.datamodel.base_models import InputFormat
        conv = _make_docling_converter()
        opts = conv.format_to_options[InputFormat.PDF].pipeline_options
        assert opts.generate_picture_images is True
        assert opts.images_scale >= 2.0


class TestFigureToMultimodalShape(unittest.TestCase):
    def test_figure_maps_to_kind_caption_content(self):
        from perspicacite.pipeline.parsers.docling_pdf import (
            DoclingFigure, figure_to_multimodal_record,
        )
        f = DoclingFigure(page=1, caption="Figure 2. Workflow.",
                          width_px=400, height_px=300, image_bytes=b"x")
        rec = figure_to_multimodal_record(f)
        assert rec["kind"] == "figure"
        assert rec["caption"] == "Figure 2. Workflow."
        assert rec["label"] == "Figure 2"
        assert "content" in rec

    def test_figure_without_label_caption(self):
        from perspicacite.pipeline.parsers.docling_pdf import (
            DoclingFigure, figure_to_multimodal_record,
        )
        f = DoclingFigure(page=1, caption="An unlabeled panel", width_px=400, height_px=300)
        rec = figure_to_multimodal_record(f)
        assert rec["kind"] == "figure"
        assert rec["label"] == ""
