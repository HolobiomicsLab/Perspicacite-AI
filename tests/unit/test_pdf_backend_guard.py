import unittest


class _Cfg:
    def __init__(self, flag=True, max_pages=40, timeout=600):
        self.docling_extract_tables_figures = flag
        self.docling_max_pages = max_pages
        self.docling_timeout_s = timeout


class TestShouldRunDoclingExtras(unittest.TestCase):
    def test_flag_off_returns_false(self):
        from perspicacite.pipeline.parsers.pdf import PDFParser
        assert PDFParser()._should_run_docling_extras(5, _Cfg(flag=False)) is False

    def test_flag_on_importable_small_returns_true(self):
        from perspicacite.pipeline.parsers import pdf as m
        orig = m._docling_importable
        m._docling_importable = lambda: True
        try:
            assert m.PDFParser()._should_run_docling_extras(5, _Cfg()) is True
        finally:
            m._docling_importable = orig

    def test_oversized_returns_false(self):
        from perspicacite.pipeline.parsers import pdf as m
        orig = m._docling_importable
        m._docling_importable = lambda: True
        try:
            assert m.PDFParser()._should_run_docling_extras(999, _Cfg(max_pages=40)) is False
        finally:
            m._docling_importable = orig

    def test_not_importable_returns_false(self):
        from perspicacite.pipeline.parsers import pdf as m
        orig = m._docling_importable
        m._docling_importable = lambda: False
        try:
            assert m.PDFParser()._should_run_docling_extras(5, _Cfg()) is False
        finally:
            m._docling_importable = orig


class TestTimeoutFallback(unittest.TestCase):
    def test_timeout_branch_via_stub(self):
        from concurrent.futures import TimeoutError as FTimeout

        from perspicacite.pipeline.parsers.pdf import PDFParser
        p = PDFParser()

        class _Fut:
            def result(self, timeout): raise FTimeout()

        class _Ex:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def submit(self, *a, **k): return _Fut()

        import concurrent.futures as cf
        orig_ex = cf.ProcessPoolExecutor
        cf.ProcessPoolExecutor = lambda *a, **k: _Ex()
        try:
            assert p._run_docling_with_timeout("/x.pdf", timeout_s=1) is None
        finally:
            cf.ProcessPoolExecutor = orig_ex

    def test_error_branch_returns_none(self):
        from perspicacite.pipeline.parsers.pdf import PDFParser
        p = PDFParser()

        class _Fut:
            def result(self, timeout): raise RuntimeError("boom")

        class _Ex:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def submit(self, *a, **k): return _Fut()

        import concurrent.futures as cf
        orig_ex = cf.ProcessPoolExecutor
        cf.ProcessPoolExecutor = lambda *a, **k: _Ex()
        try:
            assert p._run_docling_with_timeout("/x.pdf", timeout_s=1) is None
        finally:
            cf.ProcessPoolExecutor = orig_ex
