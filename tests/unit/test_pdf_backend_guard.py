import unittest


class _Cfg:
    def __init__(self, backend="auto", max_pages=40, timeout=120):
        self.pdf_backend = backend
        self.docling_max_pages = max_pages
        self.docling_timeout_s = timeout


class TestBackendSelector(unittest.TestCase):
    def _select(self, parser, pages, cfg=None):
        return parser._select_backend("/x.pdf", pages, _Cfg(**(cfg or {})))

    def test_explicit_fitz(self):
        from perspicacite.pipeline.parsers.pdf import PDFParser
        p = PDFParser()
        assert self._select(p, 5, {"backend": "fitz"}) == "fitz"

    def test_explicit_docling(self):
        from perspicacite.pipeline.parsers.pdf import PDFParser
        p = PDFParser()
        assert self._select(p, 5, {"backend": "docling"}) == "docling"

    def test_auto_uses_fitz_when_docling_absent(self):
        from perspicacite.pipeline.parsers import pdf as m
        p = m.PDFParser()
        orig = m._docling_importable
        m._docling_importable = lambda: False
        try:
            assert self._select(p, 5) == "fitz"
        finally:
            m._docling_importable = orig

    def test_auto_guard_on_pages(self):
        from perspicacite.pipeline.parsers import pdf as m
        p = m.PDFParser()
        orig = m._docling_importable
        m._docling_importable = lambda: True
        try:
            assert self._select(p, 999, {"max_pages": 40}) == "fitz"
            assert self._select(p, 10, {"max_pages": 40}) == "docling"
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
