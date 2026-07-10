"""APA .docx export — opt-in, optional-extra, tolerant of Paper objects/dicts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from perspicacite.rag.export.apa_docx_exporter import (
    export_apa_docx,
    format_authors,
    paper_to_apa,
)

_HAS_DOCX = importlib.util.find_spec("docx") is not None


class _Author:
    def __init__(self, name):
        self.name = name


class _Paper:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TestApaFormatting(unittest.TestCase):
    def test_format_authors_variants(self):
        assert format_authors([]) == ""
        assert format_authors([_Author("Smith, J.")]) == "Smith, J."
        assert format_authors([_Author("A"), _Author("B")]) == "A & B"
        assert format_authors([_Author("A"), _Author("B"), _Author("C")]) == "A, B, & C"
        # dicts and plain strings are also accepted
        assert format_authors([{"name": "X"}, "Y"]) == "X & Y"

    def test_paper_to_apa_with_object(self):
        p = _Paper(
            authors=[_Author("Doe, J."), _Author("Roe, R.")],
            year=2021, title="A study of things", journal="Journal of Things",
            doi="https://doi.org/10.1/abc",
        )
        ref = paper_to_apa(p)
        assert "Doe, J. & Roe, R." in ref
        assert "(2021)." in ref
        assert "A study of things." in ref
        assert "Journal of Things." in ref
        assert "https://doi.org/10.1/abc" in ref
        assert "https://doi.org/https://" not in ref  # doi prefix not doubled

    def test_paper_to_apa_with_dict_and_missing_year(self):
        ref = paper_to_apa({"authors": [{"name": "Solo, H."}], "title": "T"})
        assert "Solo, H." in ref and "(n.d.)." in ref and "T." in ref


class TestConfigDefaultOff(unittest.TestCase):
    def test_export_flag_defaults_false(self):
        from perspicacite.config.schema import Config
        agentic = Config().rag_modes.agentic
        assert agentic.export_apa_docx is False
        assert agentic.export_apa_docx_dir == "output"


class TestOrchestratorGate(unittest.TestCase):
    def test_helper_noop_when_disabled(self):
        # Unbound-method call on a duck-typed stub: disabled → returns without writing.
        from perspicacite.rag.agentic.orchestrator import AgenticOrchestrator

        class _Stub:
            export_apa_docx = False
        # Must return None and not raise even though no exporter is reachable.
        assert AgenticOrchestrator._maybe_export_apa_docx(_Stub(), "answer", []) is None


@unittest.skipUnless(_HAS_DOCX, "python-docx ([docx] extra) required")
class TestDocxWrite(unittest.TestCase):
    def test_export_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "m.docx"
            p = _Paper(authors=[_Author("A")], year=2020, title="T", journal="J", doi="10.1/x")
            written = export_apa_docx("Body text here.", [p, p], out)  # dup paper → dedup
            assert written == out
            assert out.exists() and out.stat().st_size > 0
