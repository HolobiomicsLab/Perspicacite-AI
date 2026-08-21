"""A BibTeX ``file`` field must not read PDFs outside local_docs.allowed_roots.

The path arrives inside an uploaded ``.bib``. Unguarded, ``file = {/…/x.pdf}``
makes the server parse that file and store its text as a KB chunk, which the
uploader can then read back through search — an arbitrary-PDF read dressed up
as an import.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from perspicacite.web.routers.kb import _validated_local_pdf


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """Point local_docs.allowed_roots at a sandbox and return it."""
    from perspicacite.web.state import app_state

    root = tmp_path / "allowed"
    root.mkdir()

    def _set(roots):
        monkeypatch.setattr(
            app_state, "config",
            SimpleNamespace(local_docs=SimpleNamespace(allowed_roots=roots)),
            raising=False,
        )

    return SimpleNamespace(root=root, outside=tmp_path / "outside", set=_set)


def _make_pdf(directory: Path, name: str = "paper.pdf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def test_pdf_inside_an_allowed_root_is_accepted(configured):
    allowed = _make_pdf(configured.root)
    configured.set([configured.root])
    assert _validated_local_pdf(str(allowed)) == allowed.resolve()


def test_pdf_outside_every_allowed_root_is_refused(configured):
    """The file exists and is a real PDF — only its location disqualifies it."""
    secret = _make_pdf(configured.outside, "private.pdf")
    configured.set([configured.root])
    assert secret.exists()
    assert _validated_local_pdf(str(secret)) is None


def test_traversal_out_of_an_allowed_root_is_refused(configured):
    secret = _make_pdf(configured.outside, "private.pdf")
    configured.set([configured.root])
    escape = f"{configured.root}/../outside/private.pdf"
    assert Path(escape).exists()  # the path does resolve to the secret
    assert _validated_local_pdf(escape) is None


def test_everything_is_refused_when_no_allow_list_is_configured(configured):
    """Empty allowed_roots disables server-side path ingest entirely."""
    anywhere = _make_pdf(configured.root)
    configured.set([])
    assert _validated_local_pdf(str(anywhere)) is None


def test_absent_and_non_pdf_paths_are_ignored(configured):
    configured.set([configured.root])
    assert _validated_local_pdf(None) is None
    assert _validated_local_pdf("") is None
    assert _validated_local_pdf("/etc/passwd") is None
