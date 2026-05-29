"""Tests for the PDF drop-zone REST endpoint (Priority 6)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from perspicacite.web.routers import pdf_dropzone as pdf_dropzone_router
from perspicacite.web.state import app_state


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A FastAPI test client wired with a fake config whose cache_dir
    points at a fresh tmp_path."""
    class _PdfDownload:
        cache_dir = str(tmp_path)
    class _Config:
        pdf_download = _PdfDownload()
    monkeypatch.setattr(app_state, "config", _Config(), raising=False)

    app = FastAPI()
    app.include_router(pdf_dropzone_router.router)
    return TestClient(app), tmp_path


def test_dropzone_accepts_valid_pdf(client):
    client_, tmp_path = client
    pdf_bytes = b"%PDF-1.4\n" + (b"abcdef" * 200)
    r = client_.post(
        "/api/pdf-dropzone",
        data={"doi": "10.1234/foo.bar"},
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["doi"] == "10.1234/foo.bar"
    assert body["stored"] is True
    assert body["size_bytes"] == len(pdf_bytes)
    cache_files = list(tmp_path.glob("*.pdf"))
    assert len(cache_files) == 1
    assert cache_files[0].read_bytes() == pdf_bytes


def test_dropzone_rejects_non_pdf(client):
    client_, _ = client
    r = client_.post(
        "/api/pdf-dropzone",
        data={"doi": "10.1234/x"},
        files={"file": ("evil.html", b"<html></html>", "text/html")},
    )
    assert r.status_code == 400
    assert "%PDF" in r.json()["detail"]


def test_dropzone_rejects_empty(client):
    client_, _ = client
    r = client_.post(
        "/api/pdf-dropzone",
        data={"doi": "10.1234/x"},
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert r.status_code == 400


def test_dropzone_strips_doi_org_prefix(client):
    client_, _tmp_path = client
    pdf_bytes = b"%PDF-1.4\n" + (b"x" * 2000)
    r = client_.post(
        "/api/pdf-dropzone",
        data={"doi": "https://doi.org/10.1234/foo.bar"},
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 200
    assert r.json()["doi"] == "10.1234/foo.bar"


def test_dropzone_check_endpoint(client):
    client_, _ = client
    # GET on missing DOI returns cached=False
    r = client_.get("/api/pdf-dropzone/10.1234/nope")
    assert r.status_code == 200
    assert r.json()["cached"] is False
    # Upload, then GET returns cached=True
    pdf_bytes = b"%PDF-1.4\n" + (b"x" * 2000)
    client_.post("/api/pdf-dropzone",
                  data={"doi": "10.1234/yes"},
                  files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
    r = client_.get("/api/pdf-dropzone/10.1234/yes")
    assert r.json()["cached"] is True
    assert r.json()["size_bytes"] == len(pdf_bytes)
