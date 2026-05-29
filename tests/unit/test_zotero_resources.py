# tests/unit/test_zotero_resources.py
"""Tests for ResourceLocator."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from perspicacite.integrations.zotero_resources import ResourceLocator


def _fake_config(tmp_path: Path) -> SimpleNamespace:
    cache_dir = tmp_path / "pdfs"
    cache_dir.mkdir()
    capsule_root = tmp_path / "capsules"
    capsule_root.mkdir()
    return SimpleNamespace(
        pdf_download=SimpleNamespace(
            cache_pdfs=True,
            cache_dir=str(cache_dir),
            unpaywall_email="test@example.com",
        ),
        capsule=SimpleNamespace(root=str(capsule_root)),
    )


def test_local_pdf_first_when_cached(tmp_path):
    cfg = _fake_config(tmp_path)
    doi = "10.1234/test"
    # Write a fake cached PDF (cached_pdf_path uses _sanitize_doi: replace / and : with _)
    pdf_path = Path(cfg.pdf_download.cache_dir) / "10.1234_test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4" + b"\x00" * 2000)

    rl = ResourceLocator(cfg)
    zotero_item = {"key": "ABC", "data": {"DOI": doi, "itemType": "journalArticle"}}
    resources = rl.build(doi=doi, zotero_item=zotero_item, attachments=[])

    pdf_resources = [r for r in resources if r["role"] == "fulltext_pdf"]
    assert len(pdf_resources) == 1
    access = pdf_resources[0]["access"]
    assert access[0]["type"] == "local"
    assert "10.1234_test.pdf" in access[0]["path"]


def test_remote_doi_resolver_always_present(tmp_path):
    cfg = _fake_config(tmp_path)
    doi = "10.1234/nopdf"
    rl = ResourceLocator(cfg)
    zotero_item = {"key": "ABC", "data": {"DOI": doi}}
    resources = rl.build(doi=doi, zotero_item=zotero_item, attachments=[])
    pdf_resources = [r for r in resources if r["role"] == "fulltext_pdf"]
    assert len(pdf_resources) == 1
    vias = [a["via"] for a in pdf_resources[0]["access"] if a["type"] == "remote"]
    assert "doi_resolver" in vias


def test_si_files_included_when_on_disk(tmp_path):
    cfg = _fake_config(tmp_path)
    doi = "10.1234/withsi"
    # Capsule sanitization: ':' → '_', '/' → '__' (double underscore)
    si_dir = Path(cfg.capsule.root) / "10.1234__withsi" / "supplementary" / "files"
    si_dir.mkdir(parents=True)
    (si_dir / "table_S1.xlsx").write_bytes(b"fake excel content")

    rl = ResourceLocator(cfg)
    zotero_item = {"key": "ABC", "data": {"DOI": doi}}
    resources = rl.build(doi=doi, zotero_item=zotero_item, attachments=[])
    si_resources = [r for r in resources if r["role"] == "supplementary"]
    assert len(si_resources) == 1
    assert si_resources[0]["filename"] == "table_S1.xlsx"
    assert si_resources[0]["access"][0]["type"] == "local"


def test_publisher_url_from_zotero_attachment(tmp_path):
    cfg = _fake_config(tmp_path)
    doi = "10.1234/pub"
    attachments = [{
        "key": "ATT1",
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "linkMode": "linked_url",
            "url": "https://publisher.com/pdf/article.pdf",
            "title": "article.pdf",
        }
    }]
    rl = ResourceLocator(cfg)
    zotero_item = {"key": "ABC", "data": {"DOI": doi}}
    resources = rl.build(doi=doi, zotero_item=zotero_item, attachments=attachments)
    pdf_resources = [r for r in resources if r["role"] == "fulltext_pdf"]
    vias = [a.get("via") for a in pdf_resources[0]["access"] if a["type"] == "remote"]
    assert "publisher" in vias


def test_no_duplicate_doi_resolver_when_publisher_present(tmp_path):
    cfg = _fake_config(tmp_path)
    doi = "10.1234/dup"
    attachments = [{
        "key": "ATT1",
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "linkMode": "linked_url",
            "url": "https://pub.com/pdf.pdf",
        }
    }]
    rl = ResourceLocator(cfg)
    resources = rl.build(doi=doi, zotero_item={"key": "X", "data": {"DOI": doi}}, attachments=attachments)
    pdf = next(r for r in resources if r["role"] == "fulltext_pdf")
    resolver_count = sum(1 for a in pdf["access"] if a.get("via") == "doi_resolver")
    assert resolver_count == 1  # appears exactly once
