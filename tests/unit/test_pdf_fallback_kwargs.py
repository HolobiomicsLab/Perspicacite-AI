"""The DOI-ingest route must not silently drop download capabilities.

``POST /api/kb/{name}/dois`` builds its ``retrieve_paper_content`` kwargs
through ``_get_pdf_fallback_kwargs``. When that helper omitted
``cookies_path``, ``cookie_domains``, ``elsevier_api_key`` and
``pdf_cache_dir``, bulk DOI ingest ran without institutional access and
threw every downloaded PDF away — while the search-to-KB path, calling the
same pipeline, kept all four.
"""
import inspect

from perspicacite.config.schema import PDFDownloadConfig
from perspicacite.pipeline.download import retrieve_paper_content
from perspicacite.web.routers.kb import _get_pdf_fallback_kwargs


def _config(**overrides) -> PDFDownloadConfig:
    base = dict(
        unpaywall_email="a@b.c",
        alternative_endpoint="https://example.org/",
        wiley_tdm_token="w", elsevier_api_key="e", aaas_api_key="a",
        rsc_api_key="r", springer_api_key="s",
        cookies_path="/tmp/cookies.txt",
        cookie_domains=["nature.com"],
    )
    base.update(overrides)
    return PDFDownloadConfig(**base)


def test_no_shared_field_is_dropped():
    """Every config field the download pipeline accepts must be forwarded.

    Guards the whole class of bug rather than the four known instances:
    add a credential to both sides and forget the router, and this fails.
    """
    accepted = set(inspect.signature(retrieve_paper_content).parameters)
    shared = accepted & set(PDFDownloadConfig.model_fields)
    passed = set(_get_pdf_fallback_kwargs(_config()))
    missing = shared - passed
    assert not missing, f"router drops {sorted(missing)}"


def test_cookie_jar_and_domains_are_forwarded():
    kwargs = _get_pdf_fallback_kwargs(_config())
    assert kwargs["cookies_path"] == "/tmp/cookies.txt"
    assert kwargs["cookie_domains"] == ["nature.com"]


def test_elsevier_key_is_forwarded():
    assert _get_pdf_fallback_kwargs(_config())["elsevier_api_key"] == "e"


def test_pdf_cache_dir_follows_the_cache_pdfs_flag():
    """A cached PDF is what Zotero push and export-kb later attach."""
    on = _get_pdf_fallback_kwargs(_config(cache_pdfs=True, cache_dir="data/papers"))
    assert on["pdf_cache_dir"] == "data/papers"
    off = _get_pdf_fallback_kwargs(_config(cache_pdfs=False))
    assert "pdf_cache_dir" not in off


def test_absent_config_yields_no_kwargs():
    assert _get_pdf_fallback_kwargs(None) == {}


if __name__ == "__main__":
    test_no_shared_field_is_dropped()
    test_cookie_jar_and_domains_are_forwarded()
    test_elsevier_key_is_forwarded()
    test_pdf_cache_dir_follows_the_cache_pdfs_flag()
    test_absent_config_yields_no_kwargs()
    print("ok")
