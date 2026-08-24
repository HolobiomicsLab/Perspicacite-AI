"""Honest exits, ranked OA candidates and throttling signal in unified.py.

Each behaviour is tested two-sided: one case that must fire on a real
positive, and one genuine negative that superficially resembles it and
must stay clean.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from perspicacite.pipeline.download import unified
from perspicacite.pipeline.download.base import (
    CONTENT_TYPE_ABSTRACT,
    CONTENT_TYPE_NONE,
    MIN_PLAUSIBLE_PDF_BYTES,
    PaperContent,
    PaperDiscovery,
)
from perspicacite.pipeline.download.pdf_cache import get_cached_pdf, store_pdf
from perspicacite.pipeline.download.rate_limit import RateLimited

# A body over the plausibility floor, so only the content decides.
PDF_BODY = b"%PDF-1.7 " + b"x" * MIN_PLAUSIBLE_PDF_BYTES
PAYWALL_BODY = b"<!DOCTYPE html><html>Get access</html>" + b" " * MIN_PLAUSIBLE_PDF_BYTES
# Same tokens as the paywall page, but inside a real PDF's metadata.
PDF_WITH_HTML_TOKEN = b"%PDF-1.7 <html>" + b"y" * MIN_PLAUSIBLE_PDF_BYTES
BOT_WALL_BODY = b"<html>cloudPMC-viewer-pow</html>" + b" " * MIN_PLAUSIBLE_PDF_BYTES
# One byte under the floor: a stub response, never a paper.
UNDERSIZED_BODY = b"%PDF-1.7 " + b"x" * (MIN_PLAUSIBLE_PDF_BYTES - 20)


class _StubParser:
    """PDF parser stub; returns fixed text so no PDF engine is loaded."""

    class _Parsed:
        text = "extracted body text " * 40

    async def parse(self, _data: bytes) -> _StubParser._Parsed:
        """Return the fixed parse result for any bytes."""
        return self._Parsed()


class _FakeDownloader:
    """PDFDownloader stub driven by a {url: bytes | Exception} table."""

    table: ClassVar[dict[str, Any]] = {}
    calls: ClassVar[list[str]] = []

    async def download(self, url: str, http_client: Any = None) -> bytes | None:
        """Return the table's entry for `url`, raising it when it is an error."""
        self.calls.append(url)
        outcome = self.table.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_downloader(monkeypatch):
    """Install the stub downloader and reset its table between tests."""
    _FakeDownloader.table = {}
    _FakeDownloader.calls = []
    monkeypatch.setattr(unified, "PDFDownloader", _FakeDownloader)
    return _FakeDownloader


# ── REGION 1: honest exits ────────────────────────────────────────────────


class TestParsePdfBytes:
    """_parse_pdf_bytes must not turn a wall page into 'full text'."""

    @pytest.mark.asyncio
    async def test_rejects_paywall_html_body(self):
        """A publisher landing page is not extractable text (positive)."""
        assert await unified._parse_pdf_bytes(PAYWALL_BODY, _StubParser()) is None

    @pytest.mark.asyncio
    async def test_accepts_pdf_carrying_html_token(self):
        """A real PDF whose metadata mentions <html> still parses (negative)."""
        text = await unified._parse_pdf_bytes(PDF_WITH_HTML_TOKEN, _StubParser())
        assert text == _StubParser._Parsed.text

    @pytest.mark.asyncio
    async def test_rejects_bot_wall_body(self):
        """A proof-of-work challenge page is rejected, not stored."""
        assert await unified._parse_pdf_bytes(BOT_WALL_BODY, _StubParser()) is None

    @pytest.mark.asyncio
    async def test_rejects_body_under_plausible_floor(self):
        """A body under MIN_PLAUSIBLE_PDF_BYTES never reaches the parser."""
        assert await unified._parse_pdf_bytes(UNDERSIZED_BODY, _StubParser()) is None


class TestClientLifecycle:
    """A failure building the client must not raise an unbound name."""

    @pytest.mark.asyncio
    async def test_client_build_failure_returns_none_result(self, monkeypatch):
        """Construction blowing up degrades to a none-result, not NameError."""

        def _boom(_cookies_path: str | None) -> Any:
            raise RuntimeError("cookie jar unreadable")

        monkeypatch.setattr(unified, "_build_owned_client", _boom)
        result = await unified.retrieve_paper_content("10.1/x")
        assert result.success is False
        assert result.content_type == CONTENT_TYPE_NONE


# ── REGION 2: ranked OA candidates ────────────────────────────────────────


def _discovery(**kwargs: Any) -> PaperDiscovery:
    """PaperDiscovery with only the fields a PDF tier reads."""
    base: dict[str, Any] = {"doi": "10.1/x", "oa_url": None, "unpaywall_pdf_url": None}
    base.update(kwargs)
    return PaperDiscovery(**base)


class TestOaCandidateTier:
    """Tier 3a walks every ranked candidate, not just the first url."""

    @pytest.mark.asyncio
    async def test_falls_through_to_second_candidate(self, fake_downloader):
        """A dead best candidate must not end the OA tier."""
        fake_downloader.table = {"https://a/x.pdf": None, "https://b/x.pdf": PDF_BODY}
        attempts: list[dict[str, Any]] = []
        disc = _discovery(oa_candidates=["https://a/x.pdf", "https://b/x.pdf"])

        result = await unified._try_pdf_sources(
            "10.1/x", None, None, disc, attempts=attempts
        )

        assert result == (PDF_BODY, "publisher_oa_pdf")
        assert fake_downloader.calls == ["https://a/x.pdf", "https://b/x.pdf"]
        assert attempts == [
            {"source": "publisher_oa_pdf", "status": "miss", "url": "https://a/x.pdf"}
        ]

    @pytest.mark.asyncio
    async def test_falls_back_to_single_oa_url(self, fake_downloader):
        """A discovery without the ranked list still tries oa_url."""
        fake_downloader.table = {"https://legacy/x.pdf": PDF_BODY}
        disc = _discovery(oa_url="https://legacy/x.pdf", oa_candidates=[])

        result = await unified._try_pdf_sources("10.1/x", None, None, disc)

        assert result == (PDF_BODY, "publisher_oa_pdf")

    @pytest.mark.asyncio
    async def test_no_openalex_refetch_tier(self, fake_downloader):
        """Discovery already ranked OpenAlex urls; no extra tier re-fetches them."""
        assert not hasattr(unified, "download_pdf_from_openalex_oa")
        result = await unified._try_pdf_sources("10.1/x", None, None, _discovery())
        assert result is None
        assert fake_downloader.calls == []


# ── REGION 3: throttling signal ───────────────────────────────────────────


class TestThrottlingSignal:
    """A 429 is recorded as rate_limited, never as a clean miss."""

    @pytest.mark.asyncio
    async def test_rate_limited_candidate_is_recorded_with_host(self, fake_downloader):
        """A throttled OA candidate produces a rate_limited attempt (positive)."""
        fake_downloader.table = {
            "https://www.biorxiv.org/x.pdf": RateLimited(
                "https://www.biorxiv.org/x.pdf", 3, None
            )
        }
        attempts: list[dict[str, Any]] = []
        disc = _discovery(oa_candidates=["https://www.biorxiv.org/x.pdf"])

        result = await unified._try_pdf_sources(
            "10.1/x", None, None, disc, attempts=attempts
        )

        assert result is None
        assert attempts[0]["status"] == "rate_limited"
        assert attempts[0]["host"] == "biorxiv.org"
        assert unified._throttled_hosts(attempts) == ["biorxiv.org"]

    @pytest.mark.asyncio
    async def test_empty_response_stays_a_miss(self, fake_downloader):
        """A host that answered with nothing is a miss, not throttling (negative)."""
        fake_downloader.table = {"https://ok.example/x.pdf": None}
        attempts: list[dict[str, Any]] = []
        disc = _discovery(oa_candidates=["https://ok.example/x.pdf"])

        await unified._try_pdf_sources("10.1/x", None, None, disc, attempts=attempts)

        assert attempts[0]["status"] == "miss"
        assert unified._throttled_hosts(attempts) == []

    @pytest.mark.asyncio
    async def test_biorxiv_throttling_reaches_the_caller(self, monkeypatch):
        """A bioRxiv 429 surfaces on the returned PaperContent."""
        doi = "10.1101/2021.01.01.425001"

        async def _disc(*_a: Any, **_k: Any) -> PaperDiscovery:
            return _discovery(doi=doi, abstract=None)

        async def _biorxiv(*_a: Any, **_k: Any) -> PaperContent:
            return PaperContent(
                success=True,
                doi=doi,
                content_type=CONTENT_TYPE_ABSTRACT,
                abstract="A bioRxiv abstract long enough to be kept.",
                content_source="biorxiv",
                rate_limited_hosts=["biorxiv.org"],
            )

        monkeypatch.setattr(unified, "discover_paper_sources", _disc)
        monkeypatch.setattr(unified, "get_content_from_biorxiv", _biorxiv)
        monkeypatch.setattr(unified, "get_fulltext_from_pmc", _no_pmc)
        monkeypatch.setattr(unified, "get_content_from_europepmc", _no_epmc)

        result = await unified.retrieve_paper_content(doi)

        assert result.content_type == CONTENT_TYPE_ABSTRACT
        assert result.rate_limited_hosts == ["biorxiv.org"]
        assert any(a["status"] == "rate_limited" for a in result.attempts)

    @pytest.mark.asyncio
    async def test_untroubled_paper_reports_no_throttling(self, monkeypatch):
        """The same path with no 429 leaves rate_limited_hosts empty (negative)."""
        doi = "10.1101/2021.01.01.425002"

        async def _disc(*_a: Any, **_k: Any) -> PaperDiscovery:
            return _discovery(doi=doi, abstract=None)

        async def _biorxiv(*_a: Any, **_k: Any) -> PaperContent:
            return PaperContent(
                success=True,
                doi=doi,
                content_type=CONTENT_TYPE_ABSTRACT,
                abstract="A bioRxiv abstract long enough to be kept.",
                content_source="biorxiv",
            )

        monkeypatch.setattr(unified, "discover_paper_sources", _disc)
        monkeypatch.setattr(unified, "get_content_from_biorxiv", _biorxiv)
        monkeypatch.setattr(unified, "get_fulltext_from_pmc", _no_pmc)
        monkeypatch.setattr(unified, "get_content_from_europepmc", _no_epmc)

        result = await unified.retrieve_paper_content(doi)

        assert result.rate_limited_hosts == []


async def _no_pmc(*_a: Any, **_k: Any) -> tuple[None, None]:
    """PMC stub: this DOI is not in PMC."""
    return None, None


async def _no_epmc(*_a: Any, **_k: Any) -> None:
    """Europe PMC stub: no record for this DOI."""
    return None


# ── pdf_cache floor ───────────────────────────────────────────────────────


class TestCacheFloor:
    """The cache must share the downloader's plausibility floor."""

    def test_undersized_body_is_not_cached(self, tmp_path):
        """A body the downloader rejects can never become a cache hit."""
        assert store_pdf("10.1/small", UNDERSIZED_BODY, tmp_path) is None
        assert get_cached_pdf("10.1/small", tmp_path) is None

    def test_plausible_body_is_cached(self, tmp_path):
        """A body at or over the floor still round-trips (negative)."""
        assert store_pdf("10.1/ok", PDF_BODY, tmp_path) is not None
        assert get_cached_pdf("10.1/ok", tmp_path) == PDF_BODY
