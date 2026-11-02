"""Offline tests for discovery-cache schema, expiry, and safe fallback."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from perspicacite.pipeline.download import discovery
from perspicacite.pipeline.download.base import PaperDiscovery


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.calls: list[str] = []

    async def get(self, url: str, **_kwargs) -> FakeResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected live lookup: {url}")
        return self.responses.pop(0)


def _openalex_payload(candidate_url: str) -> dict:
    return {
        "title": "Fresh title",
        "type": "article",
        "authorships": [],
        "publication_year": 2026,
        "ids": {},
        "locations": [{"pdf_url": candidate_url}],
        "open_access": {"is_oa": True, "oa_url": None},
        "abstract_inverted_index": None,
    }


def _write_record(tmp_path, doi: str, record: dict) -> None:
    safe = doi.replace("/", "_").replace(":", "-")
    (tmp_path / f"{safe}_discovery.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_current_record_is_a_hit_without_live_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
    disc = PaperDiscovery(
        doi="10.1234/fresh",
        title="Cached title",
        publisher_url="https://journal.example.org/article",
        oa_candidates=["https://repository.example.org/article.pdf"],
    )
    discovery._write_discovery_cache(disc)

    record = json.loads(discovery._discovery_cache_path(disc.doi).read_text())
    assert isinstance(record["schema_version"], int)
    assert not isinstance(record["schema_version"], bool)
    assert record["schema_version"] == discovery.DISCOVERY_CACHE_SCHEMA_VERSION
    fetched_at = discovery._parse_utc_timestamp(record["fetched_at"])
    assert fetched_at is not None
    assert fetched_at.utcoffset() == timedelta(0)

    client = FakeClient()
    loaded = await discovery.discover_paper_sources(disc.doi, client)
    assert client.calls == []
    assert loaded.oa_candidates == disc.oa_candidates
    assert loaded.publisher_url == disc.publisher_url


@pytest.mark.asyncio
async def test_unversioned_record_is_bypassed_for_live_candidates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    doi = "10.1234/legacy"
    _write_record(
        tmp_path,
        doi,
        {
            "doi": doi,
            "title": "Legacy title",
            "oa_url": "https://stale.example.org/article",
            "oa_candidates": ["https://stale.example.org/article.pdf"],
        },
    )
    fresh_url = "https://fresh.example.org/article.pdf"
    client = FakeClient(FakeResponse(200, _openalex_payload(fresh_url)))

    result = await discovery.discover_paper_sources(doi, client)

    assert len(client.calls) == 1
    assert result.oa_candidates == [fresh_url]
    assert result.oa_url is None


@pytest.mark.asyncio
async def test_custom_max_age_bypasses_an_expired_record(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    doi = "10.1234/expired"
    _write_record(
        tmp_path,
        doi,
        {
            "schema_version": discovery.DISCOVERY_CACHE_SCHEMA_VERSION,
            "fetched_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            "doi": doi,
            "oa_candidates": ["https://stale.example.org/article.pdf"],
        },
    )
    fresh_url = "https://fresh.example.org/article.pdf"
    client = FakeClient(FakeResponse(200, _openalex_payload(fresh_url)))

    result = await discovery.discover_paper_sources(
        doi, client, cache_max_age=timedelta(hours=1)
    )

    assert len(client.calls) == 1
    assert result.oa_candidates == [fresh_url]


@pytest.mark.parametrize(
    "record_update",
    [
        {},
        {"fetched_at": "not-a-timestamp"},
        {"fetched_at": "2026-01-01T00:00:00"},
        {"fetched_at": "2026-01-01T01:00:00+01:00"},
    ],
)
def test_missing_or_invalid_timestamp_is_a_miss(
    tmp_path, monkeypatch, record_update
):
    monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
    doi = "10.1234/invalid-time"
    record = {
        "schema_version": discovery.DISCOVERY_CACHE_SCHEMA_VERSION,
        "doi": doi,
    }
    record.update(record_update)
    _write_record(tmp_path, doi, record)
    assert discovery._read_discovery_cache(doi) is None


def test_exact_max_age_is_fresh_but_older_is_expired():
    now = datetime(2026, 11, 1, tzinfo=UTC)
    max_age = discovery.DISCOVERY_CACHE_MAX_AGE
    record = {
        "schema_version": discovery.DISCOVERY_CACHE_SCHEMA_VERSION,
        "fetched_at": (now - max_age).isoformat(),
    }
    assert discovery._is_discovery_cache_fresh(record, max_age, now=now)

    record["fetched_at"] = (now - max_age - timedelta(microseconds=1)).isoformat()
    assert not discovery._is_discovery_cache_fresh(record, max_age, now=now)


@pytest.mark.asyncio
async def test_old_metadata_survives_only_when_live_lookup_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    doi = "10.1234/failure-fallback"
    record = {
        "doi": doi,
        "title": "Legacy title",
        "abstract": "Legacy abstract",
        "pmcid": "PMC123",
        "arxiv_id": "1234.56789",
        "oa_url": "https://stale.example.org/article",
        "publisher_url": "https://stale-publisher.example.org/article",
        "oa_candidates": ["https://stale.example.org/article.pdf"],
    }
    _write_record(tmp_path, doi, record)
    cache_path = discovery._discovery_cache_path(doi)
    original = cache_path.read_text()

    result = await discovery.discover_paper_sources(
        doi, FakeClient(FakeResponse(503))
    )

    assert result.title == "Legacy title"
    assert result.abstract == "Legacy abstract"
    assert result.pmcid is None
    assert result.arxiv_id is None
    assert result.oa_url is None
    assert result.publisher_url is None
    assert result.oa_candidates == []
    assert cache_path.read_text() == original


@pytest.mark.asyncio
async def test_successful_live_lookup_does_not_seed_old_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    doi = "10.1234/live-success"
    _write_record(
        tmp_path,
        doi,
        {"doi": doi, "title": "Legacy title", "abstract": "Legacy abstract"},
    )
    payload = _openalex_payload("https://fresh.example.org/article.pdf")
    payload["title"] = None
    client = FakeClient(FakeResponse(200, payload))

    result = await discovery.discover_paper_sources(doi, client)

    assert result.title is None
    assert result.abstract is None
