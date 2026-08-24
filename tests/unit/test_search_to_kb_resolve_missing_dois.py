"""``--resolve-missing-dois`` end-to-end through search_to_kb and the CLI.

Regression cover for the failure this fixes: a Google Scholar search
returned eight usable papers and ``search-to-kb`` reported
``candidates=0``, because every hit was dropped as ``no_doi`` and
nothing on screen said so. Two things must hold — the opt-in flag lets
those hits through with a verified DOI, and when the flag is *off* the
drop is named out loud instead of looking like an empty result set.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perspicacite.models.papers import Author, Paper, PaperSource
from perspicacite.pipeline import doi_backfill, search_to_kb
from perspicacite.pipeline.search_to_kb import search_filter_and_ingest

SCHOLAR_TITLE = "Non-uniform sampling for fast 2D NMR of complex mixtures"
RESOLVED_DOI = "10.1021/acs.analchem.1c00001"


@pytest.fixture(autouse=True)
def _clean_cache():
    doi_backfill.clear_cache()
    yield
    doi_backfill.clear_cache()


def _scholar_hit() -> Paper:
    """One Google Scholar card: title, authors, year — and no DOI."""
    return Paper(
        id="scholar:deadbeef",
        title=SCHOLAR_TITLE,
        authors=[Author(name="J Dupont")],
        year=2021,
        doi=None,
        source=PaperSource.GOOGLE_SCHOLAR,
    )


@pytest.fixture
def scholar_search(monkeypatch):
    """``run_search`` returns a single DOI-less Google Scholar hit."""

    async def fake_run_search(**kwargs):
        return [_scholar_hit()]

    monkeypatch.setattr(search_to_kb, "run_search", fake_run_search)


@pytest.fixture
def resolver_finds_doi(monkeypatch):
    async def fake_resolve(title, authors, year, *, http_client, enable_browser=False):
        return RESOLVED_DOI if title == SCHOLAR_TITLE else None

    monkeypatch.setattr(
        "perspicacite.pipeline.download.title_resolver.resolve_doi_from_title",
        fake_resolve,
    )


async def test_doi_less_hit_is_dropped_when_the_flag_is_off(scholar_search):
    report = await search_filter_and_ingest(
        app_state=SimpleNamespace(config=None),
        query="fast 2D NMR",
        kb_name="throwaway",
        dry_run=True,
    )

    assert report.searched == 1
    assert report.candidates == 0
    assert report.filter_reasons == {"no_doi": 1}
    assert report.doi_backfill == {}


async def test_flag_recovers_the_hit_with_a_verified_doi(
    scholar_search, resolver_finds_doi
):
    report = await search_filter_and_ingest(
        app_state=SimpleNamespace(config=None),
        query="fast 2D NMR",
        kb_name="throwaway",
        dry_run=True,
        resolve_missing_dois=True,
    )

    assert report.candidates == 1
    assert report.selected_dois == [RESOLVED_DOI]
    assert report.filter_reasons == {}
    assert report.doi_backfill["missing_doi"] == 1
    assert report.doi_backfill["resolved"] == 1


async def test_report_still_names_the_failure_when_resolution_misses(
    scholar_search, monkeypatch
):
    """A miss must stay visible: no DOI *and* a non-empty backfill block."""

    async def never_resolves(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "perspicacite.pipeline.download.title_resolver.resolve_doi_from_title",
        never_resolves,
    )

    report = await search_filter_and_ingest(
        app_state=SimpleNamespace(config=None),
        query="fast 2D NMR",
        kb_name="throwaway",
        dry_run=True,
        resolve_missing_dois=True,
    )

    assert report.candidates == 0
    assert report.filter_reasons == {"no_doi": 1}
    assert report.doi_backfill["attempted"] == 1
    assert report.doi_backfill["unresolved"] == 1


def _run_cli(*extra_args):
    """Drive ``search-to-kb`` with AppState and the pipeline stubbed out.

    The AppState stub is not optional: the real one opens the shared
    ChromaDB store, which is single-writer and usually already held by
    a running server. ``initialized`` proves the stub was the one used.
    """
    from click.testing import CliRunner

    from perspicacite.cli import cli

    initialized: list[bool] = []

    class FakeAppState:
        async def initialize(self):
            initialized.append(True)

    with (
        patch("perspicacite.web.state.AppState", FakeAppState),
        patch(
            "perspicacite.pipeline.search_to_kb.search_filter_and_ingest",
            new=_fake_ingest,
        ),
    ):
        result = CliRunner().invoke(
            cli,
            ["search-to-kb", "-q", "fast 2D NMR", "-k", "throwaway", "--dry-run", *extra_args],
        )
    assert initialized == [True], "the real AppState ran; it would touch live ChromaDB"
    return result


async def _fake_ingest(**kwargs):
    """Report shaped like a Google-Scholar-only run: 8 hits, all DOI-less."""
    report = search_to_kb.IngestReport(query=kwargs["query"], kb_name=kwargs["kb_name"])
    report.searched = 8
    if kwargs.get("resolve_missing_dois"):
        report.candidates = 5
        report.filtered_out = 3
        report.filter_reasons = {"no_doi": 3}
        report.doi_backfill = {
            "missing_doi": 8,
            "attempted": 8,
            "resolved": 5,
            "unresolved": 3,
            "no_title": 0,
            "over_budget": 0,
            "cache_hits": 0,
        }
        report.selected_dois = [f"10.1000/hit{i}" for i in range(5)]
    else:
        report.filtered_out = 8
        report.filter_reasons = {"no_doi": 8}
    return report


def test_cli_names_the_fix_when_hits_are_dropped_for_missing_dois():
    result = _run_cli()

    assert result.exit_code == 0
    assert "no_doi=8" in result.output
    assert "--resolve-missing-dois" in result.output


def test_cli_reports_the_backfill_and_drops_the_hint_when_the_flag_is_on():
    result = _run_cli("--resolve-missing-dois")

    assert result.exit_code == 0
    assert "DOI backfill: resolved 5/8 attempted" in result.output
    # The suggestion is for people who haven't used the flag yet.
    assert "Re-run with --resolve-missing-dois" not in result.output
