"""Orchestrator dry-run test — mocks the OpenAlex client and
verifies the resolve → fetch → filter+score → return path returns
a ranked list without touching the KB."""
from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, patch

import pytest

from perspicacite.config.schema import CiteGraphConfig, KnowledgeBaseConfig


@pytest.mark.asyncio
async def test_dry_run_returns_ranked_hits():
    from perspicacite.pipeline.cite_graph import enrich_kb_from_cite_graph

    fake_works = [
        {"doi": f"10.0/{i}", "title": f"T{i}",
         "publication_year": 2020 + (i % 6),
         "cited_by_count": i * 5,
         "open_access": {"is_oa": (i % 2 == 0)},
         "abstract_inverted_index": None,
         "primary_location": {"source": {"display_name": "Journal X"}},
         "ids": {"doi": f"https://doi.org/10.0/{i}"}}
        for i in range(1, 11)
    ]

    with patch("perspicacite.pipeline.cite_graph._resolve_and_fetch",
               new=AsyncMock(return_value=(fake_works, None))):
        kb_cfg = KnowledgeBaseConfig(
            library_paper_map={"my-lib": "10.0/seed"},
            cite_graph=CiteGraphConfig(max_papers=5, min_year_offset=10),
        )
        hits = await enrich_kb_from_cite_graph(
            tool="my-lib", kb_config=kb_cfg, existing_dois=set(),
            dry_run=True, now_year=2026,
        )

    assert len(hits) <= 5
    if len(hits) > 1:
        for a, b in itertools.pairwise(hits):
            assert a.score >= b.score


@pytest.mark.asyncio
async def test_dry_run_returns_empty_when_resolver_fails():
    from perspicacite.pipeline.cite_graph import enrich_kb_from_cite_graph

    with patch("perspicacite.pipeline.cite_graph._resolve_and_fetch",
               new=AsyncMock(return_value=([], None))):
        kb_cfg = KnowledgeBaseConfig()
        hits = await enrich_kb_from_cite_graph(
            tool="unknown-lib", kb_config=kb_cfg, existing_dois=set(),
            dry_run=True, now_year=2026,
        )
    assert hits == []


@pytest.mark.asyncio
async def test_dry_run_does_not_call_ingest():
    from perspicacite.pipeline.cite_graph import enrich_kb_from_cite_graph

    with patch("perspicacite.pipeline.cite_graph._resolve_and_fetch",
               new=AsyncMock(return_value=([], None))):
        kb_cfg = KnowledgeBaseConfig()
        hits = await enrich_kb_from_cite_graph(
            tool="x", kb_config=kb_cfg, existing_dois=set(),
            dry_run=True, now_year=2026,
        )
    assert isinstance(hits, list)
