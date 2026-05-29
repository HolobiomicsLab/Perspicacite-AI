"""Unit tests for DBLP SPARQL + SemOpenAlex search provider helpers."""
from __future__ import annotations

# ── _tokenise_query ───────────────────────────────────────────────────────────

def test_tokenise_removes_stop_words():
    from perspicacite.search.dblp_sparql_search import _tokenise_query
    result = _tokenise_query("the analysis of neural networks")
    assert "the" not in result
    assert "of" not in result
    assert "neural" in result
    assert "networks" in result


def test_tokenise_caps_at_eight():
    from perspicacite.search.dblp_sparql_search import _tokenise_query
    long_query = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    result = _tokenise_query(long_query)
    assert len(result) <= 8


def test_tokenise_fallback_on_all_stop_words():
    from perspicacite.search.dblp_sparql_search import _tokenise_query
    # When all tokens are stop words or too short, fall back to first 3 raw tokens
    result = _tokenise_query("the a an")
    assert len(result) >= 1  # fallback ensures at least something is returned


def test_tokenise_short_tokens_dropped():
    from perspicacite.search.dblp_sparql_search import _tokenise_query
    result = _tokenise_query("AI ML deep learning")
    # "AI" and "ML" are 2 chars, should be dropped; "deep" and "learning" kept
    assert "deep" in result
    assert "learning" in result


# ── _build_dblp_sparql ────────────────────────────────────────────────────────

def test_build_dblp_sparql_contains_keywords():
    from perspicacite.search.dblp_sparql_search import _build_dblp_sparql
    sparql = _build_dblp_sparql(["neural", "network"], max_results=10)
    assert "CONTAINS(?lowerTitle, 'neural')" in sparql
    assert "CONTAINS(?lowerTitle, 'network')" in sparql


def test_build_dblp_sparql_year_filter_both():
    from perspicacite.search.dblp_sparql_search import _build_dblp_sparql
    sparql = _build_dblp_sparql(["graph"], max_results=5, year_min=2018, year_max=2023)
    assert "FILTER(?year >= 2018 && ?year <= 2023)" in sparql


def test_build_dblp_sparql_year_filter_min_only():
    from perspicacite.search.dblp_sparql_search import _build_dblp_sparql
    sparql = _build_dblp_sparql(["graph"], max_results=5, year_min=2020)
    assert "FILTER(?year >= 2020" in sparql


def test_build_dblp_sparql_no_year_filter_when_none():
    from perspicacite.search.dblp_sparql_search import _build_dblp_sparql
    sparql = _build_dblp_sparql(["graph"], max_results=5)
    assert "FILTER(?year" not in sparql


def test_build_dblp_sparql_limit():
    from perspicacite.search.dblp_sparql_search import _build_dblp_sparql
    sparql = _build_dblp_sparql(["test"], max_results=15)
    assert "LIMIT 15" in sparql


# ── _clean_literal + _parse_dblp_response ────────────────────────────────────

def test_clean_literal_quoted_string():
    from perspicacite.search.dblp_sparql_search import _clean_literal
    assert _clean_literal('"hello world"') == "hello world"


def test_clean_literal_typed_literal():
    from perspicacite.search.dblp_sparql_search import _clean_literal
    assert _clean_literal('"2021"^^xsd:integer') == "2021"


def test_clean_literal_iri():
    from perspicacite.search.dblp_sparql_search import _clean_literal
    assert _clean_literal("<https://example.org/foo>") == "https://example.org/foo"


def test_clean_literal_plain_string():
    from perspicacite.search.dblp_sparql_search import _clean_literal
    assert _clean_literal("plain") == "plain"


def test_parse_dblp_response_basic():
    from perspicacite.search.dblp_sparql_search import _parse_dblp_response
    data = {
        "res": [
            ['"Attention Is All You Need"', '"10.1234/abc"', '"2017"', '"5000"', '"2"'],
        ]
    }
    results = _parse_dblp_response(data)
    assert len(results) == 1
    assert results[0]["title"] == "Attention Is All You Need"
    assert results[0]["doi"] == "10.1234/abc"
    assert results[0]["year"] == 2017
    assert results[0]["cites"] == 5000


def test_parse_dblp_response_strips_doi_uri():
    from perspicacite.search.dblp_sparql_search import _parse_dblp_response
    data = {
        "res": [
            ['"A Paper"', '"https://doi.org/10.5678/xyz"', '"2020"', '"10"', '"1"'],
        ]
    }
    results = _parse_dblp_response(data)
    assert results[0]["doi"] == "10.5678/xyz"


def test_parse_dblp_response_skips_malformed_rows():
    from perspicacite.search.dblp_sparql_search import _parse_dblp_response
    data = {"res": [["only_two_cols", "x"], ['"Good"', '"10.1/a"', '"2020"', '"1"', '"1"']]}
    results = _parse_dblp_response(data)
    assert len(results) == 1
    assert results[0]["doi"] == "10.1/a"


def test_parse_dblp_response_empty():
    from perspicacite.search.dblp_sparql_search import _parse_dblp_response
    assert _parse_dblp_response({"res": []}) == []
    assert _parse_dblp_response({}) == []


# ── _build_semoa_sparql ───────────────────────────────────────────────────────

def test_build_semoa_sparql_contains_dois():
    from perspicacite.search.dblp_sparql_search import _build_semoa_sparql
    sparql = _build_semoa_sparql(["10.1234/a", "10.5678/b"])
    assert "https://doi.org/10.1234/a" in sparql
    assert "https://doi.org/10.5678/b" in sparql


def test_build_semoa_sparql_empty_dois():
    from perspicacite.search.dblp_sparql_search import _build_semoa_sparql
    sparql = _build_semoa_sparql([])
    assert "VALUES" in sparql


# ── _parse_semoa_response ─────────────────────────────────────────────────────

def test_parse_semoa_response_maps_doi_to_abstract():
    from perspicacite.search.dblp_sparql_search import _parse_semoa_response
    data = {
        "results": {
            "bindings": [
                {
                    "doiUri": {"type": "uri", "value": "https://doi.org/10.1234/abc"},
                    "abstract": {"type": "literal", "value": "This paper studies neural networks."},
                },
                {
                    "doiUri": {"type": "uri", "value": "https://doi.org/10.5678/xyz"},
                    "abstract": {"type": "literal", "value": "Graph convolutional methods."},
                },
            ]
        }
    }
    result = _parse_semoa_response(data)
    assert result["10.1234/abc"] == "This paper studies neural networks."
    assert result["10.5678/xyz"] == "Graph convolutional methods."


def test_parse_semoa_response_skips_missing_abstract():
    from perspicacite.search.dblp_sparql_search import _parse_semoa_response
    data = {
        "results": {
            "bindings": [
                {
                    "doiUri": {"type": "uri", "value": "https://doi.org/10.1234/abc"},
                    # no "abstract" key
                },
            ]
        }
    }
    result = _parse_semoa_response(data)
    assert result == {}


def test_parse_semoa_response_empty():
    from perspicacite.search.dblp_sparql_search import _parse_semoa_response
    assert _parse_semoa_response({}) == {}
    assert _parse_semoa_response({"results": {"bindings": []}}) == {}


# ── DBLPSPARQLSearchProvider ──────────────────────────────────────────────────

from unittest.mock import AsyncMock, patch

import pytest

_FAKE_DBLP_RECORDS = [
    {"title": "Graph Neural Networks Survey", "doi": "10.9999/gnn", "year": 2020, "cites": 800},
    {"title": "Deep Graph Learning", "doi": "10.9999/dgl", "year": 2019, "cites": 300},
]

_FAKE_SEMOA_ABSTRACTS = {"10.9999/gnn": "A survey of GNNs."}


@pytest.mark.asyncio
async def test_search_returns_papers():
    from perspicacite.search.dblp_sparql_search import DBLPSPARQLSearchProvider

    provider = DBLPSPARQLSearchProvider()

    with (
        patch(
            "perspicacite.search.dblp_sparql_search._query_dblp",
            new=AsyncMock(return_value=_FAKE_DBLP_RECORDS),
        ),
        patch(
            "perspicacite.search.dblp_sparql_search._enrich_semoa",
            new=AsyncMock(return_value=_FAKE_SEMOA_ABSTRACTS),
        ),
    ):
        papers = await provider.search("graph neural networks", max_results=5)

    assert len(papers) == 2
    gnn = next(p for p in papers if p.doi == "10.9999/gnn")
    assert gnn.title == "Graph Neural Networks Survey"
    assert gnn.abstract == "A survey of GNNs."
    assert gnn.year == 2020
    assert gnn.metadata["citation_count"] == 800


@pytest.mark.asyncio
async def test_search_paper_without_semoa_abstract_has_none():
    from perspicacite.search.dblp_sparql_search import DBLPSPARQLSearchProvider

    provider = DBLPSPARQLSearchProvider()

    with (
        patch(
            "perspicacite.search.dblp_sparql_search._query_dblp",
            new=AsyncMock(return_value=[
                {"title": "Deep Graph Learning", "doi": "10.9999/dgl", "year": 2019, "cites": 300},
            ]),
        ),
        patch(
            "perspicacite.search.dblp_sparql_search._enrich_semoa",
            new=AsyncMock(return_value={}),
        ),
    ):
        papers = await provider.search("graph learning")

    assert papers[0].abstract is None


@pytest.mark.asyncio
async def test_search_returns_empty_on_dblp_failure():
    from perspicacite.search.dblp_sparql_search import DBLPSPARQLSearchProvider

    provider = DBLPSPARQLSearchProvider()

    with patch(
        "perspicacite.search.dblp_sparql_search._query_dblp",
        new=AsyncMock(return_value=[]),
    ):
        papers = await provider.search("anything")

    assert papers == []


def test_provider_metadata():
    from perspicacite.search.dblp_sparql_search import DBLPSPARQLSearchProvider
    p = DBLPSPARQLSearchProvider()
    assert p.name == "dblp_sparql"
    assert p.tier == "external"
    assert p.domains == ["general"]
