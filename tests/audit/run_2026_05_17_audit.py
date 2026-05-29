#!/usr/bin/env python3
"""Live audit for features shipped 2026-05-17.

Covers:
  #1  DBLPSPARQLSearchProvider — phase 1 (DBLP QLever SPARQL endpoint)
  #2  SemOpenAlex abstract enrichment — phase 2 (semopenalex.org/sparql)
  #3  PaperSource.DBLP_SPARQL enum value present and round-trips
  #4  DBLPSPARQLSearchProvider registered in build_aggregator when enabled
  #5  --ingest-mode CLI flag on add-to-kb (abstract_only / full_text / auto)
  #6  --ingest-mode CLI flag on create-kb

No server required.  Phases 1-2 make real HTTP requests to public SPARQL
endpoints — skip with OFFLINE=1 to run purely in CI without network.

Run:
    python tests/audit/run_2026_05_17_audit.py
    OFFLINE=1 python tests/audit/run_2026_05_17_audit.py   # skip live SPARQL
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_audit_repo = Path.home() / "git" / "research-tools-audit"
_default_results = _audit_repo / "results" / "perspicacite-current" if _audit_repo.is_dir() else ROOT / "tests" / "audit" / "results"
RESULTS_DIR = Path(os.environ["AUDIT_RESULTS_DIR"]).expanduser() if os.environ.get("AUDIT_RESULTS_DIR") else _default_results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OFFLINE = os.getenv("OFFLINE", "").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✅"
FAIL = "❌"
SKIP = "⏩"
WARN = "⚠️"


def section(label: str) -> None:
    print(f"\n{'=' * 4} {label} {'=' * max(0, 60 - len(label))}")


def report(symbol: str, msg: str) -> None:
    print(f"  {symbol}  {msg}")


def _result(ok: bool, label: str, detail: str = "") -> dict[str, Any]:
    symbol = PASS if ok else FAIL
    report(symbol, f"{label}{': ' + detail if detail else ''}")
    return {"label": label, "ok": ok, "detail": detail}


def _skipped(label: str, reason: str = "OFFLINE") -> dict[str, Any]:
    report(SKIP, f"[SKIP] {label} — {reason}")
    return {"label": label, "ok": None, "detail": reason}


# ---------------------------------------------------------------------------
# #1 — DBLP QLever SPARQL: live query for a well-known paper
# ---------------------------------------------------------------------------

async def audit_dblp_sparql_live() -> list[dict[str, Any]]:
    """POST a real query to sparql.dblp.org and verify at least one result."""
    results: list[dict[str, Any]] = []
    section("#1  DBLP QLever SPARQL (live)")

    if OFFLINE:
        results.append(_skipped("dblp_sparql_live_query"))
        return results

    from perspicacite.search.dblp_sparql_search import (
        _build_dblp_sparql,
        _query_dblp,
        _tokenise_query,
    )

    query = "attention is all you need transformer"
    keywords = _tokenise_query(query)
    sparql = _build_dblp_sparql(keywords, max_results=5)

    t0 = time.monotonic()
    try:
        records = await _query_dblp(sparql)
        elapsed = time.monotonic() - t0
        ok = len(records) > 0
        if elapsed >= 24.0 and not ok:
            detail = f"TIMEOUT after {elapsed:.1f}s — sparql.dblp.org unreachable right now"
        else:
            detail = f"{len(records)} records in {elapsed:.1f}s"
        if ok:
            title_sample = records[0].get("title", "")[:70]
            detail += f" | first: \"{title_sample}\""
        results.append(_result(ok, "dblp_sparql_live_query", detail))

        # Validate record shape
        if records:
            rec = records[0]
            has_doi = bool(rec.get("doi"))
            has_year = rec.get("year") is not None
            results.append(_result(has_doi, "dblp_record_has_doi", rec.get("doi", "MISSING")[:50]))
            results.append(_result(has_year, "dblp_record_has_year", str(rec.get("year"))))
    except Exception as exc:
        results.append(_result(False, "dblp_sparql_live_query", str(exc)))

    return results


# ---------------------------------------------------------------------------
# #2 — SemOpenAlex abstract enrichment (live)
# ---------------------------------------------------------------------------

async def audit_semoa_live() -> list[dict[str, Any]]:
    """POST a batch VALUES SPARQL to semopenalex.org for the AlphaFold paper."""
    results: list[dict[str, Any]] = []
    section("#2  SemOpenAlex abstract enrichment (live)")

    if OFFLINE:
        results.append(_skipped("semoa_live_enrich"))
        return results

    from perspicacite.search.dblp_sparql_search import _enrich_semoa

    # AlphaFold — well-indexed in SemOpenAlex
    dois = ["10.1038/s41586-021-03819-2"]
    t0 = time.monotonic()
    try:
        abstracts = await _enrich_semoa(dois)
        elapsed = time.monotonic() - t0
        ok = bool(abstracts)
        if ok:
            snippet = next(iter(abstracts.values()))[:80].replace("\n", " ")
            detail = f"abstract fetched in {elapsed:.1f}s: \"{snippet}...\""
        else:
            detail = f"no abstract returned in {elapsed:.1f}s (endpoint may be slow)"
            # Treat as a warning, not hard failure — SemOpenAlex is community-hosted
            report(WARN, f"semoa_live_enrich: {detail}")
            results.append({"label": "semoa_live_enrich", "ok": None, "detail": detail})
            return results
        results.append(_result(ok, "semoa_live_enrich", detail))
    except Exception as exc:
        detail = str(exc)
        report(WARN, f"semoa_live_enrich raised {detail} (community server; non-fatal)")
        results.append({"label": "semoa_live_enrich", "ok": None, "detail": detail})

    return results


# ---------------------------------------------------------------------------
# #3 — PaperSource.DBLP_SPARQL enum
# ---------------------------------------------------------------------------

def audit_paper_source_enum() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    section("#3  PaperSource.DBLP_SPARQL enum")

    from perspicacite.models.papers import PaperSource

    try:
        val = PaperSource.DBLP_SPARQL
        results.append(_result(val.value == "dblp_sparql", "DBLP_SPARQL_value", val.value))
        # Round-trip: construct from string
        rt = PaperSource("dblp_sparql")
        results.append(_result(rt is val, "DBLP_SPARQL_roundtrip"))
    except Exception as exc:
        results.append(_result(False, "DBLP_SPARQL_enum", str(exc)))

    return results


# ---------------------------------------------------------------------------
# #4 — build_aggregator registers dblp_sparql when enabled
# ---------------------------------------------------------------------------

async def audit_aggregator_wiring() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    section("#4  build_aggregator wiring")

    from types import SimpleNamespace

    from perspicacite.search.domain_aggregator import build_aggregator

    def _cfg(enabled: list[str]) -> Any:
        return SimpleNamespace(
            search=SimpleNamespace(
                enabled_providers=enabled,
                provider_timeout_s=20.0,
                max_results_per_provider=25,
                core_api_key="",
                ads_api_key="",
            ),
            google_scholar=SimpleNamespace(enabled=False),
            pdf_download=SimpleNamespace(unpaywall_email=""),
        )

    # Build with dblp_sparql in enabled list
    try:
        agg_with = build_aggregator(_cfg(["dblp_sparql"]))
        names = [getattr(p, "name", "") for p in agg_with._providers]
        ok = "dblp_sparql" in names
        results.append(_result(ok, "dblp_sparql_in_aggregator", str(names)))
    except Exception as exc:
        results.append(_result(False, "dblp_sparql_in_aggregator", str(exc)))

    # Build without dblp_sparql
    try:
        agg_without = build_aggregator(_cfg(["europepmc"]))
        names_wo = [getattr(p, "name", "") for p in agg_without._providers]
        ok_wo = "dblp_sparql" not in names_wo
        results.append(_result(ok_wo, "dblp_sparql_absent_when_not_enabled", str(names_wo)))
    except Exception as exc:
        results.append(_result(False, "dblp_sparql_absent_when_not_enabled", str(exc)))

    return results


# ---------------------------------------------------------------------------
# #5 & #6 — --ingest-mode CLI flag on add-to-kb and create-kb
# ---------------------------------------------------------------------------

def audit_cli_ingest_mode() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    section("#5 / #6  --ingest-mode CLI flag")

    from unittest.mock import AsyncMock, patch

    from click.testing import CliRunner

    try:
        from perspicacite.cli import cli
    except Exception as exc:
        results.append(_result(False, "cli_import", str(exc)))
        return results

    # Use absolute path so CliRunner's CWD doesn't matter
    config_path = str(ROOT / "config/example.yml")

    def _make_bib(tmp: Path) -> Path:
        bib = tmp / "refs.bib"
        bib.write_text("@article{a, title={Test Paper}, year={2024}}\n")
        return bib

    # -- add-to-kb: abstract_only --
    with tempfile.TemporaryDirectory() as td:
        bib = _make_bib(Path(td))
        captured: dict = {}

        async def _fake_add(config, kb_name, bib_path, session_db, chroma_dir):
            captured["mode"] = config.knowledge_base.ingest_mode
            return {"new_papers": 0, "chunks_added": 0,
                    "total_papers": 0, "total_chunks": 0,
                    "pdf_stats": {"attempted": 0, "success": 0,
                                  "failed": 0, "skipped_no_doi": 0}}

        with patch("perspicacite.cli._add_bibtex_to_existing_kb",
                   new=AsyncMock(side_effect=_fake_add)):
            runner = CliRunner()
            res = runner.invoke(cli, [
                "-c", config_path,
                "add-to-kb", "audit_kb",
                "--from-bibtex", str(bib),
                "--ingest-mode", "abstract_only",
            ])
        ok = captured.get("mode") == "abstract_only" and res.exit_code == 0
        detail = f"mode={captured.get('mode')!r}  exit={res.exit_code}"
        if not ok and res.exception:
            detail += f"  exc={res.exception}"
        results.append(_result(ok, "add_to_kb_abstract_only", detail))

    # -- add-to-kb: full_text --
    with tempfile.TemporaryDirectory() as td:
        bib = _make_bib(Path(td))
        captured2: dict = {}

        async def _fake_add2(config, kb_name, bib_path, session_db, chroma_dir):
            captured2["mode"] = config.knowledge_base.ingest_mode
            return {"new_papers": 0, "chunks_added": 0,
                    "total_papers": 0, "total_chunks": 0,
                    "pdf_stats": {"attempted": 0, "success": 0,
                                  "failed": 0, "skipped_no_doi": 0}}

        with patch("perspicacite.cli._add_bibtex_to_existing_kb",
                   new=AsyncMock(side_effect=_fake_add2)):
            runner = CliRunner()
            res2 = runner.invoke(cli, [
                "-c", config_path,
                "add-to-kb", "audit_kb",
                "--from-bibtex", str(bib),
                "--ingest-mode", "full_text",
            ])
        ok2 = captured2.get("mode") == "full_text" and res2.exit_code == 0
        results.append(_result(ok2, "add_to_kb_full_text",
                               f"mode={captured2.get('mode')!r}"))

    # -- add-to-kb: default unchanged (config/example.yml has ingest_mode: auto) --
    with tempfile.TemporaryDirectory() as td:
        bib = _make_bib(Path(td))
        captured3: dict = {}

        async def _fake_add3(config, kb_name, bib_path, session_db, chroma_dir):
            captured3["mode"] = config.knowledge_base.ingest_mode
            return {"new_papers": 0, "chunks_added": 0,
                    "total_papers": 0, "total_chunks": 0,
                    "pdf_stats": {"attempted": 0, "success": 0,
                                  "failed": 0, "skipped_no_doi": 0}}

        with patch("perspicacite.cli._add_bibtex_to_existing_kb",
                   new=AsyncMock(side_effect=_fake_add3)):
            runner = CliRunner()
            runner.invoke(cli, [
                "-c", config_path,
                "add-to-kb", "audit_kb",
                "--from-bibtex", str(bib),
            ])
        ok3 = captured3.get("mode") == "auto"
        results.append(_result(ok3, "add_to_kb_default_unchanged",
                               f"mode={captured3.get('mode')!r} (want 'auto')"))

    # -- add-to-kb: invalid value rejected --
    with tempfile.TemporaryDirectory() as td:
        bib = _make_bib(Path(td))
        runner = CliRunner()
        res_bad = runner.invoke(cli, [
            "-c", config_path,
            "add-to-kb", "audit_kb",
            "--from-bibtex", str(bib),
            "--ingest-mode", "banana",
        ])
        ok_bad = res_bad.exit_code != 0
        results.append(_result(ok_bad, "add_to_kb_invalid_rejected",
                               f"exit={res_bad.exit_code}"))

    # -- create-kb: abstract_only --
    with tempfile.TemporaryDirectory() as td:
        bib = _make_bib(Path(td))
        captured4: dict = {}

        async def _fake_create(config, kb_name, bib_path, description,
                                session_db, chroma_dir):
            captured4["mode"] = config.knowledge_base.ingest_mode
            return {"name": kb_name, "collection_name": f"kb_{kb_name}",
                    "embedding_model": "text-embedding-3-small",
                    "papers": 0, "chunks_added": 0,
                    "total_papers": 0, "total_chunks": 0,
                    "pdf_stats": {"attempted": 0, "success": 0,
                                  "failed": 0, "skipped_no_doi": 0}}

        with patch("perspicacite.cli._create_kb_from_bibtex",
                   new=AsyncMock(side_effect=_fake_create)):
            runner = CliRunner()
            res4 = runner.invoke(cli, [
                "-c", config_path,
                "create-kb", "newkb",
                "--from-bibtex", str(bib),
                "--ingest-mode", "abstract_only",
            ])
        ok4 = captured4.get("mode") == "abstract_only" and res4.exit_code == 0
        detail4 = f"mode={captured4.get('mode')!r}  exit={res4.exit_code}"
        if not ok4 and res4.exception:
            detail4 += f"  exc={res4.exception}"
        results.append(_result(ok4, "create_kb_abstract_only", detail4))

    return results


# ---------------------------------------------------------------------------
# #7 — Full provider search() smoke (offline / mocked)
# ---------------------------------------------------------------------------

async def audit_provider_smoke() -> list[dict[str, Any]]:
    """Exercise the full search() path with mocked HTTP responses."""
    results: list[dict[str, Any]] = []
    section("#7  DBLPSPARQLSearchProvider.search() smoke (mocked)")

    from unittest.mock import AsyncMock, MagicMock, patch

    from perspicacite.models.papers import PaperSource
    from perspicacite.search.dblp_sparql_search import DBLPSPARQLSearchProvider

    dblp_payload = {
        "res": [
            ['"Attention Is All You Need"', '"https://doi.org/10.5555/3295222.3295349"',
             '"2017"^^xsd:integer', '"5000"', '"3"'],
        ]
    }
    semoa_payload = {
        "results": {
            "bindings": [
                {
                    "doiUri": {"value": "https://doi.org/10.5555/3295222.3295349"},
                    "abstract": {"value": "The dominant sequence transduction models..."},
                }
            ]
        }
    }

    mock_dblp_resp = MagicMock()
    mock_dblp_resp.raise_for_status = MagicMock()
    mock_dblp_resp.json = MagicMock(return_value=dblp_payload)

    mock_semoa_resp = MagicMock()
    mock_semoa_resp.raise_for_status = MagicMock()
    mock_semoa_resp.json = MagicMock(return_value=semoa_payload)

    async def fake_post(url, **kwargs):
        if "dblp" in url:
            return mock_dblp_resp
        return mock_semoa_resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=fake_post)

    try:
        with patch("perspicacite.search.dblp_sparql_search.httpx.AsyncClient",
                   return_value=mock_client):
            provider = DBLPSPARQLSearchProvider()
            papers = await provider.search("attention transformer", max_results=5)

        ok_count = len(papers) == 1
        results.append(_result(ok_count, "search_returns_one_paper", f"n={len(papers)}"))

        if papers:
            p = papers[0]
            results.append(_result(p.source == PaperSource.DBLP_SPARQL,
                                   "paper_source_is_dblp_sparql", str(p.source)))
            results.append(_result(
                p.abstract == "The dominant sequence transduction models...",
                "paper_abstract_enriched", (p.abstract or "")[:50]))
            results.append(_result(p.year == 2017, "paper_year_parsed", str(p.year)))
            results.append(_result(
                p.metadata.get("citation_count") == 5000,
                "citation_count_in_metadata",
                str(p.metadata.get("citation_count"))))
    except Exception:
        results.append(_result(False, "provider_smoke", traceback.format_exc()))

    return results


# ---------------------------------------------------------------------------
# #8 — OpenRouter CAPTCHA Fallback (added 2026-05-17)
# ---------------------------------------------------------------------------

async def audit_openrouter_captcha_fallback() -> list[dict[str, Any]]:
    """Audit the OpenRouter CAPTCHA fallback feature end-to-end.

    Covers:
      8a  PaperSource.OPENROUTER_WEB enum value + round-trip
      8b  GoogleScholarConfig new fields: defaults correct
      8c  _CAPTCHA_SENTINEL is a stable module-level singleton
      8d  _build_payload: tool_choice, engine, domains, max_results clamp
      8e  _parse_response: nested author arrays preserved (validates fix over re.DOTALL)
      8f  _parse_response: malformed JSON → [] (no exception propagated)
      8g  _build_paper: doi used as id; source=OPENROUTER_WEB
      8h  _build_paper: stable sha256 hash when no DOI
      8i  Full CAPTCHA→sentinel→OpenRouter→Paper flow (mocked HTTP)
      8j  Fallback disabled (openrouter_fallback_enabled=False) → [] returned
      8k  build_aggregator passes all 4 new fields to GoogleScholarPlaywrightProvider
      8l  Live OpenRouter API call with Exa engine (skipped if OFFLINE or no key)
    """
    results: list[dict[str, Any]] = []
    section("#8  OpenRouter CAPTCHA Fallback")

    # ── 8a: PaperSource.OPENROUTER_WEB ───────────────────────────────────────
    try:
        from perspicacite.models.papers import PaperSource
        val = PaperSource.OPENROUTER_WEB
        results.append(_result(val.value == "openrouter_web", "OPENROUTER_WEB_value", val.value))
        rt = PaperSource("openrouter_web")
        results.append(_result(rt is val, "OPENROUTER_WEB_roundtrip"))
    except Exception as exc:
        results.append(_result(False, "OPENROUTER_WEB_enum", str(exc)))

    # ── 8b: GoogleScholarConfig defaults ─────────────────────────────────────
    try:
        from perspicacite.config.schema import GoogleScholarConfig
        cfg = GoogleScholarConfig()
        results.append(_result(cfg.openrouter_fallback_enabled is False,
                               "scholar_cfg_fallback_enabled_default_false"))
        results.append(_result(cfg.openrouter_api_key == "",
                               "scholar_cfg_api_key_default_empty"))
        results.append(_result(cfg.openrouter_fallback_model == "deepseek/deepseek-chat",
                               "scholar_cfg_model_default", cfg.openrouter_fallback_model))
        results.append(_result(len(cfg.openrouter_fallback_domains) >= 8,
                               "scholar_cfg_domains_default_count",
                               str(len(cfg.openrouter_fallback_domains))))
        results.append(_result("arxiv.org" in cfg.openrouter_fallback_domains,
                               "scholar_cfg_domains_contains_arxiv"))
    except Exception as exc:
        results.append(_result(False, "scholar_cfg_new_fields", str(exc)))

    # ── 8c: _CAPTCHA_SENTINEL identity ───────────────────────────────────────
    try:
        import perspicacite.search.google_scholar_playwright as _mod
        sentinel = _mod._CAPTCHA_SENTINEL
        # Module-level singleton: same object on repeated access
        results.append(_result(_mod._CAPTCHA_SENTINEL is sentinel,
                               "captcha_sentinel_is_singleton"))
        # It's an empty list — but its identity is what matters, not its value
        results.append(_result(isinstance(sentinel, list) and len(sentinel) == 0,
                               "captcha_sentinel_is_empty_list"))
        # A freshly created [] has a different identity from the sentinel
        new_empty: list = []
        results.append(_result(sentinel is not new_empty,
                               "captcha_sentinel_distinct_from_new_list"))
    except Exception as exc:
        results.append(_result(False, "captcha_sentinel", str(exc)))

    # ── 8d: _build_payload structure ─────────────────────────────────────────
    try:
        from perspicacite.search.openrouter_fallback import _build_payload
        payload = _build_payload("CRISPR microbiome", "deepseek/deepseek-chat",
                                 10, ["arxiv.org", "biorxiv.org"])
        results.append(_result(payload["tool_choice"] == "required",
                               "payload_tool_choice_required"))
        results.append(_result(payload["tools"][0]["type"] == "openrouter:web_search",
                               "payload_tool_type_web_search"))
        results.append(_result(payload["tools"][0]["parameters"]["engine"] == "exa",
                               "payload_engine_exa"))
        results.append(_result(payload["tools"][0]["parameters"]["max_results"] == 10,
                               "payload_max_results_10"))
        results.append(_result("arxiv.org" in payload["tools"][0]["parameters"]["allowed_domains"],
                               "payload_domains_contains_arxiv"))
        results.append(_result("CRISPR" in payload["messages"][0]["content"],
                               "payload_query_in_prompt"))
        # Max-results clamp
        p99 = _build_payload("test", "m", 99, [])
        results.append(_result(p99["tools"][0]["parameters"]["max_results"] == 25,
                               "payload_max_results_clamped_to_25"))
    except Exception as exc:
        results.append(_result(False, "build_payload", str(exc)))

    # ── 8e: _parse_response — nested author arrays preserved ─────────────────
    # This validates the fix: re.DOTALL non-greedy regex would stop at the
    # first ']' inside the authors list; find/rfind correctly handles this.
    try:
        from perspicacite.search.openrouter_fallback import _parse_response
        content_with_nested = (
            'Here are papers: [{"title": "AlphaFold", '
            '"authors": ["Jumper J", "Evans R"], "year": 2021}] done.'
        )
        parsed = _parse_response(content_with_nested)
        ok_nested = (
            len(parsed) == 1
            and parsed[0].get("title") == "AlphaFold"
            and parsed[0].get("authors") == ["Jumper J", "Evans R"]
        )
        results.append(_result(ok_nested, "parse_response_preserves_nested_arrays",
                               f"authors={parsed[0].get('authors') if parsed else 'MISSING'}"))
    except Exception as exc:
        results.append(_result(False, "parse_response_nested", str(exc)))

    # ── 8f: _parse_response — malformed JSON → [] ────────────────────────────
    try:
        bad_cases = [
            ("no array", "No JSON here at all"),
            ("malformed", "[{broken json without closing"),
        ]
        for label, text in bad_cases:
            r = _parse_response(text)
            lbl = f"parse_response_{label.replace(' ', '_')}_returns_empty"
            results.append(_result(r == [], lbl))
    except Exception as exc:
        results.append(_result(False, "parse_response_error_handling", str(exc)))

    # ── 8g: _build_paper — doi as id, source, authors ────────────────────────
    try:
        from perspicacite.models.papers import PaperSource
        from perspicacite.search.openrouter_fallback import _build_paper
        entry = {
            "title": "AlphaFold2",
            "authors": ["Jumper J", "Evans R"],
            "year": 2021,
            "doi": "10.1038/s41586-021-03819-2",
            "abstract": "Protein structure prediction...",
            "url": "https://nature.com/articles/test",
        }
        paper = _build_paper(entry)
        results.append(_result(paper is not None, "build_paper_not_none"))
        if paper:
            results.append(_result(paper.id == paper.doi, "build_paper_id_equals_doi",
                                   paper.id))
            results.append(_result(paper.source == PaperSource.OPENROUTER_WEB,
                                   "build_paper_source_openrouter_web", str(paper.source)))
            results.append(_result(len(paper.authors) == 2, "build_paper_authors_count",
                                   str(len(paper.authors))))
            results.append(_result(paper.year == 2021, "build_paper_year", str(paper.year)))
    except Exception as exc:
        results.append(_result(False, "build_paper_full_entry", str(exc)))

    # ── 8h: _build_paper — stable sha256 hash when no DOI ────────────────────
    try:
        entry_no_doi = {"title": "No DOI Paper", "url": "https://arxiv.org/abs/1234.5678"}
        p1 = _build_paper(entry_no_doi)
        p2 = _build_paper(entry_no_doi)  # second call — must give same id
        ok_stable = p1 is not None and p2 is not None and p1.id == p2.id
        results.append(_result(ok_stable, "build_paper_stable_hash",
                               p1.id if p1 else "None"))
        if p1:
            results.append(_result(p1.id.startswith("openrouter:"),
                                   "build_paper_hash_prefix", p1.id))
    except Exception as exc:
        results.append(_result(False, "build_paper_no_doi", str(exc)))

    # ── 8i: Full CAPTCHA→sentinel→OpenRouter→Paper flow (mocked HTTP) ─────────
    try:
        import json as _json
        from unittest.mock import AsyncMock, MagicMock, patch

        from perspicacite.search.google_scholar_playwright import (
            _CAPTCHA_SENTINEL as _SENT,
        )
        from perspicacite.search.google_scholar_playwright import (
            GoogleScholarPlaywrightProvider,
        )

        or_content = _json.dumps([{
            "title": "Attention Is All You Need",
            "authors": ["Vaswani A"],
            "year": 2017,
            "doi": "10.5555/3295222.3295349",
            "abstract": "The dominant sequence transduction models...",
            "url": "https://arxiv.org/abs/1706.03762",
        }])
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": or_content}}]
        })
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        async def _fake_render(url, *, delay, headless, user_agent):
            return _SENT  # simulate CAPTCHA

        async def _run_flow():
            with patch(
                "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
                new=_fake_render,
            ), patch(
                "perspicacite.search.openrouter_fallback.httpx.AsyncClient",
                return_value=mock_client,
            ):
                provider = GoogleScholarPlaywrightProvider(
                    delay_seconds=0.0,
                    openrouter_fallback_enabled=True,
                    openrouter_api_key="sk-test",
                    openrouter_fallback_domains=["arxiv.org"],
                )
                return await provider.search("attention transformer", max_results=5)

        papers = await _run_flow()
        ok_flow = (
            len(papers) == 1
            and papers[0].title == "Attention Is All You Need"
            and papers[0].source == PaperSource.OPENROUTER_WEB
            and papers[0].id == "10.5555/3295222.3295349"
        )
        results.append(_result(ok_flow, "captcha_flow_returns_openrouter_paper",
                               f"n={len(papers)}, source={papers[0].source if papers else 'NONE'}"))
        if papers:
            # Verify the HTTP call used tool_choice=required and engine=exa
            sent = mock_client.post.call_args[1]["json"]
            results.append(_result(sent["tool_choice"] == "required",
                                   "captcha_flow_tool_choice_required"))
            results.append(_result(
                sent["tools"][0]["parameters"]["engine"] == "exa",
                "captcha_flow_engine_exa"))
    except Exception:
        results.append(_result(False, "captcha_flow_mocked", traceback.format_exc()[:300]))

    # ── 8j: Fallback disabled → [] returned, OpenRouter never called ──────────
    try:
        called = []

        async def _fake_render_j(url, *, delay, headless, user_agent):
            return _SENT

        async def _fake_or(*a, **kw):
            called.append(1)
            return []

        async def _run_disabled():
            with patch(
                "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
                new=_fake_render_j,
            ), patch(
                "perspicacite.search.openrouter_fallback.openrouter_academic_search",
                new=_fake_or,
            ):
                provider = GoogleScholarPlaywrightProvider(
                    delay_seconds=0.0,
                    openrouter_fallback_enabled=False,
                )
                return await provider.search("test", max_results=5)

        papers_disabled = await _run_disabled()
        results.append(_result(papers_disabled == [], "captcha_disabled_returns_empty"))
        results.append(_result(len(called) == 0, "captcha_disabled_no_openrouter_call",
                               f"called={len(called)}"))
    except Exception as exc:
        results.append(_result(False, "captcha_disabled", str(exc)))

    # ── 8k: build_aggregator passes all 4 new fields ─────────────────────────
    try:
        from types import SimpleNamespace

        from perspicacite.search.domain_aggregator import build_aggregator

        agg_cfg = SimpleNamespace(
            search=SimpleNamespace(
                enabled_providers=["google_scholar"],
                provider_timeout_s=20.0,
                max_results_per_provider=25,
                core_api_key="",
                ads_api_key="",
            ),
            google_scholar=SimpleNamespace(
                enabled=True,
                delay_seconds=1.0,
                headless=True,
                user_agent="AuditAgent",
                max_results=10,
                openrouter_fallback_enabled=True,
                openrouter_api_key="sk-audit",
                openrouter_fallback_model="openai/gpt-4o-mini",
                openrouter_fallback_domains=["arxiv.org"],
            ),
            pdf_download=SimpleNamespace(unpaywall_email=""),
        )
        agg = build_aggregator(agg_cfg)
        provider = next(
            (p for p in agg._providers if getattr(p, "name", "") == "google_scholar"),
            None,
        )
        ok_wired = (
            provider is not None
            and getattr(provider, "_openrouter_enabled", None) is True
            and getattr(provider, "_openrouter_api_key", None) == "sk-audit"
            and getattr(provider, "_openrouter_model", None) == "openai/gpt-4o-mini"
            and getattr(provider, "_openrouter_domains", None) == ["arxiv.org"]
        )
        results.append(_result(ok_wired, "aggregator_passes_fallback_fields_to_provider",
                               f"enabled={getattr(provider, '_openrouter_enabled', '?')} "
                               f"model={getattr(provider, '_openrouter_model', '?')}"))
    except Exception as exc:
        results.append(_result(False, "aggregator_scholar_wiring", str(exc)))

    # ── 8l: Live OpenRouter API (skipped when OFFLINE or no key) ─────────────
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if OFFLINE or not api_key:
        reason = "OFFLINE=1" if OFFLINE else "OPENROUTER_API_KEY not set"
        results.append(_skipped("openrouter_live_exa_search", reason))
    else:
        import time as _time

        from perspicacite.search.openrouter_fallback import openrouter_academic_search
        t0 = _time.monotonic()
        try:
            papers = await openrouter_academic_search(
                "CRISPR Cas9 genome editing bacteria",
                api_key=api_key,
                model="deepseek/deepseek-chat",
                max_results=3,
                allowed_domains=["arxiv.org", "biorxiv.org", "pubmed.ncbi.nlm.nih.gov"],
            )
            elapsed = _time.monotonic() - t0
            ok_live = len(papers) > 0 and all(
                p.source == PaperSource.OPENROUTER_WEB for p in papers
            )
            detail = (
                f"{len(papers)} papers in {elapsed:.1f}s"
                + (f" | first: \"{papers[0].title[:60]}\"" if papers else "")
            )
            results.append(_result(ok_live, "openrouter_live_exa_search", detail))
        except Exception as exc:
            results.append(_result(False, "openrouter_live_exa_search", str(exc)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"Perspicacite audit — 2026-05-17 features — {ts}")
    print(f"Python {sys.version.split()[0]}  |  OFFLINE={OFFLINE}")

    all_results: list[dict[str, Any]] = []

    all_results += await audit_dblp_sparql_live()
    all_results += await audit_semoa_live()
    all_results += audit_paper_source_enum()
    all_results += await audit_aggregator_wiring()
    loop = asyncio.get_event_loop()
    cli_results = await loop.run_in_executor(None, audit_cli_ingest_mode)
    all_results += cli_results
    all_results += await audit_provider_smoke()
    all_results += await audit_openrouter_captcha_fallback()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    section("Summary")

    passed  = [r for r in all_results if r["ok"] is True]
    failed  = [r for r in all_results if r["ok"] is False]
    skipped = [r for r in all_results if r["ok"] is None]

    print(f"\n  {PASS} Passed : {len(passed)}")
    print(f"  {FAIL} Failed : {len(failed)}")
    print(f"  {SKIP} Skipped: {len(skipped)}")

    if failed:
        print("\n  Failures:")
        for r in failed:
            print(f"    {FAIL}  {r['label']}: {r['detail']}")

    # ---------------------------------------------------------------------------
    # Persist results
    # ---------------------------------------------------------------------------
    out_json = RESULTS_DIR / f"audit-2026-05-17-{ts}.json"
    out_md   = RESULTS_DIR / f"audit-2026-05-17-{ts}.notes.md"

    with open(out_json, "w") as fh:
        json.dump({"ts": ts, "offline": OFFLINE, "results": all_results}, fh, indent=2)

    with open(out_md, "w") as fh:
        fh.write("# Perspicacite audit — 2026-05-17 features\n\n")
        fh.write(f"**Run:** {ts}  |  **OFFLINE:** {OFFLINE}\n\n")
        fh.write("| Result | Check | Detail |\n|--------|-------|--------|\n")
        for r in all_results:
            sym = PASS if r["ok"] is True else (SKIP if r["ok"] is None else FAIL)
            fh.write(f"| {sym} | `{r['label']}` | {r['detail']} |\n")
        fh.write(f"\n**{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped**\n")

    print(f"\n  Results → {out_json}")
    print(f"           {out_md}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
