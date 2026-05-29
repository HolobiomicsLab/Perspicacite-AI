#!/usr/bin/env python3
"""Full-pipeline live audit — Lab, ASB, KGmemory, Manuscript, Audit-instrumentation.

Maps the user-named subsystems to actual code:

  Lab           = paper-into-KB ingest (real OpenAlex metadata fetch + chunking)
  ASB           = capsule artifacts (metadata + blocks + resources + symbol-index)
  KGmemory      = cite-graph (live OpenAlex) + SessionStore + ProvenanceCollector
  Manuscript    = RAGResponse assembly + LiteratureSurveyRAGMode entry-point smoke
  Audit         = error-path logging + budget tracker + bm25s cache resilience

Three real papers exercise the harness (one is arXiv-only — stresses Task 3
arXiv-id fallback). Output: tests/audit/results/full-pipeline-<ts>.{json,md}.

Run:
    python tests/audit/run_full_pipeline_audit.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Articles (mix: 1 well-indexed Nature DOI, 1 arXiv-only, 1 well-cited preprint)
# ---------------------------------------------------------------------------

ARTICLES = [
    {
        "id": "alphafold",
        "name": "AlphaFold protein structure prediction",
        "doi": "10.1038/s41586-021-03819-2",
        "openalex_id": "W3177828909",
        "associated_library": "alphafold",
        "github_repo": "deepmind/alphafold",
        "code_file_url": (
            "https://raw.githubusercontent.com/deepmind/alphafold/main/"
            "alphafold/common/residue_constants.py"
        ),
        "code_file_path": "alphafold/common/residue_constants.py",
    },
    {
        "id": "rag",
        "name": "Retrieval-Augmented Generation (arXiv preprint, no journal DOI)",
        "doi": "10.48550/arXiv.2005.11401",
        "openalex_id": "W3098425262",
        "associated_library": "transformers",
        "github_repo": "huggingface/transformers",
        "code_file_url": (
            "https://raw.githubusercontent.com/huggingface/transformers/main/"
            "src/transformers/models/rag/modeling_rag.py"
        ),
        "code_file_path": "src/transformers/models/rag/modeling_rag.py",
    },
    {
        "id": "attention_is_all_you_need",
        "name": "Attention is All You Need (transformer original paper)",
        "doi": "10.48550/arXiv.1706.03762",
        "openalex_id": "W2963403868",
        "associated_library": "transformers",
        "github_repo": None,
        "code_file_url": None,
        "code_file_path": None,
    },
]


_audit_repo = Path.home() / "git" / "research-tools-audit"
_default_results = _audit_repo / "results" / "perspicacite-current" if _audit_repo.is_dir() else ROOT / "tests" / "audit" / "results"
RESULTS_DIR = Path(os.environ["AUDIT_RESULTS_DIR"]).expanduser() if os.environ.get("AUDIT_RESULTS_DIR") else _default_results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_url(url: str, *, timeout: float = 30.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def fetch_openalex_work(openalex_id: str, *, timeout: float = 20.0) -> dict | None:
    url = f"https://api.openalex.org/works/{openalex_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def section(label: str) -> None:
    print(f"\n{'=' * 4} {label} {'=' * (60 - len(label))}")


def report(msg: str) -> None:
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Phase 1 — LAB: ingest-pipeline primitives on real papers
# ---------------------------------------------------------------------------

async def audit_lab(findings: dict[str, Any]) -> None:
    section("Phase 1: LAB (ingest pipeline)")
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.pipeline.chunking_dispatch import chunk_document

    out: dict[str, Any] = {}
    findings["lab"] = out

    for art in ARTICLES:
        art_out: dict[str, Any] = {}
        out[art["id"]] = art_out

        # 1a) Live OpenAlex metadata fetch (the "Lab" entry point most users hit).
        report(f"[{art['id']}] fetching OpenAlex metadata for {art['openalex_id']}")
        t0 = time.perf_counter()
        oa = fetch_openalex_work(art["openalex_id"])
        art_out["oa_fetch_seconds"] = round(time.perf_counter() - t0, 3)
        if oa is None:
            art_out["status"] = "oa_fetch_failed"
            report(f"[{art['id']}] OA fetch FAILED")
            continue
        art_out["status"] = "ok"
        art_out["title"] = oa.get("title")
        art_out["year"] = oa.get("publication_year")
        art_out["cited_by_count"] = oa.get("cited_by_count")
        art_out["has_doi"] = bool(oa.get("doi"))
        art_out["openalex_doi"] = oa.get("doi")

        # 1b) Reconstruct abstract; chunk it via the dispatcher.
        abstract = reconstruct_abstract(oa.get("abstract_inverted_index"))
        art_out["abstract_len"] = len(abstract)
        if not abstract:
            art_out["chunk_count"] = 0
            report(f"[{art['id']}] no abstract — skipping chunk dispatch")
            continue

        # NOTE: PaperSource has no ARXIV/PUBMED/OPENALEX — the closest first-class
        # values are WEB_SEARCH or CITATION_FOLLOW. Tracked as audit finding.
        paper = Paper(
            id=f"doi:{art['doi']}",
            title=oa.get("title") or art["name"],
            abstract=abstract,
            source=PaperSource.WEB_SEARCH,
            doi=art["doi"],
            year=oa.get("publication_year"),
        )
        try:
            chunks = await chunk_document(
                abstract, paper, content_type="text",
                language=None, config=_minimal_chunk_config(),
            )
            art_out["chunk_count"] = len(chunks)
            art_out["first_chunk_preview"] = (chunks[0].text[:120] if chunks else "")
            report(
                f"[{art['id']}] abstract_len={len(abstract)}  "
                f"chunked into {len(chunks)} chunks"
            )
        except Exception as exc:
            art_out["chunk_error"] = f"{type(exc).__name__}: {exc}"
            report(f"[{art['id']}] CHUNK ERROR: {exc}")


def _minimal_chunk_config() -> Any:
    """Return a SimpleNamespace mimicking the parts of Config that chunk_document reads."""
    from types import SimpleNamespace
    return SimpleNamespace(
        knowledge_base=SimpleNamespace(
            chunk_size=512, chunk_overlap=64,
            embedding_model="text-embedding-3-small",
            use_two_pass=True,
            default_top_k=10,
        ),
    )


# ---------------------------------------------------------------------------
# Phase 2 — ASB: capsule artifacts on real fetched content
# ---------------------------------------------------------------------------

async def audit_asb(findings: dict[str, Any]) -> None:
    section("Phase 2: ASB (capsule artifacts)")
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.pipeline.capsule_builder import (
        capsule_dir_for,
        write_blocks,
        write_metadata,
        write_resources,
    )
    from perspicacite.pipeline.chunking_code import _chunk_python_ast
    from perspicacite.pipeline.symbol_index import (
        iter_symbols,
        write_chunks_symbols,
    )

    out: dict[str, Any] = {}
    findings["asb"] = out

    capsule_root = Path(tempfile.mkdtemp(prefix="audit_capsule_"))
    out["capsule_root"] = str(capsule_root)

    for art in ARTICLES:
        art_out: dict[str, Any] = {}
        out[art["id"]] = art_out

        paper = Paper(
            id=f"doi:{art['doi']}",
            title=art["name"],
            abstract="",
            source=PaperSource.BIBTEX,
            doi=art["doi"],
        )
        cdir = capsule_dir_for(paper, root=capsule_root)
        cdir.mkdir(parents=True, exist_ok=True)

        # 2a) metadata.json
        try:
            write_metadata(cdir, paper=paper, producer_version="audit-0.1")
            art_out["metadata_ok"] = (cdir / "metadata.json").exists()
        except Exception as exc:
            art_out["metadata_error"] = f"{type(exc).__name__}: {exc}"

        # 2b) blocks.jsonl from real OpenAlex abstract
        oa = fetch_openalex_work(art["openalex_id"])
        body_text = reconstruct_abstract(oa.get("abstract_inverted_index")) if oa else ""
        if body_text:
            try:
                n_blocks = write_blocks(cdir, text=body_text)
                art_out["blocks_written"] = n_blocks
            except Exception as exc:
                art_out["blocks_error"] = f"{type(exc).__name__}: {exc}"
        else:
            art_out["blocks_skipped"] = "no abstract"

        # 2c) resources.json — mine the real abstract for DOIs/GitHub/etc.
        try:
            n_res = write_resources(cdir, text=body_text or "")
            art_out["resources_written"] = n_res
        except Exception as exc:
            art_out["resources_error"] = f"{type(exc).__name__}: {exc}"

        # 2d) symbol-index sidecar — only when we have a real code file
        if art.get("code_file_url"):
            report(f"[{art['id']}] fetching code file for symbol-index")
            src = fetch_url(art["code_file_url"])
            if src:
                code_paper = Paper(
                    id=f"github:{art['github_repo']}@main:{art['code_file_path']}",
                    title=art["code_file_path"], abstract="", source=PaperSource.BIBTEX,
                )
                chunks = _chunk_python_ast(
                    src, code_paper, file_path=art["code_file_path"],
                    chunk_size=1000, chunk_overlap=200,
                )
                n_wrote = write_chunks_symbols(kb_dir=cdir, chunks=chunks)
                n_read = sum(1 for _ in iter_symbols(cdir))
                art_out["symbol_index"] = {
                    "n_chunks": len(chunks),
                    "n_wrote": n_wrote,
                    "n_read": n_read,
                    "kinds": sorted({c.metadata.symbol_kind for c in chunks}),
                }
                report(
                    f"[{art['id']}] symbol-index: {n_wrote} written, "
                    f"{n_read} read, kinds={sorted({c.metadata.symbol_kind for c in chunks})}"
                )

        # 2e) capsule directory inventory — what got written?
        inventory: dict[str, Any] = {}
        for child in sorted(cdir.rglob("*")):
            if child.is_file():
                rel = str(child.relative_to(cdir))
                inventory[rel] = child.stat().st_size
        art_out["capsule_inventory"] = inventory

    # cleanup temp capsule dir at end — but keep findings
    out["cleanup_ok"] = True
    shutil.rmtree(capsule_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase 3 — KGmemory: cite-graph + SessionStore + ProvenanceCollector
# ---------------------------------------------------------------------------

async def audit_kg_memory(findings: dict[str, Any]) -> None:
    section("Phase 3: KGmemory (cite-graph + session + provenance)")
    out: dict[str, Any] = {}
    findings["kg_memory"] = out

    # 3a) SessionStore round-trip
    await _audit_session_store(out)

    # 3b) ProvenanceCollector + ProvenanceStore round-trip
    await _audit_provenance(out)

    # 3c) Cite-graph live OpenAlex (per article)
    await _audit_cite_graph(out)


async def _audit_session_store(out: dict[str, Any]) -> None:
    from perspicacite.memory.session_store import SessionStore
    from perspicacite.models.kb import ChunkConfig, KnowledgeBase
    from perspicacite.models.messages import Message

    tmpdir = Path(tempfile.mkdtemp(prefix="audit_session_"))
    try:
        store = SessionStore(db_path=str(tmpdir / "session.sqlite"))
        await store.init_db()

        # KB metadata round-trip
        kb = KnowledgeBase(
            name="audit_kb", description="audit",
            collection_name="audit_coll",
            embedding_model="text-embedding-3-small",
            chunk_config=ChunkConfig(chunk_size=512, chunk_overlap=64),
        )
        await store.save_kb_metadata(kb)
        loaded_kb = await store.get_kb_metadata("audit_kb")
        out["session_kb_metadata"] = {
            "save_ok": True,
            "load_ok": loaded_kb is not None,
            "load_name_matches": (loaded_kb.name == "audit_kb") if loaded_kb else False,
        }
        kbs = await store.list_kbs()
        out["session_list_kbs"] = {"n": len(kbs)}

        # Conversation + messages
        conv = await store.create_conversation(session_id="s1", kb_name="audit_kb", title="audit")
        await store.add_message(conv.id, Message(role="user", content="What is AlphaFold?"))
        await store.add_message(conv.id, Message(role="assistant", content="A protein folding model."))
        msgs = await store.get_messages(conv.id, limit=10)
        out["session_conversation"] = {
            "create_ok": True,
            "n_messages": len(msgs),
            "first_role": msgs[0].role if msgs else None,
            "last_role": msgs[-1].role if msgs else None,
        }
        report(f"SessionStore: {len(msgs)} messages round-tripped")

    except Exception as exc:
        out["session_store_error"] = f"{type(exc).__name__}: {exc}"
        out["session_store_trace"] = traceback.format_exc()[:1200]
        report(f"SessionStore ERROR: {exc}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _audit_provenance(out: dict[str, Any]) -> None:
    """Real bug check: ProvenanceStore needs SessionStore.init_db() to create
    the `provenance` table — silent data loss when used standalone."""
    from perspicacite.memory.session_store import SessionStore
    from perspicacite.provenance.collector import ProvenanceCollector
    from perspicacite.provenance.context import collecting
    from perspicacite.provenance.store import ProvenanceStore

    tmpdir = Path(tempfile.mkdtemp(prefix="audit_prov_"))
    try:
        # First: prove the standalone-failure path (bug #1).
        col0 = ProvenanceCollector(
            conversation_id="conv-leak", message_id="msg-leak",
            rag_mode="basic", request_params={"query": "x"},
        )
        col0.add_llm_call(
            stage_label="t", provider="p", model="m",
            prompt_messages=[{"role":"user","content":"x"}],
            response_text="y", prompt_tokens=1, completion_tokens=1, latency_ms=1.0,
        )
        leak_store = ProvenanceStore(
            db_path=str(tmpdir / "standalone.sqlite"),
            sidecar_dir=str(tmpdir / "sidecar_leak"),
        )
        await leak_store.save(col0.finalize())
        leaked = await leak_store.get_for_message("msg-leak")
        out["provenance_standalone_works"] = (leaked is not None)
        # If this is False, ProvenanceStore silently dropped the row → real bug.

        # Now the supported path: SessionStore.init_db() must run on the same db.
        same_db = tmpdir / "prov.sqlite"
        sess = SessionStore(db_path=str(same_db))
        await sess.init_db()  # creates the provenance table
        col = ProvenanceCollector(
            conversation_id="conv1", message_id="msg1", rag_mode="basic",
            request_params={"query": "test", "top_k": 5},
        )
        with collecting(col) as c:
            c.add_trace("retrieval_start", query="test")
            c.add_retrieval(
                paper_id="p1", doi="10.1/x", title="Paper One",
                score=0.92, kb_name="kb1", content_type="text",
                pipeline_step="vector_search", rank=1, stage_label="initial",
            )
            c.add_llm_call(
                stage_label="answer_gen", provider="mock", model="m1",
                prompt_messages=[{"role": "user", "content": "..."}],
                response_text="answer", prompt_tokens=100, completion_tokens=50,
                latency_ms=42.0,
            )
            c.add_trace("retrieval_end", n_results=1)
        record = c.finalize()
        out["provenance_finalize"] = {
            "has_conversation_id": record.get("conversation_id") == "conv1",
            "has_message_id": record.get("message_id") == "msg1",
            "n_retrieval_events": len(record.get("retrieval_events", [])),
            "n_llm_calls": len(record.get("llm_calls", [])),
            "n_trace_steps": len(record.get("mode_trace", [])),
        }

        store = ProvenanceStore(
            db_path=str(same_db),
            sidecar_dir=str(tmpdir / "sidecar"),
        )
        await store.save(record)
        read_back = await store.get_for_message("msg1")
        sidecar_files = list((tmpdir / "sidecar").rglob("*"))
        out["provenance_store"] = {
            "save_ok": True,
            "round_trip_ok": read_back is not None,
            "n_sidecar_files": sum(1 for p in sidecar_files if p.is_file()),
            "sqlite_size_bytes": same_db.stat().st_size,
        }
        report(
            f"ProvenanceStore (with SessionStore.init_db): "
            f"round_trip_ok={read_back is not None}, "
            f"standalone_works={out['provenance_standalone_works']}"
        )

    except Exception as exc:
        out["provenance_error"] = f"{type(exc).__name__}: {exc}"
        out["provenance_trace"] = traceback.format_exc()[:1200]
        report(f"Provenance ERROR: {exc}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _audit_cite_graph(out: dict[str, Any]) -> None:
    from perspicacite.config.schema import CiteGraphConfig, KnowledgeBaseConfig
    from perspicacite.pipeline.cite_graph import enrich_kb_from_cite_graph

    cg_out: dict[str, Any] = {}
    out["cite_graph"] = cg_out

    for art in ARTICLES:
        a: dict[str, Any] = {}
        cg_out[art["id"]] = a
        kb_cfg = KnowledgeBaseConfig(
            library_paper_map={art["associated_library"]: art["doi"]},
            cite_graph=CiteGraphConfig(
                max_papers=10, min_year_offset=10, min_citations=0,
            ),
        )
        # Try BOTH paths: DOI resolution AND --openalex-id flag
        for path_kind, kwargs in [
            ("doi", {"doi": art["doi"], "tool": art["associated_library"]}),
            ("openalex_id", {"openalex_id": art["openalex_id"], "tool": art["associated_library"]}),
        ]:
            t0 = time.perf_counter()
            try:
                hits = await enrich_kb_from_cite_graph(
                    kb_config=kb_cfg, existing_dois=set(),
                    dry_run=True, **kwargs,
                )
                elapsed = round(time.perf_counter() - t0, 2)
                a[path_kind] = {
                    "status": "ok",
                    "seconds": elapsed,
                    "n_hits": len(hits),
                    "top_score": (round(hits[0].score, 3) if hits else None),
                    "top_title": (hits[0].title[:80] if hits else None),
                }
                report(
                    f"cite-graph [{art['id']}/{path_kind}]: "
                    f"{len(hits)} hits in {elapsed}s"
                )
            except Exception as exc:
                a[path_kind] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.perf_counter() - t0, 2),
                }
                report(f"cite-graph [{art['id']}/{path_kind}] ERROR: {exc}")


# ---------------------------------------------------------------------------
# Phase 4 — MANUSCRIPT: RAGResponse assembly + LiteratureSurveyRAGMode init
# ---------------------------------------------------------------------------

async def audit_manuscript(findings: dict[str, Any]) -> None:
    section("Phase 4: MANUSCRIPT (response assembly + survey mode init)")
    out: dict[str, Any] = {}
    findings["manuscript"] = out

    from perspicacite.models.rag import (
        CodeExcerpt,
        FigureRef,
        RAGMode,
        RAGRequest,
        RAGResponse,
        SourceReference,
        StreamEvent,
    )

    # 4a) RAGRequest validation — does it accept the kb_names multi-KB field?
    try:
        req = RAGRequest(
            query="Summarise AlphaFold's contributions.",
            kb_name="audit_kb",
            kb_names=["audit_kb", "other_kb"],
            mode=RAGMode.BASIC,
            provider="mock", model="mock-1",
        )
        out["rag_request_ok"] = {
            "mode_value": req.mode.value if hasattr(req.mode, "value") else str(req.mode),
            "kb_names": req.kb_names,
            "max_papers_default": req.max_papers_retrieval,
        }
    except Exception as exc:
        out["rag_request_error"] = f"{type(exc).__name__}: {exc}"

    # 4b) RAGResponse construction with attachments (Sub-project C surface)
    try:
        sources = [SourceReference(
            paper_id="doi:10.1038/s41586-021-03819-2",
            title="AlphaFold",
            authors=["Jumper et al."],
            year=2021,
            relevance_score=0.95,
            doi="10.1038/s41586-021-03819-2",
        )]
        figures = [FigureRef(
            id="pdf_p3_i1",
            paper_id="doi:10.1038/s41586-021-03819-2",
            label="Figure 1",
            caption="Network architecture.",
        )]
        excerpts = [CodeExcerpt(
            id="ex1",
            paper_id="github:deepmind/alphafold@main:alphafold/__init__.py",
            file_path="alphafold/__init__.py",
            symbol_name="model",
            symbol_kind="function",
            language="python",
            start_line=1, end_line=20,
            text="def model(): pass",
            source_url="https://github.com/deepmind/alphafold/blob/main/alphafold/__init__.py#L1-L20",
        )]
        resp = RAGResponse(
            answer="AlphaFold predicts protein structures.",
            sources=sources,
            mode=RAGMode.BASIC,
            iterations=1,
            confidence=0.9,
            figures=figures,
            code_excerpts=excerpts,
        )
        out["rag_response_ok"] = {
            "answer_len": len(resp.answer),
            "n_sources": len(resp.sources),
            "n_figures": len(resp.figures),
            "n_code_excerpts": len(resp.code_excerpts),
        }
        report(
            f"RAGResponse: {len(resp.sources)} sources, "
            f"{len(resp.figures)} figures, {len(resp.code_excerpts)} excerpts"
        )
    except Exception as exc:
        out["rag_response_error"] = f"{type(exc).__name__}: {exc}"
        out["rag_response_trace"] = traceback.format_exc()[:1200]

    # 4c) StreamEvent factories — code_excerpt + figure_ref
    try:
        ev_code = StreamEvent.code_excerpt({
            "id": "x", "language": "python", "text": "def f(): ...",
            "source_url": "https://github.com/a/b/blob/c/f.py#L1-L1",
        })
        ev_fig = StreamEvent.figure_ref({
            "id": "pdf_p1_i1", "paper_id": "p", "label": "Fig 1", "caption": "c",
        })
        # Try a thumbnail-bearing figure event too
        ev_fig_thumb = StreamEvent.figure_ref({
            "id": "pdf_p2_i1", "paper_id": "p", "label": "Fig 2",
            "caption": "with thumbnail",
            "thumbnail_b64": "iVBORw0KGgo=",
        })
        out["stream_events"] = {
            "code_event": ev_code.event,
            "figure_event": ev_fig.event,
            "figure_thumb_carries_payload": (
                "thumbnail_b64" in (ev_fig_thumb.data or {})
            ),
        }
    except Exception as exc:
        out["stream_events_error"] = f"{type(exc).__name__}: {exc}"

    # 4d) LiteratureSurveyRAGMode instantiation (real execute requires populated KB + LLM)
    try:
        from perspicacite.config.schema import Config
        from perspicacite.rag.modes.literature_survey import LiteratureSurveyRAGMode
        cfg = Config()
        mode = LiteratureSurveyRAGMode(cfg)
        out["lit_survey_mode_init"] = {
            "status": "ok",
            "batch_size": getattr(mode, "batch_size", None),
            "relevance_threshold": getattr(mode, "relevance_threshold", None),
            "max_themes": getattr(mode, "max_themes", None),
        }
        report(f"LiteratureSurveyRAGMode: init OK (batch_size={mode.batch_size})")
    except Exception as exc:
        out["lit_survey_mode_error"] = f"{type(exc).__name__}: {exc}"
        out["lit_survey_mode_trace"] = traceback.format_exc()[:1200]


# ---------------------------------------------------------------------------
# Phase 5 — AUDIT instrumentation: error-path logging, budget, bm25 cache
# ---------------------------------------------------------------------------

async def audit_instrumentation(findings: dict[str, Any]) -> None:
    section("Phase 5: AUDIT (instrumentation + budget + cache resilience)")
    out: dict[str, Any] = {}
    findings["instrumentation"] = out

    # 5a) Budget tracker basic sanity
    try:
        from perspicacite.llm.budget import BudgetTracker
        tracker = BudgetTracker(max_tokens=1000, max_cost_usd=1.0)
        tracker.add_usage(provider="mock", model="m1",
                          prompt_tokens=200, completion_tokens=100,
                          cost_usd=0.001)
        usage = tracker.summary() if hasattr(tracker, "summary") else None
        out["budget"] = {
            "tracker_ok": True,
            "has_summary_method": hasattr(tracker, "summary"),
            "summary_keys": list(usage.keys()) if isinstance(usage, dict) else None,
        }
        report(f"BudgetTracker: ok, summary keys={list(usage.keys()) if isinstance(usage, dict) else 'N/A'}")
    except Exception as exc:
        out["budget_error"] = f"{type(exc).__name__}: {exc}"

    # 5b) bm25s router cache resilience — cache hit then miss, with mocked rebuilder
    try:
        from perspicacite.rag.kb_router import _bm25_cache_clear, route_kbs
        _bm25_cache_clear()
        kb_ctx = {
            "biochem": "alphafold protein structure prediction folding",
            "ml_general": "transformer attention language model gpt",
            "math": "theorem proof lemma category topology",
        }
        chosen = route_kbs(
            query="how does alphafold predict protein structure",
            kb_contexts=kb_ctx, top_k=2,
        )

        # Second call with same corpus — should use cache
        chosen2 = route_kbs(
            query="protein folding methodology",
            kb_contexts=kb_ctx, top_k=2,
        )

        # Third call with mutated corpus — should rebuild
        route_kbs(
            query="protein folding",
            kb_contexts={**kb_ctx, "biochem": "edited"},
            top_k=2,
        )

        def _names(x):
            if not x: return []
            return [n for n, _ in x] if isinstance(x[0], tuple) else list(x)

        out["bm25_router"] = {
            "first_chose_biochem": "biochem" in _names(chosen),
            "second_chose_biochem": "biochem" in _names(chosen2),
            "third_handled_corpus_change": True,  # didn't raise
            "first_top_2": _names(chosen)[:2],
        }
        report(f"bm25s router: top-2 = {_names(chosen)[:2]}")
    except Exception as exc:
        out["bm25_router_error"] = f"{type(exc).__name__}: {exc}"
        out["bm25_router_trace"] = traceback.format_exc()[:1200]

    # 5c) Robust failure path — bad URL to snowball-public-helper
    try:
        import httpx

        from perspicacite.pipeline.snowball import openalex_id_for_doi
        async with httpx.AsyncClient() as client:
            # nonsense DOI that should miss both primary and arxiv-fallback
            result = await openalex_id_for_doi(
                client, "10.9999/totally-fake-doi-12345", headers={},
            )
        out["openalex_resolver_misses_gracefully"] = (result is None)
        report(f"openalex_id_for_doi(fake): -> {result} (expected None)")
    except Exception as exc:
        out["openalex_resolver_error"] = f"{type(exc).__name__}: {exc}"

    # 5d) chunking on malformed Python — must not raise
    try:
        from perspicacite.models.papers import Paper, PaperSource
        from perspicacite.pipeline.chunking_code import _chunk_python_ast
        bad_src = "def broken(\n   # missing close paren and colon\nclass X:::"
        paper = Paper(id="bad", title="bad.py", abstract="", source=PaperSource.BIBTEX)
        chunks = _chunk_python_ast(
            bad_src, paper, file_path="bad.py", chunk_size=1000, chunk_overlap=200,
        )
        out["chunking_malformed_python"] = {
            "did_not_raise": True,
            "n_chunks": len(chunks),
        }
        report(f"chunking malformed py: ok, fallback emitted {len(chunks)} chunks")
    except Exception as exc:
        out["chunking_malformed_error"] = f"{type(exc).__name__}: {exc}"
        out["chunking_malformed_trace"] = traceback.format_exc()[:1200]

    # 5e) collect_figure_refs with thumbnail path
    try:
        from perspicacite.models.documents import ChunkMetadata, DocumentChunk
        from perspicacite.rag.figure_refs import collect_figure_refs
        # Write a real png in a temp capsule structure
        tmp = Path(tempfile.mkdtemp(prefix="audit_fig_"))
        try:
            fig_dir = tmp / "paperX" / "figures"
            fig_dir.mkdir(parents=True)
            png = bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c6300010000000500010d0a2db4000000004945"
                "4e44ae426082"
            )
            (fig_dir / "fig1.png").write_bytes(png)
            md = ChunkMetadata(paper_id="paperX", chunk_index=0, content_type="text",
                               figure_refs=["fig1"])
            chunk = DocumentChunk(id="paperX_0", text="...", metadata=md)
            refs = collect_figure_refs([chunk], capsule_root=tmp)
            out["figure_refs_thumbnail"] = {
                "n_refs": len(refs),
                "thumbnail_loaded": (refs[0].thumbnail_b64 is not None) if refs else False,
            }
            report(f"collect_figure_refs: {len(refs)} refs, "
                   f"thumb_loaded={(refs[0].thumbnail_b64 is not None) if refs else False}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:
        out["figure_refs_thumbnail_error"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    findings: dict[str, Any] = {
        "timestamp": ts,
        "git_sha": os.popen("git rev-parse --short HEAD").read().strip(),
        "python": sys.version.split()[0],
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "articles": [{"id": a["id"], "name": a["name"], "doi": a["doi"]} for a in ARTICLES],
    }

    for phase_name, phase_fn in [
        ("lab", audit_lab),
        ("asb", audit_asb),
        ("kg_memory", audit_kg_memory),
        ("manuscript", audit_manuscript),
        ("instrumentation", audit_instrumentation),
    ]:
        try:
            await phase_fn(findings)
        except Exception as exc:
            print(f"\n  !!! phase '{phase_name}' FATAL: {exc}")
            findings[phase_name] = {
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[:3000],
            }

    # Persist
    json_path = RESULTS_DIR / f"full-pipeline-audit-{ts}.json"
    json_path.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nJSON: {json_path.relative_to(ROOT)}")

    # Human-readable summary
    md_lines = [
        f"# Full-pipeline audit — {ts}",
        "",
        f"Git SHA: `{findings['git_sha']}`",
        f"Python: `{findings['python']}` — Host: `{findings['host']}`",
        "",
        "Phases: Lab → ASB → KGmemory → Manuscript → Instrumentation",
        "",
    ]
    for phase in ["lab", "asb", "kg_memory", "manuscript", "instrumentation"]:
        if phase not in findings:
            continue
        md_lines.append(f"## {phase}")
        md_lines.append("```json")
        md_lines.append(json.dumps(findings[phase], indent=2, default=str)[:8000])
        md_lines.append("```")
        md_lines.append("")
    md_path = RESULTS_DIR / f"full-pipeline-audit-{ts}.md"
    md_path.write_text("\n".join(md_lines))
    print(f"MD:   {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
