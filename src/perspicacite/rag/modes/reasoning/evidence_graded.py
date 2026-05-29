"""Phase 1 — evidence_graded strategy.

Capability: ECO-quality-weighted answer composition. Top of the stack: uses
the full machinery from Phases 4 + 3 + 2.

Pipeline:
1. Retrieve claims by hybrid (vector → extracted claims; plus
   ``neighbors`` expansion from the claim graph).
2. Stratify by ECO tier (data > citation > knowledge > inference > speculation).
3. Compose answer with three sections; each tier annotated with ECO IRI label.
4. Sidecar JSON-LD; per-tier evidence summary.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from perspicacite.indicium_layer import queries as indicium_queries
from perspicacite.indicium_layer.builder import (
    claim_iri,
    evidence_iri,
    paper_iri,
    passage_iri,
    run_iri,
)
from perspicacite.indicium_layer.queries import (
    ASB_NS,
    IRI_CLAIM,
    IRI_CONTEXT,
    IRI_CREATED,
    IRI_EVIDENCE,
    IRI_EVIDENCE_PROP,
    IRI_OBJECT,
    IRI_QUALIFIER,
    IRI_RDF_TYPE,
    IRI_RELATION,
    IRI_SUBJECT,
    IRI_TEXT_CHUNK,
    IRI_WAS_DERIVED_FROM,
    IRI_WAS_GENERATED_BY,
    XSD_NS,
)
from perspicacite.indicium_layer.store import ClaimGraphStore
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGRequest, SourceReference, StreamEvent
from perspicacite.pipeline.claims import claims_to_graph, extract_claims
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.reasoning.evidence_graded")

# Optional indicium dependencies (graceful degradation if extra not installed)
try:
    from indicium import validate_graph as _validate_graph
    from indicium.adapters.kgmemory import claim_jsonld as _claim_jsonld

    _HAS_INDICIUM_VALIDATE = True
except ImportError:
    _validate_graph = None
    _claim_jsonld = None
    _HAS_INDICIUM_VALIDATE = False

RETRIEVAL_TOP_K = 30

_TIER_ORDER = ["data", "citation", "knowledge", "inference", "speculation"]
_TIER_HEADERS = {
    "data": "## What the data show     [ECO:0000006 data]",
    "citation": "## What the literature claims     [ECO:0000033 citation]",
    "knowledge": "## What is reported     [ECO:0000302 knowledge]",
    "inference": (
        "## What is inferred / speculated     [ECO:0000361 inference, ECO:0000034 speculation]"
    ),
    "speculation": (
        "## What is inferred / speculated     [ECO:0000361 inference, ECO:0000034 speculation]"
    ),
}

_COMPOSE_PROMPT = """You stratify an answer by ECO evidence tier.

Tiers (most to least authoritative): data > citation > knowledge > inference >
speculation. Group bullets by tier.

Return strict JSON:
{{
  "tiers": [
    {{"eco": "data|citation|knowledge|inference|speculation",
      "bullets": [
        {{"text": "<one sentence>", "papers": ["<doi>", ...]}}
      ]}}
  ]
}}

Question: {query}

Claims (each line carries its ECO tier):
{claims}
"""


def _open_store(kb_name: str) -> ClaimGraphStore:
    data_dir = pathlib.Path("data/claim_graphs") / kb_name
    try:
        return ClaimGraphStore(kb_name, data_dir=data_dir, backend="oxigraph")
    except Exception as exc:
        logger.warning("oxigraph_open_failed_fallback_memory", error=str(exc))
        return ClaimGraphStore(kb_name, backend="memory")


def _parse_json(raw: Any, default: dict) -> dict:
    s = (raw if isinstance(raw, str) else str(raw)).strip()
    if s.startswith("```"):
        s = s.split("```", 2)[-1].lstrip("json").strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        return default


def _render_tiers_markdown(tiers: list[dict]) -> str:
    out: list[str] = []
    rendered_headers: set[str] = set()
    by_eco = {t.get("eco"): t for t in tiers}
    for eco in _TIER_ORDER:
        if eco not in by_eco:
            continue
        header = _TIER_HEADERS[eco]
        if header in rendered_headers:
            continue
        rendered_headers.add(header)
        out.append(header)
        bullets = by_eco[eco].get("bullets", []) or []
        if not bullets:
            out.append("- _(none)_")
            continue
        for b in bullets:
            papers = ", ".join(b.get("papers", []) or [])
            out.append(f"- {b.get('text', '')}" + (f" — papers: {papers}" if papers else ""))
    return "\n\n".join(out)


def _chunk_meta(ch: Any) -> dict:
    """Extract metadata dict from a chunk (supports dict or Pydantic model)."""
    if isinstance(ch, dict):
        meta = ch.get("metadata") or {}
        if isinstance(meta, dict):
            return meta
        # Pydantic model stored in "metadata" key
        return {
            k: getattr(meta, k, None)
            for k in ("paper_id", "doi", "title", "year", "char_span", "chunk_index")
        }
    # Chunk is itself a Pydantic model
    raw_meta = getattr(ch, "metadata", None)
    if raw_meta is None:
        return {}
    if isinstance(raw_meta, dict):
        return raw_meta
    return {
        k: getattr(raw_meta, k, None)
        for k in ("paper_id", "doi", "title", "year", "char_span", "chunk_index")
    }


def _chunk_text(ch: Any) -> str:
    if isinstance(ch, dict):
        return ch.get("text", "") or ""
    return getattr(ch, "text", "") or ""


def _chunk_idx(ch: Any) -> int:
    if isinstance(ch, dict):
        return ch.get("chunk_idx", 0) or 0
    return getattr(ch, "chunk_idx", 0) or 0


async def run_evidence_graded_stream(
    *,
    request: RAGRequest,
    llm: Any,
    vector_store: Any,
    embedding_provider: Any,
    config: Any,
    session_store: Any = None,
) -> AsyncIterator[StreamEvent]:
    # Minimal concrete subclass to access _build_kb_retriever (can be monkeypatched)
    class _MinimalMode(BaseRAGMode):
        async def execute(self, *a: Any, **kw: Any) -> Any:
            raise NotImplementedError

        async def execute_stream(self, *a: Any, **kw: Any) -> Any:  # type: ignore[override]  # stub subclass; return type differs from AsyncIterator base
            raise NotImplementedError

    kb_name = (request.kb_names[0] if request.kb_names else request.kb_name) or "default"

    yield StreamEvent.status("Reasoning (evidence_graded): hybrid retrieval…")
    mode_obj = _MinimalMode(config)
    dkb = mode_obj._build_kb_retriever(request, vector_store, embedding_provider)
    chunks = await dkb.search(request.query, top_k=RETRIEVAL_TOP_K)

    if not chunks:
        yield StreamEvent.content("No relevant passages found for evidence-graded analysis.")
        yield StreamEvent.done(conversation_id="", tokens_used=0, mode="reasoning", iterations=1)
        return

    # Phase-4 substrate: extract + SHACL + upsert
    yield StreamEvent.status(
        f"Reasoning (evidence_graded): extracting claims from {len(chunks)} passage(s)…"
    )
    passages_for_extract: list[dict] = []
    for ch in chunks:
        meta = _chunk_meta(ch)
        passages_for_extract.append(
            {
                "chunk_text": _chunk_text(ch),
                "source": {"doi": meta.get("doi") or meta.get("paper_id")},
            }
        )
    extracted = await extract_claims(
        llm_client=llm,
        passages=passages_for_extract,
        context=request.query,
        model=getattr(request, "model", None),
    )
    if not extracted:
        yield StreamEvent.content(
            "No structured claims could be extracted from the retrieved passages."
        )
        yield StreamEvent.done(conversation_id="", tokens_used=0, mode="reasoning", iterations=1)
        return

    store = _open_store(kb_name)
    now = _dt.datetime.now(_dt.UTC)
    run = run_iri(kb_name, now.strftime("%Y%m%dT%H%M%SZ"), getattr(request, "model", None))
    valid_claims: list[dict] = []
    for ci, claim in enumerate(extracted):
        if _HAS_INDICIUM_VALIDATE and _validate_graph is not None:
            conforms, report = _validate_graph(claims_to_graph([claim]))
            if not conforms:
                logger.warning("evidence_graded_shacl_invalid", preview=str(report)[:160])
                continue
        ch = chunks[ci] if ci < len(chunks) else chunks[0]
        meta = _chunk_meta(ch)
        paper_meta = {
            "doi": meta.get("doi"),
            "title": meta.get("title"),
            "year": meta.get("year"),
            "paper_id": meta.get("paper_id") or meta.get("doi"),
        }
        p_iri = paper_iri(kb_name, paper_meta)
        paper_id_str = paper_meta["paper_id"] or "unknown"
        ps_iri = passage_iri(kb_name, paper_id_str, _chunk_idx(ch))
        ev_grade = (claim.get("evidence") or [{}])[0].get("evidence_type") or "knowledge"
        e_iri = evidence_iri(kb_name, ps_iri, ev_grade)
        c_iri = claim_iri(kb_name, claim)
        store.add(ps_iri, IRI_RDF_TYPE, IRI_TEXT_CHUNK)
        store.add(e_iri, IRI_RDF_TYPE, IRI_EVIDENCE)
        store.add(c_iri, IRI_RDF_TYPE, IRI_CLAIM)
        for slot, iri in (
            ("context", IRI_CONTEXT),
            ("subject", IRI_SUBJECT),
            ("qualifier", IRI_QUALIFIER),
            ("relation", IRI_RELATION),
            ("object", IRI_OBJECT),
        ):
            if claim.get(slot):
                store.add(c_iri, iri, ("literal", str(claim[slot]), None))
        store.add(c_iri, IRI_EVIDENCE_PROP, e_iri)
        store.add(c_iri, IRI_WAS_DERIVED_FROM, p_iri)
        store.add(c_iri, IRI_WAS_GENERATED_BY, run)
        store.add(c_iri, IRI_CREATED, ("literal", now.isoformat(), f"{XSD_NS}dateTime"))
        eco_table = indicium_queries._ECO_IRI_BY_TIER
        store.add(
            c_iri,
            f"{ASB_NS}evidenceTypeIri",
            eco_table.get(ev_grade, eco_table["knowledge"]),
        )
        rec = dict(claim)
        rec["_iri"] = c_iri
        rec["_paper_id"] = paper_id_str
        rec["_paper_doi"] = paper_meta.get("doi")
        rec["_paper_title"] = paper_meta.get("title")
        rec["_paper_year"] = paper_meta.get("year")
        rec["_evidence_grade"] = ev_grade
        rec["id"] = c_iri.rsplit("/", 1)[-1]
        valid_claims.append(rec)

    if not valid_claims:
        store.close()
        yield StreamEvent.content("No SHACL-valid claims for evidence-graded analysis.")
        yield StreamEvent.done(conversation_id="", tokens_used=0, mode="reasoning", iterations=1)
        return

    # Hybrid expansion: pull supporting claims + neighbours for each seed
    yield StreamEvent.status(
        f"Reasoning (evidence_graded): expanding via graph ({len(valid_claims)} seed claims)…"
    )
    expanded_iris: set[str] = {c["_iri"] for c in valid_claims}
    try:
        for c in valid_claims:
            for n in indicium_queries.neighbors(store, kb_name, c["_iri"]):
                if n.get("neighbor"):
                    expanded_iris.add(n["neighbor"])
    except Exception as exc:
        logger.warning("evidence_graded_graph_expand_failed", error=str(exc))

    # Build the prompt input
    rendered = "\n".join(
        f"[{c['_evidence_grade']}] "
        f"{c.get('subject', '')} {c.get('qualifier', '')} {c.get('object', '')}"
        f" — paper: {c['_paper_doi'] or c['_paper_id']}"
        for c in valid_claims
    )
    try:
        raw = await llm.complete(
            messages=[
                {
                    "role": "user",
                    "content": _COMPOSE_PROMPT.format(query=request.query, claims=rendered),
                }
            ],
            stage="reasoning.evidence_graded.compose",
        )
    except Exception as exc:
        logger.warning("evidence_graded_compose_failed", error=str(exc))
        raw = "{}"
    composed = _parse_json(raw, {"tiers": []})
    markdown = _render_tiers_markdown(composed.get("tiers", []))
    if not markdown.strip():
        markdown = (
            "(Evidence-graded composition produced no tiered output. "
            f"Seeded with {len(valid_claims)} claims; "
            f"expanded to {len(expanded_iris)} via the claim graph.)"
        )
    yield StreamEvent.content(markdown)

    if _HAS_INDICIUM_VALIDATE and _claim_jsonld is not None:
        jsonld_entries = [_claim_jsonld(c) for c in valid_claims]
    else:
        jsonld_entries = [{"@type": ["urn:indicium:Claim"], "id": c["id"]} for c in valid_claims]

    yield StreamEvent.status_kind(
        "Reasoning (evidence_graded): JSON-LD sidecar attached.",
        kind="sidecar",
        format="jsonld",
        claims=jsonld_entries,
        tier_summary={
            tier: sum(1 for c in valid_claims if c["_evidence_grade"] == tier)
            for tier in _TIER_ORDER
        },
        expanded_claim_count=len(expanded_iris),
    )

    seen: set[str] = set()
    for c in valid_claims:
        pid = c["_paper_id"]
        if pid in seen:
            continue
        seen.add(pid)
        yield StreamEvent.source(
            SourceReference(
                title=c.get("_paper_title") or pid,
                doi=c.get("_paper_doi"),
                year=c.get("_paper_year"),
                relevance_score=0.75,
                kb_name=kb_name,
            )
        )

    store.close()
    yield StreamEvent.done(
        conversation_id="",
        tokens_used=0,
        mode="reasoning",
        iterations=1,
    )
