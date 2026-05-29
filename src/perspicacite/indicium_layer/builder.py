"""Build the claim graph for one KB.

Pipeline:
1. Resolve KB papers via the caller-supplied ``papers_provider``.
2. Hash-diff against the manifest; skip unchanged.
3. For each new/changed paper:
   a. Load passages via ``passages_provider(paper_id)``.
   b. Call ``pipeline.claims.extract_claims`` (already shipped).
   c. SHACL-validate via ``indicium.validate_graph``; skip invalid claims.
   d. Upsert Claim / Evidence / Passage / Paper nodes.
4. Prune candidate pairs (cap fan-out).
5. CiTO-classify pruned pairs.
6. Write surviving edges + their confidence reification.
7. Persist manifest + append a build log line.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import indicium

from perspicacite.indicium_layer.anchor import anchor_claims
from perspicacite.indicium_layer.cito_classifier import classify_pairs
from perspicacite.indicium_layer.invalidation import (
    compute_paper_hash,
    papers_needing_rebuild,
    schema_version_changed,
)
from perspicacite.indicium_layer.manifest import (
    manifest_path,
    read_manifest,
    write_manifest,
)
from perspicacite.indicium_layer.pruner import build_candidate_pairs
from perspicacite.indicium_layer.queries import (
    ASB_NS,
    INDICIUM_NS,
    IRI_ANCHOR_STATUS,
    IRI_ASSERTED_BY,
    IRI_CAPTION,
    IRI_CLAIM,
    IRI_CLAIM_LINK,
    IRI_CLAIM_STATUS,
    IRI_CONTEXT,
    IRI_CREATED,
    IRI_DECISION_CONTEXT,
    IRI_DERIVED_FROM_PASSAGE,
    IRI_EVIDENCE,
    IRI_EVIDENCE_PROP,
    IRI_EVIDENCE_TYPE,
    IRI_FIGURE,
    IRI_FIGURE_ID,
    IRI_FIGURE_TYPE,
    IRI_FROM_CLAIM,
    IRI_LINK_TYPE,
    IRI_OA_EXACT,
    IRI_OA_PREFIX,
    IRI_OA_SUFFIX,
    IRI_OBJECT,
    IRI_QUALIFIER,
    IRI_QUOTE_EXACT,
    IRI_RDF_TYPE,
    IRI_RELATION,
    IRI_RESEARCH_PAPER,
    IRI_RUN_ACTIVITY,
    IRI_SOURCE_DOI,
    IRI_SUBJECT,
    IRI_TEXT_CHUNK,
    IRI_TEXT_QUOTE_SELECTOR,
    IRI_TO_CLAIM,
    IRI_WAS_DERIVED_FROM,
    IRI_WAS_GENERATED_BY,
    OA_NS,
    XSD_NS,
    cito_graph_iri,
    runs_graph_iri,
)
from perspicacite.logging import get_logger
from perspicacite.pipeline.claims import claims_to_graph, extract_claims

logger = get_logger("perspicacite.indicium_layer.builder")

BUILDER_VERSION = "1"
ECO_BASE = "http://purl.obolibrary.org/obo/"
_ECO_BY_TYPE = {
    "data": "ECO_0000006",
    "citation": "ECO_0000033",
    "knowledge": "ECO_0000302",
    "inference": "ECO_0000361",
    "speculation": "ECO_0000034",
}

PapersProvider = Callable[[], dict[str, dict]]
PassagesProvider = Callable[[str], list[dict]]


@dataclass
class BuildResult:
    kb_name: str
    claims_added: int
    edges_added: int
    pairs_classified: int
    papers_processed: int
    duration_seconds: float


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def claim_iri(kb_name: str, claim: dict) -> str:
    canonical = "|".join(
        str(claim.get(k, "")) for k in ("context", "subject", "qualifier", "relation", "object")
    )
    return f"kb://{kb_name}/claim/{_sha8(canonical)}"


def evidence_iri(kb_name: str, passage_id: str, eco_grade: str) -> str:
    return f"kb://{kb_name}/evidence/{_sha8(passage_id + '|' + eco_grade)}"


def paper_iri(kb_name: str, paper: dict) -> str:
    doi = paper.get("doi")
    if doi:
        return f"doi:{doi}"
    title = str(paper.get("title", ""))
    year = str(paper.get("year", ""))
    return f"kb://{kb_name}/paper/{_sha8(title + '|' + year)}"


def passage_iri(kb_name: str, paper_id: str, chunk_idx: int) -> str:
    return f"kb://{kb_name}/passage/{_sha8(paper_id)}/{chunk_idx}"


def run_iri(kb_name: str, iso8601: str, model: str | None) -> str:
    return f"kb://{kb_name}/run/{iso8601}/{_sha8(BUILDER_VERSION + '|' + (model or 'default'))}"


def _eco_iri(eco_grade: str | None) -> str:
    code = _ECO_BY_TYPE.get(eco_grade or "knowledge", _ECO_BY_TYPE["knowledge"])
    return f"{ECO_BASE}{code}"


def _write_anchor_provenance(store: Any, e_iri: str, anchor: dict, passage: dict | None) -> None:
    """Write R3 anchor triples onto an Evidence node.

    - asb:anchorStatus  : always (verified|repaired|unverified).
    - asb:quoteExact + an oa:TextQuoteSelector (oa:exact/prefix/suffix) : only
      when status is verified/repaired and a verbatim span is available — never
      for unverified (no laundering).
    """
    status = anchor.get("status", "unverified")
    store.add(e_iri, IRI_ANCHOR_STATUS, ("literal", status, None))
    quote_exact = anchor.get("quote_exact")
    if status in ("verified", "repaired") and quote_exact:
        store.add(e_iri, IRI_QUOTE_EXACT, ("literal", quote_exact, None))
        sel = f"{e_iri}#quote"
        store.add(e_iri, f"{OA_NS}hasSelector", sel)
        store.add(sel, IRI_RDF_TYPE, IRI_TEXT_QUOTE_SELECTOR)
        store.add(sel, IRI_OA_EXACT, ("literal", quote_exact, None))
        if passage is not None:
            ptext = str(passage.get("chunk_text") or passage.get("text") or "")
            pos = ptext.find(quote_exact)
            if pos != -1:
                prefix = ptext[max(0, pos - 32):pos]
                suffix = ptext[pos + len(quote_exact):pos + len(quote_exact) + 32]
                if prefix:
                    store.add(sel, IRI_OA_PREFIX, ("literal", prefix, None))
                if suffix:
                    store.add(sel, IRI_OA_SUFFIX, ("literal", suffix, None))


def _add_figure_node(
    store: Any,
    kb_name: str,
    claim_iri_str: str,
    fig: dict,
) -> str:
    """Add a Figure RDF node linked from claim_iri via prov:wasDerivedFrom.

    fig keys: fig_id, label, caption, fig_type, source_doi (optional)
    Returns the figure IRI.
    """
    fig_id = fig.get("fig_id") or fig.get("label") or "unknown"
    source_doi = fig.get("source_doi", "")
    h = hashlib.sha1(f"{source_doi}|{fig_id}".encode()).hexdigest()[:10]
    fig_iri = f"kb://{kb_name}/figure/{h}"

    store.add(fig_iri, IRI_RDF_TYPE, IRI_FIGURE)
    if fig.get("fig_id"):
        store.add(fig_iri, IRI_FIGURE_ID, ("literal", fig["fig_id"], None))
    if fig.get("caption"):
        store.add(fig_iri, IRI_CAPTION, ("literal", fig["caption"][:500], None))
    if fig.get("fig_type"):
        store.add(fig_iri, IRI_FIGURE_TYPE, ("literal", fig["fig_type"], None))
    if source_doi:
        store.add(fig_iri, IRI_SOURCE_DOI, ("literal", source_doi, None))
    # Link claim → figure
    store.add(claim_iri_str, IRI_WAS_DERIVED_FROM, fig_iri)
    return fig_iri


def _add_claim_link_node(
    store: Any,
    kb_name: str,
    from_iri: str,
    to_iri: str,
    link_type: str,
    *,
    asserted_by: str | None = None,
    graph: str | None = None,
) -> str:
    """Mint and store a ClaimLink node in the cito graph.

    Creates a new node of type cito:Citation (per Indicium v1.3 ClaimLink
    class_uri) with from_claim, to_claim, and link_type properties.
    """
    from uuid import uuid4

    link_iri = f"kb://{kb_name}/claimlink/{uuid4()}"
    g = graph or cito_graph_iri(kb_name)

    store.add(link_iri, IRI_RDF_TYPE, IRI_CLAIM_LINK, graph=g)
    store.add(link_iri, IRI_FROM_CLAIM, from_iri, graph=g)
    store.add(link_iri, IRI_TO_CLAIM, to_iri, graph=g)
    store.add(link_iri, IRI_LINK_TYPE, ("literal", link_type, None), graph=g)
    if asserted_by:
        store.add(link_iri, IRI_ASSERTED_BY, asserted_by, graph=g)

    return link_iri


async def build_claim_graph(
    *,
    kb_name: str,
    store: Any,
    llm_client: Any,
    papers_provider: PapersProvider,
    passages_provider: PassagesProvider,
    refresh: bool = False,
    max_pairs_per_claim: int = 20,
    model: str | None = None,
    builder_version: str = BUILDER_VERSION,
    progress_callback: Any = None,
    anchor_strict: bool = False,
    anchor_near_threshold: float = 0.9,
    anchor_audit_dir: str | None = None,
) -> BuildResult:
    """Build (or incrementally refresh) the claim graph for ``kb_name``."""
    t0 = _dt.datetime.now(_dt.UTC)
    iso = t0.strftime("%Y%m%dT%H%M%SZ")
    run = run_iri(kb_name, iso, model)
    runs_g = runs_graph_iri(kb_name)
    cito_g = cito_graph_iri(kb_name)

    store.add(run, IRI_RDF_TYPE, IRI_RUN_ACTIVITY, graph=runs_g)
    store.add(run, IRI_CREATED, ("literal", t0.isoformat(), f"{XSD_NS}dateTime"), graph=runs_g)

    manifest = read_manifest(kb_name)
    if schema_version_changed(manifest) or refresh:
        manifest.paper_hashes = {}

    all_papers = papers_provider()
    paper_texts = {
        pid: "\n".join(p.get("text", "") for p in passages_provider(pid)) for pid in all_papers
    }
    to_process = papers_needing_rebuild(manifest, paper_texts)

    claims_added = 0
    pairs_classified = 0
    edges_added = 0
    all_new_claims: list[dict[str, Any]] = []
    _seen_claim_iris: set[str] = set()

    for paper_id in to_process:
        paper_meta = all_papers[paper_id]
        passages = passages_provider(paper_id)
        if not passages:
            continue
        p_iri = paper_iri(kb_name, paper_meta)
        store.add(p_iri, IRI_RDF_TYPE, IRI_RESEARCH_PAPER)
        if paper_meta.get("title"):
            store.add(
                p_iri,
                "http://purl.org/dc/terms/title",
                ("literal", paper_meta["title"], None),
            )
        if paper_meta.get("year"):
            store.add(
                p_iri,
                "http://purl.org/dc/terms/issued",
                ("literal", str(paper_meta["year"]), None),
            )

        passages_for_extract = [
            {
                "chunk_text": pg.get("text", ""),
                "source": {"doi": paper_meta.get("doi")},
            }
            for pg in passages
        ]
        try:
            extracted = await extract_claims(
                llm_client=llm_client,
                passages=passages_for_extract,
                context=paper_meta.get("title"),
                model=model,
            )
        except Exception as exc:
            logger.warning("claim_extract_failed", paper_id=paper_id, error=str(exc))
            extracted = []

        # R3 — verify each claim's quote against the passages it was drawn from.
        # passages_for_extract is index-aligned with `passages`, so the resulting
        # matched_index also indexes into `passages` below.
        audit_path = (
            Path(anchor_audit_dir) / f"{kb_name}__{str(paper_id).replace('/', '_')}.anchor.jsonl"
            if anchor_audit_dir
            else None
        )
        extracted = anchor_claims(
            extracted,
            passages_for_extract,
            strict=anchor_strict,
            near_threshold=anchor_near_threshold,
            audit_path=audit_path,
        )

        for claim in extracted:
            # SHACL validate — validate_graph returns (conforms, report) tuple
            conforms, report = indicium.validate_graph(claims_to_graph([claim]))
            if not conforms:
                logger.warning(
                    "claim_shacl_invalid",
                    paper_id=paper_id,
                    preview=str(report)[:160],
                )
                continue

            c_iri = claim_iri(kb_name, claim)
            anchor = claim.get("_anchor") or {}
            matched_index = anchor.get("matched_index")
            ev_grade = (claim.get("evidence") or [{}])[0].get("evidence_type") or "knowledge"

            if matched_index is not None and matched_index < len(passages):
                # Content-matched passage: bind here and write trustworthy
                # position + quote selectors (offsets now come from the right chunk).
                pg = passages[matched_index]
                ps_iri = passage_iri(kb_name, paper_id, pg.get("chunk_idx", matched_index))
                store.add(ps_iri, IRI_RDF_TYPE, IRI_TEXT_CHUNK)
                if pg.get("char_start") is not None and pg.get("char_end") is not None:
                    selector = f"{ps_iri}#sel"
                    store.add(ps_iri, f"{OA_NS}hasSelector", selector)
                    store.add(selector, IRI_RDF_TYPE, f"{OA_NS}TextPositionSelector")
                    store.add(
                        selector,
                        f"{OA_NS}start",
                        ("literal", str(pg["char_start"]), f"{XSD_NS}nonNegativeInteger"),
                    )
                    store.add(
                        selector,
                        f"{OA_NS}end",
                        ("literal", str(pg["char_end"]), f"{XSD_NS}nonNegativeInteger"),
                    )
                e_iri = evidence_iri(kb_name, ps_iri, ev_grade)
                store.add(e_iri, IRI_RDF_TYPE, IRI_EVIDENCE)
                store.add(e_iri, IRI_EVIDENCE_TYPE, _eco_iri(ev_grade))
                store.add(e_iri, IRI_DERIVED_FROM_PASSAGE, ps_iri)
                _write_anchor_provenance(store, e_iri, anchor, pg)
            else:
                # Unverified, fail-open kept: an Evidence node with NO passage
                # binding and NO false selector — only the anchorStatus marker.
                e_iri = evidence_iri(kb_name, f"{c_iri}/unanchored", ev_grade)
                store.add(e_iri, IRI_RDF_TYPE, IRI_EVIDENCE)
                store.add(e_iri, IRI_EVIDENCE_TYPE, _eco_iri(ev_grade))
                _write_anchor_provenance(store, e_iri, anchor, None)

            is_new_claim = c_iri not in _seen_claim_iris
            _seen_claim_iris.add(c_iri)

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
            # Indicium v1.3 — claim_status, asserted_by, decision_context
            store.add(c_iri, IRI_CLAIM_STATUS, ("literal", "draft", None))
            store.add(c_iri, IRI_ASSERTED_BY, run)
            for tag in claim.get("decision_context", []):
                store.add(c_iri, IRI_DECISION_CONTEXT, ("literal", str(tag), None))
            store.add(
                c_iri,
                IRI_CREATED,
                ("literal", t0.isoformat(), f"{XSD_NS}dateTime"),
            )
            # Shortcut: ECO IRI directly on the claim for fast SPARQL filtering
            store.add(c_iri, f"{ASB_NS}evidenceTypeIri", _eco_iri(ev_grade))

            claim_record = dict(claim)
            claim_record["_iri"] = c_iri
            claim_record["_local_id"] = c_iri
            claim_record["_paper_id"] = paper_id
            claim_record["_evidence_grade"] = ev_grade
            if is_new_claim:
                all_new_claims.append(claim_record)
                claims_added += 1

        manifest.paper_hashes[paper_id] = compute_paper_hash(paper_texts[paper_id])

    # CiTO classification
    if all_new_claims:
        pairs = build_candidate_pairs(all_new_claims, max_pairs_per_claim=max_pairs_per_claim)
        pairs_classified = len(pairs)
        edges = await classify_pairs(pairs, llm_client=llm_client, model=model)
        for edge in edges:
            store.add_edge_with_confidence(
                edge["from"]["_iri"],
                f"http://purl.org/spar/cito/{edge['label']}",
                edge["to"]["_iri"],
                confidence=edge["confidence"],
                run_iri=run,
                graph=cito_g,
            )
            edges_added += 1

    manifest.indicium_schema_version = indicium.__version__
    manifest.builder_version = builder_version
    manifest.last_build_iso = t0.isoformat()
    write_manifest(manifest)

    duration = (_dt.datetime.now(_dt.UTC) - t0).total_seconds()
    log_path = manifest_path(kb_name).parent / "build_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "iso": t0.isoformat(),
                    "papers": len(to_process),
                    "claims_added": claims_added,
                    "edges_added": edges_added,
                    "pairs_classified": pairs_classified,
                    "duration_seconds": duration,
                    "model": model,
                }
            )
            + "\n"
        )

    return BuildResult(
        kb_name=kb_name,
        claims_added=claims_added,
        edges_added=edges_added,
        pairs_classified=pairs_classified,
        papers_processed=len(to_process),
        duration_seconds=duration,
    )
