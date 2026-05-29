"""Phase 3 — typed contradiction strategy (CiTO-graded consensus/dispute/open brief).

Pipeline
--------
1. Build retriever via BaseRAGMode._build_kb_retriever.
2. Search chunks from the KB.
3. Format passages and call extract_claims (pipeline/claims.py).
4. SHACL-validate each claim (indicium.validate_graph — graceful fallback when
   indicia extra not installed).
5. Upsert valid claim nodes into a ClaimGraphStore (backend="memory",
   query-scoped, no file persistence).
6. Cluster valid claims by SuperPattern signature:
   (tuple(lemmas(subject)), relation.lower(), tuple(lemmas(object))).
7. For each cluster with >=2 claims: read existing CiTO edges from store via
   cito_edges_for_claim; if none, classify pairs on-the-fly via classify_pairs
   and write edges.
8. Compose typed brief via LLM (stage="reasoning.typed_contradiction.brief").
   The LLM returns JSON with ``consensus``, ``dispute``, and ``open`` lists.
9. Render markdown with "## Consensus", "## Dispute", "## Open questions" sections.
10. Emit status event with typed=True, cito_edges=[...], clusters_count=N,
    claims_count=M.
11. Emit source events for every cited paper.
12. Emit done event.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGRequest, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.reasoning.typed_contradiction")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BRIEF_PROMPT = """\
You are a scientific analyst. Given the typed claims below (each from a specific \
paper), classify them into three buckets:
- "consensus": claims that are mutually consistent or reinforcing
- "dispute": claims that directly contradict each other
- "open": claims that raise unresolved questions

Research question: {query}

Claims (each entry: placeholder | subject | qualifier | relation | object | context | doi):
{claims_block}

CiTO edges already classified (pair label confidence):
{edges_block}

Return ONLY valid JSON with this exact structure:
{{
  "consensus": [
    {{"summary": "<text>", "claim_iris": ["__C0__", ...],
      "papers": ["<doi>", ...], "eco": "<evidence_type>"}}
  ],
  "dispute": [
    {{"summary": "<text>", "claim_iris": ["__C0__", "__C1__"],
      "papers": ["<doi>", ...], "eco": "<evidence_type>"}}
  ],
  "open": [
    {{"summary": "<text>", "claim_iris": ["__C0__", ...],
      "papers": ["<doi>", ...], "eco": "<evidence_type>"}}
  ]
}}"""

# ---------------------------------------------------------------------------
# Internal helpers — shared with provenance pattern
# ---------------------------------------------------------------------------

_CLAIM_NS = "https://asb.holobiomics.org/ns/asb#"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _chunk_text(chunk: dict) -> str:
    return chunk.get("text", "") or ""


def _chunk_meta(chunk: dict) -> dict:
    meta = chunk.get("metadata") or {}
    if hasattr(meta, "model_dump"):
        return meta.model_dump()
    return dict(meta)


def _chunk_score(chunk: dict) -> float:
    return float(chunk.get("score", 0.0))


def _passages_for_extract(chunks: list[dict]) -> list[dict]:
    """Convert raw KB chunks into the shape expected by extract_claims."""
    out = []
    for c in chunks:
        meta = _chunk_meta(c)
        out.append(
            {
                "chunk_text": _chunk_text(c),
                "source": {
                    "doi": meta.get("doi") or meta.get("paper_id") or "",
                    "title": meta.get("title") or "",
                    "year": meta.get("year"),
                },
            }
        )
    return out


def _make_claim_iri(idx: int) -> str:
    return f"urn:perspicacite:typed_contradiction:claim:{idx}"


def _placeholder(idx: int) -> str:
    return f"__C{idx}__"


def _add_claims_to_store(
    store: Any,
    claims: list[dict],
    kb_name: str,
) -> list[tuple[str, str, dict]]:
    """Add claims to the store and return list of (placeholder, iri, claim)."""
    entries: list[tuple[str, str, dict]] = []
    graph_iri = f"kb://{kb_name}/typed_contradiction"
    for idx, claim in enumerate(claims):
        iri = _make_claim_iri(idx)
        ph = _placeholder(idx)
        claim_with_id = dict(claim, id=f"typed_contradiction:{idx}")
        store.add(iri, _RDF_TYPE, f"{_CLAIM_NS}Claim", graph=graph_iri)
        for slot in ("context", "subject", "qualifier", "relation", "object"):
            if claim.get(slot):
                store.add(
                    iri,
                    f"{_CLAIM_NS}{slot}",
                    ("literal", claim[slot], None),
                    graph=graph_iri,
                )
        entries.append((ph, iri, claim_with_id))
    return entries


def _build_source_reference(meta: dict, score: float) -> SourceReference:
    return SourceReference(
        title=meta.get("title") or meta.get("doi") or "Unknown",
        authors=meta.get("authors") or [],
        year=meta.get("year"),
        doi=meta.get("doi"),
        relevance_score=min(1.0, max(0.0, score)),
    )


# ---------------------------------------------------------------------------
# SuperPattern clustering helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "to",
        "for",
        "and",
        "or",
        "with",
        "at",
        "by",
        "from",
        "that",
        "this",
        "these",
        "those",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
    }
)


def _lemmas(text: str) -> tuple[str, ...]:
    """Minimal lemmatisation: lowercase, remove stopwords, sort tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    meaningful = [t for t in tokens if t not in _STOPWORDS]
    return tuple(sorted(meaningful)) if meaningful else tuple(tokens)


def _cluster_key(claim: dict) -> tuple:
    """Build the SuperPattern signature for clustering."""
    subject = claim.get("subject", "") or ""
    relation = claim.get("relation", "") or ""
    obj = claim.get("object", "") or ""
    return (_lemmas(subject), relation.lower().strip(), _lemmas(obj))


def _cluster_claims(
    entries: list[tuple[str, str, dict]],
) -> dict[tuple, list[tuple[str, str, dict]]]:
    """Group entries by SuperPattern cluster key."""
    clusters: dict[tuple, list[tuple[str, str, dict]]] = {}
    for entry in entries:
        _ph, _iri, claim = entry
        key = _cluster_key(claim)
        clusters.setdefault(key, []).append(entry)
    return clusters


# ---------------------------------------------------------------------------
# CiTO edge helpers
# ---------------------------------------------------------------------------


def _try_read_edges_from_store(
    store: Any,
    kb_name: str,
    claim_iri: str,
) -> list[dict]:
    """Read existing CiTO edges from store — graceful fallback."""
    try:
        from perspicacite.indicium_layer.queries import cito_edges_for_claim

        return cito_edges_for_claim(store, kb_name, claim_iri)
    except Exception as exc:
        logger.debug("cito_edges_read_failed", iri=claim_iri, error=str(exc))
        return []


async def _classify_cluster_pairs(
    cluster: list[tuple[str, str, dict]],
    store: Any,
    kb_name: str,
    llm: Any,
) -> list[dict]:
    """Read or classify CiTO edges for all pairs in a cluster."""
    all_edges: list[dict] = []

    # First try to read from store for each claim in the cluster
    for _ph, iri, _claim in cluster:
        if store is not None:
            edges = _try_read_edges_from_store(store, kb_name, iri)
            all_edges.extend(edges)

    # If no stored edges and we have >=2 entries, classify on the fly
    if not all_edges and len(cluster) >= 2:
        try:
            from perspicacite.indicium_layer.cito_classifier import classify_pairs

            pairs = [(entry_a[2], entry_b[2]) for entry_a, entry_b in combinations(cluster, 2)]
            new_edges = await classify_pairs(pairs, llm_client=llm)
            all_edges.extend(new_edges)
        except Exception as exc:
            logger.warning("cito_classify_failed", error=str(exc))

    return all_edges


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_brief_markdown(brief_data: dict) -> str:
    """Render the LLM brief JSON into a structured Markdown string."""
    sections = [
        ("consensus", "## Consensus"),
        ("dispute", "## Dispute"),
        ("open", "## Open questions"),
    ]
    lines: list[str] = []
    for key, heading in sections:
        lines.append(heading)
        items = brief_data.get(key) or []
        if items:
            for item in items:
                summary = item.get("summary", "")
                papers = item.get("papers") or []
                eco = item.get("eco", "")
                paper_refs = ", ".join(papers) if papers else ""
                eco_str = f" [{eco}]" if eco else ""
                if paper_refs:
                    lines.append(f"- {summary}{eco_str} ({paper_refs})")
                else:
                    lines.append(f"- {summary}{eco_str}")
        else:
            lines.append("- *(none identified)*")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_typed_contradiction_stream(
    *,
    request: RAGRequest,
    llm: Any,
    vector_store: Any,
    embedding_provider: Any,
    config: Any,
    session_store: Any,
) -> AsyncIterator[StreamEvent]:
    """Stream a CiTO-typed consensus/dispute/open brief."""
    # Try to import optional indicium dependencies
    _claim_graph_store_cls: Any = None
    _has_store = False
    try:
        from perspicacite.indicium_layer.store import ClaimGraphStore

        _claim_graph_store_cls = ClaimGraphStore
        _has_store = True
    except ImportError:
        pass

    _has_indicium = False
    try:
        import indicium  # noqa: F401

        _has_indicium = True
    except ImportError:
        pass

    try:
        # --- Step 1: retrieve chunks ---
        yield StreamEvent.status("Reasoning (typed_contradiction): retrieving passages…")

        # Use a minimal concrete subclass so abstract-method checks pass,
        # then call _build_kb_retriever directly (can be monkeypatched in tests).
        class _MinimalMode(BaseRAGMode):
            async def execute(self, *a, **kw):
                raise NotImplementedError

            async def execute_stream(self, *a, **kw):
                raise NotImplementedError

        mode_obj = _MinimalMode(config)
        retriever = mode_obj._build_kb_retriever(request, vector_store, embedding_provider)
        chunks: list[dict] = await retriever.search(request.query, top_k=20)

        if not chunks:
            yield StreamEvent.content("No relevant passages found in the knowledge base.")
            yield StreamEvent.done(
                conversation_id="", tokens_used=0, mode="reasoning", iterations=1
            )
            return

        # --- Step 2: extract claims ---
        yield StreamEvent.status("Reasoning (typed_contradiction): extracting claims…")
        passages = _passages_for_extract(chunks)

        from perspicacite.pipeline.claims import extract_claims

        raw_claims = await extract_claims(
            llm_client=llm,
            passages=passages,
            context=request.query,
        )

        # --- Step 3: SHACL validate (graceful fallback) ---
        valid_claims: list[dict] = []
        if raw_claims:
            if _has_indicium:
                try:
                    from perspicacite.pipeline.claims import validate_claims

                    conforms, _report = validate_claims(raw_claims)
                    if not conforms:
                        logger.warning(
                            "typed_contradiction_shacl_partial",
                            report=_report[:200],
                        )
                    valid_claims = raw_claims
                except Exception as exc:
                    logger.warning("typed_contradiction_shacl_failed", error=str(exc))
                    valid_claims = raw_claims
            else:
                valid_claims = raw_claims

        # --- Step 4: upsert into ephemeral ClaimGraphStore ---
        kb_name = getattr(request, "kb_name", "default") or "default"
        store: Any = None
        entries: list[tuple[str, str, dict]] = []
        if valid_claims and _has_store and _claim_graph_store_cls is not None:
            store = _claim_graph_store_cls(kb_name, backend="memory")
            entries = _add_claims_to_store(store, valid_claims, kb_name)
        elif valid_claims:
            # Fallback: create entries without a store
            for idx, claim in enumerate(valid_claims):
                ph = _placeholder(idx)
                iri = _make_claim_iri(idx)
                entries.append((ph, iri, dict(claim, id=f"typed_contradiction:{idx}")))

        # --- Step 5: cluster by SuperPattern ---
        yield StreamEvent.status("Reasoning (typed_contradiction): clustering claims…")
        clusters = _cluster_claims(entries)

        # --- Step 6: CiTO edge classification per cluster ---
        yield StreamEvent.status("Reasoning (typed_contradiction): classifying CiTO relations…")
        all_edges: list[dict] = []
        for cluster_entries in clusters.values():
            if len(cluster_entries) >= 2:
                edges = await _classify_cluster_pairs(cluster_entries, store, kb_name, llm)
                all_edges.extend(edges)

        # --- Step 7: compose typed brief via LLM ---
        yield StreamEvent.status("Reasoning (typed_contradiction): composing brief…")

        if entries:
            claims_block = "\n".join(
                f"{ph} | {c.get('subject', '')} | {c.get('qualifier', '')} | "
                f"{c.get('relation', '')} | {c.get('object', '')} | "
                f"{c.get('context', '')} | "
                + ((c.get("evidence") or [{}])[0].get("doi", "") or c.get("source_doi", ""))
                for ph, _iri, c in entries
            )
        else:
            claims_block = "\n".join(
                f"[passage {i}]: {_chunk_text(ch)[:300]}" for i, ch in enumerate(chunks[:6])
            )

        if all_edges:
            edges_block = "\n".join(
                f"- {e.get('from', {}).get('subject', '?')} "
                f"[{e.get('label', '?')}] "
                f"{e.get('to', {}).get('subject', '?')} "
                f"(conf={e.get('confidence', 0):.2f})"
                for e in all_edges[:20]
            )
        else:
            edges_block = "(none classified)"

        brief_prompt = _BRIEF_PROMPT.format(
            query=request.query,
            claims_block=claims_block,
            edges_block=edges_block,
        )
        messages = [{"role": "user", "content": brief_prompt}]
        raw_brief = await llm.complete(
            messages=messages,
            stage="reasoning.typed_contradiction.brief",
        )
        if not isinstance(raw_brief, str):
            raw_brief = str(raw_brief)

        # --- Step 8: render markdown with three-bucket structure ---
        brief_data: dict = {}
        try:
            brief_data = json.loads(raw_brief)
        except (json.JSONDecodeError, TypeError, AttributeError):
            brief_data = {
                "consensus": [],
                "dispute": [],
                "open": [{"summary": raw_brief, "claim_iris": [], "papers": []}],
            }

        # Ensure all three buckets are present (even if LLM omitted some)
        brief_data.setdefault("consensus", [])
        brief_data.setdefault("dispute", [])
        brief_data.setdefault("open", [])

        markdown = _render_brief_markdown(brief_data)
        yield StreamEvent.content(markdown)

        # --- Step 9: emit brief status with typed=True + cito_edges ---
        yield StreamEvent(
            event="status",
            data=json.dumps(
                {
                    "message": "Typed contradiction brief attached.",
                    "kind": "brief",
                    "typed": True,
                    "cito_edges": all_edges,
                    "clusters_count": len(clusters),
                    "claims_count": len(valid_claims),
                }
            ),
        )

        # --- Step 10: emit source events ---
        seen_dois: set[str] = set()
        for chunk in chunks:
            meta = _chunk_meta(chunk)
            doi = meta.get("doi") or meta.get("paper_id") or ""
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            yield StreamEvent.source(_build_source_reference(meta, _chunk_score(chunk)))

        # --- Step 11: done ---
        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="reasoning",
            iterations=1,
        )

    except Exception as exc:
        logger.error("typed_contradiction_stream_error", error=str(exc))
        yield StreamEvent(event="error", data=json.dumps({"message": str(exc)}))
