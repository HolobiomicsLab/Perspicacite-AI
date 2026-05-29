"""Phase 4 — provenance strategy (sentence-bound citations + JSON-LD sidecar).

Pipeline
--------
1. Build retriever via BaseRAGMode._build_kb_retriever.
2. Search chunks from the KB.
3. Format passages and call extract_claims (pipeline/claims.py).
4. SHACL-validate each claim (indicium.validate_graph — graceful fallback when
   indicia extra not installed).
5. Upsert valid claim nodes into a ClaimGraphStore (backend="memory",
   query-scoped, no file persistence).
6. Compose narrative via LLM (stage="reasoning.provenance.compose").
   The LLM returns JSON with a ``narrative`` list — each item has a
   ``sentence`` and a ``supports`` list of claim placeholder IRIs.
7. Render the narrative as Markdown with [n] footnotes anchored to source DOIs.
8. Emit JSON-LD sidecar via indicium.adapters.kgmemory.claim_jsonld,
   delivered as a ``status`` event with kind="sidecar".
9. Emit ``source`` events for every cited paper.
10. Emit ``done`` event.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGRequest, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.reasoning.provenance")

_COMPOSE_PROMPT = """\
You are a scientific writing assistant. Given the claims below (each identified \
by a placeholder IRI like __CLAIM_0__), compose a concise, factual answer to the \
research question. Every sentence MUST cite at least one claim using its placeholder \
IRI in the ``supports`` list.

Research question: {query}

Claims:
{claims_block}

Return ONLY valid JSON with this exact structure:
{{
  "narrative": [
    {{"sentence": "<sentence text>", "supports": ["__CLAIM_0__", ...]}},
    ...
  ],
  "claims_used": ["__CLAIM_0__", ...]
}}"""


# ---------------------------------------------------------------------------
# Internal helpers
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
    return f"urn:perspicacite:provenance:claim:{idx}"


def _placeholder(idx: int) -> str:
    return f"__CLAIM_{idx}__"


def _add_claims_to_store(
    store: Any,
    claims: list[dict],
    kb_name: str,
) -> list[tuple[str, str, dict]]:
    """Add claims to the store and return list of (placeholder, iri, claim)."""
    entries: list[tuple[str, str, dict]] = []
    graph_iri = f"kb://{kb_name}/provenance"
    for idx, claim in enumerate(claims):
        iri = _make_claim_iri(idx)
        ph = _placeholder(idx)
        claim_with_id = dict(claim, id=f"provenance:{idx}")
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


def _render_markdown(
    narrative: list[dict],
    ph_to_doi: dict[str, str],
) -> str:
    """Render narrative sentences with [n] footnote markers.

    Each sentence's ``supports`` list contains claim placeholders; we resolve
    them to DOIs, deduplicate per-sentence, and assign incrementing footnote
    numbers across the whole answer.
    """
    doi_to_fn: dict[str, int] = {}
    fn_counter = 0
    lines: list[str] = []
    footnotes: list[str] = []

    for item in narrative:
        sentence = item.get("sentence", "")
        supports = item.get("supports") or []
        refs: list[int] = []
        seen_in_sentence: set[str] = set()
        for ph in supports:
            doi = ph_to_doi.get(ph, "")
            if not doi or doi in seen_in_sentence:
                continue
            seen_in_sentence.add(doi)
            if doi not in doi_to_fn:
                fn_counter += 1
                doi_to_fn[doi] = fn_counter
                footnotes.append(f"[{fn_counter}] {doi}")
            refs.append(doi_to_fn[doi])
        if refs:
            marker = "".join(f"[{r}]" for r in sorted(refs))
            lines.append(f"{sentence} {marker}")
        else:
            lines.append(sentence)

    body = " ".join(lines)
    if footnotes:
        body += "\n\n" + "\n".join(footnotes)
    return body


def _build_source_reference(meta: dict, score: float) -> SourceReference:
    return SourceReference(
        title=meta.get("title") or meta.get("doi") or "Unknown",
        authors=meta.get("authors") or [],
        year=meta.get("year"),
        doi=meta.get("doi"),
        relevance_score=min(1.0, max(0.0, score)),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_provenance_stream(
    *,
    request: RAGRequest,
    llm: Any,
    vector_store: Any,
    embedding_provider: Any,
    config: Any,
    session_store: Any,
) -> AsyncIterator[StreamEvent]:
    """Stream a sentence-bound-citation answer with a JSON-LD sidecar."""
    # Try to import optional indicium dependencies
    try:
        from indicium.adapters.kgmemory import claim_jsonld

        _has_indicium = True
    except ImportError:
        claim_jsonld = None
        _has_indicium = False

    _claim_graph_store_cls: Any = None
    try:
        from perspicacite.indicium_layer.store import ClaimGraphStore

        _claim_graph_store_cls = ClaimGraphStore
        _has_store = True
    except ImportError:
        _has_store = False

    try:
        # --- Step 1: retrieve chunks ---
        yield StreamEvent.status("Reasoning (provenance): retrieving passages…")

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
        yield StreamEvent.status("Reasoning (provenance): extracting claims…")
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
                            "provenance_shacl_partial",
                            report=_report[:200],
                        )
                    # Keep all claims even if SHACL has warnings — the
                    # validation report is advisory at query-time.
                    valid_claims = raw_claims
                except Exception as exc:
                    logger.warning("provenance_shacl_failed", error=str(exc))
                    valid_claims = raw_claims
            else:
                valid_claims = raw_claims

        # --- Step 4: upsert into ephemeral ClaimGraphStore ---
        kb_name = getattr(request, "kb_name", "default") or "default"
        entries: list[tuple[str, str, dict]] = []
        if valid_claims and _has_store and _claim_graph_store_cls is not None:
            store = _claim_graph_store_cls(kb_name, backend="memory")
            entries = _add_claims_to_store(store, valid_claims, kb_name)
        elif valid_claims:
            # Fallback: create entries without a store
            for idx, claim in enumerate(valid_claims):
                ph = _placeholder(idx)
                iri = _make_claim_iri(idx)
                entries.append((ph, iri, dict(claim, id=f"provenance:{idx}")))

        # --- Step 5: build placeholder-to-DOI map ---
        ph_to_doi: dict[str, str] = {}
        for ph, _iri, claim in entries:
            evidence = claim.get("evidence", [])
            doi = ""
            if isinstance(evidence, list) and evidence:
                doi = evidence[0].get("doi", "") or ""
            if not doi:
                doi = claim.get("source_doi", "") or ""
            ph_to_doi[ph] = doi

        # --- Step 6: compose narrative via LLM ---
        yield StreamEvent.status("Reasoning (provenance): composing narrative…")

        if entries:
            claims_block = "\n".join(
                f"{ph}: {c.get('subject', '')} {c.get('qualifier', '')} {c.get('object', '')} "
                f"[context: {c.get('context', '')}]"
                for ph, _iri, c in entries
            )
        else:
            # No claims extracted — fall back to plain passage summary
            claims_block = "\n".join(
                f"[passage {i}]: {_chunk_text(ch)[:300]}" for i, ch in enumerate(chunks[:6])
            )

        compose_prompt = _COMPOSE_PROMPT.format(
            query=request.query,
            claims_block=claims_block,
        )
        messages = [{"role": "user", "content": compose_prompt}]
        raw_compose = await llm.complete(messages=messages, stage="reasoning.provenance.compose")
        if not isinstance(raw_compose, str):
            raw_compose = str(raw_compose)

        # --- Step 7: render markdown with [n] footnotes ---
        narrative: list[dict] = []
        try:
            compose_data = json.loads(raw_compose)
            narrative = compose_data.get("narrative") or []
        except (json.JSONDecodeError, TypeError, AttributeError):
            narrative = [{"sentence": raw_compose, "supports": []}]

        if not narrative:
            narrative = [{"sentence": raw_compose, "supports": []}]

        markdown = _render_markdown(narrative, ph_to_doi)
        yield StreamEvent.content(markdown)

        # --- Step 8: JSON-LD sidecar ---
        jsonld_claims: list[dict] = []
        if _has_indicium and claim_jsonld is not None and entries:
            for _ph, _iri, claim in entries:
                try:
                    jsonld_claims.append(claim_jsonld(claim))
                except Exception as exc:
                    logger.warning("provenance_jsonld_failed", error=str(exc))
        elif entries:
            # Minimal sidecar without indicium
            for _ph, iri, claim in entries:
                jsonld_claims.append(
                    {
                        "@id": iri,
                        "@type": "urn:indicium:Claim",
                        "subject": claim.get("subject", ""),
                        "object": claim.get("object", ""),
                    }
                )

        yield StreamEvent(
            event="status",
            data=json.dumps(
                {
                    "message": "Reasoning (provenance): JSON-LD sidecar attached.",
                    "kind": "sidecar",
                    "claims": jsonld_claims,
                }
            ),
        )

        # --- Step 9: emit source events ---
        seen_dois: set[str] = set()
        for chunk in chunks:
            meta = _chunk_meta(chunk)
            doi = meta.get("doi") or meta.get("paper_id") or ""
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            yield StreamEvent.source(_build_source_reference(meta, _chunk_score(chunk)))

        # --- Step 10: done ---
        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="reasoning",
            iterations=1,
        )

    except Exception as exc:
        logger.error("provenance_stream_error", error=str(exc))
        yield StreamEvent(event="error", data=json.dumps({"message": str(exc)}))
