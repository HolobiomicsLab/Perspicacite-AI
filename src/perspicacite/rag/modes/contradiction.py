"""Contradiction-detection RAG mode.

Surfaces agreement / disagreement / open questions across papers by:
1. Retrieving chunks for the query.
2. Grouping chunks by paper.
3. Asking the LLM to summarise each paper's claims.
4. Clustering the summaries into consensus / disagreement / open buckets.
5. Streaming a structured three-section brief.

Degrades gracefully when fewer than MIN_PAPERS_FOR_ANALYSIS papers are found,
producing a note and a normal advanced-style answer instead.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from perspicacite.logging import get_logger
from perspicacite.provenance.context import get_collector

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from perspicacite.models.rag import (
    RAGMode,
    RAGRequest,
    RAGResponse,
    SourceReference,
    StreamEvent,
)
from perspicacite.rag import prompts as _prompts
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.multimodal import wrap_messages_for_chunks
from perspicacite.rag.paper_metadata_codec import decode_paper_metadata_json
from perspicacite.rag.telemetry import emit_phase

logger = get_logger("perspicacite.rag.modes.contradiction")

MIN_PAPERS_FOR_ANALYSIS = 3
RETRIEVAL_TOP_K = 25


class ContradictionRAGMode(BaseRAGMode):
    """Contradiction-detection RAG mode."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.settings = config.rag_modes.contradiction

    # ------------------------------------------------------------------
    # Retrieval seam (monkeypatched in tests)
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        request: RAGRequest,
        vector_store: Any,
        embedding_provider: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks from the KB (or multiple KBs), returning raw dicts from search()."""
        dkb = self._build_kb_retriever(request, vector_store, embedding_provider)
        results = await dkb.search(request.query, top_k=RETRIEVAL_TOP_K)
        if getattr(request, "recency_weight", None):
            from perspicacite.retrieval.recency import apply_recency_weighting

            results = apply_recency_weighting(
                results,
                request.recency_weight,
                getattr(request, "recency_half_life_years", None),
            )
        return results

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_by_paper(self, chunks: list[Any]) -> dict[str, list[Any]]:
        """Group chunks (dicts or objects) by paper_id."""
        out: dict[str, list[Any]] = {}
        for c in chunks:
            if isinstance(c, dict):
                meta = c.get("metadata") or {}
                if not isinstance(meta, dict):
                    # Pydantic / dataclass metadata object
                    pid = getattr(meta, "paper_id", None) or getattr(meta, "doi", None) or "?"
                else:
                    pid = meta.get("paper_id") or meta.get("doi") or "?"
            else:
                meta = getattr(c, "metadata", {}) or {}
                pid = meta.get("paper_id") or meta.get("doi") or "?"
            out.setdefault(pid, []).append(c)
        return out

    def _chunk_text(self, chunk: Any) -> str:
        """Extract text from a chunk (dict or object)."""
        if isinstance(chunk, dict):
            return chunk.get("text", "") or ""
        return getattr(chunk, "text", "") or ""

    def _chunk_score(self, chunk: Any) -> float:
        """Extract relevance score from a chunk."""
        if isinstance(chunk, dict):
            return float(chunk.get("score", 0.0))
        return float(getattr(chunk, "score", 0.0))

    def _chunk_meta(self, chunk: Any) -> dict[str, Any]:
        """Extract metadata dict from a chunk."""
        if isinstance(chunk, dict):
            meta = chunk.get("metadata") or {}
        else:
            meta = getattr(chunk, "metadata", {}) or {}
        if not isinstance(meta, dict):
            # Pydantic / dataclass object → convert to dict
            try:
                return meta.model_dump()
            except AttributeError:
                return dict(vars(meta))
        return dict(meta)

    def _chunk_kb_name(self, chunk: Any) -> str | None:
        """Extract the originating KB display name from a chunk, if present."""
        if isinstance(chunk, dict):
            return chunk.get("kb_name")
        return getattr(chunk, "kb_name", None)

    # ------------------------------------------------------------------
    # Per-paper claim summarisation
    # ------------------------------------------------------------------

    async def _summarize_claims(
        self,
        by_paper: dict[str, list[Any]],
        query: str,
        llm: Any,
        cap: int,
    ) -> list[dict[str, Any]]:
        """Ask LLM for 2-4 bullet claims per paper (up to *cap* papers)."""
        summaries: list[dict[str, Any]] = []
        for pid in list(by_paper.keys())[:cap]:
            chunks = by_paper[pid]
            excerpts = "\n---\n".join(self._chunk_text(c)[:600] for c in chunks[:4])
            meta = self._chunk_meta(chunks[0])
            title = meta.get("title") or pid
            doi = meta.get("doi") or pid
            prompt = _prompts.CONTRADICTION_CLAIM_SUMMARY_PROMPT.format(
                query=query,
                title=title,
                excerpts=excerpts,
            )
            try:
                claims = await llm.complete(messages=[{"role": "user", "content": prompt}], stage="contradiction.summarize")
                if not isinstance(claims, str):
                    claims = str(claims)
            except Exception as exc:
                logger.warning("claim_summary_failed", paper_id=pid, error=str(exc))
                claims = "(summary unavailable)"
            summaries.append(
                {
                    "paper_id": pid,
                    "title": title,
                    "doi": doi,
                    "claims": claims,
                }
            )
        return summaries

    # ------------------------------------------------------------------
    # Cluster claims into buckets
    # ------------------------------------------------------------------

    async def _cluster_claims(
        self,
        query: str,
        paper_summaries: list[dict[str, Any]],
        llm: Any,
    ) -> dict[str, Any]:
        """Cluster per-paper summaries into consensus / disagreement / open."""
        summaries_text = "\n\n".join(
            f"Paper [{s['title']} | {s['doi']}]:\n{s['claims']}" for s in paper_summaries
        )
        prompt = _prompts.CONTRADICTION_CLUSTER_PROMPT.format(
            query=query,
            summaries=summaries_text,
        )
        empty: dict[str, Any] = {"consensus": [], "disagreement": [], "open": []}
        try:
            raw = await llm.complete(messages=[{"role": "user", "content": prompt}], stage="contradiction.cluster")
            if not isinstance(raw, str):
                raw = str(raw)
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[-1].lstrip("json").strip()
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            clusters: dict[str, Any] = json.loads(raw)
            # Ensure expected keys exist
            clusters.setdefault("consensus", [])
            clusters.setdefault("disagreement", [])
            clusters.setdefault("open", [])
            return clusters
        except Exception as exc:
            logger.warning("cluster_claims_failed", error=str(exc))
            return empty

    # ------------------------------------------------------------------
    # Synthesis streaming
    # ------------------------------------------------------------------

    async def _synthesize_stream(
        self,
        request: RAGRequest,
        query: str,
        clusters: dict[str, Any],
        paper_summaries: list[dict[str, Any]],
        documents: list[Any],
        llm: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the structured three-section brief."""
        summaries_text = "\n\n".join(
            f"Paper [{s['title']} | {s['doi']}]:\n{s['claims']}" for s in paper_summaries
        )
        prompt = _prompts.CONTRADICTION_SYNTHESIS_PROMPT.format(
            query=query,
            clusters=json.dumps(clusters, indent=2),
            summaries=summaries_text,
        )
        base_messages = [{"role": "user", "content": prompt}]
        messages = wrap_messages_for_chunks(
            base_messages=base_messages,
            chunks=documents,
            model=getattr(request, "model", None),
            config=self.config,
        )
        try:
            async for chunk in llm.stream(
                messages=messages,
                model=getattr(request, "model", None) or "",
                provider=getattr(request, "provider", None),
                # 0.7 matches the complete() default that the system was accidentally using
                # before the missing-model bug was fixed. 0.3 was too conservative and
                # produced INSUFFICIENT EVIDENCE for almost every claim.
                max_tokens=2000, temperature=0.7, stage="contradiction.synthesis",
            ):
                yield StreamEvent.content(chunk)
        except AttributeError:
            # LLM has no stream() method; fall back to one-shot complete()
            try:
                answer = await llm.complete(messages=messages, stage="contradiction.synthesis")
                if not isinstance(answer, str):
                    answer = str(answer)
                yield StreamEvent.content(answer)
            except Exception as exc:
                logger.error("synthesis_failed", error=str(exc))
                yield StreamEvent.content(
                    f"(Synthesis failed: {exc}. See per-paper summaries above.)"
                )
        except Exception as exc:
            logger.error("synthesis_stream_failed", error=str(exc))
            # Try non-streaming fallback
            try:
                answer = await llm.complete(messages=messages, stage="contradiction.synthesis")
                if not isinstance(answer, str):
                    answer = str(answer)
                yield StreamEvent.content(answer)
            except Exception as exc2:
                logger.error("synthesis_fallback_failed", error=str(exc2))
                yield StreamEvent.content(
                    f"(Synthesis failed: {exc2}. See per-paper summaries above.)"
                )

    # ------------------------------------------------------------------
    # Fallback (< MIN_PAPERS)
    # ------------------------------------------------------------------

    async def _fallback_answer_stream(
        self,
        request: RAGRequest,
        chunks: list[Any],
        llm: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Simple one-shot answer from raw chunks when there are too few papers."""
        if not chunks:
            yield StreamEvent.content("No relevant documents found.")
            return
        context = "\n\n---\n\n".join(self._chunk_text(c)[:800] for c in chunks[:8])
        base_messages = self._build_messages(
            query=request.query,
            context=context,
        )
        messages = wrap_messages_for_chunks(
            base_messages=base_messages,
            chunks=chunks,
            model=getattr(request, "model", None),
            config=self.config,
        )
        try:
            async for chunk in llm.stream(
                messages=messages,
                model=getattr(request, "model", None) or "",
                provider=getattr(request, "provider", None),
                max_tokens=1500, temperature=0.3, stage="contradiction.fallback",
            ):
                yield StreamEvent.content(chunk)
        except AttributeError:
            try:
                answer = await llm.complete(messages=messages, stage="contradiction.fallback")
                if not isinstance(answer, str):
                    answer = str(answer)
                yield StreamEvent.content(answer)
            except Exception as exc:
                yield StreamEvent.content(f"(Answer generation failed: {exc})")
        except Exception:
            try:
                answer = await llm.complete(messages=messages, stage="contradiction.fallback")
                if not isinstance(answer, str):
                    answer = str(answer)
                yield StreamEvent.content(answer)
            except Exception as exc2:
                yield StreamEvent.content(f"(Answer generation failed: {exc2})")

    # ------------------------------------------------------------------
    # Build sources list
    # ------------------------------------------------------------------

    def _build_sources(
        self,
        by_paper: dict[str, list[Any]],
        real_papers: list[str],
    ) -> list[SourceReference]:
        sources: list[SourceReference] = []
        for pid in real_papers:
            chunks = by_paper[pid]
            meta = self._chunk_meta(chunks[0])
            kb_name = self._chunk_kb_name(chunks[0])
            sources.append(
                SourceReference(
                    title=meta.get("title") or pid,
                    authors=meta.get("authors"),  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                    year=meta.get("year"),
                    doi=meta.get("doi"),
                    relevance_score=min(1.0, max(0.0, self._chunk_score(chunks[0]))),
                    kb_name=kb_name,
                    paper_id=meta.get("paper_id"),
                    metadata=decode_paper_metadata_json(meta),
                )
            )
        return sources

    # ------------------------------------------------------------------
    # execute_stream — main streaming entry point
    # ------------------------------------------------------------------

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute contradiction-detection RAG with streaming."""
        _phase_sink = getattr(request, "telemetry_sink", None)
        try:
            emit_phase(_phase_sink, phase="search", state="running")
            yield StreamEvent.status("Contradiction analysis: retrieving documents...")
            chunks = await self._retrieve(request, vector_store, embedding_provider)
            emit_phase(_phase_sink, phase="search", state="done")
            _c = get_collector()
            if _c is not None:
                _c.add_trace("retrieve", detail={"kb_name": request.kb_name, "count": len(chunks)})
                for rank, ch in enumerate(chunks):
                    md = self._chunk_meta(ch)
                    if isinstance(md, dict):
                        doi = md.get("doi")
                        title = md.get("title")
                        content_type = md.get("content_type")
                        pipeline_step = md.get("content_source")
                    elif md is not None:
                        doi = getattr(md, "doi", None)
                        title = getattr(md, "title", None)
                        content_type = getattr(md, "content_type", None)
                        pipeline_step = getattr(md, "content_source", None)
                    else:
                        doi = title = content_type = pipeline_step = None
                    score = float(self._chunk_score(ch) or 0.0)
                    kb_name = self._chunk_kb_name(ch)
                    paper_id = (ch.get("paper_id") if isinstance(ch, dict) else getattr(ch, "paper_id", None))
                    _c.add_retrieval(
                        paper_id=paper_id,
                        doi=doi,
                        title=title,
                        score=score,
                        kb_name=kb_name,
                        content_type=content_type,
                        pipeline_step=pipeline_step,
                        rank=rank,
                        stage_label="contradiction.retrieve",
                    )
            by_paper = self._group_by_paper(chunks)
            if _c is not None:
                _c.add_trace("group_by_paper", detail={"papers": len(by_paper)})
            real_papers = [p for p in by_paper if p != "?"]
            n = len(real_papers)

            # Web fallback: when the KB has too few papers (or none), try
            # a live literature search. Contradiction analysis is most
            # useful when there ARE multiple competing sources to compare,
            # so a fresh aggregator fetch can often rescue an empty KB
            # scenario without forcing the user to switch modes.
            if n < MIN_PAPERS_FOR_ANALYSIS:
                _db_pretty = ", ".join(
                    d.replace("_", " ").title() for d in (request.databases or [])
                ) or "Semantic Scholar, OpenAlex, PubMed"
                yield StreamEvent.status(
                    f"Contradiction analysis: KB has {n} paper(s) "
                    f"(need {MIN_PAPERS_FOR_ANALYSIS}+) — falling back to web "
                    f"literature search across {_db_pretty}…"
                )
                try:
                    from perspicacite.rag.modes.basic import _web_fallback_papers
                    # Telemetry pattern: MCP-attached sink (live notifications)
                    # OR plain list (legacy SSE drain). See Task 2.4.
                    web_telemetry: Any = getattr(request, "telemetry_sink", None) or []
                    web_papers = await _web_fallback_papers(
                        query=request.query,
                        databases=request.databases,
                        max_docs=12,  # need >= 3, fetch a healthy pool
                        config=getattr(self, "config", None),
                        app_state=getattr(request, "app_state", None),
                        reranker_override=getattr(request, "use_reranker", None),
                        telemetry=web_telemetry,
                    )
                    # Drain telemetry into SSE only when we're holding a list
                    # (legacy path); the CallbackTelemetrySink path already
                    # forwarded events live to ctx.report_progress.
                    if isinstance(web_telemetry, list):
                        for _ev in web_telemetry:
                            _k = _ev.get("kind")
                            if _k == "query_rephrased":
                                yield StreamEvent.status_kind(
                                    f"Rewrote search query: '{_ev.get('original','')}' → '{_ev.get('rewritten','')}'",
                                    kind="query_rephrased",
                                    original=_ev.get("original", ""),
                                    rewritten=_ev.get("rewritten", ""),
                                    by=_ev.get("by", "keyword_optimizer"),
                                )
                            elif _k == "provider_progress" and _ev.get("phase") == "start":
                                _provs = ", ".join(
                                    p.replace("_", " ").title() for p in _ev.get("providers", [])
                                )
                                yield StreamEvent.status_kind(
                                    f"Querying databases: {_provs}…",
                                    kind="provider_progress",
                                    phase="start",
                                    providers=_ev.get("providers", []),
                                )
                            elif _k == "provider_progress" and _ev.get("phase") == "done":
                                _bp = _ev.get("by_provider", {}) or {}
                                _msg = ", ".join(
                                    f"{src.replace('_',' ').title()}: {nn}"
                                    for src, nn in sorted(_bp.items(), key=lambda kv: -kv[1])
                                ) if _bp else f"Total {_ev.get('total', 0)} hits"
                                yield StreamEvent.status_kind(
                                    f"Database results — {_msg}",
                                    kind="provider_progress",
                                    phase="done",
                                    total=_ev.get("total", 0),
                                    by_provider=_bp,
                                )
                except Exception as _wf_exc:
                    logger.warning("contradiction_web_fallback_failed", error=str(_wf_exc))
                    web_papers = []

                if len(web_papers) >= MIN_PAPERS_FOR_ANALYSIS:
                    yield StreamEvent.status(
                        f"Contradiction analysis: web search returned "
                        f"{len(web_papers)} paper(s) — running multi-paper claim comparison…"
                    )
                    # Synthesize via the web papers directly: turn each
                    # web paper into a "summary" entry of the right shape
                    # for _cluster_claims, bypassing the chunk-based
                    # by_paper grouping.
                    paper_summaries = []
                    for p in web_papers[:12]:
                        paper_summaries.append({
                            "paper_id": p.get("paper_id") or p.get("doi") or p.get("title"),
                            "title": p.get("title", "Untitled"),
                            "authors": p.get("authors") or [],
                            "year": p.get("year"),
                            "doi": p.get("doi"),
                            "claims": [p.get("abstract") or p.get("chunk_text") or ""],
                        })
                    clusters = await self._cluster_claims(
                        request.query, paper_summaries, llm,
                    )
                    # Reuse the main synthesis stream; chunks=[] is fine
                    # because the synthesizer reads paper_summaries +
                    # clusters for its citations and brief generation.
                    async for ev in self._synthesize_stream(
                        request=request,
                        query=request.query,
                        clusters=clusters,
                        paper_summaries=paper_summaries,
                        documents=[],
                        llm=llm,
                    ):
                        yield ev
                    # Emit source events for transparency.
                    for p in web_papers[:12]:
                        yield StreamEvent.source(
                            SourceReference(
                                title=p.get("title") or "Untitled",
                                authors=p.get("authors") or [],
                                year=p.get("year"),
                                doi=p.get("doi"),
                                url=p.get("url"),
                                source=p.get("source"),
                                source_apis=p.get("source_apis"),
                                sources_all=p.get("sources_all"),
                                enrichment_sources=p.get("enrichment_sources"),
                                relevance_score=p.get("paper_score", 0.5),
                            )
                        )
                    yield StreamEvent.done(
                        conversation_id="",
                        tokens_used=0,
                        mode="contradiction",
                        iterations=1,
                    )
                    return
                # Otherwise: fall through to the original "answer normally" path.
                yield StreamEvent.content(
                    f"_Note: contradiction analysis works best with at least "
                    f"{MIN_PAPERS_FOR_ANALYSIS} papers; "
                    f"only {n} found in this knowledge base and "
                    f"{len(web_papers)} from a live web search. "
                    f"Answering normally instead._\n\n"
                )
                async for ev in self._fallback_answer_stream(request, chunks, llm):
                    yield ev
                # Emit source events for what we have
                for pid in real_papers:
                    first_chunk = by_paper[pid][0]
                    meta = self._chunk_meta(first_chunk)
                    kb_name = self._chunk_kb_name(first_chunk)
                    yield StreamEvent.source(
                        SourceReference(
                            title=meta.get("title") or pid,
                            authors=meta.get("authors"),  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                            year=meta.get("year"),
                            doi=meta.get("doi"),
                            relevance_score=min(
                                1.0,
                                max(0.0, self._chunk_score(first_chunk)),
                            ),
                            kb_name=kb_name,
                            metadata=decode_paper_metadata_json(meta),
                        )
                    )
                yield StreamEvent.done(
                    conversation_id="",
                    tokens_used=0,
                    mode="contradiction",
                    iterations=1,
                )
                return

            # B-6: emit sources immediately after retrieval so the client can
            # display them before the slow per-paper summarisation + cluster
            # LLM calls complete (which can take 10-30 s per paper).
            sources = self._build_sources(by_paper, real_papers)
            for source in sources:
                yield StreamEvent.source(source)

            emit_phase(_phase_sink, phase="group_by_stance", state="running")
            yield StreamEvent.status(
                f"Contradiction analysis: comparing claims across {n} papers..."
            )
            cap = getattr(self.settings, "map_reduce_max_papers", 8)
            paper_summaries = await self._summarize_claims(by_paper, request.query, llm, cap)
            clusters = await self._cluster_claims(request.query, paper_summaries, llm)
            emit_phase(_phase_sink, phase="group_by_stance", state="done")
            emit_phase(_phase_sink, phase="contrast", state="running")
            if _c is not None:
                _c.add_trace("cluster", detail={
                    "agreement": len(clusters.get("agreement", [])),
                    "disagreement": len(clusters.get("disagreement", [])),
                    "open": len(clusters.get("open", [])),
                })
            if _c is not None:
                _c.add_trace("synthesize")

            emit_phase(_phase_sink, phase="contrast", state="done")
            emit_phase(_phase_sink, phase="synthesize", state="running")
            async for ev in self._synthesize_stream(
                request=request,
                query=request.query,
                clusters=clusters,
                paper_summaries=paper_summaries,
                documents=chunks,
                llm=llm,
            ):
                yield ev

            emit_phase(_phase_sink, phase="synthesize", state="done")
            yield StreamEvent.done(
                conversation_id="",
                tokens_used=0,
                mode="contradiction",
                iterations=1,
            )

        except Exception as exc:
            logger.error("contradiction_mode_error", error=str(exc))
            yield StreamEvent(event="error", data=json.dumps({"message": str(exc)}))

    # ------------------------------------------------------------------
    # execute — non-streaming entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute contradiction-detection RAG, collecting stream into a RAGResponse."""
        answer_parts: list[str] = []
        sources: list[SourceReference] = []

        async for ev in self.execute_stream(
            request=request,
            llm=llm,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            tools=tools,
        ):
            if ev.event == "content":
                try:
                    data = json.loads(ev.data)
                    delta = data.get("delta", "") or data.get("content", "") or ""
                except Exception:
                    delta = ev.data
                answer_parts.append(delta)
            elif ev.event == "source":
                try:
                    source_data = json.loads(ev.data)
                    sources.append(SourceReference(**source_data))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    pass
            elif ev.event == "error":
                try:
                    msg = json.loads(ev.data).get("message", ev.data)
                except Exception:
                    msg = ev.data
                answer_parts.append(f"\n\n[Error: {msg}]")

        return RAGResponse(
            answer="".join(answer_parts),
            sources=sources,
            mode=RAGMode.CONTRADICTION,
            iterations=1,
            web_search_used=False,
        )
