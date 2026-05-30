"""Phase 2 — graph traversal strategy.

Capability: agentic graph traversal — the planner fires structured graph
queries instead of vector retrieval, with vector fallback when graph coverage
is insufficient (true hybrid retrieval).

Loop:
1. Intent classifier: graph-shaped vs. narrative. Non-graph → degrade to
   Phase 4 (provenance).
2. Planner: pick one of five typed queries. Execute. Collect rows.
3. If <2 claims matched, planner can issue another query (up to MAX_ITERS).
4. Compose answer; emit per-query telemetry.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from perspicacite.indicium_layer import queries
from perspicacite.indicium_layer.store import ClaimGraphStore
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGRequest, SourceReference, StreamEvent

logger = get_logger("perspicacite.rag.modes.reasoning.graph_traversal")

MAX_ITERS = 5
MIN_CLAIMS_FOR_ANSWER = 2

_INTENT_PROMPT = """Classify the question's intent for an indicium reasoning system.

Return strict JSON: {{"intent": "graph"|"narrative", "rationale": "<one sentence>"}}.

Use "graph" for: "which claims support X?", "what disputes Y?", "trace evidence
from X to Z", "papers claiming pattern S R O", "neighbours of claim C".

Use "narrative" for: open-ended summaries, "what is the consensus on Y", or
anything asking for prose synthesis.

Question: {query}
"""

_PLAN_PROMPT = """You drive an indicium claim-graph traversal.

Available queries (pick exactly one and provide kwargs):
- claims_supporting(subject_or_iri: str, min_eco_grade: \
"data|citation|knowledge|inference|speculation"|null)
- claims_disputing(target_iri: str)
- evidence_trace(claim_iri: str, max_depth: int=3)
- papers_with_claim_pattern(subject: str|null, relation: str|null, object: str|null)
- neighbors(claim_iri: str, edge_types: [str, ...] |null)

Return strict JSON: {{"query": "<name>", "kwargs": {{...}}, \
"rationale": "<one sentence>"}}.

Question: {query}

Iteration: {iteration}
Already retrieved claim IRIs: {seen}
Prior rationale: {prior}
"""

_COMPOSE_PROMPT = """Answer the question USING ONLY the supplied graph rows.
Cite paper DOIs inline as [doi:...]. Plain prose, no markdown headers.

Question: {query}

Rows:
{rows}
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
    except Exception as exc:
        logger.debug("json parse failed", error=str(exc))
        return default


_QueryFn = Callable[..., list[dict]]

_QUERY_TABLE: dict[str, _QueryFn] = {
    "claims_supporting": queries.claims_supporting,
    "claims_disputing": queries.claims_disputing,
    "evidence_trace": queries.evidence_trace,
    "papers_with_claim_pattern": queries.papers_with_claim_pattern,
    "neighbors": queries.neighbors,
}


def _exec_query(store: Any, kb_name: str, name: str, kwargs: dict) -> list[dict]:
    fn = _QUERY_TABLE.get(name)
    if fn is None:
        raise ValueError(f"unknown query: {name}")
    return fn(store, kb_name, **(kwargs or {}))


async def run_graph_traversal_stream(
    *,
    request: RAGRequest,
    llm: Any,
    vector_store: Any,
    embedding_provider: Any,
    config: Any,
    session_store: Any = None,
) -> AsyncIterator[StreamEvent]:
    yield StreamEvent.status("Reasoning (graph): classifying intent…")
    intent_raw = await llm.complete(
        messages=[
            {
                "role": "user",
                "content": _INTENT_PROMPT.format(query=request.query),
            }
        ],
        stage="reasoning.graph.intent",
    )
    intent = _parse_json(intent_raw, {"intent": "narrative"})
    if intent.get("intent") != "graph":
        yield StreamEvent.status(
            f"Reasoning (graph): non-graph intent "
            f"({intent.get('rationale', '')}); degrading to provenance…"
        )
        from perspicacite.rag.modes.reasoning.provenance import run_provenance_stream

        async for ev in run_provenance_stream(
            request=request,
            llm=llm,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            config=config,
            session_store=session_store,
        ):
            yield ev
        return

    kb_name = (request.kb_names[0] if request.kb_names else request.kb_name) or "default"
    store = _open_store(kb_name)
    seen: set[str] = set()
    all_rows: list[dict] = []
    prior_rationale = "(none)"
    last_query_name = ""
    iteration = 1

    try:
        for iteration in range(1, MAX_ITERS + 1):
            plan_raw = await llm.complete(
                messages=[
                    {
                        "role": "user",
                        "content": _PLAN_PROMPT.format(
                            query=request.query,
                            iteration=iteration,
                            seen=sorted(seen)[:30],
                            prior=prior_rationale,
                        ),
                    }
                ],
                stage="reasoning.graph.plan",
            )
            plan = _parse_json(plan_raw, {})
            name = plan.get("query")
            kwargs = plan.get("kwargs") or {}
            if name not in _QUERY_TABLE:
                yield StreamEvent.status(
                    f"Reasoning (graph): planner returned invalid query '{name}'; stopping."
                )
                break

            yield StreamEvent.status_kind(
                f"Graph query #{iteration}: {name}",
                kind="graph_query",
                iteration=iteration,
                query=name,
                kwargs=kwargs,
                rationale=plan.get("rationale", ""),
            )
            try:
                rows = _exec_query(store, kb_name, name, kwargs)
            except Exception as exc:
                logger.warning("graph_query_error", query=name, error=str(exc))
                rows = []

            for r in rows:
                claim_key = r.get("claim") or r.get("from") or r.get("neighbor")
                if claim_key:
                    seen.add(claim_key)
                all_rows.append({"_query": name, **r})

            last_query_name = name
            prior_rationale = plan.get("rationale", "(none)")
            if len(seen) >= MIN_CLAIMS_FOR_ANSWER:
                break

        if not all_rows:
            yield StreamEvent.status(
                "Reasoning (graph): no rows returned by any query; "
                "degrading to provenance over vector retrieval…"
            )
            from perspicacite.rag.modes.reasoning.provenance import run_provenance_stream

            async for ev in run_provenance_stream(
                request=request,
                llm=llm,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                config=config,
                session_store=session_store,
            ):
                yield ev
            return

        rendered = "\n".join(
            f"- [{r['_query']}] " + " ".join(f"{k}={v}" for k, v in r.items() if k != "_query")
            for r in all_rows[:40]
        )
        try:
            answer = await llm.complete(
                messages=[
                    {
                        "role": "user",
                        "content": _COMPOSE_PROMPT.format(query=request.query, rows=rendered),
                    }
                ],
                stage="reasoning.graph.compose",
            )
        except Exception as exc:
            logger.warning("graph_compose_failed", error=str(exc))
            answer = (
                f"(Compose step failed: {exc}. Retrieved "
                f"{len(seen)} claim(s) via {last_query_name}.)"
            )
        if not isinstance(answer, str):
            answer = str(answer)
        yield StreamEvent.content(answer)

        # Source events from paper IRIs that appear in rows
        seen_papers: set[str] = set()
        for r in all_rows:
            paper = r.get("paper")
            if not paper or paper in seen_papers:
                continue
            seen_papers.add(paper)
            yield StreamEvent.source(
                SourceReference(
                    title=paper,
                    doi=paper.removeprefix("doi:") if paper.startswith("doi:") else None,
                    relevance_score=0.6,
                    kb_name=kb_name,
                )
            )

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="reasoning",
            iterations=iteration,
        )
    finally:
        store.close()
