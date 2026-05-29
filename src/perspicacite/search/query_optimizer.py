"""Shared LLM-assisted query rewrite for scientific literature search.

One small Haiku call rewrites the user's query into a clean scientific
phrasing before the aggregator fan-out. Used by both the MCP `search_literature`
tool and the GUI's RAG-mode aggregator call sites.

Failure modes (LLM error, unparseable JSON, timeout) silently fall back to
the verbatim query and set ``fallback_reason``. Search must never break
because rewriting failed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


_PROMPT = """You rewrite a scientific question into two specialised forms so it
performs well on different retrieval backends.

The QUERY is what the user is asking now. It is authoritative.
The CONTEXT is optional background from the conversation. Use it ONLY to
disambiguate ambiguous terms in the query (e.g., expanding an acronym, fixing
a vague pronoun). If the context conflicts with or extends beyond the query's
topic, ignore the context entirely.

Your rewrites are ADDITIVE and NORMALISING, never SUBTRACTIVE of meaning.
The only things you may remove are filler/chatty words ("can you find", "I'm
looking for", "papers about"). You must NEVER delete a content word the user
typed — names, identifiers, qualifiers, methods, organisms, units, author or
person surnames (e.g. "Libis", "Doudna"). If you are unsure whether a token
is filler or content, KEEP it.

Produce TWO rewrites:

1) WEB_QUERY — for keyword-driven academic databases (Semantic Scholar,
   OpenAlex, PubMed, arXiv, Europe PMC). 3-7 words, noun-phrase style.
   Strip chatty words. Keep proper nouns, gene/drug/species names, and
   author surnames verbatim. Prefer established terminology when the query
   is vague (e.g., "heart attack" → "myocardial infarction"), but do NOT
   replace specific terms the user already chose. NO boolean operators,
   field qualifiers, or quotes — plain text only. Resist adding extra
   subject-area context unless the query is genuinely ambiguous without it.

2) KB_QUERY — for dense / semantic retrieval over a user knowledge base.
   A full grammatical sentence or rich noun phrase (8-20 words). Keep
   the question shape if the original was a question. Domain context terms
   ARE welcome here (they boost embedding similarity). Do not invent claims
   the user did not state.

Rules common to both:
- Preserve the user's scientific intent exactly. Do not narrow or drift the topic.
- Preserve specific terms the user already chose.
- If the original is already optimal for that backend, return it unchanged
  (the two outputs may be identical).

Return JSON only:
{{"searched_query": "<WEB_QUERY>", "web_query": "<WEB_QUERY>", "kb_query": "<KB_QUERY>"}}

The legacy ``searched_query`` field MUST equal ``web_query`` so older callers
keep getting the web-friendly form.

QUERY:
{query}

CONTEXT (may be empty, may be irrelevant — ignore if it conflicts with QUERY):
{context}
"""


@dataclass
class OptimizationResult:
    """Outcome of one optimize_query call.

    - ``enabled`` is the resolved effective flag (per-call arg merged with
      config default). ``False`` means the optimizer short-circuited without
      calling Haiku.
    - ``applied`` is True only when the LLM produced a query that differs
      from the input. ``False`` covers three cases: optimizer disabled, model
      no-op, or fallback.
    - ``fallback_reason`` is None on success, on no-op, AND when disabled.
      Otherwise one of ``llm_error`` / ``unparseable`` / ``timeout``.
    """

    searched_query: str
    enabled: bool
    applied: bool
    context_used: bool
    fallback_reason: str | None
    # Backend-specific variants (since 2026-05-19). ``web_query`` is
    # tuned for keyword-driven academic databases; ``kb_query`` is a
    # richer sentence form for dense / semantic vector retrieval. Both
    # default to ``searched_query`` for back-compat when the LLM omits
    # either field or when the optimizer is disabled / falls back.
    web_query: str = ""
    kb_query: str = ""

    def for_target(self, target: str) -> str:
        """Pick the right rewrite for ``target`` ("kb" or "web")."""
        if target == "kb":
            return self.kb_query or self.searched_query
        return self.web_query or self.searched_query


def _strip_code_fence(text: str) -> str:
    """Remove leading/trailing markdown code fences. Mirrors the helper used
    by `rephrase_query` in `pipeline/search_to_kb.py`."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        cleaned = parts[1] if len(parts) >= 2 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.strip().rstrip("`")
    return cleaned


async def optimize_query(
    *,
    query: str,
    context: str | None,
    app_state: Any,
    optimize_enabled: bool | None = None,
    sink: Any = None,
) -> OptimizationResult:
    """Rewrite ``query`` into a clean scientific phrasing via one Haiku call.

    Args:
        query: The user's raw query string.
        context: Optional grounding excerpt. Truncated head-keep to the
            ``search.query_optimization.max_context_chars`` cap before being
            inserted into the prompt. ``None`` or empty string means "no
            context".
        app_state: The application state object. Must expose ``config`` and
            ``llm_client`` attributes; ``config.search.query_optimization``
            holds the runtime settings.
        optimize_enabled: Per-call override. ``True`` forces on, ``False``
            forces off, ``None`` falls back to
            ``config.search.query_optimization.enabled``.

    Returns:
        ``OptimizationResult`` — always returns a usable ``searched_query``;
        callers don't need to handle failure separately.
    """
    qo_cfg = app_state.config.search.query_optimization

    enabled = qo_cfg.enabled if optimize_enabled is None else optimize_enabled
    if not enabled:
        return OptimizationResult(
            searched_query=query, enabled=False, applied=False,
            context_used=False, fallback_reason=None,
        )

    # Year-anchor preservation: author+year queries are identifier-style lookups
    # (e.g. "retrieval augmented generation Lewis 2020"). Rewriting them strips
    # the year/author, making the specific paper unfindable. Return verbatim.
    if re.search(r'\b(19|20)\d{2}\b', query.strip()):
        return OptimizationResult(
            searched_query=query.strip(), enabled=True, applied=False,
            context_used=False, fallback_reason="year_anchor_preserved",
            web_query=query.strip(), kb_query=query.strip(),
        )

    # Truncate context head-keep.
    ctx_str = context or ""
    if len(ctx_str) > qo_cfg.max_context_chars:
        ctx_str = ctx_str[: qo_cfg.max_context_chars]
    context_used = bool(ctx_str.strip())

    # Resolve model/provider via the existing stage-routing helper.
    from perspicacite.llm.client import resolve_stage_model
    provider, model = resolve_stage_model(app_state.config, "search_optimize")

    prompt = _PROMPT.format(query=query, context=ctx_str)
    try:
        text = await asyncio.wait_for(
            app_state.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                provider=provider,
                temperature=0.2,
                max_tokens=800,
                stage="search_optimize",
                sink=sink,
            ),
            timeout=qo_cfg.timeout_s,
        )
    except TimeoutError:
        logger.warning("query_optimizer_timeout", extra={"query": query[:80]})
        return OptimizationResult(
            searched_query=query, enabled=True, applied=False,
            context_used=context_used, fallback_reason="timeout",
        )
    except Exception as exc:
        logger.warning("query_optimizer_llm_error", extra={"error": str(exc), "query": query[:80]})
        return OptimizationResult(
            searched_query=query, enabled=True, applied=False,
            context_used=context_used, fallback_reason="llm_error",
        )

    cleaned = _strip_code_fence(text)
    try:
        obj = json.loads(cleaned)
    except Exception as exc:
        logger.warning(
            "query_optimizer_unparseable",
            extra={"error": str(exc), "sample": cleaned[:200]},
        )
        return OptimizationResult(
            searched_query=query, enabled=True, applied=False,
            context_used=context_used, fallback_reason="unparseable",
        )

    rewritten = obj.get("searched_query")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return OptimizationResult(
            searched_query=query, enabled=True, applied=False,
            context_used=context_used, fallback_reason="unparseable",
        )

    rewritten = rewritten.strip()
    # Backend-specific forms. Fall back to ``rewritten`` if the LLM
    # omitted either field (older prompts, mid-deploy mismatch).
    web_q = obj.get("web_query")
    kb_q = obj.get("kb_query")
    web_q = web_q.strip() if isinstance(web_q, str) and web_q.strip() else rewritten
    kb_q = kb_q.strip() if isinstance(kb_q, str) and kb_q.strip() else rewritten
    applied = rewritten != query.strip()
    return OptimizationResult(
        searched_query=rewritten, enabled=True, applied=applied,
        context_used=context_used, fallback_reason=None,
        web_query=web_q, kb_query=kb_q,
    )
