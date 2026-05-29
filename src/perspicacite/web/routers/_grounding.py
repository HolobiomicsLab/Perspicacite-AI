"""GUI-only multi-turn grounding extractor.

The MCP-callable LLM has its own conversation transcript and applies the
`context` docstring guidance itself. The web chat router does not — it
orchestrates the search without an LLM in front. This module fills that gap
with one cheap Haiku call that decides "continuation or pivot?" and, on
continuation, extracts a short grounding phrase.

Failure modes (no prior turn, self-contained query, grounding disabled, LLM
error, timeout, unparseable output) all return ``None`` — the caller passes
``None`` as ``context`` to the search step.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger(__name__)


_PROMPT = """The user just sent a new query in a chat. Decide if it continues the prior
topic. If yes, extract one short grounding phrase (<=80 chars). If no
(topic pivot, or unrelated), return an empty string.

PRIOR TURN (truncated excerpt):
{prior}

NEW QUERY:
{query}

Output JSON only:
{{"context": "phrase or empty string"}}
"""


# Heuristic: a query longer than 80 chars containing both a capitalized
# first word AND a verb-like token (a word ending in common verb suffixes
# or a known auxiliary) is treated as self-contained — the user clearly
# restated the scope.
_VERB_TOKEN_RE = re.compile(
    r"\b(?:is|are|was|were|does|do|did|has|have|had|find|show|"
    r"compare|explain|describe|review|"
    r"\w+(?:ing|ed|ate|ize|ise)\b)",
    re.IGNORECASE,
)
_CAPITALIZED_FIRST_RE = re.compile(r"^[A-Z]")


def _looks_self_contained(query: str) -> bool:
    q = query.strip()
    if len(q) <= 80:
        return False
    if not _CAPITALIZED_FIRST_RE.match(q):
        return False
    return bool(_VERB_TOKEN_RE.search(q))


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        cleaned = parts[1] if len(parts) >= 2 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.strip().rstrip("`")
    return cleaned


async def extract_grounding_context(
    *,
    prior_excerpt: str | None,
    query: str,
    app_state: Any,
) -> str | None:
    """Decide whether the new ``query`` continues the prior topic and, if so,
    return a short grounding phrase to pass to the search step.

    Returns ``None`` on pivot, missing prior turn, self-contained query,
    grounding disabled, LLM error, timeout, or unparseable output.
    """
    qo_cfg = app_state.config.search.query_optimization

    if not qo_cfg.grounding_enabled:
        return None
    if not prior_excerpt or not prior_excerpt.strip():
        return None
    if _looks_self_contained(query):
        return None

    prior = prior_excerpt[: qo_cfg.grounding_max_prior_chars]
    q = query[: qo_cfg.grounding_max_query_chars]

    from perspicacite.llm.client import resolve_stage_model
    provider, model = resolve_stage_model(app_state.config, "grounding")

    prompt = _PROMPT.format(prior=prior, query=q)
    try:
        text = await asyncio.wait_for(
            app_state.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                provider=provider,
                temperature=0.1,
                max_tokens=60,
                stage="grounding",
            ),
            timeout=qo_cfg.grounding_timeout_s,
        )
    except TimeoutError:
        logger.warning("grounding_extractor_timeout")
        return None
    except Exception as exc:
        logger.warning("grounding_extractor_llm_error", error=str(exc))
        return None

    cleaned = _strip_code_fence(text)
    try:
        obj = json.loads(cleaned)
    except Exception as exc:
        logger.warning(
            "grounding_extractor_unparseable",
            error=str(exc), sample=cleaned[:200],
        )
        return None

    ctx = obj.get("context")
    if not isinstance(ctx, str):
        return None
    ctx = ctx.strip()
    if not ctx:
        return None
    return ctx
