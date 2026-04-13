"""Conversation history and retrieval-query helpers for non-agentic RAG modes."""

from __future__ import annotations

from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGRequest
from perspicacite.rag.prompts import GENERATE_CONTEXT_AWARE_QUERY_PROMPT

logger = get_logger("perspicacite.rag.conversation_helpers")

HISTORY_MAX_CHARS = 2000
HISTORY_MAX_MESSAGES = 8


def format_conversation_block(history: list[dict[str, str]] | None) -> str:
    """Format recent user/assistant turns for injection into the LLM prompt."""
    if not history:
        return ""
    lines: list[str] = []
    total = 0
    for msg in history[-HISTORY_MAX_MESSAGES:]:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        piece = f"{label}: {content}"
        if total + len(piece) > HISTORY_MAX_CHARS:
            remain = HISTORY_MAX_CHARS - total - 20
            if remain > 40:
                piece = f"{label}: {content[:remain]}…"
            else:
                break
        lines.append(piece)
        total += len(piece) + 1
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n\n".join(lines)


async def compute_retrieval_query(request: RAGRequest, llm: Any) -> tuple[str, str | None]:
    """Return (query_for_retrieval, refined_query_if_rewritten_else_none)."""
    history = getattr(request, "conversation_history", None) or []
    if not history:
        return request.query, None

    hist_block = format_conversation_block(history)
    if not hist_block:
        return request.query, None

    user_block = f"""{hist_block}

Current question: {request.query}

Adapt the current question into a single standalone search query (do not answer it)."""
    try:
        rewritten = (
            await llm.complete(
                messages=[
                    {"role": "system", "content": GENERATE_CONTEXT_AWARE_QUERY_PROMPT},
                    {"role": "user", "content": user_block},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=200,
                temperature=0.2,
            )
        ).strip()
        if rewritten and len(rewritten) > 3 and rewritten != request.query:
            logger.info(
                "retrieval_query_rewritten",
                original_len=len(request.query),
                rewritten_preview=rewritten[:120],
            )
            return rewritten, rewritten
    except Exception as e:
        logger.warning("retrieval_query_rewrite_failed", error=str(e))
    return request.query, None


def build_user_message_with_history(
    *,
    history_block: str,
    body: str,
) -> str:
    """Prepend conversation block before documents + question body."""
    if not history_block:
        return body
    return f"{history_block}\n\n---\n\n{body}"
