"""Chat endpoint and streaming helpers.

Owns the /api/chat POST route and the three streaming generators that
produce SSE events for the agentic and RAG-mode flows.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from perspicacite.models.rag import RAGMode
from perspicacite.pipeline.asb.response import build_asb_response_metadata
from perspicacite.provenance.collector import ProvenanceCollector
from perspicacite.web.routers._grounding import extract_grounding_context
from perspicacite.web.state import app_state

logger = logging.getLogger(__name__)

# Module-level map from mode string to RAGMode enum.
# Referenced by _stream_rag_mode and exposed for testing.
RAG_MODE_MAP = {
    "basic": RAGMode.BASIC,
    "advanced": RAGMode.ADVANCED,
    "deep_research": RAGMode.DEEP_RESEARCH,
    "profound": RAGMode.PROFOUND,   # backward-compat alias (deprecated)
    "literature_survey": RAGMode.LITERATURE_SURVEY,
    "contradiction": RAGMode.CONTRADICTION,
    "reasoning": RAGMode.REASONING,
}

router = APIRouter()


# ---------------------------------------------------------------------------
# Language detection + on-the-fly translation
# ---------------------------------------------------------------------------
#
# Literature databases (PubMed, Semantic Scholar, OpenAlex, …) are heavily
# English-biased: a French query like "Qu'est-ce que le réseautage par
# identité d'ions?" returns zero hits even when the equivalent English
# question pulls hundreds. We detect obviously non-English input with a
# cheap regex and, on a positive match, route it through the LLM for a
# one-shot translation before retrieval. Behaviour is gated by the
# request-level `auto_translate` flag (default True).

_NON_EN_LETTERS = (
    "àâçéèêëïîôûùÿœæñáíóúüöäßÀÂÇÉÈÊËÏÎÔÛÙŸŒÆÑÁÍÓÚÜÖÄ"
)
_NON_EN_HINT_RE = re.compile(
    rf"[{_NON_EN_LETTERS}]"
    r"|\b(qu'?est-?ce|qu'?un|qu'?une|quels?|quelles?|comment|pourquoi|"
    r"où|donne(?:-moi|z)?|explique(?:-moi|z)?|d[ée]finis(?:sez)?|"
    r"que\ssignifie|c'?est|je\s|nous\s|vous\s|notre\s|votre\s)\b",
    re.IGNORECASE,
)


def _looks_non_english(text: str) -> bool:
    """Cheap heuristic — flag obviously non-English text without an LLM call."""
    if not text or len(text.strip()) < 3:
        return False
    return bool(_NON_EN_HINT_RE.search(text))


async def _pre_screen_query(text: str, history: list[Any] | None) -> tuple[str, str | None]:
    """Return (verdict, reason). Verdict is "research" or "chat".

    - "research": the turn warrants the full retrieve→synthesize pipeline.
    - "chat": pure conversation / follow-up that can be answered from
      history alone (no DB hit, no rephrase).

    Best-effort: any LLM error falls back to "research" (the safer
    choice, since missing a chat-only optimisation only costs latency
    while skipping a real research turn would lose the user data).
    """
    llm = getattr(app_state, "llm_client", None)
    cfg = getattr(app_state, "config", None)
    if llm is None or cfg is None:
        return "research", None
    provider = "deepseek"
    model = "deepseek-chat"
    if getattr(cfg, "llm", None) is not None:
        provider = cfg.llm.default_provider or provider
        model = cfg.llm.default_model or model
    # Light formatting of last 3 turns so the classifier knows whether
    # the user is following up on something or starting fresh.
    history_snippet = ""
    if history:
        last = history[-3:]
        bits: list[str] = []
        for m in last:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            content = (
                getattr(m, "content", None)
                or (m.get("content") if isinstance(m, dict) else "")
            )
            if role and content:
                bits.append(f"{role}: {str(content)[:240]}")
        history_snippet = "\n".join(bits)
    try:
        out = await llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify the user's next turn in a research-assistant chat. "
                        "Output a SINGLE-LINE JSON object and nothing else: "
                        '{"verdict":"research"|"chat","reason":"<short>"}. '
                        "Use 'research' when the turn asks a new scientific question, "
                        "requests literature, compares techniques, or any task that "
                        "would benefit from new paper retrieval. "
                        "Use 'chat' when the turn is a pure follow-up to an existing answer "
                        "(\"can you summarise that\", \"explain point 2\", \"thanks\", "
                        "\"reformulate\"), small talk, or a meta-question that needs no "
                        "new sources."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent history:\n{history_snippet}\n\n"
                        f"User's next turn: {text}"
                    ),
                },
            ],
            model=model,
            provider=provider,
            temperature=0.0,
            max_tokens=80,
        )
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if not m:
            return "research", None
        obj = json.loads(m.group(0))
        verdict = obj.get("verdict")
        reason = obj.get("reason")
        if verdict in ("research", "chat"):
            return verdict, reason if isinstance(reason, str) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_screen_failed: %s", exc)
    return "research", None


async def _chat_only_reply(query: str, history: list[Any] | None) -> str:
    """Generate a no-retrieval reply for a chat-only turn. Best-effort."""
    llm = getattr(app_state, "llm_client", None)
    cfg = getattr(app_state, "config", None)
    if llm is None or cfg is None:
        return "I can answer that from the conversation above — could you rephrase?"
    provider = "deepseek"
    model = "deepseek-chat"
    if getattr(cfg, "llm", None) is not None:
        provider = cfg.llm.default_provider or provider
        model = cfg.llm.default_model or model
    msgs: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are Perspicacité, a literature-research assistant. The "
                "current turn does not require new paper retrieval — answer "
                "from the conversation context only. Be concise. If the user "
                "is asking you to cite or quote a paper, point them at the "
                "sources already shown in the prior turns and remind them "
                "that you didn't run a new search."
            ),
        }
    ]
    if history:
        for m in history[-8:]:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            content = (
                getattr(m, "content", None)
                or (m.get("content") if isinstance(m, dict) else "")
            )
            if role and content:
                msgs.append({"role": role, "content": str(content)})
    msgs.append({"role": "user", "content": query})
    try:
        out = await llm.complete(
            messages=msgs,
            model=model,
            provider=provider,
            temperature=0.4,
            max_tokens=900,
        )
        return out.strip() or "(no answer)"
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat_only_reply_failed: %s", exc)
        return "(unable to answer from context — please try again)"


async def _translate_query_to_english(text: str) -> tuple[str | None, str | None]:
    """Translate ``text`` to English. Returns (english, source_lang) or
    ``(None, None)`` on failure. Best-effort: never raises."""
    llm = getattr(app_state, "llm_client", None)
    cfg = getattr(app_state, "config", None)
    if llm is None or cfg is None:
        return None, None
    provider = "deepseek"
    model = "deepseek-chat"
    if getattr(cfg, "llm", None) is not None:
        provider = cfg.llm.default_provider or provider
        model = cfg.llm.default_model or model
    try:
        out = await llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine for scientific research queries. "
                        "Reply with a SINGLE-LINE JSON object and nothing else: "
                        '{"lang":"<ISO-639-1 code of input>","english":"<English translation>"}. '
                        "If the input is already English, set english to the input verbatim. "
                        "Never add commentary, code fences, or markdown."
                    ),
                },
                {"role": "user", "content": text},
            ],
            model=model,
            provider=provider,
            temperature=0.0,
            max_tokens=400,
        )
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if not m:
            return None, None
        obj = json.loads(m.group(0))
        english = obj.get("english")
        lang = obj.get("lang")
        if isinstance(english, str) and english.strip():
            return english.strip(), lang if isinstance(lang, str) else None
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("query_translation_failed: %s", exc)
    return None, None


# ---------------------------------------------------------------------------
# Stop-button support
# ---------------------------------------------------------------------------
# Frontend "Stop" sends a hint here before aborting its fetch. We park the
# conversation_id in a small in-memory set; the streaming generators can poll
# Cancellation now lives in the shared registry so MCP and chat both
# use the same state. The chat router only needs sync read access
# (is_chat_cancelled) — the registry's is_cancelled is sync.
from perspicacite.rag.cancellation import (
    clear as _registry_clear,
)
from perspicacite.rag.cancellation import (
    is_cancelled as _registry_is_cancelled,
)
from perspicacite.rag.cancellation import (
    mark_cancelled as _registry_mark_cancelled,
)


def is_chat_cancelled(conversation_id: str | None) -> bool:
    """Return True if the frontend asked to stop this conversation."""
    return _registry_is_cancelled(conversation_id)


def _clear_chat_cancel(conversation_id: str | None) -> None:
    if conversation_id:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_registry_clear(conversation_id))
            else:
                loop.run_until_complete(_registry_clear(conversation_id))
        except Exception:
            pass


class _CancelRequest(BaseModel):
    conversation_id: str | None = Field(default=None)


@router.post("/api/chat/cancel")
async def cancel_chat(req: _CancelRequest):
    """Mark a conversation as cancelled. Streaming generators check this
    between yields and stop emitting.
    """
    if req.conversation_id:
        await _registry_mark_cancelled(req.conversation_id)
        logger.info("chat_cancel_requested", extra={"conversation_id": req.conversation_id})
    return {"ok": True, "conversation_id": req.conversation_id}


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: str = Field(..., description="user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat endpoint - NOW WITH CONVERSATION SUPPORT.

    Accepts ``query`` (canonical) or ``message`` (Scriptorium-compat alias).
    When both are supplied, ``query`` wins.
    """

    query: str = Field(..., description="Current research question")

    @model_validator(mode="before")
    @classmethod
    def _accept_message_alias(cls, data):
        """Backward-compat with the legacy OpenAPI schema name ``message``.
        If ``query`` is absent but ``message`` is supplied, promote it.
        ``query`` always wins when both are present."""
        if isinstance(data, dict) and "query" not in data and "message" in data:
            data = {**data, "query": data["message"]}
        return data

    messages: list[ChatMessage] = Field(default_factory=list, description="Conversation history")
    session_id: str | None = Field(default=None, description="Session ID for persistence")
    conversation_id: str | None = Field(
        default=None, description="Conversation ID for persistent chat thread"
    )
    kb_name: str | None = Field(default=None, description="Knowledge base to search first")
    kb_names: list[str] | None = Field(
        default=None,
        description="Multiple knowledge bases to query together (embedding-model-compatible KBs only)",
    )
    mode: str = Field(
        default="basic",
        description=(
            "RAG mode: basic, advanced, deep_research, agentic, literature_survey, contradiction (profound is a deprecated alias for deep_research)"
        ),
    )
    stream: bool = Field(default=True, description="Stream the response")
    max_papers: int = Field(
        default=3, ge=1, le=50, description="Maximum papers to display in results"
    )
    max_papers_to_download: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum papers to download for full-text analysis (Agentic mode). Higher = more comprehensive but slower",
    )
    databases: list[str] = Field(
        default_factory=list,
        description=(
            "List of databases to search (semantic_scholar, openalex, pubmed, "
            "arxiv, ieee, springer, dblp, europepmc, core, inspire, pubchem, "
            "google_scholar, dblp_sparql). Empty list = let the backend pick "
            "from whatever providers the server's config has actually built. "
            "A non-empty list is treated as an explicit user choice — if none "
            "of the picks resolve to a built provider, the search returns no "
            "results and surfaces a 'selection_unavailable' telemetry event "
            "rather than silently falling back to unrelated databases."
        ),
    )
    context: str | None = Field(
        default=None,
        description=(
            "Optional. If set, used directly as the grounding context for "
            "query optimization. If unset and the conversation has a prior "
            "assistant turn, the server auto-extracts a short context phrase. "
            "Set to an empty string to explicitly disable grounding."
        ),
    )
    auto_translate: bool = Field(
        default=True,
        description=(
            "When true (default), detect non-English queries via a cheap "
            "heuristic and translate them to English via the configured "
            "LLM before retrieval. The original wording is reported back "
            "in a `query_translated` SSE event so the UI can show both."
        ),
    )
    pre_screen: bool = Field(
        default=True,
        description=(
            "When true (default), a cheap (~30-token) LLM classifier "
            "decides whether the user's turn actually needs literature "
            "retrieval. Pure chat / clarifications / meta-questions about "
            "an earlier answer are routed to a no-retrieval reply, saving "
            "30-60s and the cost of full search. The decision is reported "
            "via a `pre_screen` SSE event."
        ),
    )
    profond_cycles: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Profond mode only: per-request override for the number of "
            "research cycles (1-5). When unset, the server-side default "
            "(1) is used."
        ),
    )
    bm25_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Override BM25 weight for advanced-mode hybrid retrieval (0.0-1.0). "
            "bm25_weight=1.0 + vector_weight=0.0 → pure BM25. None = LLM-determined."
        ),
    )
    vector_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Override vector (dense) weight for advanced-mode hybrid retrieval. "
            "None = LLM-determined."
        ),
    )
    use_hyde: bool = Field(
        default=False,
        description=(
            "When true, generate a HyDE (Hypothetical Document Embeddings) "
            "synthetic abstract from the claim before vector search. Improves "
            "recall for hard-paraphrase queries where claim language differs "
            "from paper vocabulary. Only active in basic mode."
        ),
    )


class ChatResponse(BaseModel):
    """Response chunk for streaming."""

    type: str = Field(..., description="thinking, tool_call, tool_result, answer, error")
    content: str | None = None
    message: str | None = None
    step: str | None = None
    tool: str | None = None
    description: str | None = None
    result_summary: str | None = None
    details: str | None = None
    session_id: str | None = None


@router.post("/api/chat")
async def chat_endpoint(request: ChatRequest, raw_request: Request):
    """
    Main chat endpoint with true agentic orchestration.

    Uses LLM-driven planning, not fixed workflow.
    """
    if not app_state.initialized:
        await app_state.initialize()

    # Get or create conversation for persistence
    conversation_id = request.conversation_id
    if app_state.session_store:
        if conversation_id:
            # Verify conversation exists
            conv = await app_state.session_store.get_conversation(conversation_id)
            if not conv:
                conversation_id = None  # Will create new below

        if not conversation_id:
            # Create new conversation
            session_id = request.session_id or str(uuid.uuid4())
            kb_name = request.kb_name or "default"
            # Use first 30 chars of query as title
            title = request.query[:30] + "..." if len(request.query) > 30 else request.query

            conv = await app_state.session_store.create_conversation(
                session_id=session_id,
                kb_name=kb_name,
                title=title,
            )
            conversation_id = conv.id
            logger.info(f"Created new conversation: {conversation_id} for session {session_id}")

    # --- Grounding context resolution for basic mode ---
    resolved_context: str | None = None
    if request.mode == "basic":
        if request.context is not None:
            # Explicit client-provided context (empty string = disable)
            resolved_context = request.context or None
        else:
            # Auto-extract from last assistant turn
            prior_excerpt: str | None = None
            for msg in reversed(request.messages):
                if msg.role == "assistant":
                    prior_excerpt = msg.content
                    break
            if prior_excerpt:
                resolved_context = await extract_grounding_context(
                    prior_excerpt=prior_excerpt,
                    query=request.query,
                    app_state=app_state,
                )

    if request.stream:
        return StreamingResponse(
            agentic_chat_stream(request, conversation_id), media_type="text/event-stream"
        )
    else:
        # Non-streaming: consume the SSE stream internally, return JSON
        if request.mode == "basic":
            return await _invoke_basic_rag(request, conversation_id, context=resolved_context)

        answer = ""
        sources = []
        papers_list = []
        answer_tokens = []
        error_message = None
        event_count = 0

        async for event in agentic_chat_stream(request, conversation_id):
            event_count += 1

            # Bug 4: Check for client disconnect every 5 events
            if event_count % 5 == 0:
                if await raw_request.is_disconnected():
                    logger.warning("client_disconnected_aborting_pipeline")
                    break

            if not event.startswith("data:"):
                continue
            try:
                data = json.loads(event[5:].strip())
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "answer":
                content_b64 = data.get("content_b64")
                if content_b64:
                    answer = base64.b64decode(content_b64).decode("utf-8", errors="replace")
                elif "content" in data:
                    answer = str(data["content"])

            elif event_type == "token":
                # Bug 2: Accumulate token deltas as fallback
                delta_b64 = data.get("delta_b64")
                if delta_b64:
                    try:
                        delta = base64.b64decode(delta_b64).decode("utf-8", errors="replace")
                        answer_tokens.append(delta)
                    except Exception:
                        pass
                elif "delta" in data:
                    answer_tokens.append(str(data["delta"]))

            elif event_type == "source":
                sources.append(data.get("source", {}))

            elif event_type == "papers_found":
                # Bug 1: Use "papers" key, not "count"
                papers_list = data.get("papers", [])

            elif event_type == "error":
                # Bug 3: Capture error instead of swallowing
                error_message = data.get("message", "Unknown pipeline error")
                logger.warning("non_streaming_pipeline_error: %s", error_message)

        # Bug 2: If no "answer" event arrived, fall back to accumulated tokens
        if not answer and answer_tokens:
            answer = "".join(answer_tokens)

        # Bug 3: If pipeline errored with no answer, return HTTP 502
        if not answer and error_message:
            raise HTTPException(status_code=502, detail=error_message)

        return {
            "answer": answer,
            "sources": sources,
            "papers_found": len(papers_list) or len(sources),
            "conversation_id": conversation_id,
        }


async def agentic_chat_stream(request: ChatRequest, conversation_id: str | None = None):
    """
    Stream chat responses using selected RAG mode.

    Routes to appropriate handler based on request.mode:
    - agentic: Uses AgenticOrchestrator (intent-based, tool use)
    - basic/advanced/deep_research: Uses RAGEngine with respective mode

    Yields SSE events with thinking steps, tool calls, and final answer.
    """
    from perspicacite.models.messages import Message

    # Save user message to conversation if we have one. We persist the
    # *original* wording (in whatever language the user typed) — the
    # English translation, if any, is purely a retrieval-side detail.
    if conversation_id and app_state.session_store:
        try:
            await app_state.session_store.add_message(
                conversation_id, Message(role="user", content=request.query)
            )
        except Exception as e:
            logger.warning(f"Failed to save user message: {e}")

    # --- Optional auto-translation to English for retrieval ---
    # We translate before dispatching to any mode so KB search + web
    # search + query rephrasing all see English text. The original
    # query stays available for display via the SSE event we emit.
    if request.auto_translate and _looks_non_english(request.query):
        original_query = request.query
        english, lang = await _translate_query_to_english(original_query)
        if english and english.strip().lower() != original_query.strip().lower():
            yield (
                "data: "
                + json.dumps(
                    {
                        "kind": "query_translated",
                        "original": original_query,
                        "translated": english,
                        "source_lang": lang or "non-en",
                    },
                    separators=(",", ":"),
                )
                + "\n\n"
            )
            # Replace the query field so all downstream code (modes,
            # retrievers, LLM prompts) sees the English text.
            try:
                request.query = english
            except Exception:
                pass

    # --- Optional pre-screen: route chat-only turns away from retrieval ---
    # A cheap LLM classifier decides whether the user's turn warrants the
    # full retrieve→rephrase→synthesize pipeline. Pure follow-ups
    # ("rephrase the second paragraph", "thanks", "explain point 2") are
    # answered from history alone — no DB hit, no rephrase, no wait.
    if request.pre_screen:
        verdict, reason = await _pre_screen_query(request.query, request.messages)
        # Always surface the verdict so the UI can show what we decided
        # and let the user override (e.g. "no, please do search the
        # literature for this") on the next turn.
        yield (
            "data: "
            + json.dumps(
                {
                    "kind": "pre_screen",
                    "verdict": verdict,
                    "reason": reason or "",
                },
                separators=(",", ":"),
            )
            + "\n\n"
        )
        if verdict == "chat":
            reply = await _chat_only_reply(request.query, request.messages)
            assistant_message_id = str(uuid.uuid4())
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "answer",
                        "message_id": assistant_message_id,
                        "conversation_id": conversation_id,
                        "content_b64": base64.b64encode(reply.encode("utf-8")).decode("ascii"),
                        "sources": [],
                    },
                    separators=(",", ":"),
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps({"type": "done", "message_id": assistant_message_id})
                + "\n\n"
            )
            # Persist the assistant reply too.
            if conversation_id and app_state.session_store:
                try:
                    from perspicacite.rag.utils import collapse_runaway_repeats

                    await app_state.session_store.add_message(
                        conversation_id,
                        Message(
                            id=assistant_message_id,
                            role="assistant",
                            content=collapse_runaway_repeats(reply),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to save chat-only assistant message: {e}")
            return

    assistant_content = ""
    assistant_message_id_outer: str | None = None

    try:
        logger.info(f"Chat request: {request.query[:50]}... | Mode: {request.mode}")

        # Route based on selected mode
        if request.mode == "agentic":
            # Use agentic orchestrator for full agentic behavior
            async for event in _stream_agentic(request, conversation_id):
                # Collect assistant content for saving
                if event.startswith("data:"):
                    try:
                        data = json.loads(event[5:].strip())
                        if data.get("type") == "answer":
                            content_b64 = data.get("content_b64")
                            if content_b64:
                                assistant_content = base64.b64decode(content_b64).decode("utf-8")
                    except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError):
                        pass
                yield event
        else:
            # Use RAGEngine for other modes (basic, advanced, deep_research)
            async for event in _stream_rag_mode(request, conversation_id):
                # Collect assistant content and message_id for saving
                if event.startswith("data:"):
                    try:
                        data = json.loads(event[5:].strip())
                        if data.get("type") == "answer":
                            content_b64 = data.get("content_b64")
                            if content_b64:
                                assistant_content = base64.b64decode(content_b64).decode("utf-8")
                            new_msg_id = data.get("message_id")
                            if new_msg_id:
                                assistant_message_id_outer = new_msg_id
                    except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError):
                        pass
                yield event

    except Exception as e:
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        error_data = json.dumps({"type": "error", "message": str(e)})
        yield f"data: {error_data}\n\n"

    # Save assistant message to conversation
    if conversation_id and app_state.session_store and assistant_content:
        try:
            from perspicacite.rag.utils import collapse_runaway_repeats

            msg_id = assistant_message_id_outer or str(uuid.uuid4())
            await app_state.session_store.add_message(
                conversation_id,
                Message(
                    id=msg_id,
                    role="assistant",
                    content=collapse_runaway_repeats(assistant_content),
                ),
            )
            logger.info(f"Saved conversation messages to {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to save assistant message: {e}")


async def _copyright_filter_for_router(
    *, full_answer: str, sources: list[dict[str, Any]],
) -> str:
    """Router-level copyright filter applied to the assembled answer.

    The sources list emitted to the client contains paper metadata
    (title, doi, year, relevance_score) but typically not full_text.
    Without the original chunk text we can only run the detector on
    whatever is in ``sources[i].get("text")`` / ``chunk_text`` (which
    some modes do populate for citation snippets).

    For action=log, this is a final observability hook. For
    quote/strip/rewrite, per-mode synthesis is the better place to run
    the filter (it has the actual chunk full_text) — see
    rag/modes/basic.py::_apply_copyright_filter.
    """
    if not full_answer or not sources:
        return full_answer
    try:
        from perspicacite.rag.copyright_filter import CopyrightFilter
        cf_cfg = getattr(getattr(app_state, "config", None), "copyright_filter", None)
        if cf_cfg is None or not getattr(cf_cfg, "enabled", True):
            return full_answer
        sources_for_check = [
            {
                "text": s.get("chunk_text") or s.get("text") or "",
                "title": s.get("title"),
            }
            for s in sources
        ]
        cf = CopyrightFilter(
            enabled=cf_cfg.enabled,
            action=getattr(cf_cfg, "action", "log"),
            min_ngram=getattr(cf_cfg, "min_ngram", 8),
            llm_client=getattr(app_state, "llm_client", None),
            rewrite_model=getattr(cf_cfg, "rewrite_model", "claude-haiku-4-5"),
            rewrite_provider=getattr(cf_cfg, "rewrite_provider", "anthropic"),
        )
        return await cf.apply(full_answer, sources_for_check)
    except Exception as exc:
        logger.warning(f"router_copyright_filter_inner_failed: {exc}")
        return full_answer


async def _stream_agentic(request: ChatRequest, conversation_id: str | None = None):
    """Stream using AgenticOrchestrator."""
    assistant_message_id = str(uuid.uuid4())

    collector = ProvenanceCollector(
        conversation_id=conversation_id,
        message_id=assistant_message_id,
        rag_mode="agentic",
        request_params={
            "kb_name": request.kb_name,
            "kb_names": getattr(request, "kb_names", None),
        },
    )

    # See engine.py:query_stream — `with collecting()` inside an async
    # generator consumed by StreamingResponse can raise
    # "Token was created in a different Context" when the contextvar token
    # is reset across asyncio Context boundaries. Set without resetting.
    from perspicacite.provenance.context import set_collector
    set_collector(collector)

    # Accumulate `papers_found` events to populate `answer.sources`.
    # The orchestrator emits `papers_found` AFTER `answer` (so the
    # browser UI can append citations without overwriting the answer
    # innerHTML), but the chat-route contract puts sources on the
    # answer event. We buffer the answer event until papers_found
    # arrives or the stream ends, then merge.
    accumulated_papers: list[dict[str, Any]] = []
    pending_answer_event: dict[str, Any] | None = None
    # Track whether papers_found was ever received.  The end-of-stream flush
    # must NOT re-emit source events when papers_found already ran the loop —
    # otherwise papers appear twice when papers_found arrives before answer.
    papers_found_received: bool = False

    def _emit_answer(content: str, sess_id: str | None) -> str:
        sources = [
            {
                "title": p.get("title"),
                "authors": (
                    ", ".join(p["authors"]) if isinstance(p.get("authors"), list)
                    else p.get("authors")
                ),
                "year": p.get("year"),
                "doi": (
                    p["doi"].replace("https://doi.org/", "")
                    if isinstance(p.get("doi"), str)
                    else p.get("doi")
                ),
                "url": p.get("url"),
                "relevance_score": p.get("relevance_score"),
                "kb_name": p.get("kb_name"),
            }
            for p in accumulated_papers
        ]
        safe = {
            "type": "answer",
            "session_id": sess_id,
            "conversation_id": conversation_id,
            "message_id": assistant_message_id,
            "content_b64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "sources": sources,
        }
        return f"data: {json.dumps(safe, separators=(',', ':'))}\n\n"

    async for event in app_state.orchestrator.chat(
        query=request.query,
        session_id=request.session_id,
        kb_name=request.kb_name,
        stream=True,
        max_papers_to_download=request.max_papers_to_download,
        databases=request.databases,
    ):
        if event.get("type") == "papers_found":
            papers_found_received = True
            new_papers: list[dict] = []
            for p in event.get("papers") or []:
                key = (p.get("doi") or p.get("id") or p.get("title") or "").strip()
                if not key:
                    continue
                if any(
                    (q.get("doi") or q.get("id") or q.get("title") or "").strip() == key
                    for q in accumulated_papers
                ):
                    continue
                accumulated_papers.append(p)
                new_papers.append(p)
            # Item 10 (NEXT_STEPS_2026_05_25): emit individual `source` events so
            # SSE clients have the same single-contract interface regardless of mode.
            # We emit them BEFORE flushing the buffered answer so ordering matches
            # _stream_rag_mode (source events → answer event).
            for p in new_papers:
                src_event = {
                    "type": "source",
                    "source": {
                        "title": p.get("title"),
                        "authors": (
                            ", ".join(p["authors"]) if isinstance(p.get("authors"), list)
                            else p.get("authors")
                        ),
                        "year": p.get("year"),
                        "doi": (
                            p["doi"].replace("https://doi.org/", "")
                            if isinstance(p.get("doi"), str)
                            else p.get("doi")
                        ),
                        "url": p.get("url"),
                        "relevance_score": p.get("relevance_score"),
                        "kb_name": p.get("kb_name"),
                        # B-10: include paper_id so eval/client can match KB papers
                        # via "scifact:N" prefix even when the paper has no DOI.
                        "paper_id": p.get("paper_id"),
                    },
                }
                yield f"data: {json.dumps(src_event, separators=(',', ':'))}\n\n"
            # Flush any pending answer event now that we have papers.
            if pending_answer_event is not None:
                pending = pending_answer_event
                pending_answer_event = None
                yield _emit_answer(pending["content"], pending["session_id"])
            # Forward the papers_found event for UI rendering.
            yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            continue

        if event.get("type") == "answer":
            # Buffer until papers_found arrives (or stream ends).
            pending_answer_event = {
                "content": event.get("content") or "",
                "session_id": event.get("session_id"),
            }
            continue

        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

    # If no papers_found came after the answer, flush the buffered answer
    # event so the client still gets a response (with empty sources).
    if pending_answer_event is not None:
        # Emit any accumulated papers as individual source events first —
        # but only when papers_found was never received.  If papers_found DID
        # arrive (even before `answer`), the source events were already emitted
        # inside the loop above, and re-emitting here would produce duplicates.
        for p in (accumulated_papers if not papers_found_received else []):
            src_event = {
                "type": "source",
                "source": {
                    "title": p.get("title"),
                    "authors": (
                        ", ".join(p["authors"]) if isinstance(p.get("authors"), list)
                        else p.get("authors")
                    ),
                    "year": p.get("year"),
                    "doi": (
                        p["doi"].replace("https://doi.org/", "")
                        if isinstance(p.get("doi"), str)
                        else p.get("doi")
                    ),
                    "url": p.get("url"),
                    "relevance_score": p.get("relevance_score"),
                    "kb_name": p.get("kb_name"),
                    # B-10: include paper_id so KB papers without DOI can be matched.
                    "paper_id": p.get("paper_id"),
                },
            }
            yield f"data: {json.dumps(src_event, separators=(',', ':'))}\n\n"
        yield _emit_answer(
            pending_answer_event["content"], pending_answer_event["session_id"]
        )

    # End of stream
    yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_message_id})}\n\n"

    # Persist provenance record — best-effort, never raise
    if app_state.provenance_store is not None:
        try:
            await app_state.provenance_store.save(collector.finalize())
        except Exception as exc:
            logger.warning(f"provenance_save_failed (agentic): {exc}")


async def _stream_rag_mode(request: ChatRequest, conversation_id: str | None = None):
    """Stream using RAGEngine with selected mode (basic, advanced, deep_research, contradiction…)."""
    from perspicacite.models.rag import RAGMode
    from perspicacite.models.rag import RAGRequest as RAGReq

    # Map string mode to RAGMode enum using the module-level constant
    rag_mode = RAG_MODE_MAP.get(request.mode, RAGMode.BASIC)

    logger.info(f"Using RAGEngine with mode: {rag_mode.value}")

    # Pre-generate assistant message id so it can be threaded into the engine
    # and included in SSE frames so the UI and provenance rows share the same id.
    assistant_message_id = str(uuid.uuid4())

    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    conv_hist = (
        [{"role": m.role, "content": m.content} for m in request.messages]
        if request.messages
        else None
    )

    # Determine effective kb_name / kb_names for the RAG request
    effective_kb_name = request.kb_name or "default"
    effective_kb_names: list[str] | None = None

    # `kb_name="auto"` (or `kb_names=["auto"]`) routes the query: pick the
    # top-N most-relevant KBs by description + sampled paper titles, then
    # query them in parallel via the multi-KB path. BM25 by default
    # (no LLM cost); set rag_modes.route_method='llm' to use the LLM
    # router instead (one cheap call).
    is_auto = (
        (request.kb_name or "").strip().lower() == "auto"
        or (request.kb_names and len(request.kb_names) == 1
            and (request.kb_names[0] or "").strip().lower() == "auto")
    )
    if is_auto and app_state.session_store:
        from perspicacite.rag.kb_router import auto_route_kbs
        all_kbs = await app_state.session_store.list_kbs()
        cfg_rag = getattr(getattr(app_state, "config", None), "rag_modes", None)
        method = getattr(cfg_rag, "route_method", "bm25") if cfg_rag else "bm25"
        top_k = getattr(cfg_rag, "route_top_k", 3) if cfg_rag else 3
        threshold = getattr(cfg_rag, "route_threshold", 0.1) if cfg_rag else 0.1
        from perspicacite.llm.client import resolve_stage_model
        rp, rm = resolve_stage_model(getattr(app_state, "config", None), "routing")
        hits = await auto_route_kbs(
            query=request.query, kb_metas=all_kbs,
            vector_store=app_state.vector_store, method=method,
            top_k=top_k, score_threshold=threshold,
            llm_client=getattr(app_state, "llm_client", None),
            llm_model=rm, llm_provider=rp,
        )
        if not hits:
            yield (
                "data: " + json.dumps({
                    "type": "error",
                    "message": (
                        "auto-routing found no relevant KBs. Try a more "
                        "specific query, or specify kb_name / kb_names "
                        "directly."
                    ),
                }) + "\n\n"
            )
            return
        # Surface the routing decision to the client so the user sees
        # which KBs were picked + why (BM25 has no reason field).
        yield (
            "data: " + json.dumps({
                "type": "kb_route",
                "method": method,
                "hits": [h.to_dict() for h in hits],
            }) + "\n\n"
        )
        # Auto-routing (esp. the default BM25 text router) can surface KBs
        # across embedding models — e.g. a legacy all-MiniLM KB alongside
        # text-embedding-3-large ones. They can't be queried together, and a
        # KB in a different family than the running embedding provider can't
        # be queried here at all (dimension mismatch). Narrow to the family
        # the server can actually query instead of failing outright.
        picked = [h.kb_name for h in hits]
        if len(picked) == 1:
            effective_kb_name = picked[0]
        else:
            from perspicacite.retrieval.multi_kb import (
                canonical_embedding_fingerprint,
            )

            metas = [await app_state.session_store.get_kb_metadata(n) for n in picked]
            server_fp = canonical_embedding_fingerprint(
                getattr(getattr(app_state, "embedding_provider", None), "model_name", None)
            )
            # Prefer the running server's embedding family; fall back to the
            # most-relevant (top-ranked) KB's family if the server model is
            # unknown.
            target_fp = server_fp or next(
                (
                    canonical_embedding_fingerprint(getattr(m, "embedding_model", None))
                    for m in metas
                    if m is not None
                ),
                None,
            )
            compatible = [
                n
                for n, m in zip(picked, metas)
                if m is not None
                and canonical_embedding_fingerprint(getattr(m, "embedding_model", None))
                == target_fp
            ]
            dropped = [n for n in picked if n not in compatible]
            if dropped:
                logger.info(
                    f"auto-route narrowed to embedding family {target_fp!r}: "
                    f"querying {compatible}, dropped {dropped}"
                )
            if not compatible:
                yield (
                    "data: " + json.dumps({
                        "type": "error",
                        "message": (
                            f"Auto-routing matched {picked}, but none share the "
                            f"running server's embedding model ({target_fp}). Pick a "
                            "knowledge base with that embedding directly, or restart "
                            "the server with the matching embedding config."
                        ),
                    }) + "\n\n"
                )
                return
            if len(compatible) == 1:
                effective_kb_name = compatible[0]
            else:
                effective_kb_names = compatible

    elif request.kb_names and len(request.kb_names) > 1 and app_state.session_store:
        from perspicacite.retrieval.multi_kb import check_embedding_compat

        metas = [await app_state.session_store.get_kb_metadata(n) for n in request.kb_names]
        missing = next((request.kb_names[i] for i, m in enumerate(metas) if m is None), None)
        if missing is not None:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Knowledge base not found: {missing}'})}\n\n"
            return
        compat_msg = check_embedding_compat(metas)
        if compat_msg:
            yield f"data: {json.dumps({'type': 'error', 'message': compat_msg})}\n\n"
            return
        effective_kb_names = request.kb_names
    elif request.kb_names and len(request.kb_names) == 1:
        effective_kb_name = request.kb_names[0]

    # Resolve provider/model from server-side config so the chat router
    # respects llm.default_provider / llm.default_model from config.yml
    # rather than the hard-coded RAGRequest defaults (deepseek).
    cfg_llm = getattr(app_state, "config", None)
    default_provider = "deepseek"
    default_model = "deepseek-chat"
    if cfg_llm is not None and getattr(cfg_llm, "llm", None) is not None:
        default_provider = cfg_llm.llm.default_provider or default_provider
        default_model = cfg_llm.llm.default_model or default_model

    # Create RAG request
    rag_request = RAGReq(
        query=request.query,
        kb_name=effective_kb_name,
        kb_names=effective_kb_names,
        mode=rag_mode,
        stream=True,
        databases=request.databases,
        conversation_history=conv_hist,
        max_papers_retrieval=request.max_papers,
        provider=default_provider,
        model=default_model,
        max_iterations=(
            request.profond_cycles
            if rag_mode == RAGMode.PROFOUND and request.profond_cycles is not None
            else None
        ),
        bm25_weight=request.bm25_weight,
        vector_weight=request.vector_weight,
        use_hyde=getattr(request, "use_hyde", False),
    )
    # Thread app_state and grounding context onto the RAGRequest so that
    # BasicRAGMode.execute() can pass them to the optimizer.
    try:
        object.__setattr__(rag_request, "app_state", app_state)
        object.__setattr__(
            rag_request, "_resolved_context",
            getattr(request, "_resolved_context", None),
        )
    except Exception:
        pass
    # Execute using RAGEngine streaming
    full_answer = ""
    sources = []

    try:
        async for event in app_state.rag_engine.query_stream(
            rag_request,
            message_id=assistant_message_id,
            conversation_id=conversation_id,
        ):
            if event.event == "status":
                # Forward status updates
                status_data = json.loads(event.data)
                # Include full status data (for literature survey session info, etc.)
                yield f"data: {json.dumps({'type': 'status', **status_data})}\n\n"

            elif event.event == "source":
                # Collect sources
                source_data = json.loads(event.data)
                sources.append(source_data)
                # Also forward to UI for display
                yield f"data: {json.dumps({'type': 'source', 'source': source_data})}\n\n"

            elif event.event == "content":
                # Accumulate answer content
                delta = json.loads(event.data)["delta"]
                full_answer += delta
                # Live token deltas for the UI (base64 avoids mid-chunk JSON breakage)
                token_payload = {
                    "type": "token",
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "delta_b64": base64.b64encode(delta.encode("utf-8")).decode("ascii"),
                }
                yield f"data: {json.dumps(token_payload, separators=(',', ':'))}\n\n"

            elif event.event == "done":
                # Defense-in-depth copyright filter: run BEFORE we emit
                # the final answer event so any rewritten content goes
                # into the same answer envelope (rather than as a
                # follow-up revision event the client might miss).
                # Sources here carry titles + relevance_score but not
                # full_text; the per-mode synthesis path already ran the
                # filter against full chunk text when configured for
                # quote/strip/rewrite actions. This emits a log-only
                # warning event when verbatim copies still escape.
                try:
                    full_answer = await _copyright_filter_for_router(
                        full_answer=full_answer, sources=sources,
                    )
                except Exception as _exc:
                    logger.warning(f"router_copyright_filter_failed: {_exc}")
                # Send final answer as base64
                safe = {
                    "type": "answer",
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "message_id": assistant_message_id,
                    "content_b64": base64.b64encode(full_answer.encode("utf-8")).decode("ascii"),
                    "sources": sources,
                }
                yield f"data: {json.dumps(safe, separators=(',', ':'))}\n\n"
                # Derive ASB skill/workflow metadata blocks from collected
                # sources. Each source carries its underlying paper's
                # ``metadata`` dict; the helper coalesces by skill_id /
                # task_id and ignores non-ASB sources. Emit a separate
                # SSE event only when at least one block is non-empty so
                # non-ASB conversations don't get an extra noise frame.
                try:
                    asb_md = build_asb_response_metadata(
                        [
                            {"metadata": (s.get("metadata") if isinstance(s, dict) else None)}
                            for s in sources
                        ]
                    )
                    if asb_md.get("skill_metadata") or asb_md.get("workflow_metadata"):
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "asb_metadata",
                                    "message_id": assistant_message_id,
                                    **asb_md,
                                },
                                separators=(",", ":"),
                            )
                            + "\n\n"
                        )
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(f"asb_metadata_emit_failed: {_exc}")
                yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_message_id})}\n\n"
                return

            elif event.event == "code_excerpt":
                # Sub-project C: forward code-excerpt attachment to UI.
                try:
                    payload = json.loads(event.data)
                    yield f"data: {json.dumps({'type': 'code_excerpt', **payload})}\n\n"
                except Exception as _exc:
                    logger.warning(f"code_excerpt_forward_failed: {_exc}")

            elif event.event == "figure_ref":
                # Sub-project C: forward figure-ref attachment to UI.
                try:
                    payload = json.loads(event.data)
                    yield f"data: {json.dumps({'type': 'figure_ref', **payload})}\n\n"
                except Exception as _exc:
                    logger.warning(f"figure_ref_forward_failed: {_exc}")

            elif event.event == "metadata":
                try:
                    payload = json.loads(event.data)
                    yield f"data: {json.dumps({'type': 'metadata', **payload})}\n\n"
                except Exception as _exc:
                    logger.warning(f"metadata_forward_failed: {_exc}")

            elif event.event == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': event.data})}\n\n"
                return

    except Exception as e:
        logger.error(f"RAG engine error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': f'Error in {rag_mode.value} mode: {e!s}'})}\n\n"

    # End of stream (fallback if no done event)
    yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_message_id})}\n\n"


async def _invoke_basic_rag(
    request: "ChatRequest",
    conversation_id: str | None = None,
    *,
    context: str | None = None,
) -> dict:
    """Dispatch basic RAG mode with optional grounding context.

    Wraps ``_stream_rag_mode`` for basic mode.
    Separated into a named helper so tests can patch it and verify the
    grounding context was threaded through correctly.
    """
    answer = ""
    sources: list = []
    papers_list: list = []
    answer_tokens: list[str] = []

    # Propagate the resolved grounding context so _stream_rag_mode can
    # thread it onto the RAGRequest for BasicRAGMode.execute to consume.
    try:
        object.__setattr__(request, "_resolved_context", context)
    except Exception:
        pass

    async for event in _stream_rag_mode(request, conversation_id):
        if not event.startswith("data:"):
            continue
        try:
            data = json.loads(event[5:].strip())
        except Exception:
            continue
        etype = data.get("type", "")
        if etype == "answer":
            cb64 = data.get("content_b64")
            if cb64:
                answer = base64.b64decode(cb64).decode("utf-8", errors="replace")
            elif "content" in data:
                answer = str(data["content"])
        elif etype == "token":
            db64 = data.get("delta_b64")
            if db64:
                answer_tokens.append(base64.b64decode(db64).decode("utf-8", errors="replace"))
            elif "delta" in data:
                answer_tokens.append(str(data["delta"]))
        elif etype == "source":
            sources.append(data.get("source", {}))
        elif etype == "papers_found":
            papers_list = data.get("papers", [])

    if not answer and answer_tokens:
        answer = "".join(answer_tokens)

    return {
        "answer": answer,
        "sources": sources,
        "papers_found": len(papers_list) or len(sources),
        "conversation_id": conversation_id,
    }
