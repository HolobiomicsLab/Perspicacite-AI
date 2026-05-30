"""Conversation CRUD routes."""

from __future__ import annotations

import json as _json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from perspicacite.logging import get_logger
from perspicacite.web.state import app_state

logger = get_logger("perspicacite.web.routers.conversations")

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(session_id: str | None = None) -> list[Any]:
    """List all conversations (optionally filtered by session_id)."""
    if not app_state.session_store:
        return []

    # If no session_id provided, return all conversations
    conversations: list[Any]
    if session_id:
        conversations = await app_state.session_store.list_conversations(session_id)
    else:
        # Get all conversations from all sessions
        import aiosqlite

        async with aiosqlite.connect(app_state.session_store.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall("SELECT * FROM conversations ORDER BY updated_at DESC")
            conversations = [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "kb_name": r["kb_name"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    return conversations


@router.get("/api/conversations/search")
async def search_conversations(q: str = ""):
    """Full-text search across saved conversations."""
    if not app_state.session_store:
        return {"results": []}
    if not q or not q.strip():
        return {"results": []}
    return {"results": await app_state.session_store.search_conversations(q.strip())}


@router.get("/api/conversations/{conv_id}/messages/{message_id}/provenance")
async def get_message_provenance(conv_id: str, message_id: str):
    """Return the provenance record for a specific message."""
    if app_state.provenance_store is None:
        raise HTTPException(status_code=503, detail="provenance not configured")
    rec = await app_state.provenance_store.get_for_message(message_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="no provenance for that message")
    if rec.get("conversation_id") and rec["conversation_id"] != conv_id:
        raise HTTPException(status_code=404, detail="provenance not in this conversation")
    return rec


@router.get("/api/conversations/{conv_id}/provenance")
async def list_conversation_provenance(conv_id: str):
    """Return all provenance records for a conversation."""
    if app_state.provenance_store is None:
        raise HTTPException(status_code=503, detail="provenance not configured")
    return await app_state.provenance_store.get_for_conversation(conv_id)


@router.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a specific conversation with all messages."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    conversation = await app_state.session_store.get_conversation(conv_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": conversation.id,
        "title": conversation.title,
        "kb_name": conversation.kb_name,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in conversation.messages
        ],
    }


@router.post("/api/conversations")
async def create_conversation(request: dict):
    """Create a new conversation."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    session_id = request.get("session_id", "default")
    kb_name = request.get("kb_name", "default")
    title = request.get("title", "New Conversation")

    conversation = await app_state.session_store.create_conversation(
        session_id=session_id,
        kb_name=kb_name,
        title=title,
    )

    return {
        "id": conversation.id,
        "title": conversation.title,
        "kb_name": conversation.kb_name,
        "session_id": session_id,
    }


@router.post("/api/conversations/{conv_id}/messages")
async def add_message(conv_id: str, request: dict):
    """Add a message to a conversation."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    from perspicacite.models.messages import Message

    message = Message(
        role=request.get("role", "user"),
        content=request.get("content", ""),
    )

    await app_state.session_store.add_message(conv_id, message)

    return {"status": "ok"}


@router.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    success = await app_state.session_store.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "deleted", "conversation_id": conv_id}


@router.delete("/api/conversations")
async def delete_all_conversations():
    """Delete all conversations for the current user."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    count = await app_state.session_store.delete_all_conversations()
    return {"status": "deleted", "count": count}


def _parse_sources(raw) -> list[dict]:
    """Return a list of source dicts; tolerates list-of-dicts or a JSON string."""
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except Exception as exc:
            logger.debug("failed to parse sources json", error=str(exc))
            return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    return []


def _render_conversation_markdown(conv) -> str:
    """Build a Markdown string from a Conversation object."""
    conv_id = getattr(conv, "id", "unknown")
    lines: list[str] = []
    title = getattr(conv, "title", None) or f"Conversation {conv_id}"
    lines.append(f"# {title}\n")
    created = getattr(conv, "created_at", None)
    if created:
        lines.append(f"_Exported from Perspicacité — created {created}_\n")

    all_sources: list[dict] = []
    msgs = list(getattr(conv, "messages", []) or [])

    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = getattr(m, "role", "")
        if role == "user":
            lines.append(f"## {getattr(m, 'content', '').strip()}\n")
            # find next assistant message
            j = i + 1
            ans = None
            while j < len(msgs):
                if getattr(msgs[j], "role", "") == "assistant":
                    ans = msgs[j]
                    break
                j += 1
            if ans is not None:
                lines.append(getattr(ans, "content", "").rstrip() + "\n")
                srcs = _parse_sources(getattr(ans, "sources", []))
                if srcs:
                    cited = ", ".join(
                        (
                            f"[{s.get('title') or s.get('doi') or '?'}](https://doi.org/{s['doi']})"
                            if s.get("doi")
                            else f"{s.get('title') or '?'}"
                        )
                        for s in srcs
                    )
                    lines.append(f"**Sources:** {cited}\n")
                    all_sources.extend(srcs)
                i = j + 1
                continue
        elif role == "assistant":
            # standalone assistant message — render as a section
            lines.append(getattr(m, "content", "").rstrip() + "\n")
            srcs = _parse_sources(getattr(m, "sources", []))
            all_sources.extend(srcs)
        i += 1

    if all_sources:
        lines.append("\n## References\n")
        seen: set[str] = set()
        for s in all_sources:
            key = s.get("doi") or s.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            doi = s.get("doi")
            entry = s.get("title") or doi or "?"
            if doi:
                entry += f" — https://doi.org/{doi}"
            lines.append(f"- {entry}")

    return "\n".join(lines) + "\n"


@router.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str, format: str = "markdown"):
    """Export a conversation as Markdown or RO-Crate zip."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="System not initialized")
    if format not in ("markdown", "ro-crate"):
        raise HTTPException(status_code=400, detail="Supported formats: markdown, ro-crate")
    conv = await app_state.session_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    md = _render_conversation_markdown(conv)

    if format == "markdown":
        return PlainTextResponse(
            md,
            media_type="text/markdown",
            headers={"Content-Disposition": (f'attachment; filename="conversation-{conv_id}.md"')},
        )

    # ro-crate branch
    from fastapi.responses import Response

    from perspicacite.provenance.rocrate import build_rocrate_bundle

    prov_records: list = []
    jsonl_bytes: bytes = b""
    if app_state.provenance_store is not None:
        prov_records = await app_state.provenance_store.get_for_conversation(conv_id)
        sidecar = app_state.provenance_store.sidecar_dir / f"{conv_id}.jsonl"
        if sidecar.exists():
            jsonl_bytes = sidecar.read_bytes()

    conv_dict = conv.model_dump() if hasattr(conv, "model_dump") else dict(conv)
    # Normalize datetime values to ISO strings for JSON serialization
    from datetime import datetime as _datetime
    conv_dict = {
        k: (v.isoformat() if isinstance(v, _datetime) else v)
        for k, v in conv_dict.items()
    }
    msgs_raw = list(getattr(conv, "messages", []) or [])
    msgs_dicts: list[dict] = []
    for m in msgs_raw:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        # Normalize sources (may be a JSON string)
        d["sources"] = _parse_sources(d.get("sources"))
        # Normalize datetime values to ISO strings
        d = {k: (v.isoformat() if isinstance(v, _datetime) else v) for k, v in d.items()}
        msgs_dicts.append(d)

    blob = build_rocrate_bundle(
        conversation=conv_dict,
        messages=msgs_dicts,
        conversation_markdown=md,
        provenance_records=prov_records,
        llm_calls_jsonl=jsonl_bytes,
    )
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="conversation-{conv_id}.rocrate.zip"'},
    )
