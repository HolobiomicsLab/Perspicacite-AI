"""RO-Crate 1.1-flavored bundle builder (not SHACL-validated)."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any


def build_rocrate_bundle(
    *,
    conversation: dict[str, Any],
    messages: list[dict[str, Any]],
    conversation_markdown: str,
    provenance_records: Iterable[dict[str, Any]],
    llm_calls_jsonl: bytes,
) -> bytes:
    """Build an in-memory zip containing an RO-Crate-flavored conversation bundle.

    Layout:
        ro-crate-metadata.json
        conversation.md
        provenance/answer-<message_id>.json  (one per assistant message that has a record)
        provenance/llm-calls.jsonl           (copy of the sidecar)
        sources.json                         (flat, deduplicated list of cited papers)
    """
    prov_records_list = list(provenance_records)
    prov_by_msg = {p["message_id"]: p for p in prov_records_list if p.get("message_id")}

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in messages:
        for s in (m.get("sources") or []):
            doi = s.get("doi")
            key = doi or s.get("title") or json.dumps(s, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "doi": doi,
                    "title": s.get("title"),
                    "year": s.get("year"),
                    "journal": s.get("journal"),
                    "kb_name": s.get("kb_name"),
                    "content_type": s.get("content_type"),
                }
            )

    metadata = _build_ro_crate_metadata(conversation, messages, prov_by_msg, sources)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ro-crate-metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        z.writestr("conversation.md", conversation_markdown)
        z.writestr("sources.json", json.dumps(sources, indent=2, ensure_ascii=False))
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            mid = msg.get("id")
            rec = prov_by_msg.get(mid)
            if rec is None:
                continue
            z.writestr(f"provenance/answer-{mid}.json", json.dumps(rec, indent=2, ensure_ascii=False))
        z.writestr("provenance/llm-calls.jsonl", llm_calls_jsonl or b"")
    return buf.getvalue()


def _build_ro_crate_metadata(
    conversation: dict[str, Any],
    messages: list[dict[str, Any]],
    prov_by_msg: dict[str, dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    graph: list[dict[str, Any]] = [
        {
            "@type": "CreativeWork",
            "@id": "ro-crate-metadata.json",
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
            "about": {"@id": "./"},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": conversation.get("title") or "Conversation",
            "datePublished": conversation.get("created_at") or now,
            "description": "Perspicacité conversation with provenance trace",
            "identifier": conversation.get("id"),
            "hasPart": [
                {"@id": "conversation.md"},
                {"@id": "provenance/llm-calls.jsonl"},
                {"@id": "sources.json"},
            ],
        },
        {"@id": "conversation.md", "@type": "File", "encodingFormat": "text/markdown",
         "name": "Conversation transcript"},
        {"@id": "provenance/llm-calls.jsonl", "@type": "File", "encodingFormat": "application/jsonl",
         "name": "LLM call audit"},
        {"@id": "sources.json", "@type": "File", "encodingFormat": "application/json",
         "name": "Cited papers manifest"},
    ]

    for s in sources:
        sid = (
            f"https://doi.org/{s['doi']}"
            if s.get("doi")
            else f"#paper-{abs(hash(s.get('title') or '')) % 10_000_000}"
        )
        graph.append({
            "@id": sid,
            "@type": "ScholarlyArticle",
            "name": s.get("title"),
            "identifier": s.get("doi"),
            "datePublished": str(s.get("year")) if s.get("year") else None,
            "journal": s.get("journal"),
        })

    pending_q: dict[str, Any] | None = None
    for m in messages:
        role = m.get("role")
        if role == "user":
            pending_q = m
            continue
        if role == "assistant":
            mid = m.get("id")
            rec = prov_by_msg.get(str(mid) if mid is not None else "", {})
            instruments = sorted({c.get("model") for c in (rec.get("llm_calls_index") or []) if c.get("model")})
            mentions = [
                {
                    "@id": (
                        f"https://doi.org/{s['doi']}"
                        if s.get("doi")
                        else f"#paper-{abs(hash(s.get('title') or '')) % 10_000_000}"
                    )
                }
                for s in (m.get("sources") or [])
            ]
            entry: dict[str, Any] = {
                "@id": f"#answer-{mid}",
                "@type": "CreateAction",
                "name": f"Answer {mid}",
                "object": pending_q.get("content") if pending_q else None,
                "result": m.get("content"),
                "instrument": [
                    {"@id": f"#model-{x}", "@type": "SoftwareApplication", "name": x}
                    for x in instruments
                ],
                "mentions": mentions,
                "additionalProperty": [
                    {"@type": "PropertyValue", "name": "rag_mode", "value": rec.get("rag_mode")},
                    {
                        "@type": "PropertyValue",
                        "name": "kb_name",
                        "value": (rec.get("request_params") or {}).get("kb_name"),
                    },
                ],
                "subjectOf": {"@id": f"provenance/answer-{mid}.json"},
            }
            graph.append(entry)
            pending_q = None
    return {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": graph,
    }
