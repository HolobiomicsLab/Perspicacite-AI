"""Zotero integration router — push papers to Zotero via the web API.

Endpoints:
- GET  /api/zotero/status — returns {"enabled": bool}
- POST /api/zotero/push   — push a list of DOIs to the configured Zotero library
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.web.routers.zotero")

router = APIRouter(prefix="/api/zotero", tags=["zotero"])


class PushRequest(BaseModel):
    dois: list[str]


@router.get("/status")
async def zotero_status() -> dict[str, Any]:
    """Return whether Zotero is configured.

    Reports ``enabled`` when ``zotero.enabled`` is true — that alone is
    sufficient when talking to the desktop API on loopback. ``mode`` is
    "local" / "cloud" so the UI can hint at the credentials needed for
    write operations.
    """
    from perspicacite.web.state import app_state

    cfg = getattr(getattr(app_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled):
        return {"enabled": False, "mode": None}
    base_url = (getattr(cfg, "base_url", "") or "").strip()
    is_local = bool(base_url) and ("localhost" in base_url or "127.0.0.1" in base_url)
    return {
        "enabled": True,
        "mode": "local" if is_local else "cloud",
        "has_api_key": bool(cfg.api_key),
        "has_library_id": bool(cfg.library_id),
    }


@router.post("/push")
async def zotero_push(payload: PushRequest) -> dict[str, Any]:
    """Push a list of DOIs to the configured Zotero library.

    Fetches metadata (abstract-only, no PDF download) via the unified
    pipeline and calls ZoteroClient.create_item for each DOI.

    Returns:
        {"created": [...], "skipped": [], "failed": [...]}

    Raises:
        503 when Zotero is not configured.
        400 when more than 100 DOIs are submitted.
    """
    from perspicacite.web.state import app_state

    cfg = getattr(getattr(app_state, "config", None), "zotero", None)
    if not cfg or not cfg.enabled or not cfg.api_key or not cfg.library_id:
        raise HTTPException(status_code=503, detail="zotero not configured")
    if len(payload.dois) > 100:
        raise HTTPException(status_code=400, detail="At most 100 DOIs per request")

    from perspicacite.integrations.zotero import ZoteroClient
    from perspicacite.pipeline.download import retrieve_paper_content

    # Re-use app_state's http_client if available; otherwise open a temporary one.
    http_client = getattr(app_state, "http_client", None)
    own_client = None
    if http_client is None:
        import httpx

        own_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        http_client = own_client

    try:
        zotero = ZoteroClient(
            api_key=cfg.api_key,
            library_id=cfg.library_id,
            library_type=cfg.library_type,
            collection_key=cfg.collection_key,
            base_url=getattr(cfg, "base_url", "") or None,
            http_client=http_client,
        )
        created: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for doi in payload.dois:
            doi = (doi or "").strip().replace("https://doi.org/", "")
            if not doi:
                continue
            try:
                content = await retrieve_paper_content(
                    doi,
                    http_client=http_client,
                    pdf_parser=None,  # metadata-only — no slow PDF download
                )
                paper: dict[str, Any] = dict(content.metadata or {})
                paper["doi"] = doi
                paper["abstract"] = content.abstract or paper.get("abstract")
                key = await zotero.create_item(paper)
                if key:
                    created.append({"doi": doi, "key": key})
                else:
                    failed.append({"doi": doi, "reason": "no key returned"})
            except Exception as exc:
                logger.warning("zotero push failed", error=str(exc))
                failed.append({"doi": doi, "reason": str(exc)})

        return {"created": created, "skipped": [], "failed": failed}
    finally:
        if own_client is not None:
            await own_client.aclose()
