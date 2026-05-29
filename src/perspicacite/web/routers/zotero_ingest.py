"""Zotero → KB ingest endpoints: plan + async build."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from perspicacite.integrations.zotero import ZoteroClient
from perspicacite.integrations.zotero_ingest import (
    ZoteroKBPlanEntry,
    build_kbs_from_zotero,
    plan_kbs_from_zotero,
)
from perspicacite.web.state import app_state

router = APIRouter(prefix="/api/zotero", tags=["zotero-ingest"])

# Hold strong refs to in-flight tasks (asyncio can GC them otherwise).
_ingest_tasks: set[asyncio.Task[Any]] = set()


class BuildKBsRequest(BaseModel):
    plan: list[ZoteroKBPlanEntry]
    # Optional per-call overrides so a caller can drive multiple libraries
    # (e.g. BioMedOmicsAI + MetaboLinkAI) without restarting the server.
    library_id: str | None = None
    library_type: str | None = None


def _client_from_state(
    *, library_id: str | None = None, library_type: str | None = None
) -> ZoteroClient:
    cfg = getattr(getattr(app_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled):
        raise HTTPException(status_code=503, detail="Zotero not configured")
    eff_library_type = library_type or cfg.library_type
    base_url = getattr(cfg, "base_url", "") or None
    _base_url_str = base_url or ""
    is_local = bool(base_url) and ("localhost" in _base_url_str or "127.0.0.1" in _base_url_str)
    if not cfg.api_key and not is_local:
        raise HTTPException(
            status_code=503,
            detail="Zotero api_key required for cloud access. Set zotero.api_key (or switch base_url to a local Zotero).",
        )
    # Local loopback: library_id is optional — the desktop API exposes a
    # default library when the user is signed in. We pass "0" as a
    # placeholder, which the local API treats as "the current library".
    eff_library_id = library_id or cfg.library_id
    if not eff_library_id:
        if is_local:
            eff_library_id = "0"
        else:
            raise HTTPException(
                status_code=400,
                detail="library_id required for cloud Zotero. Set zotero.library_id in config.",
            )
    return ZoteroClient(
        api_key=cfg.api_key,
        library_id=eff_library_id,
        library_type=eff_library_type,
        collection_key=cfg.collection_key,
        base_url=base_url,
    )


@router.get("/plan")
async def get_plan(
    include_unfiled: bool = True,
    library_id: str | None = None,
    library_type: str | None = None,
) -> dict[str, Any]:
    """Plan the KBs that would be built from a Zotero library.

    Pass ``library_id`` / ``library_type`` to preview a different library
    than the one in config without restarting the server.
    """
    client = _client_from_state(library_id=library_id, library_type=library_type)
    # Resolve the real library name so KB names get a unique per-group
    # prefix instead of "Library_*" — prevents collisions when running
    # builds across multiple groups.
    library_name = await client.get_library_name() or "Library"
    plan = await plan_kbs_from_zotero(
        client, include_unfiled=include_unfiled, library_label=library_name,
    )
    return {"library_name": library_name, "plan": [p.model_dump() for p in plan]}


@router.post("/build-kbs/async")
async def build_kbs_async(payload: BuildKBsRequest) -> dict[str, Any]:
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    client = _client_from_state(
        library_id=payload.library_id, library_type=payload.library_type
    )
    job_id = await app_state.job_registry.create("zotero_ingest", total=len(payload.plan))
    task = asyncio.create_task(
        build_kbs_from_zotero(
            client,
            plan=payload.plan,
            app_state=app_state,
            registry=app_state.job_registry,
            job_id=job_id,
        )
    )
    _ingest_tasks.add(task)
    task.add_done_callback(_ingest_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}
