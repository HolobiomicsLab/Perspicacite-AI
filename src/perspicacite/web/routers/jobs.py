"""Jobs router — GET /api/jobs/{id} and GET /api/jobs/{id}/events (SSE)."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from perspicacite.web.state import app_state

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(job_id: str):
    reg = app_state.job_registry
    if reg is None:
        raise HTTPException(status_code=503, detail="jobs not configured")
    row = await reg.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return row


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str):
    reg = app_state.job_registry
    if reg is None:
        raise HTTPException(status_code=503, detail="jobs not configured")
    row = await reg.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen():
        if row["status"] in ("done", "error"):
            payload = {
                "type": row["status"],
                "result": row.get("result"),
                "error": row.get("error"),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            return
        async for ev in reg.subscribe(job_id):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
