"""Small in-process job registry: SQLite row + in-memory event queues."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.jobs")


class JobRegistry:
    """create / publish / finish / fail / subscribe / get.

    The SQLite row is the source of truth (survives restart). In-memory
    `_queues[job_id]` drives SSE streams. After server restart in-memory
    queues are gone; clients fall back to polling get().
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}

    async def create(self, kind: str, total: int) -> str:
        job_id = str(uuid.uuid4())
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO jobs (id, kind, status, total) VALUES (?, ?, 'running', ?)",
                (job_id, kind, total),
            )
            await db.commit()
        self._queues[job_id] = asyncio.Queue()
        return job_id

    async def publish(self, job_id: str, event: dict[str, Any]) -> None:
        q = self._queues.get(job_id)
        if q is not None:
            await q.put(event)
        if event.get("type") == "progress" and "done" in event:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        "UPDATE jobs SET done_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (int(event["done"]), job_id),
                    )
                    await db.commit()
            except Exception as exc:
                logger.warning("jobs_progress_persist_failed", error=str(exc))

    async def finish(self, job_id: str, result: dict[str, Any]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='done', result=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(result), job_id),
            )
            await db.commit()
        q = self._queues.get(job_id)
        if q is not None:
            await q.put({"type": "done", "result": result})
            await q.put(None)
        # Remove dict entry so _queues doesn't grow unbounded; active subscribers
        # already hold a local reference to q and continue draining it fine.
        self._queues.pop(job_id, None)

    async def fail(self, job_id: str, err: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='error', error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (err, job_id),
            )
            await db.commit()
        q = self._queues.get(job_id)
        if q is not None:
            await q.put({"type": "error", "error": err})
            await q.put(None)
        # Remove dict entry so _queues doesn't grow unbounded.
        self._queues.pop(job_id, None)

    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        # For already-terminal jobs (finished/failed before subscriber arrived),
        # yield a single synthesised final frame and return immediately so callers
        # never hang on an empty queue created by setdefault below.
        row = await self.get(job_id)
        if row is not None and row.get("status") in ("done", "error"):
            if row["status"] == "done":
                yield {"type": "done", "result": row.get("result") or {}}
            else:
                yield {"type": "error", "error": row.get("error") or ""}
            return
        q = self._queues.setdefault(job_id, asyncio.Queue())
        while True:
            ev = await q.get()
            if ev is None:
                return
            yield ev

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        # aiosqlite.Row is NOT a dict: iterating it yields column *values*, so we
        # must go through .keys() to get column names. noqa: SIM118 (the rule
        # assumes a real mapping where `in row` == `in row.keys()`).
        d = {k: row[k] for k in row.keys()}  # noqa: SIM118
        if d.get("result"):
            with contextlib.suppress(json.JSONDecodeError):
                d["result"] = json.loads(d["result"])
        return d
