"""Crash-resilient checkpoint store for multi-item ingests (Wave 3.3).

A single JSON file per (KB, operation) tracks which planned IDs have
been processed and how. Saved atomically (tmp + rename) so a SIGKILL
mid-write never leaves a half-written file.

See docs/superpowers/specs/2026-05-14-checkpoint-resume-design.md.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

_REASON_MAX = 200


@dataclass
class CheckpointState:
    kb_name: str
    operation: str
    planned_ids: list[str]
    processed: dict[str, str] = field(default_factory=dict)
    started_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def record(self, item_id: str, outcome: str, reason: str | None = None) -> None:
        if outcome == "failed":
            r = (reason or "").strip()[:_REASON_MAX]
            self.processed[item_id] = f"failed: {r}" if r else "failed"
        else:
            self.processed[item_id] = outcome
        self.updated_at = int(time.time())

    def remaining_ids(self, *, retry_failed: bool = False) -> Iterator[str]:
        for pid in self.planned_ids:
            outcome = self.processed.get(pid)
            if outcome is None or (retry_failed and outcome.startswith("failed")):
                yield pid

    def is_complete(self) -> bool:
        return all(pid in self.processed for pid in self.planned_ids)

    def to_dict(self) -> dict:
        return {
            "kb_name": self.kb_name,
            "operation": self.operation,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "total_planned": len(self.planned_ids),
            "planned_ids": list(self.planned_ids),
            "processed": dict(self.processed),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointState:
        return cls(
            kb_name=d.get("kb_name", ""),
            operation=d.get("operation", ""),
            planned_ids=list(d.get("planned_ids", [])),
            processed=dict(d.get("processed", {})),
            started_at=int(d.get("started_at", time.time())),
            updated_at=int(d.get("updated_at", time.time())),
        )


class CheckpointStore:
    """File-backed accessor for a CheckpointState."""

    def __init__(self, *, path: Path | str, kb_name: str, operation: str):
        self.path = Path(path)
        self.kb_name = kb_name
        self.operation = operation
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> CheckpointState | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Corrupt / unreadable file → treat as missing. Caller can
            # decide whether to overwrite with load_or_create.
            return None
        return CheckpointState.from_dict(data)

    def load_or_create(self, planned_ids: list[str]) -> CheckpointState:
        existing = self.load()
        if existing is not None:
            return existing
        return CheckpointState(
            kb_name=self.kb_name,
            operation=self.operation,
            planned_ids=list(planned_ids),
        )

    def save(self, state: CheckpointState) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def delete(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
