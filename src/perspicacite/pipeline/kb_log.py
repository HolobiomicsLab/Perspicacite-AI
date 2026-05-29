"""Per-KB append-only JSONL event log (Wave 4.3).

One file per KB at ``data/kb_logs/<kb_name>.jsonl``. Each line is one
:class:`KBEvent` rendered as JSON. Append is atomic for sub-PIPE_BUF
lines via POSIX append-mode write — concurrent writers don't
interleave.

See docs/superpowers/specs/2026-05-14-versioned-kbs-design.md.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.kb_log")


EventKind = Literal[
    "kb_created", "paper_added", "paper_skipped",
    "paper_failed", "kb_pruned", "external_link",
]


@dataclass
class KBEvent:
    """One event in a KB's history."""

    event: EventKind
    kb_name: str
    paper_id: str = ""
    title: str | None = None
    chunks: int = 0
    source_command: str | None = None
    reason: str | None = None
    ts: int = field(default_factory=lambda: int(time.time()))
    operator_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class KBLogWriter:
    """Append-only JSONL log for one KB."""

    def __init__(self, *, path: Path | str):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Don't crash the caller — provenance is best-effort.
            logger.warning(
                "kb_log_mkdir_failed", path=str(self.path), error=str(exc),
            )

    # ---- write side --------------------------------------------------

    def append(self, event: KBEvent) -> None:
        """Atomically append one event line. Never raises — write
        failures log + drop."""
        try:
            line = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True)
            # ``a`` mode is atomic for sub-PIPE_BUF (~4 KB) writes on POSIX.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                # fsync is best-effort
                with contextlib.suppress(OSError):
                    os.fsync(fh.fileno())
        except Exception as exc:
            logger.warning(
                "kb_log_append_failed",
                path=str(self.path),
                event_kind=event.event,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    # ---- read side ---------------------------------------------------

    def read_all(self) -> list[KBEvent]:
        """Return all events in append order. Missing file → []."""
        if not self.path.exists():
            return []
        events: list[KBEvent] = []
        try:
            with open(self.path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:
            logger.warning("kb_log_read_failed", path=str(self.path), error=str(exc))
            return []

        for i, raw in enumerate(lines):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Only the LAST line can be a SIGKILL-induced partial;
                # log middle-line corruption but keep reading.
                if i == len(lines) - 1:
                    logger.debug("kb_log_partial_last_line_skipped",
                                 path=str(self.path))
                else:
                    logger.warning("kb_log_malformed_line",
                                   path=str(self.path), line_no=i + 1)
                continue
            try:
                events.append(KBEvent(**data))
            except TypeError:
                # Schema drift — log and skip.
                logger.warning("kb_log_schema_drift",
                               path=str(self.path), line_no=i + 1)
        return events

    def read_after(self, *, ts: int) -> list[KBEvent]:
        """Events with ``ts > ts`` (strict)."""
        return [e for e in self.read_all() if e.ts > ts]

    # ---- rollback ----------------------------------------------------

    def rollback_after(self, *, ts: int) -> list[str]:
        """Return the paper IDs added after ``ts`` and record a
        ``kb_pruned`` event."""
        all_events = self.read_all()
        paper_ids = [
            e.paper_id for e in all_events
            if e.event == "paper_added" and e.ts > ts and e.paper_id
        ]
        kb_name = ""
        for e in all_events:
            if e.kb_name:
                kb_name = e.kb_name
                break
        self.append(KBEvent(
            event="kb_pruned",
            kb_name=kb_name,
            extra={
                "rolled_back_paper_ids": paper_ids,
                "ts_cutoff": ts,
            },
        ))
        return paper_ids
