"""TelemetrySink: unified protocol for in-RAG-pipeline progress events.

Three implementations:

- ``ListTelemetrySink``     : drop-in replacement for the legacy
  ``telemetry: list[dict]`` pattern; the SSE chat router and existing
  call sites continue to use this.

- ``CallbackTelemetrySink`` : invokes an awaitable on each event.
  The MCP layer wraps ``ctx.report_progress`` in this so external
  agents see live progress notifications during long-running tools.

- ``NullTelemetrySink``     : drops every event. Useful in tests /
  batch scripts that don't care about progress notifications.

Both `append` (sync, list-style) and `on_event_async` (async, callback-style)
APIs are exposed on every sink so mode code can use whichever feels
natural without conditionally checking the sink type.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class TelemetrySink(Protocol):
    """Any object usable as the ``telemetry`` parameter on RAG helpers."""

    def append(self, event: dict[str, Any]) -> None: ...
    async def on_event_async(self, event: dict[str, Any]) -> None: ...


class ListTelemetrySink:
    """Stores events in a plain list; drain after the await.

    Preserves the legacy semantics. Existing code that does
    ``telemetry.append({...})`` keeps working.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def on_event_async(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)


class CallbackTelemetrySink:
    """Invokes ``callback(event)`` (awaitable) on every event.

    Used by the MCP progress adapter. Provides ``append`` as a sync
    fire-and-forget shim that schedules the callback on the running
    loop; prefer ``on_event_async`` from async contexts.
    """

    def __init__(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._callback = callback
        # Mirror events into a buffer for diagnostics.
        self.events: list[dict[str, Any]] = []
        # Strong refs to fire-and-forget callback tasks. The event loop only
        # holds weak references, so without this the task can be GC'd before
        # the callback runs, silently dropping telemetry (ruff RUF006).
        self._tasks: set[asyncio.Task[None]] = set()

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._callback(event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except RuntimeError:
            # No running loop — caller is sync; drop event silently
            # (the legacy SSE drain path uses .events directly).
            pass

    async def on_event_async(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        # never let telemetry errors break the RAG pipeline
        with contextlib.suppress(Exception):
            await self._callback(event)


def emit_phase(
    sink: Any,
    phase: str,
    state: str,
    **extra: Any,
) -> None:
    """Append a ``phase_progress`` event to a telemetry sink.

    ``sink`` may be ``None``, a plain ``list``, or any object exposing
    an ``append`` method (ListTelemetrySink, CallbackTelemetrySink).
    Designed for use from inside RAG modes:

        emit_phase(_telemetry, phase="retrieve", state="running")
        ... do work ...
        emit_phase(_telemetry, phase="retrieve", state="done")

    The MCP progress adapter recognises ``kind="phase_progress"`` and
    forwards a "Phase <name>: <state>" message to clients.
    """
    if sink is None:
        return
    event = {"kind": "phase_progress", "phase": phase, "state": state}
    event.update(extra)
    # Sink is some other mapping/callable — drop silently rather than
    # crashing the pipeline.
    with contextlib.suppress(AttributeError):
        sink.append(event)


def emit_tokens(
    sink: Any,
    *,
    input_tokens: int,
    output_tokens: int,
    cumulative_in: int | None = None,
    cumulative_out: int | None = None,
    **extra: Any,
) -> None:
    """Append a ``tokens`` event to a telemetry sink."""
    if sink is None:
        return
    event: dict[str, Any] = {
        "kind": "tokens",
        "in": int(input_tokens or 0),
        "out": int(output_tokens or 0),
    }
    if cumulative_in is not None:
        event["cumulative_in"] = int(cumulative_in)
    if cumulative_out is not None:
        event["cumulative_out"] = int(cumulative_out)
    event.update(extra)
    with contextlib.suppress(AttributeError):
        sink.append(event)


def emit_cost(
    sink: Any,
    *,
    usd: float,
    model: str,
    **extra: Any,
) -> None:
    """Append a ``cost_estimate`` event to a telemetry sink."""
    if sink is None:
        return
    event: dict[str, Any] = {
        "kind": "cost_estimate",
        "usd": float(usd or 0.0),
        "model": model,
    }
    event.update(extra)
    with contextlib.suppress(AttributeError):
        sink.append(event)


class NullTelemetrySink:
    """Drops every event. Useful in tests / batch scripts."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:
        return None

    async def on_event_async(self, event: dict[str, Any]) -> None:
        return None

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False


class ResponseMetadataCollector:
    """Accumulates response-level metadata from telemetry events.

    Conforms to the .append(event) sink protocol so it can be passed
    alongside other sinks (e.g. fan-out a request.telemetry_sink to
    [MCPProgressAdapter sink, ResponseMetadataCollector]).

    Call ``.as_response_extras()`` after the RAG run to get a dict
    that can be merged into the final JSON response.
    """

    def __init__(self) -> None:
        self._attempts: list[dict] = []
        self._query_rephrasings: list[dict] = []
        self._usage_tokens_in: int = 0
        self._usage_tokens_out: int = 0
        self._usage_cost_usd: float = 0.0
        self._usage_model: str | None = None

    def append(self, event: Any) -> None:
        """Sink-protocol entry. Tolerates non-dict and unknown-kind input."""
        if not isinstance(event, dict):
            return
        kind = event.get("kind")
        if kind == "provider_progress" and event.get("phase") == "done":
            self._attempts.append(
                {
                    "query": event.get("query"),
                    "provider_counts": dict(event.get("by_provider") or {}),
                    "hit_count": int(event.get("total", 0)),
                }
            )
        elif kind == "query_rephrased":
            self._query_rephrasings.append(
                {
                    "original": event.get("original"),
                    "refined": event.get("rewritten") or event.get("refined"),
                    "reason": event.get("reason"),
                }
            )
        elif kind == "tokens":
            try:
                self._usage_tokens_in += int(event.get("in") or 0)
                self._usage_tokens_out += int(event.get("out") or 0)
            except (TypeError, ValueError):
                pass
        elif kind == "cost_estimate":
            try:
                self._usage_cost_usd += float(event.get("usd") or 0.0)
                self._usage_model = event.get("model") or self._usage_model
            except (TypeError, ValueError):
                pass

    async def on_event_async(self, event: Any) -> None:
        """Async sink-protocol entry (delegates to .append)."""
        self.append(event)

    def as_response_extras(self) -> dict:
        out: dict = {}
        if self._attempts:
            out["attempts"] = list(self._attempts)
        if self._query_rephrasings:
            out["query_rephrasings"] = list(self._query_rephrasings)
        if (
            self._usage_tokens_in
            or self._usage_tokens_out
            or self._usage_cost_usd
        ):
            out["usage"] = {
                "tokens_in": self._usage_tokens_in,
                "tokens_out": self._usage_tokens_out,
                "model": self._usage_model,
                "cost_usd_estimate": round(self._usage_cost_usd, 6),
            }
        return out
