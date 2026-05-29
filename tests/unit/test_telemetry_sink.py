"""Unit tests for TelemetrySink implementations."""
import asyncio

import pytest

from perspicacite.rag.telemetry import (
    CallbackTelemetrySink,
    ListTelemetrySink,
    NullTelemetrySink,
)


def test_list_sink_append_and_iterate():
    s = ListTelemetrySink()
    s.append({"a": 1})
    s.append({"b": 2})
    assert list(s) == [{"a": 1}, {"b": 2}]
    assert len(s) == 2
    assert bool(s) is True


def test_list_sink_empty_is_falsey():
    s = ListTelemetrySink()
    assert bool(s) is False
    assert len(s) == 0


@pytest.mark.asyncio
async def test_callback_sink_invokes_callback():
    received: list[dict] = []

    async def cb(e):
        received.append(e)

    s = CallbackTelemetrySink(cb)
    await s.on_event_async({"x": 1})
    await s.on_event_async({"y": 2})
    assert received == [{"x": 1}, {"y": 2}]


@pytest.mark.asyncio
async def test_callback_sink_buffers_events():
    """The .events list lets diagnostics read what was emitted."""
    async def cb(_e):
        return

    s = CallbackTelemetrySink(cb)
    await s.on_event_async({"a": 1})
    assert s.events == [{"a": 1}]


@pytest.mark.asyncio
async def test_callback_sink_swallows_callback_errors():
    """Callback exceptions must not break the RAG pipeline."""
    async def bad(e):
        raise RuntimeError("boom")

    s = CallbackTelemetrySink(bad)
    await s.on_event_async({"x": 1})  # must not raise
    assert s.events == [{"x": 1}]


def test_null_sink_drops_everything():
    s = NullTelemetrySink()
    s.append({"x": 1})
    assert len(s) == 0
    assert bool(s) is False


@pytest.mark.asyncio
async def test_callback_sink_append_in_async_context():
    """append() schedules callback via create_task when a loop is running."""
    received: list[dict] = []

    async def cb(e):
        received.append(e)

    s = CallbackTelemetrySink(cb)
    s.append({"z": 99})
    # The task is scheduled but not yet run; yield control to let it execute.
    await asyncio.sleep(0)
    assert received == [{"z": 99}]
    assert s.events == [{"z": 99}]


def test_callback_sink_append_no_loop_silently_drops():
    """append() in a sync context with no running loop must not raise."""
    received: list[dict] = []

    async def cb(e):
        received.append(e)

    s = CallbackTelemetrySink(cb)
    # No event loop is running here — RuntimeError is caught silently.
    s.append({"w": 42})
    # The event was buffered locally even though the callback was not invoked.
    assert s.events == [{"w": 42}]
    assert received == []
