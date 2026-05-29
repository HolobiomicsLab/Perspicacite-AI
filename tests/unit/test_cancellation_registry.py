"""Unit tests for the cancellation registry."""
import pytest

from perspicacite.rag import cancellation as cr


@pytest.fixture(autouse=True)
async def _reset():
    await cr.reset_for_tests()
    yield
    await cr.reset_for_tests()


@pytest.mark.asyncio
async def test_mark_and_check():
    await cr.mark_cancelled("abc")
    assert cr.is_cancelled("abc") is True
    assert cr.is_cancelled("other") is False


@pytest.mark.asyncio
async def test_empty_and_none_ids_safe():
    assert cr.is_cancelled(None) is False
    assert cr.is_cancelled("") is False
    await cr.mark_cancelled("")  # no-op, must not raise
    await cr.mark_cancelled(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_clear_removes_entry():
    await cr.mark_cancelled("x")
    await cr.clear("x")
    assert cr.is_cancelled("x") is False


@pytest.mark.asyncio
async def test_max_entries_bound():
    """Inserting 1500 entries -> registry stays at <= 1000 (MAX cap)."""
    for i in range(1500):
        await cr.mark_cancelled(f"id-{i}")
    snap = await cr.snapshot()
    assert len(snap) <= 1000
    # Oldest IDs were evicted; newest should still be present.
    assert "id-1499" in snap
    assert "id-0" not in snap


@pytest.mark.asyncio
async def test_ttl_expiry(monkeypatch):
    fake_now = [1000.0]

    def fake_monotonic():
        return fake_now[0]

    monkeypatch.setattr("perspicacite.rag.cancellation.time.monotonic", fake_monotonic)
    await cr.mark_cancelled("ttl-test")
    assert cr.is_cancelled("ttl-test")
    # Jump 1 hour + 1 second forward and trigger another mark to invoke prune.
    fake_now[0] += 3601.0
    await cr.mark_cancelled("trigger-prune")
    snap = await cr.snapshot()
    assert "ttl-test" not in snap
    assert "trigger-prune" in snap
