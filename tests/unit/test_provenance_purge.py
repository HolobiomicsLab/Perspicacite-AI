"""Deleting a conversation must take its provenance with it.

The provenance sidecar holds verbatim prompts and responses, so a delete that
leaves it behind is not a delete.
"""
from __future__ import annotations

import pytest

from perspicacite.provenance.store import ProvenanceStore


def _record(conv_id: str, message_id: str) -> dict:
    return {
        "message_id": message_id,
        "conversation_id": conv_id,
        "rag_mode": "basic",
        "llm_calls": [
            {
                "stage_label": "synthesis",
                "provider": "test",
                "model": "test-model",
                "prompt_messages": [{"role": "user", "content": "SECRET UNPUBLISHED QUESTION"}],
                "response_text": "SECRET ANSWER",
            }
        ],
    }


@pytest.fixture
async def store(tmp_path):
    s = ProvenanceStore(db_path=tmp_path / "p.db", sidecar_dir=tmp_path / "sidecars")
    await s.init_db()
    return s


@pytest.mark.asyncio
async def test_purge_conversation_removes_rows_and_sidecar(store, tmp_path):
    await store.save(_record("conv-1", "msg-1"))
    sidecar = tmp_path / "sidecars" / "conv-1.jsonl"
    assert sidecar.exists()
    assert "SECRET UNPUBLISHED QUESTION" in sidecar.read_text()

    deleted = await store.purge_conversation("conv-1")

    assert deleted == 1
    assert await store.get_for_conversation("conv-1") == []
    assert not sidecar.exists(), "verbatim prompt sidecar survived the delete"


@pytest.mark.asyncio
async def test_purge_conversation_leaves_other_conversations_alone(store, tmp_path):
    await store.save(_record("conv-1", "msg-1"))
    await store.save(_record("conv-2", "msg-2"))

    await store.purge_conversation("conv-1")

    assert await store.get_for_conversation("conv-2") != []
    assert (tmp_path / "sidecars" / "conv-2.jsonl").exists()


@pytest.mark.asyncio
async def test_purge_all_removes_every_row_and_sidecar(store, tmp_path):
    await store.save(_record("conv-1", "msg-1"))
    await store.save(_record("conv-2", "msg-2"))

    deleted = await store.purge_all()

    assert deleted == 2
    assert await store.get_for_conversation("conv-1") == []
    assert await store.get_for_conversation("conv-2") == []
    assert list((tmp_path / "sidecars").glob("*.jsonl")) == []


@pytest.mark.asyncio
async def test_purge_is_quiet_when_nothing_matches(store):
    assert await store.purge_conversation("never-existed") == 0


@pytest.mark.asyncio
async def test_traversing_conversation_id_cannot_delete_outside_the_sidecar_dir(
    store, tmp_path
):
    """The id arrives from a URL path segment and this path gets unlinked."""
    outsider = tmp_path / "important.jsonl"
    outsider.write_text("do not delete me")

    await store.purge_conversation("../important")

    assert outsider.exists(), "path traversal escaped the sidecar directory"
