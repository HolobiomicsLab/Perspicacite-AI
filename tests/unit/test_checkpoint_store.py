"""Tests for CheckpointStore + atomic save (Wave 3.3)."""
from pathlib import Path

from perspicacite.pipeline.checkpoint import CheckpointStore

# The ingest loop composes this reason for a throttled DOI; record()
# prepends the "failed" prefix that remaining_ids() keys the retry on.
from perspicacite.pipeline.search_to_kb import RATE_LIMITED_REASON


def _store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(
        path=tmp_path / "ck.json",
        kb_name="kb1",
        operation="ingest_dois",
    )


def test_load_returns_none_when_file_missing(tmp_path):
    s = _store(tmp_path)
    assert s.load() is None


def test_save_then_load_roundtrip(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a", "b", "c"])
    state.record("a", "added")
    s.save(state)

    s2 = _store(tmp_path)
    loaded = s2.load()
    assert loaded is not None
    assert loaded.processed == {"a": "added"}
    assert loaded.planned_ids == ["a", "b", "c"]


def test_record_adds_to_processed(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a", "b"])
    state.record("a", "added")
    state.record("b", "failed", reason="timeout")
    assert state.processed["a"] == "added"
    assert "failed" in state.processed["b"]
    assert "timeout" in state.processed["b"]


def test_remaining_ids_excludes_processed(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a", "b", "c", "d"])
    state.record("a", "added")
    state.record("c", "failed", reason="x")
    assert list(state.remaining_ids()) == ["b", "d"]


def test_retry_failed_re_includes_failed_ids(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a", "b", "c"])
    state.record("a", "added")
    state.record("b", "failed", reason="x")
    assert list(state.remaining_ids(retry_failed=True)) == ["b", "c"]


def test_is_complete(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a", "b"])
    state.record("a", "added")
    assert state.is_complete() is False
    state.record("b", "added")
    assert state.is_complete() is True


def test_atomic_save_no_tmp_left_behind(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a"])
    s.save(state)
    # tmp suffix file should not exist after save.
    assert not (tmp_path / "ck.json.tmp").exists()
    assert (tmp_path / "ck.json").exists()


def test_delete_removes_file(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a"])
    s.save(state)
    assert (tmp_path / "ck.json").exists()
    s.delete()
    assert not (tmp_path / "ck.json").exists()
    # Delete on absent file is a no-op.
    s.delete()


def test_record_failed_reason_truncated_to_200_chars(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["a"])
    long_reason = "x" * 500
    state.record("a", "failed", reason=long_reason)
    assert len(state.processed["a"]) <= 220   # "failed: " + 200 chars + slack
    assert "failed:" in state.processed["a"]


def test_rate_limited_reason_keeps_the_failed_prefix(tmp_path):
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["throttled", "clean"])
    state.record("clean", "added")
    state.record(
        "throttled", "failed",
        reason=RATE_LIMITED_REASON.format(hosts="api.publisher.example"),
    )

    assert state.processed["throttled"].startswith("failed")
    assert "api.publisher.example" in state.processed["throttled"]
    assert list(state.remaining_ids(retry_failed=True)) == ["throttled"]


def test_added_outcome_is_never_reoffered(tmp_path):
    """Negative case: an abstract-only paper recorded as added must stay done."""
    s = _store(tmp_path)
    state = s.load_or_create(planned_ids=["abstract_only"])
    state.record("abstract_only", "added")

    assert list(state.remaining_ids(retry_failed=True)) == []
