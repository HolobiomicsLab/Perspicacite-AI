"""In-place skill_kb.json round-trip. ASB writes a stub; Perspicacité
fills it with chunk metadata after ingest. Idempotent against re-runs."""
import json
import shutil
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures" / "asb" / "metlinkr_subset"


def test_write_entries_updates_skill_kb_json(tmp_path):
    from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries

    target = tmp_path / "run"
    shutil.copytree(FIXTURE, target)
    skill_kb = (target / "skills" / "cross-identifier-reconciliation"
                / "skill_kb.json")
    before = json.loads(skill_kb.read_text())
    # ASB writes entries=[] in the placeholder
    assert before.get("entries") in ([], None)

    entries = [
        {
            "kind": "skill_body",
            "source_url": "skills/cross-identifier-reconciliation/skill.md",
            "kb_name": "metlinkr_bundle",
            "chunk_ids": ["c1", "c2"],
            "chunk_count": 2,
            "bytes": 5400,
            "content_type": "text",
            "embedding_model": "text-embedding-3-small",
            "ingested_at": "2026-05-15T20:30:00Z",
        }
    ]
    n = write_skill_kb_entries(skill_kb, entries=entries)
    assert n == 1

    after = json.loads(skill_kb.read_text())
    assert len(after["entries"]) == 1
    assert after["total_bytes"] == 5400
    assert "perspicacite_ingest_completed=" in after.get("notes", "")


def test_write_entries_idempotent_by_source_url(tmp_path):
    """Re-running with the same source_url replaces, not duplicates."""
    from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries

    target = tmp_path / "run"
    shutil.copytree(FIXTURE, target)
    skill_kb = (target / "skills" / "cross-identifier-reconciliation"
                / "skill_kb.json")

    entries = [{
        "kind": "skill_body",
        "source_url": "skills/cross-identifier-reconciliation/skill.md",
        "kb_name": "kb",
        "chunk_ids": [],
        "chunk_count": 0,
        "bytes": 100,
        "content_type": "text",
        "embedding_model": "x",
        "ingested_at": "2026-05-15T20:00:00Z",
    }]
    write_skill_kb_entries(skill_kb, entries=entries)
    write_skill_kb_entries(skill_kb, entries=entries)  # re-run
    after = json.loads(skill_kb.read_text())
    assert len(after["entries"]) == 1


def test_write_preserves_original_notes(tmp_path):
    """If the source skill_kb.json has notes, append rather than overwrite."""
    from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries

    skill_kb = tmp_path / "skill_kb.json"
    skill_kb.write_text(json.dumps({
        "entries": [],
        "total_bytes": 0,
        "truncated": False,
        "notes": "original_asb_note=true",
    }))
    write_skill_kb_entries(skill_kb, entries=[
        {"kind": "skill_body", "source_url": "x", "kb_name": "kb",
         "chunk_ids": [], "chunk_count": 0, "bytes": 50, "content_type": "text",
         "embedding_model": "x", "ingested_at": "2026-05-15T20:00:00Z"}
    ])
    after = json.loads(skill_kb.read_text())
    assert "original_asb_note=true" in after.get("notes", "")
    assert "perspicacite_ingest_completed=" in after.get("notes", "")


def test_write_replaces_previous_perspicacite_stamp(tmp_path):
    """If the file already has a previous perspicacite_ingest_completed
    stamp, it should be replaced (not appended again)."""
    from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries

    skill_kb = tmp_path / "skill_kb.json"
    skill_kb.write_text(json.dumps({
        "entries": [],
        "notes": "prefix | perspicacite_ingest_completed=2025-01-01T00:00:00Z",
    }))
    write_skill_kb_entries(skill_kb, entries=[
        {"kind": "skill_body", "source_url": "y", "kb_name": "kb",
         "chunk_ids": [], "chunk_count": 0, "bytes": 50, "content_type": "text",
         "embedding_model": "x", "ingested_at": "2026-05-15T20:00:00Z"}
    ])
    after = json.loads(skill_kb.read_text())
    notes = after.get("notes", "")
    # Only ONE stamp present
    assert notes.count("perspicacite_ingest_completed=") == 1


def test_write_missing_file_raises(tmp_path):
    import pytest

    from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries
    with pytest.raises(FileNotFoundError):
        write_skill_kb_entries(tmp_path / "no_such.json", entries=[])


def test_total_bytes_sums_entry_bytes(tmp_path):
    from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries
    skill_kb = tmp_path / "skill_kb.json"
    skill_kb.write_text(json.dumps({"entries": []}))
    entries = [
        {"kind": "skill_body", "source_url": "a", "kb_name": "k",
         "chunk_ids": [], "chunk_count": 0, "bytes": 100, "content_type": "text",
         "embedding_model": "x", "ingested_at": "2026-05-15T20:00:00Z"},
        {"kind": "workflow_card", "source_url": "b", "kb_name": "k",
         "chunk_ids": [], "chunk_count": 0, "bytes": 200, "content_type": "text",
         "embedding_model": "x", "ingested_at": "2026-05-15T20:00:00Z"},
    ]
    write_skill_kb_entries(skill_kb, entries=entries)
    after = json.loads(skill_kb.read_text())
    assert after["total_bytes"] == 300
