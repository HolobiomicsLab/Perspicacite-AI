"""In-place writer for skills/{slug}/skill_kb.json.

Preserves ASB's original notes; appends Perspicacité's completion
stamp. Idempotent against re-ingest: entries keyed by source_url.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def write_skill_kb_entries(
    skill_kb_path: Path | str,
    *,
    entries: list[dict],
) -> int:
    """Update the skill_kb.json file at ``skill_kb_path``.

    Entries with the same ``source_url`` as an existing entry are
    replaced (not duplicated). Returns the total number of entries
    after the update.

    Raises FileNotFoundError if the file doesn't exist. ASB always
    writes a placeholder so this is the right contract: if there's
    no placeholder, something upstream is broken.
    """
    path = Path(skill_kb_path)
    if not path.exists():
        raise FileNotFoundError(f"skill_kb.json not found at {path}")
    data = json.loads(path.read_text())
    existing: list[dict] = data.get("entries") or []
    # Merge by source_url. None / missing source_url are deduped together
    # (only one None-keyed entry can exist).
    by_url: dict = {e.get("source_url"): e for e in existing}
    for e in entries:
        by_url[e.get("source_url")] = e
    merged = list(by_url.values())

    data["entries"] = merged
    data["total_bytes"] = sum(int(e.get("bytes") or 0) for e in merged)
    data["truncated"] = any(bool(e.get("truncated")) for e in merged)

    stamp = f"perspicacite_ingest_completed={_now_iso()}"
    original_notes = data.get("notes") or ""
    if "perspicacite_ingest_completed=" in original_notes:
        # Replace inline so re-runs don't accumulate stamps.
        prefix, _sep, rest = original_notes.partition("perspicacite_ingest_completed=")
        # rest may contain " | suffix"; preserve trailing content past the next
        # ' | ' if any (defensive — unlikely in practice).
        tail = ""
        if " | " in rest:
            tail = " | " + rest.split(" | ", 1)[1]
        new_notes = (prefix + stamp + tail).strip()
        # Collapse a leading " | " separator if prefix is empty. Use
        # removeprefix (exact substring), not lstrip, which strips any
        # leading run of ' ' and '|' chars — the classic multi-char footgun.
        new_notes = new_notes.removeprefix(" | ")
        data["notes"] = new_notes
    else:
        sep = " | " if original_notes else ""
        data["notes"] = f"{original_notes}{sep}{stamp}"

    path.write_text(json.dumps(data, indent=2) + "\n")
    return len(merged)


def _now_iso() -> str:
    """UTC RFC3339 timestamp (Z-suffixed)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
