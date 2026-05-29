"""Enforce: no production code reads/writes legacy metadata['sources']."""
import re
from pathlib import Path

import perspicacite

SRC_DIR = Path(perspicacite.__file__).parent

_LEGACY_PAT = re.compile(
    r"""metadata\s*\[\s*["'](?:sources|enrichment_sources)["']\s*\]"""
)


def test_no_legacy_metadata_sources_writes():
    """grep src/ for metadata["sources"] hits — should find nothing."""
    hits: list[str] = []
    for f in SRC_DIR.rglob("*.py"):
        # Self-reference in this test file is OK.
        if "test_no_legacy_metadata_sources" in str(f):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for ln, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # comments are documentation, not live usage
            if _LEGACY_PAT.search(line):
                hits.append(f"{f}:{ln}: {line.strip()}")
    assert hits == [], (
        "Legacy metadata['sources'] usage found — migrate to "
        "Paper.discovery_sources / Paper.enrichment_sources:\n"
        + "\n".join(hits)
    )
