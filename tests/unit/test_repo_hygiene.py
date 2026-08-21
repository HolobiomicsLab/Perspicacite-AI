"""Repository hygiene guards for a public repo.

Two things have drifted into the tracked tree before and are cheap to keep
out: a real person's address in a shipped config preset, and any reference to
a shadow-library PDF source. Both are checked against the files git actually
tracks, so an untracked local config never trips them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Addresses that are meant to be in the tree: the maintainer's own contact on
# the security policy and the outgoing User-Agent, plus obvious placeholders.
_ALLOWED_EMAILS = {
    "your@email.com",
    "you@your-domain.org",
    "your.email@domain.com",
    "researcher@university.edu",
    "me@x.org",
    "user@example.com",
    "you@example.com",
    "louisfelix.nothias@gmail.com",  # maintainer, published deliberately
}

# Config presets are copied verbatim by users, so they must not carry anyone's
# real address — an unpaywall_email is sent to a third-party API on every call.
_EMAIL_VALUE = re.compile(r'unpaywall_email:\s*["\']([^"\']+)["\']')

_SHADOW_LIBRARY = re.compile(r"sci-?hub", re.IGNORECASE)

_TEXT_SUFFIXES = {".py", ".yml", ".yaml", ".md", ".toml", ".txt", ".json", ".sh", ".html"}


def _tracked_files() -> list[Path]:
    """Files git tracks, as absolute paths. Skips the test itself."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable — hygiene scan needs the tracked file list")
    here = Path(__file__).resolve()
    paths = [_REPO_ROOT / name for name in out.split("\0") if name]
    return [p for p in paths if p.resolve() != here and p.suffix in _TEXT_SUFFIXES]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def test_no_third_party_email_in_shipped_configs():
    """A shipped preset must not carry a real person's address.

    ``unpaywall_email`` is transmitted to Unpaywall/PubMed on every request, so
    a preset that keeps one contributor's address makes every user impersonate
    them.
    """
    offenders = []
    for path in _tracked_files():
        for value in _EMAIL_VALUE.findall(_read(path)):
            if value not in _ALLOWED_EMAILS:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {value}")
    assert not offenders, (
        "shipped config presets carry a non-placeholder address:\n  "
        + "\n  ".join(offenders)
    )


def test_no_shadow_library_reference_in_tracked_tree():
    """No tracked file may reference a shadow library as a PDF source."""
    offenders = []
    for path in _tracked_files():
        text = _read(path)
        for match in _SHADOW_LIBRARY.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line_no}")
    assert not offenders, (
        "shadow-library reference reintroduced into the tracked tree:\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    test_no_third_party_email_in_shipped_configs()
    test_no_shadow_library_reference_in_tracked_tree()
    print("repo hygiene OK")
