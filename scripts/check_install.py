#!/usr/bin/env python3
"""Perspicacité install preflight — verify the environment is ready to `serve`.

Prints a status table for every prerequisite (python, uv venv, config,
provider keys, optional server probe). Exits 0 if all critical checks pass.

Mirrors the Scriptorium ``scripts/check_services.py`` convention. Mirror'd
intentionally — the same agent flow that runs ``/setup`` against Scriptorium
should produce the same shape of output here.

Run from the repo root::

    uv run python scripts/check_install.py
    # or, without uv:
    python3 scripts/check_install.py
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COL1 = 26
COL2 = 6
SEP = " " + "─" * 60


def _check_python() -> tuple[str, str]:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 12):
        return "OK", f"{v.major}.{v.minor}.{v.micro}"
    return "FAIL", f"need >=3.12, got {v.major}.{v.minor}.{v.micro}"


def _check_uv() -> tuple[str, str]:
    if uv := shutil.which("uv"):
        return "OK", uv
    return "FAIL", "uv not on PATH — `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`"


def _check_venv() -> tuple[str, str]:
    venv = REPO_ROOT / ".venv"
    if not venv.exists():
        return "FAIL", "no .venv — run: uv sync"
    pkg = venv / "lib"
    if not pkg.exists():
        return "WARN", ".venv exists but looks empty"
    return "OK", ".venv present"


def _check_config() -> tuple[str, str]:
    config = REPO_ROOT / "config.yml"
    if not config.exists():
        return "FAIL", "config.yml missing — run: cp config/example.yml config.yml"
    return "OK", "config.yml present"


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Tiny .env parser — no shell quoting tricks; one key=value per line."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _check_provider_key() -> tuple[str, str]:
    """The CLI loads `.env` from CWD at startup (see cli.py: main → load_dotenv)."""
    PROVIDERS = ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
    dotenv = _parse_dotenv(REPO_ROOT / ".env")
    found_in_env: list[str] = []
    found_in_dotenv: list[str] = []
    for p in PROVIDERS:
        if os.environ.get(p):
            found_in_env.append(p)
        if dotenv.get(p):
            found_in_dotenv.append(p)
    if found_in_env or found_in_dotenv:
        srcs = []
        if found_in_env:
            srcs.append(f"shell({len(found_in_env)})")
        if found_in_dotenv:
            srcs.append(f".env({len(found_in_dotenv)})")
        sample = (found_in_env or found_in_dotenv)[0]
        return "OK", f"{sample} (+others) via {', '.join(srcs)}"
    return "FAIL", (
        "no LLM provider key in shell or .env — "
        "add ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY "
        "(tip: if set in your shell config, run `source ~/.zshrc` or `source ~/.profile`)"
    )


def _check_embedding_key() -> tuple[str, str]:
    """OPENAI_API_KEY is the default for embeddings; without it, fallback to MiniLM."""
    if os.environ.get("OPENAI_API_KEY") or _parse_dotenv(REPO_ROOT / ".env").get("OPENAI_API_KEY"):
        return "OK", "OPENAI_API_KEY set (text-embedding-3-small)"
    return "WARN", "no OPENAI_API_KEY → embeddings will fall back to local all-MiniLM-L6-v2 (~3s first boot)"


def _check_unpaywall_email() -> tuple[str, str]:
    """Required for open-access PDF discovery; non-critical for basic serve."""
    config_path = REPO_ROOT / "config.yml"
    if not config_path.exists():
        return "SKIP", "(config.yml missing)"
    text = config_path.read_text()
    if "your@email.com" in text or "your_email@example.com" in text:
        return "WARN", "placeholder email in config — set pdf_download.unpaywall_email"
    # Coarse check: a real email contains @ in a non-comment line
    found = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if "unpaywall_email" in s and "@" in s:
            found = True
            break
    return ("OK", "configured") if found else ("WARN", "unpaywall_email not configured")


def _check_port_8000() -> tuple[str, str]:
    """Optional — is the server already up? Useful when paired with an audit run."""
    try:
        with socket.create_connection(("localhost", 8000), timeout=0.5):
            return "OK", "server already listening on :8000"
    except (ConnectionRefusedError, OSError, TimeoutError, socket.timeout):
        return "SKIP", "no server on :8000 (start with: uv run perspicacite -c config.yml serve)"


def _check_import() -> tuple[str, str]:
    """Confirm perspicacite + python-dotenv (used by the new `main()` shim) import."""
    try:
        import perspicacite  # noqa: F401
        try:
            from dotenv import load_dotenv  # noqa: F401
            return "OK", "perspicacite + dotenv importable"
        except ImportError:
            return "WARN", "perspicacite importable but python-dotenv missing — .env won't load"
    except ImportError as e:
        return "FAIL", f"import failed: {e} (run `uv sync`)"


def main() -> int:
    rows: list[tuple[str, str, str]] = [
        ("Python ≥ 3.12", *_check_python()),
        ("uv on PATH", *_check_uv()),
        (".venv installed", *_check_venv()),
        ("Importable", *_check_import()),
        ("config.yml", *_check_config()),
        ("Provider key (chat)", *_check_provider_key()),
        ("Provider key (embed)", *_check_embedding_key()),
        ("Unpaywall email", *_check_unpaywall_email()),
        ("Server :8000", *_check_port_8000()),
    ]

    print("Perspicacité — install check")
    print()
    print(f" {'Check':<{COL1}}  {'Status':<{COL2}}  Details")
    print(SEP)
    for name, status, detail in rows:
        print(f" {name:<{COL1}}  {status:<{COL2}}  {detail}")
    print(SEP)
    n_fail = sum(1 for _, s, _ in rows if s == "FAIL")
    n_warn = sum(1 for _, s, _ in rows if s == "WARN")
    n_ok = sum(1 for _, s, _ in rows if s == "OK")
    print(f" {n_ok} OK · {n_warn} WARN · {n_fail} FAIL")
    if n_fail == 0:
        print()
        print(" Next: start the server →")
        print("   uv run perspicacite -c config.yml serve")
        print("   curl -s http://localhost:8000/api/health")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
