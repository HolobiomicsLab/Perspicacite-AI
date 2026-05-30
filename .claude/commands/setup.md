---
name: setup
description: Pre-flight install check — verify the Perspicacité environment before `serve`.
---

Run the install verification:

```bash
uv run python scripts/check_install.py
```

Prints a status table for Python version, uv, the `.venv`, `config.yml`,
provider API keys (chat + embedding), Unpaywall email config, and whether a
server is already running on :8000. Exits 0 if all critical checks pass.

If something fails, the script names the exact next command to run.
For the full step-by-step install walkthrough see [`INSTALL_AGENT.md`](../../docs/INSTALL_AGENT.md).
