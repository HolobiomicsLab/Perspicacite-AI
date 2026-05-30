"""Synced from AgenticScienceBuilder @ a10eced — keep API in sync.

Notebook output stripper used before ingesting fetched .ipynb files so
embedded image blobs / stderr don't pollute the KB.
"""
from __future__ import annotations

import json

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.external.notebooks")


def strip_notebook_outputs(raw: str) -> str:
    """Remove cell outputs and execution counts from a Jupyter notebook JSON.

    Reduces a typical notebook from 500 KB to 30–80 KB by dropping output
    blobs (images, stderr) while preserving source code and markdown cells.
    Returns the original string unchanged on any parse error.
    """
    try:
        nb = json.loads(raw)
        for cell in nb.get("cells", []):
            cell["outputs"] = []
            cell["execution_count"] = None
        return json.dumps(nb, indent=1, ensure_ascii=False)
    except Exception as exc:
        logger.debug("notebook output stripping failed", error=str(exc))
        return raw
