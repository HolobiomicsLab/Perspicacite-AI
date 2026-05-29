"""Minimal AppState surface for CLI / MCP isolated execution.

The full ``AppState`` in ``web/state.py`` carries FastAPI router
state (lifespan handles, job registry, ...) that CLI subcommands and
the MCP server don't need. ``MinimalAppState`` exposes just the
attributes RAG mode code reads via ``request.app_state``:

- ``config``       : full Config object (for search.query_optimization etc.)
- ``llm_client``   : AsyncLLMClient instance for optimizer / Haiku rewrites

Constructed from a Config in one call. Used by the CLI and the MCP
``generate_report`` / ``web_search`` tools so they no longer need the
heavyweight web AppState singleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MinimalAppState:
    """Subset of AppState that RAG mode code actually reads."""

    config: Any
    llm_client: Any = None

    @classmethod
    def from_config(cls, config: Any) -> MinimalAppState:
        """Build a minimal state with a fresh LLM client."""
        from perspicacite.llm.client import AsyncLLMClient
        client = AsyncLLMClient(config)
        return cls(config=config, llm_client=client)
