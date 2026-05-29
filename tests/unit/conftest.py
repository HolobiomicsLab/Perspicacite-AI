"""Unit test configuration.

Provides fixtures that apply to all unit tests under this directory.
"""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _mock_app_state_initialized(monkeypatch):
    """Prevent app_state.initialize() from running in unit tests.

    Unit tests that exercise router functions directly need app_state to
    appear already initialized so the preflight LLM-key check and full
    component bootstrap (ChromaDB, embeddings, etc.) are skipped. Each
    test that needs specific app_state attributes should patch those
    attributes individually via monkeypatch or unittest.mock.patch.
    """
    from perspicacite.web import state as _state_mod

    monkeypatch.setattr(_state_mod.app_state, "initialized", True, raising=False)
    monkeypatch.setattr(
        _state_mod.app_state,
        "initialize",
        AsyncMock(return_value=None),
        raising=False,
    )
    monkeypatch.setattr(_state_mod.app_state, "session_store", None, raising=False)
    monkeypatch.setattr(_state_mod.app_state, "provenance_store", None, raising=False)
