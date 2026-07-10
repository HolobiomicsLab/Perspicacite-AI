"""Regression tests: the KB-metadata SQLite must honour `database.path`.

The vector store has always read `database.chroma_path`, while the session
store was hardcoded to a working-directory-relative `./data/perspicacite.db`.
A second instance therefore listed the main hub's knowledge bases while writing
its vectors elsewhere, so every KB it created "existed" but retrieved nothing.
"""

from pathlib import Path

import pytest

from perspicacite.config.paths import LEGACY_SESSION_DB_PATH, resolve_session_db_path
from perspicacite.config.schema import Config


def test_unset_path_keeps_the_legacy_location():
    """Existing deployments keep their metadata where it already lives."""
    assert resolve_session_db_path(Config()) == LEGACY_SESSION_DB_PATH


def test_setting_only_chroma_path_keeps_the_legacy_location():
    """Touching the vector-store path must not move the metadata store."""
    config = Config(database={"chroma_path": "/tmp/isolated-chroma"})
    assert resolve_session_db_path(config) == LEGACY_SESSION_DB_PATH


def test_explicit_path_is_honoured_and_home_is_expanded():
    """`~` must be expanded; SessionStore would otherwise create a `~` directory."""
    resolved = resolve_session_db_path(Config(database={"path": "~/isolated/memory.db"}))
    assert resolved == Path.home() / "isolated" / "memory.db"
    assert "~" not in str(resolved)


def test_env_var_override_reaches_the_resolver(monkeypatch, tmp_path):
    """PERSPICACITE_DB_PATH is documented as the isolation switch."""
    from perspicacite.config.loader import load_config

    target = tmp_path / "instance" / "memory.db"
    monkeypatch.setenv("PERSPICACITE_DB_PATH", str(target))
    assert resolve_session_db_path(load_config(None)) == target


@pytest.mark.asyncio
async def test_app_state_opens_the_configured_database(monkeypatch, tmp_path):
    """AppState.initialize must hand the configured path to SessionStore."""
    from perspicacite.web import state as state_module

    target = tmp_path / "instance" / "memory.db"
    monkeypatch.setenv("PERSPICACITE_DB_PATH", str(target))
    monkeypatch.setenv("PERSPICACITE_ALLOW_MISSING_LLM_KEYS", "1")
    # A throwaway instance whose configured DB does not exist yet.
    monkeypatch.setenv("PERSPICACITE_ALLOW_NEW_DB", "1")
    monkeypatch.setenv("PERSPICACITE_DB_CHROMA_PATH", str(tmp_path / "chroma"))

    opened: list[Path] = []

    class _RecordingSessionStore:
        def __init__(self, db_path):
            opened.append(Path(db_path))
            self.db_path = Path(db_path)

        async def init_db(self):
            return None

    monkeypatch.setattr(state_module, "SessionStore", _RecordingSessionStore)

    app_state = state_module.AppState()
    await app_state.initialize()

    assert opened == [target]
    assert opened[0] != LEGACY_SESSION_DB_PATH


def _make_populated_db(path: Path) -> None:
    """Create a SQLite file with one kb_metadata row, like a real registry."""
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE kb_metadata (name TEXT)")
    connection.execute("INSERT INTO kb_metadata (name) VALUES ('existing-kb')")
    connection.commit()
    connection.close()


def test_guard_refuses_to_orphan_an_existing_registry(monkeypatch, tmp_path):
    """The documented `cp config.example.yml` path must not open an empty DB."""
    from perspicacite.config import paths as paths_module

    legacy = tmp_path / "data" / "perspicacite.db"
    _make_populated_db(legacy)
    monkeypatch.setattr(paths_module, "LEGACY_SESSION_DB_PATH", legacy)
    monkeypatch.delenv(paths_module.ALLOW_NEW_DB_ENV, raising=False)

    config = Config(database={"path": str(tmp_path / "home" / "memory.db")})
    with pytest.raises(paths_module.SessionDatabaseMisconfiguredError) as excinfo:
        paths_module.guard_session_db_path(config)
    assert str(legacy) in str(excinfo.value)


def test_guard_allows_a_deliberately_fresh_instance(monkeypatch, tmp_path):
    """PERSPICACITE_ALLOW_NEW_DB opts out of the orphan guard."""
    from perspicacite.config import paths as paths_module

    legacy = tmp_path / "data" / "perspicacite.db"
    _make_populated_db(legacy)
    monkeypatch.setattr(paths_module, "LEGACY_SESSION_DB_PATH", legacy)
    monkeypatch.setenv(paths_module.ALLOW_NEW_DB_ENV, "1")

    target = tmp_path / "home" / "memory.db"
    resolved = paths_module.guard_session_db_path(Config(database={"path": str(target)}))
    assert resolved == target


def test_guard_is_silent_when_no_legacy_registry_exists(monkeypatch, tmp_path):
    """A first-ever install has no legacy DB, so the guard must not fire."""
    from perspicacite.config import paths as paths_module

    monkeypatch.setattr(paths_module, "LEGACY_SESSION_DB_PATH", tmp_path / "data" / "absent.db")
    monkeypatch.delenv(paths_module.ALLOW_NEW_DB_ENV, raising=False)

    target = tmp_path / "home" / "memory.db"
    resolved = paths_module.guard_session_db_path(Config(database={"path": str(target)}))
    assert resolved == target


def test_guard_ignores_an_empty_legacy_database(monkeypatch, tmp_path):
    """A legacy file with no knowledge bases is not worth blocking startup for."""
    from perspicacite.config import paths as paths_module

    legacy = tmp_path / "data" / "perspicacite.db"
    legacy.parent.mkdir(parents=True)
    legacy.touch()  # exists but has no kb_metadata table
    monkeypatch.setattr(paths_module, "LEGACY_SESSION_DB_PATH", legacy)
    monkeypatch.delenv(paths_module.ALLOW_NEW_DB_ENV, raising=False)

    target = tmp_path / "home" / "memory.db"
    resolved = paths_module.guard_session_db_path(Config(database={"path": str(target)}))
    assert resolved == target


def test_guard_passes_through_when_config_points_at_the_legacy_path(monkeypatch, tmp_path):
    """Pinning database.path to the legacy DB is the intended fix, not an error."""
    from perspicacite.config import paths as paths_module

    legacy = tmp_path / "data" / "perspicacite.db"
    _make_populated_db(legacy)
    monkeypatch.setattr(paths_module, "LEGACY_SESSION_DB_PATH", legacy)
    monkeypatch.delenv(paths_module.ALLOW_NEW_DB_ENV, raising=False)

    resolved = paths_module.guard_session_db_path(Config(database={"path": str(legacy)}))
    assert resolved == legacy
