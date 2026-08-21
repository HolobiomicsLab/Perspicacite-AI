"""Filesystem paths derived from configuration.

The vector store has always honoured ``database.chroma_path`` while the KB
metadata SQLite was hardcoded to a working-directory-relative path. A second
instance therefore read the main hub's KB listing and wrote its vectors
elsewhere, producing knowledge bases that exist in the listing but return
nothing. These helpers give every entry point (web, MCP, CLI) one answer.
"""

import os
import sqlite3
from pathlib import Path

from perspicacite.config.schema import Config

# Where the web app, MCP server and CLI historically kept KB metadata, relative
# to the process working directory.
LEGACY_SESSION_DB_PATH = Path("./data/perspicacite.db")

# Set to start a deliberately fresh instance whose configured database does not
# yet exist, bypassing the orphaned-registry guard below.
ALLOW_NEW_DB_ENV = "PERSPICACITE_ALLOW_NEW_DB"


class SessionDatabaseMisconfiguredError(RuntimeError):
    """The configured session database is absent while a populated one is not.

    Honouring the configuration would open an empty database and hide every
    existing knowledge base. Raised at startup so the operator fixes the path
    (or opts into a fresh instance) rather than discovering an empty registry.
    """


def resolve_session_db_path(config: Config) -> Path:
    """Return the KB-metadata SQLite path for this configuration.

    ``database.path`` is honoured only when it was actually supplied, by config
    file or by ``PERSPICACITE_DB_PATH``. Applying the schema default instead
    would silently relocate the metadata of every existing deployment, whose
    knowledge bases live in the legacy working-directory path.
    """
    if "path" in config.database.model_fields_set:
        return config.database.path.expanduser()
    return LEGACY_SESSION_DB_PATH


def _holds_knowledge_bases(db_path: Path) -> bool:
    """Return True when ``db_path`` is a SQLite file with a non-empty registry."""
    if not db_path.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kb_metadata'"
        ).fetchone()
        if not has_table:
            return False
        return connection.execute("SELECT 1 FROM kb_metadata LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def guard_session_db_path(config: Config) -> Path:
    """Resolve the session-DB path, refusing to orphan an existing registry.

    The shipped ``config.example.yml`` sets ``database.path`` to a home-directory
    location, while deployments created before the path was honoured keep their
    knowledge bases in the legacy working-directory database. Starting against
    the configured (empty) path would present an empty KB list and write new
    metadata somewhere the vectors do not follow. Rather than move data
    implicitly, refuse to start and name the fix. ``PERSPICACITE_ALLOW_NEW_DB=1``
    opts into a deliberately fresh instance.
    """
    resolved = resolve_session_db_path(config)
    if os.environ.get(ALLOW_NEW_DB_ENV):
        return resolved
    if resolved.exists() or resolved == LEGACY_SESSION_DB_PATH:
        return resolved
    if not _holds_knowledge_bases(LEGACY_SESSION_DB_PATH):
        return resolved
    raise SessionDatabaseMisconfiguredError(
        f"configured session database {resolved} does not exist, but "
        f"{LEGACY_SESSION_DB_PATH.resolve()} holds existing knowledge bases. "
        f"Starting here would present an empty registry. Set database.path to "
        f"'{LEGACY_SESSION_DB_PATH}' in your config, or move the file to "
        f"{resolved}, or set {ALLOW_NEW_DB_ENV}=1 to start a fresh instance."
    )


if __name__ == "__main__":
    assert resolve_session_db_path(Config()) == LEGACY_SESSION_DB_PATH
    assert guard_session_db_path(Config()) == LEGACY_SESSION_DB_PATH
    print("paths smoke check OK")
