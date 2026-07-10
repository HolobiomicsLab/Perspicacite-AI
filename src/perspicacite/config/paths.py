"""Filesystem paths derived from configuration.

The vector store has always honoured ``database.chroma_path`` while the KB
metadata SQLite was hardcoded to a working-directory-relative path. A second
instance therefore read the main hub's KB listing and wrote its vectors
elsewhere, producing knowledge bases that exist in the listing but return
nothing. These helpers give every entry point (web, MCP, CLI) one answer.
"""

from pathlib import Path

from perspicacite.config.schema import Config

# Where the web app, MCP server and CLI historically kept KB metadata, relative
# to the process working directory.
LEGACY_SESSION_DB_PATH = Path("./data/perspicacite.db")


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
