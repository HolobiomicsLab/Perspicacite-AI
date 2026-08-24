"""Regression: chroma's HNSW cache must bound what the server holds open.

chromadb sizes its resident-index cache from the soft ``RLIMIT_NOFILE`` it sees
while the client is built (``chromadb/api/rust.py``: ``max_file_handles // 5``,
"4 data files and 1 metadata file"). On a host whose ``ulimit -n`` is in the
millions that budget is effectively unbounded, so a long-lived server keeps
every collection's index resident and the files it holds grow linearly with the
number of collections touched.

Measured on chromadb 1.5.9 / macOS: an unbounded client retains exactly 4 files
per collection, matching the production signature of 1,981 collections against
7,924 held files. These tests pin the bound by comparing a budgeted client
against an unbounded control built in the same process.
"""

import os
import shutil
import subprocess
from pathlib import Path

import chromadb
import pytest

from perspicacite.retrieval import chroma_store
from perspicacite.retrieval.chroma_store import (
    FD_COUNT_UNAVAILABLE,
    ChromaVectorStore,
    _bounded_chroma_client,
    _hnsw_index_budget,
)

# Enough collections to overrun the shrunken cache; ~1s per client to build.
COLLECTION_COUNT = 120
# Vectors per collection: enough to force a real on-disk HNSW index.
VECTORS_PER_COLLECTION = 20
# Tiny embeddings keep the index files small and the test fast.
EMBEDDING_DIM = 8
# Soft limit for the budgeted client; below ~500 chroma's rust runtime aborts.
TEST_SOFT_FD_LIMIT = 500
# Unbounded retains 4 files per collection, so demand a clear separation.
MAX_BOUNDED_FRACTION_OF_CONTROL = 0.75


def _files_held_under(directory: str) -> int:
    """Count files this process holds open or memory-mapped under a directory.

    HNSW index files are memory-mapped, so ``/proc/self/fd`` alone misses them;
    ``/proc/self/maps`` and ``lsof`` both report mappings.

    Args:
        directory: Absolute, resolved path to count entries beneath.

    Returns:
        Number of matching entries, or FD_COUNT_UNAVAILABLE when neither
        source is present.
    """
    maps = Path("/proc/self/maps")
    if maps.is_file():
        return _count_lines_mentioning(maps.read_text(), directory)
    listing = _lsof_self()
    if listing is None:
        return FD_COUNT_UNAVAILABLE
    return _count_lines_mentioning(listing, directory)


def _count_lines_mentioning(listing: str, directory: str) -> int:
    """Count listing lines that reference a directory.

    Args:
        listing: Text of a maps or lsof listing.
        directory: Absolute path to look for.

    Returns:
        Number of lines mentioning the directory.
    """
    return sum(1 for line in listing.splitlines() if directory in line)


def _lsof_self() -> str | None:
    """Read this process's open-file listing via lsof.

    Returns:
        The lsof output, or None when lsof is unavailable or fails.
    """
    if shutil.which("lsof") is None:
        return None
    try:
        result = subprocess.run(
            ["lsof", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout or None


def _fill_client(client, persist_dir: str) -> int:
    """Create many small collections and query each one back.

    Args:
        client: A chromadb client to populate.
        persist_dir: Resolved path the client persists to.

    Returns:
        Files held under persist_dir once every collection has been queried.
    """
    for index in range(COLLECTION_COUNT):
        collection = client.create_collection(f"plateau{index:04d}")
        collection.add(
            ids=[f"{index}-{n}" for n in range(VECTORS_PER_COLLECTION)],
            embeddings=_vectors(index),
            documents=[f"doc{n}" for n in range(VECTORS_PER_COLLECTION)],
        )
        collection.query(query_embeddings=_vectors(index)[:1], n_results=2)
    return _files_held_under(persist_dir)


def _vectors(seed: int) -> list[list[float]]:
    """Build deterministic embeddings for one collection.

    Args:
        seed: Collection index, so vectors differ between collections.

    Returns:
        A list of VECTORS_PER_COLLECTION embeddings.
    """
    return [
        [0.01 * n + 0.02 * seed] * EMBEDDING_DIM
        for n in range(VECTORS_PER_COLLECTION)
    ]


@pytest.fixture
def shrunken_budget(monkeypatch):
    """Shrink the descriptor budget so eviction happens within the test."""
    monkeypatch.setattr(chroma_store, "CHROMA_HNSW_FD_BUDGET", TEST_SOFT_FD_LIMIT)
    monkeypatch.setattr(chroma_store, "CHROMA_MIN_SOFT_FD_LIMIT", TEST_SOFT_FD_LIMIT)
    monkeypatch.setattr(chroma_store, "CHROMA_FD_HEADROOM", 0)
    chroma_store._budgeted_persist_dirs.clear()
    yield
    chroma_store._budgeted_persist_dirs.clear()


def test_held_files_plateau_against_an_unbounded_control(tmp_path, shrunken_budget):
    """The budgeted client must hold far fewer files than an unbounded one."""
    bounded_dir = str((tmp_path / "bounded").resolve())
    control_dir = str((tmp_path / "control").resolve())
    if _files_held_under(bounded_dir) == FD_COUNT_UNAVAILABLE:
        pytest.skip("platform reports no open-file listing")

    bounded_held = _fill_client(_bounded_chroma_client(bounded_dir), bounded_dir)
    control_held = _fill_client(chromadb.PersistentClient(path=control_dir), control_dir)

    assert control_held >= COLLECTION_COUNT, (
        f"control held only {control_held} files for {COLLECTION_COUNT} "
        "collections; it is not the unbounded baseline this test compares to"
    )
    assert bounded_held < control_held * MAX_BOUNDED_FRACTION_OF_CONTROL, (
        f"bounded client held {bounded_held} files vs control {control_held}: "
        "the HNSW cache is growing with collection count, not plateauing"
    )


def test_store_client_gets_the_budgeted_cache_size(tmp_path, mock_embedding_provider):
    """The production path must wire the budget into chroma's own knob."""
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "kb"), embedding_provider=mock_embedding_provider
    )
    server = getattr(store.client, "_server", None)
    if not hasattr(server, "hnsw_cache_size"):
        pytest.skip("chromadb no longer exposes hnsw_cache_size")

    assert server.hnsw_cache_size == _hnsw_index_budget()


def test_one_persist_dir_is_budgeted_once(tmp_path, mock_embedding_provider):
    """Two spellings of one directory share a single budgeted System."""
    chroma_store._budgeted_persist_dirs.clear()
    (tmp_path / "kb").mkdir()
    first = ChromaVectorStore(
        persist_dir=str(tmp_path / "kb"), embedding_provider=mock_embedding_provider
    )
    second = ChromaVectorStore(
        persist_dir=str(tmp_path / "." / "kb"), embedding_provider=mock_embedding_provider
    )

    assert first.persist_dir == second.persist_dir
    assert list(chroma_store._budgeted_persist_dirs).count(first.persist_dir) == 1


if __name__ == "__main__":
    # Offline smoke check: no client, no chroma_db/ access.
    print(f"held_under_root={_files_held_under('/')}")
    assert _count_lines_mentioning("a/b\nc/d\n", "a/b") == 1
    assert _count_lines_mentioning("a/b\nc/d\n", "zz") == 0
    assert len(_vectors(3)) == VECTORS_PER_COLLECTION
    assert len(_vectors(3)[0]) == EMBEDDING_DIM
    print("fd plateau helpers smoke ok")
