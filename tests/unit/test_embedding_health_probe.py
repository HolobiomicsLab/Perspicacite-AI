"""Regression tests for the embedding-health probe in GET /api/kb/{name}/stats.

The probe samples stored vectors to flag a knowledge base poisoned with zero
vectors. A mid-ingest embedding outage leaves a contiguous run of zeros, usually
at the end of the collection, so a prefix-only sample would report a heavily
poisoned KB as healthy. The probe reads both ends and states whether the whole
collection was checked, so a sample is never mistaken for a proof.
"""

import chromadb

from perspicacite.web.routers.kb import EMBEDDING_PROBE_LIMIT, _probe_embedding_health

DIMENSION = 8
_UNIT = [1.0] + [0.0] * (DIMENSION - 1)


def _collection(tmp_path, zero_from: int, total: int):
    """A Chroma collection of `total` vectors, zero-filled from index `zero_from`."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    coll = client.create_collection(name="probe-kb")
    embeddings = [[0.0] * DIMENSION if i >= zero_from else _UNIT for i in range(total)]
    coll.add(
        ids=[f"c{i}" for i in range(total)],
        embeddings=embeddings,
        documents=[f"doc {i}" for i in range(total)],
    )
    return coll


def test_empty_collection_is_complete_and_not_degraded():
    assert _probe_embedding_health(None, 0) == {
        "probed_chunks": 0,
        "zero_vector_chunks": 0,
        "degraded": False,
        "total_chunks": 0,
        "complete": True,
    }


def test_small_clean_collection_is_probed_completely(tmp_path):
    coll = _collection(tmp_path, zero_from=999, total=20)
    health = _probe_embedding_health(coll, 20)
    assert health["degraded"] is False
    assert health["complete"] is True
    assert health["probed_chunks"] == 20


def test_late_poisoning_beyond_the_prefix_is_detected(tmp_path):
    """Zeros past the first 128 chunks must still flip degraded to True."""
    total = EMBEDDING_PROBE_LIMIT * 4
    coll = _collection(tmp_path, zero_from=total - 10, total=total)
    health = _probe_embedding_health(coll, total)
    assert health["degraded"] is True, "tail poisoning must not be reported healthy"
    assert health["zero_vector_chunks"] >= 1
    assert health["complete"] is False
    assert health["total_chunks"] == total


def test_large_clean_sample_is_flagged_incomplete(tmp_path):
    """A clean sample of a big KB is not proof the whole KB is clean."""
    total = EMBEDDING_PROBE_LIMIT * 4
    coll = _collection(tmp_path, zero_from=total + 1, total=total)
    health = _probe_embedding_health(coll, total)
    assert health["degraded"] is False
    assert health["complete"] is False
    assert health["probed_chunks"] == EMBEDDING_PROBE_LIMIT
