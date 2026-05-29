"""Perf regression baseline (Wave 6.3).

Runs the same pipeline (deterministic embedder, no network) against a
fixed 5-paper corpus at ``tests/data/perf_corpus/`` and compares
against a stored baseline at ``tests/data/perf_baseline.json``.

Behaviour:
- ``PERSPICACITE_UPDATE_PERF_BASELINE=1 pytest -m perf`` regenerates
  the baseline and skips the assertion.
- Plain ``pytest -m perf`` compares current run against baseline.
- ``PERSPICACITE_PERF_TOLERANCE=<float>`` overrides the 1.30 default.
- Marked ``perf`` so CI can run it selectively (and skip in the fast
  developer loop).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

# Reuse the e2e conftest's deterministic mocks. pytest plugin loading
# picks up tests/e2e/conftest.py automatically only for tests under
# tests/e2e/, so we re-import the helper class here.
chromadb = pytest.importorskip("chromadb")
np = pytest.importorskip("numpy")

from perspicacite.models.papers import Author, Paper, PaperSource

pytestmark = pytest.mark.perf


CORPUS_DIR = Path(__file__).parent.parent / "data" / "perf_corpus"
BASELINE_PATH = Path(__file__).parent.parent / "data" / "perf_baseline.json"
UPDATE = os.environ.get("PERSPICACITE_UPDATE_PERF_BASELINE") == "1"
TOLERANCE = float(os.environ.get("PERSPICACITE_PERF_TOLERANCE", "1.30"))

# Timings below ~10 ms are dominated by syscall / GC jitter on a typical
# laptop. Comparing them against a fixed baseline produces flake. We skip
# the regression check for any metric whose baseline is below the noise
# floor; we still record the current value so trends are visible.
_NOISE_FLOOR_SECONDS = 0.010
_NOISE_FLOOR_THROUGHPUT_PER_S = 1000.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_corpus() -> list[Paper]:
    """Build Paper instances from the JSON corpus."""
    papers: list[Paper] = []
    for f in sorted(CORPUS_DIR.glob("paper_*.json")):
        d = json.loads(f.read_text())
        papers.append(
            Paper(
                id=d["id"],
                doi=d["doi"],
                title=d["title"],
                authors=[Author(**a) for a in d["authors"]],
                year=d["year"],
                abstract=d["abstract"],
                full_text=d["full_text"],
                source=PaperSource.WEB_SEARCH,
            )
        )
    return papers


class _PerfEmbedder:
    """Deterministic embedder for perf measurement.

    Same shape as the e2e DeterministicEmbeddingProvider — but inlined
    here so this file does not depend on tests/e2e/conftest.py being
    loaded.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "perf-mock-embedder"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        self.calls += 1
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec: list[float] = []
            while len(vec) < self._dim:
                for b in h:
                    vec.append((b / 127.5) - 1.0)
                    if len(vec) >= self._dim:
                        break
                h = hashlib.sha256(h).digest()
            # Cosine-normalize for Chroma's cosine space.
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            out.append(arr.tolist())
        return out


@pytest.mark.asyncio
async def test_perf_baseline(tmp_path: Path) -> None:
    from perspicacite.pipeline.kb_log import KBEvent, KBLogWriter
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig
    from perspicacite.retrieval.chroma_store import ChromaVectorStore

    papers = _load_corpus()
    assert len(papers) == 5, f"Expected 5 papers in {CORPUS_DIR}, got {len(papers)}"

    embedder = _PerfEmbedder()
    vs = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_provider=embedder,
    )
    cfg = KnowledgeBaseConfig(
        vector_size=embedder.dimension,
        chunk_size=500,
        chunk_overlap=50,
        top_k=10,
    )
    kb = DynamicKnowledgeBase(
        vector_store=vs,
        embedding_service=embedder,
        config=cfg,
    )
    await kb.initialize()

    # Warm-up: hit each hot path once so imports / lazy initialisation
    # don't taint the timing of the actual measurement below.
    await embedder.embed(["warmup-text"])
    await kb.search(query="warmup query", top_k=5)

    # 1. Ingest timing (chunking + embedding + chroma writes)
    t0 = time.perf_counter()
    await kb.add_papers(papers)
    ingest_s = time.perf_counter() - t0

    # 2. Search timing (averaged over 10 queries)
    queries = [
        "energy efficient transformer inference consumer GPU",
        "retrieval augmented generation noisy citation graph",
        "differential privacy vector database clinical text",
        "multilingual sentence transformer scientific abstract",
        "reproducibility crisis NLP benchmark meta analysis",
        "stellar nucleosynthesis red giant evolution",
        "AlphaFold protein structure prediction",
        "cryo-em ribosome assembly intermediates",
        "exoplanet biosignature atmospheric spectroscopy",
        "fused softmax attention kernel HBM traffic",
    ]
    t0 = time.perf_counter()
    for q in queries:
        await kb.search(query=q, top_k=10)
    search_s_avg = (time.perf_counter() - t0) / len(queries)

    # 3. "Report synthesis" stand-in: 20 embeddings of medium text.
    medium_text = "The results indicate a strong correlation. " * 20
    t0 = time.perf_counter()
    await embedder.embed([medium_text] * 20)
    report_s = time.perf_counter() - t0

    # 4. Embeddings/sec on small batches of 100 short texts.
    t0 = time.perf_counter()
    await embedder.embed(["sample text " + str(i) for i in range(100)])
    emb_dt = time.perf_counter() - t0
    emb_per_s = 100 / max(emb_dt, 1e-9)

    # 5. KB log writes/sec — 500 events on a fresh log file.
    log = KBLogWriter(path=tmp_path / "perf_log.jsonl")
    t0 = time.perf_counter()
    for i in range(500):
        log.append(KBEvent(
            event="paper_added",
            kb_name="perf",
            paper_id=f"perf_{i}",
            title="perf paper",
            chunks=1,
            source_command="perf_test",
        ))
    log_dt = time.perf_counter() - t0
    log_writes_per_s = 500 / max(log_dt, 1e-9)

    metrics = {
        "ingest_5_papers_seconds": round(ingest_s, 4),
        "search_top10_seconds_avg": round(search_s_avg, 4),
        "report_synthesis_seconds": round(report_s, 4),
        "embeddings_per_second": round(emb_per_s, 1),
        "kb_log_writes_per_second": round(log_writes_per_s, 1),
        "git_sha": _git_sha(),
        "timestamp": time.time(),
    }

    if UPDATE or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
        pytest.skip(
            f"Baseline written to {BASELINE_PATH}. "
            f"Re-run without PERSPICACITE_UPDATE_PERF_BASELINE."
        )

    baseline = json.loads(BASELINE_PATH.read_text())
    failures: list[str] = []
    speedups: list[str] = []
    skipped: list[str] = []

    # Lower-is-better metrics
    for k in (
        "ingest_5_papers_seconds",
        "search_top10_seconds_avg",
        "report_synthesis_seconds",
    ):
        cur, base = metrics[k], baseline.get(k, metrics[k])
        if base < _NOISE_FLOOR_SECONDS:
            skipped.append(f"{k} (baseline={base:.4f}s below noise floor)")
            continue
        ratio = cur / max(base, 1e-9)
        if ratio > TOLERANCE:
            failures.append(
                f"{k}: {ratio:.2f}× slower (current={cur:.4f}, baseline={base:.4f})"
            )
        elif ratio < (1 / TOLERANCE):
            speedups.append(f"{k}: {1 / ratio:.2f}× faster")

    # Higher-is-better metrics
    for k in ("embeddings_per_second", "kb_log_writes_per_second"):
        cur, base = metrics[k], baseline.get(k, metrics[k])
        if base < _NOISE_FLOOR_THROUGHPUT_PER_S:
            skipped.append(f"{k} (baseline={base:.1f}/s below noise floor)")
            continue
        ratio = base / max(cur, 1e-9)
        if ratio > TOLERANCE:
            failures.append(
                f"{k}: {ratio:.2f}× slower (current={cur:.1f}/s, baseline={base:.1f}/s)"
            )
        elif ratio < (1 / TOLERANCE):
            speedups.append(f"{k}: {1 / ratio:.2f}× faster")

    if skipped:
        print("Perf metrics skipped (below noise floor): " + "; ".join(skipped))

    if speedups:
        print("Perf speedups vs baseline: " + "; ".join(speedups))
    if failures:
        pytest.fail("Perf regression beyond tolerance:\n  " + "\n  ".join(failures))
