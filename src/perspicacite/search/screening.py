from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rank_bm25 import BM25Plus

from perspicacite.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "that",
        "this",
        "it",
        "its",
        "as",
        "not",
        "no",
        "can",
        "will",
        "do",
        "did",
        "so",
        "if",
        "he",
        "she",
        "we",
        "they",
        "their",
        "our",
        "which",
        "who",
        "what",
        "all",
        "also",
        "more",
        "than",
        "up",
        "out",
        "about",
        "into",
        "such",
        "may",
        "each",
        "how",
        "when",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumeric tokens, drop stopwords and single chars."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _candidate_text(candidate: dict) -> str:
    title = candidate.get("title", "") or ""
    abstract = candidate.get("abstract", "") or ""
    return f"{title} {abstract}"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity clamped to [0, 1] (negatives -> 0). Zero vectors -> 0."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, dot / (na * nb))


def _topn_mean(scores: list[float], n: int) -> float:
    """Mean of the top-``n`` scores (all of them if fewer). Empty -> 0."""
    if not scores:
        return 0.0
    top = sorted(scores, reverse=True)[: max(1, n)]
    return sum(top) / len(top)


@dataclass
class ScreenResult:
    item: dict
    score: float
    kept: bool
    reason: str = field(default="")


def screen_papers(
    candidates: Sequence[dict],
    reference: str | Sequence[str],
    method: str = "bm25",
    threshold: float = 0.3,
) -> list[ScreenResult]:
    """Screen candidate papers against a reference query using BM25 text similarity.

    Args:
        candidates: List of paper dicts with at least 'title' and/or 'abstract'.
        reference: Reference query string or list of query strings.
        method: Must be 'bm25'. Use screen_papers_llm for LLM scoring.
        threshold: Normalised score threshold above which a paper is 'kept'.

    Returns:
        List of ScreenResult objects sorted by score descending.
    """
    if method != "bm25":
        raise ValueError(
            "screen_papers only supports method='bm25'; use screen_papers_llm for LLM scoring"
        )

    refs: list[str] = [reference] if isinstance(reference, str) else list(reference)

    # Tokenize reference strings and drop empty ones
    ref_token_lists = [_tokenize(r) for r in refs]
    ref_token_lists = [t for t in ref_token_lists if t]

    if not ref_token_lists:
        return [ScreenResult(item=c, score=0.0, kept=False) for c in candidates]

    # Tokenize candidate documents
    candidate_tokens = [_tokenize(_candidate_text(c)) for c in candidates]

    # BM25Okapi crashes on empty token lists — substitute a sentinel
    safe_tokens = [t if t else ["__empty__"] for t in candidate_tokens]
    bm25 = BM25Plus(safe_tokens)

    # Score each candidate as the max score across all reference queries
    raw_scores = [0.0] * len(candidates)
    for ref_tokens in ref_token_lists:
        scores = bm25.get_scores(ref_tokens)
        for i, s in enumerate(scores):
            if s > raw_scores[i]:
                raw_scores[i] = float(s)

    # Absolute saturation function: s / (s + k). With k≈3 this gives a
    # threshold of 0.5 at BM25 score ≈3, matching where ms-marco rerank
    # and LLM-judge produce "borderline relevant" papers in practice.
    # The 'score' field is comparable across BM25/rerank/LLM tiers and
    # the threshold has consistent semantics. Tie-broken at score ties by
    # raw BM25 to preserve in-batch ordering.
    _K = 3.0
    norm_scores = [float(s) / (float(s) + _K) for s in raw_scores]

    results = [
        ScreenResult(
            item=candidates[i],
            score=norm_scores[i],
            kept=norm_scores[i] >= threshold,
            reason=f"bm25_raw={raw_scores[i]:.2f}",
        )
        for i in range(len(candidates))
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    kept_count = sum(r.kept for r in results)
    logger.info(
        "screen_papers_bm25",
        n=len(candidates),
        kept=kept_count,
        threshold=threshold,
    )
    return results


def screen_papers_setwise_bm25(
    candidates: Sequence[dict],
    *,
    reference_papers: Sequence[Sequence[str]],
    intra_k: int = 3,
    top_n: int = 5,
    threshold: float = 0.3,
) -> list[ScreenResult]:
    """BM25 two-level aggregation mirroring the embedding scorer. Per-paper raw
    score = mean of the candidate's top-``intra_k`` BM25 scores over that paper's
    texts; candidate raw = mean of the top-``top_n`` papers; then ``s/(s+k)``
    saturation to [0,1] (same k as ``screen_papers``).
    """
    cands = list(candidates)
    if not cands:
        return []
    # Tokenize each paper's texts (drop empties); keep paper grouping.
    papers_tokens: list[list[list[str]]] = []
    for p in reference_papers:
        toks = [t for t in (_tokenize(x) for x in p) if t]
        if toks:
            papers_tokens.append(toks)
    if not papers_tokens:
        return [
            ScreenResult(item=c, score=0.0, kept=False, reason="no_reference_papers")
            for c in cands
        ]

    candidate_tokens = [_tokenize(_candidate_text(c)) for c in cands]
    safe_tokens = [t if t else ["__empty__"] for t in candidate_tokens]
    bm25 = BM25Plus(safe_tokens)

    # per_paper_raw[i] = list of paper-level raw BM25 scores for candidate i
    per_paper_raw: list[list[float]] = [[] for _ in cands]
    for paper_toks in papers_tokens:
        # score every candidate against each text of this paper
        per_text_scores: list[list[float]] = [[] for _ in cands]
        for text_tokens in paper_toks:
            scores = bm25.get_scores(text_tokens)
            for i, s in enumerate(scores):
                per_text_scores[i].append(float(s))
        for i in range(len(cands)):
            per_paper_raw[i].append(_topn_mean(per_text_scores[i], intra_k))

    sat_k = 3.0
    results: list[ScreenResult] = []
    for i, c in enumerate(cands):
        raw = _topn_mean(per_paper_raw[i], top_n)
        norm = raw / (raw + sat_k)
        results.append(
            ScreenResult(
                item=c, score=norm, kept=norm >= threshold,
                reason=f"bm25_top{min(top_n, len(per_paper_raw[i]))}papers_mean={norm:.3f}",
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results


async def screen_papers_rerank(
    candidates: Sequence[dict],
    query: str,
    threshold: float = 0.3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[ScreenResult]:
    """Cross-encoder rerank screening (tier B).

    More accurate than BM25 (tier A) because the cross-encoder sees the
    query + paper jointly and can score semantic similarity beyond
    keyword overlap — catches "wrong domain entirely" hits that share
    surface keywords (e.g. a "graph neural network" query that BM25
    matches against combinatorial graph theory).

    Requires ``sentence-transformers`` (already a project dep). The
    default ``ms-marco-MiniLM-L-6-v2`` model is small (~80MB) and runs
    on CPU at ~5ms per pair. First call loads + caches the model.

    Raw model scores are logit-shaped (roughly -10..+10 for the
    ms-marco family); we apply a sigmoid to normalize them to [0,1]
    so the ``threshold`` parameter is consistent with the BM25 and LLM
    paths.

    Args:
        candidates: List of paper dicts with at least ``title`` and/or ``abstract``.
        query: Free-text relevance query.
        threshold: Normalized score >= threshold means kept=True.
        model_name: Override the default cross-encoder.

    Returns:
        List of ``ScreenResult`` sorted by score descending.
    """
    candidates_list = list(candidates)
    if not candidates_list:
        return []

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers required for rerank screening. "
            "Install with: pip install sentence-transformers"
        ) from exc

    import asyncio
    import math

    loop = asyncio.get_running_loop()
    # Try the cache first. CrossEncoder's default load does a HEAD against
    # huggingface.co to check for updates, which 429s under modest load and
    # then sleeps 31 s × 5 retries — wedging the whole RAG path. When the
    # model is already cached locally (the common case), local_files_only
    # skips the network entirely. Fall back to a full load only when the
    # cache miss raises.
    def _load_offline_or_fallback() -> CrossEncoder:
        try:
            return CrossEncoder(model_name, local_files_only=True)
        except Exception:
            return CrossEncoder(model_name)

    model = await loop.run_in_executor(None, _load_offline_or_fallback)

    pairs = [
        (query, _candidate_text(c)[:2000])  # cap each input for throughput
        for c in candidates_list
    ]
    raw_scores = await loop.run_in_executor(None, lambda: model.predict(pairs))

    # Tempered sigmoid -> [0,1]. The ms-marco family emits logits roughly
    # -10..+10; a plain sigmoid saturates above ~10 so every "good" hit
    # collapses to 1.0 and the threshold loses resolution. Dividing by
    # T=4.0 keeps the active band wide enough to distinguish a logit-of-6
    # hit (≈0.82) from a logit-of-10 hit (≈0.92), so the returned score
    # is actually informative as a ranking signal.
    _T = 4.0
    norm_scores = [1.0 / (1.0 + math.exp(-float(s) / _T)) for s in raw_scores]

    results = [
        ScreenResult(
            item=candidates_list[i],
            score=norm_scores[i],
            kept=norm_scores[i] >= threshold,
            reason=f"rerank_logit={float(raw_scores[i]):.2f}",
        )
        for i in range(len(candidates_list))
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    kept_count = sum(r.kept for r in results)
    logger.info(
        "screen_papers_rerank",
        n=len(candidates_list),
        kept=kept_count,
        threshold=threshold,
        model=model_name,
    )
    return results


async def screen_papers_llm(
    candidates: Sequence[dict],
    query: str,
    llm: Any,
    threshold: float = 0.5,
    batch_size: int = 20,
    model: str | None = None,
    provider: str | None = None,
) -> list[ScreenResult]:
    """Screen candidate papers against a query using LLM 0-1 relevance scoring.

    Args:
        candidates: List of paper dicts with at least 'title' and/or 'abstract'.
        query: Research query to score papers against.
        llm: AsyncLLMClient (or compatible) with async complete(messages) method.
        threshold: Score >= threshold means kept=True.
        batch_size: Number of papers per LLM call.

    Returns:
        List of ScreenResult objects sorted by score descending.
    """
    candidates = list(candidates)
    results: list[ScreenResult] = []

    system_msg = (
        "You rate how relevant each paper is to a research query. "
        "Respond ONLY with a JSON array of objects "
        '{"index": int, "score": float in [0,1], "reason": short string}.'
    )

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        lines = []
        for local_i, c in enumerate(batch):
            title = c.get("title", "") or ""
            abstract = (c.get("abstract", "") or "")[:400]
            lines.append(f"{local_i}. {title} — {abstract}")
        listing = "\n".join(lines)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Query: {query}\n\nPapers:\n{listing}"},
        ]

        try:
            kw: dict[str, Any] = {"messages": messages, "stage": "screening"}
            if model is not None:
                kw["model"] = model
            if provider is not None:
                kw["provider"] = provider
            raw = await llm.complete(**kw)
            text: str = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            match = re.search(r"\[.*\]", text, re.S)
            if match is None:
                raise ValueError("No JSON array found in response")
            parsed = json.loads(match.group())
        except Exception:
            logger.warning("screen_papers_llm_parse_failed", batch_start=batch_start)
            parsed = []

        # Build index -> object map for items that have a valid index key
        index_map: dict[int, dict] = {}
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj["index"])
                index_map[idx] = obj
            except (KeyError, TypeError, ValueError):
                continue

        for local_i, candidate in enumerate(batch):
            obj = index_map.get(local_i, {})
            score = float(obj.get("score", 0.0) or 0.0)
            reason = str(obj.get("reason", ""))
            results.append(
                ScreenResult(
                    item=candidate,
                    score=score,
                    kept=score >= threshold,
                    reason=reason,
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)

    kept_count = sum(r.kept for r in results)
    logger.info("screen_papers_llm", n=len(candidates), kept=kept_count)
    return results


async def screen_papers_embedding(
    candidates: Sequence[dict],
    *,
    reference_papers: Sequence[Sequence[str]],
    embedding_provider: Any,
    intra_k: int = 3,
    top_n: int = 5,
    threshold: float = 0.3,
) -> list[ScreenResult]:
    """Embedding similarity, two-level. Each KB paper is a list of texts
    (``[abstract]`` or chunk texts). Per-paper similarity = mean of the
    candidate's top-``intra_k`` cosines to that paper's texts; the candidate's
    score = mean of the top-``top_n`` papers. Candidate w/o abstract -> 0.0.
    """
    cands = list(candidates)
    papers = [[t for t in p if (t or "").strip()] for p in reference_papers]
    papers = [p for p in papers if p]
    if not cands:
        return []
    if not papers:
        return [
            ScreenResult(item=c, score=0.0, kept=False, reason="no_reference_papers")
            for c in cands
        ]

    # Flatten reference texts, remembering each paper's [start, end) slice.
    flat: list[str] = []
    bounds: list[tuple[int, int]] = []
    for p in papers:
        start = len(flat)
        flat.extend(p)
        bounds.append((start, len(flat)))

    # Candidates that have an abstract get embedded; others score 0.
    to_embed: list[str] = []
    embed_idx: dict[int, int] = {}
    for c in cands:
        if (c.get("abstract") or "").strip():
            embed_idx[id(c)] = len(to_embed)
            to_embed.append(_candidate_text(c))

    try:
        ref_vecs = await embedding_provider.embed(list(flat))
        cand_vecs = await embedding_provider.embed(to_embed) if to_embed else []
    except Exception as exc:
        return [
            ScreenResult(item=c, score=0.0, kept=False, reason=f"embedding_error: {exc}")
            for c in cands
        ]

    results: list[ScreenResult] = []
    for c in cands:
        if id(c) not in embed_idx:
            results.append(ScreenResult(item=c, score=0.0, kept=False, reason="no abstract"))
            continue
        cv = cand_vecs[embed_idx[id(c)]]
        per_paper = [
            _topn_mean([_cosine(cv, ref_vecs[j]) for j in range(s, e)], intra_k)
            for (s, e) in bounds
        ]
        score = _topn_mean(per_paper, top_n)
        results.append(
            ScreenResult(
                item=c, score=score, kept=score >= threshold,
                reason=f"emb_top{min(top_n, len(per_paper))}papers_mean={score:.3f}",
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "screen_papers_embedding",
        n=len(cands), papers=len(papers), top_n=top_n, intra_k=intra_k,
        kept=sum(r.kept for r in results), threshold=threshold,
    )
    return results


async def screen_papers_hybrid(
    candidates: Sequence[dict],
    *,
    reference_papers: Sequence[Sequence[str]],
    embedding_provider: Any,
    weights: tuple[float, float] = (0.25, 0.75),
    intra_k: int = 3,
    top_n: int = 5,
    threshold: float = 0.3,
) -> list[ScreenResult]:
    """Blend set-wise BM25 with abstract-to-abstract embedding, both two-level
    over the same ``reference_papers``. ``weights = (w_bm25, w_emb)``; default
    favours the semantic signal 3:1. Realignment by object identity.
    """
    cands = list(candidates)
    if not cands:
        return []
    w_bm25, w_emb = weights
    bm25_results = screen_papers_setwise_bm25(
        cands, reference_papers=reference_papers, intra_k=intra_k, top_n=top_n, threshold=0.0
    )
    emb_results = await screen_papers_embedding(
        cands, reference_papers=reference_papers, embedding_provider=embedding_provider,
        intra_k=intra_k, top_n=top_n, threshold=0.0,
    )
    bm25_by_id = {id(r.item): r.score for r in bm25_results}
    emb_by_id = {id(r.item): r.score for r in emb_results}

    results: list[ScreenResult] = []
    for c in cands:
        b = bm25_by_id.get(id(c), 0.0)
        e = emb_by_id.get(id(c), 0.0)
        score = w_bm25 * b + w_emb * e
        results.append(
            ScreenResult(
                item=c, score=score, kept=score >= threshold,
                reason=f"hybrid bm25={b:.3f} emb={e:.3f}",
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "screen_papers_hybrid",
        n=len(cands), top_n=top_n, intra_k=intra_k,
        kept=sum(r.kept for r in results), threshold=threshold, weights=list(weights),
    )
    return results


def select_calibration_samples(
    results: Sequence[ScreenResult], n: int = 4
) -> list[ScreenResult]:
    """Pick ``n`` samples spanning the score distribution, for human labeling.

    Targets ``n`` evenly-spaced points across the observed score range
    (high -> low) and picks the nearest not-yet-chosen result to each. Returns
    all results (sorted descending) when there are <= ``n`` of them.
    """
    items = sorted(results, key=lambda r: r.score, reverse=True)
    if len(items) <= n:
        return items

    lo, hi = items[-1].score, items[0].score
    if hi == lo:
        step = len(items) / n
        return [items[min(len(items) - 1, int(i * step))] for i in range(n)]

    # Evenly spaced fractions, high to low: for n=4 -> 0.875, 0.625, 0.375, 0.125.
    fractions = [1.0 - (i + 0.5) / n for i in range(n)]
    picked: list[ScreenResult] = []
    seen: set[int] = set()
    for f in fractions:
        target = lo + f * (hi - lo)
        best = min(
            (r for r in items if id(r) not in seen),
            key=lambda r: abs(r.score - target),
            default=None,
        )
        if best is not None:
            seen.add(id(best))
            picked.append(best)
    return picked


def cutoff_from_labels(
    labeled: Sequence[tuple[ScreenResult, bool]],
) -> float:
    """Return the score cutoff that best separates relevant (True) samples
    from not-relevant (False) ones.

    Tries every boundary (each sample score, plus just below the min and just
    above the max) and returns the one minimising misclassified samples -- a
    'relevant' that falls below the cutoff, or a 'not-relevant' kept at/above
    it. Ties break toward the HIGHER cutoff (more conservative -- keep fewer).
    Empty input returns 0.0 (keep everything).
    """
    labels = list(labeled)
    if not labels:
        return 0.0

    distinct = sorted({r.score for r, _ in labels})
    eps = 1e-6
    candidates = [distinct[0] - eps, *distinct, distinct[-1] + eps]

    best_cut = candidates[0]
    best_err: int | None = None
    for cut in candidates:  # ascending
        err = 0
        for r, is_relevant in labels:
            kept = r.score >= cut
            if kept != is_relevant:
                err += 1
        if best_err is None or err < best_err or (err == best_err and cut > best_cut):
            best_err = err
            best_cut = cut
    return float(max(0.0, best_cut))
