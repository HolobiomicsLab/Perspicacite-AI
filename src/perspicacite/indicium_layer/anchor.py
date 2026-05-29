"""Anchor orchestration: verify each extracted claim's quote against the
passages it could have come from, tag the claim, and (optionally) emit an audit
sidecar. Project-side glue around indicium's verify_quote kernel.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from perspicacite.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def _claim_quote(claim: dict) -> str | None:
    ev = claim.get("evidence") or []
    if ev and isinstance(ev[0], dict):
        return ev[0].get("quote")
    return None


def _claim_doi(claim: dict) -> str | None:
    ev = claim.get("evidence") or []
    if ev and isinstance(ev[0], dict):
        return ev[0].get("doi")
    return None


def _emit_audit(audit_path: Path, records: list[dict]) -> None:
    """Append one JSONL record per claim. Fail-soft (never raises into caller)."""
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("anchor_audit_write_failed", path=str(audit_path), error=str(exc))


def anchor_claims(
    claims: list[dict],
    passages: list[dict],
    *,
    strict: bool = False,
    near_threshold: float = 0.9,
    audit_path: Path | None = None,
) -> list[dict]:
    """Verify each claim's quote against `passages` (index-aligned with the
    builder's passage list) and attach an `_anchor` record.

    Default (fail-open): keep every claim, tagged with its status.
    strict=True: drop claims whose status is "unverified".

    Each kept claim carries:
        claim["_anchor"] = {status, matched_index, quote_exact, score,
                            positional_index, divergent}
    where divergent = (matched_index is not None and matched_index != positional_index).
    Fail-soft: a verification error degrades that claim to "unverified". Modifies
    the input claim dicts in place (adds "_anchor") and returns the kept subset.
    If the verifier itself is unavailable (indicia extra absent), every claim is
    tagged "unverified" and returned regardless of `strict` — dropping everything
    when we cannot judge would be worse than keeping it tagged.
    """
    try:
        from indicium import verify_quote
    except Exception:  # indicia extra absent — graph path already gates on it
        logger.warning("anchor_verifier_unavailable")
        for i, claim in enumerate(claims):
            claim["_anchor"] = {
                "status": "unverified", "matched_index": None,
                "quote_exact": None, "score": 0.0,
                "positional_index": i, "divergent": False,
            }
        return claims

    candidates = [
        str(p.get("chunk_text", "")) if isinstance(p, dict) else "" for p in passages
    ]
    audit_records: list[dict] = []
    kept: list[dict] = []

    for i, claim in enumerate(claims):
        quote = _claim_quote(claim)
        try:
            if quote:
                res = verify_quote(quote, candidates, near_threshold=near_threshold)
                status, matched_index = res.status, res.matched_index
                quote_exact, score = res.quote_exact, res.score
            else:
                status, matched_index, quote_exact, score = "unverified", None, None, 0.0
        except Exception as exc:  # never let verification break a build
            logger.warning("anchor_verify_error", error=str(exc))
            status, matched_index, quote_exact, score = "unverified", None, None, 0.0

        divergent = matched_index is not None and matched_index != i
        claim["_anchor"] = {
            "status": status,
            "matched_index": matched_index,
            "quote_exact": quote_exact,
            "score": score,
            "positional_index": i,
            "divergent": divergent,
        }
        audit_records.append({
            "claim_id": claim.get("id"),
            "doi": _claim_doi(claim),
            "status": status,
            "score": round(score, 4),
            "matched_index": matched_index,
            "positional_index": i,
            "divergent": divergent,
        })

        if strict and status == "unverified":
            continue
        kept.append(claim)

    logger.info(
        "anchor_claims_done",
        total=len(claims), kept=len(kept), dropped=len(claims) - len(kept),
        strict=strict, divergent=sum(1 for r in audit_records if r["divergent"]),
    )
    if audit_path is not None:
        _emit_audit(audit_path, audit_records)
    return kept
