"""LLM-backed structured extraction from passages.

Shared core behind ``extract_parameters_from_passages`` and
``extract_failure_modes_from_passages`` MCP tools.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from perspicacite.logging import get_logger

logger = get_logger(__name__)

# Tier A — verbatim safe.
_TIER_A_PATTERNS = (
    r"^cc0", r"^public.?domain", r"^cc.?by(?!-)", r"^mit", r"^apache",
    r"^bsd", r"^isc", r"^unlicense",
)
# Tier B — quote with caution (CC-BY-NC / -ND / -SA combinations).
_TIER_B_PATTERNS = (
    r"^cc.?by.?nc", r"^cc.?by.?nd", r"^cc.?by.?sa",
)
_TIER_B_VERBATIM_MAX_CHARS = 300
_BATCH_SIZE = 8


def classify_license_tier(license_id: str | None) -> str:
    """Return 'A', 'B', or 'C' for the given license_id."""
    norm = (license_id or "").strip().lower().replace(" ", "-")
    if not norm:
        return "C"
    for pat in _TIER_A_PATTERNS:
        if re.match(pat, norm):
            return "A"
    for pat in _TIER_B_PATTERNS:
        if re.match(pat, norm):
            return "B"
    return "C"


def handle_quote_for_license(
    text: str,
    *,
    license_id: str | None,
    paraphraser: Callable[[str], str] | None = None,
) -> str | None:
    """Apply Tier A/B/C policy to a quoted source string.

    Returns None when the policy says drop and no paraphraser is supplied.
    """
    tier = classify_license_tier(license_id)
    if tier == "A":
        return text
    if tier == "B":
        if len(text) <= _TIER_B_VERBATIM_MAX_CHARS:
            return text
        if paraphraser is None:
            return None
        return paraphraser(text)
    # Tier C
    if paraphraser is None:
        return None
    return paraphraser(text)


@dataclass(frozen=True)
class Passage:
    text: str
    source_doi: str
    license_id: str | None = None
    source_url: str | None = None


class _LLM(Protocol):
    async def complete(self, messages: list[dict], **kwargs: Any) -> str:
        ...


def _try_parse_json(raw: str) -> list[dict] | None:
    """Two-stage parse: direct, then trivial salvage (extract first [...] block).

    Strips control chars via the project-wide salvage util when available.
    """
    try:
        from perspicacite.rag.utils.json_salvage import clean_control_chars
        raw_clean = clean_control_chars(raw)
    except Exception:
        raw_clean = raw

    try:
        v = json.loads(raw_clean)
        return v if isinstance(v, list) else None
    except json.JSONDecodeError:
        pass
    # Salvage: take first [...] block
    m = re.search(r"\[.*\]", raw_clean, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else None
    except json.JSONDecodeError:
        return None


def _build_prompt(template: str, batch: list[Passage], context: str | None) -> str:
    lines = [template]
    if context:
        lines.append(f"Context: {context}")
    lines.append("Passages:")
    for i, p in enumerate(batch, 1):
        lines.append(f"[{i}] DOI={p.source_doi}\n{p.text}")
    lines.append(
        "Return JSON array. Each item must include the keys you were asked to "
        "extract, plus 'source_doi' (the [n] DOI it came from)."
    )
    return "\n\n".join(lines)


async def extract_structured(
    *,
    llm_client: _LLM,
    passages: list[Passage],
    prompt_template: str,
    schema: dict[str, Any],
    what: str,
    dedup_key: Callable[[dict], tuple],
    context: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Run LLM extraction across batches of passages with JSON salvage + dedup.

    Returns [] on total failure rather than raising — callers log and move on.
    """
    if not passages:
        return []

    seen: set[tuple] = set()
    out: list[dict] = []

    for start in range(0, len(passages), _BATCH_SIZE):
        batch = passages[start : start + _BATCH_SIZE]
        prompt = _build_prompt(prompt_template, batch, context)
        # timeout: HTTP-level cap per attempt (LiteLLM).
        # max_tokens=800: large enough for JSON extraction, small enough to stay fast.
        # The outer asyncio.timeout(80s) in the MCP tool caps the total time across all batches.
        kwargs: dict[str, Any] = {"temperature": 0.1, "max_tokens": 800, "timeout": 40}
        if model:
            kwargs["model"] = model

        try:
            raw = await asyncio.wait_for(
                llm_client.complete(
                    messages=[{"role": "user", "content": prompt}], **kwargs
                ),
                timeout=50.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "extraction_llm_call_timeout",
                what=what, batch_start=start, timeout_s=45.0,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "extraction_llm_call_failed",
                what=what, batch_start=start, error=str(exc),
            )
            continue

        records = _try_parse_json(raw)
        if not records:
            logger.warning(
                "extraction_json_parse_failed",
                what=what, batch_start=start, raw_preview=raw[:200],
            )
            continue

        for r in records:
            if not isinstance(r, dict):
                continue
            key = dedup_key(r)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)

    return out


def annotate_anchor_status(
    records: list[dict],
    passages: list[Passage],
    *,
    quote_key: str = "source_quote",
) -> list[dict]:
    """Tag each extracted record with anchor verification of its quote against
    the source passage texts. Mutates and returns `records`.

    Adds:
        record["anchor_status"]  -> "verified" | "repaired" | "unverified" | "unchecked"
        record["quote_exact"]    -> verbatim span (only when verified/repaired)

    Verifies BEFORE any license-tier rewrite of source_quote, so the check runs
    against the model's original quote. Degrades to "unchecked" (never raises)
    when the indicium verifier is unavailable.
    """
    if not records:
        return records
    try:
        from indicium import verify_quote
    except Exception:  # indicia extra absent
        for r in records:
            if isinstance(r, dict):
                r["anchor_status"] = "unchecked"
        logger.warning("anchor_verifier_unavailable_extraction")
        return records

    candidates = [p.text for p in passages]
    for r in records:
        if not isinstance(r, dict):
            continue
        quote = r.get(quote_key)
        if not quote:
            r["anchor_status"] = "unverified"
            continue
        try:
            res = verify_quote(str(quote), candidates)
        except Exception:
            r["anchor_status"] = "unchecked"
            continue
        r["anchor_status"] = res.status
        if res.status in ("verified", "repaired") and res.quote_exact:
            r["quote_exact"] = res.quote_exact
    return records
