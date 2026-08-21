"""Minimal, dependency-free projection of claims onto ASTRA records.

Vendored on purpose. The richer projection lives in ``indicium``, which is
private and unpublishable, so relying on it made the only ASTRA feature in this
repo unusable for everyone outside the lab. The shapes here follow ASTRA 0.0.12:

* ``Insight``  — ``claim``, ``created_at`` and ``evidence`` are all required.
* ``Evidence`` — exactly one of ``doi`` or ``artifact``, plus an optional
  ``quote`` selector.
* ``Analysis`` — the root record; its ``narrative.findings`` prose is required
  whenever structured findings are present.

A claim that cannot be grounded in a DOI or an artifact cannot become a
schema-valid Insight, so it is reported as skipped rather than emitted with an
empty evidence list.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

ASTRA_SCHEMA_VERSION = "0.0.12"
_ID_SAFE = re.compile(r"[^a-z0-9_]+")


def _slug(value: str, fallback: str) -> str:
    """ASTRA ids are lowercase snake_case."""
    out = _ID_SAFE.sub("_", str(value).strip().lower()).strip("_")
    return out or fallback


def claim_string(claim: dict[str, Any]) -> str:
    """Flatten a 5-slot SuperPattern claim into one sentence.

    Mirrors indicium's projection: ``subject relation object (context)``.
    Missing slots are dropped rather than rendered as ``None``.
    """
    subject = str(claim.get("subject") or "").strip()
    relation = str(claim.get("relation") or "").strip()
    obj = str(claim.get("object") or "").strip()
    context = str(claim.get("context") or "").strip()

    core = " ".join(part for part in (subject, relation, obj) if part)
    if not core:
        core = str(claim.get("text") or claim.get("claim") or "").strip()
    return f"{core} ({context})" if context and core else core


def _evidence_nodes(claim: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence entries that satisfy ASTRA's doi-xor-artifact rule."""
    nodes: list[dict[str, Any]] = []
    for item in claim.get("evidence") or []:
        node: dict[str, Any] = {}
        if item.get("doi"):
            node["doi"] = item["doi"]
        elif item.get("artifact"):
            node["artifact"] = item["artifact"]
        else:
            continue
        quote = item.get("quote")
        if quote:
            node["quote"] = {"exact": str(quote)}
        nodes.append(node)
    return nodes


def claim_to_insight(
    claim: dict[str, Any], *, index: int = 0, created_at: str | None = None
) -> dict[str, Any] | None:
    """Project one claim onto an ASTRA Insight, or None if it cannot be grounded."""
    evidence = _evidence_nodes(claim)
    if not evidence:
        return None
    text = claim_string(claim)
    if not text:
        return None
    return {
        "id": _slug(claim.get("id") or f"c{index}", f"c{index}"),
        "claim": text,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "evidence": evidence,
    }


def build_analysis(
    claims: list[dict[str, Any]],
    *,
    analysis_id: str = "perspicacite_claim_export",
    name: str = "Perspicacité claim export",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Wrap projected claims in a root ASTRA Analysis.

    Returns the Analysis plus a ``skipped`` list naming every claim that had no
    DOI or artifact to stand on, so an empty export is never mistaken for a
    clean one.
    """
    stamp = created_at or datetime.now(UTC).isoformat()
    findings: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []

    for index, claim in enumerate(claims):
        insight = claim_to_insight(claim, index=index, created_at=stamp)
        if insight is None:
            skipped.append({
                "id": str(claim.get("id") or f"c{index}"),
                "reason": "no DOI or artifact to ground the claim",
            })
            continue
        findings[insight["id"]] = insight

    analysis: dict[str, Any] = {
        "id": _slug(analysis_id, "perspicacite_claim_export"),
        "name": name,
        "schema_version": ASTRA_SCHEMA_VERSION,
        "narrative": {
            "summary": (
                f"{len(findings)} claim(s) exported from a Perspicacité knowledge base "
                "as ASTRA Insights."
            ),
        },
        "findings": findings,
    }
    if findings:
        # ASTRA requires findings prose whenever structured findings exist.
        analysis["narrative"]["findings"] = (
            "Each finding is one extracted claim, grounded in the DOI or artifact "
            "it was read from. Claims that could not be grounded were omitted."
        )
    return {"analysis": analysis, "skipped": skipped}
