"""Extract typed indicium claims (5-slot SuperPattern + ECO-typed evidence)
from retrieved passages, via the project LLM client."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

# Matches ```...``` or ```json\n...\n``` wrappers that some LLMs add when
# asked to "return JSON" without strict JSON-mode. The leading/trailing
# whitespace and optional ``json`` language tag are both stripped.
_JSON_FENCE_RE: re.Pattern[str] = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL
)


def _strip_markdown_fence(raw: str) -> str:
    """Return ``raw`` with a surrounding ```` ``` ```` fence removed (if any).

    Pure string -> string. Returns ``raw`` unchanged when no fence is present.
    """
    if not isinstance(raw, str):
        return raw
    m = _JSON_FENCE_RE.match(raw)
    return m.group(1) if m else raw

_QUALIFIERS: frozenset[str] = frozenset({
    "causes", "prevents", "inhibits", "activates", "increases", "decreases",
    "correlates_with", "is_associated_with", "predicts", "interacts_with",
    "produces", "requires", "enables", "treats", "enhances", "reduces",
    "has_property", "is_part_of", "is_a", "consistent_with",
})
_EVIDENCE_TYPES = {"data", "citation", "knowledge", "inference", "speculation"}
_SOURCE_TYPES = {"text", "figure", "table", "image", "code", "data"}

_PROMPT = """You extract structured scientific claims from passages.
For each well-supported claim, output the 5-slot pattern:
context, subject, qualifier (one of: {qualifiers}), relation, object;
plus claim_type (explicit|implicit), evidence_type (one of: {evidence_types}),
source_type (one of: {source_types}), an exact quote, and source_doi.
Context: {context}
{domain_context}Passages:
{passages}
Return JSON: {{"claims": [{{...}}]}}"""


async def extract_claims(
    *, llm_client: Any, passages: list[dict], context: str | None = None,
    model: str | None = None,
    domain_adapter: Any | None = None,
) -> list[dict]:
    """Extract typed claims from passages via LLM.

    Args:
        llm_client: AsyncLLMClient instance.
        passages: List of passage dicts with ``chunk_text`` and ``source`` keys.
        context: Optional free-text context prepended to the prompt.
        model: Optional model override forwarded to ``llm_client.complete()``.
        domain_adapter: Optional DomainAdapter (indicium-adapters). When provided:
            - its qualifiers are merged into the valid qualifier set;
            - its extraction_context() is appended to the LLM prompt;
            - its enrich_claim() is called on every coerced claim.
            No import from indicium_adapters is required — structural typing only.
    """
    valid_qualifiers = (
        _QUALIFIERS | domain_adapter.qualifiers
        if domain_adapter is not None
        else _QUALIFIERS
    )
    domain_context = (
        domain_adapter.extraction_context() + "\n"
        if domain_adapter is not None
        else ""
    )
    rendered = "\n\n".join(
        f"[{i}] doi={p.get('source', {}).get('doi')}: {p.get('chunk_text', '')}"
        for i, p in enumerate(passages)
    )
    prompt = _PROMPT.format(
        qualifiers=", ".join(sorted(valid_qualifiers)),
        evidence_types=", ".join(sorted(_EVIDENCE_TYPES)),
        source_types=", ".join(sorted(_SOURCE_TYPES)),
        context=context or "(none)",
        domain_context=domain_context,
        passages=rendered,
    )
    messages = [{"role": "user", "content": prompt}]
    raw = await (llm_client.complete(messages=messages, model=model) if model
                 else llm_client.complete(messages=messages))
    # Some providers (e.g. OpenAI gpt-4o-mini, Anthropic Haiku/Sonnet when
    # not in strict JSON mode) wrap their response in a ```json … ``` fence
    # even when the prompt asks for raw JSON. Strip it before parsing so we
    # don't silently return zero claims.
    raw = _strip_markdown_fence(raw)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[dict] = []
    for raw_claim in data.get("claims", []):
        claim = _coerce_claim(raw_claim, valid_qualifiers)
        if claim is not None:
            out.append(claim)
    if domain_adapter is not None:
        out = [domain_adapter.enrich_claim(c) for c in out]
    return out


def _coerce_claim(c: dict, qualifiers: frozenset[str] = _QUALIFIERS) -> dict | None:
    required = ("context", "subject", "qualifier", "relation", "object")
    if not all(c.get(k) for k in required):
        return None
    if c["qualifier"] not in qualifiers:
        return None  # out-of-vocabulary qualifier => drop (don't void the run)
    ev_type = c.get("evidence_type")
    if ev_type is not None and ev_type not in _EVIDENCE_TYPES:
        ev_type = None
    src_type = c.get("source_type")
    if src_type is not None and src_type not in _SOURCE_TYPES:
        src_type = None
    # Mint a stable id so every claim dict is Indicium-adapter-compatible.
    # Callers may override by setting claim["id"] after coercion if they have
    # a stable identifier from the upstream source (e.g. a DOI-scoped hash).
    claim: dict = {"id": f"perspicacite:{uuid.uuid4().hex[:12]}"}
    claim.update({k: c[k] for k in required})
    if c.get("claim_type") in {"explicit", "implicit"}:
        claim["claim_type"] = c["claim_type"]
    evidence: dict = {}
    if c.get("source_doi"):
        evidence["doi"] = c["source_doi"]
    if c.get("quote"):
        evidence["quote"] = c["quote"]
    if ev_type:
        evidence["evidence_type"] = ev_type
    if src_type:
        evidence["source_type"] = src_type
    if evidence:
        claim["evidence"] = [evidence]
    return claim


_ASB = "https://asb.holobiomics.org/ns/asb#"


def claims_to_graph(claims: list[dict]):
    """Serialize claim dicts to an rdflib Graph using the asb: vocabulary that
    indicium's SHACL targets (a asb:Claim with asb:context/subject/...)."""
    import rdflib

    g = rdflib.Graph()
    asb = rdflib.Namespace(_ASB)
    for i, c in enumerate(claims):
        cid = c.get("id") or f"pos:{i}"
        node = rdflib.URIRef(f"urn:perspicacite:claim:{cid}")
        g.add((node, rdflib.RDF.type, asb.Claim))
        for slot in ("context", "subject", "qualifier", "relation", "object"):
            if c.get(slot):
                g.add((node, asb[slot], rdflib.Literal(c[slot])))
        for slot, curie in (c.get("ontology_terms") or {}).items():
            if curie:
                g.add((node, asb[f"{slot}_ontology_term"], rdflib.Literal(str(curie))))
    return g


def validate_claims(
    claims: list[dict],
    domain_adapter: Any | None = None,
) -> tuple[bool, str]:
    """SHACL-validate claims against indicium's shapes. Returns (conforms, report).

    Args:
        claims: List of claim dicts (as returned by extract_claims()).
        domain_adapter: Optional DomainAdapter. If it has a ``shacl_shapes()``
            method (i.e. satisfies SHACLProvider), its shapes are merged with
            indicium's base shapes before validation. No import from
            indicium_adapters is required — duck-typing via hasattr only.
    """
    import indicium

    g = claims_to_graph(claims)
    extra = (
        domain_adapter.shacl_shapes()
        if domain_adapter is not None and hasattr(domain_adapter, "shacl_shapes")
        else None
    )
    return indicium.validate_graph(g, extra_shapes=extra)
