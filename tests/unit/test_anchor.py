"""Unit tests for anchor_claims orchestration (R3)."""
from __future__ import annotations

import json

import pytest

from perspicacite.indicium_layer.anchor import anchor_claims


def _claim(cid: str, quote: str | None, doi: str = "10.1/x") -> dict:
    ev = {"doi": doi}
    if quote is not None:
        ev["quote"] = quote
    return {
        "id": cid,
        "context": "in vitro", "subject": "A", "qualifier": "inhibits",
        "relation": "inhibits", "object": "B",
        "evidence": [ev],
    }


@pytest.mark.unit
def test_positional_bug_regression_binds_to_content_match():
    # Quote is verbatim from passage index 2, but the claim is positionally at
    # output index 0. anchor_claims must bind it to passage 2, not passage 0.
    passages = [
        {"chunk_text": "Totally unrelated passage about the weather."},
        {"chunk_text": "Another unrelated passage about traffic."},
        {"chunk_text": "We found that compound A inhibits enzyme B strongly."},
    ]
    out = anchor_claims([_claim("c0", "compound A inhibits enzyme B")], passages)
    assert len(out) == 1
    anc = out[0]["_anchor"]
    assert anc["status"] == "verified"
    assert anc["matched_index"] == 2
    assert anc["positional_index"] == 0
    assert anc["divergent"] is True
    assert anc["quote_exact"] == "compound A inhibits enzyme B"


@pytest.mark.unit
def test_laundering_paraphrase_is_unverified():
    passages = [{"chunk_text": "The cat sat on the mat in the sun."}]
    out = anchor_claims([_claim("c0", "Felines rest upon textiles during daylight")], passages)
    assert out[0]["_anchor"]["status"] == "unverified"
    assert out[0]["_anchor"]["matched_index"] is None


@pytest.mark.unit
def test_strict_drops_unverified_failopen_keeps():
    passages = [{"chunk_text": "Real source text about A inhibits B clearly."}]
    good = _claim("good", "A inhibits B")
    bad = _claim("bad", "completely fabricated unrelated nonsense phrase")
    kept_open = anchor_claims([dict(good), dict(bad)], passages, strict=False)
    assert {c["id"] for c in kept_open} == {"good", "bad"}
    kept_strict = anchor_claims([dict(good), dict(bad)], passages, strict=True)
    assert {c["id"] for c in kept_strict} == {"good"}


@pytest.mark.unit
def test_audit_sidecar_written_with_divergent_flag(tmp_path):
    passages = [
        {"chunk_text": "unrelated A"},
        {"chunk_text": "the measured value was 42 units exactly"},
    ]
    audit = tmp_path / "anchor_audit.jsonl"
    anchor_claims([_claim("c0", "measured value was 42 units")], passages, audit_path=audit)
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["claim_id"] == "c0"
    assert rec["status"] == "verified"
    assert rec["matched_index"] == 1
    assert rec["positional_index"] == 0
    assert rec["divergent"] is True


@pytest.mark.unit
def test_missing_quote_is_unverified_kept_failopen():
    passages = [{"chunk_text": "source text"}]
    claim = {
        "id": "c0", "context": "c", "subject": "s", "qualifier": "inhibits",
        "relation": "r", "object": "o",
    }  # no evidence / no quote
    out = anchor_claims([claim], passages)
    assert len(out) == 1
    assert out[0]["_anchor"]["status"] == "unverified"
    assert out[0]["_anchor"]["matched_index"] is None


@pytest.mark.unit
def test_verifier_unavailable_degrades_all_to_unverified(monkeypatch):
    # Simulate the indicia extra being absent: `from indicium import verify_quote`
    # raises. Even though the quote IS verbatim in the passage and strict=True,
    # fail-soft must keep the claim, tagged unverified (we cannot judge).
    import sys

    monkeypatch.setitem(sys.modules, "indicium", None)
    passages = [{"chunk_text": "We found that compound A inhibits enzyme B strongly."}]
    out = anchor_claims(
        [_claim("c0", "compound A inhibits enzyme B")], passages, strict=True
    )
    assert len(out) == 1
    anc = out[0]["_anchor"]
    assert anc["status"] == "unverified"
    assert anc["matched_index"] is None
    assert anc["positional_index"] == 0
    assert anc["divergent"] is False
