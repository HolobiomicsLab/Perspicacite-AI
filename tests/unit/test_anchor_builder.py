"""Integration of anchor binding into the claim-graph builder (R3)."""
from __future__ import annotations

import pytest

from perspicacite.indicium_layer.builder import build_claim_graph
from perspicacite.indicium_layer.store import ClaimGraphStore


class _FakeLLM:
    """Returns one claim whose quote is verbatim from passage index 2."""

    def __init__(self, payload: str):
        self._payload = payload

    async def complete(self, *, messages, model=None, **kwargs):
        return self._payload


_PAYLOAD = (
    '{"claims": [{"context": "in vitro", "subject": "compound A", '
    '"qualifier": "inhibits", "relation": "inhibits", "object": "enzyme B", '
    '"claim_type": "explicit", "evidence_type": "data", "source_type": "text", '
    '"quote": "compound A inhibits enzyme B", "source_doi": "10.1/x"}]}'
)


@pytest.mark.unit
async def test_builder_binds_quote_to_content_matched_passage(tmp_path, monkeypatch):
    # Redirect the on-disk manifest to tmp so the rebuild decision is hermetic
    # (otherwise a prior run's manifest marks the paper unchanged and it's skipped).
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    papers = {"10.1/x": {"paper_id": "10.1/x", "doi": "10.1/x", "title": "T"}}
    passages = {
        "10.1/x": [
            {"chunk_idx": 0, "text": "Unrelated weather passage.",
             "char_start": 0, "char_end": 25},
            {"chunk_idx": 1, "text": "Unrelated traffic passage.",
             "char_start": 26, "char_end": 51},
            {"chunk_idx": 2, "text": "We found that compound A inhibits enzyme B strongly.",
             "char_start": 52, "char_end": 103},
        ]
    }
    store = ClaimGraphStore("kbAnchor", backend="memory")
    try:
        await build_claim_graph(
            kb_name="kbAnchor",
            store=store,
            llm_client=_FakeLLM(_PAYLOAD),
            papers_provider=lambda: papers,
            passages_provider=lambda pid: passages.get(pid, []),
            max_pairs_per_claim=0,
            anchor_audit_dir=str(tmp_path),
        )
        # The verified quote_exact must be present on some Evidence node.
        rows = store.select(
            'SELECT ?q WHERE { ?e <https://asb.holobiomics.org/ns/asb#quoteExact> ?q }'
        )
        quotes = {r["q"] for r in rows}
        assert "compound A inhibits enzyme B" in quotes
        # anchorStatus verified is recorded.
        status_rows = store.select(
            'SELECT ?s WHERE { ?e <https://asb.holobiomics.org/ns/asb#anchorStatus> ?s }'
        )
        assert "verified" in {r["s"] for r in status_rows}
    finally:
        store.close()


_PAYLOAD_FABRICATED = (
    '{"claims": [{"context": "in vitro", "subject": "compound A", '
    '"qualifier": "inhibits", "relation": "inhibits", "object": "enzyme B", '
    '"claim_type": "explicit", "evidence_type": "data", "source_type": "text", '
    '"quote": "penguins migrate across antarctic ice during the polar winter", '
    '"source_doi": "10.1/x"}]}'
)


@pytest.mark.unit
async def test_builder_does_not_launder_unverified_quote(tmp_path, monkeypatch):
    # A fabricated quote (verbatim in NO passage) must be kept (fail-open) but
    # tagged unverified with NO quoteExact / oa:TextQuoteSelector written. This
    # pins the no-laundering invariant at the builder layer, where the
    # selector-writing logic actually lives.
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    papers = {"10.1/x": {"paper_id": "10.1/x", "doi": "10.1/x", "title": "T"}}
    passages = {
        "10.1/x": [
            {"chunk_idx": 0, "text": "We found that compound A inhibits enzyme B strongly.",
             "char_start": 0, "char_end": 52},
        ]
    }
    store = ClaimGraphStore("kbLaunder", backend="memory")
    try:
        await build_claim_graph(
            kb_name="kbLaunder",
            store=store,
            llm_client=_FakeLLM(_PAYLOAD_FABRICATED),
            papers_provider=lambda: papers,
            passages_provider=lambda pid: passages.get(pid, []),
            max_pairs_per_claim=0,
            anchor_audit_dir=str(tmp_path),
        )
        # No quoteExact may be laundered onto any Evidence node.
        quote_rows = store.select(
            'SELECT ?q WHERE { ?e <https://asb.holobiomics.org/ns/asb#quoteExact> ?q }'
        )
        assert quote_rows == []
        # The claim is still kept (fail-open) and tagged unverified.
        status_rows = store.select(
            'SELECT ?s WHERE { ?e <https://asb.holobiomics.org/ns/asb#anchorStatus> ?s }'
        )
        assert "unverified" in {r["s"] for r in status_rows}
    finally:
        store.close()
