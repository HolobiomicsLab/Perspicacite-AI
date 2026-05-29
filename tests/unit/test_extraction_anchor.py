"""Unit tests for annotate_anchor_status on the extraction path (R3)."""
from __future__ import annotations

import pytest

from perspicacite.pipeline.extraction import Passage, annotate_anchor_status


@pytest.mark.unit
def test_annotate_anchor_status_verified():
    passages = [Passage(text="The threshold was set to 0.85 for all runs.", source_doi="10.1/x")]
    records = [{
        "name": "threshold", "value": "0.85", "source_doi": "10.1/x",
        "source_quote": "threshold was set to 0.85",
    }]
    out = annotate_anchor_status(records, passages)
    assert out[0]["anchor_status"] == "verified"
    assert out[0]["quote_exact"] == "threshold was set to 0.85"


@pytest.mark.unit
def test_annotate_anchor_status_unverified_paraphrase():
    passages = [Passage(text="The cutoff used was eighty-five hundredths.", source_doi="10.1/x")]
    records = [{"name": "threshold", "source_quote": "we picked 0.85 arbitrarily"}]
    out = annotate_anchor_status(records, passages)
    assert out[0]["anchor_status"] == "unverified"
    assert "quote_exact" not in out[0]


@pytest.mark.unit
def test_annotate_anchor_status_missing_quote():
    passages = [Passage(text="source", source_doi="10.1/x")]
    out = annotate_anchor_status([{"name": "x"}], passages)
    assert out[0]["anchor_status"] == "unverified"
