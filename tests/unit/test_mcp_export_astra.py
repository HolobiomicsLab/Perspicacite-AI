"""export_astra must work without the private indicium package and emit a
schema-valid ASTRA Analysis.

ASTRA 0.0.12 requires ``claim``, ``created_at`` and ``evidence`` on every
Insight, and exactly one of ``doi``/``artifact`` on every Evidence node.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from perspicacite.mcp import server as mcp_server

_GROUNDED_CLAIM = {
    "id": "c1",
    "context": "in vitro",
    "subject": "A",
    "qualifier": "inhibits",
    "relation": "inhibits",
    "object": "B",
    "evidence": [{"doi": "10.1/x", "quote": "A inhibits B"}],
}


async def _export(claims):
    state = MagicMock()
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.export_astra(claims=claims)
    return json.loads(raw)


@pytest.mark.unit
async def test_export_astra_emits_a_valid_analysis():
    payload = await _export([_GROUNDED_CLAIM])

    assert payload["success"] is True
    analysis = payload["analysis"]
    assert analysis["id"] and analysis["narrative"]["summary"]
    # Findings prose is required whenever structured findings exist.
    assert analysis["narrative"]["findings"]

    insight = analysis["findings"]["c1"]
    assert insight["claim"] == "A inhibits B (in vitro)"
    assert insight["created_at"]
    assert insight["evidence"][0]["doi"] == "10.1/x"
    assert insight["evidence"][0]["quote"]["exact"] == "A inhibits B"


@pytest.mark.unit
async def test_export_astra_needs_no_private_package():
    """The tool used to hard-fail for anyone without the private indicium."""
    import sys

    blocked = {name: None for name in list(sys.modules) if name.startswith("indicium")}
    blocked["indicium"] = None
    with patch.dict(sys.modules, blocked):
        payload = await _export([_GROUNDED_CLAIM])

    assert payload["success"] is True
    assert payload["analysis"]["findings"]["c1"]["evidence"]


@pytest.mark.unit
async def test_ungrounded_claims_are_reported_not_silently_dropped():
    ungrounded = {**_GROUNDED_CLAIM, "id": "c2", "evidence": []}
    payload = await _export([_GROUNDED_CLAIM, ungrounded])

    assert set(payload["analysis"]["findings"]) == {"c1"}
    assert payload["skipped"] == [
        {"id": "c2", "reason": "no DOI or artifact to ground the claim"}
    ]


@pytest.mark.unit
async def test_evidence_without_doi_or_artifact_is_never_emitted():
    """ASTRA requires exactly one of doi/artifact; a bare quote is not evidence."""
    quote_only = {**_GROUNDED_CLAIM, "id": "c3", "evidence": [{"quote": "no anchor"}]}
    payload = await _export([quote_only])

    assert payload["analysis"]["findings"] == {}
    assert payload["skipped"][0]["id"] == "c3"


@pytest.mark.unit
async def test_artifact_backed_evidence_is_accepted():
    artifact_claim = {
        **_GROUNDED_CLAIM,
        "id": "c4",
        "evidence": [{"artifact": "run_20260524.json"}],
    }
    payload = await _export([artifact_claim])

    assert payload["analysis"]["findings"]["c4"]["evidence"][0]["artifact"] == (
        "run_20260524.json"
    )


@pytest.mark.unit
async def test_empty_export_is_still_a_valid_analysis():
    payload = await _export([])

    analysis = payload["analysis"]
    assert analysis["findings"] == {}
    # No structured findings, so no findings prose is required.
    assert "findings" not in analysis["narrative"]
    assert analysis["narrative"]["summary"]
