"""Access-tier gate on ``ingest_asb_run`` — audit item N7.

``ingest_asb_run`` must NOT ingest verbatim Evidence-section content
from skills/cards whose source paper is non-OA (or whose access tier
is unknown). The release-time sanitization pass in
``AgenticScienceBuilder/release/promote.py`` runs OUTSIDE the run-dir
and AFTER the build, so a Perspicacité KB built directly from a fresh
run-dir would otherwise leak closed-access verbatim spans into Chroma.

These tests use the same in-memory mock-KB seam that the existing
integration tests in ``tests/integration/test_asb_run_ingest_end_to_end.py``
use, so KB / DOI ingest side effects are stubbed out.

Audit doc: ``asb_performance_eval/bench/closed_access_audit_2026_05_27.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_NON_OA_DOI = "10.9999/closed.paper.example"
_UNKNOWN_DOI = "10.9999/unknown.paper.example"

_SKILL_MD_WITH_EVIDENCE = """---
description: A test skill for the access-tier gate.
edam_operation: http://edamontology.org/operation_9999
---

## Overview

Test skill body for the access-tier gate test.

## When to Use

Whenever testing the gate.

## Procedure

1. Build the skill.
2. Ingest it.
3. Assert verbatim is gone.

## Evidence

- [section] paraphrase: "VERBATIM_QUOTE_FROM_NON_OA_PAPER_MUST_NOT_LEAK"
- [section] paraphrase: "ANOTHER_VERBATIM_QUOTE_FROM_NON_OA_PAPER"

## References

- DOI: """ + _NON_OA_DOI + "\n"


def _write_run_dir(
    root: Path,
    *,
    slug: str,
    source_doi: str,
    corpus_papers: list[dict] | None,
) -> Path:
    """Construct a minimal run-dir with one skill + (optional) corpus.yaml.

    Mirrors the layout the skill_parser expects:
      run/
        skills/
          _index.json
          {slug}/
            skill.md
            papers.json
        corpus.yaml  (optional)
    """
    run_dir = root / "run"
    skills_dir = run_dir / "skills" / slug
    skills_dir.mkdir(parents=True)

    (run_dir / "skills" / "_index.json").write_text(json.dumps({
        "skills": [{
            "slug": slug,
            "name": slug,
            "description": "test skill",
            "schema_version": "0.2.0",
            "body_path": f"skills/{slug}/skill.md",
        }]
    }))
    (skills_dir / "skill.md").write_text(_SKILL_MD_WITH_EVIDENCE)
    (skills_dir / "papers.json").write_text(json.dumps([
        {"doi": source_doi, "title": "Closed paper",
         "year": 2025, "role": "method"}
    ]))

    if corpus_papers is not None:
        import yaml
        (run_dir / "corpus.yaml").write_text(
            yaml.safe_dump({"papers": corpus_papers})
        )
    return run_dir


def _stub_kb(added: list):
    fake_kb = MagicMock()
    fake_kb.name = "test_kb"
    fake_kb.description = ""

    async def fake_add_papers(papers, **kw):
        added.extend(papers)
        return len(papers)

    fake_kb.add_papers = fake_add_papers
    return fake_kb


@pytest.mark.asyncio
async def test_non_oa_paper_evidence_is_redacted(tmp_path):
    """A skill whose source DOI is flagged ``access.type=closed`` in
    corpus.yaml must have its ``## Evidence`` section stripped from
    ``Paper.full_text`` before ingest into the KB.
    """
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    run_dir = _write_run_dir(
        tmp_path,
        slug="closed-skill",
        source_doi=_NON_OA_DOI,
        corpus_papers=[{
            "doi": _NON_OA_DOI,
            "status": "included",
            "access": {"type": "closed"},
        }],
    )

    added: list = []
    fake_kb = _stub_kb(added)

    async def fake_make_or_get_kb(name, description="", **kw):
        fake_kb.name = name
        fake_kb.description = description
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(run_dir),
            kb_name="test_kb",
            include=("skills",),
            mode="composite",
            app_state=None,
        )

    assert result["skills_ingested"] == 1
    assert len(added) == 1
    body = added[0].full_text or ""
    # Verbatim quote must be stripped.
    assert "VERBATIM_QUOTE_FROM_NON_OA_PAPER_MUST_NOT_LEAK" not in body
    assert "ANOTHER_VERBATIM_QUOTE_FROM_NON_OA_PAPER" not in body
    # Redaction marker must be present.
    assert "REDACTED" in body
    # Non-Evidence sections preserved.
    assert "## Overview" in body
    assert "## Procedure" in body
    # Telemetry surfaces the redaction.
    assert result["access_gate"]["redacted_n"] == 1
    assert result["access_gate"]["active"] is True


@pytest.mark.asyncio
async def test_allow_non_oa_ingest_bypasses_redaction(tmp_path):
    """``allow_non_oa_ingest=True`` is an explicit operator opt-in:
    even a closed-access skill keeps its verbatim Evidence section.
    A warning is logged (not asserted here — log capture would be brittle).
    """
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    run_dir = _write_run_dir(
        tmp_path,
        slug="closed-skill-bypass",
        source_doi=_NON_OA_DOI,
        corpus_papers=[{
            "doi": _NON_OA_DOI,
            "status": "included",
            "access": {"type": "closed"},
        }],
    )

    added: list = []
    fake_kb = _stub_kb(added)

    async def fake_make_or_get_kb(name, description="", **kw):
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(run_dir),
            kb_name="test_kb",
            include=("skills",),
            mode="composite",
            app_state=None,
            allow_non_oa_ingest=True,
        )

    assert result["skills_ingested"] == 1
    body = added[0].full_text or ""
    assert "VERBATIM_QUOTE_FROM_NON_OA_PAPER_MUST_NOT_LEAK" in body
    assert "REDACTED" not in body
    assert result["access_gate"]["redacted_n"] == 0
    assert result["access_gate"]["active"] is False


@pytest.mark.asyncio
async def test_paper_absent_from_corpus_yaml_is_redacted_fail_safe(tmp_path):
    """A skill whose source DOI is NOT listed in corpus.yaml at all
    (unknown access tier) must be redacted — fail-safe per N7 spec.
    """
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    # corpus.yaml lists a DIFFERENT, OA paper. The skill's source DOI
    # is unknown to the corpus.
    other_doi = "10.9999/some-other-oa-paper"
    run_dir = _write_run_dir(
        tmp_path,
        slug="unknown-doi-skill",
        source_doi=_UNKNOWN_DOI,
        corpus_papers=[{
            "doi": other_doi,
            "status": "included",
            "access": {"type": "open-access"},
        }],
    )

    added: list = []
    fake_kb = _stub_kb(added)

    async def fake_make_or_get_kb(name, description="", **kw):
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(run_dir),
            kb_name="test_kb",
            include=("skills",),
            mode="composite",
            app_state=None,
        )

    body = added[0].full_text or ""
    # Unknown DOI → fail-safe → redacted.
    assert "VERBATIM_QUOTE_FROM_NON_OA_PAPER_MUST_NOT_LEAK" not in body
    assert "REDACTED" in body
    assert result["access_gate"]["redacted_n"] == 1


@pytest.mark.asyncio
async def test_known_oa_paper_evidence_is_preserved(tmp_path):
    """Sanity check: a skill whose source DOI is explicitly OA in
    corpus.yaml keeps its full Evidence section. The gate is not
    over-broad.
    """
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    oa_doi = "10.9999/oa.paper.example"
    run_dir = _write_run_dir(
        tmp_path,
        slug="oa-skill",
        source_doi=oa_doi,
        corpus_papers=[{
            "doi": oa_doi,
            "status": "included",
            "access": {"type": "open-access"},
        }],
    )

    added: list = []
    fake_kb = _stub_kb(added)

    async def fake_make_or_get_kb(name, description="", **kw):
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(run_dir),
            kb_name="test_kb",
            include=("skills",),
            mode="composite",
            app_state=None,
        )

    body = added[0].full_text or ""
    # OA → Evidence preserved verbatim.
    assert "VERBATIM_QUOTE_FROM_NON_OA_PAPER_MUST_NOT_LEAK" in body
    assert "REDACTED" not in body
    assert result["access_gate"]["redacted_n"] == 0


@pytest.mark.asyncio
async def test_missing_corpus_yaml_triggers_fail_safe_redaction(tmp_path):
    """When corpus.yaml is absent from the run-dir, the operator has
    not declared access tiers → fail-safe is to redact everything.
    """
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    run_dir = _write_run_dir(
        tmp_path,
        slug="no-corpus-skill",
        source_doi=_NON_OA_DOI,
        corpus_papers=None,  # NO corpus.yaml at all
    )

    added: list = []
    fake_kb = _stub_kb(added)

    async def fake_make_or_get_kb(name, description="", **kw):
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(run_dir),
            kb_name="test_kb",
            include=("skills",),
            mode="composite",
            app_state=None,
        )

    body = added[0].full_text or ""
    assert "VERBATIM_QUOTE_FROM_NON_OA_PAPER_MUST_NOT_LEAK" not in body
    assert "REDACTED" in body
    assert result["access_gate"]["redacted_n"] == 1
    assert result["access_gate"].get("fail_safe_redact_all") is True
