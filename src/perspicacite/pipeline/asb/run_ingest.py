"""Top-level ASB-run ingestion.

Steps:
  1. Parse skills (skill_parser.parse_skill_bundle)
  2. Parse workflow cards (card_parser.parse_cards)
  3. Load workflow DAG (dag.load_workflow_dag)
  4. Get/create the KB (composite, or per-skill when mode='per-skill')
  5. For each ParsedSkill: build a Paper + add to KB; ingest backing-paper
     DOIs via existing ingest_dois_into_kb
  6. For each ParsedCard: build a Paper (with DAG neighbors) + add
  7. Store workflow_dag.json contents on the KB description
  8. Write skill_kb.json entries per skill

The two seam helpers — `_make_or_get_kb` and `_ingest_backing_paper_dois`
— are module-level so tests can patch them. In production, the orchestrator
is invoked from the MCP tool / CLI handler with a populated `app_state`;
both helpers consume `app_state` from there.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perspicacite.pipeline.asb.card_parser import parse_cards
from perspicacite.pipeline.asb.chunk_producer import (
    _should_redact,
    card_to_paper,
    skill_to_paper,
)
from perspicacite.pipeline.asb.dag import WorkflowDag, load_workflow_dag
from perspicacite.pipeline.asb.models import ParsedCard, ParsedSkill
from perspicacite.pipeline.asb.skill_kb_writer import write_skill_kb_entries
from perspicacite.pipeline.asb.skill_parser import parse_skill_bundle

logger = logging.getLogger(__name__)

# OA-equivalent access-tier values per corpus.yaml schema. Anything
# NOT in this set (including missing/empty/"unknown"/"closed"/"hybrid")
# is treated as non-OA → Evidence section redacted at ingest time.
# See audit item N7 in
# asb_performance_eval/bench/closed_access_audit_2026_05_27.md.
_OA_ACCESS_TYPES = frozenset({
    "open-access", "open_access", "oa",
    "gold-oa", "diamond", "green",
})


def _load_non_oa_dois(run_dir: Path) -> tuple[set[str] | None, dict[str, Any]]:
    """Parse corpus.yaml in the run-dir and return the redaction map.

    Returns:
        (non_oa_dois, info) where
          - ``non_oa_dois`` is a set of lower-cased DOIs that should
            have their Evidence section redacted on ingest. None means
            "no gate active" (operator opt-in via
            ``allow_non_oa_ingest=True``).
          - ``info`` is a small dict of telemetry for the result payload
            and structured logging.

    Fail-safe semantics:
      - If ``corpus.yaml`` is absent → return an empty set (gate active;
        any record with no source DOI is redacted; records WITH a DOI
        are NOT in the non-OA set, so they're allowed through. This
        matches the "no information" case — see N7 spec for why the
        chunk_producer also treats DOI-less records as redact-by-default
        when the gate is active).
      - If PyYAML import fails → log warning, return an empty set with
        ``corpus_yaml_parsed=False`` and the chunk_producer treats every
        DOI as unknown (redact all). This is the "we cannot prove OA
        for anything" fail-safe.
      - If corpus.yaml parses but contains no papers → return empty set.
    """
    corpus_path = run_dir / "corpus.yaml"
    info: dict[str, Any] = {
        "corpus_yaml_present": corpus_path.is_file(),
        "corpus_yaml_parsed": False,
        "known_dois": 0,
        "non_oa_dois": 0,
    }
    if not corpus_path.is_file():
        return set(), info
    try:
        import yaml  # local import — PyYAML is already a runtime dep
    except ImportError:
        logger.warning(
            "asb_corpus_yaml_pyyaml_missing extra=%s",
            {"run_dir": str(run_dir)},
        )
        # Fail-safe: treat *every* DOI as unknown by returning a
        # sentinel set containing a marker. The chunk_producer's
        # _should_redact logic redacts records whose DOI is NOT in the
        # known-OA set, which we can't compute without yaml. Easiest
        # fail-safe is to redact unconditionally by passing a set
        # containing a marker AND returning known_dois=0 so the
        # chunk_producer treats every record as "unknown".
        # We achieve unconditional redaction by returning a set with a
        # token that won't match real DOIs but causes _should_redact
        # to operate (non-None) and treats all unknown DOIs as redact.
        return {"__pyyaml_missing__"}, info
    try:
        corpus = yaml.safe_load(corpus_path.read_text()) or {}
    except Exception as e:  # malformed yaml
        logger.warning(
            "asb_corpus_yaml_parse_failed extra=%s",
            {"run_dir": str(run_dir), "error": str(e)},
        )
        return {"__corpus_yaml_unparseable__"}, info
    info["corpus_yaml_parsed"] = True

    papers = corpus.get("papers") or []
    non_oa: set[str] = set()
    known: set[str] = set()
    for p in papers:
        if not isinstance(p, dict):
            continue
        doi = (p.get("doi") or "").lower().strip()
        if not doi:
            continue
        known.add(doi)
        access = (p.get("access") or {}) if isinstance(p.get("access"), dict) else {}
        atype = (access.get("type") or "").lower().strip()
        if atype not in _OA_ACCESS_TYPES:
            non_oa.add(doi)
    info["known_dois"] = len(known)
    info["non_oa_dois"] = len(non_oa)
    # Stash the OA-known DOIs so chunk_producer can distinguish
    # "known-OA → allow" from "unknown → redact (fail-safe)". We encode
    # this by including in non_oa_dois every DOI that is NOT known-OA.
    # Records with a DOI that is in known-OA pass through unredacted;
    # records with any other DOI (or no DOI) are redacted.
    # Practically: non_oa_dois set returned here is the explicit non-OA
    # set; the chunk_producer additionally redacts DOI-less records
    # whenever the gate is active. Records carrying an unknown DOI
    # (not listed in corpus.yaml at all) are NOT yet redacted by the
    # existing _should_redact — we extend the contract by adding those
    # unknown DOIs to non_oa here. We can't know them up-front, but
    # the caller (ingest_asb_run) will compute "all skill/card DOIs
    # minus known" and union them in before passing to chunk_producer.
    return non_oa, info


async def ingest_asb_run(
    *,
    asb_run_dir: str | Path,
    kb_name: str | None = None,
    include: Iterable[str] = ("skills", "workflows"),
    mode: str = "composite",
    update_skill_kb_json: bool = True,
    app_state: Any = None,
    allow_non_oa_ingest: bool = False,
) -> dict[str, Any]:
    """Ingest an ASB run directory into one or more Perspicacité KBs.

    Args:
        asb_run_dir: Path to the run dir (must contain skills/_index.json
            and/or cards/).
        kb_name: KB to write to. Defaults to the run-dir name.
        include: Subset of {"skills", "workflows"}.
        mode: "composite" (single KB) or "per-skill" (one KB per skill;
            workflows still go to the composite KB).
        update_skill_kb_json: Write back the integration seam.
        app_state: Application state object (for KB creation + DOI ingest).
            May be None in tests when both seam helpers are patched.
        allow_non_oa_ingest: When True, BYPASS the access-tier gate and
            ingest verbatim skill/card bodies regardless of corpus.yaml
            access tier. Default False (safe). Use only when the operator
            has independently confirmed that the run-dir's content is
            permissively licensed. Logs a clear warning when enabled.
            See audit item N7 for context.

    Returns:
        Dict with:
          - kb_names: list of KBs created/updated
          - skills_ingested: int
          - workflows_ingested: int
          - papers_ingested: int (total Paper objects added)
          - failed: list of (paper_id, error)
          - workflow_dag: nodes/edges as dict (or None when DAG empty)
          - access_gate: dict with corpus.yaml telemetry and the number
            of records redacted (``redacted_n``).
    """
    include_set = set(include)
    if not include_set:
        raise ValueError("include must contain at least one of {skills, workflows}")
    if mode not in ("composite", "per-skill"):
        raise ValueError("mode must be 'composite' or 'per-skill'")

    run_dir = Path(asb_run_dir)
    skills = parse_skill_bundle(run_dir) if "skills" in include_set else []
    cards = parse_cards(run_dir) if "workflows" in include_set else []
    dag = load_workflow_dag(run_dir)

    # ---- Access-tier gate (audit item N7) ------------------------------
    # Determine which records' Evidence sections must be redacted at
    # ingest time. ingest_asb_run reads corpus.yaml directly; sanitization
    # at release time (promote.py) runs LATER + OUTSIDE the run-dir, so
    # raw run-dir ingest WITHOUT this gate would leak closed-access
    # verbatim into the KB.
    access_gate: dict[str, Any] = {
        "active": not allow_non_oa_ingest,
        "redacted_n": 0,
        "allow_non_oa_ingest": allow_non_oa_ingest,
    }
    if allow_non_oa_ingest:
        logger.warning(
            "ingest_asb_run_allow_non_oa_ingest_enabled extra=%s",
            {
                "run_dir": str(run_dir),
                "note": "non-OA verbatim WILL be ingested",
            },
        )
        non_oa_dois: set[str] | None = None
    else:
        non_oa_dois, gate_info = _load_non_oa_dois(run_dir)
        access_gate.update(gate_info)

        # Collect every skill/card source DOI seen in the run-dir so
        # we can implement the "unknown DOI = fail-safe redact" rule.
        seen_dois: set[str] = set()
        for s in skills:
            for p in s.papers:
                if p.doi:
                    seen_dois.add(p.doi.lower().strip())
        for c in cards:
            if c.crossref_doi:
                seen_dois.add(c.crossref_doi.lower().strip())

        # Sentinel cases (corpus.yaml missing / PyYAML missing /
        # unparseable corpus.yaml) → fail-safe = redact ALL records.
        # Add every seen DOI to the redaction set so chunk_producer
        # treats them all as non-OA. DOI-less records are already
        # handled by _should_redact's missing-DOI branch.
        sentinel_present = any(
            d.startswith("__") and d.endswith("__")
            for d in (non_oa_dois or set())
        )
        if sentinel_present or not gate_info.get("corpus_yaml_present"):
            non_oa_dois = (non_oa_dois or set()) | seen_dois
            access_gate["fail_safe_redact_all"] = True

        # Extend the redaction set with any skill/card source DOIs that
        # do NOT appear in corpus.yaml at all — fail-safe per N7 spec.
        # Known-OA set = (papers in corpus.yaml with OA access type).
        # non_oa = explicitly-non-OA. Anything in seen_dois that is
        # NEITHER explicitly-non-OA NOR known-OA is "unknown" → redact.
        if gate_info.get("corpus_yaml_parsed"):
            corpus_path = run_dir / "corpus.yaml"
            try:
                import yaml
                corpus = yaml.safe_load(corpus_path.read_text()) or {}
            except Exception:
                corpus = {}
            known_dois = {
                ((p or {}).get("doi") or "").lower().strip()
                for p in (corpus.get("papers") or [])
                if isinstance(p, dict)
            }
            known_dois.discard("")
            unknown = seen_dois - known_dois
            non_oa_dois = (non_oa_dois or set()) | unknown

    composite_name = kb_name or run_dir.name
    kb_names: list[str] = []
    papers_ingested = 0
    failed: list[dict] = []

    composite_kb = await _make_or_get_kb(
        composite_name,
        description=_kb_description(skills=skills, cards=cards, dag=dag),
        app_state=app_state,
    )
    kb_names.append(composite_name)

    # ---- Skills ---------------------------------------------------------
    for skill in skills:
        target_kb = composite_kb
        target_kb_name = composite_name
        if mode == "per-skill":
            per_skill_name = f"{composite_name}__{skill.slug}"
            target_kb = await _make_or_get_kb(
                per_skill_name,
                description=_kb_description(skills=[skill], cards=[], dag=dag),
                app_state=app_state,
            )
            target_kb_name = per_skill_name
            kb_names.append(per_skill_name)

        # 1. Skill body
        skill_entries: list[dict] = []
        try:
            paper = skill_to_paper(skill, non_oa_dois=non_oa_dois)
            if _should_redact(
                [p.doi for p in skill.papers if p.doi], non_oa_dois
            ):
                access_gate["redacted_n"] += 1
            await target_kb.add_papers([paper])
            papers_ingested += 1
            skill_entries.append({
                "kind": "skill_body",
                "source_url": f"skills/{skill.slug}/skill.md",
                "kb_name": target_kb_name,
                "chunk_ids": [],
                "chunk_count": 0,
                "bytes": len((skill.body_markdown or "").encode("utf-8")),
                "content_type": "text",
                "embedding_model": "text-embedding-3-small",
                "ingested_at": _now_iso(),
            })
        except Exception as e:
            logger.warning("asb_skill_add_failed", extra={
                "slug": skill.slug, "error": str(e)
            })
            failed.append({"id": f"asb_skill:{skill.slug}", "error": str(e)})

        # 2. Backing-paper DOIs (existing literature ingest path)
        skill_dois = [p.doi for p in skill.papers if p.doi]
        if skill_dois:
            try:
                await _ingest_backing_paper_dois(
                    kb=target_kb, dois=skill_dois, app_state=app_state,
                )
            except Exception as e:
                logger.warning("asb_skill_backing_dois_failed", extra={
                    "slug": skill.slug, "error": str(e)
                })

        # 3. skill_kb.json round-trip
        if update_skill_kb_json and skill_entries:
            skill_kb_path = run_dir / "skills" / skill.slug / "skill_kb.json"
            if skill_kb_path.exists():
                try:
                    write_skill_kb_entries(skill_kb_path, entries=skill_entries)
                except Exception as e:
                    logger.warning("asb_skill_kb_write_failed", extra={
                        "slug": skill.slug, "error": str(e)
                    })

    # ---- Workflow cards (always composite KB) ---------------------------
    for card in cards:
        try:
            paper = card_to_paper(card, dag=dag, non_oa_dois=non_oa_dois)
            if _should_redact(
                [card.crossref_doi] if card.crossref_doi else [], non_oa_dois
            ):
                access_gate["redacted_n"] += 1
            await composite_kb.add_papers([paper])
            papers_ingested += 1
        except Exception as e:
            logger.warning("asb_card_add_failed", extra={
                "task_id": card.task_id, "error": str(e)
            })
            failed.append({"id": f"asb_card:{card.task_id}", "error": str(e)})

    if access_gate["redacted_n"]:
        logger.info(
            "ingest_asb_run_redacted_n=%d extra=%s",
            access_gate["redacted_n"],
            {
                "run_dir": str(run_dir),
                "kb": composite_name,
                "corpus_yaml_present": access_gate.get("corpus_yaml_present"),
                "corpus_yaml_parsed": access_gate.get("corpus_yaml_parsed"),
            },
        )

    return {
        "kb_names": kb_names,
        "skills_ingested": len(skills),
        "workflows_ingested": len(cards),
        "papers_ingested": papers_ingested,
        "failed": failed,
        "workflow_dag": dag.to_dict() if dag.nodes else None,
        "access_gate": access_gate,
    }


def _kb_description(
    *,
    skills: list[ParsedSkill],
    cards: list[ParsedCard],
    dag: WorkflowDag,
) -> str:
    """Serialise bundle-level metadata onto the KB description.

    Stored as a JSON blob inside a descriptive prefix so the
    auto-KB-routing layer can extract it.
    """
    payload = {
        "skills": [s.slug for s in skills],
        "task_ids": [c.task_id for c in cards],
        "workflow_dag": dag.to_dict() if dag.nodes else None,
    }
    return (
        "ASB bundle ingest. Includes skills + workflow cards + "
        f"DAG metadata. {json.dumps(payload)}"
    )


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _make_or_get_kb(name: str, *, description: str = "", app_state: Any = None):
    """Get or create a KB by name. Production: routes through app_state's
    session_store + DynamicKnowledgeBase factory. Patched in tests.

    NOTE for D8/D9 wiring: the production helper in search_to_kb.py is
    named ``_create_kb_if_missing(app_state, kb_name, description)`` and
    returns ``(kb_meta, created)`` — it does NOT return a DynamicKnowledgeBase
    directly. The production path below sketches the two-step dance needed:
    first call ``_create_kb_if_missing``, then construct a DynamicKnowledgeBase
    from the returned metadata + app_state stores.
    """
    if app_state is None:
        raise RuntimeError(
            "_make_or_get_kb requires app_state in production. "
            "Tests should patch this function."
        )
    # Production path: re-use search_to_kb's _create_kb_if_missing pattern.
    from perspicacite.pipeline.search_to_kb import _create_kb_if_missing
    _kb_meta, _created = await _create_kb_if_missing(
        app_state, name, description,
    )
    # Construct the in-memory DynamicKnowledgeBase backed by that metadata.
    from perspicacite.models.kb import chroma_collection_name_for_kb
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

    config = KnowledgeBaseConfig(
        collection_prefix=chroma_collection_name_for_kb(name) + "_",
    )
    kb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        # Both AppState and MCPState expose `embedding_provider`; there is no
        # `embedding_service` attribute. Reading it broke ingest_asb_run on
        # every path ("'MCPState' object has no attribute 'embedding_service'").
        embedding_service=app_state.embedding_provider,
        config=config,
    )
    # Pin the collection to the KB's CANONICAL name, exactly as the read path
    # does (search_knowledge_base sets `dkb.collection_name =
    # chroma_collection_name_for_kb(name)`). Without this, DKB.__init__ leaves
    # collection_name = "<prefix><session_id>", so ingest wrote to a
    # session-suffixed collection while search/stats read the canonical one —
    # ingested skills were invisible (chunk_count=0 despite a "success" ingest).
    kb.collection_name = chroma_collection_name_for_kb(name)
    kb.name = name
    kb.description = description
    return kb


async def _ingest_backing_paper_dois(
    *, kb: Any, dois: list[str], app_state: Any,
) -> dict[str, Any]:
    """Run the existing DOI ingest path. Patched in tests."""
    if app_state is None:
        raise RuntimeError(
            "_ingest_backing_paper_dois requires app_state in production. "
            "Tests should patch this function."
        )
    from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb
    return await ingest_dois_into_kb(
        app_state, kb_name=getattr(kb, "name", "asb_bundle"), dois=dois,
    )
