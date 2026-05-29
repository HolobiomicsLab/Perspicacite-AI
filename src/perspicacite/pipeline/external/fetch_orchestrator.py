"""Cycle C — fetch_paper_resources orchestrator.

Reads a capsule's ``resources.json`` (produced by Cycle A V1 mining) and
dispatches to the right ``fetch_*`` helper for each resource kind. Emits
per-resource progress events via the JobRegistry. Optionally ingests
fetched text-like files into the KB tagging chunks with
``is_external=True, parent_paper_id=<paper.id>``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper
from perspicacite.pipeline.external.fetch_doi import (
    fetch_crossref,
    fetch_unpaywall,
)
from perspicacite.pipeline.external.fetch_github import fetch_github_docs
from perspicacite.pipeline.external.fetch_zenodo import fetch_zenodo

logger = get_logger("perspicacite.external.orchestrator")

# Resource kinds we know how to fetch on demand. Other kinds (e.g.
# data-archive accessions like ``pride``, ``geo_series``) are left as
# mining-only references for now.
SUPPORTED_KINDS = frozenset({"github", "zenodo", "doi"})


async def fetch_paper_resources(
    *,
    paper: Paper,
    capsule_dir: Path,
    kinds: list[str] | None,
    app_state,
    registry,
    job_id: str,
    ingest: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve ``resources.json`` → fetch by kind → (optional) ingest.

    Args:
        paper: Source paper (used for ``parent_paper_id`` on ingested chunks).
        capsule_dir: Capsule directory (must already exist).
        kinds: Filter list, e.g. ``["github", "zenodo"]``. ``None`` = all
            supported kinds present in resources.json.
        app_state: Provides ``config.external_resources``, embedding/vector
            stores (used when ``ingest=True``).
        registry: JobRegistry with ``publish/finish/fail`` async methods.
        job_id: Job identifier for progress events.
        ingest: When True, fetched text-like files are routed through
            ``ingest_local_documents`` with the ``is_external=True``,
            ``parent_paper_id=paper.id`` annotation.
        force: Currently informational; underlying helpers are cache-keyed
            so re-calls are idempotent. Reserved for future cache-busting.

    Returns:
        Dict summary, e.g.
        ``{"github": 1, "zenodo": 2, "doi": 0, "files_fetched": 12,
        "bytes_fetched": 45000, "ingested_chunks": 0}``.
    """
    cfg = app_state.config.external_resources
    if not cfg.fetch_on_demand:
        logger.info("external_fetch_disabled_by_config")
        result: dict[str, object] = {"disabled": True}
        await registry.finish(job_id, result)
        return result

    res_path = capsule_dir / "resources.json"
    if not res_path.is_file():
        await registry.fail(job_id, f"resources.json missing under {capsule_dir}")
        return {}
    try:
        resources = json.loads(res_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        await registry.fail(job_id, f"corrupt resources.json: {exc}")
        return {}

    selected_kinds = (
        {k for k in kinds if k in SUPPORTED_KINDS}
        if kinds is not None else SUPPORTED_KINDS
    )

    cache_dir = Path(cfg.cache_dir)
    ttl_seconds = int(cfg.cache_ttl_days) * 86400

    counts: dict[str, int] = {k: 0 for k in SUPPORTED_KINDS}
    total_files = 0
    total_bytes = 0
    fetched_paths: list[Path] = []

    for r in resources:
        kind = r.get("kind")
        identifier = r.get("identifier") or ""
        if kind not in selected_kinds:
            continue
        try:
            if kind == "github":
                # identifier is "owner/repo"
                if "/" not in identifier:
                    continue
                owner, repo = identifier.split("/", 1)
                summary = await fetch_github_docs(
                    owner, repo,
                    capsule_dir=capsule_dir,
                    cache_dir=cache_dir,
                    text_file_extensions=list(cfg.text_file_extensions),
                    ttl_seconds=ttl_seconds,
                )
                counts["github"] += 1
                total_files += int(summary.get("files_fetched", 0))
                total_bytes += int(summary.get("bytes_fetched", 0))
                base = capsule_dir / "external" / "github" / f"{owner}__{repo}"
                for sub in ("docs", "env", "notebooks", "scripts"):
                    if (base / sub).is_dir():
                        for p in (base / sub).rglob("*"):
                            if p.is_file():
                                fetched_paths.append(p)
                if (base / "README.md").is_file():
                    fetched_paths.append(base / "README.md")

            elif kind == "zenodo":
                summary = await fetch_zenodo(
                    identifier,
                    capsule_dir=capsule_dir,
                    cache_dir=cache_dir,
                    text_file_extensions=list(cfg.text_file_extensions),
                    max_bytes_per_file=int(cfg.zenodo_max_bytes_per_file),
                    max_bytes_per_record=int(cfg.zenodo_max_bytes_per_record),
                    metadata_only=True,
                    ttl_seconds=ttl_seconds,
                )
                counts["zenodo"] += 1
                total_files += int(summary.get("files_fetched", 0))
                total_bytes += int(summary.get("bytes_fetched", 0))

            elif kind == "doi":
                _ = await fetch_crossref(
                    identifier,
                    capsule_dir=capsule_dir,
                    cache_dir=cache_dir,
                    ttl_seconds=ttl_seconds,
                )
                _ = await fetch_unpaywall(
                    identifier,
                    capsule_dir=capsule_dir,
                    cache_dir=cache_dir,
                    ttl_seconds=ttl_seconds,
                )
                counts["doi"] += 1

            await registry.publish(job_id, {
                "type": "progress",
                "kind": kind,
                "identifier": identifier,
                "status": "fetched",
            })
        except Exception as exc:
            logger.warning(
                "external_fetch_resource_failed",
                kind=kind, identifier=identifier, error=str(exc),
            )
            await registry.publish(job_id, {
                "type": "progress",
                "kind": kind,
                "identifier": identifier,
                "status": "failed",
                "error": str(exc),
            })

    ingested_chunks = 0
    if ingest and fetched_paths:
        from perspicacite.integrations.local_docs import ingest_local_documents
        r2 = await ingest_local_documents(
            kb_name=getattr(paper, "kb_name", None) or _kb_name_from_app_state(app_state, paper),
            paths=fetched_paths,
            app_state=app_state,
            registry=registry,
            job_id=job_id,
            recursive=False,
            external_metadata={
                "parent_paper_id": paper.id,
            },
        )
        ingested_chunks = int(r2.get("added_chunks", 0))

    result = {
        **counts,
        "files_fetched": total_files,
        "bytes_fetched": total_bytes,
        "ingested_chunks": ingested_chunks,
    }
    await registry.finish(job_id, result)
    return result


def _kb_name_from_app_state(app_state, paper: Paper) -> str:
    """Best-effort KB lookup. Caller is expected to pass kb_name via
    ``paper.kb_name`` or attach it to ``app_state`` for orchestration paths.
    Falls back to an empty string so the downstream call fails fast and
    visibly rather than silently routing to the wrong KB."""
    return getattr(paper, "_kb_name", "") or ""
