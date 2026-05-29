"""Parser for ASB skill bundles.

Reads {run_dir}/skills/_index.json and walks each per-skill
directory, returning a list[ParsedSkill] for downstream conversion
to Papers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from perspicacite.pipeline.asb.models import (
    ParsedEnvironment,
    ParsedLink,
    ParsedPaperRef,
    ParsedParameter,
    ParsedSkill,
    ParsedTool,
)

logger = logging.getLogger(__name__)


def parse_skill_bundle(run_dir: Path | str) -> list[ParsedSkill]:
    """Walk an ASB run directory and return one ParsedSkill per
    entry in skills/_index.json. Missing sidecar files yield empty
    fields rather than errors."""
    run_dir = Path(run_dir)
    index_path = run_dir / "skills" / "_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"ASB skills index not found at {index_path}")
    index = json.loads(index_path.read_text())
    out: list[ParsedSkill] = []
    for entry in index.get("skills", []):
        # Both 'slug' and 'name' may serve as the directory name.
        slug = entry.get("slug") or entry.get("name")
        if not slug:
            logger.warning("asb_skill_index_entry_missing_slug: %s", entry)
            continue
        skill_dir = run_dir / "skills" / slug
        if not skill_dir.is_dir():
            logger.warning("asb_skill_missing: %s", slug)
            continue
        out.append(_parse_one_skill(skill_dir=skill_dir, index_entry=entry))
    return out


def _parse_one_skill(*, skill_dir: Path, index_entry: dict) -> ParsedSkill:
    slug = index_entry.get("slug") or index_entry.get("name")

    # 1. skill.md → frontmatter + body
    frontmatter, body = _split_frontmatter(skill_dir / "skill.md")

    # 2. JSON sidecars (all optional)
    tools_raw = _load_json(skill_dir / "tools.json", default={"tools": []})
    envs_raw = _load_json(skill_dir / "environments.json", default=[])
    params_raw = _load_json(skill_dir / "parameters.json", default=[])
    papers_raw = _load_json(skill_dir / "papers.json", default=[])
    links_raw = _load_json(skill_dir / "links.json", default=[])
    provenance_raw = _load_json(skill_dir / "artifact_provenance.json", default={})

    # Tolerate both list-of-dicts and {tools: [...]} shapes
    tools_list = (
        tools_raw.get("tools") if isinstance(tools_raw, dict) else tools_raw
    ) or []
    envs_list = envs_raw.get("environments") or [] if isinstance(envs_raw, dict) else envs_raw or []
    if isinstance(params_raw, dict):
        params_list = params_raw.get("parameters") or []
    else:
        params_list = params_raw or []
    if isinstance(papers_raw, dict):
        papers_list = papers_raw.get("papers") or []
    else:
        papers_list = papers_raw or []
    links_list = links_raw.get("links") or [] if isinstance(links_raw, dict) else links_raw or []

    return ParsedSkill(
        slug=slug or "",
        name=index_entry.get("name") or slug or "",
        description=index_entry.get("description") or frontmatter.get("description") or "",
        edam_operation=index_entry.get("edam_operation") or frontmatter.get("edam_operation"),
        edam_topics=frontmatter.get("edam_topics") or [],
        when_to_use_negative=frontmatter.get("when_to_use_negative") or [],
        schema_version=index_entry.get("schema_version") or frontmatter.get("schema_version"),
        body_markdown=body,
        tools=[ParsedTool(**t) for t in tools_list if isinstance(t, dict)],
        environments=[ParsedEnvironment(**e) for e in envs_list if isinstance(e, dict)],
        parameters=[ParsedParameter(**p) for p in params_list if isinstance(p, dict)],
        papers=[ParsedPaperRef(**p) for p in papers_list if isinstance(p, dict)],
        links=[ParsedLink(**lnk) for lnk in links_list if isinstance(lnk, dict)],
        asb_task_ids=_task_ids_from_provenance(provenance_raw, frontmatter),
        bundle_dir=str(skill_dir.relative_to(skill_dir.parent.parent)),
    )


def _split_frontmatter(skill_md: Path) -> tuple[dict, str]:
    if not skill_md.exists():
        return {}, ""
    text = skill_md.read_text()
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        logger.warning("asb_skill_frontmatter_unparseable: %s", str(skill_md))
        meta = {}
    body = parts[2].lstrip("\n")
    return meta if isinstance(meta, dict) else {}, body


def _load_json(path: Path, *, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("asb_sidecar_unparseable: %s", str(path))
        return default


def _task_ids_from_provenance(provenance: dict, frontmatter: dict) -> list[str]:
    ids: list[str] = []
    # frontmatter.provenance.source_task_ids
    fm_prov = (frontmatter or {}).get("provenance") or {}
    ids.extend(fm_prov.get("source_task_ids") or [])
    # artifact_provenance.json may also carry task ids
    if isinstance(provenance, dict):
        for k in ("source_task_ids", "task_ids"):
            ids.extend(provenance.get(k) or [])
    # dedup, preserve order
    seen, out = set(), []
    for t in ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
