#!/usr/bin/env python3
"""Extract paper-only chunks from a mixed asb-skills-* KB into a fresh
asb-paper-* KB. Reuses the already-computed embeddings — no re-embed cost.

This is the migration step for the lifecycle clean-up:
  - asb-skills-{doi}  ← mixed (paper + skills + workflows + asb_traces)
  - asb-paper-{doi}   ← paper + SI only

Both KBs share the same paper_id (DOI-derived or sha256-derived since
2026-05-26 commit 042433c), so re-attaching them later is trivial. The
paper-only KB is what Agent 6, the Phase 2 validator, and the eval
bench should query going forward — structurally impossible to leak
ASB-emitted artifacts into grounding.

The migration is read-only on the source: it never modifies the
asb-skills-* KB. Idempotent on the target: re-running adds 0 new
chunks if the paper_ids are already present.

Usage:
    uv run python scripts/migrate_to_paper_only_kb.py \\
        --src asb-skills-pesticide-v8d \\
        --dst asb-paper-pesticide

    # Or run on multiple KBs:
    uv run python scripts/migrate_to_paper_only_kb.py \\
        --src asb-skills-haffner-v8 --dst asb-paper-haffner \\
        --src asb-skills-jeong-v8   --dst asb-paper-jeong
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import chromadb


CHROMA_DB_PATH = REPO_ROOT / "chroma_db"
MCP_URL = "http://127.0.0.1:8002/mcp"  # default; override via --mcp-url


def kb_collection_name(kb: str) -> str:
    """Mirror perspicacite.models.kb.chroma_collection_name_for_kb."""
    return f"kb_{kb}"


async def _ensure_kb_registered(dst_kb: str, mcp_url: str) -> None:
    """Register the dst KB in Perspicacité's session_store via the
    running MCP server's create_knowledge_base tool.

    Chroma collection creation alone isn't enough — the SQLite
    metadata must also have a row, or MCP-side KB lookups
    (get_relevant_passages, bench judge) return "KB not found".

    Idempotent: list_knowledge_bases first; skip if dst_kb is there.
    """
    try:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp.client.session import ClientSession
    except ImportError:
        return  # MCP SDK absent; user can manually register

    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Check whether the KB is already registered
            listed = await session.call_tool("list_knowledge_bases", {})
            listed_text = listed.content[0].text if listed.content else "{}"
            try:
                listed_data = __import__("json").loads(listed_text)
            except Exception:
                listed_data = {}
            existing = {
                k.get("name")
                for k in (listed_data.get("knowledge_bases") or [])
            }
            if dst_kb in existing:
                return
            # Create the KB metadata. The MCP create_knowledge_base
            # tool also creates a fresh Chroma collection, but since
            # ours already exists Chroma's get_or_create returns
            # the existing one — no data loss.
            await session.call_tool("create_knowledge_base", {
                "name": dst_kb,
                "description": f"Paper-only KB (migrated) for {dst_kb}",
            })


def migrate_one(src_kb: str, dst_kb: str) -> dict:
    """Copy source_type='paper' chunks from src to dst KB.

    Reuses embeddings — no re-computation.
    Idempotent: chunks whose `id` is already in dst are skipped.

    Side effect: ensures dst is registered in Perspicacité's
    session_store metadata so MCP tools can find it.

    Returns:
        {src, dst, n_paper_chunks_src, n_existing_dst, n_added,
         papers_migrated: list[paper_id]}
    """
    # Register the KB metadata first (so MCP-side lookups work)
    try:
        asyncio.run(_ensure_kb_registered(dst_kb, mcp_url=MCP_URL))
    except Exception as exc:
        print(f"WARN: could not register {dst_kb!r} in session_store: {exc}",
              file=sys.stderr)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    src_coll_name = kb_collection_name(src_kb)
    dst_coll_name = kb_collection_name(dst_kb)
    try:
        src_coll = client.get_collection(src_coll_name)
    except Exception as exc:
        raise RuntimeError(f"source KB not found: {src_coll_name}") from exc

    # Pull all paper chunks from source
    src_res = src_coll.get(
        where={"source_type": "paper"},
        include=["documents", "metadatas", "embeddings"],
    )
    n_paper_chunks_src = len(src_res.get("ids") or [])
    if n_paper_chunks_src == 0:
        return {
            "src": src_kb, "dst": dst_kb,
            "n_paper_chunks_src": 0, "n_existing_dst": 0, "n_added": 0,
            "papers_migrated": [],
            "note": "no paper chunks in source",
        }

    # Ensure destination collection exists (chroma creates on get_or_create)
    dst_coll = client.get_or_create_collection(dst_coll_name)

    # Check which ids already exist in dst (idempotence)
    src_ids = src_res["ids"]
    try:
        existing = dst_coll.get(ids=src_ids, include=[])
        existing_ids = set(existing.get("ids") or [])
    except Exception:
        existing_ids = set()

    # Build add lists for ids NOT already present. Use explicit
    # `is None` / index checks since chroma returns numpy arrays for
    # embeddings (truthy-boolean on ndarray raises).
    src_docs = src_res.get("documents")
    src_metas = src_res.get("metadatas")
    src_embs = src_res.get("embeddings")
    if src_docs is None or src_metas is None or src_embs is None:
        raise RuntimeError(
            "source KB missing documents/metadatas/embeddings"
        )
    to_add_ids: list[str] = []
    to_add_docs: list[str] = []
    to_add_metas: list[dict] = []
    to_add_embs: list[list[float]] = []
    paper_ids_seen: set[str] = set()
    for i, cid in enumerate(src_ids):
        if cid in existing_ids:
            continue
        to_add_ids.append(cid)
        to_add_docs.append(src_docs[i] or "")
        meta = src_metas[i] or {}
        to_add_metas.append(dict(meta))
        emb = src_embs[i]
        # Chroma returns numpy arrays; convert to plain list for re-add
        to_add_embs.append(list(emb))
        pid = meta.get("paper_id")
        if pid:
            paper_ids_seen.add(pid)

    if not to_add_ids:
        return {
            "src": src_kb, "dst": dst_kb,
            "n_paper_chunks_src": n_paper_chunks_src,
            "n_existing_dst": len(existing_ids),
            "n_added": 0,
            "papers_migrated": [],
            "note": "all paper chunks already in dst",
        }

    # Batch the add (chroma handles ~5000 at a time well)
    BATCH = 500
    for i in range(0, len(to_add_ids), BATCH):
        dst_coll.add(
            ids=to_add_ids[i:i + BATCH],
            documents=to_add_docs[i:i + BATCH],
            metadatas=to_add_metas[i:i + BATCH],
            embeddings=to_add_embs[i:i + BATCH],
        )

    return {
        "src": src_kb, "dst": dst_kb,
        "n_paper_chunks_src": n_paper_chunks_src,
        "n_existing_dst": len(existing_ids),
        "n_added": len(to_add_ids),
        "papers_migrated": sorted(paper_ids_seen),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", action="append", required=True,
                   help="Source KB name (repeatable; pair with --dst)")
    p.add_argument("--dst", action="append", required=True,
                   help="Destination KB name (repeatable; same order as --src)")
    args = p.parse_args()
    if len(args.src) != len(args.dst):
        print("--src and --dst must be paired", file=sys.stderr)
        return 1

    print(f"{'src':35s} {'dst':35s} {'src_n':>6s} {'added':>6s} {'papers':>7s}")
    for src, dst in zip(args.src, args.dst):
        try:
            r = migrate_one(src, dst)
        except Exception as exc:
            print(f"{src:35s} {dst:35s} ERR: {exc}")
            continue
        n_papers = len(r.get("papers_migrated") or [])
        print(
            f"{src:35s} {dst:35s} "
            f"{r['n_paper_chunks_src']:>6d} {r['n_added']:>6d} "
            f"{n_papers:>7d}"
        )
        if r.get("note"):
            print(f"   note: {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
