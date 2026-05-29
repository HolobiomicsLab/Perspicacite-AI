"""Implementation helpers for the Perspicacité CLI commands.

These functions hold the business logic the ``@cli.command()`` wrappers in
[cli.py](cli.py) delegate to. They are re-imported into ``perspicacite.cli`` so
that ``perspicacite.cli.<name>`` stays a valid attribute (several unit tests
``monkeypatch.setattr`` these names there).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from perspicacite.pipeline.github_kb import IngestSummary


def _start_mcp_and_web(config, app) -> None:
    """Start MCP server and web server on a single port."""
    import asyncio
    from contextlib import asynccontextmanager

    import uvicorn

    # Initialize MCP state
    from perspicacite.mcp.server import mcp, mcp_state

    asyncio.run(mcp_state.initialize(config))

    # Get MCP ASGI app
    mcp_app = mcp.http_app()

    # Combine web app + MCP app lifespans
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(app_instance):
        async with original_lifespan(app_instance), mcp_app.lifespan(app_instance):
            yield

    app.router.lifespan_context = combined_lifespan

    # Mount MCP ASGI app — its internal routes are at /mcp
    app.mount("/", mcp_app)

    # Run single server
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


async def _create_kb_from_bibtex(
    config: Any,
    *,
    kb_name: str,
    bib_path: Path,
    description: str | None,
    session_db: Path,
    chroma_dir: Path,
) -> dict[str, Any]:
    from perspicacite.pipeline.bibtex_kb import create_kb_from_bibtex

    return await create_kb_from_bibtex(
        config,
        kb_name=kb_name,
        bib_path=bib_path,
        description=description,
        session_db=session_db,
        chroma_dir=chroma_dir,
    )


async def _add_bibtex_to_existing_kb(
    config: Any,
    *,
    kb_name: str,
    bib_path: Path,
    session_db: Path,
    chroma_dir: Path,
) -> dict[str, Any]:
    from perspicacite.pipeline.bibtex_kb import add_bibtex_to_existing_kb

    return await add_bibtex_to_existing_kb(
        config,
        kb_name=kb_name,
        bib_path=bib_path,
        session_db=session_db,
        chroma_dir=chroma_dir,
    )


async def _run_query(
    config: Any,
    query: str,
    kb: str,
    mode: str,
    provider: str,
    model: str | None,
) -> None:
    """Run a RAG query and print the answer + sources to stdout."""
    from perspicacite.models.rag import RAGMode, RAGRequest
    from perspicacite.web.state import AppState

    state = AppState()
    await state.initialize()
    assert state.session_store is not None

    # Verify the KB exists so we fail fast with a clear message instead
    # of letting the RAG engine spit a chroma error.
    if await state.session_store.get_kb_metadata(kb) is None:
        click.echo(f"\nError: KB '{kb}' not found. List with: perspicacite list-kb", err=True)
        sys.exit(1)

    mode_map = {
        "basic": RAGMode.BASIC,
        "advanced": RAGMode.ADVANCED,
        "deep_research": RAGMode.DEEP_RESEARCH,
        "profound": RAGMode.PROFOUND,  # backward-compat alias
        "contradiction": RAGMode.CONTRADICTION,
    }
    rag_mode = mode_map.get(mode, RAGMode.BASIC)

    # Effective model/provider: explicit flag → config default → dataclass default.
    eff_provider = provider or config.llm.default_provider
    eff_model = model or config.llm.default_model

    request = RAGRequest(
        query=query,
        kb_name=kb,
        mode=rag_mode,
        provider=eff_provider,
        model=eff_model,
    )

    # Use the same RAGEngine the web/MCP layers use.
    full_answer_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    try:
        async for event in state.rag_engine.query_stream(request):
            etype = getattr(event, "event", None)
            data = getattr(event, "data", None)
            if etype == "content" and data:
                # data is a JSON envelope { "delta": "..." }
                try:
                    import json as _json
                    delta = _json.loads(data).get("delta", "")
                except Exception:
                    delta = str(data)
                if delta:
                    full_answer_parts.append(delta)
            elif etype == "source" and data:
                try:
                    import json as _json
                    s = _json.loads(data)
                    sources.append(s)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            elif etype == "error" and data:
                click.echo(f"\n❌ Error from RAG engine: {data}", err=True)
                sys.exit(1)
    except Exception as exc:
        click.echo(f"\n❌ Query failed: {exc}", err=True)
        sys.exit(1)

    answer = "".join(full_answer_parts).strip()
    click.echo("\n📝 Answer:")
    if not answer:
        click.echo("  (no answer — KB might be empty for this query)")
    else:
        click.echo(answer)
    if sources:
        click.echo("\n📎 Sources:")
        for i, s in enumerate(sources, 1):
            title = s.get("title") or s.get("doi") or "(untitled)"
            year = s.get("year")
            doi = s.get("doi")
            tag = f" ({year})" if year else ""
            doi_tag = f"  doi:{doi}" if doi else ""
            click.echo(f"  [{i}] {title}{tag}{doi_tag}")


async def _build_app_state_for_cli(config: Any) -> Any:
    """Test seam: thin wrapper so unit tests can patch this without
    constructing the full AppState."""
    from perspicacite.web.state import AppState
    state = AppState()
    await state.initialize()
    return state


def _print_github_repo_summary(summary: IngestSummary) -> None:
    """Human-readable summary line for the raw-repo path."""
    coords = ""
    if summary.repo_org and summary.repo_name:
        coords = f" ({summary.repo_org}/{summary.repo_name}"
        if summary.commit_sha:
            coords += f"@{summary.commit_sha}"
        coords += ")"
    click.echo(f"GitHub repo ingested into KB: {summary.kb_name}{coords}")
    click.echo(f"  files:  {summary.files_added}")
    click.echo(f"  chunks: {summary.chunks_added}")


def _print_skill_bundle_summary(summary: IngestSummary) -> None:
    """Human-readable summary line for a single bundle ingest."""
    suffix = ""
    if summary.bundle_name:
        suffix = f" (bundle: {summary.bundle_name})"
    click.echo(f"Skill bundle ingested into KB: {summary.kb_name}{suffix}")
    click.echo(f"  files:         {summary.files_added}")
    click.echo(f"  chunks:        {summary.chunks_added}")
    click.echo(f"  linked papers: {summary.linked_papers_added}")
    if summary.linked_papers_skipped_non_doi:
        kinds = ", ".join(
            f"{kind}={value}"
            for kind, value in summary.linked_papers_skipped_non_doi[:5]
        )
        click.echo(
            f"  skipped (non-DOI): {len(summary.linked_papers_skipped_non_doi)} ({kinds})"
        )
