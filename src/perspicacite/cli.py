"""Command-line interface for Perspicacité v2."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from perspicacite import __version__
from perspicacite.config import load_config
from perspicacite.logging import get_logger, setup_logging

logger = get_logger("perspicacite.cli")


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.pass_context
def cli(ctx: click.Context, config: Path | None, verbose: bool) -> None:
    """Perspicacité v2 - AI-powered scientific literature research assistant."""
    # Ensure context dict exists
    ctx.ensure_object(dict)

    # Load config
    try:
        cfg = load_config(str(config) if config else None)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    # Override log level if verbose
    if verbose:
        cfg.logging.level = "DEBUG"

    # Setup logging
    setup_logging(cfg.logging)
    logger.info("perspicacite_started", version=__version__)

    # Store config in context
    ctx.obj["config"] = cfg


@cli.command()
@click.option(
    "--host",
    default=None,
    help="Server host (default: from config)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Server port (default: from config)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload (development)",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    help="Disable MCP server",
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Headless mode (API only)",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    reload: bool,
    no_mcp: bool,
    no_ui: bool,
) -> None:
    """Start the Perspicacité server."""
    config = ctx.obj["config"]

    # Override with CLI args
    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if reload:
        config.server.reload = True
    if no_mcp:
        config.mcp.enabled = False

    logger.info(
        "starting_server",
        host=config.server.host,
        port=config.server.port,
        mcp_enabled=config.mcp.enabled,
    )

    click.echo(f"🚀 Starting Perspicacité v{__version__}")
    click.echo(f"   Server: http://{config.server.host}:{config.server.port}")
    if config.mcp.enabled:
        click.echo(f"   MCP: http://{config.mcp.host}:{config.mcp.port}")

    # Import and run server from web_app_full
    import uvicorn
    
    # Dynamically import the web app module
    import sys
    web_app_path = Path(__file__).parent.parent.parent / "web_app_full.py"
    
    # Add the repo root to path temporarily
    repo_root = web_app_path.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Import using importlib to handle the module loading
    import importlib.util
    spec = importlib.util.spec_from_file_location("web_app_full", web_app_path)
    web_module = importlib.util.module_from_spec(spec)
    
    # Set up the app state with config
    import os
    os.environ["PERSPICACITE_CONFIG"] = str(config)
    
    spec.loader.exec_module(web_module)
    app = web_module.app

    # TODO: Start MCP server alongside if enabled
    # For now, just run FastAPI
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


@cli.command()
@click.argument("name")
@click.option(
    "--description",
    "-d",
    help="Knowledge base description",
)
@click.option(
    "--from-bibtex",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    help="Create from BibTeX file (.bib)",
)
@click.option(
    "--session-db",
    type=click.Path(path_type=Path),
    default=None,
    help="SQLite DB for KB metadata (default: data/perspicacite.db, same as web app)",
)
@click.option(
    "--chroma-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Chroma persist directory (default: from config database.chroma_path)",
)
@click.pass_context
def create_kb(
    ctx: click.Context,
    name: str,
    description: str | None,
    from_bibtex: Path | None,
    session_db: Path | None,
    chroma_dir: Path | None,
) -> None:
    """Create a new knowledge base (from BibTeX when --from-bibtex is set)."""
    config = ctx.obj["config"]

    if not from_bibtex:
        click.echo(
            "Creating an empty KB from the CLI is not implemented yet.\n"
            "Use: perspicacite create-kb NAME --from-bibtex path/to/file.bib",
            err=True,
        )
        sys.exit(1)

    session_db = session_db or Path("data/perspicacite.db")
    chroma_dir = chroma_dir or config.database.chroma_path

    click.echo(f"Creating knowledge base '{name}' from {from_bibtex}...")
    click.echo(f"  Session DB: {session_db}")
    click.echo(f"  Chroma: {chroma_dir}")
    click.echo(f"  Embedding model (from config): {config.knowledge_base.embedding_model}")

    try:
        result = asyncio.run(
            _create_kb_from_bibtex(
                config=config,
                kb_name=name,
                bib_path=from_bibtex,
                description=description,
                session_db=session_db,
                chroma_dir=chroma_dir,
            )
        )
    except FileExistsError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("create_kb_failed", error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo("\n✅ Knowledge base created")
    click.echo(f"   Name (sanitized): {result['name']}")
    click.echo(f"   Chroma collection: {result['collection_name']}")
    click.echo(f"   Papers: {result['papers']}, chunks: {result['chunks_added']}")
    st = result["pdf_stats"]
    click.echo(
        f"   PDF download: attempted={st['attempted']} success={st['success']} "
        f"failed={st['failed']} no_doi={st['skipped_no_doi']}"
    )


@cli.command()
@click.pass_context
def list_kb(ctx: click.Context) -> None:
    """List all knowledge bases."""
    click.echo("📚 Knowledge Bases:")
    click.echo("  (not implemented yet)")


@cli.command()
@click.argument("query")
@click.option(
    "--kb",
    "-k",
    default="default",
    help="Knowledge base to query",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["quick", "standard", "advanced", "deep", "citation"]),
    default="standard",
    help="RAG mode",
)
@click.option(
    "--provider",
    "-p",
    default="anthropic",
    help="LLM provider",
)
@click.option(
    "--model",
    default=None,
    help="LLM model",
)
@click.pass_context
def query(
    ctx: click.Context,
    query: str,
    kb: str,
    mode: str,
    provider: str,
    model: str | None,
) -> None:
    """Query a knowledge base."""
    config = ctx.obj["config"]

    click.echo(f"🔍 Querying '{kb}' with {mode} mode...")
    click.echo(f"   Query: {query}")
    click.echo(f"   Provider: {provider}")

    # Run async query
    asyncio.run(_run_query(config, query, kb, mode, provider, model))


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


async def _run_query(
    config: Any,
    query: str,
    kb: str,
    mode: str,
    provider: str,
    model: str | None,
) -> None:
    """Run query (placeholder)."""
    click.echo("\n📝 Answer:")
    click.echo("  (RAG not implemented yet)")


@cli.command()
def version() -> None:
    """Print version information."""
    click.echo(f"Perspicacité v{__version__}")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
