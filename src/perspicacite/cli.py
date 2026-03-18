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

    # Import and run server
    from perspicacite.api.app import create_app
    import uvicorn

    app = create_app(config)

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
    help="Create from BibTeX file",
)
@click.option(
    "--embedding-model",
    "-e",
    default="text-embedding-3-small",
    help="Embedding model to use",
)
@click.pass_context
def create_kb(
    ctx: click.Context,
    name: str,
    description: str | None,
    from_bibtex: Path | None,
    embedding_model: str,
) -> None:
    """Create a new knowledge base."""
    config = ctx.obj["config"]

    click.echo(f"Creating knowledge base '{name}'...")

    if from_bibtex:
        click.echo(f"  From BibTeX: {from_bibtex}")

    click.echo(f"  Embedding model: {embedding_model}")
    click.echo("\n✅ Knowledge base created (not implemented yet)")


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
