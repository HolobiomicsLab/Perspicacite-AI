"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from perspicacite.config.schema import Config
from perspicacite.logging import get_logger
from perspicacite.api.routes import chat, knowledge_bases, sessions, health

logger = get_logger("perspicacite.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("api_startup")
    yield
    # Shutdown
    logger.info("api_shutdown")


def create_app(config: Config) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Application configuration

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Perspicacité v2 API",
        description="AI-powered scientific literature research assistant",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router, prefix="/api")
    app.include_router(knowledge_bases.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(health.router)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Perspicacité v2",
            "version": "2.0.0",
            "docs": "/docs",
        }

    return app
