"""FastAPI application — instantiation, lifespan, root + favicon, router mounts.

Logging is configured at the top of this module before any structlog-using
imports (e.g. perspicacite.memory.session_store via perspicacite.logging),
because basicConfig is a no-op once handlers are attached.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# ── Logging setup MUST run before any imports that may trigger structlog ──
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"web_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_log_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_file_handler = logging.FileHandler(_log_file)
_file_handler.setFormatter(_log_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.addHandler(_file_handler)
_root.addHandler(_stream_handler)

# Now safe to import modules that use structlog.
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from perspicacite.llm.embeddings import EmbeddingFailedError
from perspicacite.web.routers import (
    chat as chat_router,
)
from perspicacite.web.routers import (
    conversations as conversations_router,
)
from perspicacite.web.routers import (
    health as health_router,
)
from perspicacite.web.routers import (
    jobs as jobs_router,
)
from perspicacite.web.routers import (
    kb as kb_router,
)
from perspicacite.web.routers import (
    llm_proxy as llm_proxy_router,
)
from perspicacite.web.routers import (
    pdf_dropzone as pdf_dropzone_router,
)
from perspicacite.web.routers import (
    survey as survey_router,
)
from perspicacite.web.routers import (
    zotero as zotero_router,
)
from perspicacite.web.routers import (
    zotero_ingest as zotero_ingest_router,
)
from perspicacite.web.state import app_state

logger = logging.getLogger("perspicacite.web")


# Resolve repo-root paths once at import time.
# This file lives at: <repo>/src/perspicacite/web/app.py
# parents[3] from __file__ is <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATE_PATH = _REPO_ROOT / "templates" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await app_state.initialize()
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Perspicacité v2 - True Agentic RAG", lifespan=lifespan)


# Mount static assets (CSS/JS extracted from templates/index.html)
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Disable browser caching of frontend assets and the index template. Without
# this the browser holds onto stale chat.js/chat.css across server restarts
# (only ETag/Last-Modified are sent, and a "hard refresh" doesn't always
# defeat aggressive cache heuristics). Cheap to revalidate every load.
@app.middleware("http")
async def _no_cache_for_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.exception_handler(EmbeddingFailedError)
async def _embedding_failed_handler(request, exc: EmbeddingFailedError):
    """Report a dead embedder as an upstream failure, never as a successful ingest."""
    logger.error("Embedding failed on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=502, content={"success": False, "error": str(exc)})


# Mount routers
app.include_router(health_router.router)
app.include_router(conversations_router.router)
app.include_router(chat_router.router)
app.include_router(kb_router.router)
app.include_router(llm_proxy_router.router)
app.include_router(survey_router.router)
app.include_router(jobs_router.router)
app.include_router(zotero_router.router)
app.include_router(zotero_ingest_router.router)
app.include_router(pdf_dropzone_router.router)


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface."""
    if _TEMPLATE_PATH.exists():
        content = _TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        content = "<h1>Perspicacité v2</h1><p>Template not found. Please ensure templates/index.html exists.</p>"
    return HTMLResponse(content=content)


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404 errors in logs."""
    return Response(content=b"", media_type="image/x-icon")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("perspicacite.web.app:app", host="0.0.0.0", port=8000, reload=False)
