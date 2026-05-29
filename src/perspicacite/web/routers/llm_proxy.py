"""Raw-LLM proxy endpoint.

Lets external clients (e.g. Scriptorium) use Perspicacité as an
LLM gateway. **No RAG, no KB awareness** — just credentials routing
+ the existing stage-tiering rules.

Added 2026-05-15 per Scriptorium-v0.13 integration feedback.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/llm", tags=["llm-proxy"])
logger = structlog.get_logger("perspicacite.web.llm_proxy")


class LLMProxyRequest(BaseModel):
    prompt: str = Field(..., description="Raw prompt to send to the LLM")
    model: str | None = Field(
        default=None,
        description="Override model. Defaults to the config's default model.",
    )
    max_tokens: int = Field(default=2048, ge=1, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stage: str | None = Field(
        default=None,
        description="Stage-tiering hint passed through to the LLM client.",
    )


@router.post("/proxy")
async def llm_proxy(request: LLMProxyRequest, http_request: Request) -> StreamingResponse:
    """Stream the model's response as text/plain chunks.

    No RAG, no KB. The endpoint exists so that clients with their own
    retrieval logic can use Perspicacité's API keys + provider config.
    """
    async def gen() -> AsyncIterator[bytes]:
        async for chunk in _call_llm_streaming(
            prompt=request.prompt,
            model=request.model,
            stage=request.stage,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            if isinstance(chunk, bytes):
                yield chunk
            else:
                yield chunk.encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


async def _call_llm_streaming(
    *,
    prompt: str,
    model: str | None,
    stage: str | None,
    max_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """Resolve the LLM client and stream chunks. No retrieval — pure
    pass-through. Kept as a module-level helper so tests can monkey-patch it.
    """
    from perspicacite.config.loader import load_config
    from perspicacite.llm.client import AsyncLLMClient

    config = load_config()
    llm_client = AsyncLLMClient(config=config.llm)

    resolved_model = model or config.llm.default_model
    messages = [{"role": "user", "content": prompt}]

    kwargs = {}
    if stage:
        kwargs["stage"] = stage

    logger.info(
        "llm_proxy_stream_start",
        model=resolved_model,
        stage=stage,
        prompt_chars=len(prompt),
    )

    async for chunk in llm_client.stream(
        messages=messages,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    ):
        yield chunk
