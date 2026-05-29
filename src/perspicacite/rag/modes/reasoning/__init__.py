"""RAGMode.REASONING dispatcher.

Routes on ``request.reasoning_strategy`` to one of four strategy modules.
Subplan A ships ``provenance`` and ``contradiction``; unshipped strategies
return a single error StreamEvent pointing to the planned sprint.

The indicia extra (``indicium`` + ``pyoxigraph``) is required. When missing,
we yield a friendly error event explaining how to enable.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from perspicacite.logging import get_logger
from perspicacite.models.rag import (
    RAGMode,
    RAGRequest,
    RAGResponse,
    SourceReference,
    StreamEvent,
)
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.reasoning")

# After Subplan B: all four strategies ship; default is evidence_graded.
SHIPPED_STRATEGIES: frozenset[str] = frozenset(
    {"provenance", "contradiction", "graph", "evidence_graded"}
)
DEFAULT_REASONING_STRATEGY: str = "evidence_graded"

# Module-level toggle so tests can monkey-patch without touching sys.modules.
try:
    import indicium  # noqa: F401
    import pyoxigraph  # noqa: F401

    _HAS_INDICIA = True
except ImportError:
    _HAS_INDICIA = False


def _resolve_strategy(request: RAGRequest) -> str:
    return request.reasoning_strategy or DEFAULT_REASONING_STRATEGY


class ReasoningRAGMode(BaseRAGMode):
    """RAG mode dispatcher for indicium-backed reasoning."""

    def __init__(self, config: Any, *, session_store: Any = None) -> None:
        super().__init__(config)
        self.session_store = session_store

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        if not _HAS_INDICIA:
            yield StreamEvent(
                event="error",
                data=json.dumps(
                    {
                        "message": (
                            "Reasoning mode requires the 'indicia' extra. "
                            "Install with: uv sync --extra indicia"
                        )
                    }
                ),
            )
            return

        strategy = _resolve_strategy(request)
        if strategy not in SHIPPED_STRATEGIES:
            yield StreamEvent(
                event="error",
                data=json.dumps(
                    {
                        "message": (
                            f"reasoning_strategy='{strategy}' is not yet shipped "
                            f"(planned in Subplan B). Use one of: "
                            f"{sorted(SHIPPED_STRATEGIES)}."
                        )
                    }
                ),
            )
            return

        if strategy == "provenance":
            from perspicacite.rag.modes.reasoning.provenance import (
                run_provenance_stream,
            )

            async for ev in run_provenance_stream(
                request=request,
                llm=llm,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                config=self.config,
                session_store=self.session_store,
            ):
                yield ev
            return

        elif strategy == "graph":
            from perspicacite.rag.modes.reasoning.graph_traversal import (
                run_graph_traversal_stream,
            )

            async for ev in run_graph_traversal_stream(
                request=request,
                llm=llm,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                config=self.config,
                session_store=self.session_store,
            ):
                yield ev
            return

        elif strategy == "evidence_graded":
            from perspicacite.rag.modes.reasoning.evidence_graded import (
                run_evidence_graded_stream,
            )

            async for ev in run_evidence_graded_stream(
                request=request,
                llm=llm,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                config=self.config,
                session_store=self.session_store,
            ):
                yield ev
            return

        # strategy == "contradiction"
        from perspicacite.rag.modes.reasoning.typed_contradiction import (
            run_typed_contradiction_stream,
        )

        async for ev in run_typed_contradiction_stream(
            request=request,
            llm=llm,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            config=self.config,
            session_store=self.session_store,
        ):
            yield ev

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        answer_parts: list[str] = []
        sources: list[SourceReference] = []
        async for ev in self.execute_stream(
            request=request,
            llm=llm,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            tools=tools,
        ):
            if ev.event == "content":
                try:
                    delta = json.loads(ev.data).get("delta", "")
                except Exception:
                    delta = ev.data
                answer_parts.append(delta)
            elif ev.event == "source":
                with contextlib.suppress(Exception):
                    sources.append(SourceReference(**json.loads(ev.data)))
            elif ev.event == "error":
                try:
                    msg = json.loads(ev.data).get("message", ev.data)
                except Exception:
                    msg = ev.data
                answer_parts.append(f"\n\n[Error: {msg}]")
        return RAGResponse(
            answer="".join(answer_parts),
            sources=sources,
            mode=RAGMode.REASONING,
            iterations=1,
            web_search_used=False,
        )
