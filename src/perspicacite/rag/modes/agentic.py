"""Agentic RAG - True agent-based research with self-reflection and tool use.

DEPRECATION NOTICE:
This module is now a thin wrapper around AgenticOrchestrator for backward compatibility.
All core functionality has been consolidated into:
    - perspicacite.rag.agentic.orchestrator.AgenticOrchestrator

Key capabilities (now in orchestrator):
- Document quality assessment
- Early exit based on confidence
- Dynamic plan adjustment
- Web search fallback
- Tool selection and execution
- Self-evaluation and refinement
- Hybrid retrieval support

Migration path:
- For direct usage: Use AgenticOrchestrator instead
- For RAGEngine: This wrapper delegates to AgenticOrchestrator internally
"""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.agentic")


class AgenticRAGMode(BaseRAGMode):
    """
    Agentic RAG - True agent-based research.
    
    DEPRECATED: This class is now a thin wrapper around AgenticOrchestrator.
    All functionality has been consolidated into the orchestrator module.
    
    Use AgenticOrchestrator directly for new code:
        from perspicacite.rag.agentic import AgenticOrchestrator
        orchestrator = AgenticOrchestrator(...)
        async for event in orchestrator.chat(query, ...):
            ...
    
    This wrapper maintains backward compatibility for:
    - RAGEngine integration
    - Existing tests that import AgenticRAGMode
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self._orchestrator = None
        self._config = config
        
        # Extract settings for backward compatibility
        rag_settings = getattr(config.rag_modes, "agentic", None)
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            rag_settings = rag_settings.dict()
        
        self.early_exit_confidence = rag_settings.get("early_exit_confidence", 0.85)
        self.max_iterations = rag_settings.get("max_iterations", 3)
        self.use_hybrid = rag_settings.get("use_hybrid", True)

    def _get_orchestrator(
        self,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> "AgenticOrchestrator":
        """Lazy initialization of the underlying orchestrator."""
        if self._orchestrator is None:
            from perspicacite.rag.agentic import AgenticOrchestrator
            
            # Create LLM adapter if needed
            if not hasattr(llm, "complete"):
                raise ValueError("LLM must have a complete method")
            
            self._orchestrator = AgenticOrchestrator(
                llm_client=llm,
                tool_registry=tools,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                max_iterations=self.max_iterations,
                use_hybrid=self.use_hybrid,
                early_exit_confidence=self.early_exit_confidence,
            )
        return self._orchestrator

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute agentic RAG with full tool use and self-reflection.
        
        Delegates to AgenticOrchestrator internally.
        """
        orchestrator = self._get_orchestrator(llm, vector_store, embedding_provider, tools)
        
        logger.info("AgenticRAGMode delegating to AgenticOrchestrator", query=request.query)
        
        # Collect all events from orchestrator
        iterations = 0
        research_plan = []
        final_answer = ""
        papers_found = []
        
        async for event in orchestrator.chat(
            query=request.query,
            session_id=None,  # Stateless mode
            kb_name=request.kb_name,
            stream=False,  # We collect everything
        ):
            event_type = event.get("type", "")
            
            if event_type == "thinking":
                # Track research plan from thinking steps
                message = event.get("message", "")
                if message and message not in research_plan:
                    research_plan.append(message)
                    
            elif event_type == "tool_call":
                iterations += 1
                
            elif event_type == "answer":
                final_answer = event.get("content", "")
                
            elif event_type == "papers_found":
                papers = event.get("papers", [])
                papers_found.extend(papers)
        
        # Convert papers to SourceReference format
        sources = []
        seen = set()
        for paper in papers_found:
            title = paper.get("title", "Unknown")
            if title in seen:
                continue
            seen.add(title)
            
            sources.append(
                SourceReference(
                    title=title,
                    authors=paper.get("authors", []),
                    year=paper.get("year"),
                    relevance_score=paper.get("relevance_score", 0) / 5.0,  # Normalize to 0-1
                )
            )
        
        # Sort by relevance
        sources.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return RAGResponse(
            answer=final_answer or "No answer generated",
            sources=sources[:10],
            mode=RAGMode.AGENTIC,
            iterations=max(iterations, 1),
            research_plan=research_plan[:5],
        )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute agentic RAG with streaming output.
        
        Delegates to AgenticOrchestrator internally.
        """
        orchestrator = self._get_orchestrator(llm, vector_store, embedding_provider, tools)
        
        logger.info("AgenticRAGMode streaming via AgenticOrchestrator", query=request.query)
        
        yield StreamEvent.status("Initializing agentic research...")
        
        async for event in orchestrator.chat(
            query=request.query,
            session_id=None,
            kb_name=request.kb_name,
            stream=True,
        ):
            event_type = event.get("type", "")
            
            if event_type == "thinking":
                yield StreamEvent.status(event.get("message", ""))
                
            elif event_type == "tool_call":
                yield StreamEvent.status(
                    f"Executing: {event.get('description', event.get('tool', 'tool'))}"
                )
                
            elif event_type == "tool_result":
                # Optionally yield tool results as partial content
                pass
                
            elif event_type == "answer":
                content = event.get("content", "")
                # Stream the answer in chunks
                chunk_size = 100
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    yield StreamEvent.token(chunk)
                    
            elif event_type == "papers_found":
                papers = event.get("papers", [])
                yield StreamEvent.status(f"Found {len(papers)} relevant papers")
        
        yield StreamEvent.done()


# Keep backward-compatible exports
__all__ = ["AgenticRAGMode"]
