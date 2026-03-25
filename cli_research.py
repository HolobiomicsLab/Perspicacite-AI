#!/usr/bin/env python3
"""
Perspicacité v2 - CLI Research Tool

Command-line interface for the agentic research system.
Same functionality as web UI but runs in terminal.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/cli_research.log")
    ]
)
logger = logging.getLogger("perspicacite.cli")

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)


async def initialize_system():
    """Initialize the research system."""
    from perspicacite.config.loader import load_config
    from perspicacite.llm import AsyncLLMClient, LiteLLMEmbeddingProvider
    from perspicacite.retrieval import ChromaVectorStore
    from perspicacite.rag.agentic import AgenticOrchestrator, LLMAdapter
    from perspicacite.rag.tools import ToolRegistry, LotusSearchTool
    
    logger.info("=" * 80)
    logger.info("INITIALIZING RESEARCH SYSTEM")
    logger.info("=" * 80)
    
    config = load_config()
    logger.info(f"Config loaded - Provider: {config.llm.default_provider}, Model: {config.llm.default_model}")
    
    llm_client = AsyncLLMClient(config.llm)
    embedding_provider = LiteLLMEmbeddingProvider(model=config.knowledge_base.embedding_model)
    vector_store = ChromaVectorStore(persist_dir="./chroma_db", embedding_provider=embedding_provider)
    
    tool_registry = ToolRegistry()
    lotus_tool = LotusSearchTool()
    tool_registry.register(lotus_tool)
    logger.info(f"Registered tools: {tool_registry.list_tools()}")
    
    llm_adapter = LLMAdapter(
        client=llm_client,
        model=config.llm.default_model,
        provider=config.llm.default_provider
    )
    
    orchestrator = AgenticOrchestrator(
        llm_client=llm_adapter,
        tool_registry=tool_registry,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        max_iterations=5
    )
    
    logger.info("System initialized successfully!")
    return orchestrator


async def run_research(orchestrator, query: str):
    """Run a research query and print results."""
    logger.info("=" * 80)
    logger.info(f"QUERY: {query}")
    logger.info("=" * 80)
    
    print(f"\n🔍 Researching: {query}\n")
    
    step_count = 0
    async for event in orchestrator.chat(query=query, stream=True):
        step_count += 1
        
        if event["type"] == "thinking":
            message = event.get("message", "")
            details = event.get("details", "")
            print(f"🧠 {message}")
            if details:
                print(f"   {details}")
        
        elif event["type"] == "tool_call":
            step = event.get("step", "")
            tool = event.get("tool", "")
            desc = event.get("description", "")
            print(f"🔧 [{step}] Using {tool}: {desc}")
        
        elif event["type"] == "tool_result":
            step = event.get("step", "")
            summary = event.get("result_summary", "")
            print(f"📄 [{step}] Result: {summary[:200]}...")
        
        elif event["type"] == "answer":
            print("\n" + "=" * 80)
            print("📝 ANSWER:")
            print("=" * 80)
            print(event["content"])
            print("=" * 80)
            
            session_id = event.get("session_id", "")
            logger.info(f"Session ID: {session_id}")
            return event["content"]
        
        elif event["type"] == "error":
            print(f"❌ Error: {event.get('message', 'Unknown error')}")
            logger.error(f"Error: {event}")
    
    print(f"\n⚠️ No answer received after {step_count} events")
    return None


async def main():
    """Main CLI entry point."""
    # Initialize system
    orchestrator = await initialize_system()
    
    # Test query
    query = "I want to learn about feature based molecular network"
    
    print("\n" + "=" * 80)
    print("PERSPICACITÉ v2 - CLI RESEARCH TOOL")
    print("=" * 80)
    
    await run_research(orchestrator, query)
    
    # Interactive mode
    print("\n💡 Enter your own queries (or 'quit' to exit):\n")
    
    while True:
        try:
            user_query = input("> ").strip()
            
            if user_query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not user_query:
                continue
            
            await run_research(orchestrator, user_query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    asyncio.run(main())
