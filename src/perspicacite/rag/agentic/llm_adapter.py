"""Adapter for LLM clients to provide simple complete() interface."""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from perspicacite.llm import AsyncLLMClient


class LLMAdapter:
    """Simple adapter that wraps AsyncLLMClient with complete(prompt) interface."""
    
    def __init__(
        self, 
        client: "AsyncLLMClient",
        model: Optional[str] = None,
        provider: Optional[str] = None
    ):
        self.client = client
        self.model = model
        self.provider = provider
    
    async def complete(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        Simple completion interface.
        
        Args:
            prompt: The prompt text
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        
        return await self.client.complete(
            messages=messages,
            model=self.model,
            provider=self.provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
