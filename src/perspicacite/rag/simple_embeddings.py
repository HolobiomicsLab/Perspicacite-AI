"""Simple embedding provider using OpenAI directly.

This avoids the LiteLLM import issues.
"""

from typing import Any

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.simple_embeddings")


class SimpleOpenAIEmbeddingProvider:
    """
    Simple embedding provider using OpenAI API directly.
    
    This is a lightweight alternative to LiteLLMEmbeddingProvider
    that avoids the heavy LiteLLM import.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 32,
    ):
        self.model = model
        self.api_key = api_key or self._get_api_key()
        self.batch_size = batch_size
        self._dimension = self._get_dimension()

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        import os
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        return key

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * self.dimension for _ in texts]

        logger.debug("embedding_start", text_count=len(valid_texts), model=self.model)

        try:
            all_embeddings = []
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                for i in range(0, len(valid_texts), self.batch_size):
                    batch = valid_texts[i:i + self.batch_size]
                    
                    response = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "input": batch,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)

            logger.debug(
                "embedding_complete",
                text_count=len(valid_texts),
                dimension=self.dimension,
            )

            return all_embeddings

        except Exception as e:
            logger.error(
                "embedding_error",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
