"""Search provider protocol definitions."""

from typing import Any, Protocol

from perspicacite.models.papers import Paper


class SearchProvider(Protocol):
    """Protocol for literature search providers."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        apis: list[str] | None = None,
    ) -> list[Paper]: ...
