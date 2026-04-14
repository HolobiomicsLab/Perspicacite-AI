"""Literature search providers."""

from perspicacite.search.scilex_adapter import SciLExAdapter, SciLExSearchProvider
from perspicacite.search.google_scholar import GoogleScholarSearch, SearchAggregator
from perspicacite.search.doi_resolver import resolve_doi, resolve_dois_batch
from perspicacite.search.semantic_scholar import lookup_paper, normalize_paper_id
from perspicacite.search.protocols import SearchProvider

__all__ = [
    "SciLExAdapter",
    "SciLExSearchProvider",
    "GoogleScholarSearch",
    "SearchAggregator",
    "resolve_doi",
    "resolve_dois_batch",
    "lookup_paper",
    "normalize_paper_id",
    "SearchProvider",
]
