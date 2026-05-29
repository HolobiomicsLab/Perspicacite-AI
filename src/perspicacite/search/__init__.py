"""Literature search providers."""

from perspicacite.search.ads_search import ADSSearchProvider
from perspicacite.search.core_search import CORESearchProvider
from perspicacite.search.dblp_sparql_search import DBLPSPARQLSearchProvider
from perspicacite.search.doi_resolver import resolve_doi, resolve_dois_batch
from perspicacite.search.domain_aggregator import DomainAwareAggregator, build_aggregator
from perspicacite.search.domain_classifier import DomainClassifier
from perspicacite.search.europepmc_search import EuropePMCSearchProvider
from perspicacite.search.google_scholar import GoogleScholarSearch, SearchAggregator
from perspicacite.search.google_scholar_playwright import GoogleScholarPlaywrightProvider
from perspicacite.search.inspire_search import INSPIREHEPSearchProvider
from perspicacite.search.protocols import SearchProvider
from perspicacite.search.pubchem_search import PubChemSearchProvider
from perspicacite.search.scilex_adapter import SciLExAdapter, SciLExSearchProvider
from perspicacite.search.semantic_scholar import lookup_paper, normalize_paper_id

__all__ = [
    "ADSSearchProvider",
    "CORESearchProvider",
    "DBLPSPARQLSearchProvider",
    "DomainAwareAggregator",
    "DomainClassifier",
    "EuropePMCSearchProvider",
    "GoogleScholarPlaywrightProvider",
    "GoogleScholarSearch",
    "INSPIREHEPSearchProvider",
    "PubChemSearchProvider",
    "SciLExAdapter",
    "SciLExSearchProvider",
    "SearchAggregator",
    "SearchProvider",
    "build_aggregator",
    "lookup_paper",
    "normalize_paper_id",
    "resolve_doi",
    "resolve_dois_batch",
]
