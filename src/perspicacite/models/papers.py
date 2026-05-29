"""Paper and document models."""

import contextlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PaperSource(str, Enum):
    """Source of a paper.

    Legacy values (BIBTEX, SCILEX, WEB_SEARCH, USER_UPLOAD,
    CITATION_FOLLOW, LOCAL) are kept for backward compat.
    Audit 2026-05-15 finding #5 added explicit database sources
    (OPENALEX, PUBMED, ARXIV, CROSSREF). The 2026-05-15 follow-up
    migration added SEMANTIC_SCHOLAR for direct S2 API hits.
    Added 2026-05-15: SKILL_BUNDLE for ASB (Agent Skill Bundle) ingest —
    chunks derived from skills and workflow cards rather than literature.
    """

    BIBTEX = "bibtex"
    SCILEX = "scilex"
    WEB_SEARCH = "web_search"
    USER_UPLOAD = "user_upload"
    CITATION_FOLLOW = "citation_follow"
    LOCAL = "local"
    OPENALEX = "openalex"
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    CROSSREF = "crossref"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    EUROPE_PMC = "europe_pmc"
    PUBCHEM = "pubchem"
    CORE = "core"
    INSPIRE_HEP = "inspire_hep"
    ADS = "ads"
    OPENCITATIONS = "opencitations"
    GOOGLE_SCHOLAR = "google_scholar"
    DBLP_SPARQL = "dblp_sparql"
    OPENROUTER_WEB = "openrouter_web"
    # Added 2026-05-15: tags chunks from ASB skill/card ingest so downstream
    # consumers can distinguish them from literature-derived chunks.
    SKILL_BUNDLE = "skill_bundle"


class Author(BaseModel):
    """Author of a paper."""

    model_config = {"frozen": True}

    name: str
    given: str | None = None
    family: str | None = None
    orcid: str | None = None

    def __repr__(self) -> str:
        return f"Author(name='{self.name}')"

    def __str__(self) -> str:
        return self.name


class Paper(BaseModel):
    """Canonical paper representation used across the entire system."""

    id: str = Field(description="Unique ID: DOI, PMID, or generated UUID")
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    pmid: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    source: PaperSource = PaperSource.BIBTEX
    keywords: list[str] = Field(default_factory=list)
    # === Provenance ===
    # Which DBs returned this specific paper (e.g. ["openalex", "pubmed"]).
    discovery_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Upstream databases that returned this paper. Filled by "
            "the aggregator merge step."
        ),
    )
    # Which DBs ENRICHED the metadata (Crossref, Unpaywall, OpenAlex
    # fill-in, etc.). Distinct from discovery_sources.
    enrichment_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Secondary databases that contributed metadata enrichment "
            "(Crossref bibliographic patch, OpenAlex abstract fill, "
            "Unpaywall OA detection)."
        ),
    )
    full_text: str | None = None
    # Pipeline content-tier marker carried from PaperContent.content_type:
    # one of "structured" | "full_text" | "abstract" | "none". None when
    # the paper was loaded outside the unified download pipeline.
    content_type: str | None = None
    license: str | None = None  # OA license id from discovery (e.g. "cc-by")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int | None) -> int | None:
        """Validate year is reasonable."""
        if v is None:
            return v
        current_year = datetime.now().year
        if v < 1800 or v > current_year + 1:
            raise ValueError(f"Year must be between 1800 and {current_year + 1}")
        return v

    def __repr__(self) -> str:
        return f"Paper(id='{self.id}', title='{self.title[:50]}...')"

    @property
    def first_author(self) -> str | None:
        """Get first author name or None."""
        if self.authors:
            return self.authors[0].name
        return None

    @property
    def citation_key(self) -> str:
        """Generate a citation key (AuthorYear format)."""
        author_part = "Unknown"
        if self.authors and self.authors[0].family:
            author_part = self.authors[0].family
        elif self.authors:
            author_part = self.authors[0].name.split()[-1]
        year_part = str(self.year) if self.year else "n.d."
        return f"{author_part}{year_part}"

    @classmethod
    def from_bibtex(cls, entry: dict[str, Any]) -> "Paper":
        """Create Paper from BibTeX entry dict."""
        # Extract authors
        authors = []
        author_field = entry.get("author", "")
        if author_field:
            for author_str in author_field.split(" and "):
                author_str = author_str.strip()
                if not author_str:
                    continue
                # Try to parse "Family, Given" format
                if "," in author_str:
                    parts = author_str.split(",", 1)
                    family = parts[0].strip()
                    given = parts[1].strip() if len(parts) > 1 else None
                    name = f"{given} {family}" if given else family
                    authors.append(
                        Author(name=name, given=given, family=family)
                    )
                else:
                    # "Given Family" format
                    parts = author_str.rsplit(" ", 1)
                    if len(parts) == 2:
                        family = parts[1]
                        given = parts[0]
                        authors.append(
                            Author(name=author_str, given=given, family=family)
                        )
                    else:
                        authors.append(Author(name=author_str))

        # Extract year
        year = None
        year_str = entry.get("year")
        if year_str:
            with contextlib.suppress(ValueError):
                year = int(year_str)

        # Generate ID from DOI or PMID, or create from title
        doi = entry.get("doi")
        pmid = entry.get("pmid")
        if doi:
            paper_id = f"doi:{doi}"
        elif pmid:
            paper_id = f"pmid:{pmid}"
        else:
            # Generate from title hash
            import hashlib

            title = entry.get("title", "")
            paper_id = f"generated:{hashlib.md5(title.encode()).hexdigest()[:12]}"

        return cls(
            id=paper_id,
            title=entry.get("title", ""),
            authors=authors,
            abstract=entry.get("abstract"),
            year=year,
            journal=entry.get("journal") or entry.get("journaltitle"),
            doi=doi,
            pmid=pmid,
            url=entry.get("url"),
            pdf_url=entry.get("file"),
            keywords=entry.get("keywords", "").split(", ") if entry.get("keywords") else [],
            source=PaperSource.BIBTEX,
            metadata={k: v for k, v in entry.items() if k not in {
                "title", "author", "abstract", "year", "journal", "journaltitle",
                "doi", "pmid", "url", "file", "keywords"
            }},
        )


def normalize_paper_dict(raw: dict[str, Any], source: PaperSource = PaperSource.WEB_SEARCH) -> dict[str, Any]:
    """Normalize any provider's paper response into a standardized dict.

    This is the single source of truth for paper data formatting across the
    entire system. All paper-returning functions should pass their raw
    provider response through this before returning.

    Args:
        raw: Raw paper data from any provider (OpenAlex, Semantic Scholar, arXiv, etc.)
        source: The source identifier for where this paper came from

    Returns:
        Normalized dict with keys matching the Paper schema:
        {
            "id": str,
            "title": str,
            "authors": list[str],  # List of author names (strings), not Author objects
            "abstract": str | None,
            "year": int | None,
            "doi": str | None,
            "pmid": str | None,
            "url": str | None,
            "pdf_url": str | None,
            "citation_count": int | None,
            "source": PaperSource,
            "full_text": str | None,
            "metadata": dict,
        }
    """
    # Extract title
    title = raw.get("title") or raw.get("display_name") or "Unknown"

    # Extract and normalize authors
    # Multiple possible input formats:
    # - OpenAlex: authorships=[{"author": {"display_name": "..."}}]
    # - Semantic Scholar: authors=[{"name": "..."}]
    # - List of strings: ["Author One", "Author Two"]
    # - Comma-separated string: "Author One, Author Two"
    authors: list[str] = []

    # First check authorships (OpenAlex format)
    authorships = raw.get("authorships")
    if authorships and isinstance(authorships, list):
        for a in authorships[:10]:  # Limit to first 10
            if isinstance(a, dict):
                author_obj = a.get("author") or a
                name = author_obj.get("display_name") or author_obj.get("name")
                if name:
                    authors.append(name)

    # If no authorships, check authors directly
    if not authors:
        authors_raw = raw.get("authors")
        if authors_raw:
            if isinstance(authors_raw, list):
                for a in authors_raw[:10]:
                    if isinstance(a, Author):
                        authors.append(a.name)
                    elif isinstance(a, dict):
                        name = a.get("display_name") or a.get("name")
                        if name:
                            authors.append(name)
                    elif isinstance(a, str):
                        authors.append(a)
            elif isinstance(authors_raw, str):
                # Comma-separated or " and "-separated
                for part in authors_raw.replace(" and ", ",").split(","):
                    part = part.strip()
                    if part:
                        authors.append(part)

    # Extract year
    year = raw.get("year")
    if year is None:
        year = raw.get("publication_year")
    if year and isinstance(year, str):
        import re
        year_match = re.search(r"\b(19|20)\d{2}\b", year)
        year = int(year_match.group()) if year_match else None
    if year and isinstance(year, int):
        # Validate reasonable range (Paper model does 1800 to current_year+1)
        current_year = datetime.now().year
        if 1800 <= year <= current_year + 1:
            pass  # Valid year
        else:
            year = None
    else:
        year = None

    # Extract DOI
    doi = raw.get("doi")
    if doi:
        # Strip URL prefixes
        for prefix in ("https://doi.org/", "http://dx.doi.org/", "doi:", "DOI:"):
            if isinstance(doi, str) and doi.lower().startswith(prefix.lower()):
                doi = doi[len(prefix):].strip()
    # Validate DOI has a slash (basic check)
    if not doi or not isinstance(doi, str) or "/" not in doi:
        doi = None

    # Extract PMID
    pmid = raw.get("pmid") or raw.get("pmid_id")

    # Extract URLs
    url = raw.get("url") or raw.get("link")
    pdf_url = raw.get("pdf_url") or raw.get("fulltext_pdf") or raw.get("oa_link")

    # Extract citation count (try multiple field names)
    # Use .get() with None default to avoid treating 0 as falsy
    citation_count = raw.get("citation_count")
    if citation_count is None:
        citation_count = raw.get("cited_by_count")
    if citation_count is None:
        citation_count = raw.get("citationCount")  # Semantic Scholar format

    if citation_count is not None and isinstance(citation_count, str):
        try:
            citation_count = int(citation_count)
        except ValueError:
            citation_count = None

    # Generate ID if not provided
    paper_id = raw.get("id")
    if not paper_id:
        if doi:
            paper_id = f"doi:{doi}"
        elif pmid:
            paper_id = f"pmid:{pmid}"
        elif raw.get("arxiv_id"):
            paper_id = f"arxiv:{raw['arxiv_id']}"
        else:
            import hashlib
            title_hash = hashlib.md5(str(title).encode()).hexdigest()[:12]
            paper_id = f"generated:{title_hash}"

    return {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": raw.get("abstract") or raw.get("description"),
        "year": year,
        "doi": doi,
        "pmid": pmid,
        "url": url,
        "pdf_url": pdf_url,
        "citation_count": citation_count,
        "source": source,
        "full_text": raw.get("full_text"),
        "metadata": {k: v for k, v in raw.items() if k not in {
            "id", "title", "authors", "authorships", "abstract", "year", "doi", "pmid",
            "url", "link", "pdf_url", "fulltext_pdf", "oa_link", "citation_count",
            "full_text", "display_name", "publication_year", "publication_date",
            "cited_by_count", "citationCount", "description"
        }},
    }
