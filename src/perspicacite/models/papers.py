"""Paper and document models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PaperSource(str, Enum):
    """Source of a paper."""

    BIBTEX = "bibtex"
    SCILEX = "scilex"
    WEB_SEARCH = "web_search"
    USER_UPLOAD = "user_upload"
    CITATION_FOLLOW = "citation_follow"


class Author(BaseModel):
    """Author of a paper."""

    model_config = {"frozen": True}

    name: str
    given: Optional[str] = None
    family: Optional[str] = None
    orcid: Optional[str] = None

    def __repr__(self) -> str:
        return f"Author(name='{self.name}')"

    def __str__(self) -> str:
        return self.name


class Paper(BaseModel):
    """Canonical paper representation used across the entire system."""

    id: str = Field(description="Unique ID: DOI, PMID, or generated UUID")
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    source: PaperSource = PaperSource.BIBTEX
    keywords: list[str] = Field(default_factory=list)
    full_text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
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
    def first_author(self) -> Optional[str]:
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
            try:
                year = int(year_str)
            except ValueError:
                pass

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
