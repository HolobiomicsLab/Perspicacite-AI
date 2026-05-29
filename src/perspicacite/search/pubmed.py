"""PubMed deep-search adapter using Biopython Entrez (esearch → efetch)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, ClassVar

from perspicacite.logging import get_logger
from perspicacite.models.papers import Author, Paper, PaperSource

try:
    from Bio import Entrez
except ImportError:
    Entrez = None  # type: ignore[assignment]

logger = get_logger(__name__)

_OBVIOUS_PLACEHOLDERS = {
    "",
    "user@example.com",
    "you@example.com",
    "your.email@domain.com",
    "email@example.com",
    "test@test.com",
}


class PubMedConfigError(RuntimeError):
    """Raised when PubMed adapter is misconfigured."""


def _parse_efetch(handle: Any) -> list[dict[str, Any]]:
    """Parse a PubMed efetch XML handle into plain dicts.

    Each dict contains: pmid, title, year, doi, abstract, journal, authors.
    Malformed records yield partial data rather than raising.
    """
    if Entrez is None:
        return []

    try:
        records = Entrez.read(handle)
    except Exception:
        logger.warning("pubmed_efetch_parse_failed")
        return []

    results: list[dict[str, Any]] = []

    for article in records.get("PubmedArticle", []):
        row: dict[str, Any] = {
            "pmid": None,
            "title": None,
            "year": None,
            "doi": None,
            "abstract": None,
            "journal": None,
            "authors": [],
        }

        # --- PMID ---
        try:
            medline = article["MedlineCitation"]
            row["pmid"] = str(medline["PMID"])
        except (KeyError, TypeError, AttributeError):
            pass

        # --- Article fields ---
        try:
            art = article["MedlineCitation"]["Article"]
        except (KeyError, TypeError, AttributeError):
            results.append(row)
            continue

        with contextlib.suppress(Exception):
            row["title"] = str(art["ArticleTitle"])

        try:
            abstract_texts = art["Abstract"]["AbstractText"]
            if isinstance(abstract_texts, list):
                row["abstract"] = " ".join(str(t) for t in abstract_texts)
            else:
                row["abstract"] = str(abstract_texts)
        except (KeyError, TypeError, AttributeError):
            pass

        with contextlib.suppress(Exception):
            row["journal"] = str(art["Journal"]["Title"])

        try:
            pub_date = art["Journal"]["JournalIssue"]["PubDate"]
            year_str = str(pub_date.get("Year", "") or pub_date.get("MedlineDate", ""))
            if year_str:
                import re

                m = re.search(r"\b(19|20)\d{2}\b", year_str)
                if m:
                    row["year"] = int(m.group())
        except (KeyError, TypeError, AttributeError, ValueError):
            pass

        try:
            authors: list[str] = []
            for auth in art.get("AuthorList", []):
                try:
                    fore = auth.get("ForeName", "") or ""
                    last = auth.get("LastName", "") or ""
                    name = f"{fore} {last}".strip() if fore or last else None
                    if name:
                        authors.append(name)
                except (AttributeError, TypeError):
                    pass
            row["authors"] = authors
        except (KeyError, TypeError, AttributeError):
            pass

        # --- DOI from PubmedData ---
        try:
            for id_entry in article["PubmedData"]["ArticleIdList"]:
                if id_entry.attributes.get("IdType") == "doi":
                    row["doi"] = str(id_entry)
                    break
        except (KeyError, TypeError, AttributeError):
            pass

        results.append(row)

    return results


class PubMedSearchAdapter:
    """Search PubMed via NCBI Entrez (esearch → efetch), returning Paper models."""

    name = "pubmed"
    description = "Direct NCBI PubMed search via Biopython Entrez (esearch → efetch)"
    # PubMed indexes biomedical literature but the boundary with adjacent
    # fields (chemistry, ML-in-medicine, public health) is fuzzy, and the
    # domain classifier returns "general" for many biomedical-leaning chatty
    # queries (e.g. "statins heart attack risk last 5 years"). Adding
    # "general" lets PubMed serve as the baseline biomedical fallback for
    # those queries too; it just returns 0 hits for genuinely non-biomedical
    # queries (CS/ML), which the aggregator merges harmlessly with the
    # other providers.
    domains: ClassVar[list[str]] = ["general", "biomedical"]
    tier: str = "reliable"
    retry: int = 0

    def __init__(
        self,
        email: str,
        api_key: str | None = None,
        rate_limit_per_sec: float | None = None,
    ) -> None:
        if Entrez is None:
            raise PubMedConfigError(
                "Biopython is not installed. Install it with: pip install biopython"
            )
        if not email or email.strip().lower() in _OBVIOUS_PLACEHOLDERS:
            raise PubMedConfigError(
                "PubMed search requires a real NCBI email. "
                "Set config.scilex.pubmed_email (or pdf_download.unpaywall_email) "
                "to your address."
            )
        self.email = email
        self.api_key = api_key or None

        Entrez.email = email  # type: ignore[assignment]  # Biopython stubs type email/api_key as None; runtime accepts str
        if self.api_key:
            Entrez.api_key = self.api_key  # type: ignore[assignment]  # same as above

        # Rate-limit metadata: 10 req/s with key, 3 req/s without (NCBI guidelines)
        self._min_interval = 1.0 / (rate_limit_per_sec or (10.0 if self.api_key else 3.0))

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        **_: Any,
    ) -> list[Paper]:
        """Search PubMed and return a list of Paper objects.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            year_min: Restrict to papers published on or after this year.
            year_max: Restrict to papers published on or before this year.

        Returns:
            List of Paper objects ordered by relevance (PubMed default).
        """
        term = query
        if year_min or year_max:
            term += f" AND ({year_min or 1800}:{year_max or 2100}[dp])"

        def _run() -> list[dict[str, Any]]:
            h = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
            ids = Entrez.read(h).get("IdList", [])
            if not ids:
                return []
            fh = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml", retmode="xml")
            return _parse_efetch(fh)

        raw_records: list[dict[str, Any]] = await asyncio.to_thread(_run)

        papers: list[Paper] = []
        for r in raw_records:
            doi = r.get("doi")
            pmid = r.get("pmid")
            paper_id = doi or (f"pmid:{pmid}" if pmid else "pmid:unknown")
            papers.append(
                Paper(
                    id=paper_id,
                    title=r.get("title") or "",
                    authors=[Author(name=a) for a in r.get("authors", [])],
                    year=r.get("year"),
                    doi=doi,
                    abstract=r.get("abstract"),
                    journal=r.get("journal"),
                    source=PaperSource.PUBMED,
                    metadata={"pmid": pmid},
                )
            )

        logger.info("pubmed_search", query=query, results=len(papers))
        return papers
