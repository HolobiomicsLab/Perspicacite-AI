"""Unit tests for Paper source fields and open-access candidate discovery."""
import json

from perspicacite.models.papers import Paper
from perspicacite.pipeline.download import discovery
from perspicacite.pipeline.download.base import PaperDiscovery


def test_discovery_sources_default_empty():
    p = Paper(id="x", title="t")
    assert p.discovery_sources == []
    assert p.enrichment_sources == []


def test_legacy_metadata_sources_not_mirrored():
    """Back-compat shim removed: metadata['sources'] does NOT populate
    discovery_sources. Callers must use the typed field directly."""
    p = Paper(
        id="x", title="t",
        metadata={"sources": ["openalex", "pubmed"]},
    )
    assert p.discovery_sources == []


def test_legacy_metadata_enrichment_sources_not_mirrored():
    """Back-compat shim removed: metadata['enrichment_sources'] does NOT
    populate enrichment_sources. Callers must use the typed field directly."""
    p = Paper(
        id="x", title="t",
        metadata={"enrichment_sources": ["crossref"]},
    )
    assert p.enrichment_sources == []


def test_explicit_typed_field_used_directly():
    """Typed fields are populated when passed as kwargs, independent of
    metadata."""
    p = Paper(
        id="x", title="t",
        discovery_sources=["new_value"],
        metadata={"sources": ["ignored_legacy_value"]},
    )
    assert p.discovery_sources == ["new_value"]


def test_fields_are_independent_lists():
    p1 = Paper(id="x", title="t")
    p2 = Paper(id="y", title="t")
    p1.discovery_sources.append("openalex")
    assert p2.discovery_sources == []


# --- Open-access candidate discovery (pipeline/download/discovery.py) -------
#
# These cover the ranked oa_candidates list, the id extractors that mine
# location urls, and the discovery cache round-trip. All offline: the
# functions under test are pure payload parsers.

# A url that mentions arXiv but is not an arXiv article path; the negative
# twin of the real landing pages below.
ARXIV_LOOKALIKE_URL = "https://www.semanticscholar.org/search?q=arxiv/2310.11511"
# A bare-digit article path on a host that is not PMC; the negative twin of
# the legacy PMC url.
PMC_LOOKALIKE_URL = "https://journals.example.org/articles/6940144"


class TestArxivIdFromOpenAlexLocations:
    """_extract_arxiv_id_from_openalex must mine locations, not just ids."""

    def test_id_recovered_from_locations_without_arxiv_key(self):
        """work['ids'] has no arxiv key, but a location names the record."""
        work = {
            "ids": {"openalex": "https://openalex.org/W123"},
            "doi": "https://doi.org/10.1234/journal.abc",
            "locations": [
                {"landing_page_url": "https://example.org/record/1"},
                {"pdf_url": "https://arxiv.org/pdf/2310.11511"},
            ],
        }
        assert discovery._extract_arxiv_id_from_openalex(work) == "2310.11511"

    def test_id_recovered_from_best_oa_location(self):
        """best_oa_location is scanned too, not only the locations array."""
        work = {
            "best_oa_location": {
                "landing_page_url": "https://arxiv.org/abs/2310.11511v2",
            },
        }
        assert discovery._extract_arxiv_id_from_openalex(work) == "2310.11511v2"

    def test_direct_ids_key_still_wins(self):
        """Regression guard: the pre-existing ids['arxiv'] path is unchanged."""
        work = {"ids": {"arxiv": "https://arxiv.org/abs/1706.03762"}}
        assert discovery._extract_arxiv_id_from_openalex(work) == "1706.03762"

    def test_lookalike_url_stays_clean(self):
        """A url merely containing 'arxiv' yields None, not a bogus id."""
        work = {"locations": [{"landing_page_url": ARXIV_LOOKALIKE_URL}]}
        assert discovery._extract_arxiv_id_from_openalex(work) is None

    def test_non_arxiv_work_yields_none(self):
        """No arXiv anywhere is reported as None (unknown), never a guess."""
        work = {
            "ids": {"pmid": "12345"},
            "doi": "https://doi.org/10.1234/journal.abc",
            "locations": [{"pdf_url": "https://publisher.example.org/a.pdf"}],
        }
        assert discovery._extract_arxiv_id_from_openalex(work) is None


class TestPmcidFromUnpaywallLocations:
    """The PMC url matcher must accept both id spellings and both hosts."""

    def test_legacy_bare_digit_url_yields_pmcid(self):
        """The old /pmc/articles/6940144 form has no PMC prefix in the path."""
        locations = [{"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/6940144/"}]
        result = discovery._extract_pmcid_from_unpaywall_locations(locations)
        assert result == "PMC6940144"

    def test_prefixed_url_still_yields_pmcid(self):
        """Regression guard: the currently-working PMC-prefixed form."""
        locations = [{"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6940144"}]
        result = discovery._extract_pmcid_from_unpaywall_locations(locations)
        assert result == "PMC6940144"

    def test_url_for_pdf_is_scanned(self):
        """url_for_pdf was omitted before; a PMC pdf-only location must hit."""
        locations = [
            {"url_for_pdf": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6940144/pdf"}
        ]
        result = discovery._extract_pmcid_from_unpaywall_locations(locations)
        assert result == "PMC6940144"

    def test_europepmc_host_yields_pmcid(self):
        """Europe PMC serves the same articles under its own host."""
        locations = [{"url": "https://europepmc.org/articles/PMC6940144"}]
        result = discovery._extract_pmcid_from_unpaywall_locations(locations)
        assert result == "PMC6940144"

    def test_lookalike_article_path_stays_clean(self):
        """A bare-digit /articles/ path off-host must not be read as a PMCID."""
        locations = [{"url": PMC_LOOKALIKE_URL}]
        assert discovery._extract_pmcid_from_unpaywall_locations(locations) is None

    def test_no_locations_yields_none(self):
        """None input means 'not checked' and must not raise."""
        assert discovery._extract_pmcid_from_unpaywall_locations(None) is None


class TestOpenAlexCandidateRanking:
    """_collect_openalex_oa_candidates ranks pdf urls ahead of oa_url."""

    def test_candidates_are_ranked_pdf_first(self):
        """locations[].pdf_url outranks the open_access.oa_url landing page."""
        work = {
            "locations": [{"pdf_url": "https://repo.example.org/paper.pdf"}],
            "best_oa_location": {"pdf_url": "https://mirror.example.org/p.pdf"},
            "open_access": {
                "is_oa": True,
                "oa_url": "https://publisher.example.org/landing",
            },
        }
        assert discovery._collect_openalex_oa_candidates(work) == [
            "https://repo.example.org/paper.pdf",
            "https://mirror.example.org/p.pdf",
            "https://publisher.example.org/landing",
        ]

    def test_closed_work_yields_empty_list(self):
        """No OA anywhere gives [], never [None] and never a landing page."""
        work = {
            "locations": [{"pdf_url": None, "landing_page_url": "https://x.org/a"}],
            "best_oa_location": None,
            "primary_location": {"pdf_url": None},
            "open_access": {"is_oa": False, "oa_url": None},
        }
        candidates = discovery._collect_openalex_oa_candidates(work)
        assert candidates == []
        assert None not in candidates

    def test_oa_url_gated_on_is_oa(self):
        """A closed work's oa_url is not promoted into the candidate list."""
        work = {"open_access": {"is_oa": False, "oa_url": "https://x.org/paper"}}
        assert discovery._collect_openalex_oa_candidates(work) == []


class TestUnpaywallCandidateRanking:
    """_collect_unpaywall_oa_candidates keeps every url, best first."""

    def test_repository_pdf_beats_bare_doi_redirect(self):
        """url_for_pdf outranks the same location's doi.org landing url."""
        data = {
            "best_oa_location": {
                "url_for_pdf": "https://repo.example.org/paper.pdf",
                "url": "https://doi.org/10.1234/journal.abc",
            },
        }
        assert discovery._collect_unpaywall_oa_candidates(data) == [
            "https://repo.example.org/paper.pdf",
            "https://doi.org/10.1234/journal.abc",
        ]

    def test_secondary_locations_are_kept(self):
        """Every oa_locations entry survives instead of being discarded."""
        data = {
            "best_oa_location": {"url": "https://a.example.org/landing"},
            "oa_locations": [
                {"url": "https://a.example.org/landing"},
                {"url_for_pdf": "https://b.example.org/p.pdf",
                 "url": "https://b.example.org/landing"},
            ],
        }
        assert discovery._collect_unpaywall_oa_candidates(data) == [
            "https://a.example.org/landing",
            "https://b.example.org/p.pdf",
            "https://b.example.org/landing",
        ]

    def test_empty_payload_yields_empty_list(self):
        """A payload naming no OA url gives [], never [None]."""
        data = {"best_oa_location": None, "oa_locations": []}
        assert discovery._collect_unpaywall_oa_candidates(data) == []


class TestDiscoveryCacheRoundTrip:
    """oa_candidates must survive the disk cache, old files must still load."""

    def test_candidates_survive_the_cache(self, tmp_path, monkeypatch):
        """Without this, every cache hit collapses the ranked list."""
        monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
        disc = PaperDiscovery(
            doi="10.1234/journal.abc",
            oa_url="https://a.example.org/landing",
            oa_candidates=["https://a.example.org/p.pdf",
                           "https://a.example.org/landing"],
        )
        discovery._write_discovery_cache(disc)
        loaded = discovery._read_discovery_cache("10.1234/journal.abc")
        assert loaded is not None
        assert loaded.oa_candidates == disc.oa_candidates
        assert loaded.oa_url == disc.oa_url

    def test_old_cache_file_without_new_key_loads(self, tmp_path, monkeypatch):
        """Files written before oa_candidates existed must still be readable."""
        monkeypatch.setattr(discovery, "_CACHE_DIR", tmp_path)
        legacy = {
            "doi": "10.1234/journal.abc",
            "pmcid": "PMC6940144",
            "arxiv_id": None,
            "oa_url": "https://a.example.org/landing",
            "abstract": "An abstract.",
            "title": "A title",
            "is_oa": True,
            "work_type": "article",
            "unpaywall_pdf_url": None,
        }
        path = tmp_path / "10.1234_journal.abc_discovery.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = discovery._read_discovery_cache("10.1234/journal.abc")
        assert loaded is not None
        assert loaded.oa_candidates == []
        assert loaded.pmcid == "PMC6940144"
        assert loaded.oa_url == "https://a.example.org/landing"
