"""Tests for the BibTeX → paper-dict resolver in push_to_zotero.

Two live-discovered bugs the resolver had to start handling correctly:

- BibTeX case-preservation braces (``{LLM}``, ``The {Evolving} {Role}``) were
  appearing verbatim in Zotero titles. Only the outermost ``{}`` were
  stripped, leaving inner braces. Fix: recursively strip ``{...}`` pairs.
- Entries with both ``url=arxiv.org/abs/<id>`` and ``eprint=<id>`` (the
  format ``arxiv`` writes when you click "Cite") fell through to the URL
  route → ``item_type=webpage``, no DOI, no PDF fetch. Fix: synthesize
  ``10.48550/arXiv.<id>`` from ``eprint``/``archivePrefix=arXiv`` or from
  the arxiv-shaped ``url``, so the entry routes through the DOI path.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from perspicacite.mcp.server import _resolve_push_input


@pytest.mark.asyncio
async def test_bibtex_strips_nested_case_preservation_braces():
    """``The {Evolving} {Role} of {LLM}`` → ``The Evolving Role of LLM``."""
    bib = """
@article{example2025,
  title={{The {Evolving} {Role} of {Large} {Language} {Models}: {Evaluator}, {Collaborator}, and {Scientist}}},
  author={Smith, John and Doe, Jane},
  year={2025},
  doi={10.1234/abc}
}
"""
    fake_content = AsyncMock()
    fake_content.return_value.metadata = {}
    fake_content.return_value.abstract = ""
    async with httpx.AsyncClient() as http:
        with patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=fake_content,
        ):
            paper, _doi, _url = await _resolve_push_input(
                {"bibtex": bib}, http_client=http,
            )
    assert "{" not in paper["title"]
    assert "}" not in paper["title"]
    assert paper["title"] == (
        "The Evolving Role of Large Language Models: "
        "Evaluator, Collaborator, and Scientist"
    )


@pytest.mark.asyncio
async def test_bibtex_synthesizes_arxiv_doi_from_eprint():
    """A misc/article entry with ``eprint`` + ``archivePrefix=arXiv`` and a
    parallel ``url`` field should route via DOI ``10.48550/arXiv.<id>``,
    not via the URL → webpage path."""
    bib = """
@misc{zhou2025autonomousagentsscientificdiscovery,
  title={{Autonomous Agents for Scientific Discovery}},
  author={Zhou, Lianhao and Ling, Hongyi},
  year={2025},
  eprint={2510.09901},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2510.09901}
}
"""
    fake_content = AsyncMock()
    fake_content.return_value.metadata = {"title": "x"}
    fake_content.return_value.abstract = ""
    async with httpx.AsyncClient() as http:
        with patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=fake_content,
        ):
            _paper, doi, _url = await _resolve_push_input(
                {"bibtex": bib}, http_client=http,
            )
    assert doi == "10.48550/arXiv.2510.09901"


@pytest.mark.asyncio
async def test_bibtex_synthesizes_arxiv_doi_from_arxiv_url_only():
    """When the bib has only ``url=arxiv.org/abs/<id>`` (no ``eprint``),
    the resolver should still build a DOI from the URL."""
    bib = """
@misc{paperbench2025,
  title={{PaperBench}},
  author={Starace, Giulio},
  year={2025},
  url={https://arxiv.org/abs/2504.01848}
}
"""
    fake_content = AsyncMock()
    fake_content.return_value.metadata = {"title": "x"}
    fake_content.return_value.abstract = ""
    async with httpx.AsyncClient() as http:
        with patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=fake_content,
        ):
            _paper, doi, _url = await _resolve_push_input(
                {"bibtex": bib}, http_client=http,
            )
    assert doi == "10.48550/arXiv.2504.01848"


@pytest.mark.asyncio
async def test_bibtex_no_doi_no_url_resolves_via_title(respx_mock):
    """A ``@misc`` with only title + author + year (no DOI, no URL, no
    eprint) should trigger the title resolver and route through the
    DOI path instead of erroring out."""
    bib = """
@misc{vaswani2017,
  title={{Attention Is All You Need}},
  author={Vaswani, Ashish and Shazeer, Noam},
  year={2017}
}
"""
    # OpenAlex returns a clean hit
    respx_mock.get(url__regex=r"https://api\.openalex\.org/works.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Attention Is All You Need",
                        "publication_year": 2017,
                        "doi": "https://doi.org/10.48550/arXiv.1706.03762",
                        "authorships": [
                            {"author": {"display_name": "Ashish Vaswani"}},
                        ],
                    }
                ]
            },
        )
    )
    fake_content = AsyncMock()
    fake_content.return_value.metadata = {"title": "x"}
    fake_content.return_value.abstract = ""
    async with httpx.AsyncClient() as http:
        with patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=fake_content,
        ):
            _paper, doi, _url = await _resolve_push_input(
                {"bibtex": bib}, http_client=http,
            )
    assert doi == "10.48550/arXiv.1706.03762"


@pytest.mark.asyncio
async def test_bibtex_explicit_doi_wins_over_arxiv_synthesis():
    """If the bib has both an explicit ``doi`` and an ``eprint``, the
    explicit ``doi`` is authoritative — it might point to the published
    journal version, not the preprint."""
    bib = """
@article{example,
  title={{Example}},
  doi={10.1038/s41586-024-12345-6},
  eprint={2401.00001},
  archivePrefix={arXiv}
}
"""
    fake_content = AsyncMock()
    fake_content.return_value.metadata = {"title": "x"}
    fake_content.return_value.abstract = ""
    async with httpx.AsyncClient() as http:
        with patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=fake_content,
        ):
            _paper, doi, _url = await _resolve_push_input(
                {"bibtex": bib}, http_client=http,
            )
    assert doi == "10.1038/s41586-024-12345-6"
