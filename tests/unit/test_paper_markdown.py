"""Rendering a retrieved paper as markdown for an external builder."""
from dataclasses import dataclass, field

import pytest

from perspicacite.pipeline.paper_markdown import (
    MAX_REFERENCES,
    author_names,
    render_paper_markdown,
)


@dataclass
class _Result:
    full_text: str | None = None
    abstract: str | None = None
    sections: dict | None = None
    references: list | None = field(default_factory=list)
    content_source: str = "pmc"
    metadata: dict | None = field(default_factory=dict)


def test_empty_retrieval_is_refused():
    """A title-only stub would build a capsule that measures nothing."""
    with pytest.raises(ValueError):
        render_paper_markdown("10.1038/x", _Result(metadata={"title": "T"}))


def test_structured_sections_keep_their_headings():
    md = render_paper_markdown("10.1038/x", _Result(
        full_text="ignored",
        sections={"methods": "We did things.", "results": "It worked."},
    ))
    assert "## Methods" in md and "We did things." in md
    assert "## Results" in md and "It worked." in md


def test_flat_text_is_used_when_there_are_no_sections():
    md = render_paper_markdown("10.1038/x", _Result(full_text="Body."))
    assert "## Full text" in md and "Body." in md


def test_empty_sections_do_not_suppress_the_full_text():
    """A structured result whose sections are all blank still has a body."""
    md = render_paper_markdown("10.1038/x", _Result(
        full_text="Body.", sections={"methods": "   "}))
    assert "Body." in md


def test_abstract_is_included_alongside_the_body():
    md = render_paper_markdown("10.1038/x", _Result(
        abstract="Short.", full_text="Long."))
    assert "## Abstract" in md and "Short." in md and "Long." in md


def test_abstract_alone_is_enough_to_render():
    assert "Short." in render_paper_markdown("10.1038/x", _Result(abstract="Short."))


def test_provenance_is_stated_in_the_document():
    """Whoever reads the markdown should see how it was obtained."""
    md = render_paper_markdown("10.1038/x", _Result(
        full_text="B.", content_source="unpaywall_pdf"))
    assert "unpaywall_pdf" in md and "DOI: 10.1038/x" in md


def test_authors_may_be_strings_or_records():
    assert author_names({"authors": ["Ada L."]}) == "Ada L."
    assert author_names({"authors": [{"name": "Ada L."}]}) == "Ada L."
    assert author_names({"authors": [{}, None, "Bob"]}) == "Bob"


def test_reference_list_is_capped():
    refs = [{"title": f"Ref {i}"} for i in range(MAX_REFERENCES + 25)]
    md = render_paper_markdown("10.1038/x", _Result(full_text="B.", references=refs))
    assert "Ref 0" in md
    assert f"Ref {MAX_REFERENCES + 10}" not in md
    assert "25 further references omitted" in md


def test_no_reference_heading_when_there_are_none():
    md = render_paper_markdown("10.1038/x", _Result(full_text="B.", references=[]))
    assert "## References" not in md


def test_title_falls_back_to_the_doi():
    md = render_paper_markdown("10.1038/x", _Result(full_text="B."))
    assert md.startswith("# 10.1038/x")
