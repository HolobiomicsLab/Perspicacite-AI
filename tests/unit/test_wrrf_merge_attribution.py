"""WRRF neighbor merging must not mix papers under one citation.

Neighbors in ``select_wrrf_merged_documents`` are adjacent in WRRF *score*, not
inside the source document. Merging them unconditionally put another paper's
text under the current chunk's citation, which misattributes every quote drawn
from the merged context.
"""
from __future__ import annotations

from types import SimpleNamespace

from perspicacite.rag.wrrf_v1 import (
    doc_source_id,
    from_same_paper,
    select_wrrf_merged_documents,
)


def _doc(paper_id: str, text: str, citation: str | None = None):
    meta = SimpleNamespace(
        paper_id=paper_id,
        citation=citation or f"{paper_id} et al.",
        title=f"Title of {paper_id}",
        url="",
        source_type="",
    )
    return SimpleNamespace(chunk=SimpleNamespace(text=text, metadata=meta), score=0.5)


def _select(docs, final_max_docs=5, max_docs_per_source=5):
    sorted_docs = [(i, 1.0 - i / 100) for i in range(len(docs))]
    documents_info = dict(enumerate(docs))
    return select_wrrf_merged_documents(
        sorted_docs, documents_info, final_max_docs, max_docs_per_source
    )


class TestSourceIdentity:
    def test_paper_id_is_the_identity(self):
        assert doc_source_id(_doc("paperA", "x")) == "paperA"

    def test_unknown_identity_is_none(self):
        assert doc_source_id(SimpleNamespace()) is None
        assert doc_source_id(None) is None

    def test_unknown_identities_never_count_as_the_same_paper(self):
        blank = SimpleNamespace()
        assert from_same_paper(blank, blank) is False

    def test_same_paper_matches_and_different_papers_do_not(self):
        a1, a2, b1 = _doc("A", "1"), _doc("A", "2"), _doc("B", "1")
        assert from_same_paper(a1, a2) is True
        assert from_same_paper(a1, b1) is False


class TestMergeStaysWithinOnePaper:
    def test_neighbors_from_other_papers_are_not_merged_in(self):
        """The regression: B's merged text must not contain A's or C's words."""
        docs = [
            _doc("A", "ALPHA-ONLY-TEXT"),
            _doc("B", "BRAVO-ONLY-TEXT"),
            _doc("C", "CHARLIE-ONLY-TEXT"),
        ]
        selected = _select(docs)

        by_paper = {doc_source_id(s): s.chunk.text for s in selected}
        assert "ALPHA-ONLY-TEXT" not in by_paper["B"], (
            "merged context carries paper A's text under paper B's citation"
        )
        assert "CHARLIE-ONLY-TEXT" not in by_paper["B"]
        assert "BRAVO-ONLY-TEXT" in by_paper["B"]

    def test_same_paper_neighbors_are_still_merged(self):
        """The feature must survive the fix: adjacent chunks of one paper merge."""
        docs = [
            _doc("A", "FIRST-CHUNK"),
            _doc("A", "SECOND-CHUNK"),
            _doc("A", "THIRD-CHUNK"),
        ]
        selected = _select(docs, max_docs_per_source=5)

        middle = selected[1].chunk.text
        assert "FIRST-CHUNK" in middle
        assert "SECOND-CHUNK" in middle
        assert "THIRD-CHUNK" in middle

    def test_every_selected_chunk_only_contains_its_own_paper_text(self):
        docs = [
            _doc("A", "AAA"),
            _doc("B", "BBB"),
            _doc("A", "AAA2"),
            _doc("C", "CCC"),
        ]
        selected = _select(docs)

        foreign = {"A": ("BBB", "CCC"), "B": ("AAA", "AAA2", "CCC"), "C": ("AAA", "BBB", "AAA2")}
        for result in selected:
            paper = doc_source_id(result)
            for marker in foreign[paper]:
                assert marker not in result.chunk.text, (
                    f"{paper}'s merged text leaked {marker}"
                )
