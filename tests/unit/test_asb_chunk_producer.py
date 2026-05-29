"""ASB chunk producer: ParsedSkill | ParsedCard → Paper with metadata.

Papers carry PaperSource.SKILL_BUNDLE; metadata is propagated by the
existing chunker. IDs are stable so re-ingest is idempotent.
"""
from pathlib import Path

METLINKR = Path(__file__).parent.parent / "fixtures" / "asb" / "metlinkr_subset"
ARTICLE = Path(__file__).parent.parent / "fixtures" / "asb" / "article_878_v4_subset"


# ---------- skills ----------

def test_skill_becomes_paper_with_correct_metadata():
    from perspicacite.models.papers import PaperSource
    from perspicacite.pipeline.asb.chunk_producer import skill_to_paper
    from perspicacite.pipeline.asb.skill_parser import parse_skill_bundle

    skill = parse_skill_bundle(METLINKR)[0]
    paper = skill_to_paper(skill)
    assert paper.source is PaperSource.SKILL_BUNDLE
    assert paper.id == "asb_skill:cross-identifier-reconciliation"
    assert paper.full_text  # the skill.md body
    md = paper.metadata
    assert md["content_kind"] == "skill_body"
    assert md["skill_id"] == "cross-identifier-reconciliation"
    assert md["skill_name"]
    assert isinstance(md["tools"], list)
    assert isinstance(md["environment"], list)
    assert isinstance(md["parameters"], list)


def test_skill_to_paper_title_is_skill_name():
    from perspicacite.pipeline.asb.chunk_producer import skill_to_paper
    from perspicacite.pipeline.asb.skill_parser import parse_skill_bundle
    skill = parse_skill_bundle(METLINKR)[0]
    paper = skill_to_paper(skill)
    assert paper.title == skill.name


def test_skill_to_paper_abstract_is_description():
    from perspicacite.pipeline.asb.chunk_producer import skill_to_paper
    from perspicacite.pipeline.asb.skill_parser import parse_skill_bundle
    skill = parse_skill_bundle(METLINKR)[0]
    paper = skill_to_paper(skill)
    assert paper.abstract == skill.description


# ---------- cards (2026-05-15 metlinkr) ----------

def test_card_becomes_paper_with_workflow_metadata_metlinkr():
    from perspicacite.models.papers import PaperSource
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    card = next(c for c in parse_cards(METLINKR) if c.task_id == "task_001")
    paper = card_to_paper(card, dag=None)
    assert paper.source is PaperSource.SKILL_BUNDLE
    assert paper.id == "asb_card:task_001"
    md = paper.metadata
    assert md["content_kind"] == "workflow_card"
    assert md["task_id"] == "task_001"
    assert isinstance(md["tools_used"], list)
    # domain/primary_domain populated on real cards
    assert md["domain"] or md["primary_domain"]


def test_card_to_paper_attaches_dag_neighbors_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper
    from perspicacite.pipeline.asb.dag import load_workflow_dag

    dag = load_workflow_dag(METLINKR)
    card = next(c for c in parse_cards(METLINKR) if c.task_id == "task_001")
    paper = card_to_paper(card, dag=dag)
    md = paper.metadata
    assert "task_002" in md["downstream_tasks"]
    assert md["upstream_tasks"] == []


def test_card_to_paper_doi_propagated_from_crossref():
    """When the card has a DOI, surface it on Paper.doi for KB-side
    de-duplication / Zotero export."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    card = next(c for c in parse_cards(METLINKR) if c.task_id == "task_001")
    paper = card_to_paper(card, dag=None)
    if card.crossref_doi:
        assert paper.doi == card.crossref_doi


# ---------- cards (2026-05-16 article_878_v4) ----------

def test_card_to_paper_surfaces_executable_dict_article():
    """2026-05-16 cards have executable as a dict; metadata carries it through."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    card = cards["task_001"]
    paper = card_to_paper(card, dag=None)
    md = paper.metadata
    # executable field present (may be None or dict — propagated as-is)
    assert "executable" in md
    if card.executable is not None:
        assert md["executable"] == card.executable


def test_card_to_paper_surfaces_execution_profile_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    card = cards["task_001"]
    paper = card_to_paper(card, dag=None)
    md = paper.metadata
    assert isinstance(md["execution_profile"], dict)


def test_card_to_paper_surfaces_task_inputs_outputs_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    card = cards["task_001"]
    paper = card_to_paper(card, dag=None)
    md = paper.metadata
    assert isinstance(md["task_inputs"], list)
    assert isinstance(md["task_outputs"], list)


def test_card_to_paper_uses_task_objective_as_abstract_when_present():
    """2026-05-16: task_objective is the canonical abstract for cards."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    card = cards["task_001"]
    paper = card_to_paper(card, dag=None)
    if card.task_objective:
        assert paper.abstract == card.task_objective


def test_card_to_paper_paper_github_prefers_name_over_legacy():
    """2026-05-16: github_name supersedes github."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper

    for fixture in (METLINKR, ARTICLE):
        cards = {c.task_id: c for c in parse_cards(fixture)}
        card = cards["task_001"]
        paper = card_to_paper(card, dag=None)
        expected = card.github_name or card.github
        assert paper.metadata.get("paper_github") == expected


# ---------- idempotency ----------

def test_paper_ids_are_stable_for_idempotent_reingest():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    from perspicacite.pipeline.asb.chunk_producer import card_to_paper, skill_to_paper
    from perspicacite.pipeline.asb.skill_parser import parse_skill_bundle

    skill = parse_skill_bundle(METLINKR)[0]
    p1 = skill_to_paper(skill)
    p2 = skill_to_paper(skill)
    assert p1.id == p2.id

    card = parse_cards(METLINKR)[0]
    c1 = card_to_paper(card, dag=None)
    c2 = card_to_paper(card, dag=None)
    assert c1.id == c2.id
