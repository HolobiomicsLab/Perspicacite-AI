"""ASB workflow-card parser. Covers both fixture sets:
- metlinkr_subset (2 cards, schema 0.17.0, github_name populated)
- article_878_v4_subset (3 cards, schema 0.17.0, github_name=None)

Both fixtures carry the 2026-05-16 rich schema (task_objective,
executable-as-dict, task_inputs/outputs, execution_profile, etc.).
The plan's "2026-05-15 bool executable" form is tested synthetically
via the tmp_path fixture.
"""
from pathlib import Path

METLINKR = Path(__file__).parent.parent / "fixtures" / "asb" / "metlinkr_subset"
ARTICLE = Path(__file__).parent.parent / "fixtures" / "asb" / "article_878_v4_subset"


# ---------- metlinkr_subset (2 cards) ----------

def test_parse_cards_finds_two_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = parse_cards(METLINKR)
    ids = {c.task_id for c in cards}
    assert ids == {"task_001", "task_002"}


def test_parse_card_extracts_tools_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    # metlinkr task_001 uses MetLinkR and R
    assert "MetLinkR" in t1.tools_used


def test_parse_card_extracts_skills_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    # skills field has at least one entry
    assert t1.skills_used
    assert any("metabolite" in s or "mapping" in s for s in t1.skills_used)


def test_parse_card_includes_body_text_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    body = cards["task_001"].body_markdown
    assert body
    assert "metabolite" in body.lower() or "identifier" in body.lower()


def test_parse_card_extracts_domain_facets_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    assert t1.domain == "mass-spectrometry / metabolomics"
    assert t1.primary_domain == "metabolomics"


def test_parse_card_extracts_evaluation_strategy_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    assert isinstance(t1.evaluation_strategy, dict)
    assert t1.evaluation_strategy  # non-empty


def test_parse_card_extracts_task_objective_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    assert t1.task_objective
    assert "RefMet" in t1.task_objective or "mapping" in t1.task_objective.lower()


def test_parse_card_executable_is_dict_metlinkr():
    """Both fixture schemas use executable as a structured dict."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    assert isinstance(t1.executable, dict)
    assert "cmd" in t1.executable


def test_parse_card_github_name_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    assert t1.github_name == "ncats/MetLinkR"


def test_parse_card_task_inputs_outputs_metlinkr():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(METLINKR)}
    t1 = cards["task_001"]
    assert isinstance(t1.task_inputs, list)
    assert len(t1.task_inputs) > 0
    assert isinstance(t1.task_outputs, list)
    assert len(t1.task_outputs) > 0


# ---------- article_878_v4_subset (3 cards) ----------

def test_parse_cards_finds_three_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = parse_cards(ARTICLE)
    ids = {c.task_id for c in cards}
    assert ids == {"task_001", "task_002", "task_003"}


def test_parse_card_extracts_task_objective_article():
    """task_objective must be captured."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert t1.task_objective  # non-empty
    assert "matchms" in t1.task_objective.lower() or "library" in t1.task_objective.lower()


def test_parse_card_executable_is_dict_in_article():
    """executable is a structured dict with cmd/env."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert isinstance(t1.executable, dict)
    assert "cmd" in t1.executable


def test_parse_card_captures_task_inputs_outputs_article():
    """Cards carry typed input/output port lists."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert isinstance(t1.task_inputs, list)
    assert len(t1.task_inputs) > 0
    assert isinstance(t1.task_outputs, list)
    assert len(t1.task_outputs) > 0


def test_parse_card_captures_execution_profile_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert isinstance(t1.execution_profile, dict)
    assert t1.execution_profile  # non-empty: has compute_tier etc.


def test_parse_card_captures_run_timeout_seconds_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert t1.run_timeout_seconds is not None
    assert t1.run_timeout_seconds > 0


def test_parse_card_captures_reproducibility_tier_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert t1.reproducibility_tier is not None


def test_parse_card_captures_linked_result_ids_article():
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards = {c.task_id: c for c in parse_cards(ARTICLE)}
    t1 = cards["task_001"]
    assert isinstance(t1.linked_result_ids, list)
    assert len(t1.linked_result_ids) > 0


# ---------- cross-schema compatibility ----------

def test_parse_card_bool_executable_is_dropped(tmp_path):
    """When executable is a bool (old schema), it is dropped (becomes None)."""
    import json

    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "task_001.json").write_text(
        json.dumps({"title": "t", "executable": True, "skills": [], "tools": []})
    )
    (cards_dir / "task_001.md").write_text("# Old schema card")
    cards = parse_cards(tmp_path)
    assert len(cards) == 1
    assert cards[0].executable is None  # bool form is dropped


def test_parse_card_skips_pair_with_missing_md(tmp_path):
    """If task_NNN.json exists without a matching .md, the card is skipped."""
    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "task_999.json").write_text("{}")
    # No task_999.md
    cards = parse_cards(tmp_path)
    assert all(c.task_id != "task_999" for c in cards)


def test_parse_cards_missing_dir_returns_empty(tmp_path):
    from perspicacite.pipeline.asb.card_parser import parse_cards
    assert parse_cards(tmp_path) == []


def test_parse_card_task_inputs_default_empty_when_absent(tmp_path):
    """When task_inputs/outputs are absent, they default to empty lists."""
    import json

    from perspicacite.pipeline.asb.card_parser import parse_cards
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "task_001.json").write_text(
        json.dumps({"title": "minimal", "skills": [], "tools": []})
    )
    (cards_dir / "task_001.md").write_text("# Minimal card")
    cards = parse_cards(tmp_path)
    assert cards[0].task_inputs == []
    assert cards[0].task_outputs == []
    assert cards[0].task_objective in (None, "")
