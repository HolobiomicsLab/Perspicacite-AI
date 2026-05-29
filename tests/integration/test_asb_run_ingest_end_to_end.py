"""End-to-end orchestrator test. KB layer + DOI ingest are mocked
at the orchestrator's seam points; the test asserts the flow:
parsers run → Papers built → KB receives Papers → skill_kb.json
updated → DAG stored on KB description.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

METLINKR = Path(__file__).parent.parent / "fixtures" / "asb" / "metlinkr_subset"
ARTICLE = Path(__file__).parent.parent / "fixtures" / "asb" / "article_878_v4_subset"


@pytest.mark.asyncio
async def test_ingest_asb_run_composite_mode_metlinkr(tmp_path):
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(METLINKR, target)

    added_papers: list = []
    fake_kb = MagicMock()
    fake_kb.name = "metlinkr_bundle"
    fake_kb.description = ""

    async def fake_add_papers(papers, **kw):
        added_papers.extend(papers)
        return len(papers)

    fake_kb.add_papers = fake_add_papers

    async def fake_make_or_get_kb(name, description="", **kw):
        fake_kb.name = name
        fake_kb.description = description
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": len(dois), "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name="metlinkr_bundle",
            include=("skills", "workflows"),
            mode="composite",
            app_state=None,
        )

    assert result["kb_names"] == ["metlinkr_bundle"]
    assert result["skills_ingested"] == 1
    assert result["workflows_ingested"] == 2
    assert result["papers_ingested"] >= 1
    paper_ids = {p.id for p in added_papers}
    assert "asb_skill:cross-identifier-reconciliation" in paper_ids
    assert "asb_card:task_001" in paper_ids
    assert "asb_card:task_002" in paper_ids
    # DAG stored on KB description (JSON-encoded under "workflow_dag" key)
    assert "workflow_dag" in (fake_kb.description or "")
    # skill_kb.json updated
    sk = json.loads(
        (target / "skills" / "cross-identifier-reconciliation"
         / "skill_kb.json").read_text()
    )
    assert sk["entries"]


@pytest.mark.asyncio
async def test_ingest_asb_run_composite_mode_article(tmp_path):
    """Same flow, 2026-05-16 fixture: 2 skills, 3 cards, dict-edge DAG."""
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(ARTICLE, target)

    added_papers: list = []
    fake_kb = MagicMock()
    fake_kb.name = "article_bundle"
    fake_kb.description = ""

    async def fake_add_papers(papers, **kw):
        added_papers.extend(papers)
        return len(papers)

    fake_kb.add_papers = fake_add_papers

    async def fake_make_or_get_kb(name, description="", **kw):
        fake_kb.name = name
        fake_kb.description = description
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": len(dois), "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name="article_bundle",
            include=("skills", "workflows"),
            mode="composite",
            app_state=None,
        )

    assert result["skills_ingested"] == 2
    assert result["workflows_ingested"] == 3
    paper_ids = {p.id for p in added_papers}
    assert "asb_skill:mass-spectral-library-curation" in paper_ids
    assert "asb_skill:chemical-structure-annotation-repair" in paper_ids
    assert "asb_card:task_001" in paper_ids
    assert "asb_card:task_002" in paper_ids
    assert "asb_card:task_003" in paper_ids


@pytest.mark.asyncio
async def test_ingest_asb_run_include_skills_only(tmp_path):
    """include=('skills',) skips workflow card ingestion."""
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(METLINKR, target)

    added_papers: list = []
    fake_kb = MagicMock()
    fake_kb.name = "kb"
    fake_kb.description = ""

    async def fake_add_papers(papers, **kw):
        added_papers.extend(papers)
        return len(papers)

    fake_kb.add_papers = fake_add_papers

    async def fake_make_or_get_kb(name, description="", **kw):
        fake_kb.name = name
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name="kb",
            include=("skills",),
            mode="composite",
            app_state=None,
        )

    assert result["workflows_ingested"] == 0
    paper_ids = {p.id for p in added_papers}
    assert not any(p.startswith("asb_card:") for p in paper_ids)


@pytest.mark.asyncio
async def test_ingest_asb_run_include_workflows_only(tmp_path):
    """include=('workflows',) skips skill ingestion."""
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(METLINKR, target)

    added_papers: list = []
    fake_kb = MagicMock()
    fake_kb.name = "kb"
    fake_kb.description = ""

    async def fake_add_papers(papers, **kw):
        added_papers.extend(papers)
        return len(papers)

    fake_kb.add_papers = fake_add_papers

    async def fake_make_or_get_kb(name, description="", **kw):
        fake_kb.name = name
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name="kb",
            include=("workflows",),
            mode="composite",
            app_state=None,
        )

    assert result["skills_ingested"] == 0
    paper_ids = {p.id for p in added_papers}
    assert not any(p.startswith("asb_skill:") for p in paper_ids)


@pytest.mark.asyncio
async def test_ingest_asb_run_per_skill_mode(tmp_path):
    """mode='per-skill' creates one KB per skill (plus composite for workflows)."""
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(ARTICLE, target)

    kbs_created: list[str] = []
    fake_kbs: dict = {}

    def make_fake_kb(name):
        fake = MagicMock()
        fake.name = name
        fake.description = ""
        fake.added_papers = []

        async def fake_add_papers(papers, **kw):
            fake.added_papers.extend(papers)
            return len(papers)

        fake.add_papers = fake_add_papers
        return fake

    async def fake_make_or_get_kb(name, description="", **kw):
        if name not in fake_kbs:
            fake_kbs[name] = make_fake_kb(name)
            kbs_created.append(name)
        fake_kbs[name].description = description
        return fake_kbs[name]

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    with patch(
        "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
        side_effect=fake_make_or_get_kb,
    ), patch(
        "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
        side_effect=fake_ingest_backing_dois,
    ):
        result = await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name="bundle",
            include=("skills", "workflows"),
            mode="per-skill",
            app_state=None,
        )

    # per-skill mode creates: composite "bundle" + one per skill
    assert "bundle" in result["kb_names"]
    assert any("__mass-spectral-library-curation" in n for n in result["kb_names"])
    assert any("__chemical-structure-annotation-repair" in n for n in result["kb_names"])


@pytest.mark.asyncio
async def test_ingest_asb_run_raises_on_embedding_conflict(tmp_path):
    """A pre-existing KB built with a *different* embedding model must
    raise :class:`EmbeddingModelConflictError` at the ASB orchestrator's
    KB-acquisition seam.

    Uses the real ``_make_or_get_kb`` path (no patch) so the production
    chain ``_make_or_get_kb → _create_kb_if_missing →
    check_embedding_compat_for_ingest`` runs.
    """
    import types

    from perspicacite.memory.session_store import SessionStore
    from perspicacite.models.kb import (
        ChunkConfig,
        KnowledgeBase,
        chroma_collection_name_for_kb,
    )
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run
    from perspicacite.rag.kb_compat import EmbeddingModelConflictError

    target = tmp_path / "run"
    shutil.copytree(METLINKR, target)

    # Stand up a real session_store with a KB that was built under
    # a *different* embedding model than the one we'll configure on
    # the stub app_state. The shape ``check_embedding_compat_for_ingest``
    # reads is ``kb_meta.embedding_model`` vs
    # ``app_state.embedding_provider.model_name``.
    ss = SessionStore(db_path=str(tmp_path / "session.db"))
    await ss.init_db()
    pre_existing_name = "asb_conflict_kb"
    pre_existing = KnowledgeBase(
        name=pre_existing_name,
        description="pre-existing — model-A",
        collection_name=chroma_collection_name_for_kb(pre_existing_name),
        embedding_model="model-A",
        chunk_config=ChunkConfig(),
    )
    await ss.save_kb_metadata(pre_existing)

    # Stub app_state — only the attributes the production chain touches
    # before the check are exercised. Anything farther downstream
    # (vector_store / embedding_service) is never reached because the
    # check raises first.
    stub_embedder = types.SimpleNamespace(model_name="model-B")
    stub_app_state = types.SimpleNamespace(
        session_store=ss,
        embedding_provider=stub_embedder,
    )

    with pytest.raises(EmbeddingModelConflictError) as excinfo:
        await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name=pre_existing_name,
            include=("skills", "workflows"),
            mode="composite",
            app_state=stub_app_state,
        )

    err = excinfo.value
    assert err.kb_name == pre_existing_name
    assert err.existing_model == "model-A"
    assert err.attempted_model == "model-B"


@pytest.mark.asyncio
async def test_ingest_asb_run_validates_include(tmp_path):
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run
    with pytest.raises(ValueError, match="include"):
        await ingest_asb_run(
            asb_run_dir=str(tmp_path),
            include=(),
            mode="composite",
            app_state=None,
        )


@pytest.mark.asyncio
async def test_ingest_asb_run_validates_mode(tmp_path):
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run
    with pytest.raises(ValueError, match="mode"):
        await ingest_asb_run(
            asb_run_dir=str(tmp_path),
            include=("skills",),
            mode="bogus",
            app_state=None,
        )


# ---------------------------------------------------------------------------
# Live end-to-end (gated)
# ---------------------------------------------------------------------------

import os


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("PERSPICACITE_E2E_ASB") != "1",
    reason="Set PERSPICACITE_E2E_ASB=1 to run the live ingest test",
)
async def test_ingest_asb_run_against_real_chroma(tmp_path):
    """Full pipeline against the real chunker + Chroma. Gated so it
    doesn't run in CI by default (requires the embedding provider
    env variables + a chroma backend reachable from this process).
    """
    from perspicacite.web.app_state import AppState

    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(METLINKR, target)

    # Boot the real AppState — same path the CLI uses.
    app_state = AppState()
    await app_state.initialize()

    try:
        result = await ingest_asb_run(
            asb_run_dir=str(target),
            kb_name="asb_e2e_test",
            include=("skills", "workflows"),
            mode="composite",
            update_skill_kb_json=True,
            app_state=app_state,
        )
    finally:
        shutdown = getattr(app_state, "shutdown", None)
        if shutdown is not None:
            if hasattr(shutdown, "__await__"):
                await shutdown()
            else:
                shutdown()

    assert result["skills_ingested"] == 1
    assert result["workflows_ingested"] == 2
    # papers_ingested counts skill body + each card; backing-paper DOIs
    # ingest through a separate path with its own counter
    assert result["papers_ingested"] >= 3
    assert result["workflow_dag"]
    assert result["workflow_dag"]["nodes"]


# ---------------------------------------------------------------------------
# Response payload regression
# ---------------------------------------------------------------------------


def test_chat_response_includes_asb_blocks_when_chunks_match():
    """If the chat/MCP response builder receives chunks with
    content_kind=skill_* or workflow_card, the response carries
    skill_metadata / workflow_metadata blocks via the helper from D10."""
    from perspicacite.pipeline.asb.response import build_asb_response_metadata

    chunks = [
        {"metadata": {
            "content_kind": "skill_body", "skill_id": "x",
            "skill_name": "X", "tools": [], "environment": [], "parameters": [],
        }},
        {"metadata": {
            "content_kind": "workflow_card", "task_id": "t1",
            "task_card_title": "T1",
            "task_objective": "Do the thing",
            "executable": {"cmd": ["bash", "-c", "echo hi"]},
            "execution_profile": {"compute_tier": "fast"},
        }},
        {"metadata": {"content_kind": "literature_chunk"}},  # unrelated — ignored
    ]
    out = build_asb_response_metadata(chunks)
    assert len(out["skill_metadata"]) == 1
    assert out["skill_metadata"][0]["skill_id"] == "x"
    assert len(out["workflow_metadata"]) == 1
    assert out["workflow_metadata"][0]["task_id"] == "t1"
    # 2026-05-16 fields surface
    assert out["workflow_metadata"][0]["task_objective"] == "Do the thing"
    assert out["workflow_metadata"][0]["executable"] == {"cmd": ["bash", "-c", "echo hi"]}


def test_end_to_end_orchestrator_output_feeds_response_metadata(tmp_path):
    """Smoke: an orchestrator pass produces Papers whose metadata
    would, after retrieval, feed build_asb_response_metadata correctly."""
    import asyncio
    from unittest.mock import MagicMock

    from perspicacite.pipeline.asb.response import build_asb_response_metadata
    from perspicacite.pipeline.asb.run_ingest import ingest_asb_run

    target = tmp_path / "run"
    shutil.copytree(METLINKR, target)

    added_papers: list = []
    fake_kb = MagicMock()
    fake_kb.name = "kb"
    fake_kb.description = ""

    async def fake_add_papers(papers, **kw):
        added_papers.extend(papers)
        return len(papers)

    fake_kb.add_papers = fake_add_papers

    async def fake_make_or_get_kb(name, description="", **kw):
        fake_kb.name = name
        fake_kb.description = description
        return fake_kb

    async def fake_ingest_backing_dois(*, kb, dois, app_state):
        return {"added": 0, "failed": []}

    async def run():
        with patch(
            "perspicacite.pipeline.asb.run_ingest._make_or_get_kb",
            side_effect=fake_make_or_get_kb,
        ), patch(
            "perspicacite.pipeline.asb.run_ingest._ingest_backing_paper_dois",
            side_effect=fake_ingest_backing_dois,
        ):
            await ingest_asb_run(
                asb_run_dir=str(target),
                kb_name="kb",
                include=("skills", "workflows"),
                mode="composite",
                app_state=None,
            )

    asyncio.run(run())

    # Simulate the retrieval layer surfacing chunks: each Paper.metadata
    # becomes a chunk metadata dict for our purposes here.
    chunks = [{"metadata": p.metadata} for p in added_papers]
    out = build_asb_response_metadata(chunks)
    skill_ids = {s["skill_id"] for s in out["skill_metadata"]}
    task_ids = {w["task_id"] for w in out["workflow_metadata"]}
    assert "cross-identifier-reconciliation" in skill_ids
    assert {"task_001", "task_002"}.issubset(task_ids)
