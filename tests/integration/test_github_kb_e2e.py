"""End-to-end integration tests for ``perspicacite.pipeline.github_kb``.

Covers the top-level orchestrator entry points introduced by Task 5 of
the 2026-05-15 GitHub / skill-bundle ingest plan:

* ``ingest_github_repo`` — fetch a repo, walk it, add Papers to a KB.
* ``ingest_skill_bundle`` — parse bundle.yml, walk + chunk, optionally
  route linked DOIs through ``ingest_dois_into_kb``.
* ``ingest_skill_bundles_batch`` — repeat per subdir (per-skill mode)
  or aggregate into a single composite KB.

Both ``ingest_dois_into_kb`` and the ``GitHubFetcher`` are seam-points
patched here so the tests stay fully offline. Real Chroma is used via
the top-level ``deterministic_embedder`` fixture (defined in
``tests/conftest.py``) so KB writes survive the round trip.

See:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` Task 5
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip whole file cleanly if Chroma isn't available. We importorskip
# *before* importing the orchestrator so a Chroma-less environment
# short-circuits cleanly rather than failing inside dynamic_kb's
# transitive imports.
chromadb = pytest.importorskip("chromadb")

from perspicacite.config.schema import Config
from perspicacite.pipeline.github.fetcher import GitHubFetcher

SAMPLE_BUNDLE = Path(__file__).parent.parent / "data" / "sample_bundle"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chroma_store(tmp_path: Path, embedder):
    from perspicacite.retrieval.chroma_store import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_provider=embedder,
    )


async def _make_session_store(tmp_path: Path):
    from perspicacite.memory.session_store import SessionStore

    store = SessionStore(db_path=str(tmp_path / "session.db"))
    await store.init_db()
    return store


def _make_config() -> Config:
    """A Config with defaults. The orchestrator only touches the
    ``bundles`` and ``knowledge_base`` sub-configs in v1."""
    return Config()


# ---------------------------------------------------------------------------
# ingest_skill_bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_skill_bundle_per_skill_mode(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """A local bundle path → per-skill KB created, files chunked + added,
    linked DOIs routed through ingest_dois_into_kb."""
    from perspicacite.pipeline import github_kb

    captured_calls: list[dict] = []

    async def fake_ingest_dois(app_state, kb_name, dois, **kw):
        captured_calls.append({"kb_name": kb_name, "dois": list(dois), "kw": kw})
        return {
            "added_papers": len(dois),
            "added_chunks": len(dois),
            "skipped_duplicates": 0,
            "failed": [],
        }

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    # Stub out the external-id resolvers so this test stays offline AND
    # preserves its original "arxiv/pmc surface as skipped" semantics.
    # The dedicated resolution test below patches them to return DOIs.
    async def _no_resolve(_id: str, *, client=None):
        return None

    monkeypatch.setattr(github_kb, "resolve_arxiv_to_doi", _no_resolve)
    monkeypatch.setattr(github_kb, "resolve_pmc_to_doi", _no_resolve)

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name="test_bundle_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=True,
        app_state_for_doi_ingest=MagicMock(),  # only forwarded to the patched ingest_dois
    )

    assert summary.bundle_name == "sample-bundle"
    assert summary.kb_name == "test_bundle_kb"
    assert summary.mode == "per-skill"
    # README + intro + qc.ipynb + qc.py = 4 papers; the producer also
    # ingests bundle.yml under default globs (yaml/yml). Spec asks for >= 3.
    assert summary.files_added >= 3
    assert summary.chunks_added > 0
    # 4 DOIs survive routing: 2 YAML DOIs + 2 README inline DOIs.
    assert summary.linked_papers_added == 4
    # arxiv + PMC mentioned but not routed (v1 only routes DOIs).
    skipped_kinds = {kind for kind, _ in summary.linked_papers_skipped_non_doi}
    assert skipped_kinds == {"arxiv", "pmc"}

    # Exactly one ingest_dois_into_kb call, targeting our KB, with the
    # exact dedup'd DOI set.
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["kb_name"] == "test_bundle_kb"
    assert set(call["dois"]) == {
        "10.1234/yaml-paper-1",
        "10.1234/yaml-paper-2",
        "10.5678/readme-paper-1",
        "10.5678/readme-paper-2",
    }


@pytest.mark.asyncio
async def test_ingest_skill_bundle_without_linked_papers(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """ingest_linked_papers=False → ingest_dois_into_kb is NOT called."""
    from perspicacite.pipeline import github_kb

    fake = AsyncMock()
    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake)

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name="no_links_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )
    assert summary.linked_papers_added == 0
    assert summary.linked_papers_skipped_non_doi == []
    fake.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_skill_bundle_kb_name_from_template(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """When kb_name is None, derive from bundles.default_kb_name_template."""
    from perspicacite.pipeline import github_kb

    async def fake_ingest_dois(*a, **kw):
        return {
            "added_papers": 0,
            "added_chunks": 0,
            "skipped_duplicates": 0,
            "failed": [],
        }

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    cfg = _make_config()
    cfg.bundles.default_kb_name_template = "kb-from-{name}"
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name=None,
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )
    # bundle name is "sample-bundle".
    assert summary.kb_name == "kb-from-sample-bundle"


@pytest.mark.asyncio
async def test_ingest_skill_bundle_requires_app_state_when_linking_papers(
    tmp_path: Path, deterministic_embedder
):
    """ingest_linked_papers=True without app_state_for_doi_ingest → ValueError."""
    from perspicacite.pipeline import github_kb

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    with pytest.raises(ValueError, match="app_state_for_doi_ingest"):
        await github_kb.ingest_skill_bundle(
            source=SAMPLE_BUNDLE,
            kb_name="x",
            config=cfg,
            vector_store=vs,
            embedding_service=deterministic_embedder,
            session_store=ss,
            ingest_linked_papers=True,
            app_state_for_doi_ingest=None,
        )


# ---------------------------------------------------------------------------
# ingest_github_repo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_github_repo_minimal(
    tmp_path: Path, deterministic_embedder
):
    """Mock the GitHubFetcher to point at the sample_bundle dir; assert
    the orchestrator wires SHA + repo metadata into the summary."""
    from perspicacite.pipeline import github_kb

    fake_fetcher = MagicMock(spec=GitHubFetcher)
    fake_fetcher.fetch = AsyncMock(
        return_value=(SAMPLE_BUNDLE, "abcdef1234567890")
    )

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_github_repo(
        url="https://github.com/example/sample-bundle",
        kb_name="repo_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        fetcher=fake_fetcher,
    )

    assert summary.kb_name == "repo_kb"
    assert summary.bundle_name is None  # raw-repo mode
    assert summary.mode == "repo"
    assert summary.repo_org == "example"
    assert summary.repo_name == "sample-bundle"
    assert summary.commit_sha == "abcdef1234567890"
    assert summary.files_added >= 3
    assert summary.chunks_added > 0
    # ingest_github_repo never auto-routes linked papers (per spec).
    assert summary.linked_papers_added == 0
    assert summary.linked_papers_skipped_non_doi == []
    fake_fetcher.fetch.assert_awaited_once()


# ---------------------------------------------------------------------------
# ingest_skill_bundles_batch
# ---------------------------------------------------------------------------


def _materialise_two_bundles(root: Path) -> None:
    """Stamp two minimal bundles under ``root`` for the batch tests."""
    for name in ("bundle-a", "bundle-b"):
        sub = root / name
        sub.mkdir(parents=True, exist_ok=True)
        # Each bundle gets a unique YAML name so the per-skill KBs differ.
        (sub / "bundle.yml").write_text(
            f"name: {name}\n"
            "papers:\n"
            f"  - doi: '10.1234/{name}-paper'\n"
        )
        (sub / "README.md").write_text(f"# {name}\n\nIntro.\n")


@pytest.mark.asyncio
async def test_ingest_skill_bundle_resolves_arxiv_pmc_to_doi(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """arXiv + PMC refs in the bundle are resolved to DOIs upstream and
    routed through the existing DOI ingest path.

    Setup uses the sample bundle, which carries:
      - 4 native DOIs (2 YAML + 2 README inline)
      - 1 arXiv id (``2204.12345`` in YAML)
      - 1 PMC id (``PMC9123456`` in YAML)

    The two resolvers are patched on :mod:`perspicacite.pipeline.github_kb`
    so the test stays offline. We assert the resolved DOIs land in
    :func:`ingest_dois_into_kb`'s captured call, the summary's new
    ``linked_papers_resolved_via_external_id`` field counts them, and
    ``linked_papers_skipped_non_doi`` is left with only the
    unresolvable refs.
    """
    from perspicacite.pipeline import github_kb

    captured_calls: list[dict] = []

    async def fake_ingest_dois(app_state, kb_name, dois, **kw):
        captured_calls.append({"kb_name": kb_name, "dois": list(dois), "kw": kw})
        return {
            "added_papers": len(dois),
            "added_chunks": len(dois),
            "skipped_duplicates": 0,
            "failed": [],
        }

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    async def fake_resolve_arxiv(arxiv_id: str, *, client=None):
        # Only resolve the known sample-bundle arXiv id; anything else
        # falls through to "unresolvable" so the skipped list still has
        # something to assert on if the sample expands.
        if arxiv_id == "2204.12345":
            return "10.9999/resolved-from-arxiv"
        return None

    async def fake_resolve_pmc(pmc_id: str, *, client=None):
        if pmc_id == "PMC9123456":
            return "10.9999/resolved-from-pmc"
        return None

    monkeypatch.setattr(github_kb, "resolve_arxiv_to_doi", fake_resolve_arxiv)
    monkeypatch.setattr(github_kb, "resolve_pmc_to_doi", fake_resolve_pmc)

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name="resolved_ids_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=True,
        app_state_for_doi_ingest=MagicMock(),
    )

    # Two refs resolved (1 arxiv + 1 pmc).
    assert summary.linked_papers_resolved_via_external_id == 2
    # All 4 original DOIs + 2 resolved DOIs = 6 routed.
    assert summary.linked_papers_added == 6
    # Nothing unresolvable in the sample bundle → empty skip list.
    assert summary.linked_papers_skipped_non_doi == []

    # The two resolved DOIs must have landed in the captured ingest call.
    assert len(captured_calls) == 1
    call_dois = set(captured_calls[0]["dois"])
    assert "10.9999/resolved-from-arxiv" in call_dois
    assert "10.9999/resolved-from-pmc" in call_dois
    # And the 4 original DOIs are still present (no eviction).
    assert {
        "10.1234/yaml-paper-1",
        "10.1234/yaml-paper-2",
        "10.5678/readme-paper-1",
        "10.5678/readme-paper-2",
    } <= call_dois


@pytest.mark.asyncio
async def test_ingest_skill_bundles_batch_per_skill_mode(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    from perspicacite.pipeline import github_kb

    root = tmp_path / "bundles_root"
    _materialise_two_bundles(root)

    async def fake_ingest_dois(*a, **kw):
        return {
            "added_papers": 0,
            "added_chunks": 0,
            "skipped_duplicates": 0,
            "failed": [],
        }

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summaries = await github_kb.ingest_skill_bundles_batch(
        root=root,
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        composite_kb=None,
        ingest_linked_papers=False,
    )
    assert len(summaries) == 2
    kb_names = sorted(s.kb_name for s in summaries)
    assert kb_names == ["bundle-a", "bundle-b"]
    for s in summaries:
        assert s.mode == "per-skill"
        assert s.bundle_name in {"bundle-a", "bundle-b"}


# ---------------------------------------------------------------------------
# external_link KB-log events (Task 9)
# ---------------------------------------------------------------------------


def _read_external_link_events(log_path: Path) -> list[dict]:
    """Read the KB-log JSONL and return all ``external_link`` event payloads."""
    if not log_path.exists():
        return []
    events: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("event") == "external_link":
            events.append(payload)
    return events


@pytest.mark.asyncio
async def test_external_links_emitted_to_kb_log(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """Sample bundle's README references a figshare dataset + a github
    tool URL → at least one ``external_link`` event lands in the KB log."""
    from perspicacite.pipeline import github_kb

    async def fake_ingest_dois(*a, **kw):
        return {"added_papers": 0, "added_chunks": 0,
                "skipped_duplicates": 0, "failed": []}

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    cfg = _make_config()
    cfg.knowledge_base.log_dir = tmp_path / "kb_logs"
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name="ext_link_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )
    assert summary.external_links_logged >= 1

    log_path = cfg.knowledge_base.log_dir / "ext_link_kb.jsonl"
    events = _read_external_link_events(log_path)
    assert len(events) >= 1


@pytest.mark.asyncio
async def test_external_link_event_carries_url_in_extra(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """Each ``external_link`` event's ``extra`` dict has a ``url`` key."""
    from perspicacite.pipeline import github_kb

    async def fake_ingest_dois(*a, **kw):
        return {"added_papers": 0, "added_chunks": 0,
                "skipped_duplicates": 0, "failed": []}

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    cfg = _make_config()
    cfg.knowledge_base.log_dir = tmp_path / "kb_logs"
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name="ext_link_url_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )
    log_path = cfg.knowledge_base.log_dir / "ext_link_url_kb.jsonl"
    events = _read_external_link_events(log_path)
    assert events  # precondition for the carry-url assertion
    for ev in events:
        assert "url" in ev["extra"]
        assert ev["extra"]["url"].startswith(("http://", "https://"))
        assert ev["extra"]["category"] in {"dataset", "tool"}


@pytest.mark.asyncio
async def test_ingest_summary_external_links_logged_counts_correctly(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """The summary's ``external_links_logged`` matches the distinct URLs
    mined from the bundle's README + docs."""
    from perspicacite.pipeline import github_kb
    from perspicacite.pipeline.github.bundle import BundleManifest

    async def fake_ingest_dois(*a, **kw):
        return {"added_papers": 0, "added_chunks": 0,
                "skipped_duplicates": 0, "failed": []}

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    cfg = _make_config()
    cfg.knowledge_base.log_dir = tmp_path / "kb_logs"
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name="count_ext_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )
    # Compute the expected count straight from the orchestrator's
    # collection method so the test stays self-consistent with the
    # mining logic.
    manifest = BundleManifest.from_directory(SAMPLE_BUNDLE)
    bag = manifest.collect_external_links()
    expected = len(bag.datasets) + len(bag.tools)
    assert summary.external_links_logged == expected


@pytest.mark.asyncio
async def test_no_external_link_events_when_readme_only_bundle(
    tmp_path: Path, deterministic_embedder
):
    """A README-only bundle with no URLs → 0 external_link events +
    ``external_links_logged == 0``."""
    from perspicacite.pipeline import github_kb

    bundle = tmp_path / "url-free"
    bundle.mkdir()
    (bundle / "README.md").write_text(
        "# url-free\n\nPure prose, no URLs.\n", encoding="utf-8"
    )

    cfg = _make_config()
    cfg.knowledge_base.log_dir = tmp_path / "kb_logs"
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summary = await github_kb.ingest_skill_bundle(
        source=bundle,
        kb_name="urlfree_kb",
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )
    assert summary.external_links_logged == 0

    log_path = cfg.knowledge_base.log_dir / "urlfree_kb.jsonl"
    events = _read_external_link_events(log_path)
    assert events == []


# ---------------------------------------------------------------------------
# ingest_skill_bundles_batch (continued)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_skill_bundles_batch_composite_mode(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    from perspicacite.pipeline import github_kb

    root = tmp_path / "bundles_root"
    _materialise_two_bundles(root)

    async def fake_ingest_dois(*a, **kw):
        return {
            "added_papers": 0,
            "added_chunks": 0,
            "skipped_duplicates": 0,
            "failed": [],
        }

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", fake_ingest_dois)

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    ss = await _make_session_store(tmp_path)

    summaries = await github_kb.ingest_skill_bundles_batch(
        root=root,
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        composite_kb="my-composite",
        ingest_linked_papers=False,
    )
    assert len(summaries) == 2
    for s in summaries:
        assert s.kb_name == "my-composite"
        assert s.mode == "composite"


# ---------------------------------------------------------------------------
# Embedding-model conflict detection (2026-05-16 P1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_skill_bundle_raises_on_embedding_conflict(
    tmp_path: Path, deterministic_embedder, monkeypatch
):
    """A pre-existing KB built with a *different* embedding model must
    raise :class:`EmbeddingModelConflictError` before any new chunks
    are written. The check lives in ``github_kb._add_papers_to_kb``.
    """
    from perspicacite.models.kb import (
        ChunkConfig,
        KnowledgeBase,
        chroma_collection_name_for_kb,
    )
    from perspicacite.pipeline import github_kb
    from perspicacite.rag.kb_compat import EmbeddingModelConflictError

    # Pre-stamp a KB record under the target name with a *different*
    # embedding model than the deterministic embedder reports.
    ss = await _make_session_store(tmp_path)
    target_kb_name = "conflict_kb"
    pre_existing = KnowledgeBase(
        name=target_kb_name,
        description="pre-existing — embedded with model-A",
        collection_name=chroma_collection_name_for_kb(target_kb_name),
        embedding_model="model-A",  # NOT "deterministic-mock"
        chunk_config=ChunkConfig(),
    )
    await ss.save_kb_metadata(pre_existing)

    # Avoid touching the DOI ingest path; the conflict must trip before
    # we get anywhere near it.
    async def _no_ingest(*a, **kw):
        raise AssertionError(
            "linked-paper ingest should not run when the embedding "
            "conflict trips first"
        )

    monkeypatch.setattr(github_kb, "ingest_dois_into_kb", _no_ingest)

    cfg = _make_config()
    vs = _make_chroma_store(tmp_path, deterministic_embedder)

    with pytest.raises(EmbeddingModelConflictError) as excinfo:
        await github_kb.ingest_skill_bundle(
            source=SAMPLE_BUNDLE,
            kb_name=target_kb_name,
            config=cfg,
            vector_store=vs,
            embedding_service=deterministic_embedder,
            session_store=ss,
            ingest_linked_papers=False,
        )

    err = excinfo.value
    assert err.kb_name == target_kb_name
    assert err.existing_model == "model-A"
    assert err.attempted_model == deterministic_embedder.model_name


@pytest.mark.asyncio
async def test_ingest_skill_bundle_ok_when_models_match(
    tmp_path: Path, deterministic_embedder
):
    """Sanity check: when the pre-existing KB's embedding model matches
    the current service, ingest proceeds normally (no false-positive
    conflict)."""
    from perspicacite.models.kb import (
        ChunkConfig,
        KnowledgeBase,
        chroma_collection_name_for_kb,
    )
    from perspicacite.pipeline import github_kb

    ss = await _make_session_store(tmp_path)
    target_kb_name = "matched_kb"
    pre_existing = KnowledgeBase(
        name=target_kb_name,
        description="pre-existing — same model",
        collection_name=chroma_collection_name_for_kb(target_kb_name),
        embedding_model=deterministic_embedder.model_name,
        chunk_config=ChunkConfig(),
    )
    await ss.save_kb_metadata(pre_existing)
    # The Chroma collection has to exist for the dynamic_kb add path,
    # so create it explicitly to match the pre-stamped metadata.
    vs = _make_chroma_store(tmp_path, deterministic_embedder)
    await vs.create_collection(pre_existing.collection_name)

    cfg = _make_config()

    summary = await github_kb.ingest_skill_bundle(
        source=SAMPLE_BUNDLE,
        kb_name=target_kb_name,
        config=cfg,
        vector_store=vs,
        embedding_service=deterministic_embedder,
        session_store=ss,
        ingest_linked_papers=False,
    )

    assert summary.kb_name == target_kb_name
    assert summary.chunks_added > 0
