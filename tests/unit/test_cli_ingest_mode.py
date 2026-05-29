"""Tests for --ingest-mode CLI flag on create-kb and add-to-kb commands."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _make_bib(tmp_path: Path) -> Path:
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{a, title={Test Paper}, year={2024}}\n")
    return bib


def test_add_to_kb_ingest_mode_overrides_config(tmp_path):
    """--ingest-mode abstract_only sets config.knowledge_base.ingest_mode before pipeline call."""
    from click.testing import CliRunner

    from perspicacite.cli import cli

    bib = _make_bib(tmp_path)
    captured: dict = {}

    async def fake_add_bibtex(config, kb_name, bib_path, session_db, chroma_dir):
        captured["mode"] = config.knowledge_base.ingest_mode
        return {
            "new_papers": 0,
            "chunks_added": 0,
            "pdf_stats": {"attempted": 0, "success": 0, "failed": 0, "skipped_no_doi": 0},
        }

    with patch("perspicacite.cli._add_bibtex_to_existing_kb", new=fake_add_bibtex):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", "config.example.yml",
                "add-to-kb", "mytest",
                "--from-bibtex", str(bib),
                "--ingest-mode", "abstract_only",
            ],
        )

    assert captured.get("mode") == "abstract_only", (
        f"Expected 'abstract_only', got {captured.get('mode')!r}. "
        f"CLI output:\n{result.output}"
    )


def test_add_to_kb_default_ingest_mode_unchanged(tmp_path):
    """Omitting --ingest-mode leaves config.knowledge_base.ingest_mode as config default."""
    from click.testing import CliRunner

    from perspicacite.cli import cli

    bib = _make_bib(tmp_path)
    captured: dict = {}

    async def fake_add_bibtex(config, kb_name, bib_path, session_db, chroma_dir):
        captured["mode"] = config.knowledge_base.ingest_mode
        return {
            "new_papers": 0,
            "chunks_added": 0,
            "pdf_stats": {"attempted": 0, "success": 0, "failed": 0, "skipped_no_doi": 0},
        }

    with patch("perspicacite.cli._add_bibtex_to_existing_kb", new=fake_add_bibtex):
        runner = CliRunner()
        runner.invoke(
            cli,
            ["-c", "config.example.yml", "add-to-kb", "mytest", "--from-bibtex", str(bib)],
        )

    # config.example.yml has ingest_mode: "auto" — flag not given, stays "auto"
    assert captured.get("mode") == "auto"


def test_add_to_kb_ingest_mode_full_text(tmp_path):
    """--ingest-mode full_text is a valid choice."""
    from click.testing import CliRunner

    from perspicacite.cli import cli

    bib = _make_bib(tmp_path)
    captured: dict = {}

    async def fake_add_bibtex(config, kb_name, bib_path, session_db, chroma_dir):
        captured["mode"] = config.knowledge_base.ingest_mode
        return {
            "new_papers": 0,
            "chunks_added": 0,
            "pdf_stats": {"attempted": 0, "success": 0, "failed": 0, "skipped_no_doi": 0},
        }

    with patch("perspicacite.cli._add_bibtex_to_existing_kb", new=fake_add_bibtex):
        runner = CliRunner()
        runner.invoke(
            cli,
            ["-c", "config.example.yml", "add-to-kb", "mytest",
             "--from-bibtex", str(bib), "--ingest-mode", "full_text"],
        )

    assert captured.get("mode") == "full_text"


def test_add_to_kb_invalid_ingest_mode_rejected(tmp_path):
    """--ingest-mode banana is rejected by Click with a non-zero exit code."""
    from click.testing import CliRunner

    from perspicacite.cli import cli

    bib = _make_bib(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["-c", "config.example.yml", "add-to-kb", "mytest",
         "--from-bibtex", str(bib), "--ingest-mode", "banana"],
    )
    assert result.exit_code != 0
    assert "banana" in result.output or "Invalid value" in result.output


def test_create_kb_ingest_mode_overrides_config(tmp_path):
    """--ingest-mode abstract_only also works on create-kb --from-bibtex."""
    from click.testing import CliRunner

    from perspicacite.cli import cli

    bib = _make_bib(tmp_path)
    captured: dict = {}

    async def fake_create_bibtex(config, kb_name, bib_path, description, session_db, chroma_dir):
        captured["mode"] = config.knowledge_base.ingest_mode
        return {
            "name": kb_name,
            "collection_name": f"kb_{kb_name}",
            "papers": 0,
            "chunks_added": 0,
            "pdf_stats": {"attempted": 0, "success": 0, "failed": 0, "skipped_no_doi": 0},
        }

    with patch("perspicacite.cli._create_kb_from_bibtex", new=fake_create_bibtex):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["-c", "config.example.yml", "create-kb", "newkb",
             "--from-bibtex", str(bib), "--ingest-mode", "abstract_only"],
        )

    assert captured.get("mode") == "abstract_only", (
        f"Expected 'abstract_only', got {captured.get('mode')!r}. Output:\n{result.output}"
    )
