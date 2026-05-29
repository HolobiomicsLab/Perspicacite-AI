"""CLI command `perspicacite ingest-asb-run <run_dir>` — wraps the
ASB orchestrator and prints a summary."""
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

# The exact command name (kebab-case) the CLI exposes
COMMAND_NAME = "ingest-asb-run"


def test_cli_ingest_asb_run_help_lists_command():
    from perspicacite.cli import cli
    runner = CliRunner()
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0, r.output
    assert COMMAND_NAME in r.output


def test_cli_ingest_asb_run_invokes_orchestrator(tmp_path):
    """The command should call the orchestrator with the run dir + flags."""
    from perspicacite.cli import cli

    fake_result = {
        "kb_names": ["my_bundle"],
        "skills_ingested": 1,
        "workflows_ingested": 2,
        "papers_ingested": 3,
        "failed": [],
        "workflow_dag": None,
    }

    captured = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return fake_result

    # Patch the orchestrator's import inside cli.py
    with patch(
        "perspicacite.cli.ingest_asb_run_pipeline",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        run_dir = tmp_path / "asb"
        run_dir.mkdir()
        r = runner.invoke(
            cli,
            [COMMAND_NAME, str(run_dir), "--kb-name", "my_bundle"],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("kb_name") == "my_bundle"
    assert str(run_dir) in captured.get("asb_run_dir", "")
    assert "my_bundle" in r.output
    # Summary line includes counts
    assert "1" in r.output and "2" in r.output


def test_cli_ingest_asb_run_supports_include_flag(tmp_path):
    from perspicacite.cli import cli

    fake_result = {
        "kb_names": ["b"], "skills_ingested": 1, "workflows_ingested": 0,
        "papers_ingested": 1, "failed": [], "workflow_dag": None,
    }
    captured = {}

    async def fake_orchestrator(**kw):
        captured.update(kw)
        return fake_result

    with patch(
        "perspicacite.cli.ingest_asb_run_pipeline",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        run_dir = tmp_path / "asb"
        run_dir.mkdir()
        r = runner.invoke(
            cli,
            [COMMAND_NAME, str(run_dir), "--include", "skills"],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert set(captured.get("include", [])) == {"skills"}


def test_cli_ingest_asb_run_per_skill_mode(tmp_path):
    from perspicacite.cli import cli

    captured = {}
    async def fake_orchestrator(**kw):
        captured.update(kw)
        return {"kb_names": ["a", "a__s1"], "skills_ingested": 1,
                "workflows_ingested": 0, "papers_ingested": 1,
                "failed": [], "workflow_dag": None}

    with patch(
        "perspicacite.cli.ingest_asb_run_pipeline",
        new=fake_orchestrator,
    ), patch(
        "perspicacite.cli._build_app_state_for_cli",
        new=AsyncMock(return_value=MagicMock()),
    ):
        runner = CliRunner()
        run_dir = tmp_path / "asb"
        run_dir.mkdir()
        r = runner.invoke(
            cli,
            [COMMAND_NAME, str(run_dir), "--mode", "per-skill"],
            catch_exceptions=False,
        )

    assert r.exit_code == 0, r.output
    assert captured.get("mode") == "per-skill"


def test_cli_ingest_asb_run_missing_dir_errors(tmp_path):
    """Click should error out if the run dir doesn't exist (path type check)."""
    from perspicacite.cli import cli
    runner = CliRunner()
    r = runner.invoke(cli, [COMMAND_NAME, str(tmp_path / "missing")])
    assert r.exit_code != 0
