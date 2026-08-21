"""The cookie exporter must not dump the whole jar without an explicit opt-in."""
import pytest
import structlog
from click.testing import CliRunner

from perspicacite.cli import cli


@pytest.fixture(autouse=True)
def restore_structlog_config():
    """Undo the global logging setup that invoking the CLI performs.

    The CLI configures structlog with ``cache_logger_on_first_use=True``. That
    is process-wide, and once it is on, ``structlog.testing.capture_logs``
    cannot intercept an already-bound logger — which silently breaks any later
    test that asserts on captured log records.
    """
    saved = structlog.get_config().copy()
    yield
    structlog.reset_defaults()
    structlog.configure(**saved)

def test_refuses_without_domain_filter(tmp_path):
    r = CliRunner().invoke(cli, ["import-browser-cookies", "--output", str(tmp_path / "c.txt")])
    assert r.exit_code == 2
    assert "Refusing to export every cookie" in r.output
    assert not (tmp_path / "c.txt").exists()

def test_all_domains_flag_is_accepted(tmp_path):
    """--all-domains gets past the guard (then fails later on browser access)."""
    r = CliRunner().invoke(cli, ["import-browser-cookies", "--all-domains", "--output", str(tmp_path / "c.txt")])
    assert "Refusing to export every cookie" not in r.output

def test_domain_filter_gets_past_the_guard(tmp_path):
    r = CliRunner().invoke(cli, ["import-browser-cookies", "--domain", "nature.com", "--output", str(tmp_path / "c.txt")])
    assert "Refusing to export every cookie" not in r.output
