"""Guard: a stdlib logger must never be called with structlog-style keyword args.

Most modules bind `logger` with `perspicacite.logging.get_logger`, a structlog
logger where `logger.warning("event", kb=name)` is the intended API. A handful
bind it with `logging.getLogger`, where the same call raises
`TypeError: _log() got an unexpected keyword argument 'kb'`.

Every offending call found so far sat inside an `except` block whose job was to
degrade gracefully, so the handler replaced a recoverable failure with a crash.
"""

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"

# Keyword arguments logging.Logger.debug/info/warning/... actually accepts.
STDLIB_LOG_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})
LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log"})


def _logger_names_by_binding(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return the names bound to a stdlib logger and to a structlog logger."""
    stdlib: set[str] = set()
    structlog: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "getLogger":
            stdlib |= targets
        elif isinstance(func, ast.Name) and func.id == "get_logger":
            structlog |= targets
    return stdlib, structlog


def _rejected_keyword_calls(tree: ast.Module, stdlib_names: set[str]) -> list[tuple[int, str]]:
    """Find stdlib-logger calls passing a keyword the stdlib will reject."""
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = node.func.value
        if node.func.attr not in LOG_METHODS or not isinstance(target, ast.Name):
            continue
        if target.id not in stdlib_names:
            continue
        rejected = [kw.arg for kw in node.keywords if kw.arg not in STDLIB_LOG_KWARGS]
        if rejected:
            offenders.append((node.lineno, ", ".join(sorted(rejected))))
    return offenders


def test_no_stdlib_logger_is_called_with_structlog_keywords():
    """A rejected keyword turns any log line into a TypeError at call time."""
    failures = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stdlib_names, structlog_names = _logger_names_by_binding(tree)
        # A name rebound to get_logger later is a structlog logger; leave it alone.
        for line, keywords in _rejected_keyword_calls(tree, stdlib_names - structlog_names):
            failures.append(f"{path.relative_to(SOURCE_ROOT.parent)}:{line} passes {keywords}")

    assert not failures, (
        "stdlib logging.Logger rejects these keyword arguments, so each call raises "
        "TypeError when it runs:\n  " + "\n  ".join(failures)
    )
