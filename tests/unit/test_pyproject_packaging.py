"""Guards the packaging metadata that keeps `uv sync` working on fresh clones.

Regression test for issue #29 (and #5 before it). uv resolves *every* extra and
*every* [tool.uv.sources] entry when it locks, regardless of which extra the
user asked for. So a local-path source, or a requirement on a package that is
not published, makes `uv sync` fail for anyone without the private sibling
checkouts — before a single line of Python runs.
"""

import re
import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Private, unpublished siblings of this repository. See the pyproject comment.
PRIVATE_DISTRIBUTIONS = frozenset(
    {"indicium", "indicium-adapters", "indicium-adapters-metabolomics"}
)


def _load_pyproject() -> dict:
    """Parse the repository's pyproject.toml."""
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _uses_local_path(source_spec) -> bool:
    """True when a [tool.uv.sources] entry points at a filesystem path."""
    entries = source_spec if isinstance(source_spec, list) else [source_spec]
    return any(isinstance(entry, dict) and "path" in entry for entry in entries)


def _distribution_name(requirement: str) -> str:
    """Return the normalised distribution name of a PEP 508 requirement."""
    name = re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _declared_requirements(pyproject: dict) -> list[str]:
    """Every requirement uv resolves: base deps, extras, and dependency-groups."""
    project = pyproject["project"]
    requirements = list(project.get("dependencies", []))
    for extra_requirements in project.get("optional-dependencies", {}).values():
        requirements.extend(extra_requirements)
    for group_requirements in pyproject.get("dependency-groups", {}).values():
        # PEP 735 groups may also hold {"include-group": ...} tables; skip those.
        requirements.extend(r for r in group_requirements if isinstance(r, str))
    return requirements


def test_no_local_path_uv_sources():
    """Local-path sources break `uv sync` for clones without the siblings."""
    sources = _load_pyproject().get("tool", {}).get("uv", {}).get("sources", {})
    offenders = sorted(name for name, spec in sources.items() if _uses_local_path(spec))
    assert not offenders, (
        f"[tool.uv.sources] entries {offenders} point at local paths, which makes "
        "`uv sync` fail on any fresh clone lacking those sibling directories"
    )


def test_no_private_distributions_declared():
    """Unpublished packages must not appear in any uv-resolved requirement."""
    declared = {_distribution_name(req) for req in _declared_requirements(_load_pyproject())}
    leaked = sorted(declared & PRIVATE_DISTRIBUTIONS)
    assert not leaked, (
        f"{leaked} are private and absent from PyPI; declaring them in any extra "
        "or dependency-group makes `uv lock` unsatisfiable for every external user"
    )


def test_graph_extra_declares_the_rdf_backend():
    """indicium_layer/store.py imports rdflib and pyoxigraph directly."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    graph_distributions = {_distribution_name(req) for req in extras["graph"]}
    assert {"rdflib", "pyoxigraph"} <= graph_distributions
