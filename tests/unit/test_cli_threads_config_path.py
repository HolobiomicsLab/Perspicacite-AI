"""Regression: every CLI command must thread the -c config into AppState.

Commands that did `AppState()` (no config_path) silently loaded the default config.yml
embedder, so ingesting into a KB built with a different embedder (e.g. openai-large)
failed with dimension mismatches. Guard at the source level: no bare AppState() in cli.py.
"""

from __future__ import annotations

import pathlib
import re


def test_no_bare_appstate_in_cli():
    src = pathlib.Path("src/perspicacite/cli.py").read_text()
    bare = re.findall(r"AppState\(\)", src)
    assert not bare, f"{len(bare)} bare AppState() — must pass config_path=ctx.obj.get(...)"


def test_appstate_calls_thread_config_path():
    src = pathlib.Path("src/perspicacite/cli.py").read_text()
    # every AppState( in cli.py should carry a config_path argument
    calls = re.findall(r"AppState\((?!config_path)", src)
    assert not calls, "an AppState(...) in cli.py is constructed without config_path"
