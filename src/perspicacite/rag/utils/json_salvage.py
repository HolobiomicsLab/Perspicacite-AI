"""LLM-emitted JSON salvage helpers.

Two failure modes are common:

- Truncation mid-array (the LLM hit max_tokens before closing ``]``).
  ``salvage_truncated_array`` walks the partial string, extracts every
  complete ``{...}`` object inside the named array, and returns them
  as a list of parsed dicts. Better to keep 23/25 entries than throw.

- Raw control characters inside string values (some providers emit
  literal ``\\x01`` etc. that ``json.loads`` rejects with
  "Invalid control character"). ``clean_control_chars`` strips them
  while preserving valid whitespace (``\\t``, ``\\n``, ``\\r``).
"""
from __future__ import annotations

import json
import re
from typing import Any


def clean_control_chars(json_str: str) -> str:
    """Strip raw ASCII control chars (0x00-0x1F) except whitespace.

    Keeps ``\\t`` (0x09), ``\\n`` (0x0A), ``\\r`` (0x0D) intact. Drops
    every other char in the 0x00-0x1F range, which is what makes
    json.loads explode with "Invalid control character at: ...".
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", json_str)


def salvage_truncated_array(
    json_str: str, array_key: str,
) -> list[dict[str, Any]] | None:
    """Recover complete ``{...}`` entries from a truncated array.

    Looks for ``"array_key": [`` in ``json_str``, then scans forward
    extracting every complete brace-balanced object until the array
    ends or the string runs out. Quote-aware so braces inside string
    values don't confuse the depth counter.

    Returns ``None`` when the array_key isn't found OR when no complete
    entries could be extracted; the caller falls back to the original
    JSONDecodeError.
    """
    m = re.search(rf'"{re.escape(array_key)}"\s*:\s*\[', json_str)
    if not m:
        return None
    start = m.end()
    depth = 0
    i = start
    complete_objects: list[str] = []
    obj_start = -1
    in_str = False
    esc = False
    while i < len(json_str):
        c = json_str[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    complete_objects.append(json_str[obj_start : i + 1])
                    obj_start = -1
            elif c == "]" and depth == 0:
                break
        i += 1

    recovered: list[dict[str, Any]] = []
    for obj_str in complete_objects:
        try:
            recovered.append(json.loads(obj_str))
        except (json.JSONDecodeError, ValueError):
            continue
    return recovered or None
