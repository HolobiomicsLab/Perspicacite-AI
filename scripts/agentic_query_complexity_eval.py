#!/usr/bin/env python3
"""Light offline eval: heuristic + optional LLM query_complexity vs labeled expectations.

Run from repo root:
  python scripts/agentic_query_complexity_eval.py
  python scripts/agentic_query_complexity_eval.py --llm   # needs configured API like rest of app
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Repo src on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perspicacite.rag.agentic.intent import (  # noqa: E402
    heuristic_query_complexity,
    HEURISTIC_WEAK_COMPLEXITY_TAGS,
)

# (query, expected: simple | composite)
LABELED: list[tuple[str, str]] = [
    ("What is molecular networking?", "simple"),
    ("Explain feature-based molecular networking for metabolomics.", "simple"),
    ("Compare FBMN vs classical molecular networking pros and cons", "composite"),
    ("CRISPR versus zinc-finger nucleases: differences", "composite"),
    ("Advantages and disadvantages of LC-MS/MS for lipids", "composite"),
    ("Effect of temperature on enzyme kinetics in vitro", "simple"),
    ("Tea polyphenols and their effect on gut microbiota", "composite"),
    ("Difference between GNPS and MZmine workflows", "composite"),
    ("Search papers on natural products from Streptomyces", "simple"),
    ("Lotus database metabolite lookup for quercetin", "simple"),
    ("Trade-offs between sensitivity and specificity in mass spec", "composite"),
    ("What are the applications of tandem MS in metabolomics?", "simple"),
    ("X versus Y versus Z in ion mobility", "composite"),
    ("How does database search work in proteomics?", "simple"),
    ("Compare retention time prediction models A and B for metabolites", "composite"),
    ("Pros and cons of open vs proprietary spectral libraries", "composite"),
    ("Tell me about the history of GNPS", "simple"),
    ("Metabolomics and proteomics integration challenges", "simple"),
    ("Differences between ESI and MALDI ionization", "composite"),
    ("Find recent reviews on exposomics", "simple"),
]


def heuristic_with_demotion(query: str) -> tuple[str, str]:
    """Apply the same weak-tag demotion used in production (without LLM)."""
    h, tag = heuristic_query_complexity(query)
    if tag in HEURISTIC_WEAK_COMPLEXITY_TAGS:
        return "simple", tag
    return h, tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also run IntentClassifier.classify (async, needs LLM env).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Test raw heuristic (without weak-tag demotion).",
    )
    args = parser.parse_args()

    n = len(LABELED)
    correct = 0
    rows = []
    fn = heuristic_query_complexity if args.raw else heuristic_with_demotion
    label = "Raw heuristic" if args.raw else "Heuristic (with demotion)"
    for q, exp in LABELED:
        h, tag = fn(q)
        pred = h
        ok = pred == exp
        correct += int(ok)
        rows.append(
            {
                "query": q[:120],
                "expected": exp,
                "heuristic": pred,
                "tag": tag,
                "match": ok,
            }
        )

    print(f"{label} accuracy: {correct}/{n} ({100.0 * correct / n:.1f}%)")
    mismatches = [r for r in rows if not r["match"]]
    if mismatches:
        print(json.dumps(mismatches, indent=2))
    else:
        print("All queries matched.")

    if args.llm:
        asyncio.run(_run_llm_eval())


async def _run_llm_eval() -> None:
    try:
        from perspicacite.config.loader import load_config
        from perspicacite.llm import AsyncLLMClient
        from perspicacite.rag.agentic.intent import IntentClassifier
        from perspicacite.rag.agentic.llm_adapter import LLMAdapter
    except Exception as e:
        print("LLM eval skipped (import error):", e)
        return

    cfg = load_config()
    llm = LLMAdapter(AsyncLLMClient(cfg.llm))
    clf = IntentClassifier(llm)
    n = len(LABELED)
    correct = 0
    for q, exp in LABELED:
        r = await clf.classify(q, conversation_history=None, active_kb_name=None)
        pred = getattr(r, "query_complexity", "simple")
        correct += int(pred == exp)
    print(f"LLM+merged accuracy: {correct}/{n} ({100.0 * correct / n:.1f}%)")


if __name__ == "__main__":
    main()
