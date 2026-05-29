"""Tests for src/perspicacite/pipeline/extraction.py."""
from __future__ import annotations

from unittest.mock import AsyncMock

from perspicacite.pipeline.extraction import (
    Passage,
    classify_license_tier,
    extract_structured,
    handle_quote_for_license,
)


def _llm(json_str: str):
    client = AsyncMock()
    client.complete = AsyncMock(return_value=json_str)
    return client


def test_classify_license_tier():
    assert classify_license_tier("CC-BY") == "A"
    assert classify_license_tier("MIT") == "A"
    assert classify_license_tier("CC0") == "A"
    assert classify_license_tier("CC-BY-NC") == "B"
    assert classify_license_tier("CC-BY-ND") == "B"
    assert classify_license_tier("all rights reserved") == "C"
    assert classify_license_tier(None) == "C"
    assert classify_license_tier("") == "C"


def test_handle_quote_for_license_tier_a_keeps_verbatim():
    out = handle_quote_for_license("hello world", license_id="CC-BY")
    assert out == "hello world"


def test_handle_quote_for_license_tier_b_short_keeps():
    short = "x" * 250
    out = handle_quote_for_license(short, license_id="CC-BY-NC")
    assert out == short


def test_handle_quote_for_license_tier_b_long_paraphrases():
    long = "y" * 350
    out = handle_quote_for_license(
        long, license_id="CC-BY-NC",
        paraphraser=lambda s: f"PARAPHRASED:{s[:5]}",
    )
    assert out == "PARAPHRASED:yyyyy"


def test_handle_quote_for_license_tier_c_paraphrases():
    out = handle_quote_for_license(
        "secret text", license_id=None,
        paraphraser=lambda s: f"P:{s}",
    )
    assert out == "P:secret text"


def test_handle_quote_for_license_tier_c_no_paraphraser_drops():
    out = handle_quote_for_license("secret", license_id=None, paraphraser=None)
    assert out is None


async def test_extract_structured_happy_path():
    llm = _llm('[{"name":"temp","typical":"37","units":"C"}]')
    schema = {"type": "array"}
    passages = [Passage(text="grew at 37 C", source_doi="10/a", license_id="CC-BY")]

    out = await extract_structured(
        llm_client=llm,
        passages=passages,
        prompt_template="Extract {what}",
        schema=schema,
        what="parameters",
        dedup_key=lambda r: (r.get("name"), r.get("units")),
    )

    assert out == [{"name": "temp", "typical": "37", "units": "C"}]
    llm.complete.assert_awaited_once()


async def test_extract_structured_dedups():
    llm = _llm(
        '[{"name":"temp","units":"C"},{"name":"temp","units":"C"},{"name":"pH","units":""}]'
    )
    out = await extract_structured(
        llm_client=llm,
        passages=[Passage(text="x", source_doi="d", license_id="CC-BY")],
        prompt_template="x",
        schema={},
        what="p",
        dedup_key=lambda r: (r.get("name"), r.get("units")),
    )
    assert [r["name"] for r in out] == ["temp", "pH"]


async def test_extract_structured_invalid_json_returns_empty_with_warning():
    llm = _llm("not json at all {{{")
    out = await extract_structured(
        llm_client=llm,
        passages=[Passage(text="x", source_doi="d", license_id="CC-BY")],
        prompt_template="x",
        schema={},
        what="p",
        dedup_key=lambda r: tuple(r.items()),
    )
    assert out == []


async def test_extract_structured_empty_passages_returns_empty():
    llm = AsyncMock()
    llm.complete = AsyncMock()
    out = await extract_structured(
        llm_client=llm,
        passages=[],
        prompt_template="x",
        schema={},
        what="p",
        dedup_key=lambda r: tuple(r.items()),
    )
    assert out == []
    llm.complete.assert_not_awaited()
