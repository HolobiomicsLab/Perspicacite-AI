"""Unit tests for rag.utils.json_salvage."""

from perspicacite.rag.utils.json_salvage import (
    clean_control_chars,
    salvage_truncated_array,
)


def test_clean_control_chars_strips_invalid():
    raw = "hello\x01world\x02!"
    assert clean_control_chars(raw) == "helloworld!"


def test_clean_control_chars_keeps_whitespace():
    raw = "line1\nline2\tcol\r\nline3"
    assert clean_control_chars(raw) == "line1\nline2\tcol\r\nline3"


def test_clean_control_chars_strips_all_bad_range():
    for code in range(0, 9):
        assert clean_control_chars(chr(code)) == ""
    assert clean_control_chars("\x0b") == ""
    assert clean_control_chars("\x0c") == ""
    for code in range(0x0e, 0x20):
        assert clean_control_chars(chr(code)) == ""


def test_salvage_truncated_array_recovers_complete_entries():
    truncated = """
    {
      "analyses": [
        {"id": "p1", "score": 4},
        {"id": "p2", "score": 5},
        {"id": "p3", "scor
    """
    result = salvage_truncated_array(truncated, "analyses")
    assert result == [
        {"id": "p1", "score": 4},
        {"id": "p2", "score": 5},
    ]


def test_salvage_no_array_key_returns_none():
    assert salvage_truncated_array('{"other": []}', "missing") is None


def test_salvage_handles_braces_inside_strings():
    """Braces inside string values must not throw off depth counting."""
    payload = """
    {"analyses": [
      {"text": "method uses { and } chars", "score": 4},
      {"text": "incomplete...
    """
    result = salvage_truncated_array(payload, "analyses")
    assert result == [{"text": "method uses { and } chars", "score": 4}]


def test_salvage_empty_array_returns_none():
    """No entries inside the array -> return None (caller handles)."""
    assert salvage_truncated_array('{"analyses": []}', "analyses") is None


def test_salvage_complete_array_returns_all():
    """A valid, non-truncated array returns all entries."""
    data = '{"analyses": [{"id": 1}, {"id": 2}]}'
    result = salvage_truncated_array(data, "analyses")
    assert result == [{"id": 1}, {"id": 2}]
