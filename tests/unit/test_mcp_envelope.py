"""Pin the MCP JSON envelope shape. Both 'success' and 'ok' keys
are emitted for one minor cycle to ease the Scriptorium-v0.13
downstream client migration; 'ok' is the deprecated alias."""
import json

from perspicacite.mcp.server import _json_error, _json_ok


def test_json_ok_emits_both_success_and_ok():
    payload = json.loads(_json_ok({"x": 1}))
    assert payload["success"] is True
    assert payload["ok"] is True
    assert payload["x"] == 1


def test_json_error_emits_both_success_and_ok_false():
    payload = json.loads(_json_error("boom"))
    assert payload["success"] is False
    assert payload["ok"] is False
    assert payload["error"] == "boom"
