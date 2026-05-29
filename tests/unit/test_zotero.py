import httpx
import pytest

from perspicacite.integrations.zotero import (
    ZoteroAPIError,
    ZoteroAuthError,
    ZoteroClient,
    _text_to_html,
    _TokenBucket,
)


@pytest.mark.asyncio
async def test_create_item_maps_doi_and_returns_key(respx_mock):
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    create_route = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json={
            "success": {"0": "ABC123"},
            "successful": {"0": {"key": "ABC123"}},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        key = await c.create_item(paper={
            "doi": "10.1/x", "title": "T", "year": 2024, "journal": "J",
            "authors": ["A B"], "abstract": "abs",
        })
    assert key == "ABC123"
    body_bytes = create_route.calls[0].request.read()
    assert b"10.1/x" in body_bytes
    assert b"journalArticle" in body_bytes


@pytest.mark.asyncio
async def test_dedup_returns_existing_key(respx_mock):
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[
            {"key": "EXIST", "data": {"DOI": "10.1/x", "title": "T"}}
        ])
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        key = await c.create_item(paper={"doi": "10.1/x", "title": "T"})
    assert key == "EXIST"


@pytest.mark.asyncio
async def test_group_library_uses_groups_path(respx_mock):
    respx_mock.post("https://api.zotero.org/groups/999/items").mock(
        return_value=httpx.Response(200, json={"successful": {"0": {"key": "G1"}}, "success": {"0": "G1"}, "failed": {}})
    )
    respx_mock.get("https://api.zotero.org/groups/999/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="999", library_type="group", http_client=http)
        key = await c.create_item(paper={"doi": "10.1/x", "title": "T"})
    assert key == "G1"


def test_zotero_client_requires_credentials():
    with pytest.raises(ValueError):
        ZoteroClient(api_key="", library_id="123")
    with pytest.raises(ValueError):
        ZoteroClient(api_key="k", library_id="")


@pytest.mark.asyncio
async def test_create_item_with_collection_key(respx_mock):
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    create_route = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json={"successful": {"0": {"key": "X"}}, "success": {"0": "X"}, "failed": {}})
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user",
                         collection_key="COLL", http_client=http)
        await c.create_item(paper={"doi": "10.1/x", "title": "T"})
    body_bytes = create_route.calls[0].request.read()
    assert b"COLL" in body_bytes


@pytest.mark.asyncio
async def test_dedup_null_doi_does_not_crash_and_no_false_match(respx_mock):
    """When Zotero returns DOI: null in response, dedup must not crash and
    must not consider it a match for any real DOI."""
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(
            200,
            json=[{"key": "OLD", "data": {"DOI": None, "title": "Something"}}],
        )
    )
    # Provide a post mock so the item can be created without error
    respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(
            200,
            json={"successful": {"0": {"key": "NEW"}}, "success": {"0": "NEW"}, "failed": {}},
        )
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        # Should not raise AttributeError, and should NOT return "OLD"
        key = await c.create_item(paper={"doi": "10.1/real", "title": "Real"})
    # The null-DOI item must not be returned as a duplicate
    assert key != "OLD"


@pytest.mark.asyncio
async def test_dedup_falls_back_to_recent_items_when_search_misses(respx_mock):
    """Zotero's full-text search index is eventually consistent; items
    pushed via the API can take minutes-to-hours to become searchable.
    During that window the q=<DOI> search returns []. Dedup must fall
    back to scanning recent items so we don't double-create.

    Regression for 2026-05-16 audit: pushing 10.48550/arxiv.2603.08127
    twice within 15 minutes produced two Zotero items because the
    second push's q=<DOI> search hadn't yet indexed the first.
    """
    # Search by ?q=... returns empty (indexing lag) — match the first GET only.
    # The second GET hits the same /items endpoint with different params
    # (direction=desc, limit=100) and returns the actual item.
    calls_by_q: list[str | None] = []

    def _handler(request):
        q = request.url.params.get("q")
        direction = request.url.params.get("direction")
        calls_by_q.append(q)
        if q:
            # Indexing-lag path: search returns nothing
            return httpx.Response(200, json=[])
        if direction == "desc":
            # Fallback: recent items includes the duplicate
            return httpx.Response(200, json=[
                {"key": "RECENT_DUP",
                 "data": {"DOI": "10.48550/arxiv.2603.08127",
                          "title": "EvoScientist"}}
            ])
        return httpx.Response(200, json=[])

    respx_mock.get("https://api.zotero.org/groups/6555390/items").mock(side_effect=_handler)
    respx_mock.post("https://api.zotero.org/groups/6555390/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "WOULDVE_BEEN_DUP"}},
            "success": {"0": "WOULDVE_BEEN_DUP"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="6555390", library_type="group", http_client=http)
        key = await c.create_item(paper={
            "doi": "10.48550/arxiv.2603.08127",
            "title": "EvoScientist",
        })
    assert key == "RECENT_DUP", (
        f"dedup should have found RECENT_DUP via recent-items fallback, got {key!r}; "
        f"GET calls: {calls_by_q}"
    )


@pytest.mark.asyncio
async def test_dedup_normalizes_doi_prefix_and_case(respx_mock):
    """DOI normalization must strip https://doi.org/ prefix and be
    case-insensitive so https://doi.org/10.1038/X matches 10.1038/x."""
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[
            {"key": "MATCH",
             "data": {"DOI": "https://doi.org/10.1038/X", "title": "T"}}
        ])
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        key = await c.create_item(paper={"doi": "10.1038/x", "title": "T"})
    assert key == "MATCH"


@pytest.mark.asyncio
async def test_create_item_url_route_creates_webpage(respx_mock):
    """When paper has no DOI but has a URL + title, create a webpage item."""
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    create = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "WP1"}},
            "success": {"0": "WP1"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        key = await c.create_item(paper={
            "url": "https://github.com/langchain-ai/langgraph",
            "title": "LangGraph",
            "authors": ["LangChain Inc."],
        })
    assert key == "WP1"
    body = create.calls[0].request.read()
    assert b"webpage" in body
    assert b"langgraph" in body.lower()


@pytest.mark.asyncio
async def test_create_item_url_route_dedups_by_url(respx_mock):
    """URL-route items dedup by URL when the DOI is empty."""
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[
            {"key": "EXISTING_URL_ITEM",
             "data": {"itemType": "webpage",
                       "url": "https://github.com/langchain-ai/langgraph",
                       "title": "LangGraph"}}
        ])
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        key = await c.create_item(paper={
            "url": "https://github.com/langchain-ai/langgraph",
            "title": "LangGraph",
        })
    assert key == "EXISTING_URL_ITEM"


@pytest.mark.asyncio
async def test_create_item_explicit_item_type(respx_mock):
    """Caller can override item_type — e.g. for preprints or software."""
    respx_mock.get("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    create = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "PRE1"}},
            "success": {"0": "PRE1"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        await c.create_item(paper={
            "doi": "10.48550/arXiv.1234",
            "title": "Foo",
            "item_type": "preprint",
            "repository": "arXiv",
            "archive_id": "1234.5678",
        })
    body = create.calls[0].request.read()
    assert b"preprint" in body
    assert b"arXiv" in body


@pytest.mark.asyncio
async def test_create_item_group_local_writes_to_cloud(respx_mock):
    """Group library + local base_url → writes must still go to cloud."""
    respx_mock.get("https://api.zotero.org/groups/999/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    create_route = respx_mock.post("https://api.zotero.org/groups/999/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "K1"}},
            "success": {"0": "K1"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="real-cloud-key",
            library_id="999",
            library_type="group",
            base_url="http://localhost:23119/api",
            http_client=http,
        )
        key = await c.create_item(paper={"doi": "10.1/x", "title": "T"})
    assert key == "K1"
    posted_url = str(create_route.calls[0].request.url)
    assert "api.zotero.org" in posted_url
    assert "localhost" not in posted_url


@pytest.mark.asyncio
async def test_create_item_user_local_writes_to_localhost(respx_mock):
    """User library + local base_url → writes go directly to localhost, no api_key needed."""
    respx_mock.get("http://localhost:23119/api/users/42/items").mock(
        return_value=httpx.Response(200, json=[])
    )
    create_route = respx_mock.post("http://localhost:23119/api/users/42/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "LOCAL1"}},
            "success": {"0": "LOCAL1"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="",
            library_id="42",
            library_type="user",
            base_url="http://localhost:23119/api",
            http_client=http,
        )
        key = await c.create_item(paper={"doi": "10.1/x", "title": "T"})
    assert key == "LOCAL1"
    posted_url = str(create_route.calls[0].request.url)
    assert "localhost" in posted_url
    assert "api.zotero.org" not in posted_url


@pytest.mark.asyncio
async def test_create_note_user_local_writes_to_localhost(respx_mock):
    """User library + local base_url → note POST goes to localhost."""
    create_route = respx_mock.post("http://localhost:23119/api/users/42/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "NOTE_LOCAL"}},
            "success": {"0": "NOTE_LOCAL"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="",
            library_id="42",
            library_type="user",
            base_url="http://localhost:23119/api",
            http_client=http,
        )
        key = await c.create_note(parent_item_key="PARENT1", content="Local note")
    assert key == "NOTE_LOCAL"
    posted_url = str(create_route.calls[0].request.url)
    assert "localhost" in posted_url
    assert "api.zotero.org" not in posted_url


@pytest.mark.asyncio
async def test_create_item_group_local_without_api_key_raises():
    """Group library + local base_url + no api_key → ZoteroWriteUnsupportedError."""
    from perspicacite.integrations.zotero import ZoteroWriteUnsupportedError
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="",
            library_id="999",
            library_type="group",
            base_url="http://localhost:23119/api",
            http_client=http,
        )
        with pytest.raises(ZoteroWriteUnsupportedError) as exc:
            await c.create_item(paper={"doi": "10.1/x", "title": "T"})
    assert "api_key" in str(exc.value)


@pytest.mark.asyncio
async def test_upload_attachment_writes_to_cloud_when_configured_for_local(
    respx_mock, tmp_path
):
    """Symmetric to create_item: when base_url is local, the attachment
    upload protocol (3-step + finalize) must use the cloud API."""
    # Write a small fake PDF that upload_attachment can read.
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    # md5-dedup pre-check looks up parent's children (empty = no dedup).
    respx_mock.get(
        "https://api.zotero.org/groups/999/items/PARENT/children"
    ).mock(return_value=httpx.Response(200, json=[]))
    register_route = respx_mock.post("https://api.zotero.org/groups/999/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "ATT_K"}},
            "success": {"0": "ATT_K"},
            "failed": {},
        })
    )
    creds_route = respx_mock.post(
        "https://api.zotero.org/groups/999/items/ATT_K/file"
    ).mock(
        # Server-side dedup path: exists=1 — no upload needed.
        return_value=httpx.Response(200, json={"exists": 1})
    )

    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="real-cloud-key",
            library_id="999",
            library_type="group",
            base_url="http://localhost:23119/api",
            http_client=http,
        )
        key = await c.upload_attachment(
            parent_item_key="PARENT", file_path=str(pdf_path), filename="x.pdf",
        )
    assert key == "ATT_K"
    assert register_route.called
    assert creds_route.called
    assert "api.zotero.org" in str(register_route.calls[0].request.url)


@pytest.mark.asyncio
async def test_upload_attachment_skips_when_md5_already_attached(
    respx_mock, tmp_path
):
    """Live-discovered (2026-05-16): re-pushing the same DOI with
    attach_pdf=True hits HTTP 412 on step 2 ("If-None-Match: * set but
    file exists") because Zotero already has the same content under
    that parent. The 3-step upload protocol must be skipped entirely
    when the parent's children include an attachment with the same md5."""
    import hashlib

    pdf_bytes = b"%PDF-1.4 some bytes"
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(pdf_bytes)
    md5 = hashlib.md5(pdf_bytes).hexdigest()

    respx_mock.get(
        "https://api.zotero.org/groups/999/items/PARENT/children"
    ).mock(return_value=httpx.Response(200, json=[
        {"key": "EXISTING_ATT", "data": {
            "itemType": "attachment", "md5": md5, "filename": "x.pdf",
        }}
    ]))
    # The register POST should NEVER be reached.
    register_route = respx_mock.post("https://api.zotero.org/groups/999/items")

    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="cloud-key", library_id="999", library_type="group",
            http_client=http,
        )
        key = await c.upload_attachment(
            parent_item_key="PARENT", file_path=str(pdf_path), filename="x.pdf",
        )
    assert key == "EXISTING_ATT"
    assert not register_route.called


@pytest.mark.asyncio
async def test_upload_attachment_uses_if_match_and_prefix_suffix_protocol(
    respx_mock, tmp_path
):
    """Live-discovered (2026-05-16): Zotero's file-upload protocol on
    step 2 requires ``If-Match: <md5>`` (not ``If-None-Match: *``) — the
    latter returns 412 because step 1 already records the md5 in the
    shell data, which Zotero treats as "file is associated". Step 3
    must use the documented ``prefix + body + suffix`` raw POST, not
    multipart/form-data with a non-existent ``params`` key (that
    returns 400 from S3 because ``key`` field is missing)."""
    import hashlib

    html_bytes = b"<html><body>x</body></html>"
    html_path = tmp_path / "stub.html"
    html_path.write_bytes(html_bytes)
    md5 = hashlib.md5(html_bytes).hexdigest()

    respx_mock.get(
        "https://api.zotero.org/groups/999/items/PARENT/children"
    ).mock(return_value=httpx.Response(200, json=[]))
    respx_mock.post("https://api.zotero.org/groups/999/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "NEW_ATT"}},
            "failed": {},
        })
    )
    # Step 2: capture the request to assert the precondition.
    captured_headers: dict[str, str] = {}

    def _step2(request):
        captured_headers.update(request.headers)
        return httpx.Response(200, json={
            "url": "https://zoterofilestorage.s3.example.com/",
            "contentType": "multipart/form-data; boundary=----X",
            "prefix": "------X\r\nContent-Disposition: form-data; name=\"key\"\r\n\r\nfoo/bar\r\n------X\r\n",
            "suffix": "\r\n------X--",
            "uploadKey": "UPLOAD_KEY",
        })

    respx_mock.post(
        "https://api.zotero.org/groups/999/items/NEW_ATT/file"
    ).mock(side_effect=_step2)

    # Step 3: S3-style POST. Capture body to verify prefix+bytes+suffix.
    captured_body: dict[str, bytes] = {}

    def _step3(request):
        captured_body["body"] = request.content
        return httpx.Response(204)

    respx_mock.post("https://zoterofilestorage.s3.example.com/").mock(
        side_effect=_step3
    )

    # Step 4: finalize — re-uses /file endpoint; mock already set above
    # returns 200 (which is acceptable for finalize too).

    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="cloud-key", library_id="999", library_type="group",
            http_client=http,
        )
        key = await c.upload_attachment(
            parent_item_key="PARENT", file_path=str(html_path),
            filename="stub.html", content_type="text/html",
        )
    assert key == "NEW_ATT"
    # Precondition assertion: If-Match: <our_md5>, NOT If-None-Match
    assert captured_headers.get("if-match") == md5
    assert "if-none-match" not in {k.lower() for k in captured_headers}
    # Step 3 body assertion: prefix + bytes + suffix concatenation
    assert html_bytes in captured_body["body"]
    assert b"name=\"key\"" in captured_body["body"]  # prefix carries the S3 key field
    assert captured_body["body"].endswith(b"------X--")



# ---------------------------------------------------------------------------
# Production hardening: validate_credentials() + 401/403 fail-fast + rate limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_credentials_returns_key_info_on_200(respx_mock):
    respx_mock.get("https://api.zotero.org/keys/cloud-key").mock(
        return_value=httpx.Response(
            200,
            json={"key": "cloud-key", "access": {"user": {"library": True}}},
        )
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="cloud-key", library_id="123", library_type="user",
            http_client=http,
        )
        info = await c.validate_credentials()
    assert info["key"] == "cloud-key"


@pytest.mark.asyncio
async def test_validate_credentials_raises_authError_on_401_without_retry(
    respx_mock,
):
    """The whole point: on 401 we must raise immediately, NEVER retry.
    Looping 401s on Zotero triggers the ~15min IP-level lockout that
    was the original motivation for this hardening."""
    route = respx_mock.get("https://api.zotero.org/keys/bad-key").mock(
        return_value=httpx.Response(401, text="bad api key")
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="bad-key", library_id="123", library_type="user",
            http_client=http,
        )
        with pytest.raises(ZoteroAuthError, match="api_key"):
            await c.validate_credentials()
    # Single attempt — no retry burst
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_create_item_raises_authError_on_403_without_retry(respx_mock):
    """Same fail-fast contract on the write path."""
    respx_mock.get(
        url__regex=r"https://api\.zotero\.org/users/123/items.*"
    ).mock(return_value=httpx.Response(200, json=[]))
    route = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(403, text="permission denied")
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(
            api_key="k", library_id="123", library_type="user",
            http_client=http,
        )
        with pytest.raises(ZoteroAuthError, match="create_item"):
            await c.create_item({"title": "X", "doi": "10.1/x"})
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# _text_to_html helpers
# ---------------------------------------------------------------------------

def test_text_to_html_single_paragraph():
    assert _text_to_html("Hello world") == "<p>Hello world</p>"


def test_text_to_html_multiple_paragraphs():
    result = _text_to_html("Para one\n\nPara two")
    assert result == "<p>Para one</p>\n<p>Para two</p>"


def test_text_to_html_inline_newline_becomes_br():
    result = _text_to_html("Line one\nLine two")
    assert "<br>" in result


def test_text_to_html_escapes_html_chars():
    result = _text_to_html("<b>bold</b> & \"quotes\"")
    assert "<b>" not in result
    assert "&lt;" in result
    assert "&amp;" in result


# ---------------------------------------------------------------------------
# ZoteroClient.create_note
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_note_returns_key(respx_mock):
    create_route = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "NOTE01"}},
            "success": {"0": "NOTE01"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        key = await c.create_note(parent_item_key="PARENT1", content="My note text")
    assert key == "NOTE01"
    body = create_route.calls[0].request.read()
    assert b"note" in body
    assert b"PARENT1" in body


@pytest.mark.asyncio
async def test_create_note_sends_tags(respx_mock):
    create_route = respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(200, json={
            "successful": {"0": {"key": "NOTE02"}},
            "success": {"0": "NOTE02"},
            "failed": {},
        })
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        await c.create_note(parent_item_key="P1", content="text", tags=["ai-forge", "review"])
    body = create_route.calls[0].request.read()
    assert b"ai-forge" in body
    assert b"review" in body


@pytest.mark.asyncio
async def test_create_note_raises_auth_error_on_403(respx_mock):
    respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(403, text="Forbidden")
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        with pytest.raises(ZoteroAuthError):
            await c.create_note(parent_item_key="P1", content="x")


@pytest.mark.asyncio
async def test_create_note_raises_api_error_on_unexpected_status(respx_mock):
    respx_mock.post("https://api.zotero.org/users/123/items").mock(
        return_value=httpx.Response(500, text="Server error")
    )
    async with httpx.AsyncClient() as http:
        c = ZoteroClient(api_key="k", library_id="123", library_type="user", http_client=http)
        with pytest.raises(ZoteroAPIError):
            await c.create_note(parent_item_key="P1", content="x")


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------

def test_token_bucket_basic_consume():
    """A burst of N tokens should be consumable immediately."""
    import asyncio
    bucket = _TokenBucket(rate_per_sec=10.0, burst=3.0)

    async def consume_three():
        for _ in range(3):
            await bucket.acquire()
        return True

    assert asyncio.run(consume_three()) is True


@pytest.mark.asyncio
async def test_token_bucket_blocks_when_burst_exhausted():
    """After consuming the burst, the next acquire should wait
    ~(1/rate) seconds. We use a high rate so the test is fast but
    measurable."""
    import time
    bucket = _TokenBucket(rate_per_sec=50.0, burst=1.0)
    await bucket.acquire()  # consume the only burst token
    t0 = time.monotonic()
    await bucket.acquire()  # should sleep ~20ms
    elapsed = time.monotonic() - t0
    # Allow generous slack (CI noise) but reject "no wait at all"
    assert 0.005 < elapsed < 0.5
