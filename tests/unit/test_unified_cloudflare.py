"""Cloudflare challenge handling across ranked OA PDF candidates."""

from unittest.mock import AsyncMock

import httpx
import pytest

from perspicacite.pipeline.download import unified
from perspicacite.pipeline.download.base import MIN_PLAUSIBLE_PDF_BYTES, PaperDiscovery

BLOCKED_URL = "https://publisher.test/article.pdf"
CHALLENGE_FINAL_URL = "https://publisher.test/cdn-cgi/challenge-platform/redirect"
REPOSITORY_PDF_URL = "https://repository.test/article.pdf"
CLOUDFLARE_CHALLENGE_BODY = (
    b"<!DOCTYPE html><title>Just a moment...</title>"
    b'<script src="/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1"></script>'
    b'<div id="cf-chl-widget"></div>' + b" " * MIN_PLAUSIBLE_PDF_BYTES
)
VALID_PDF_BODY = b"%PDF-1.7\n" + b"0" * MIN_PLAUSIBLE_PDF_BYTES


def _response(status: int, url: str, body: bytes, content_type: str) -> httpx.Response:
    """Build a response whose request URL represents its final redirect URL."""
    return httpx.Response(
        status,
        headers={"content-type": content_type},
        content=body,
        request=httpx.Request("GET", url),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("challenge_status", [200, 403])
async def test_cloudflare_challenge_advances_to_repository_pdf(
    challenge_status: int,
) -> None:
    """A 200 or 403 challenge is recorded once before the next PDF wins."""
    client = AsyncMock()
    client.get = AsyncMock(
        side_effect=[
            _response(
                challenge_status,
                CHALLENGE_FINAL_URL,
                CLOUDFLARE_CHALLENGE_BODY,
                "text/html; charset=UTF-8",
            ),
            _response(200, REPOSITORY_PDF_URL, VALID_PDF_BODY, "application/pdf"),
        ]
    )
    attempts: list[dict[str, object]] = []
    discovery = PaperDiscovery(
        doi="10.1/cloudflare",
        oa_candidates=[BLOCKED_URL, REPOSITORY_PDF_URL],
    )

    result = await unified._try_pdf_sources(
        "10.1/cloudflare", None, client, discovery, attempts=attempts
    )

    assert result == (VALID_PDF_BODY, "publisher_oa_pdf")
    assert client.get.await_count == 2
    assert [call.args[0] for call in client.get.await_args_list] == [
        BLOCKED_URL,
        REPOSITORY_PDF_URL,
    ]
    assert len(attempts) == 1
    assert attempts[0]["source"] == "publisher_oa_pdf"
    assert attempts[0]["status"] == "blocked"
    assert attempts[0]["marker"] == "cloudflare_challenge"
    assert attempts[0]["url"] == BLOCKED_URL
