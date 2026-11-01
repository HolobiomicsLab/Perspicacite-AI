"""URL → paper-metadata extractors for the ``ingest_url`` MCP tool.

Four extractor branches:

- :func:`extract_github`         — github.com/<owner>/<repo> via the public API.
- :func:`extract_openreview`     — openreview.net/forum?id=<id> via api.openreview.net.
- :func:`extract_preprints_org`  — preprints.org/manuscript/<id>(/v<n>) — find
  the direct PDF URL embedded in the ``<meta name="citation_pdf_url">`` tag.
- :func:`extract_generic_html`   — anything else: scrape ``<meta name="citation_*">``,
  OpenGraph tags, and ``<title>`` for a best-effort metadata record.

Each extractor returns a normalized dict matching the input shape
:meth:`ZoteroClient.create_item` expects (``title``, ``authors``, ``url``,
optional ``year``, ``abstract``, ``doi``, ``item_type``, ``repository``,
etc.) plus an ``ingest_format`` flag describing what we got.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download.url_extractors")


_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/([\w.-]+)/([\w.-]+?)(?:/.*)?/?$",
    re.IGNORECASE,
)
_OPENREVIEW_RE = re.compile(
    r"^https?://openreview\.net/forum\?id=([\w-]+)",
    re.IGNORECASE,
)
_PREPRINTS_ORG_RE = re.compile(
    r"^https?://(?:www\.)?preprints\.org/manuscript/(\d+(?:\.\d+)?)(?:/v\d+)?/?",
    re.IGNORECASE,
)


def classify_url(url: str) -> str:
    """Return one of: github, openreview, preprints_org, generic."""
    if not url:
        return "generic"
    if _GITHUB_REPO_RE.match(url):
        return "github"
    if _OPENREVIEW_RE.match(url):
        return "openreview"
    if _PREPRINTS_ORG_RE.match(url):
        return "preprints_org"
    return "generic"


async def extract_github(url: str, *, http_client: httpx.AsyncClient) -> dict[str, Any]:
    """GitHub repo → repo + README metadata via the public API.

    Public repos: no auth needed. Returns a paper dict with
    ``item_type=computerProgram``, the README in ``abstract``, and
    the top contributors as ``authors``.
    """
    m = _GITHUB_REPO_RE.match(url)
    if not m:
        raise ValueError(f"not a github URL: {url}")
    owner, repo = m.group(1), m.group(2).rstrip(".git")
    api = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        r = await http_client.get(api, timeout=10.0, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        repo_data = r.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"GitHub API error for {owner}/{repo}: {exc}") from exc

    # Pull the README — GitHub returns the raw decoded content when
    # ``Accept: application/vnd.github.raw`` is used.
    readme_text = ""
    try:
        rr = await http_client.get(
            f"{api}/readme",
            timeout=10.0,
            headers={"Accept": "application/vnd.github.raw"},
        )
        if rr.status_code == 200:
            readme_text = rr.text
    except httpx.HTTPError:
        pass

    # Top 5 contributors → authors
    authors: list[str] = []
    try:
        rc = await http_client.get(
            f"{api}/contributors", params={"per_page": 5}, timeout=10.0,
            headers={"Accept": "application/vnd.github+json"},
        )
        if rc.status_code == 200:
            authors = [c.get("login") for c in (rc.json() or []) if c.get("login")]
    except httpx.HTTPError:
        pass
    if not authors:
        # Fall back to owner if contributors aren't accessible
        authors = [owner]

    description = repo_data.get("description") or ""
    return {
        "url": f"https://github.com/{owner}/{repo}",
        "title": f"{owner}/{repo}",
        "authors": authors,
        "abstract": description + ("\n\n" + readme_text if readme_text else ""),
        "item_type": "computerProgram",
        "year": (repo_data.get("created_at") or "")[:4],
        "repository": "GitHub",
        "programming_language": repo_data.get("language") or "",
        "version": repo_data.get("default_branch") or "",
        "tags": [t for t in (repo_data.get("topics") or []) if t][:10],
        "ingest_format": "github_api",
    }


async def extract_openreview(url: str, *, http_client: httpx.AsyncClient) -> dict[str, Any]:
    """OpenReview forum → title, authors, abstract, PDF URL."""
    m = _OPENREVIEW_RE.match(url)
    if not m:
        raise ValueError(f"not an openreview URL: {url}")
    note_id = m.group(1)
    api_url = f"https://api.openreview.net/notes?id={note_id}"
    try:
        r = await http_client.get(api_url, timeout=10.0)
        r.raise_for_status()
        notes = (r.json() or {}).get("notes") or []
    except httpx.HTTPError as exc:
        raise RuntimeError(f"OpenReview API error for {note_id}: {exc}") from exc
    if not notes:
        raise RuntimeError(f"OpenReview note not found: {note_id}")
    note = notes[0]
    content = note.get("content") or {}
    # OpenReview API v2 wraps fields in {"value": ...}; v1 is flat. Handle both.
    def _v(field: str) -> str:
        x = content.get(field)
        if isinstance(x, dict):
            return str(x.get("value") or "")
        return str(x or "")
    title = _v("title")
    abstract = _v("abstract")
    authors_field = content.get("authors")
    if isinstance(authors_field, dict):
        authors_field = authors_field.get("value")
    authors = list(authors_field or [])
    pdf_path = _v("pdf")  # e.g., "/pdf?id=..."
    pdf_url = f"https://openreview.net{pdf_path}" if pdf_path else ""
    venue = _v("venue") or _v("venueid")
    return {
        "url": url,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "item_type": "preprint",
        "repository": venue or "OpenReview",
        "archive_id": note_id,
        "pdf_url": pdf_url,
        "ingest_format": "openreview_api",
    }


class _MetaTagCollector(HTMLParser):
    """Collect <meta> and <title> tags for citation_* / OpenGraph extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, list[str]] = {}
        self.title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
            return
        if tag.lower() != "meta":
            return
        d = {(k or "").lower(): (v or "") for k, v in attrs}
        key = d.get("name") or d.get("property") or ""
        content = d.get("content") or ""
        if not key or not content:
            return
        self.meta.setdefault(key.lower(), []).append(content)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and self.title is None:
            self.title = data.strip()


def _first(meta: dict[str, list[str]], *keys: str) -> str:
    for k in keys:
        v = meta.get(k.lower())
        if v:
            return v[0]
    return ""


def _all(meta: dict[str, list[str]], *keys: str) -> list[str]:
    out: list[str] = []
    for k in keys:
        out.extend(meta.get(k.lower()) or [])
    return out


class _RepositoryLinkCollector(_MetaTagCollector):
    """Collect citation metadata and repository download links from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag.lower() == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def extract_repository_pdf_url(html: str, landing_url: str) -> str | None:
    """Find one HTTP PDF link in already-fetched repository HTML.

    Citation metadata takes precedence over DSpace bitstream and HAL document
    links. Relative links resolve against the final landing URL. No fetching
    occurs here; the caller limits retrieval to one additional hop.
    """
    parser = _RepositoryLinkCollector()
    parser.feed(html)
    candidates = list(parser.meta.get("citation_pdf_url", []))
    candidates.extend(link for link in parser.links if _is_repository_download(link))
    for candidate in candidates:
        resolved = urldefrag(urljoin(landing_url, candidate.strip()))[0]
        if urlparse(resolved).scheme not in {"http", "https"}:
            continue
        if resolved != urldefrag(landing_url)[0]:
            return resolved
    return None


def _is_repository_download(url: str) -> bool:
    """Recognise DSpace bitstream and HAL document paths."""
    path = urlparse(url).path.rstrip("/")
    return (
        "/bitstream/" in path
        or bool(re.search(r"/bitstreams/[^/]+/download$", path))
        or path.endswith("/document")
    )


async def extract_generic_html(url: str, *, http_client: httpx.AsyncClient) -> dict[str, Any]:
    """Generic HTML page → metadata from citation_* / OpenGraph / <title>.

    Most academic publishers and many blogs emit ``<meta name='citation_title'>``
    / ``citation_author`` / ``citation_publication_date`` tags (the Google
    Scholar Dublin Core profile). Some emit ``<meta property='og:*'>`` for
    sharing. We mine both and fall back to the page title.
    """
    # Publisher landing pages gate non-browser UAs with 403; send a
    # realistic Chrome/Mac UA matching the rest of the pipeline.
    try:
        r = await http_client.get(
            url, timeout=15.0, follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"
                ),
            },
        )
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"HTML fetch error: {exc}") from exc
    if "html" not in r.headers.get("content-type", "").lower():
        raise RuntimeError(f"not an HTML page (content-type={r.headers.get('content-type')})")

    parser = _MetaTagCollector()
    try:
        parser.feed(r.text)
    except Exception as exc:
        raise RuntimeError(f"HTML parse error: {exc}") from exc

    title = (
        _first(parser.meta, "citation_title", "dc.title", "og:title", "twitter:title")
        or parser.title
        or urlparse(url).path.strip("/").split("/")[-1]
        or url
    )
    authors_raw = _all(parser.meta, "citation_author", "dc.creator", "author")
    abstract = _first(
        parser.meta,
        "citation_abstract", "dc.description", "description",
        "og:description", "twitter:description",
    )
    pub_date = _first(parser.meta, "citation_publication_date", "citation_date", "dc.date", "article:published_time")
    doi = _first(parser.meta, "citation_doi", "dc.identifier", "prism.doi")
    if doi:
        doi = doi.replace("doi:", "").strip().rstrip(".")
    journal = _first(parser.meta, "citation_journal_title", "citation_conference_title", "og:site_name")
    pdf_url = _first(parser.meta, "citation_pdf_url")

    # arXiv pages don't emit citation_doi, but they do emit citation_arxiv_id.
    # The arxiv→DOI mapping is the standard 10.48550/arXiv.<id> form, which
    # downstream code (push_to_zotero item_type routing) already handles
    # as a preprint.
    arxiv_id = _first(parser.meta, "citation_arxiv_id")
    repository = None
    archive_id = None
    item_type = "journalArticle" if doi else "webpage"
    if arxiv_id and not doi:
        doi = f"10.48550/arXiv.{arxiv_id}"
        repository = "arXiv"
        archive_id = arxiv_id
        item_type = "preprint"
    elif arxiv_id:
        repository = "arXiv"
        archive_id = arxiv_id

    out: dict[str, Any] = {
        "url": url,
        "title": title,
        "authors": [a for a in authors_raw if a],
        "abstract": abstract,
        "year": (pub_date or "")[:4],
        "doi": doi or "",
        "journal": journal,
        "item_type": item_type,
        "pdf_url": pdf_url,
        "ingest_format": "html_meta_tags",
    }
    if repository:
        out["repository"] = repository
    if archive_id:
        out["archive_id"] = archive_id
    return out


async def extract_preprints_org(url: str, *, http_client: httpx.AsyncClient) -> dict[str, Any]:
    """preprints.org pages have a ``citation_pdf_url`` meta tag with a
    token-signed direct PDF URL that works without Cloudflare. Reuse the
    generic extractor and just tag the format."""
    paper = await extract_generic_html(url, http_client=http_client)
    paper["item_type"] = "preprint"
    paper["repository"] = paper.get("journal") or "preprints.org"
    paper["ingest_format"] = "preprints_org"
    return paper


async def extract_url(url: str, *, http_client: httpx.AsyncClient) -> dict[str, Any]:
    """Dispatch to the right extractor by URL pattern."""
    kind = classify_url(url)
    if kind == "github":
        return await extract_github(url, http_client=http_client)
    if kind == "openreview":
        return await extract_openreview(url, http_client=http_client)
    if kind == "preprints_org":
        return await extract_preprints_org(url, http_client=http_client)
    return await extract_generic_html(url, http_client=http_client)
