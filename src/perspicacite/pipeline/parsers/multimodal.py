"""Multimodal PDF visual extractor — figures, tables, formulas (Wave 4.1).

Renders each PDF page as PNG via PyMuPDF, sends to a vision-capable
LLM, and parses the structured JSON response into VisualExtract
records.

See docs/superpowers/specs/2026-05-14-multimodal-pdf-extraction-design.md.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.parsers.multimodal")


_VALID_KINDS: frozenset[str] = frozenset({"figure", "table", "formula"})


_EXTRACTION_PROMPT = """\
You are extracting figures, tables, and formulas from a single page of a
scientific PDF. The image shows the page rendered at the requested DPI.

Return a JSON object with this exact shape:

{
  "visuals": [
    {
      "kind": "figure" | "table" | "formula",
      "page": <page-number-as-int>,
      "label": "<figure/table number e.g. 'Figure 3' or 'Table 1'>",
      "caption": "<full caption text>",
      "content": "<markdown for tables, plain-text description for figures, latex for formulas>"
    }
  ]
}

Rules:
- "figure": include a 1-3 sentence semantic description of what the figure
  shows (axes, trends, comparison). Don't repeat the caption text in
  "content" — that goes in "caption".
- "table": render as a markdown table in "content". Preserve column headers
  and row labels exactly. For huge tables, summarise the structure.
- "formula": only include numbered display equations. "content" is
  best-effort LaTeX.
- Return {"visuals": []} when the page has no figures, tables, or numbered
  formulas.
- Output VALID JSON only. No markdown fences, no commentary.
"""


@dataclass
class VisualExtract:
    """One figure/table/formula extracted from a PDF page."""

    kind: Literal["figure", "table", "formula"]
    page: int
    label: str
    caption: str
    content: str


class MultimodalPDFExtractor:
    """Render PDF pages and ask a multimodal LLM what's on them.

    Stateless — one instance can process many PDFs. The LLM disk cache
    (Wave 2.1) keys on the rendered PNG bytes, so re-running on the
    same PDF / DPI hits cache.
    """

    def __init__(
        self,
        *,
        llm_client: Any,
        model: str,
        provider: str,
        dpi: int = 150,
    ):
        self.llm_client = llm_client
        self.model = model
        self.provider = provider
        self.dpi = dpi

    # ---- internals: rendering -----------------------------------------

    def _open_pdf(self, pdf_path: Path) -> Any:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for multimodal extraction. "
                "Install with: pip install pymupdf"
            ) from exc
        return fitz.open(str(pdf_path))

    def _page_count(self, pdf_path: Path) -> int:
        doc = self._open_pdf(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    def _render_png(self, page_num: int) -> bytes:
        """Render a single page (1-indexed) as PNG bytes.

        Reads ``self._active_pdf_path`` which is set by ``extract_visuals``
        before the rendering loop. The test suite mocks this method at the
        instance level — the signature takes only ``page_num`` so mock
        side_effect callables with a single positional arg work without
        modification.
        """
        pdf_path: Path | None = getattr(self, "_active_pdf_path", None)
        if pdf_path is None:
            raise ValueError("_active_pdf_path not set; call extract_visuals")
        doc = self._open_pdf(pdf_path)
        try:
            page = doc.load_page(page_num - 1)  # fitz is 0-indexed
            pix = page.get_pixmap(dpi=self.dpi)
            return pix.tobytes("png")
        finally:
            doc.close()

    # ---- internals: response parsing ----------------------------------

    def _parse_response(self, raw: str) -> list[VisualExtract]:
        """Parse the LLM's JSON response into VisualExtract records.

        Robust against malformed JSON, missing keys, and unknown
        ``kind`` values. Returns ``[]`` on any parse failure rather
        than raising — one bad page shouldn't kill a multi-page run.
        """
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("multimodal_parse_json_decode_failed",
                           preview=raw[:200])
            return []
        if not isinstance(payload, dict):
            return []
        visuals_raw = payload.get("visuals")
        if not isinstance(visuals_raw, list):
            return []
        out: list[VisualExtract] = []
        for item in visuals_raw:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind not in _VALID_KINDS:
                continue
            try:
                out.append(VisualExtract(
                    kind=kind,
                    page=int(item.get("page", 0)),
                    label=str(item.get("label", "")),
                    caption=str(item.get("caption", "")),
                    content=str(item.get("content", "")),
                ))
            except (TypeError, ValueError):
                continue
        return out

    # ---- public API ---------------------------------------------------

    async def extract_visuals(
        self,
        *,
        pdf_path: Path,
        paper_id: str,
        page_range: tuple[int, int] | None = None,
    ) -> list[VisualExtract]:
        """Extract all visuals from a PDF.

        ``page_range`` is ``(start, end)`` inclusive, 1-indexed.
        ``None`` means all pages.

        Per-page failures are logged + skipped — the run continues on
        the next page so a single corrupt page doesn't lose the rest.
        """
        # Store so _render_png can open the right file without a signature
        # that clashes with test mocks (see _render_png docstring).
        self._active_pdf_path = pdf_path

        n = self._page_count(pdf_path)
        if page_range is None:
            page_nums = list(range(1, n + 1))
        else:
            start, end = page_range
            page_nums = list(range(max(1, start), min(n, end) + 1))

        all_visuals: list[VisualExtract] = []
        for page_num in page_nums:
            try:
                png_bytes = self._render_png(page_num)
            except Exception as e:
                logger.warning(
                    "multimodal_render_failed",
                    paper_id=paper_id, page=page_num, error=str(e),
                )
                continue
            page_visuals = await self._extract_one_page(
                png_bytes=png_bytes,
                page_num=page_num,
                paper_id=paper_id,
            )
            all_visuals.extend(page_visuals)
        return all_visuals

    async def _extract_one_page(
        self,
        *,
        png_bytes: bytes,
        page_num: int,
        paper_id: str,
    ) -> list[VisualExtract]:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        # Anthropic-style image content block; OpenAI accepts a similar
        # shape via litellm's drop_params. For agent_cli providers
        # without multimodal support, this call will fail — caller is
        # expected to route to a vision-capable provider.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            _EXTRACTION_PROMPT
                            + f"\n\n(This is page {page_num} of paper "
                            + f"`{paper_id}`.)"
                        ),
                    },
                ],
            },
        ]
        try:
            response = await self.llm_client.complete(
                messages=messages,
                model=self.model,
                provider=self.provider,
                temperature=0.0,
                max_tokens=4000,
                stage="multimodal_extraction",
            )
        except Exception as e:
            logger.warning(
                "multimodal_llm_failed",
                paper_id=paper_id, page=page_num,
                error=str(e), error_type=type(e).__name__,
            )
            return []
        return self._parse_response(response)

    def to_chunks(
        self,
        visuals: list[VisualExtract],
        *,
        paper_id: str,
        chunk_index_offset: int = 0,
    ) -> list[Any]:
        """Wrap each VisualExtract in a DocumentChunk.

        ``chunk_index_offset`` lets callers continue the index numbering
        after the text chunks in the same paper.
        """
        from perspicacite.models.documents import ChunkMetadata, DocumentChunk
        chunks: list[Any] = []
        for i, v in enumerate(visuals):
            idx = chunk_index_offset + i
            text = f"{v.label}\n{v.caption}\n\n{v.content}".strip()
            md = ChunkMetadata(
                paper_id=paper_id,
                chunk_index=idx,
                section=v.label,
                page_number=v.page,
                content_type=v.kind,
            )
            chunks.append(DocumentChunk(
                id=f"{paper_id}__visual_{v.page}_{i}",
                text=text,
                metadata=md,
            ))
        return chunks
