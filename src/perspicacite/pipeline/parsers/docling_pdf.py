"""Docling-backed PDF extraction (R2).

Ports the converter configuration proven in AgenticScienceBuilder's
figures.py: picture images MUST be rendered (generate_picture_images=True)
or PictureItem.get_image() returns None and every figure is dropped; figure
pixel dimensions MUST be read from the rendered image or the size filter
discards them. No dependency on ASB.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

from perspicacite.logging import get_logger
from perspicacite.pipeline.parsers.pdf import ParsedContent

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = get_logger("perspicacite.pipeline.parsers.docling")

_MIN_AREA_PX = 50_000  # drop logos/icons (mirrors ASB)


@dataclass
class DoclingTable:
    page: int
    caption: str
    markdown: str
    headers: list[str]
    rows: list[list[str]]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)


@dataclass
class DoclingFigure:
    page: int
    caption: str
    width_px: int
    height_px: int
    image_bytes: bytes = b""


def docling_importable() -> bool:
    return importlib.util.find_spec("docling") is not None


def _make_docling_converter():
    # Picture images MUST be enabled or get_image() returns None (zero figures).
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    # Force CPU. On Apple Silicon docling auto-selects the MPS (Metal) backend,
    # which raises "Cannot convert a MPS Tensor to float64 ... MPS doesn't
    # support float64" and fails conversion on every page. CPU is portable and
    # matches the documented R2 device intent.
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _page_of(item) -> int:
    prov = getattr(item, "prov", None) or []
    if prov and getattr(prov[0], "page_no", None) is not None:
        return int(prov[0].page_no)
    return 1


_FIG_LABEL_RE = re.compile(
    r"^\s*((?:supplementary\s+)?(?:fig(?:ure|\.)?|scheme)\s+[A-Za-z]?\d+[A-Za-z]?)",
    re.IGNORECASE,
)


def figure_to_multimodal_record(fig: DoclingFigure) -> dict:
    """Map a DoclingFigure to the existing multimodal record shape
    {kind, label, caption, content} used by parsers/multimodal.py. `content`
    is left empty: docling supplies the image, not a semantic description."""
    m = _FIG_LABEL_RE.match(fig.caption or "")
    label = m.group(1).strip() if m else ""
    return {"kind": "figure", "label": label, "caption": fig.caption or "", "content": ""}


class DoclingPDFParser:
    """Extracts text + structured tables + figures via docling."""

    def __init__(self, converter_factory: Callable[[], Any] = _make_docling_converter):
        self._converter_factory = converter_factory

    def extract(self, source: str | Path) -> ParsedContent:
        conv = self._converter_factory()
        doc = conv.convert(str(source)).document
        figures = self._figures(doc)
        tables = self._tables(doc)
        text = self._text(doc)
        return ParsedContent(
            text=text,
            sections=None,
            metadata={"extractor": "docling"},
            tables=tables,
            figures=figures,
        )

    def _text(self, doc) -> str:
        try:
            return doc.export_to_markdown()
        except Exception:
            return ""

    def _figures(self, doc) -> list[DoclingFigure]:
        out: list[DoclingFigure] = []
        for pic in getattr(doc, "pictures", []) or []:
            try:
                pil = pic.get_image(doc)
                w, h = pil.width, pil.height
                buf = BytesIO()
                pil.save(buf, "PNG")
                image_bytes = buf.getvalue()
            except Exception:
                continue
            if len(image_bytes) < 1024:
                continue
            try:
                caption = pic.caption_text(doc) or ""
            except Exception:
                caption = ""
            out.append(
                DoclingFigure(
                    page=_page_of(pic), caption=caption,
                    width_px=w, height_px=h, image_bytes=image_bytes,
                )
            )
        return out

    def _tables(self, doc) -> list[DoclingTable]:
        out: list[DoclingTable] = []
        for tbl in getattr(doc, "tables", []) or []:
            try:
                df = tbl.export_to_dataframe(doc)
                headers = [str(c) for c in df.columns.tolist()]
                rows = [[str(v) for v in row] for row in df.values.tolist()]
                markdown = tbl.export_to_markdown(doc)
            except Exception:
                continue
            try:
                caption = tbl.caption_text(doc) or ""
            except Exception:
                caption = ""
            out.append(
                DoclingTable(
                    page=_page_of(tbl), caption=caption,
                    markdown=markdown, headers=headers, rows=rows,
                )
            )
        return out
