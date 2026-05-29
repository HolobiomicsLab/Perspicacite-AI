"""ASB-aligned PDF figure extraction.

Synced from AgenticScienceBuilder @ 809f478 — keep API in sync.

PyMuPDF rasterizes each embedded image individually, but a single scientific
figure is often a composite (Figure 1A/1B/1C panels). This module pairs every
extracted image with its parent caption and, when the caption enumerates
panels, assigns each image a ``subcomponent_label`` (A/B/…) by spatial
position (row-major).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.parsers.figures")

_CAPTION_PREFIXES = ("fig", "figure", "scheme", "supplementary")
_MAX_CAPTION_DISTANCE_PT = 120  # was 80; captions sometimes sit further below
_MAX_ASPECT_RATIO = 10.0
_MIN_ASPECT_RATIO = 0.1
_MIN_AREA_PX = 50_000  # drop small icons / decorations (< 224x224 equivalent)

# "Figure 1.", "Fig. 2:", "Scheme 3", "Supplementary Figure S1" — captures
# the alphanumeric label after the prefix.
_FIGURE_NUMBER_RE = re.compile(
    r"^\s*(?:supplementary\s+)?(?:fig(?:ure|\.)?|scheme)\s+([A-Za-z]?\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)
# Panel markers like "(A)", "(B)", "(a)", "(i)", "(ii)" appearing in the
# caption body. We capture the inner token verbatim and de-duplicate
# preserving order.
_PANEL_MARKER_RE = re.compile(r"\((?P<label>[A-Za-z]{1,3}|\d{1,2})\)")


@dataclass
class FigureRecord:
    source_pdf: str
    page: int
    index: int
    width_px: int
    height_px: int
    caption: str
    filename: str
    ext: str
    # Sub-component fields (None when undetectable):
    figure_number: str | None = None
    subcomponent_label: str | None = None
    # Bounding box (x0, y0, x1, y1) in PDF user-space points. Used by the
    # post-processor to assign panel labels by spatial position; carried
    # forward to index.json for downstream tooling.
    bbox: tuple[float, float, float, float] | None = None
    # Sprint 8: per-panel crop records set by crop_panels().  Each entry is
    # {label, filename, w, h}.  Empty until crop_panels() runs.
    panel_files: list[dict] = field(default_factory=list)


@dataclass
class RawFigure:
    record: FigureRecord
    image_bytes: bytes


def extract_figures(pdf_path: Path, min_px: int = 100) -> list[RawFigure]:
    """Extract embedded images from a PDF. Returns [] on any failure."""
    try:
        return _extract_via_pymupdf(pdf_path, min_px)
    except ImportError:
        return []
    except Exception:
        return []


def _extract_via_pymupdf(pdf_path: Path, min_px: int) -> list[RawFigure]:
    import fitz

    doc = fitz.open(str(pdf_path))
    results: list[RawFigure] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            img_idx = 0
            page_results: list[RawFigure] = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                w = base_image["width"]
                h = base_image["height"]
                if w < min_px or h < min_px:
                    continue
                if w * h < _MIN_AREA_PX:
                    continue
                ratio = w / h if h > 0 else 0.0
                if ratio > _MAX_ASPECT_RATIO or ratio < _MIN_ASPECT_RATIO:
                    continue
                img_idx += 1
                ext = base_image.get("ext", "png")
                image_bytes = base_image["image"]
                # Convert non-RGB colorspaces (CMYK, Lab, etc.) to RGB so browsers
                # render correctly — native CMYK PNGs appear as solid black boxes.
                cs_name = base_image.get("cs-name", "")
                if cs_name and cs_name not in ("DeviceRGB", "sRGB", ""):
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.colorspace and pix.colorspace.n != 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        image_bytes = pix.tobytes("png")
                        ext = "png"
                    except Exception as exc:
                        logger.debug("colorspace_conversion_failed", cs_name=cs_name, error=str(exc))  # keep original bytes
                # Skip images that are essentially blank (< 1 KB after encoding).
                if len(image_bytes) < 1024:
                    continue
                bbox = None
                bbox_tuple: tuple[float, float, float, float] | None = None
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        bbox = rects[0]
                        bbox_tuple = (
                            float(bbox.x0), float(bbox.y0),
                            float(bbox.x1), float(bbox.y1),
                        )
                except Exception as exc:
                    logger.debug("image_rect_lookup_failed", xref=xref, error=str(exc))
                caption = _detect_caption(page, bbox) if bbox is not None else ""
                filename = f"fig_{page_num:03d}_i{img_idx:02d}.{ext}"
                page_results.append(
                    RawFigure(
                        record=FigureRecord(
                            source_pdf=pdf_path.name,
                            page=page_num,
                            index=img_idx,
                            width_px=w,
                            height_px=h,
                            caption=caption,
                            filename=filename,
                            ext=ext,
                            bbox=bbox_tuple,
                        ),
                        image_bytes=image_bytes,
                    )
                )
            # Post-process this page's images: parse figure numbers and
            # assign sub-component labels when multiple panels share a
            # caption.
            assign_subcomponents(page_results)
            results.extend(page_results)
    finally:
        doc.close()
    return results


# ---------------------------------------------------------------------------
# Sub-component / panel-label post-processing (pure, testable)
# ---------------------------------------------------------------------------


def parse_figure_number(caption: str) -> str | None:
    """Return ``"1"``, ``"S1"``, etc. parsed from a caption prefix; else None."""
    if not caption:
        return None
    m = _FIGURE_NUMBER_RE.match(caption)
    return m.group(1) if m else None


def parse_panel_labels(caption: str) -> list[str]:
    """Return the panel-marker tokens in caption order, deduplicated.

    Examples
    --------
    >>> parse_panel_labels("Figure 1. (A) Methanol; (B) ethanol; (C) water.")
    ['A', 'B', 'C']
    >>> parse_panel_labels("Fig 2: panels (a)-(d) summarize…")
    ['a', 'd']
    """
    if not caption:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for m in _PANEL_MARKER_RE.finditer(caption):
        tok = m.group("label")
        if tok in seen:
            continue
        seen.add(tok)
        labels.append(tok)
    return labels


def assign_subcomponents(page_results: list[RawFigure]) -> None:
    """In-place: tag every record with figure_number + subcomponent_label.

    Strategy
    --------
    1. Group images by ``record.caption`` (when non-empty). Images with the
       same caption are interpreted as panels of one composite figure.
    2. For each group, parse the caption once for ``figure_number`` and
       ``parse_panel_labels``. If the panel-label list has at least as
       many entries as the group, assign labels by spatial position
       (row-major: ascending y0 with a row-tolerance bucket, then
       ascending x0 within each row). Otherwise leave
       ``subcomponent_label`` as None.
    3. Singleton groups still get ``figure_number`` populated from the
       caption when available.
    """
    if not page_results:
        return

    # Bucket by caption (empty captions skip grouping but still get
    # figure_number=None — unchanged).
    by_caption: dict[str, list[RawFigure]] = {}
    for rf in page_results:
        cap = rf.record.caption or ""
        by_caption.setdefault(cap, []).append(rf)

    for caption, group in by_caption.items():
        fig_num = parse_figure_number(caption)
        for rf in group:
            rf.record.figure_number = fig_num
        if len(group) <= 1:
            continue
        panels = parse_panel_labels(caption)
        if len(panels) < len(group):
            continue
        # Sort by spatial position. Use a row-tolerance equal to the
        # smallest image height in the group (works for typical 2x2 / 1xN
        # grid layouts; falls back gracefully on degenerate bboxes).
        sortable = [rf for rf in group if rf.record.bbox is not None]
        if len(sortable) < len(group):
            # Some images have no bbox; can't position-assign reliably.
            continue
        min_h = min((rf.record.bbox[3] - rf.record.bbox[1]) for rf in sortable if rf.record.bbox is not None) or 1.0
        row_tol = max(min_h * 0.5, 5.0)

        def _key(rf: RawFigure, _row_tol: float = row_tol) -> tuple[int, float]:
            x0, y0, _, _ = rf.record.bbox  # type: ignore[misc]
            row_bucket = int(y0 // _row_tol)
            return (row_bucket, x0)

        ordered = sorted(sortable, key=_key)
        # strict=False: panels may be shorter than ordered; we label as many
        # figures as we have panel labels for and leave the rest unlabeled.
        for rf, label in zip(ordered, panels[: len(ordered)], strict=False):
            rf.record.subcomponent_label = label


def _detect_caption_via_ocr(page, bbox) -> str:
    """Run OCR on a strip below *bbox* and return the first caption-like line.

    Requires ``pytesseract`` and ``Pillow`` (optional deps). Returns "" on
    ``ImportError`` or any failure. Enabled only when ``ASB_OCR=1`` is set in
    the environment (OCR is slow; off by default).
    """
    if not os.environ.get("ASB_OCR"):
        return ""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        import fitz
        strip_rect = fitz.Rect(
            bbox.x0,
            bbox.y1,
            bbox.x1,
            bbox.y1 + _MAX_CAPTION_DISTANCE_PT,
        )
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
        pix = page.get_pixmap(matrix=mat, clip=strip_rect)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        ocr_text = pytesseract.image_to_string(img)
        for line in ocr_text.splitlines():
            if line.strip().lower().startswith(_CAPTION_PREFIXES):
                return line.strip()
    except Exception as exc:
        logger.debug("ocr_caption_failed", error=str(exc))
    return ""


def _detect_caption(page, bbox) -> str:
    """Return the first text line immediately below the image bounding box."""
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except (AttributeError, TypeError, KeyError):
        return ""

    img_bottom = bbox.y1
    img_left = bbox.x0
    img_right = bbox.x1

    candidates: list[tuple[float, str]] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, _ = block["bbox"]
        if by0 < img_bottom or by0 > img_bottom + _MAX_CAPTION_DISTANCE_PT:
            continue
        overlap = min(bx1, img_right) - max(bx0, img_left)
        if overlap < 20:
            continue
        text = " ".join(
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if text:
            candidates.append((by0, text))

    if not candidates:
        # No text blocks found under the figure — try OCR if enabled.
        return _detect_caption_via_ocr(page, bbox)

    candidates.sort()
    for _, text in candidates:
        if text.lower().startswith(_CAPTION_PREFIXES):
            return text
    return candidates[0][1]
