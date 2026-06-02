"""Docling-backed PDF extraction (R2).

Ports the converter configuration proven in AgenticScienceBuilder's
figures.py: picture images MUST be rendered (generate_picture_images=True)
or PictureItem.get_image() returns None and every figure is dropped; figure
pixel dimensions MUST be read from the rendered image or the size filter
discards them. No dependency on ASB.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass

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
