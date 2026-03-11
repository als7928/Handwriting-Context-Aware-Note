"""EasyOCR wrapper for handwritten text recognition inside PDF regions."""

from __future__ import annotations

import logging

import pymupdf as fitz

logger = logging.getLogger(__name__)

# None  = not yet initialised
# False = initialisation failed / easyocr not installed
# <Reader> = ready
_reader_initialized: bool = False
_reader = None


def _get_reader():
    """Return an EasyOCR Reader; returns None when unavailable."""
    global _reader_initialized, _reader
    if _reader_initialized:
        return _reader
    _reader_initialized = True
    try:
        import easyocr  # noqa: PLC0415

        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        logger.info("EasyOCR initialised (Korean + English, CPU mode)")
    except ImportError:
        logger.warning("easyocr not installed – handwriting OCR disabled")
    except Exception as exc:
        logger.warning("EasyOCR init failed: %s", exc)
    return _reader


def ocr_region(
    page: fitz.Page,
    rect: fitz.Rect,
    dpi: int = 200,
    padding: float = 15.0,
) -> str:
    """Render a PDF region and return OCR'd text.

    Returns an empty string when EasyOCR is unavailable or produces no output.
    """
    reader = _get_reader()
    if reader is None:
        return ""
    try:
        import numpy as np  # noqa: PLC0415

        padded = fitz.Rect(
            max(page.rect.x0, rect.x0 - padding),
            max(page.rect.y0, rect.y0 - padding),
            min(page.rect.x1, rect.x1 + padding),
            min(page.rect.y1, rect.y1 + padding),
        )
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=padded, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
        results = reader.readtext(img, detail=0, paragraph=True)
        return " ".join(str(r) for r in results).strip()
    except Exception as exc:
        logger.debug("EasyOCR region error: %s", exc)
        return ""


def ocr_page(page: fitz.Page, dpi: int = 150) -> list[tuple[str, fitz.Rect]]:
    """OCR a full page; returns list of (text, rect_in_page_coords).

    Used for pages that contain no selectable text (scanned / image PDFs or
    fully handwritten pages).
    """
    reader = _get_reader()
    if reader is None:
        return []
    try:
        import numpy as np  # noqa: PLC0415

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
        results = reader.readtext(img, detail=1)
        scale = 72.0 / dpi
        output: list[tuple[str, fitz.Rect]] = []
        for bbox_pts, text, conf in results:
            if conf < 0.3 or not (text or "").strip():
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            rect = fitz.Rect(
                min(xs) * scale,
                min(ys) * scale,
                max(xs) * scale,
                max(ys) * scale,
            )
            output.append((text.strip(), rect))
        return output
    except Exception as exc:
        logger.debug("EasyOCR page error: %s", exc)
        return []
