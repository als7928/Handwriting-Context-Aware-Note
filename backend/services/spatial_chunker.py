"""Spatial-aware PDF chunking with PyMuPDF.

Extracts text blocks with bounding-box coordinates and maps nearby
handwritten markers (drawings / annotations) to their closest text chunks
using Euclidean proximity search.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field

import pymupdf as fitz

logger = logging.getLogger(__name__)


@dataclass
class RawTextBlock:
    """A single text block extracted from a PDF page."""

    page_no: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class MarkerAnnotation:
    """A detected handwritten marker or annotation found near printed text."""

    page_no: int
    marker_type: str
    cx: float  # centre-x
    cy: float  # centre-y
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


@dataclass
class SpatialChunk:
    """Text block enriched with the nearest marker metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    page_no: int = 0
    text: str = ""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    marker_type: str | None = None
    marker_distance: float | None = None


# ── Marker heuristics ────────────────────────────────────────────────────────

# Annotation types drawn OVER printed text – exact covered text is read
# directly via page.get_textbox() rather than proximity mapping.
_TEXT_OVERLAY_TYPES: frozenset[int] = frozenset({
    fitz.PDF_ANNOT_HIGHLIGHT,
    fitz.PDF_ANNOT_UNDERLINE,
    fitz.PDF_ANNOT_SQUIGGLY,
    fitz.PDF_ANNOT_STRIKE_OUT,
})


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_text_blocks(page: fitz.Page, page_no: int) -> list[RawTextBlock]:
    """Extract text blocks with bounding boxes from a single page."""
    blocks: list[RawTextBlock] = []
    for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if b["type"] != 0:  # skip image blocks
            continue
        text_parts: list[str] = []
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text_parts.append(span["text"])
        text = " ".join(text_parts).strip()
        if not text:
            continue
        blocks.append(
            RawTextBlock(
                page_no=page_no,
                text=text,
                x0=b["bbox"][0],
                y0=b["bbox"][1],
                x1=b["bbox"][2],
                y1=b["bbox"][3],
            )
        )
    return blocks


def _cluster_drawings(drawings: list[dict], gap: float = 15.0) -> list[dict]:
    """Group nearby vector drawing paths into composite clusters.

    Many PDF annotation apps (e.g. Notability) save each pen stroke as a
    separate path element.  Clustering nearby paths lets us correctly classify
    multi-stroke shapes (circles, boxes) as a whole.
    """
    clusters: list[dict] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        r = fitz.Rect(rect)
        # Skip degenerate / sub-pixel paths
        if r.is_empty or r.width < 2 or r.height < 2:
            continue
        merged = False
        for cl in clusters:
            cr = cl["rect"]
            # Merge when bounding boxes are within *gap* pixels of each other
            if (
                r.x0 - gap <= cr.x1
                and r.x1 + gap >= cr.x0
                and r.y0 - gap <= cr.y1
                and r.y1 + gap >= cr.y0
            ):
                cl["rect"] = fitz.Rect(
                    min(cr.x0, r.x0),
                    min(cr.y0, r.y0),
                    max(cr.x1, r.x1),
                    max(cr.y1, r.y1),
                )
                cl["items"].extend(d.get("items", []))
                merged = True
                break
        if not merged:
            clusters.append(
                {
                    "rect": fitz.Rect(r),
                    "items": list(d.get("items", [])),
                    "color": d.get("color"),
                    "fill": d.get("fill"),
                    "width": d.get("width", 1.0),
                }
            )
    return clusters


def extract_markers(page: fitz.Page, page_no: int) -> list[MarkerAnnotation]:
    """Detect handwritten markers from annotations and vector drawings."""
    markers: list[MarkerAnnotation] = []

    # 1) PDF annotations – skip text-overlay types (handled by
    #    extract_annotated_text_chunks) and free-text (handled by
    #    extract_freetext_and_ink_chunks) to avoid duplicate chunks.
    _skip_in_proximity = _TEXT_OVERLAY_TYPES | frozenset({fitz.PDF_ANNOT_FREE_TEXT})
    for annot in page.annots() or []:
        type_id = annot.type[0] if annot.type else -1
        if type_id in _skip_in_proximity:
            continue
        rect = annot.rect
        markers.append(
            MarkerAnnotation(
                page_no=page_no,
                marker_type="annotated",
                cx=(rect.x0 + rect.x1) / 2,
                cy=(rect.y0 + rect.y1) / 2,
                x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1,
            )
        )

    # 2) Vector drawings – cluster nearby paths before classifying so that
    #    multi-stroke shapes (circles, boxes) are recognised as one unit.
    for cluster in _cluster_drawings(page.get_drawings()):
        r = cluster["rect"]
        markers.append(
            MarkerAnnotation(
                page_no=page_no,
                marker_type="annotated",
                cx=(r.x0 + r.x1) / 2,
                cy=(r.y0 + r.y1) / 2,
                x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1,
            )
        )

    return markers


# ── Direct annotation text extraction ───────────────────────────────────────

def extract_annotated_text_chunks(page: fitz.Page, page_no: int) -> list[SpatialChunk]:
    """Extract printed text directly beneath highlight / underline / squiggly /
    strikethrough annotations.

    These annotation types are drawn ON TOP of existing text, so the exact
    covered content is retrieved via ``page.get_textbox()``.  The resulting
    chunks get ``marker_distance=0`` (perfect spatial match) and are stored
    alongside proximity-mapped chunks in the Qdrant collection.
    """
    chunks: list[SpatialChunk] = []
    for annot in page.annots() or []:
        type_id = annot.type[0] if annot.type else -1
        if type_id not in _TEXT_OVERLAY_TYPES:
            continue
        rect = annot.rect
        text = page.get_textbox(rect).strip()
        # Fallback: rebuild rect from quad vertices (more precise for multi-line highlights)
        if not text and annot.vertices:
            verts = annot.vertices
            xs = [p[0] if hasattr(p, "__getitem__") else p.x for p in verts]
            ys = [p[1] if hasattr(p, "__getitem__") else p.y for p in verts]
            quad_rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            text = page.get_textbox(quad_rect).strip()
            if text:
                rect = quad_rect
        if not text:
            continue
        chunks.append(
            SpatialChunk(
                page_no=page_no,
                text=text,
                x0=rect.x0,
                y0=rect.y0,
                x1=rect.x1,
                y1=rect.y1,
                marker_type="annotated",
                marker_distance=0.0,
            )
        )
    return chunks


def extract_freetext_and_ink_chunks(page: fitz.Page, page_no: int) -> list[SpatialChunk]:
    """Extract free-text (typed) annotation content and contextual text for ink strokes.

    - ``PDF_ANNOT_FREE_TEXT``: typed note/label added by the reader app;
      content is read directly from the annotation metadata.
    - ``PDF_ANNOT_INK``: freehand pen strokes (digital handwriting). The stroke
      paths themselves are not machine-readable without OCR, but any printed
      text within the ink bounding box is captured as context.
    """
    chunks: list[SpatialChunk] = []
    for annot in page.annots() or []:
        type_id = annot.type[0] if annot.type else -1
        rect = annot.rect

        if type_id == fitz.PDF_ANNOT_FREE_TEXT:
            # Typed note added directly onto the PDF page
            content = (annot.info.get("content") or "").strip()
            if content:
                chunks.append(
                    SpatialChunk(
                        page_no=page_no,
                        text=f"[Annotation note] {content}",
                        x0=rect.x0,
                        y0=rect.y0,
                        x1=rect.x1,
                        y1=rect.y1,
                        marker_type="annotated",
                        marker_distance=0.0,
                    )
                )

        elif type_id == fitz.PDF_ANNOT_INK:
            # Ink strokes: try to read printed text the user wrote over/around.
            # Works when the stylus was used over or very near existing text.
            underlying = page.get_textbox(rect).strip()
            if underlying:
                chunks.append(
                    SpatialChunk(
                        page_no=page_no,
                        text=underlying,
                        x0=rect.x0,
                        y0=rect.y0,
                        x1=rect.x1,
                        y1=rect.y1,
                        marker_type="annotated",
                        marker_distance=0.0,
                    )
                )

    return chunks


def _try_ocr_chunks(page: fitz.Page, page_no: int) -> list[SpatialChunk]:
    """EasyOCR-based extraction for handwritten content.

    Mode 1 – No selectable text on page (scanned / fully handwritten PDF):
      OCR the entire page and return one chunk per recognised text region.

    Mode 2 – Ink annotations whose bounding area contains no printed text:
      OCR just those annotation regions to surface handwritten notes.
    """
    from services.ocr import ocr_page, ocr_region  # lazy import to avoid circular deps

    chunks: list[SpatialChunk] = []
    page_has_text = bool(page.get_text().strip())

    if not page_has_text:
        # Mode 1: full-page handwriting / scanned page
        for text, rect in ocr_page(page):
            chunks.append(
                SpatialChunk(
                    page_no=page_no,
                    text=text,
                    x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1,
                    marker_type="annotated",
                    marker_distance=0.0,
                )
            )
    else:
        # Mode 2: OCR ink annotation regions that have no underlying printed text.
        # The OCR'd text is the handwritten content itself – store it as annotated.
        for annot in page.annots() or []:
            type_id = annot.type[0] if annot.type else -1
            if type_id != fitz.PDF_ANNOT_INK:
                continue
            rect = annot.rect
            if page.get_textbox(rect).strip():
                continue  # already captured by extract_freetext_and_ink_chunks
            text = ocr_region(page, rect)
            if text:
                chunks.append(
                    SpatialChunk(
                        page_no=page_no,
                        text=f"[Handwritten] {text}",
                        x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1,
                        marker_type="annotated",
                        marker_distance=0.0,
                    )
                )

    return chunks


# ── Proximity mapping ────────────────────────────────────────────────────────

def _block_centre(block: RawTextBlock) -> tuple[float, float]:
    return ((block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2)


def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _rects_overlap(
    bx0: float, by0: float, bx1: float, by1: float,
    mx0: float, my0: float, mx1: float, my1: float,
    h_padding: float = 15.0,
    v_padding: float = 35.0,
) -> bool:
    """Return True when the marker rect overlaps or is near the text block.

    Asymmetric padding: larger *v_padding* catches markers drawn below text
    (hand-drawn underlines) and markers that surround text (circles, boxes).
    """
    return (
        mx0 - h_padding <= bx1
        and mx1 + h_padding >= bx0
        and my0 - v_padding <= by1
        and my1 + v_padding >= by0
    )


def map_markers_to_blocks(
    blocks: list[RawTextBlock],
    markers: list[MarkerAnnotation],
    max_distance: float = 250.0,
) -> dict[int, list[tuple[str, float]]]:
    """For each marker, find the nearest text block within *max_distance*.

    Rect overlap is treated as distance 0.  Falls back to Euclidean distance.
    Returns ``block_index -> [(marker_type, distance), ...]``.
    A single block can have MULTIPLE distinct marker types (e.g. underline AND
    circle), producing one SpatialChunk per marker type.
    """
    block_markers: dict[int, list[tuple[str, float]]] = {}

    for marker in markers:
        best_idx, best_dist = -1, float("inf")
        has_rect = marker.x1 > marker.x0 and marker.y1 > marker.y0
        for idx, block in enumerate(blocks):
            if block.page_no != marker.page_no:
                continue
            if has_rect and _rects_overlap(
                block.x0, block.y0, block.x1, block.y1,
                marker.x0, marker.y0, marker.x1, marker.y1,
            ):
                dist = 0.0
            else:
                cx, cy = _block_centre(block)
                dist = _euclidean(cx, cy, marker.cx, marker.cy)
            if dist < best_dist:
                best_idx, best_dist = idx, dist

        if best_idx < 0 or best_dist > max_distance:
            continue

        entry = block_markers.setdefault(best_idx, [])
        # Per block, keep one entry per marker_type (closest wins)
        type_idx = {m[0]: i for i, m in enumerate(entry)}
        if marker.marker_type not in type_idx:
            entry.append((marker.marker_type, best_dist))
        elif best_dist < entry[type_idx[marker.marker_type]][1]:
            entry[type_idx[marker.marker_type]] = (marker.marker_type, best_dist)

    return block_markers


# ── Public API ───────────────────────────────────────────────────────────────

def process_pdf(file_path: str) -> tuple[list[SpatialChunk], int]:
    """Parse a PDF and return spatial chunks enriched with marker proximity.

    Returns:
        (chunks, page_count)
    """
    doc = fitz.open(file_path)
    all_blocks: list[RawTextBlock] = []
    all_markers: list[MarkerAnnotation] = []
    direct_chunks: list[SpatialChunk] = []

    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        page_1 = page_no + 1  # 1-indexed throughout

        page_blocks = extract_text_blocks(page, page_1)
        page_markers = extract_markers(page, page_1)
        page_direct = (
            extract_annotated_text_chunks(page, page_1)
            + extract_freetext_and_ink_chunks(page, page_1)
            + _try_ocr_chunks(page, page_1)
        )

        all_blocks.extend(page_blocks)
        all_markers.extend(page_markers)
        direct_chunks.extend(page_direct)

        annot_count = sum(1 for _ in (page.annots() or []))
        drawing_count = len(page.get_drawings())
        logger.info(
            "PDF page %d: %d text blocks | %d PDF annots | %d drawings | "
            "%d markers extracted | %d direct chunks",
            page_1, len(page_blocks), annot_count, drawing_count,
            len(page_markers), len(page_direct),
        )

    marker_map = map_markers_to_blocks(all_blocks, all_markers)

    chunks: list[SpatialChunk] = []
    for idx, block in enumerate(all_blocks):
        marker_list = marker_map.get(idx, [])
        if not marker_list:
            # Plain, unannotated text block
            chunks.append(
                SpatialChunk(
                    page_no=block.page_no,
                    text=block.text,
                    x0=block.x0, y0=block.y0,
                    x1=block.x1, y1=block.y1,
                    marker_type=None,
                    marker_distance=None,
                )
            )
        else:
            # Emit one chunk per distinct marker type (cap at 3 per block)
            for m_type, m_dist in marker_list[:3]:
                chunks.append(
                    SpatialChunk(
                        page_no=block.page_no,
                        text=block.text,
                        x0=block.x0, y0=block.y0,
                        x1=block.x1, y1=block.y1,
                        marker_type=m_type,
                        marker_distance=m_dist,
                    )
                )

    # Append directly extracted annotation chunks (highlights, typed notes, OCR)
    chunks.extend(direct_chunks)

    logger.info(
        "PDF processing done: %d total chunks (%d proximity-mapped, %d direct)",
        len(chunks), len(chunks) - len(direct_chunks), len(direct_chunks),
    )

    page_count = len(doc)
    doc.close()
    return chunks, page_count
