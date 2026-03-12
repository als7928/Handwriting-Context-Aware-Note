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


# ── Layout-element filter ─────────────────────────────────────────────────────

# Marks wider / taller than these fractions of the page are almost certainly
# printed layout rules or borders, not handwritten annotations.
_MAX_MARK_WIDTH_RATIO: float = 0.45
_MAX_MARK_HEIGHT_RATIO: float = 0.45
# Minimum dimension (PDF points) – anything smaller is noise / sub-pixel artifact
_MIN_MARK_PTS: float = 4.0


def _is_layout_element(rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    """Return True when a vector drawing is almost certainly a printed layout
    element (horizontal rule, table border, decorative line) rather than a
    handwritten mark.

    Heuristics used (all in PDF-point units):
    - Bounding box is sub-pixel / too small to be a real mark.
    - Drawing spans ≥ 45 % of the page width or height  →  layout rule / border.
    """
    w, h = rect.width, rect.height
    pw, ph = page_rect.width, page_rect.height

    if w < _MIN_MARK_PTS and h < _MIN_MARK_PTS:
        return True
    if w > pw * _MAX_MARK_WIDTH_RATIO:
        return True
    if h > ph * _MAX_MARK_HEIGHT_RATIO:
        return True

    return False


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_text_blocks(page: fitz.Page, page_no: int) -> list[RawTextBlock]:
    """Extract text *blocks* (paragraphs) with bounding boxes from a single page.

    Block-level granularity is intentional: a single annotation may span
    multiple lines, and we want to surface all nearby context blocks as
    candidates rather than over-precisely matching a single line.
    """
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
        # Skip printed layout elements (rules, borders, table lines, etc.)
        if _is_layout_element(r, page.rect):
            logger.debug(
                "Page %d: skipping layout drawing %.0fx%.0f at (%.0f,%.0f)",
                page_no, r.width, r.height, r.x0, r.y0,
            )
            continue
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


def extract_freetext_and_ink_chunks(
    page: fitz.Page,
    page_no: int,
    text_lines: list[RawTextBlock] | None = None,
) -> list[SpatialChunk]:
    """Extract free-text (typed) annotation content and contextual text for ink strokes.

    - ``PDF_ANNOT_FREE_TEXT``: typed note/label added by the reader app;
      content is read directly from the annotation metadata.
    - ``PDF_ANNOT_INK``: freehand pen strokes (digital handwriting).  The exact
      search strategy is:

      1. Direct: ``page.get_textbox(ink_rect)`` – works when ink is drawn
         directly over text.
      2. Expanded upward: many apps draw underlines *below* the text baseline,
         so we expand the search rect upward by ``_INK_UP_EXPAND`` points.
      3. Nearest-line fallback: if expansion still finds nothing, pick the
         pre-extracted line whose rect overlaps an even wider search window.

    Pass ``text_lines`` (from ``extract_text_blocks``) to enable strategies
    2 and 3.
    """
    # How far above the ink bbox to look for the annotated text line.
    _INK_UP_EXPAND = 36  # ≈ 12.7 mm – covers one or two typical text lines
    _INK_H_EXPAND = 6   # small horizontal tolerance
    _INK_DOWN_EXPAND = 4

    chunks: list[SpatialChunk] = []
    for annot in page.annots() or []:
        type_id = annot.type[0] if annot.type else -1
        rect = annot.rect

        if type_id == fitz.PDF_ANNOT_FREE_TEXT:
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
            # Strategy 1: direct textbox lookup
            underlying = page.get_textbox(rect).strip()

            if not underlying:
                # Strategy 2: expand upward – underlines are drawn below text
                expanded = fitz.Rect(
                    rect.x0 - _INK_H_EXPAND,
                    rect.y0 - _INK_UP_EXPAND,
                    rect.x1 + _INK_H_EXPAND,
                    rect.y1 + _INK_DOWN_EXPAND,
                ).intersect(page.rect)
                underlying = page.get_textbox(expanded).strip()

            if not underlying and text_lines:
                # Strategy 3: find pre-extracted lines whose bbox overlaps
                # the expanded rect and join their text.
                expanded = fitz.Rect(
                    rect.x0 - _INK_H_EXPAND,
                    rect.y0 - _INK_UP_EXPAND,
                    rect.x1 + _INK_H_EXPAND,
                    rect.y1 + _INK_DOWN_EXPAND,
                ).intersect(page.rect)
                matched = [
                    ln for ln in text_lines
                    if ln.page_no == page_no
                    and not fitz.Rect(ln.x0, ln.y0, ln.x1, ln.y1).intersect(expanded).is_empty
                ]
                if matched:
                    underlying = " ".join(ln.text for ln in matched)
                    logger.debug(
                        "Page %d INK: matched %d lines via expanded search",
                        page_no, len(matched),
                    )

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
            # Check both the direct rect and the expanded rect used by
            # extract_freetext_and_ink_chunks – skip if text was already found.
            expanded = fitz.Rect(
                rect.x0 - 6, rect.y0 - 36,
                rect.x1 + 6, rect.y1 + 4,
            ).intersect(page.rect)
            if page.get_textbox(expanded).strip():
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


def _distance_to_rect(
    px: float, py: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> float:
    """Minimum distance from point (px, py) to a rectangle.

    Returns 0 when the point is inside the rectangle.  This is more accurate
    than centre-to-centre distance for margin marks (stars, arrows) that are
    placed to the left/right of a line rather than directly over it.
    """
    dx = max(bx0 - px, 0.0, px - bx1)
    dy = max(by0 - py, 0.0, py - by1)
    return math.sqrt(dx * dx + dy * dy)


def _rects_overlap(
    bx0: float, by0: float, bx1: float, by1: float,
    mx0: float, my0: float, mx1: float, my1: float,
    h_padding: float = 8.0,
    v_padding: float = 12.0,
) -> bool:
    """Return True when the marker rect overlaps or is near the text block.

    Padding values are intentionally conservative to avoid tagging unrelated
    text blocks as annotated:
    - h_padding  8 pts ≈ 2.8 mm – small horizontal tolerance
    - v_padding 12 pts ≈ 4.2 mm – catches underlines drawn just below a line
      of text and circles that slightly miss the text bounding box
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
    max_distance: float = 150.0,
    top_k: int = 4,
) -> dict[int, list[tuple[str, float]]]:
    """For each marker, collect the *top_k* nearest text blocks within
    *max_distance* and tag all of them as annotated.

    Recall-first strategy: rather than trying to pinpoint the exact block
    a user marked, we surface all plausible candidates in the vicinity so
    that retrieval can rank them.  A single annotation can therefore tag
    multiple nearby blocks.

    max_distance=150 PDF points ≈ 53 mm – wide enough to catch margin marks
    and annotations that span paragraph boundaries.
    top_k=4 – at most 4 blocks per marker to avoid tagging the whole page.

    Returns ``block_index -> [(marker_type, distance), ...]``.
    """
    block_markers: dict[int, list[tuple[str, float]]] = {}

    for marker in markers:
        # Collect all blocks on the same page, sorted by edge distance
        candidates: list[tuple[int, float]] = []
        for idx, block in enumerate(blocks):
            if block.page_no != marker.page_no:
                continue
            dist = _distance_to_rect(
                marker.cx, marker.cy,
                block.x0, block.y0, block.x1, block.y1,
            )
            if dist <= max_distance:
                candidates.append((idx, dist))

        # Keep only the top_k closest
        candidates.sort(key=lambda t: t[1])
        for idx, dist in candidates[:top_k]:
            entry = block_markers.setdefault(idx, [])
            type_idx = {m[0]: i for i, m in enumerate(entry)}
            if marker.marker_type not in type_idx:
                entry.append((marker.marker_type, dist))
            elif dist < entry[type_idx[marker.marker_type]][1]:
                entry[type_idx[marker.marker_type]] = (marker.marker_type, dist)

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
            + extract_freetext_and_ink_chunks(page, page_1, text_lines=page_blocks)
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
