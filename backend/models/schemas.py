"""Pydantic schemas for request/response serialization."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    """Schema returned when listing / retrieving documents."""

    id: uuid.UUID
    filename: str
    page_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] = Field(default_factory=list)


class HighlightLocation(BaseModel):
    """Exact location to highlight on the PDF viewer."""

    document_id: uuid.UUID
    filename: str = ""
    page_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    marker_type: str | None = None
    score: float = 0.0  # cosine similarity in [0, 1]


class ChatResponse(BaseModel):
    """Response returned to the frontend chat interface."""

    answer: str
    highlights: list[HighlightLocation] = Field(default_factory=list)


# ── Spatial Chunk (internal) ──────────────────────────────────────────────────

class SpatialChunkPayload(BaseModel):
    """Payload stored alongside the vector in Qdrant."""

    chunk_id: str
    document_id: str
    page_no: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    marker_type: str | None = None
    marker_distance: float | None = None
