"""Shared state schema used across all LangGraph nodes."""

from __future__ import annotations

from typing import TypedDict


class RetrievedChunk(TypedDict):
    """A single retrieved spatial chunk with its relevance score."""

    chunk_id: str
    document_id: str
    page_no: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    marker_type: str | None
    marker_distance: float | None
    score: float


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph workflow."""

    # Input
    raw_query: str
    document_ids: list[str]

    # After Query Rewrite
    semantic_query: str
    marker_filter: str | None

    # After Retrieval
    retrieved_chunks: list[RetrievedChunk]

    # After Reranking
    reranked_chunks: list[RetrievedChunk]

    # After Synthesis
    answer: str
    highlights: list[dict]
