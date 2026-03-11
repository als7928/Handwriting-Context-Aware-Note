"""Retriever Node – performs hybrid search in Qdrant (vector + metadata)."""

from __future__ import annotations

from agents.state import AgentState, RetrievedChunk
from services.vector_store import hybrid_search


async def retriever_node(state: AgentState) -> AgentState:
    """Execute a hybrid search combining semantic similarity and marker filter."""
    results = await hybrid_search(
        query=state["semantic_query"],
        document_ids=state.get("document_ids") or None,
        marker_type=state.get("marker_filter"),
        top_k=15,
    )

    chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id", ""),
            page_no=r.get("page_no", 0),
            text=r.get("text", ""),
            x0=r.get("x0", 0),
            y0=r.get("y0", 0),
            x1=r.get("x1", 0),
            y1=r.get("y1", 0),
            marker_type=r.get("marker_type"),
            marker_distance=r.get("marker_distance"),
            score=r.get("score", 0),
        )
        for r in results
    ]

    return {**state, "retrieved_chunks": chunks}
