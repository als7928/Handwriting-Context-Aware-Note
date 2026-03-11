"""Reranker Node – verifies proximity and relevance of retrieved chunks."""

from __future__ import annotations

from agents.state import AgentState, RetrievedChunk


def _relevance_score(chunk: RetrievedChunk, has_marker_filter: bool) -> float:
    """Compute a combined relevance score.

    Higher is better. Boosts chunks that:
    - have high vector similarity (score)
    - are close to a handwritten marker (low marker_distance)
    - match the requested marker_type when a filter is active
    """
    base = chunk["score"]

    # Boost when a marker is present and close
    if chunk["marker_type"] and chunk["marker_distance"] is not None:
        proximity_bonus = max(0, 1 - chunk["marker_distance"] / 200.0) * 0.3
        base += proximity_bonus

    # Penalise chunks without a marker when the user explicitly asked for one
    if has_marker_filter and not chunk["marker_type"]:
        base *= 0.5

    return base


async def reranker_node(state: AgentState) -> AgentState:
    """Re-rank retrieved chunks by combined spatial-semantic relevance."""
    has_marker = state.get("marker_filter") is not None
    chunks = list(state.get("retrieved_chunks", []))
    chunks.sort(key=lambda c: _relevance_score(c, has_marker), reverse=True)

    # Keep top-5 most relevant
    return {**state, "reranked_chunks": chunks[:5]}
