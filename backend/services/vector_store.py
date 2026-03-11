"""Qdrant vector store operations for spatial chunks."""

from __future__ import annotations

import uuid
import logging

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings
from models.schemas import SpatialChunkPayload
from services.embedding import embed_single, embed_texts


logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None


def _get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


async def ensure_collection() -> None:
    """Create the Qdrant collection if it does not yet exist."""
    client = _get_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if settings.qdrant_collection not in names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )


async def upsert_chunks(
    chunks: list[SpatialChunkPayload],
) -> list[str]:
    """Embed and upsert spatial chunks into Qdrant. Returns point IDs."""
    client = _get_client()
    texts = [c.text for c in chunks]
    vectors = await embed_texts(texts)

    points: list[PointStruct] = []
    point_ids: list[str] = []
    for chunk, vec in zip(chunks, vectors):
        pid = str(uuid.uuid4())
        point_ids.append(pid)
        points.append(
            PointStruct(
                id=pid,
                vector=vec,
                payload=chunk.model_dump(),
            )
        )

    await client.upsert(collection_name=settings.qdrant_collection, points=points)
    return point_ids


def _apply_mmr(
    query_vec: list[float],
    candidates: list[dict],
    k: int,
    lambda_: float = 0.7,
) -> list[dict]:
    """Apply Maximal Marginal Relevance to select k diverse documents.

    Args:
        query_vec:  Embedding of the user query.
        candidates: List of result dicts, each containing a ``"_vec"`` key
                    with the document embedding.
        k:          Number of results to select.
        lambda_:    Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        Up to *k* results ordered by MMR selection order (most relevant/diverse
        first).  The ``"_vec"`` helper key is stripped from the returned dicts.
    """
    if len(candidates) <= k:
        for c in candidates:
            c.pop("_vec", None)
        return candidates

    q = np.array(query_vec, dtype=np.float32)
    vecs = np.array([c["_vec"] for c in candidates], dtype=np.float32)

    # L2-normalise for cosine similarity
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_norm = vecs / norms

    sim_to_query = vecs_norm @ q_norm  # shape (n,)

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: float(sim_to_query[i]))
        else:
            sel_mat = vecs_norm[selected]  # (m, d)
            best_score = float("-inf")
            best = remaining[0]
            for idx in remaining:
                redundancy = float((vecs_norm[idx] @ sel_mat.T).max())
                mmr = lambda_ * float(sim_to_query[idx]) - (1 - lambda_) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best = idx
        selected.append(best)
        remaining.remove(best)

    results = []
    for i in selected:
        c = dict(candidates[i])
        c.pop("_vec", None)
        results.append(c)
    logger.debug("MMR selected %d/%d candidates (lambda=%.2f)", len(results), len(candidates), lambda_)
    return results


async def hybrid_search(
    query: str,
    document_ids: list[str] | None = None,
    marker_type: str | None = None,
    top_k: int = 10,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
) -> list[dict]:
    """Semantic search with optional metadata filters and MMR re-ranking.

    When *marker_type* is specified, performs a primary search filtered to that
    marker type.  If the filtered search returns fewer than
    ``ceil(top_k / 3)`` results, a secondary unfiltered search runs and the
    results are merged.  This makes annotation-aware queries robust even when
    some chunks were mis-classified at ingestion time.

    When *use_mmr* is True (default), Maximal Marginal Relevance is applied on
    the over-fetched candidate pool to maximise both relevance and diversity.

    Returns a list of payload dicts ordered by relevance.
    """
    client = _get_client()

    try:
        query_vec = await embed_single(query)
    except Exception as exc:
        logger.warning("Embedding generation failed in hybrid_search. error=%s", exc)
        return []

    # Fetch more candidates for MMR re-selection
    fetch_k = top_k * 2 if use_mmr else top_k

    async def _search(extra_marker: str | None) -> list[dict]:
        must: list = []
        if document_ids:
            must.append(Filter(should=[
                FieldCondition(key="document_id", match=MatchValue(value=did))
                for did in document_ids
            ]))
        if extra_marker:
            must.append(FieldCondition(key="marker_type", match=MatchValue(value=extra_marker)))
        f = Filter(must=must) if must else None
        try:
            resp = await client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_vec,
                query_filter=f,
                limit=fetch_k,
                with_payload=True,
                with_vectors=use_mmr,
            )
            return [
                {
                    **hit.payload,
                    "score": hit.score,
                    **({"_vec": list(hit.vector)} if use_mmr and hit.vector is not None else {}),
                }
                for hit in resp.points
            ]
        except Exception as exc:
            logger.warning("Qdrant query failed. error=%s", exc)
            return []

    results = await _search(marker_type)
    logger.debug("Primary search returned %d results (marker_type=%s)", len(results), marker_type)

    # Fallback: if marker-filtered search is sparse, blend in unfiltered results
    fallback_threshold = max(3, top_k // 3)
    if marker_type and len(results) < fallback_threshold:
        fallback = await _search(None)
        seen_ids = {r.get("chunk_id") for r in results}
        for r in fallback:
            if r.get("chunk_id") not in seen_ids:
                results.append(r)
                seen_ids.add(r.get("chunk_id"))
        logger.debug("After fallback merge: %d results", len(results))

    if use_mmr and results:
        results = _apply_mmr(query_vec, results, k=top_k, lambda_=mmr_lambda)
    else:
        results = results[:top_k]

    return results


async def delete_by_document(document_id: str) -> None:
    """Remove all points belonging to a document."""
    client = _get_client()
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )
