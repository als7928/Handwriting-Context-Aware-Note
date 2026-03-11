"""Chat endpoint – runs the LangGraph agent pipeline."""

from __future__ import annotations

import logging
import uuid as _uuid
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.graph import agent_graph
from models.database import Document
from models.schemas import ChatRequest, ChatResponse, HighlightLocation
from services.db import get_db

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    """Process a user query through the multi-agent workflow."""
    # When no specific documents are selected, search across ALL uploaded documents
    document_ids = [str(d) for d in req.document_ids]
    if not document_ids:
        result_q = await db.execute(select(Document.id))
        document_ids = [str(row[0]) for row in result_q.fetchall()]

    initial_state = {
        "raw_query": req.query,
        "document_ids": document_ids,
    }

    try:
        result = await agent_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Chat pipeline failed: %s", exc)
        return ChatResponse(
            answer=(
                "RAG pipeline is temporarily unavailable. "
                "Please check Qdrant/OpenAI configuration and try again."
            ),
            highlights=[],
        )

    raw_highlights = result.get("highlights", [])

    # Fetch filenames for cited documents
    highlight_doc_ids = list({h["document_id"] for h in raw_highlights})
    filename_map: dict[str, str] = {}
    if highlight_doc_ids:
        try:
            doc_uuids = [_uuid.UUID(did) for did in highlight_doc_ids]
            rows = await db.execute(
                select(Document.id, Document.filename).where(
                    Document.id.in_(doc_uuids)
                )
            )
            filename_map = {str(row[0]): row[1] for row in rows.fetchall()}
        except Exception:
            pass

    highlights = [
        HighlightLocation(
            document_id=h["document_id"],
            filename=filename_map.get(h["document_id"], ""),
            page_no=h["page_no"],
            x0=h["x0"],
            y0=h["y0"],
            x1=h["x1"],
            y1=h["y1"],
            text=h["text"],
            marker_type=h.get("marker_type"),
            score=h.get("score", 0.0),
        )
        for h in raw_highlights
    ]

    logger.info(
        "Chat response: query=%r highlights=%d scores=%s",
        req.query[:80],
        len(highlights),
        [round(h.score * 100) for h in highlights],
    )
    return ChatResponse(
        answer=result.get("answer", ""),
        highlights=highlights,
    )
