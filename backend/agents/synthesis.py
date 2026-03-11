"""Synthesis Node – generates a natural-language answer from reranked chunks
and returns highlight coordinates for the frontend PDF viewer.
"""

from __future__ import annotations

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from config import settings


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful study assistant. The user is asking about content in their \
annotated PDF documents. You will be given relevant text excerpts with their \
spatial coordinates and marker information.

Marker types you may encounter:
- highlight: text marked with a fluorescent highlighter pen
- underline: text with a line drawn beneath it
- squiggly: text with a wavy underline
- strikethrough: text that has been struck through
- circle: text surrounded by a hand-drawn circle
- box: text enclosed in a hand-drawn rectangle or box
- checkmark: item marked with a check or tick
- star: item marked with a star symbol
- arrow: item pointed to by an arrow
- ink_drawing: freehand pen/stylus mark near or over text
- free_text: a typed annotation note added by the user
- handwriting: OCR'd text from a handwritten page

Rules:
- Provide a clear, concise answer based ONLY on the provided excerpts.
- Reference specific pages and marker types when relevant (e.g. "the highlighted text on page 3").
- Treat 'free_text' excerpts as the user's own typed notes.
- Treat 'handwriting' excerpts as OCR'd handwritten content.
- If no relevant content is found, say so honestly.
- Format your answer in Markdown for readability.
"""


async def synthesis_node(state: AgentState) -> AgentState:
    """Generate a summary answer and collect highlight locations."""
    chunks = state.get("reranked_chunks", [])

    if not chunks:
        return {
            **state,
            "answer": "I couldn't find any relevant content matching your query in the uploaded documents.",
            "highlights": [],
        }

    # Build context from chunks
    context_parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        marker_info = f" [Marker: {c['marker_type']}]" if c["marker_type"] else ""
        context_parts.append(
            f"Excerpt {i} (Page {c['page_no']}{marker_info}):\n{c['text']}"
        )
    context = "\n\n".join(context_parts)

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User question: {state['raw_query']}\n\nRelevant excerpts:\n{context}"
        ),
    ]
    try:
        response = await llm.ainvoke(messages)
        answer_text = response.content
    except Exception as exc:
        logger.warning("Synthesis LLM call failed. Falling back to extractive answer. error=%s", exc)
        bullets = [f"- p.{c['page_no']}: {c['text'][:220]}" for c in chunks[:3]]
        answer_text = "I found relevant excerpts, but generation failed. Here are the top matches:\n\n" + "\n".join(bullets)

    highlights = [
        {
            "document_id": c["document_id"],
            "page_no": c["page_no"],
            "x0": c["x0"],
            "y0": c["y0"],
            "x1": c["x1"],
            "y1": c["y1"],
            "text": c["text"],
            "marker_type": c.get("marker_type"),
            "score": c.get("score", 0.0),
        }
        for c in chunks
    ]

    return {
        **state,
        "answer": answer_text,
        "highlights": highlights,
    }
