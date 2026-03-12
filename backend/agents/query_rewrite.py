"""Query Rewrite Node – uses DSPy to decompose the user query into
a semantic intent string and an optional visual-marker filter.
"""

from __future__ import annotations

import logging
import dspy

from agents.state import AgentState
from config import settings


logger = logging.getLogger(__name__)


# ── DSPy signature ────────────────────────────────────────────────────────────

class QueryRewriteSignature(dspy.Signature):
    """Decompose a user question about handwritten-annotated PDFs into a
    cleaned semantic query and an optional marker-type filter."""

    raw_query: str = dspy.InputField(desc="The user's original question.")
    semantic_query: str = dspy.OutputField(
        desc="Rewritten query focusing on the semantic meaning, without references to visual markers."
    )
    marker_filter: str = dspy.OutputField(
        desc=(
            "Output 'annotated' when the user is asking about content that was marked, highlighted, "
            "underlined, circled, boxed, starred, written on, or otherwise annotated with handwriting "
            "or drawings – regardless of the specific shape or style of the marking. "
            "Output 'none' when the user is asking about general content without any reference to "
            "annotations or markings."
        )
    )


class QueryRewriter(dspy.Module):
    """DSPy module that rewrites spatial queries."""

    def __init__(self) -> None:
        super().__init__()
        self.rewrite = dspy.ChainOfThought(QueryRewriteSignature)

    def forward(self, raw_query: str) -> dspy.Prediction:
        return self.rewrite(raw_query=raw_query)


# Singleton – initialised lazily so import never fails without an API key.
_rewriter: QueryRewriter | None = None


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        lm = dspy.LM(f"openai/{settings.llm_model}", api_key=settings.openai_api_key)
        dspy.configure(lm=lm)
        _rewriter = QueryRewriter()
    return _rewriter


# ── LangGraph node function ──────────────────────────────────────────────────

async def query_rewrite_node(state: AgentState) -> AgentState:
    """Rewrite the raw user query into structured search params."""
    try:
        rewriter = _get_rewriter()
        result = rewriter(raw_query=state["raw_query"])

        marker = result.marker_filter.strip().lower()
        if marker == "none" or not marker:
            marker = None

        return {
            **state,
            "semantic_query": result.semantic_query.strip(),
            "marker_filter": marker,
        }
    except Exception as exc:
        logger.warning("Query rewrite failed. Falling back to raw query. error=%s", exc)
        raw_query = state["raw_query"]
        lowered = raw_query.lower()
        marker = None
        # Keyword → marker type mapping including Korean terms
        _kw_map = {
            "star": ["star", "별", "별표"],
            "underline": ["underline", "밑줄"],
            "squiggly": ["squiggly", "물결", "구불구불", "파도선"],
            "strikethrough": ["strikethrough", "strikeout", "취소선", "지운", "긋기"],
            "circle": ["circle", "동그라미", "원", "동그"],
            "highlight": ["highlight", "형광", "하이라이트", "형광펜", "칠한", "마킹"],
            "arrow": ["arrow", "화살표", "화살"],
            "bracket": ["bracket", "괄호"],
            "checkmark": ["check", "checkmark", "체크", "체크표시", "체크마크"],
            "box": ["box", "네모", "사각형", "박스", "직사각형", "rectangle"],
            "ink_drawing": ["ink", "펜", "필기", "드로잉"],
            "free_text": ["note", "메모", "주석", "노트"],
            "handwriting": ["handwriting", "손글씨", "필기체", "수기"],
        }
        for marker_name, keywords in _kw_map.items():
            if any(kw in lowered for kw in keywords):
                marker = marker_name
                break

        return {
            **state,
            "semantic_query": raw_query,
            "marker_filter": marker,
        }
