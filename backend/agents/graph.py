"""LangGraph state-machine that orchestrates the multi-agent workflow.

Flow:  query_rewrite → retriever → reranker → synthesis
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.query_rewrite import query_rewrite_node
from agents.reranker import reranker_node
from agents.retriever import retriever_node
from agents.state import AgentState
from agents.synthesis import synthesis_node


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.set_entry_point("query_rewrite")
    workflow.add_edge("query_rewrite", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


# Module-level compiled graph (singleton)
agent_graph = build_graph()
