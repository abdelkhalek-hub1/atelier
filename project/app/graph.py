"""
LangGraph workflow assembly.

Wires together all nodes and the conditional router into a compiled
StateGraph that can be invoked with a single question string.

Graph topology
--------------

    START
      |
      v
    wikipedia_node
      |
      +-- success --> llm_node --> END
      |
      +-- failure --> dlq_node --> END
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]

from app.config import config
from app.logger import get_logger
from app.nodes import dlq_node, llm_node, wikipedia_node
from app.router import FAILURE_ROUTE, SUCCESS_ROUTE, router_function
from app.state import AgentState

logger = get_logger("langgraph.graph")


# ---------------------------------------------------------------------------
# LangSmith tracing (optional – activated by env var)
# ---------------------------------------------------------------------------

def _configure_langsmith() -> None:
    """
    Enable LangSmith tracing when LANGSMITH_TRACING=true is set.

    This must be called before any LangChain / LangGraph objects are
    created so the tracer callback is registered globally.
    """
    if config.LANGSMITH_TRACING_ENABLED and config.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
        logger.info(
            "LangSmith tracing enabled",
            extra={"project": config.LANGSMITH_PROJECT},
        )
    else:
        # Ensure tracing is off when credentials are absent
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


_configure_langsmith()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph object ready to be invoked.
    """
    # Use AgentState directly as the state schema
    graph = StateGraph(AgentState)

    # Register nodes directly
    graph.add_node("wikipedia_node", wikipedia_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("dlq_node", dlq_node)

    # Entry point
    graph.add_edge(START, "wikipedia_node")

    # Conditional routing after wikipedia_node
    graph.add_conditional_edges(
        "wikipedia_node",
        router_function,
        {
            SUCCESS_ROUTE: "llm_node",
            FAILURE_ROUTE: "dlq_node",
        },
    )

    # Terminal edges
    graph.add_edge("llm_node", END)
    graph.add_edge("dlq_node", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Singleton compiled graph – built once at module import time
_compiled_graph = build_graph()


def run_workflow(question: str) -> AgentState:
    """
    Execute the full LangGraph workflow for a given question.

    Args:
        question: The user's natural-language question.

    Returns:
        The final AgentState after the graph has completed.

    Raises:
        ValueError: If question is empty or whitespace-only.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    initial_state = AgentState(question=question.strip())

    logger.info(
        "Workflow started",
        extra={
            "execution_id": initial_state.execution_id,
            "question": initial_state.question,
        },
    )

    result = _compiled_graph.invoke(initial_state)
    if isinstance(result, AgentState):
        final_state = result
    else:
        final_state = AgentState(**result)

    logger.info(
        "Workflow completed",
        extra={
            "execution_id": final_state.execution_id,
            "has_answer": bool(final_state.final_answer),
            "has_error": bool(final_state.error),
        },
    )

    return final_state
