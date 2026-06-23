"""
Router function for the LangGraph workflow.

Inspects the current AgentState and decides whether the graph
should continue along the **success** path (→ llm_node) or the
**failure** path (→ dlq_node).
"""

from __future__ import annotations

from typing import Literal

from app.logger import get_logger, log_node_event
from app.state import AgentState

logger = get_logger("langgraph.router")

# String literals that LangGraph uses as edge labels
SUCCESS_ROUTE: Literal["success"] = "success"
FAILURE_ROUTE: Literal["failure"] = "failure"


def router_function(state: AgentState) -> Literal["success", "failure"]:
    """
    Conditional edge function for the LangGraph workflow.

    Rules
    -----
    - If ``state.error`` is set **or** ``state.wiki_result`` is absent/empty,
      route to ``"failure"`` → DLQ node.
    - Otherwise route to ``"success"`` → LLM node.

    Returns
    -------
    Literal["success"] | Literal["failure"]
        The edge label LangGraph will follow.
    """
    has_error: bool = bool(state.error)
    has_content: bool = bool(state.wiki_result and state.wiki_result.strip())

    if has_error or not has_content:
        log_node_event(
            execution_id=state.execution_id,
            node="router",
            status="routed_to_failure",
            extra={
                "has_error": has_error,
                "has_content": has_content,
                "error": state.error,
            },
        )
        return FAILURE_ROUTE

    log_node_event(
        execution_id=state.execution_id,
        node="router",
        status="routed_to_success",
        extra={"has_content": has_content},
    )
    return SUCCESS_ROUTE
