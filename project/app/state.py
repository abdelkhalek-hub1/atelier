"""
State model for the LangGraph agentic workflow.

Defines the strongly-typed AgentState used across all nodes
in the Wikipedia → LLM / DLQ workflow.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """
    Strongly-typed state object shared across all LangGraph nodes.

    This model is passed through the graph and mutated by each node.
    All fields are optional to allow partial updates at each step.
    """

    question: str = Field(..., description="The user's original question.")
    wiki_result: Optional[str] = Field(
        default=None,
        description="Raw text returned by the Wikipedia API on success.",
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The LLM-generated answer based on the Wikipedia context.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message if any node encountered a failure.",
    )
    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this workflow execution.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when the execution was created.",
    )

    model_config = ConfigDict(frozen=False)

    def with_updates(self, **kwargs: object) -> "AgentState":
        """
        Return a new AgentState with the given fields overridden.

        This preserves immutability semantics while still allowing
        LangGraph nodes to return state diffs cleanly.
        """
        return self.model_copy(update=kwargs)
