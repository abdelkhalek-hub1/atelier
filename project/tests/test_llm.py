"""
Unit tests for llm_node.

Covers:
- Normal response with valid Wikipedia context
- Response with empty context (no wiki_result)
- Response with None context
- Response with very short context (single sentence)
- final_answer is always a non-empty string
- execution_id is preserved
"""

from __future__ import annotations

import pytest

from app.nodes import llm_node
from app.state import AgentState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RICH_CONTEXT = (
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasizes code readability with the use of significant indentation. "
    "Python is dynamically typed and garbage-collected. "
    "It supports multiple programming paradigms, including structured, object-oriented and functional programming."
)


@pytest.fixture()
def state_with_context() -> AgentState:
    """State with a rich Wikipedia extract available."""
    return AgentState(
        question="What is Python?",
        wiki_result=RICH_CONTEXT,
        error=None,
    )


@pytest.fixture()
def state_with_empty_context() -> AgentState:
    """State where wiki_result is an empty string."""
    return AgentState(
        question="What is Python?",
        wiki_result="",
        error=None,
    )


@pytest.fixture()
def state_with_none_context() -> AgentState:
    """State where wiki_result is None."""
    return AgentState(
        question="What is Python?",
        wiki_result=None,
        error=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_llm_node_normal_response(state_with_context: AgentState) -> None:
    """
    LLM node should populate final_answer with a non-empty string
    when valid Wikipedia context is provided.
    """
    result = llm_node(state_with_context)

    assert result.final_answer is not None
    assert isinstance(result.final_answer, str)
    assert len(result.final_answer.strip()) > 0


def test_llm_node_empty_context(state_with_empty_context: AgentState) -> None:
    """
    LLM node should still produce a final_answer when context is empty,
    indicating that no information was available.
    """
    result = llm_node(state_with_empty_context)

    assert result.final_answer is not None
    assert isinstance(result.final_answer, str)
    assert len(result.final_answer.strip()) > 0


def test_llm_node_none_context(state_with_none_context: AgentState) -> None:
    """
    LLM node should produce a meaningful final_answer even when
    wiki_result is None.
    """
    result = llm_node(state_with_none_context)

    assert result.final_answer is not None
    assert isinstance(result.final_answer, str)
    assert len(result.final_answer.strip()) > 0


def test_llm_node_none_context_answer_mentions_unavailability(
    state_with_none_context: AgentState,
) -> None:
    """
    When no context is available, the offline fallback should
    mention that context was unavailable.
    """
    result = llm_node(state_with_none_context)

    # The fallback answer should communicate the absence of context
    answer_lower = result.final_answer.lower()
    assert any(
        phrase in answer_lower
        for phrase in ["no", "not", "unavailable", "could not", "empty"]
    )


def test_llm_node_single_sentence_context() -> None:
    """LLM node should work correctly with a single-sentence context."""
    state = AgentState(
        question="What is gravity?",
        wiki_result="Gravity is the force that attracts objects with mass towards each other.",
        error=None,
    )
    result = llm_node(state)

    assert result.final_answer is not None
    assert len(result.final_answer.strip()) > 0


def test_llm_node_preserves_execution_id(state_with_context: AgentState) -> None:
    """execution_id must be unchanged after LLM node execution."""
    original_id = state_with_context.execution_id
    result = llm_node(state_with_context)

    assert result.execution_id == original_id


def test_llm_node_does_not_clear_wiki_result(state_with_context: AgentState) -> None:
    """LLM node should not overwrite the wiki_result field."""
    result = llm_node(state_with_context)

    assert result.wiki_result == RICH_CONTEXT


def test_llm_node_malformed_context() -> None:
    """LLM node should handle unicode-heavy and malformed context without raising."""
    state = AgentState(
        question="test",
        wiki_result="Ünïcödé têxt wïth spécïàl chàràctérs 🐍🔬 \x00\x01\x02",
        error=None,
    )
    result = llm_node(state)

    assert result.final_answer is not None
    assert len(result.final_answer.strip()) > 0
