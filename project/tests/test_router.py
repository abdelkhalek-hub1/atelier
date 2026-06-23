"""
Unit tests for router_function.

Covers:
- Routes to success when wiki_result is present and error is None
- Routes to failure when error is set
- Routes to failure when wiki_result is empty string
- Routes to failure when wiki_result is None
- Routes to failure when both error and wiki_result are set (error takes precedence)
- Routes to failure when wiki_result is whitespace-only
"""

from __future__ import annotations

import pytest

from app.router import router_function, SUCCESS_ROUTE, FAILURE_ROUTE
from app.state import AgentState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def success_state() -> AgentState:
    """State representing a successful Wikipedia fetch."""
    return AgentState(
        question="What is Python?",
        wiki_result="Python is a high-level programming language.",
        error=None,
    )


@pytest.fixture()
def failure_state_with_error() -> AgentState:
    """State representing a failed Wikipedia fetch with an error message."""
    return AgentState(
        question="What is Python?",
        wiki_result=None,
        error="Wikipedia returned 404 for query: What_is_Python",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_router_routes_to_success_when_wiki_result_present(
    success_state: AgentState,
) -> None:
    """Router should return 'success' when wiki_result is populated and error is None."""
    route = router_function(success_state)
    assert route == SUCCESS_ROUTE


def test_router_routes_to_failure_when_error_is_set(
    failure_state_with_error: AgentState,
) -> None:
    """Router should return 'failure' when error field is set."""
    route = router_function(failure_state_with_error)
    assert route == FAILURE_ROUTE


def test_router_routes_to_failure_when_wiki_result_is_none() -> None:
    """Router should return 'failure' when wiki_result is None (no error either)."""
    state = AgentState(
        question="What is gravity?",
        wiki_result=None,
        error=None,
    )
    route = router_function(state)
    assert route == FAILURE_ROUTE


def test_router_routes_to_failure_when_wiki_result_is_empty_string() -> None:
    """Router should return 'failure' when wiki_result is an empty string."""
    state = AgentState(
        question="What is gravity?",
        wiki_result="",
        error=None,
    )
    route = router_function(state)
    assert route == FAILURE_ROUTE


def test_router_routes_to_failure_when_wiki_result_is_whitespace() -> None:
    """Router should treat a whitespace-only wiki_result as a failure."""
    state = AgentState(
        question="What is gravity?",
        wiki_result="   \n\t  ",
        error=None,
    )
    route = router_function(state)
    assert route == FAILURE_ROUTE


def test_router_failure_takes_precedence_over_wiki_result() -> None:
    """
    When both wiki_result and error are set (edge case), error takes precedence.
    This should not normally occur but the router must handle it defensively.
    """
    state = AgentState(
        question="What is Python?",
        wiki_result="Python is a language.",
        error="Partial failure occurred",
    )
    route = router_function(state)
    assert route == FAILURE_ROUTE


def test_router_return_type_is_string(success_state: AgentState) -> None:
    """Router must return a plain string, not an enum or other type."""
    route = router_function(success_state)
    assert isinstance(route, str)
