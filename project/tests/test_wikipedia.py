"""
Unit tests for wikipedia_node.

Covers:
- Successful Wikipedia response
- HTTP 404
- HTTP 500
- HTTP 503
- Request timeout
- Connection error
- Empty extract in response
- Unexpected HTTP status code
"""

from __future__ import annotations

import pytest
import responses as resp_lib
import requests

from app.nodes import wikipedia_node
from app.state import AgentState
from app.config import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_state() -> AgentState:
    """Return a fresh AgentState with a simple question."""
    return AgentState(question="Python programming language")


WIKI_API = config.WIKIPEDIA_API_URL  # e.g. "https://en.wikipedia.org/api/rest_v1/page/summary/"
ARTICLE_URL = f"{WIKI_API}Python_programming_language"

SAMPLE_EXTRACT = (
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasizes code readability with the use of significant indentation."
)


# ---------------------------------------------------------------------------
# Test: successful response
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_success(base_state: AgentState) -> None:
    """Node should populate wiki_result on a 200 response with a valid extract."""
    resp_lib.add(
        resp_lib.GET,
        ARTICLE_URL,
        json={"extract": SAMPLE_EXTRACT, "title": "Python (programming language)"},
        status=200,
    )

    result = wikipedia_node(base_state)

    assert result.error is None
    assert result.wiki_result == SAMPLE_EXTRACT


# ---------------------------------------------------------------------------
# Test: HTTP 404
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_404(base_state: AgentState) -> None:
    """Node should set error and clear wiki_result on a 404 response."""
    resp_lib.add(resp_lib.GET, ARTICLE_URL, status=404)

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None
    assert "404" in result.error


# ---------------------------------------------------------------------------
# Test: HTTP 500
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_500(base_state: AgentState) -> None:
    """Node should set error and clear wiki_result on a 500 response."""
    resp_lib.add(resp_lib.GET, ARTICLE_URL, status=500)

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None
    assert "500" in result.error


# ---------------------------------------------------------------------------
# Test: HTTP 503
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_503(base_state: AgentState) -> None:
    """Node should set error and clear wiki_result on a 503 response."""
    resp_lib.add(resp_lib.GET, ARTICLE_URL, status=503)

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None
    assert "503" in result.error


# ---------------------------------------------------------------------------
# Test: timeout
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_timeout(base_state: AgentState) -> None:
    """Node should set error on a requests.Timeout."""
    resp_lib.add(
        resp_lib.GET,
        ARTICLE_URL,
        body=requests.Timeout("Connection timed out"),
    )

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None
    assert "timed out" in result.error.lower()


# ---------------------------------------------------------------------------
# Test: connection error
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_connection_error(base_state: AgentState) -> None:
    """Node should set error on a requests.ConnectionError."""
    resp_lib.add(
        resp_lib.GET,
        ARTICLE_URL,
        body=requests.ConnectionError("Name or service not known"),
    )

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None
    assert "connection error" in result.error.lower()


# ---------------------------------------------------------------------------
# Test: empty extract in response body
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_empty_extract(base_state: AgentState) -> None:
    """Node should treat an empty extract as a failure."""
    resp_lib.add(
        resp_lib.GET,
        ARTICLE_URL,
        json={"extract": "", "title": "Python (programming language)"},
        status=200,
    )

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None


# ---------------------------------------------------------------------------
# Test: unexpected HTTP status code
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_unexpected_status(base_state: AgentState) -> None:
    """Node should handle unexpected status codes gracefully."""
    resp_lib.add(resp_lib.GET, ARTICLE_URL, status=418)

    result = wikipedia_node(base_state)

    assert result.wiki_result is None
    assert result.error is not None
    assert "418" in result.error


# ---------------------------------------------------------------------------
# Test: execution_id is preserved across node call
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_preserves_execution_id(base_state: AgentState) -> None:
    """The execution_id must remain unchanged after node execution."""
    resp_lib.add(resp_lib.GET, ARTICLE_URL, status=404)

    result = wikipedia_node(base_state)

    assert result.execution_id == base_state.execution_id


# ---------------------------------------------------------------------------
# Test: question with spaces gets slugified correctly
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_wikipedia_node_question_slugification() -> None:
    """Spaces in the question should be converted to underscores in the URL."""
    state = AgentState(question="Albert Einstein")
    expected_url = f"{WIKI_API}Albert_Einstein"

    resp_lib.add(
        resp_lib.GET,
        expected_url,
        json={"extract": "Albert Einstein was a theoretical physicist."},
        status=200,
    )

    result = wikipedia_node(state)

    assert result.error is None
    assert "Albert Einstein" in result.wiki_result
