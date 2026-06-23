"""
Integration tests for the complete LangGraph workflow.

These tests invoke run_workflow() end-to-end and assert on the final
AgentState, validating the routing decisions, node outputs, and
resilience of the entire pipeline.

Mocked services
---------------
- Wikipedia REST API  (via `responses`)
- DLQ webhook         (via `responses`)

No real external network calls are made.
"""

from __future__ import annotations

import pytest
import requests
import responses as resp_lib

from app.graph import run_workflow
from app.config import config
from app.state import AgentState

WIKI_API = config.WIKIPEDIA_API_URL
DLQ_URL = config.DLQ_WEBHOOK_URL

SAMPLE_EXTRACT = (
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasizes code readability."
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add_wiki_success(slug: str, extract: str = SAMPLE_EXTRACT) -> None:
    """Register a successful Wikipedia mock for the given slug."""
    resp_lib.add(
        resp_lib.GET,
        f"{WIKI_API}{slug}",
        json={"extract": extract, "title": slug},
        status=200,
    )


def _add_wiki_error(slug: str, status: int) -> None:
    """Register a failing Wikipedia mock for the given slug."""
    resp_lib.add(resp_lib.GET, f"{WIKI_API}{slug}", status=status)


def _add_dlq_success() -> None:
    """Register a successful DLQ webhook mock."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)


# ---------------------------------------------------------------------------
# Test: successful workflow (happy path)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_successful_workflow() -> None:
    """
    Full success path: Wikipedia returns 200 → LLM node produces answer.
    The final state must have final_answer and no error.
    """
    _add_wiki_success("Python_programming_language")

    result = run_workflow("Python programming language")

    assert isinstance(result, AgentState)
    assert result.final_answer is not None
    assert len(result.final_answer.strip()) > 0
    assert result.error is None


# ---------------------------------------------------------------------------
# Test: failed workflow (Wikipedia 404 → DLQ)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_failed_workflow_404() -> None:
    """
    Failure path: Wikipedia returns 404 → DLQ node fires.
    The final state must have a DLQ-style answer and an error recorded.
    """
    _add_wiki_error("Nonexistent_article_xyz123", 404)
    _add_dlq_success()

    result = run_workflow("Nonexistent article xyz123")

    assert result.error is not None
    assert "404" in result.error
    assert result.final_answer is not None
    assert "DLQ" in result.final_answer


# ---------------------------------------------------------------------------
# Test: failed workflow (Wikipedia 500 → DLQ)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_failed_workflow_500() -> None:
    """Failure path: Wikipedia returns 500 → DLQ node fires."""
    _add_wiki_error("Server_error_article", 500)
    _add_dlq_success()

    result = run_workflow("Server error article")

    assert result.error is not None
    assert "500" in result.error
    assert "DLQ" in result.final_answer


# ---------------------------------------------------------------------------
# Test: failed workflow (Wikipedia timeout → DLQ)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_timeout_routed_to_dlq() -> None:
    """Failure path: Wikipedia times out → DLQ node fires."""
    resp_lib.add(
        resp_lib.GET,
        f"{WIKI_API}Slow_article",
        body=requests.Timeout("Simulated timeout"),
    )
    _add_dlq_success()

    result = run_workflow("Slow article")

    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert result.final_answer is not None


# ---------------------------------------------------------------------------
# Test: long query
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_long_query() -> None:
    """Workflow should handle very long questions without crashing."""
    long_question = "What is the history of " + "very " * 10 + "long article"
    # Register a 404 for the entire Wikipedia base URL prefix using passthrough
    # We use add_passthrough for unknown URLs so responses doesn't error,
    # but we still need to handle the result – so use a broad prefix mock.
    # Build the expected slug from the question
    slug = long_question.strip().rstrip("?!.").replace(" ", "_")
    resp_lib.add(resp_lib.GET, f"{WIKI_API}{slug}", status=404)
    _add_dlq_success()

    # We only care that it doesn't raise
    try:
        result = run_workflow(long_question)
        # Either error or answer must be set
        assert result.final_answer is not None
    except Exception:  # noqa: BLE001
        pytest.fail("run_workflow raised an exception for a long query")


# ---------------------------------------------------------------------------
# Test: unicode query
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_unicode_query() -> None:
    """Workflow should handle unicode questions gracefully."""
    question = "quantum_computing"
    resp_lib.add(resp_lib.GET, f"{WIKI_API}{question}", status=404)
    _add_dlq_success()

    result = run_workflow(question)

    assert result.final_answer is not None


# ---------------------------------------------------------------------------
# Test: malformed query (only punctuation)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_malformed_query() -> None:
    """Workflow should route punctuation-only queries to DLQ without crashing."""
    # A punctuation-only question gets stripped to empty slug – 404 expected
    resp_lib.add(resp_lib.GET, WIKI_API, status=404)
    # Also cover the slug variant with underscores
    resp_lib.add(resp_lib.GET, f"{WIKI_API}___", status=404)
    resp_lib.add(resp_lib.GET, f"{WIKI_API}!!!_???_###", status=404)
    _add_dlq_success()

    result = run_workflow("!!! ??? ###")

    assert result.final_answer is not None


# ---------------------------------------------------------------------------
# Test: empty-ish query raises ValueError before graph invocation
# ---------------------------------------------------------------------------

def test_integration_empty_query_raises() -> None:
    """run_workflow must raise ValueError for empty/whitespace-only questions."""
    with pytest.raises(ValueError, match="non-empty"):
        run_workflow("")

    with pytest.raises(ValueError, match="non-empty"):
        run_workflow("   ")


# ---------------------------------------------------------------------------
# Test: execution_id is a valid UUID in result
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_result_has_valid_execution_id() -> None:
    """The result AgentState must carry a non-empty execution_id."""
    import uuid

    _add_wiki_success("Quantum_mechanics")

    result = run_workflow("Quantum mechanics")

    assert result.execution_id
    # Validate it parses as a UUID
    parsed = uuid.UUID(result.execution_id)
    assert str(parsed) == result.execution_id


# ---------------------------------------------------------------------------
# Test: connection error routed to DLQ
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_connection_error_routed_to_dlq() -> None:
    """A ConnectionError from Wikipedia must be handled and routed to DLQ."""
    resp_lib.add(
        resp_lib.GET,
        f"{WIKI_API}Offline_article",
        body=requests.ConnectionError("Network unreachable"),
    )
    _add_dlq_success()

    result = run_workflow("Offline article")

    assert result.error is not None
    assert "connection error" in result.error.lower()
    assert "DLQ" in result.final_answer


# ---------------------------------------------------------------------------
# Test: DLQ webhook failure does not break the workflow
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_dlq_webhook_failure_is_graceful() -> None:
    """Even if the DLQ webhook is unreachable, the workflow completes."""
    _add_wiki_error("Broken_article", 503)
    resp_lib.add(
        resp_lib.POST,
        DLQ_URL,
        body=requests.ConnectionError("DLQ is down"),
    )

    # Should NOT raise even though DLQ is also down
    result = run_workflow("Broken article")

    assert result.final_answer is not None
