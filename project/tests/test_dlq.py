"""
Unit tests for dlq_node.

Covers:
- DLQ webhook is called with correct HTTP method
- Payload contains all required fields
- Payload status field equals "FAILED_ROUTED_TO_DLQ"
- Payload error field is present and matches state.error
- Payload execution_id matches state.execution_id
- Node completes gracefully when webhook returns 4xx
- Node completes gracefully when webhook connection fails
- Node completes gracefully when webhook times out
- final_answer is set after DLQ node execution
"""

from __future__ import annotations

import json

import pytest
import requests
import responses as resp_lib

from app.nodes import dlq_node
from app.state import AgentState
from app.config import config


DLQ_URL = config.DLQ_WEBHOOK_URL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def failed_state() -> AgentState:
    """State representing a failed Wikipedia execution ready for DLQ routing."""
    return AgentState(
        question="What is the speed of light?",
        wiki_result=None,
        error="Wikipedia returned 503 for query: What_is_the_speed_of_light",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_dlq_node_calls_webhook(failed_state: AgentState) -> None:
    """DLQ node must issue exactly one POST request to the webhook URL."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    dlq_node(failed_state)

    assert len(resp_lib.calls) == 1
    assert resp_lib.calls[0].request.method == "POST"


@resp_lib.activate
def test_dlq_node_payload_contains_required_fields(failed_state: AgentState) -> None:
    """The POST body must contain execution_id, question, error, and status."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    dlq_node(failed_state)

    body = json.loads(resp_lib.calls[0].request.body)
    assert "execution_id" in body
    assert "question" in body
    assert "error" in body
    assert "status" in body


@resp_lib.activate
def test_dlq_node_payload_status_is_correct(failed_state: AgentState) -> None:
    """The payload status field must equal FAILED_ROUTED_TO_DLQ."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    dlq_node(failed_state)

    body = json.loads(resp_lib.calls[0].request.body)
    assert body["status"] == "FAILED_ROUTED_TO_DLQ"


@resp_lib.activate
def test_dlq_node_payload_error_matches_state(failed_state: AgentState) -> None:
    """The payload error field must match state.error."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    dlq_node(failed_state)

    body = json.loads(resp_lib.calls[0].request.body)
    assert body["error"] == failed_state.error


@resp_lib.activate
def test_dlq_node_payload_execution_id_matches_state(failed_state: AgentState) -> None:
    """The payload execution_id must match state.execution_id."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    dlq_node(failed_state)

    body = json.loads(resp_lib.calls[0].request.body)
    assert body["execution_id"] == failed_state.execution_id


@resp_lib.activate
def test_dlq_node_graceful_on_webhook_4xx(failed_state: AgentState) -> None:
    """DLQ node must not raise even if the webhook returns a 4xx status."""
    resp_lib.add(resp_lib.POST, DLQ_URL, status=400)

    # Should not raise
    result = dlq_node(failed_state)
    assert result.final_answer is not None


@resp_lib.activate
def test_dlq_node_graceful_on_connection_error(failed_state: AgentState) -> None:
    """DLQ node must not raise on a requests.ConnectionError."""
    resp_lib.add(
        resp_lib.POST,
        DLQ_URL,
        body=requests.ConnectionError("Connection refused"),
    )

    result = dlq_node(failed_state)
    assert result.final_answer is not None


@resp_lib.activate
def test_dlq_node_graceful_on_timeout(failed_state: AgentState) -> None:
    """DLQ node must not raise on a requests.Timeout."""
    resp_lib.add(
        resp_lib.POST,
        DLQ_URL,
        body=requests.Timeout("Webhook timed out"),
    )

    result = dlq_node(failed_state)
    assert result.final_answer is not None


@resp_lib.activate
def test_dlq_node_sets_final_answer(failed_state: AgentState) -> None:
    """DLQ node must always populate final_answer after execution."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    result = dlq_node(failed_state)

    assert result.final_answer is not None
    assert isinstance(result.final_answer, str)
    assert len(result.final_answer) > 0


@resp_lib.activate
def test_dlq_node_final_answer_contains_execution_id(failed_state: AgentState) -> None:
    """final_answer should reference the execution_id for traceability."""
    resp_lib.add(resp_lib.POST, DLQ_URL, json={"status": "ok"}, status=200)

    result = dlq_node(failed_state)

    assert failed_state.execution_id in result.final_answer
