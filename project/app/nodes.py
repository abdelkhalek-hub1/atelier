"""
LangGraph node implementations.

Each node is a pure function  ``(AgentState) -> AgentState``  that
performs one well-defined responsibility, handles its own errors, and
never lets exceptions propagate outside the node boundary.

Nodes
-----
- wikipedia_node  – fetches a Wikipedia summary; handles HTTP / network errors
- llm_node        – generates an LLM answer from the Wikipedia context
- dlq_node        – POSTs a failure payload to the DLQ webhook
"""

from __future__ import annotations

import time
from typing import Any

import requests

from app.config import config
from app.logger import get_logger, log_node_event
from app.state import AgentState

logger = get_logger("langgraph.nodes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_query(question: str) -> str:
    """
    Convert a free-text question into a Wikipedia-friendly slug.

    Strips trailing punctuation, collapses whitespace, and replaces
    spaces with underscores so the REST v1 summary endpoint can accept
    the value directly as a path segment.
    """
    slug = question.strip().rstrip("?!.")
    slug = " ".join(slug.split())          # normalise internal whitespace
    slug = slug.replace(" ", "_")
    return slug


# ---------------------------------------------------------------------------
# wikipedia_node
# ---------------------------------------------------------------------------

def wikipedia_node(state: AgentState) -> AgentState:
    """
    Fetch a Wikipedia article summary for the user's question.

    Handles
    -------
    - HTTP 404, 500, 503
    - requests.Timeout
    - requests.ConnectionError
    - Any unexpected exception

    On success the ``wiki_result`` field is populated.
    On failure the ``error`` field is populated; the workflow continues.

    Args:
        state: Current agent state containing the user's question.

    Returns:
        Updated AgentState with either wiki_result or error set.
    """
    start_time = time.perf_counter()
    log_node_event(
        execution_id=state.execution_id,
        node="wikipedia_node",
        status="started",
        extra={"question": state.question},
    )

    slug = _sanitise_query(state.question)
    url = f"{config.WIKIPEDIA_API_URL}{slug}"

    try:
        headers = {
            "User-Agent": "LangGraphWikipediaWorkflow/1.0 (contact@example.com; educational research bot)"
        }
        response = requests.get(url, headers=headers, timeout=config.WIKIPEDIA_TIMEOUT_SECONDS)

        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            extract: str = data.get("extract", "")
            if not extract:
                raise ValueError("Wikipedia returned an empty extract.")

            duration_ms = (time.perf_counter() - start_time) * 1_000
            log_node_event(
                execution_id=state.execution_id,
                node="wikipedia_node",
                status="success",
                duration_ms=duration_ms,
                extra={"url": url, "extract_length": len(extract)},
            )
            return state.with_updates(wiki_result=extract, error=None)

        elif response.status_code == 404:
            error_msg = f"Wikipedia article not found (404) for query: '{slug}'"
        elif response.status_code == 500:
            error_msg = (
                f"Wikipedia server error (500) for query: '{slug}'"
            )
        elif response.status_code == 503:
            error_msg = (
                f"Wikipedia service unavailable (503) for query: '{slug}'"
            )
        else:
            error_msg = (
                f"Wikipedia returned unexpected status {response.status_code} "
                f"for query: '{slug}'"
            )

    except requests.Timeout:
        error_msg = (
            f"Wikipedia API timed out after {config.WIKIPEDIA_TIMEOUT_SECONDS}s "
            f"for query: '{slug}'"
        )
    except requests.ConnectionError as exc:
        error_msg = f"Wikipedia API connection error for query '{slug}': {exc}"
    except ValueError as exc:
        error_msg = str(exc)
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"Unexpected error in wikipedia_node: {exc}"

    duration_ms = (time.perf_counter() - start_time) * 1_000
    log_node_event(
        execution_id=state.execution_id,
        node="wikipedia_node",
        status="failure",
        duration_ms=duration_ms,
        extra={"error": error_msg, "url": url},
    )
    return state.with_updates(error=error_msg, wiki_result=None)


# ---------------------------------------------------------------------------
# llm_node
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> AgentState:
    """
    Generate a final answer using the Wikipedia context.

    The node uses LangChain's ChatGoogleGenerativeAI under the hood.  In testing /
    offline environments the GEMINI_API_KEY may be absent; the node
    then falls back to a deterministic offline answer so the graph can
    still complete for integration tests.

    Args:
        state: Current agent state with wiki_result populated.

    Returns:
        Updated AgentState with final_answer set.
    """
    start_time = time.perf_counter()
    log_node_event(
        execution_id=state.execution_id,
        node="llm_node",
        status="started",
    )

    context: str = (state.wiki_result or "").strip()
    question: str = state.question.strip()

    # ------------------------------------------------------------------
    # Attempt real LLM call
    # ------------------------------------------------------------------
    answer: str | None = None
    llm_error: str | None = None

    if config.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-untyped]
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore[import-untyped]

            llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_output_tokens=config.LLM_MAX_TOKENS,
                google_api_key=config.GEMINI_API_KEY,
            )

            system_prompt = (
                "You are a knowledgeable assistant. "
                "Answer the user's question using ONLY the provided Wikipedia context. "
                "Be concise and accurate. "
                "If the context is empty or unhelpful, say so explicitly."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer:"
                    )
                ),
            ]

            result = llm.invoke(messages)
            answer = str(result.content).strip()

        except Exception as exc:  # pylint: disable=broad-except
            llm_error = str(exc)
            logger.warning(
                "LLM call failed, falling back to offline answer",
                extra={"error": llm_error, "execution_id": state.execution_id},
            )

    # ------------------------------------------------------------------
    # Offline / fallback answer
    # ------------------------------------------------------------------
    if answer is None:
        if context:
            # Use the first two sentences of the Wikipedia extract
            sentences = [s.strip() for s in context.replace("\n", " ").split(". ") if s.strip()]
            summary = ". ".join(sentences[:2])
            if summary and not summary.endswith("."):
                summary += "."
            answer = f"Based on Wikipedia: {summary}"
        else:
            answer = (
                "I could not generate an answer because no Wikipedia context "
                "was available for your question."
            )

    duration_ms = (time.perf_counter() - start_time) * 1_000
    log_node_event(
        execution_id=state.execution_id,
        node="llm_node",
        status="success",
        duration_ms=duration_ms,
        extra={"answer_length": len(answer)},
    )

    return state.with_updates(final_answer=answer)


# ---------------------------------------------------------------------------
# dlq_node
# ---------------------------------------------------------------------------

def dlq_node(state: AgentState) -> AgentState:
    """
    Forward a failed execution to the Dead Letter Queue webhook.

    The node POSTs a structured JSON payload to the configured DLQ URL.
    If the webhook POST itself fails, the error is logged but the
    workflow still completes gracefully – we never re-raise here.

    Args:
        state: Current agent state with error field populated.

    Returns:
        Updated AgentState with final_answer set to a DLQ acknowledgement.
    """
    start_time = time.perf_counter()
    log_node_event(
        execution_id=state.execution_id,
        node="dlq_node",
        status="started",
        extra={"error": state.error},
    )

    payload: dict[str, Any] = {
        "execution_id": state.execution_id,
        "question": state.question,
        "error": state.error or "Unknown error",
        "status": "FAILED_ROUTED_TO_DLQ",
        "timestamp": state.timestamp,
    }

    dlq_success = False
    try:
        response = requests.post(
            config.DLQ_WEBHOOK_URL,
            json=payload,
            timeout=config.DLQ_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        dlq_success = True
        logger.info(
            "DLQ webhook accepted payload",
            extra={
                "execution_id": state.execution_id,
                "dlq_status_code": response.status_code,
            },
        )
    except requests.Timeout:
        logger.error(
            "DLQ webhook timed out",
            extra={"execution_id": state.execution_id, "url": config.DLQ_WEBHOOK_URL},
        )
    except requests.ConnectionError as exc:
        logger.error(
            "DLQ webhook connection error",
            extra={"execution_id": state.execution_id, "error": str(exc)},
        )
    except requests.HTTPError as exc:
        logger.error(
            "DLQ webhook returned an error status",
            extra={
                "execution_id": state.execution_id,
                "error": str(exc),
                "status_code": exc.response.status_code if exc.response else None,
            },
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unexpected error posting to DLQ",
            extra={"execution_id": state.execution_id, "error": str(exc)},
        )

    duration_ms = (time.perf_counter() - start_time) * 1_000
    log_node_event(
        execution_id=state.execution_id,
        node="dlq_node",
        status="success" if dlq_success else "dlq_webhook_failed",
        duration_ms=duration_ms,
        extra={"payload": payload},
    )

    final_answer = (
        f"[DLQ] Execution {state.execution_id} was routed to the Dead Letter Queue. "
        f"Error: {state.error}"
    )
    return state.with_updates(final_answer=final_answer)
