"""
Structured JSON logger for the LangGraph workflow.

Every node emits a structured log entry with the shape:

    {
        "execution_id": "...",
        "node": "...",
        "status": "...",
        "duration_ms": 123,
        ...extra fields...
    }

The logger writes to stdout so it works seamlessly with container
runtimes and GitHub Actions log streaming.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from app.config import config


# ---------------------------------------------------------------------------
# Low-level formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields the caller attached
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                payload[key] = value
        return json.dumps(payload, default=str)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    handler = logging.StreamHandler(sys.stdout)
    if config.LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    logger.addHandler(handler)
    logger.setLevel(config.LOG_LEVEL)
    logger.propagate = False
    return logger


# Module-level default logger
_logger = _build_logger("langgraph.workflow")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_logger(name: str = "langgraph.workflow") -> logging.Logger:
    """Return (or create) a named structured logger."""
    return _build_logger(name)


def log_node_event(
    *,
    execution_id: str,
    node: str,
    status: str,
    duration_ms: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Emit a standardised structured log entry for a node lifecycle event.

    Args:
        execution_id: Workflow execution UUID.
        node: Name of the LangGraph node.
        status: One of "started", "success", "failure", etc.
        duration_ms: Wall-clock time spent in the node (milliseconds).
        extra: Additional key/value pairs to include in the log record.
    """
    payload: dict[str, Any] = {
        "execution_id": execution_id,
        "node": node,
        "status": status,
    }
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 2)
    if extra:
        payload.update(extra)

    _logger.info("node_event", extra=payload)


@contextmanager
def timed_node(
    *,
    execution_id: str,
    node: str,
) -> Generator[None, None, None]:
    """
    Context manager that logs node start/end with wall-clock timing.

    Usage::

        with timed_node(execution_id=state.execution_id, node="wikipedia_node"):
            # do work
    """
    log_node_event(execution_id=execution_id, node=node, status="started")
    start = time.perf_counter()
    try:
        yield
        duration_ms = (time.perf_counter() - start) * 1_000
        log_node_event(
            execution_id=execution_id,
            node=node,
            status="success",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1_000
        log_node_event(
            execution_id=execution_id,
            node=node,
            status="failure",
            duration_ms=duration_ms,
            extra={"error": str(exc)},
        )
        raise
