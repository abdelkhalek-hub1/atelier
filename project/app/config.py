"""
Application configuration loaded from environment variables.

All external URLs, timeouts, and feature flags are centralised here
so they can be overridden in CI/CD without touching source code.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()


class Config:
    """
    Singleton-style configuration object.

    Values are read from environment variables at import time so
    the entire test suite can override them via monkeypatching or
    by setting variables before import.
    """

    # ------------------------------------------------------------------
    # Wikipedia API
    # ------------------------------------------------------------------
    WIKIPEDIA_API_URL: str = os.getenv(
        "WIKIPEDIA_API_URL",
        "https://en.wikipedia.org/api/rest_v1/page/summary/",
    )
    WIKIPEDIA_TIMEOUT_SECONDS: float = float(
        os.getenv("WIKIPEDIA_TIMEOUT_SECONDS", "10")
    )

    # ------------------------------------------------------------------
    # Dead Letter Queue webhook
    # ------------------------------------------------------------------
    DLQ_WEBHOOK_URL: str = os.getenv(
        "DLQ_WEBHOOK_URL",
        "https://webhook.site/langgraph-dlq",
    )
    DLQ_TIMEOUT_SECONDS: float = float(os.getenv("DLQ_TIMEOUT_SECONDS", "5"))

    # ------------------------------------------------------------------
    # LangSmith observability
    # ------------------------------------------------------------------
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv(
        "LANGSMITH_PROJECT", "langgraph-wikipedia-workflow"
    )
    LANGSMITH_TRACING_ENABLED: bool = (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )

    # ------------------------------------------------------------------
    # LLM (Google Gemini)
    # ------------------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # "json" | "text"


# Expose a single shared instance
config = Config()
