"""
Shared pytest configuration and fixtures.

This module is automatically discovered by pytest and applies to all
test files in the ``tests/`` directory tree.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _disable_real_http_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure that no real external API calls can accidentally be made
    during the test suite by removing live credentials from the
    environment.

    The ``responses`` library handles HTTP-level interception; this
    fixture provides an additional defence-in-depth layer.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # Keep test output clean


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "unit: fast isolated unit tests")
    config.addinivalue_line(
        "markers", "integration: end-to-end workflow tests with mocked HTTP"
    )
    config.addinivalue_line("markers", "slow: tests that may take >1 s")
