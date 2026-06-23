# LangGraph Wikipedia Agentic Workflow

[![CI](https://github.com/your-org/langgraph-wikipedia-workflow/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/langgraph-wikipedia-workflow/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green.svg)
![Tests](https://img.shields.io/badge/tests-45-brightgreen.svg)

A **production-ready agentic AI pipeline** built with LangGraph that answers user questions using Wikipedia as a knowledge source, with automatic failure detection, Dead Letter Queue (DLQ) routing, LangSmith observability, and full CI/CD.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Running the Workflow](#running-the-workflow)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [GitHub Actions CI/CD](#github-actions-cicd)
8. [LangSmith Tracing](#langsmith-tracing)
9. [DLQ Explained](#dlq-explained)
10. [LangGraph Explained](#langgraph-explained)

---

## Architecture

```
START
  │
  ▼
┌─────────────────────────────┐
│       wikipedia_node        │
│  GET /api/rest_v1/summary/  │
└─────────────────────────────┘
         │              │
    success         failure
  (200 + text)  (404/500/503/Timeout)
         │              │
         ▼              ▼
   ┌──────────┐   ┌──────────┐
   │ llm_node │   │ dlq_node │
   │ (answer) │   │ (webhook)│
   └──────────┘   └──────────┘
         │              │
         └──────┬───────┘
                ▼
               END
```

### Node Responsibilities

| Node | Responsibility | On Failure |
|------|---------------|------------|
| `wikipedia_node` | Fetch Wikipedia article summary | Sets `state.error` |
| `router_function` | Inspect state and choose path | N/A (conditional edge) |
| `llm_node` | Generate LLM answer from context | Falls back to offline answer |
| `dlq_node` | POST failure payload to DLQ webhook | Logs, never re-raises |

---

## Project Structure

```
project/
├── app/
│   ├── __init__.py         # Package marker
│   ├── config.py           # Centralised env-var configuration
│   ├── graph.py            # LangGraph workflow assembly + public API
│   ├── logger.py           # Structured JSON logger
│   ├── nodes.py            # wikipedia_node, llm_node, dlq_node
│   ├── router.py           # router_function (conditional edge)
│   └── state.py            # AgentState Pydantic model
│
├── tests/
│   ├── __init__.py
│   ├── test_wikipedia.py   # 9 unit tests
│   ├── test_router.py      # 7 unit tests
│   ├── test_dlq.py         # 10 unit tests
│   ├── test_llm.py         # 8 unit tests
│   └── test_integration.py # 11 integration tests
│
├── docs/
│   ├── architecture.md     # Detailed architecture documentation
│   └── workflow.json       # Machine-readable workflow spec
│
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
│
├── pyproject.toml          # Pytest + coverage configuration
├── requirements.txt        # All dependencies
└── README.md               # This file
```

---

## Setup & Installation

### Prerequisites

- Python **3.11** or higher
- `pip`

### Install

```bash
# Clone the repository
git clone https://github.com/your-org/langgraph-wikipedia-workflow.git
cd langgraph-wikipedia-workflow/project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows PowerShell

# Install all dependencies
pip install -r requirements.txt
```

### Environment Variables (Optional)

Copy and edit the template:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For real LLM calls | OpenAI API key |
| `LANGSMITH_API_KEY` | For tracing | LangSmith API key |
| `LANGSMITH_TRACING` | Optional | Set `true` to enable |
| `DLQ_WEBHOOK_URL` | Optional | Override default DLQ URL |

> **Note:** Without `OPENAI_API_KEY`, the LLM node uses an offline fallback (sentence extraction from Wikipedia). All tests pass without any API keys.

---

## Running the Workflow

```python
from app.graph import run_workflow

result = run_workflow("What is quantum mechanics?")

print(result.final_answer)
print(result.execution_id)
print(result.error)  # None on success
```

### Command-line (quick test)

```bash
python -c "
from app.graph import run_workflow
r = run_workflow('Python programming language')
print('Answer:', r.final_answer[:200])
print('Error:', r.error)
"
```

---

## Configuration

All settings are in [`app/config.py`](app/config.py) and read from environment variables:

```bash
# Wikipedia
export WIKIPEDIA_TIMEOUT_SECONDS=10

# DLQ Webhook
export DLQ_WEBHOOK_URL=https://your-dlq-endpoint.com/webhook

# LLM
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
export LLM_TEMPERATURE=0.2

# LangSmith
export LANGSMITH_API_KEY=ls__...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=my-project

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json   # or "text" for local dev
```

---

## Testing

### Run All Tests

```bash
pytest
```

### Run with Verbose Output

```bash
pytest -v
```

### Run Specific Test File

```bash
pytest tests/test_wikipedia.py -v
pytest tests/test_integration.py -v
```

### Run with Coverage Report

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### Test Summary

| File | Type | Tests |
|------|------|-------|
| `test_wikipedia.py` | Unit | 9 |
| `test_router.py` | Unit | 7 |
| `test_dlq.py` | Unit | 10 |
| `test_llm.py` | Unit | 8 |
| `test_integration.py` | Integration | 11 |
| **Total** | | **45** |

All tests use the `responses` library to mock HTTP calls. **No real API calls are made during testing.**

---

## GitHub Actions CI/CD

The pipeline (`.github/workflows/ci.yml`) runs on every push and pull request:

```
push / pull_request
        │
        ▼
   ┌─────────┐
   │  lint   │  ruff + mypy
   └─────────┘
        │
        ▼
   ┌─────────┐
   │  test   │  pytest + coverage + JUnit XML
   └─────────┘
        │
        ▼
   ┌─────────┐
   │  docs   │  validate workflow.json
   └─────────┘
```

### Key Features

- **Python 3.11** on `ubuntu-latest`
- **Dependency caching** via `actions/setup-python`
- **JUnit XML** test report uploaded as an artifact
- **Coverage XML** uploaded as an artifact
- **Test results** published as PR comments via `EnricoMi/publish-unit-test-result-action`
- **Fails immediately** if any test fails (`exit code ≠ 0`)
- **No real API calls** — all external services are mocked in tests

### Secrets Required in GitHub

| Secret | Description |
|--------|-------------|
| None required | All tests run with mocked services |
| `OPENAI_API_KEY` | Optional: only if you enable real LLM calls in CI |
| `LANGSMITH_API_KEY` | Optional: only if you enable tracing in CI |

---

## LangSmith Tracing

[LangSmith](https://smith.langchain.com/) provides full observability for LangChain and LangGraph workflows.

### Enable Tracing

```bash
export LANGSMITH_API_KEY=ls__your_key_here
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=langgraph-wikipedia-workflow
```

### What is Traced

- Every node invocation with input/output
- LLM calls (model, prompt, response, token usage)
- Routing decisions (`router_function` output)
- Wall-clock timing per node
- Errors and exception details

### Structured Logs (always on)

Even without LangSmith, every node emits JSON structured logs:

```json
{
  "level": "INFO",
  "logger": "langgraph.nodes",
  "message": "node_event",
  "execution_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "node": "wikipedia_node",
  "status": "success",
  "duration_ms": 142.5,
  "extract_length": 1024
}
```

---

## DLQ Explained

A **Dead Letter Queue (DLQ)** is an architectural pattern for handling messages that cannot be successfully processed. Instead of silently dropping failed requests, they are forwarded to a dedicated sink for inspection, alerting, and reprocessing.

### How It Works in This Workflow

```
wikipedia_node fails
        │
        ▼
router_function → "failure"
        │
        ▼
dlq_node
  ├── POST payload to DLQ_WEBHOOK_URL
  ├── Log structured error entry
  └── Set final_answer = "[DLQ] Execution ... routed to DLQ"
```

### DLQ Payload

```json
{
  "execution_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "question": "What is the speed of light?",
  "error": "Wikipedia returned 503 for query: What_is_the_speed_of_light",
  "status": "FAILED_ROUTED_TO_DLQ",
  "timestamp": "2026-06-23T12:00:00+00:00"
}
```

### Resilience

The DLQ node itself is fault-tolerant:
- If the webhook is unreachable, the error is **logged** but the workflow still completes.
- The `final_answer` is always populated (with a DLQ acknowledgement message).
- The `execution_id` is carried through for end-to-end traceability.

### Integration Options

The DLQ webhook can forward to:
- **AWS SQS / SNS** — for retry pipelines
- **Google Pub/Sub** — for event-driven processing
- **Slack / PagerDuty** — for alerting
- **Datadog / Grafana** — for monitoring dashboards
- **webhook.site** — for local development inspection

---

## LangGraph Explained

[LangGraph](https://langchain-ai.github.io/langgraph/) is a framework for building stateful, multi-actor applications with LLMs using a graph-based programming model.

### Key Concepts Used

| Concept | Usage in This Project |
|---------|----------------------|
| `StateGraph` | Holds the `AgentState` dict and drives execution |
| **Nodes** | Pure functions `(state) → state` for each step |
| **Edges** | Unconditional links between nodes |
| **Conditional Edges** | `router_function` selects next node dynamically |
| `START` / `END` | LangGraph built-in entry/exit sentinels |
| `compile()` | Validates the graph and returns an invocable object |

### Why LangGraph?

- **Explicit control flow** — routing logic is code, not prompt engineering
- **Type-safe state** — Pydantic models prevent runtime type errors
- **Built-in observability** — native LangSmith integration
- **Resilience patterns** — easy to add retries, fallbacks, and DLQ routing
- **Testability** — pure functions are trivial to unit test

---

## License

MIT License — see `LICENSE` for details.
