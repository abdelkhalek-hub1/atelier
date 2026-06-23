# Architecture — LangGraph Wikipedia Workflow

## 1. Overview

This document describes the technical architecture of the **LangGraph Wikipedia Agentic Workflow** — a production-ready agentic AI pipeline that answers user questions using Wikipedia as a knowledge source, with automatic failure detection and Dead Letter Queue (DLQ) routing.

---

## 2. Workflow Topology

```
START
  │
  ▼
┌─────────────────────┐
│   wikipedia_node    │  ← fetches Wikipedia REST API
└─────────────────────┘
  │               │
  │ success       │ failure (404 / 500 / 503 / Timeout / ConnectionError)
  ▼               ▼
┌──────────┐  ┌──────────┐
│ llm_node │  │ dlq_node │
└──────────┘  └──────────┘
  │               │
  └───────┬───────┘
          ▼
         END
```

The `router_function` conditional edge inspects `AgentState` after `wikipedia_node` completes and directs the graph to either `llm_node` (success path) or `dlq_node` (failure path).

---

## 3. Nodes

### 3.1 `wikipedia_node`

| Attribute | Value |
|-----------|-------|
| **Module** | `app/nodes.py` |
| **Responsibility** | Fetch a Wikipedia article summary |
| **External call** | `GET https://en.wikipedia.org/api/rest_v1/page/summary/{slug}` |
| **Timeout** | `WIKIPEDIA_TIMEOUT_SECONDS` (default: 10 s) |

**Error handling matrix:**

| Condition | Behaviour |
|-----------|-----------|
| HTTP 200 + non-empty extract | Sets `wiki_result`, clears `error` |
| HTTP 200 + empty extract | Sets `error` "empty extract" |
| HTTP 404 | Sets `error` with status code |
| HTTP 500 | Sets `error` with status code |
| HTTP 503 | Sets `error` with status code |
| Any other HTTP status | Sets `error` with status code |
| `requests.Timeout` | Sets `error` "timed out" |
| `requests.ConnectionError` | Sets `error` "connection error" |
| Unexpected exception | Sets `error` "unexpected error …" |

The node **never raises an exception**. All failures are captured into `state.error`.

### 3.2 `router_function`

| Attribute | Value |
|-----------|-------|
| **Module** | `app/router.py` |
| **Type** | LangGraph conditional edge function |
| **Returns** | `"success"` or `"failure"` |

**Decision logic:**

```python
if state.error or not state.wiki_result.strip():
    return "failure"   # → dlq_node
return "success"       # → llm_node
```

### 3.3 `llm_node`

| Attribute | Value |
|-----------|-------|
| **Module** | `app/nodes.py` |
| **Responsibility** | Generate a natural-language answer |
| **LLM provider** | OpenAI (via `langchain-openai`) |
| **Model** | `LLM_MODEL` env var (default: `gpt-4o-mini`) |
| **Fallback** | Offline sentence extraction (no API key needed) |

When `OPENAI_API_KEY` is absent or the LLM call fails, the node extracts the first two sentences of the Wikipedia extract as its answer. This ensures the workflow remains fully functional in offline/CI environments.

### 3.4 `dlq_node`

| Attribute | Value |
|-----------|-------|
| **Module** | `app/nodes.py` |
| **Responsibility** | Forward failures to the Dead Letter Queue webhook |
| **External call** | `POST DLQ_WEBHOOK_URL` |
| **Timeout** | `DLQ_TIMEOUT_SECONDS` (default: 5 s) |

**Payload:**

```json
{
  "execution_id": "uuid4",
  "question": "user question",
  "error": "error description",
  "status": "FAILED_ROUTED_TO_DLQ",
  "timestamp": "2026-06-23T12:00:00+00:00"
}
```

The DLQ node also **never raises**. Webhook failures are logged as errors, but the workflow still produces a `final_answer`.

---

## 4. State Model

`AgentState` (defined in `app/state.py`) is a **Pydantic v2 BaseModel** threaded through the entire graph:

```python
class AgentState(BaseModel):
    question: str                     # user input
    wiki_result: Optional[str]        # Wikipedia extract (success)
    final_answer: Optional[str]       # LLM or DLQ answer (terminal)
    error: Optional[str]              # error message (failure)
    execution_id: str                 # UUID4 (auto-generated)
    timestamp: str                    # ISO-8601 UTC (auto-generated)
```

LangGraph's `StateGraph(dict)` is used internally; `AgentState` is serialised to/from dict at the graph boundary via adapters in `app/graph.py`.

---

## 5. DLQ Mechanism

The **Dead Letter Queue** pattern captures messages that cannot be processed successfully and routes them to a separate sink for review, alerting, or reprocessing.

In this workflow:

1. `wikipedia_node` encounters a failure (network, HTTP error, etc.).
2. `router_function` detects `state.error` and returns `"failure"`.
3. `dlq_node` posts a structured JSON payload to the DLQ webhook URL.
4. The DLQ entry can be consumed by:
   - A monitoring dashboard (e.g., Datadog, Grafana)
   - A retry queue (e.g., SQS, Pub/Sub)
   - An alerting pipeline (e.g., PagerDuty, Slack)

The DLQ URL is configured via the `DLQ_WEBHOOK_URL` environment variable, making it trivial to swap between environments.

---

## 6. Observability

### 6.1 Structured Logging

Every node emits JSON log records to stdout:

```json
{
  "level": "INFO",
  "logger": "langgraph.nodes",
  "message": "node_event",
  "execution_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "node": "wikipedia_node",
  "status": "success",
  "duration_ms": 142.5
}
```

Use `LOG_FORMAT=text` for human-readable output during local development.

### 6.2 LangSmith Tracing

When `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set, the workflow automatically enables LangChain's `LANGCHAIN_TRACING_V2` integration. Every node invocation, LLM call, and routing decision is traced and visible in the LangSmith dashboard.

---

## 7. Configuration

All configuration is in `app/config.py` and read from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKIPEDIA_API_URL` | `https://en.wikipedia.org/api/rest_v1/page/summary/` | Wikipedia REST endpoint |
| `WIKIPEDIA_TIMEOUT_SECONDS` | `10` | Request timeout |
| `DLQ_WEBHOOK_URL` | `https://webhook.site/langgraph-dlq` | DLQ POST endpoint |
| `DLQ_TIMEOUT_SECONDS` | `5` | DLQ request timeout |
| `OPENAI_API_KEY` | *(empty)* | OpenAI key for LLM node |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_MAX_TOKENS` | `512` | Maximum tokens |
| `LANGSMITH_API_KEY` | *(empty)* | LangSmith API key |
| `LANGSMITH_PROJECT` | `langgraph-wikipedia-workflow` | LangSmith project |
| `LANGSMITH_TRACING` | `false` | Enable tracing |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FORMAT` | `json` | `json` or `text` |

---

## 8. Testing Strategy

### 8.1 Unit Tests

Each node is tested in complete isolation using the `responses` library to mock HTTP calls. No real network calls are made.

| Module | File | Tests |
|--------|------|-------|
| `wikipedia_node` | `tests/test_wikipedia.py` | 9 |
| `router_function` | `tests/test_router.py` | 7 |
| `dlq_node` | `tests/test_dlq.py` | 10 |
| `llm_node` | `tests/test_llm.py` | 8 |

### 8.2 Integration Tests

End-to-end tests invoke `run_workflow()` with both Wikipedia and DLQ endpoints mocked. These validate routing decisions, state transitions, and resilience.

| File | Tests |
|------|-------|
| `tests/test_integration.py` | 11 |

**Total: 45 tests** (exceeds the 20-test minimum).

### 8.3 Mocking

The `responses` library intercepts `requests` HTTP calls at the socket level. This guarantees:
- No real Wikipedia calls in CI
- No real DLQ webhook posts in CI
- Deterministic test results

---

## 9. CI/CD Pipeline

Three jobs run sequentially in GitHub Actions:

```
lint ──► test ──► docs
```

1. **lint** — `ruff` + `mypy` static analysis
2. **test** — `pytest` with coverage and JUnit XML artefacts
3. **docs** — Validate `docs/workflow.json` is well-formed

The pipeline fails on any test failure, preventing broken code from reaching `main`.
