#!/usr/bin/env bash
#
# Reproduce TongAgent's local quality gates. Dependency installation may use the
# configured package index; the test process itself is denied network sockets.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/learning/search-agent"
UV_BIN="${UV_BIN:-uv}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "error: uv was not found; install uv or set UV_BIN to its executable" >&2
  exit 127
fi

# Default to the canonical index while still allowing an explicit mirror:
#   UV_DEFAULT_INDEX=https://your-mirror.example/simple scripts/check_tongagent.sh
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"

# Empty values also prevent search_agent.py's setdefault-based .env loader from
# restoring local credentials. Keep this list explicit so CI cannot accidentally
# turn a unit test into a model, search-provider, or tracing request.
offline_environment=(
  SEARCH_AGENT_API_KEY
  SEARCH_AGENT_BASE_URL
  SEARCH_AGENT_MODEL
  SEARCH_AGENT_WORKER_MODEL
  OPENAI_API_KEY
  OPENAI_BASE_URL
  OPENAI_API_BASE
  OPENAI_MODEL
  ANTHROPIC_API_KEY
  GOOGLE_API_KEY
  GEMINI_API_KEY
  TAVILY_API_KEY
  SERPAPI_API_KEY
  BING_SEARCH_API_KEY
  LANGCHAIN_API_KEY
  LANGCHAIN_ENDPOINT
  LANGCHAIN_PROJECT
  LANGCHAIN_TRACING
  LANGCHAIN_TRACING_V2
  LANGSMITH_API_KEY
  LANGSMITH_ENDPOINT
  LANGSMITH_PROJECT
  LANGSMITH_TRACING
  OTEL_EXPORTER_OTLP_ENDPOINT
  OTEL_EXPORTER_OTLP_HEADERS
)
for variable in "${offline_environment[@]}"; do
  export "${variable}="
done

cd "${PROJECT_DIR}"

echo "==> Sync locked TongAgent test environment"
"${UV_BIN}" sync --locked --group test

uv_run=("${UV_BIN}" run --locked --no-sync --group test)
pytest_offline=(
  "${uv_run[@]}"
  python
  -m
  pytest
  --disable-socket
  --allow-unix-socket
)

echo "==> Check formatting"
"${uv_run[@]}" ruff format --check .

echo "==> Lint"
"${uv_run[@]}" ruff check .

echo "==> Compile and import"
"${uv_run[@]}" python -m compileall \
  -q \
  -x '(^|/)(\.git|\.venv|output)(/|$)' \
  .
"${uv_run[@]}" python -c \
  'import adaptive_control, agent_policy, evidence_graph, research_graph, research_state, search_agent, telemetry'

echo "==> Run benchmark-metric regression tests"
"${pytest_offline[@]}" -q \
  tests/test_metric_semantics.py \
  tests/test_search_quality.py \
  tests/test_evidence_graph.py \
  tests/test_adaptive_control.py \
  tests/test_stage03d_safety.py

echo "==> Run complete offline test suite"
"${pytest_offline[@]}" -q tests

echo "==> TongAgent checks passed"
