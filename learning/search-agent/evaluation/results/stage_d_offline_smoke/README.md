# Tracked Stage D evidence

> Deterministic offline fixture smoke test; not a formal benchmark result.

This directory is a sanitized, repository-sized export of the canonical local
Stage D run. `summary.json`, `summary.csv`, and `summary.md` preserve the full
18-result matrix and aggregate evidence. `representative_run.json` preserves
one complete `RunResult` plus its task, resolved config, canonical trace,
metrics, failure, and answer companions in a single reviewable envelope.

Machine-specific absolute artifact paths were replaced with one stable
repository-relative example path. No credentials, provider cache, downloaded
web pages, SQLite database, or worker environment are included.
