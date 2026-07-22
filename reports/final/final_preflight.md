# Final Transparent Harness Preflight

## Decision

The offline preflight is complete, but the formal online benchmark is **not
authorized**. Dataset materialization, historical-leakage checks, the unified
configuration fingerprint, and the 60-job dry-run all pass. Two strict gate
conditions do not:

1. `vanilla_deepagents` does not use the same explicit system prompt as
   `bare_simple_react` and `tongagent_standard`.
2. The manifest's `$18.80` cost figure is a conservative projection, not a hard
   upper bound. At the configured per-run token ceilings, the mathematical
   GPT-5.5 upper bound is `$37.50` for the 60 online jobs.

No model endpoint was called and no benchmark attempt directory was created by
this preflight.

## Repository snapshot

| Field | Value |
| --- | --- |
| Branch | `exp/final-transparent-harness` |
| Baseline HEAD | `778d5f99b40b17237556d7fdae46b6ec5d13d2fc` |
| Baseline latest commit | `778d5f9 chore(evals): freeze final harness benchmark tasks` |
| Harness implementation commit | `73c05c944436012a9966e95d5c35d9048799984a` |
| Baseline worktree | clean |

Commit `73c05c9` contains the execution, transparent harness, scoring, and
fault-injection implementation and tests. Commit `778d5f9` adds only the frozen
benchmark manifest. The preflight dataset, live configuration, and this report
are the only files added after that frozen selection.

## Offline validation

The canonical commands were run from `learning/search-agent`:

```text
uv run --group test ruff format --check .
uv run --group test ruff check .
uv run --group test python -m compileall -q evaluation tests scripts *.py
PYTHONPATH=. uv run --group test python -m pytest --disable-socket -q
git diff --check
```

Results:

```text
ruff format: 85 files already formatted
ruff check: All checks passed
compileall: passed
pytest: 405 passed, 78 subtests passed
git diff --check: passed
```

An initial `uv run pytest` invocation omitted `PYTHONPATH=.` and therefore
failed during collection because repository-local modules were not importable.
It executed no tests and is not treated as a product failure. The complete
suite was immediately rerun with the repository's canonical import path and
passed as shown above.

## Frozen dataset materialization

The frozen manifest is:

```text
learning/search-agent/evaluation/manifests/final_harness_benchmark_v1.json
```

The materialized dataset is:

```text
learning/search-agent/evaluation/datasets/final_harness_benchmark_v1.jsonl
```

Dataset properties:

| Property | Result |
| --- | ---: |
| GAIA Level-1 text-only tasks | 10 |
| BrowseComp tasks | 10 |
| Total tasks | 20 |
| Unique task IDs | 20 |
| Manifest task-ID set match | exact |
| Manifest source-index match | 20/20 |
| Manifest question SHA-256 match | 20/20 |
| Dataset parser validation | passed |
| Dataset SHA-256 | `beafdf7d56b214d114910038a096f96d885da0b54df75a7e2ea969c0904a7a05` |

Source revisions recorded in the JSONL are:

```text
sayan1101/gaia_filtered_text_only: 8b387ca6207e63345b380f5741b3c91103d2ac85
smolagents/browse_comp: f975f60afc3c811de82951165f5a9a921d024e14
```

BrowseComp questions and post-run scoring answers were decoded with the
official `openai/simple-evals` SHA-256-derived XOR procedure. The Agent runtime
receives only `task.question`; `reference_answer` is read by common scoring
after the system returns.

## Selection integrity and historical leakage

Selection was frozen in commit `778d5f9` before the scoring answers were
materialized. That commit contains only the manifest. Its selection rules are:

- GAIA: filter to Level 1 with no file input, sort by `task_id`, take the first
  10. Selection fields exclude `Final answer`.
- BrowseComp: decrypt the problem only, hash the problem with SHA-256, sort by
  that hash, take the first 10. The answer field was not used for selection.

After materialization, the frozen IDs were compared with all other repository
evaluation JSONL files and persisted historical result/failure/task artifacts:

| Historical source | Unique IDs inspected |
| --- | ---: |
| Other evaluation JSONL datasets | 29 |
| Persisted evaluation outputs | 33 |
| Overlap with the frozen 20 | 0 |

Reference answers were read only after selection to populate the scoring
field. They were not used to select, replace, rank, or remove tasks.

## Live configuration and fairness audit

The secret-free live configuration is:

```text
learning/search-agent/evaluation/configs/final_harness_gpt55_seed17.live.json
```

It names `OPENAI_API_KEY` as the credential environment variable but contains
no credential value. The resolved shared configuration is:

| Field | Shared value |
| --- | --- |
| Model | `gpt-5.5` |
| Temperature | `1.0` |
| Seed | `17` |
| Runtime mode | `tongagent_standard` |
| Retrieval backend | unified TongAgent search/fetch backend |
| Search / fetch | `4 / 6` |
| Total tools / model calls | `12 / 16` |
| Total tokens / max output | `100000 / 5000` |
| Watchdog | `600s` |
| Task input | question only |
| Scoring | raw whole-string EM, standard normalized EM, answer rate |

All 60 prospective jobs resolve to one fairness fingerprint:

```text
sha256:168a65ea19973b86da83eb4ec7adc3467f38ed1916eb5d6e357ae7984d17c16d
```

Model, seed, retrieval tools, external budgets, watchdog, task input policy,
and common scoring are identical. The prompt audit is not fully equal:

| Prompt | SHA-256 |
| --- | --- |
| Bare Simple ReAct / TongAgent Standard | `73a4e7cc90c0ed0b2f5f743de28f35747647fc33c80e1bb6af706b1e5ada6eef` |
| Vanilla DeepAgents | `8bf254b3638c8fdf41adbf65a7477aadb94060414bdc55798115c7198c79ab03` |

Bare Simple ReAct and TongAgent Standard use the same graph builder and exact
prompt. Vanilla DeepAgents retains its native planning/delegation-oriented
system prompt. This is an intrinsic policy difference in the mature harness,
but it does not satisfy the strict requirement that all three explicit prompts
be identical. The preflight does not change it because changing the prompt
would change Agent reasoning behavior.

## Dry-run schedule

The dry-run was executed with `OPENAI_API_KEY` explicitly removed from the
process environment. It therefore validated construction and scheduling
without any possibility of calling the model endpoint.

```text
20 tasks × 3 systems = 60 jobs
dry_run = 60
executed = 0
skipped = 0
```

Systems:

```text
bare_simple_react
vanilla_deepagents
tongagent_standard
```

No experiment lock, attempt, result, or failure artifact was created because
the execution layer resolves all prospective configurations before dry-run and
does not launch workers in dry-run mode.

## Conservative cost audit

The manifest records `$18.80` as a projection based on observed GPT-5.5
attribution usage with a 1.5× margin. Controlled fault injection uses only the
local fixture model and fixture retrieval backend, so its API cost is `$0.00`.

Using the configured hard ceilings and the supplied GPT-5.5 prices of `$5/M`
input tokens and `$30/M` output tokens, the worst permitted mix per run is
95,000 input plus 5,000 output tokens:

```text
per-run upper bound = 95,000 × $5/M + 5,000 × $30/M = $0.625
60-run upper bound = 60 × $0.625 = $37.50
fault-injection API upper bound = $0.00
combined mathematical upper bound = $37.50
```

Therefore `$18.80` is a conservative forecast but not an enforceable upper
bound. No experiment-level dollar limiter currently proves that all 60 jobs
can complete below `$20`. The cost gate fails under the requested hard-cap
interpretation.

## Final gate

```text
WORKTREE_CLEAN = YES
FULL_TESTS_PASS = YES
MANIFEST_JSONL_MATCH = YES
HISTORICAL_TASK_LEAKAGE = NO
FAIRNESS_CHECK_PASS = NO
DRY_RUN_JOBS = 60
COST_CAP_PASS = NO
```

Formal benchmark authorization:

```text
FINAL_PREFLIGHT_GATE = NO
FORMAL_ONLINE_BENCHMARK_ALLOWED = NO
```

The formal benchmark must not start until the prompt-equivalence requirement
is clarified or changed and the `$20` hard cap is made compatible with the
60-job ceilings. No Agent reasoning mechanism was modified during this
preflight.
