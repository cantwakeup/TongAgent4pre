# Final Transparent Harness Preflight v2

## Decision

The revised offline gate passes. It separates policy parity between Bare Simple
ReAct and TongAgent Standard from resource parity across all three evaluated
systems. The 16-task formal subset is a deterministic source-index prefix of
the previously frozen 20-task candidate pool, and its exact mathematical
GPT-5.5 cost upper bound is `$18.00`.

No Agent reasoning mechanism, prompt, retrieval implementation, or scoring
rule was changed. No model endpoint was called during this preflight.

## Repository snapshot

| Field | Value |
| --- | --- |
| Branch | `exp/final-transparent-harness` |
| Preflight baseline HEAD | `e70d12fee382a174dc85f528f067dede502dd1f9` |
| Baseline latest commit | `e70d12f chore(evals): complete final harness preflight` |
| Harness implementation commit | `73c05c944436012a9966e95d5c35d9048799984a` |
| Execution/fault-injection code | committed |
| Agent behavior changes in v2 | none |

The commit containing this report adds only manifests, a deterministic JSONL
subset, a shared live configuration, and preflight documentation.

## Candidate pool and deterministic selection

The original 20-task manifest remains in place and is now explicitly marked:

```text
manifest_type = full_candidate_manifest_v1
candidate_role = frozen_pool_for_final_harness_benchmark_v2
```

Candidate manifest:

```text
learning/search-agent/evaluation/manifests/final_harness_benchmark_v1.json
SHA-256: 93e5f796e4a66265ee68dd488ddb9d0cca456960b99bbc3891b5c6d57b015e79
```

The v2 rule is applied independently within each dataset:

```text
sort the already frozen candidates by source_index ascending
take the first 8
do not inspect answers, historical runs, or estimated difficulty
```

Selected source indices:

| Dataset | Frozen source indices |
| --- | --- |
| GAIA | `16, 29, 32, 41, 75, 83, 86, 91` |
| BrowseComp | `195, 257, 429, 515, 537, 560, 969, 1139` |

The excluded suffix is GAIA `103, 105` and BrowseComp `1148, 1225`. The rule
was evaluated from the answer-free task metadata in the frozen manifest. No
reference answer was read to select, replace, or order a task.

## Manifest and JSONL integrity

Formal artifacts:

```text
learning/search-agent/evaluation/manifests/final_harness_benchmark_v2.json
learning/search-agent/evaluation/datasets/final_harness_benchmark_v2.jsonl
learning/search-agent/evaluation/configs/final_harness_gpt55_seed17_v2.live.json
```

| Artifact | SHA-256 |
| --- | --- |
| v2 manifest | `ec9617dbc2c01d9c5d60b44af2fa00045069fccc3c52d61f83ac712ab9a86162` |
| v2 JSONL | `0e5dbf0dcf04b43a9dee035af53bce17be65f1742b0a792d22a50b77786c9d9e` |
| v2 live config | `1fbc55241358ed196af7eaca4e3c8512c59728b722bfa803507f162c08be538a` |

Validation results:

| Check | Result |
| --- | ---: |
| Total tasks | 16 |
| GAIA / BrowseComp | 8 / 8 |
| Unique task IDs | 16 |
| Manifest task-ID set match | 16/16 |
| Source-index match | 16/16 |
| Question SHA-256 match | 16/16 |
| Parent candidate manifest hash | match |
| JSONL schema/parser | pass |

The runtime passes only `task.question` to each Agent. Reference answers remain
available solely to the common post-run scorer.

## Historical task exclusion

The selected IDs were compared with all other evaluation JSONL datasets and
persisted historical task/result/failure artifacts. The frozen 20-task
candidate pool and its v2 derivative were excluded from the historical set
because neither has been executed.

| Historical source | Unique IDs inspected |
| --- | ---: |
| Other evaluation JSONL files | 29 |
| Persisted evaluation outputs | 33 |
| v2 overlap | 0 |

```text
HISTORICAL_TASK_LEAKAGE = NO
```

## Two-layer fairness audit

### Policy parity

Policy parity applies only to:

```text
bare_simple_react
tongagent_standard
```

Both use `build_transparent_react_graph`, the same explicit system prompt, the
same model and shared model parameters, the same model-facing `web_search` and
`fetch_url` tools, the same ReAct decision loop, and the same external and
token budgets. Existing offline parity tests verify identical fixture tool
actions and raw/final answers. TongAgent Standard adds checkpoint, retry,
structured trace, telemetry, and post-hoc audit services without rewriting the
model answer.

| System | Explicit prompt SHA-256 |
| --- | --- |
| `bare_simple_react` | `73a4e7cc90c0ed0b2f5f743de28f35747647fc33c80e1bb6af706b1e5ada6eef` |
| `tongagent_standard` | `73a4e7cc90c0ed0b2f5f743de28f35747647fc33c80e1bb6af706b1e5ada6eef` |

```text
BARE_STANDARD_PROMPT_PARITY = YES
```

### Resource parity

Resource parity applies across:

```text
bare_simple_react
vanilla_deepagents
tongagent_standard
```

All 48 prospective jobs resolve to one resource fairness fingerprint:

```text
sha256:516c790a52a97ee393c990af08ec8229658469529e7264dad2fa77feeb487233
```

Shared resources:

| Field | Value |
| --- | --- |
| Model | `gpt-5.5` |
| Temperature | `1.0` |
| Seed | `17` |
| Dataset | v2 JSONL digest above |
| Retrieval | unified TongAgent live search/fetch backend |
| Search / fetch calls | `4 / 6` |
| Total external/model-facing tools | `12` |
| Model calls | `16` |
| Total token ceiling | `60000` |
| Maximum output tokens | `3000` |
| Watchdog | `600s` |
| Input policy | question only |
| Scoring | raw whole-string EM, standard normalized EM, answer rate |

```text
RESOURCE_PARITY_ACROSS_SYSTEMS = YES
```

### Vanilla native prompt disclosure

Vanilla DeepAgents retains its native planning/delegation prompt because this
prompt is part of that baseline's implementation. It is disclosed but is not a
resource mismatch:

| System | Explicit prompt SHA-256 |
| --- | --- |
| `vanilla_deepagents` | `8bf254b3638c8fdf41adbf65a7477aadb94060414bdc55798115c7198c79ab03` |

```text
VANILLA_NATIVE_PROMPT_DISCLOSED = YES
FAIRNESS_CHECK_PASS = YES
```

## Exact mathematical cost bound

Rates used:

```text
uncached input: $5.00 / 1M tokens
cached input:   $0.50 / 1M tokens
output:         $30.00 / 1M tokens
```

For each run, let `I`, `C`, and `O` be uncached input, cached input, and output
tokens. The shared ceilings are:

```text
I + C + O <= 60,000
O <= 3,000
I, C, O >= 0
```

Because output has the highest price and uncached input is more expensive than
cached input, the cost-maximizing allocation is:

```text
O = 3,000
I = 57,000
C = 0
```

Exact cost:

```text
per run = 57,000 × $5/1M + 0 × $0.50/1M + 3,000 × $30/1M
        = $0.375

48 runs = 48 × $0.375 = $18.00
fault-injection API cost = $0.00
combined mathematical upper bound = $18.00
```

The `$20` cap passes without reducing the formal set to 7+7 and without giving
any system a different budget.

```text
COST_CAP_PASS = YES
```

## Dry-run

Dry-run was executed with `OPENAI_API_KEY` explicitly removed from the process
environment, so no model call could occur:

```text
16 tasks × 3 systems = 48 jobs
dry_run = 48
executed = 0
skipped = 0
```

## Offline verification

```text
ruff format --check: 85 files already formatted
ruff check: passed
compileall: passed
socket-disabled pytest: 405 passed, 78 subtests passed
git diff --check: passed
```

## Final gate

```text
WORKTREE_CLEAN = YES
FULL_TESTS_PASS = YES
MANIFEST_JSONL_MATCH = YES
HISTORICAL_TASK_LEAKAGE = NO
BARE_STANDARD_PROMPT_PARITY = YES
RESOURCE_PARITY_ACROSS_SYSTEMS = YES
VANILLA_NATIVE_PROMPT_DISCLOSED = YES
FAIRNESS_CHECK_PASS = YES
DRY_RUN_JOBS = 48
COST_CAP_PASS = YES
FINAL_PREFLIGHT_GATE = YES
```

The formal online benchmark is authorized exactly once from the commit
containing this report. During the run, code, task selection, configuration,
and scoring are frozen; algorithm failures are retained and are not rerun.
