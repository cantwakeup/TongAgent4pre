# GPT-5.5 Scaling Smoke Timeout Diagnosis

## Scope and outcome

The formal scaling matrix remained paused throughout this diagnosis. No Agent
reasoning, prompt, Retrieval, scoring, budget, or stopping behavior changed.
Only runner/telemetry persistence changed in commit:

```text
4da895df0d36eff47518fbc6b23cad5454769113
```

The frozen four-job strong smoke was rerun exactly once as:

```text
final-model-harness-scaling-smoke-strong-telemetry-20260723T140500Z
```

## Configuration diff

The successful attribution reference is the canonical resolved config from
`constraint-guided-gpt55-attribution-canary-20260722T141647Z`. The failed
scaling reference is the canonical resolved config from
`final-model-harness-scaling-smoke-strong-20260723T041100Z`.

| Field | Attribution canary | Scaling smoke | Difference |
| --- | --- | --- | --- |
| Model ID | `gpt-5.5` | `gpt-5.5` | none |
| Provider | `openai` | `openai` | none |
| Base URL | same frozen OpenAI-compatible endpoint | same | none |
| Reasoning effort | not configured | not configured | none |
| Temperature | `1.0` | `1.0` | none |
| Max output tokens | `5000` | `3000` | reduced by 2000 |
| Total token ceiling | `100000` | `60000` | reduced by 40000 |
| Max model calls | `16` | `16` | none |
| Max turns | no independent field | no independent field | none |
| Recursion limit | `125` | `125` | none |
| Request timeout | not configured | not configured | none |
| Business watchdog | `600s` | `600s` | none |
| Job concurrency | sequential CLI, one worker | sequential CLI, one worker | none |
| Streaming | not configured | not configured | none |
| Search/fetch budget | `4/6` | `4/6` | none |
| Total external tools | `12` | `12` | none |
| Search/fetch backend | TongAgent shared web search/fetch | same | none |
| Results/page limits | `5` results, `12000` chars | same | none |
| Runtime/system | `long_react` / `simple_react` | `tongagent_standard` / Bare or Standard | intentional policy experiment difference |

The material resource differences were therefore the lower per-call output cap
and lower total token ceiling. Endpoint identity, tool configuration, external
budgets, watchdog, seed, and effective concurrency did not change.

## Original timeout artifact audit

No reference answer was used for this audit.

| System/task | Last auditable stage | Evidence |
| --- | --- | --- |
| Bare / `0000` | worker/request started; first model response unknown | only parent timeout artifacts; no worker trace/checkpoint/counters |
| Bare / `0001` | worker/request started; first model response unknown | only parent timeout artifacts; no worker trace/checkpoint/counters |
| Standard / `0000` | `search` | 3 searches and 3 fetches completed; the model then emitted a fourth search, with no completion checkpoint |
| Standard / `0001` | `fetch` | 3 searches completed; 2 of 3 requested fetches were present in pending writes when killed |

None of the four original artifacts proves that an answer was generated or
that finalization began. The two Bare artifacts also cannot prove that no model
response arrived; they can only fail closed at the last persisted stage.

## Single-task foreground diagnostic

The frozen first FRAMES task was run once with GPT-5.5, Bare Simple ReAct, and
concurrency 1. It completed as a legal token-budget terminal rather than a
startup or endpoint failure:

```text
experiment: final-model-harness-single-diagnostic-20260723T130000Z
status: budget_exhausted
wall: 551.39s
model calls: 3
search/fetch: 4/4
tokens: 28,325
runner error: 0
```

The first model response arrived about 22 seconds after request start. The
dominant latency was repeated public search/fetch work, including initial
searches lasting roughly three minutes. This identifies the failure as a long
research trajectory near the 600-second boundary, not an API startup outage.

## Timeout-safe telemetry change

The runner now:

- atomically persists sanitized trace events as they occur;
- records tool-start counters before invoking the provider-facing handler;
- writes `partial_telemetry.json` with request/model/tool milestones, budget
  snapshot, reported usage status, and a five-second heartbeat;
- snapshots that file to `watchdog_telemetry.json` before terminating a worker;
- restores observed search/fetch/token fields into the parent timeout result;
- keeps unknown provider usage as JSON `null` with
  `token_usage_status=usage_unavailable` rather than inventing zero usage.

Offline validation:

```text
ruff format/check: PASS
compile/import: PASS
socket-disabled suite: 409 passed, 78 subtests passed
git diff check: PASS
```

## Frozen strong smoke rerun

| System | Task | Terminal status | Search/fetch | Input/output/total | Wall | Last auditable stage |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Bare | `0000` | `budget_exhausted` | 4/6 | 50,398 / 1,122 / 51,520 | 539.97s | post-retrieval model reservation denied by token budget |
| Bare | `0001` | `timed_out` | 2/4 | 43,721 / 1,000 / 44,721 | 605.10s | third search started; watchdog snapshot persisted before termination |
| Standard | `0000` | `budget_exhausted` | 4/6 | 51,188 / 1,330 / 52,518 | 303.21s | post-retrieval model reservation denied by token budget |
| Standard | `0001` | `budget_exhausted` | 4/6 | 50,871 / 2,006 / 52,877 | 590.92s | post-retrieval model reservation denied by token budget |

All four jobs have request start, first model response, first tool call, last
progress timestamp, partial counters, and provider-reported token usage. There
were no runner errors and no `attempt-0002` directories.

Using the frozen rates (`$5/M` uncached input, `$0.50/M` cached input, `$30/M`
output), the artifact-computed costs are:

| System/task | Cost |
| --- | ---: |
| Bare / `0000` | `$0.153170` |
| Bare / `0001` | `$0.139165` |
| Standard / `0000` | `$0.195616` |
| Standard / `0001` | `$0.160167` |
| **Total** | **`$0.648118`** |

Reasoning tokens are reported as an output-token subset and are not charged a
second time.

## Gate

```text
STRONG_SMOKE_TELEMETRY = 4/4
STRONG_SMOKE_COMPLETED = 0/4
RUNNER_ERROR = 0
COST_AUDITABLE = YES
STRONG_SMOKE_GATE = FAIL
FORMAL_40_RUN_MATRIX_STARTED = NO
```

The telemetry defect is fixed, but the frozen GPT-5.5 policy/resource setup did
not complete at least three smoke jobs. The scaling study therefore stops here
without launching the deterministic FRAMES-8 matrix.
