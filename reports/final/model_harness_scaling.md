# Final Model-Harness Scaling Study

## Status

The formal 60-run core matrix was **not started** because the preregistered
Smoke Gate did not produce usable strong-model telemetry. No Agent, Retrieval,
scoring, recovery, or fault-injection behavior was changed.

```text
FRAMES_SET_FROZEN = YES
SMOKE_TERMINAL_ARTIFACTS = 8/8
SMOKE_RUNNER_ERRORS = 0
STRONG_SMOKE_TIMEOUTS = 4/4
STRONG_SMOKE_KNOWN_TOKEN_RUNS = 0/4
SMOKE_GATE_PASSED = NO
FORMAL_CORE_RUNS_STARTED = NO
```

The frozen set is bound to commit:

```text
6d24599c0d1d21bbd8d5d63fdbadb4ec7b6b4b56
```

Selection used only source index/task ID, question text, and the original
question-level `reasoning_types`. Reference answers were attached after the
indices were immutable for use by the existing scorer. Gold articles,
`source_urls`, historical results, and manual difficulty judgments were not
available to the selection function.

The 12 source indices are:

```text
0, 1, 2, 3, 4, 5, 6, 7, 12, 25, 30, 39
```

They contain three tasks in each frozen category:

- relation/multi-constraint;
- numerical/temporal;
- table/enumeration/count;
- post-processing/mixed.

Thirty-two prior FRAMES task IDs were excluded. A second scan of 7,950 JSON or
JSONL artifacts found no selected task overlapping a prior task.

## Smoke experiments

The first attempted weak Smoke used CLI `--limit 2`, which applies after the
runner's seed shuffle and therefore selected `0030` first. It was interrupted
before any terminal result and is excluded from the gate. Its directory is
retained as:

```text
final-model-harness-scaling-smoke-weak-20260723T000000Z
```

The corrected smoke dataset is byte-for-byte equal to the first two lines of
the frozen FRAMES-12 dataset (`0000`, `0001`). It produced these immutable
experiments:

```text
final-model-harness-scaling-smoke-weak-v2-20260723T073000Z
final-model-harness-scaling-smoke-strong-20260723T041100Z
```

| Model | System | Task | Terminal status | Standard EM | Tokens | Wall |
| --- | --- | --- | --- | ---: | ---: | ---: |
| gpt-5.4-nano | Bare | 0000 | completed | 0 | 32,230 | 336.95s |
| gpt-5.4-nano | Bare | 0001 | budget_exhausted | 0 | 44,980 | 593.63s |
| gpt-5.4-nano | Standard | 0000 | completed | 0 | 32,111 | 383.20s |
| gpt-5.4-nano | Standard | 0001 | timed_out | 0 | unknown | 605.05s |
| gpt-5.5 | Bare | 0000 | timed_out | 0 | unknown | 605.05s |
| gpt-5.5 | Bare | 0001 | timed_out | 0 | unknown | 605.05s |
| gpt-5.5 | Standard | 0000 | timed_out | 0 | unknown | 605.05s |
| gpt-5.5 | Standard | 0001 | timed_out | 0 | unknown | 605.05s |

All eight corrected Smoke jobs produced a terminal `result.json` with scorer
fields, and none produced a runner error or a second attempt. The endpoint
listed both model IDs and accepted live calls. The strong jobs also wrote
in-progress checkpoint data, so this was not a startup failure. However, all
four were killed by the unchanged 600-second business watchdog before the
runner could publish provider usage.

The Smoke Gate exists to check endpoint, artifacts, scoring, telemetry, and
cost projection. Endpoint/artifact/scoring passed, but strong-model telemetry
did not. Launching the formal 36 strong jobs without any measurable strong
Smoke usage would make the `$15` monitored-cost requirement unauditable.
Accordingly, the core matrix was not launched.

## Frozen conclusions

```text
MODEL_CAPABILITY_EFFECT = NOT_ESTIMABLE_FROM_NEW_FRAMES_RUNS
HARNESS_ACCURACY_EFFECT = NOT_ESTIMABLE_FROM_NEW_FRAMES_RUNS
MODEL_HARNESS_INTERACTION = NOT_ESTIMABLE_FROM_NEW_FRAMES_RUNS
BENCHMARK_DIFFICULTY_EFFECT = PARTIAL_EXISTING_EVIDENCE_ONLY
OPERATIONAL_RECOVERY_EFFECT = CONFIRMED_BY_FROZEN_FAULT_INJECTION
```

No accuracy conclusion is inferred from Smoke timeouts.
