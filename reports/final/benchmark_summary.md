# Final Harness Benchmark Summary

## Experiment

```text
experiment: final-transparent-harness-v2-formal-20260722T162330Z
commit: 7a94dd2a583fc9770586a5de5fbad1e37a2639c1
dataset: 8 GAIA Level-1 text-only + 8 BrowseComp
systems: bare_simple_react, vanilla_deepagents, tongagent_standard
results: 48/48 terminal artifacts
reruns: 0
attempt-0002 directories: 0
```

## Aggregate results

| System | Answer rate | Raw EM | Standard normalized EM | Completed | Budget exhausted | Runner error | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare Simple ReAct | 7/16 | 4/16 | 6/16 | 7 | 8 | 1 | 0 |
| Vanilla DeepAgents | 6/16 | 2/16 | 5/16 | 6 | 10 | 0 | 0 |
| TongAgent Standard | 7/16 | 3/16 | 5/16 | 7 | 9 | 0 | 0 |

TongAgent Standard is one standard-normalized exact match behind Bare Simple
ReAct and tied with Vanilla DeepAgents. It matches Bare's answer rate and has no
runner error. It does **not** establish accuracy superiority.

All three systems scored zero exact matches on the eight BrowseComp tasks.
Their exact matches came from the GAIA subset. BrowseComp was dominated by
token exhaustion; Vanilla produced one BrowseComp answer, but it did not match
under the frozen scorer.

## Per-task standard-normalized correctness

| Task | Bare | Vanilla | TongAgent Standard |
| --- | ---: | ---: | ---: |
| `0383a3ee…` | ✓ | ✓ | ✓ |
| `11af4e1a…` | — | — | — |
| `27d5d136…` | ✓ | ✓ | ✓ |
| `2d83110e…` | ✓ | ✓ | ✓ |
| `305ac316…` | ✓ | — | ✓ |
| `3cef3a44…` | ✓ | ✓ | — |
| `42576abe…` | ✓ | ✓ | ✓ |
| `46719c30…` | — | — | — |
| eight BrowseComp tasks | 0/8 | 0/8 | 0/8 |

The TongAgent miss on `3cef3a44…` was an answer-selection error: the response
omitted `fresh basil`. This is not evidence that the transparent Harness
rewrote the answer; the persisted raw and final answers are identical.

## Non-inferiority decision

| Criterion | Observation | Result |
| --- | --- | ---: |
| Standard EM at least Bare minus one | `5 >= 6 - 1` | pass |
| Answer rate at least Bare minus one | `7 >= 7 - 1` | pass |
| TongAgent runner errors | `0` | pass |
| TongAgent timeouts | `0` | pass |
| Mean wall time no more than 1.25× Bare | `96.88s / 139.60s = 0.694×` | pass |
| Mean tokens no more than 1.10× Bare | `34,068 / 31,694.2 = 1.075×` on known-usage runs | observed pass |

The token comparison has an important integrity caveat: one Bare runner-error
artifact has no token telemetry, so Bare has 15/16 known token totals while
TongAgent has 16/16. The repository aggregate correctly reports Bare total
tokens as null. The observed-known ratio passes, but the full 16-run token
criterion cannot be proved from canonical artifacts. The final binary
performance decision therefore fails closed rather than imputing the missing
usage.

```text
ACCURACY_SUPERIORITY = NO
PERFORMANCE_NON_INFERIORITY = NO
```
