# Benchmark Difficulty Ladder

## Frozen evidence

| Regime | Dataset | Systems/runs | Observable result | Interpretation |
| --- | --- | ---: | --- | --- |
| easier text/tool regime | GAIA Level-1 text-only | 8 tasks × 3 | Bare 6/8, Vanilla 5/8, Standard 5/8 standard EM | differentiating but largely solvable |
| intended middle regime | FRAMES-12 | core matrix not started | Smoke only; strong 4/4 timeout | formal difficulty estimate unavailable |
| hard stress regime | BrowseComp | 8 tasks × 3 | all three systems 0/8 | shared capability ceiling under frozen budget |

The GAIA and BrowseComp values come from the immutable
`final-transparent-harness-v2-formal-20260722T162330Z` experiment. BrowseComp
remains visible as a hard-regime capability ceiling and was not rerun with the
weak or intermediate model.

The answer-blind FRAMES-12 set was successfully frozen, but its model-scaling
matrix did not pass Smoke because every GPT-5.5 Smoke job reached the unchanged
600-second watchdog without terminal token telemetry. It is therefore invalid
to place a FRAMES accuracy point between GAIA and BrowseComp from this run.

```text
BENCHMARK_DIFFICULTY_EFFECT = PARTIAL
```

The supported statement is limited to: GAIA produced nonzero exact matches,
while BrowseComp produced none for any frozen system. FRAMES remains an
unmeasured middle-regime hypothesis in this study.
