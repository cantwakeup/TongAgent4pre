# Benchmark Difficulty Ladder v2

## Frozen and exploratory evidence

| Regime | Dataset and budget | Systems | Standard-normalized EM | Answer-rate observation | Interpretation |
| --- | --- | --- | --- | --- | --- |
| easier text/tool regime | GAIA Level-1 text-only, frozen formal 60k budget | Bare / Vanilla / Standard | 6/8, 5/8, 5/8 | formal 48-run subset | differentiating but largely solvable |
| middle exploratory regime | FRAMES source indices 0–7, 100k / 900s | Bare / Standard | 3/8, 3/8 | 5/8, 4/8 canonical | measurable, but completion remains budget-sensitive |
| hard stress regime | BrowseComp, frozen formal 60k budget | Bare / Vanilla / Standard | 0/8, 0/8, 0/8 | token exhaustion dominated | shared capability ceiling |

The GAIA and BrowseComp rows remain the immutable
`final-transparent-harness-v2-formal-20260722T162330Z` result. The FRAMES row
comes only from `frames-high-budget-completion-v1`, which is explicitly an
**exploratory high-budget benchmark**. It does not replace any formal row.

## Interpretation limits

The observed ordering is:

```text
GAIA: 5–6 correct of 8
FRAMES high-budget: 3 correct of 8
BrowseComp: 0 correct of 8
```

This supports a descriptive difficulty ladder, but not a controlled causal
comparison. FRAMES used a 100k-token, 900-second envelope and only two systems,
whereas the formal GAIA/BrowseComp run used 60k tokens and three systems. The
FRAMES tasks also include two previously attempted scaling-smoke IDs; that
overlap was preregistered and disclosed rather than hidden.

The earlier 60k/600s FRAMES smoke established that the strong-model path did
not complete stably under that envelope. The new high-budget run shows that
FRAMES becomes measurable with more headroom, but seven of sixteen jobs still
ended in budget exhaustion and the forced-finalization boundary recovered no
answers.

```text
BENCHMARK_DIFFICULTY_EFFECT = DESCRIPTIVE_ONLY
```

The supported claim is therefore narrow: under their frozen study envelopes,
GAIA yielded the highest accuracy, this FRAMES subset yielded intermediate
nonzero accuracy, and BrowseComp remained a zero-EM hard-regime ceiling.
