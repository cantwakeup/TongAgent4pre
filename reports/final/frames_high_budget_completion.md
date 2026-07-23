# FRAMES High-Budget Completion Study

## Status and scope

```text
experiment: frames-high-budget-completion-v1
label: exploratory high-budget benchmark
implementation commit: bd00f32da77c4d8d9338fdf5e782cdaacfcb10cd
core results: 16/16 terminal artifacts
attempt-0002 directories: 0
known-cost ledger: $2.722600
conservative-cost ledger: $3.347600
cost cap: $6.00
```

This experiment is a separately labelled high-budget supplement. It does not
replace, extend, or reinterpret the frozen 48-run GAIA/BrowseComp benchmark.
The selected FRAMES tasks are source indices `0..7` from the frozen
`final_model_harness_frames12` set. Validation remained:

```text
historical leakage = 0
question hash match = 8/8
source index match = 8/8
prior scaling-smoke overlap = frames-test-0000, frames-test-0001
```

Bare Simple ReAct and TongAgent Standard used the same GPT-5.5 model, prompt
hash `73a4e7cc90c0ed0b2f5f743de28f35747647fc33c80e1bb6af706b1e5ada6eef`,
seed 17, tools, Retrieval Backend, budgets, scoring, and finalization boundary.
Their fairness fingerprint was
`sha256:3829950012a3d732a6a5c5d10c037808dbbd2281ce8b370646de5d01355f4a0a`.

## Aggregate outcomes

“Natural answer” means the policy produced a short answer without the shared
forced-finalization call. “Canonical answer after finalization” additionally
requires a valid terminal result artifact. This distinction matters for
Standard task `0005`, where the natural answer was saved but result validation
failed.

| System | Clean completed status | Natural answers | Canonical answers after finalization | Raw EM | Standard EM | Natural completion EM | Budget exhausted | Runner error | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare Simple ReAct | 3/8 | 5/8 | 5/8 | 3/8 | 3/8 | 3/8 | 5 | 0 | 0 |
| TongAgent Standard | 4/8 | 5/8 | 4/8 | 3/8 | 3/8 | 3/8 | 2 | 1 | 0 |

The systems tie on both raw and standard-normalized exact match. TongAgent
Standard has one fewer canonical answer because task `0005` ended in a
`RunResult` counter-invariant validation error after the natural answer had
already been saved. That natural answer was incorrect, so the error changed
answer rate but not EM.

## Per-task results

| Task | Bare status | Bare final answer | Bare EM | Standard status | Standard final answer | Standard EM | Finalization boundary |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| `0000` | budget exhausted | — | — | completed | `Jane Ballou` | ✓ | not triggered |
| `0001` | budget exhausted | — | — | budget exhausted | — | — | not triggered |
| `0002` | budget exhausted | — | — | failed / invalid output | — | — | Standard: total tokens; forced answer null |
| `0003` | budget exhausted | `France` | ✓ | completed | `France` | ✓ | not triggered |
| `0004` | completed | `Jens Kidman` | ✓ | completed | `Jens Kidman` | ✓ | not triggered |
| `0005` | completed | `506,000` | — | failed / runner error | canonical `ABSTAIN`; natural `506,000` | — | not triggered |
| `0006` | completed | `Dmitri Mendeleev` | — | completed | `Dmitri Mendeleev` | — | not triggered |
| `0007` | budget exhausted | `2` | ✓ | budget exhausted | — | — | not triggered |

## Shared finalization boundary

The boundary triggered in only `1/16` jobs: TongAgent Standard `0002` at the
total-token condition. It blocked a subsequent fetch and executed the shared
tool-free finalization call, but that call returned no valid short answer.
Consequently:

```text
forced final answers = 0/16
EM gained by forced finalization = 0
answers gained by forced finalization = 0
```

Seven budget-exhausted jobs stopped before the explicit 90k trigger. Their next
full-context model reservation could not fit within the remaining token budget,
so the existing boundary did not activate. This is retained as an experimental
finding; it was not patched or rerun.

## Paired accuracy

```text
both correct = 2
Bare only = 1
TongAgent Standard only = 1
both wrong = 4
Harness standard-EM delta = 3/8 - 3/8 = 0
```

The task-level differences are symmetric: Standard uniquely solved `0000`,
while Bare uniquely solved `0007`. Eight tasks are insufficient for a broad
statistical claim, and the observed tie provides no evidence of Harness
accuracy superiority.

## Interruption and Vanilla decision

The user-requested pause interrupted Bare `0005` after 15,658 reported tokens.
Its trace, telemetry, and `$0.092815` known cost were preserved under
`native/infrastructure_interruptions/user-pause-20260723`. The single allowed
infrastructure recovery reused the same `attempt-0001`; no `attempt-0002` was
created, and the interruption cost is included in the ledger.

Vanilla DeepAgents was not run. Although the core matrix completed and known
cost was below `$4.50`, eight additional worst-case runs require `$5.00`. The
remaining hard-cap headroom was at most `$3.047201` after reconciling available
telemetry, so the full optional extension could not be guaranteed to remain
below `$6.00`.

## Conclusions

```text
NATURAL_COMPLETION_PERFORMANCE = 5/8 answers and 3/8 EM for each system
HIGH_BUDGET_FINALIZED_PERFORMANCE = Bare 5/8 answers, Standard 4/8 answers, both 3/8 EM
HARNESS_ACCURACY_EFFECT = NONE_OBSERVED
HARNESS_RECOVERY_EFFECT = NOT_DEMONSTRATED_IN_THIS_STUDY
BENCHMARK_DIFFICULTY_EFFECT = DESCRIPTIVE_ONLY
```

High budget made this FRAMES subset measurable and produced nonzero accuracy,
but it did not create an accuracy separation between the transparent Harness
and its matched Bare policy. The previously frozen fault-injection result
remains the evidence for operational recovery superiority; this accuracy study
does not strengthen or weaken that separate claim.
