# FRAMES High-Budget Cost and Latency

## Cost ledger

GPT-5.5 pricing used by the frozen launcher was `$5/M` uncached input,
`$0.50/M` cached input, and `$30/M` output. The per-run mathematical worst case
was `$0.625`, and the hard study cap was `$6.00`.

| Ledger | Amount | Meaning |
| --- | ---: | --- |
| Launcher known-cost total | `$2.722600` | terminal `RunResult` usage plus the preserved Bare interruption |
| Standard `0005` supplemental reported telemetry | `$0.230199` | usage was reported in partial telemetry but omitted from its failed terminal result |
| Trace-reconciled observed cost | `$2.952799` | known launcher cost plus supplemental telemetry |
| Frozen conservative launcher total | `$3.347600` | charges missing terminal usage at the `$0.625` worst case |
| Hard cap | `$6.000000` | never reached |

The conservative run-state ledger was not rewritten after completion. The
supplemental number is reported separately so that the canonical failure
remains visible and no usage is invented.

## Usage and latency

| System | Canonical usage-known | Canonical tokens | Trace-reconciled tokens | Mean reconciled tokens/task | Mean canonical wall | Interruption-adjusted wall | Actual search/fetch | Reconciled observed cost | Conservative cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare Simple ReAct | 8/8 | 434,288 | 449,946 | 56,243.2 | 154.44s | 162.63s/task | 27 / 39 | `$1.400099` | `$1.400099` |
| TongAgent Standard | 7/8 | 416,130 known | 485,972 | 60,746.5 | 156.59s | 156.59s/task | 23 / 35 | `$1.552700` | `$1.947501` |

Bare's reconciled totals include the user-paused partial trajectory: 15,658
tokens, 3 searches, 4 fetches, 65.53 seconds, and `$0.092815`. Standard `0005`
has no terminal usage object because result validation failed, but its flushed
telemetry reports 69,842 tokens, 4 budget-accounted searches, 6 fetches, and
`token_usage_status=reported`. Those values are included only in the explicitly
labelled trace-reconciled columns.

Across the full sequential core, interruption-adjusted worker time was about
2,553.79 seconds (42.56 minutes), excluding the period while execution was
paused. The reconciled mean-token ratio was:

```text
TongAgent Standard / Bare = 60,746.5 / 56,243.2 = 1.080×
```

The corresponding mean-wall ratio was approximately `0.963×`. These are
descriptive resource comparisons, not an accuracy result.

## Why Vanilla was skipped

Eight Vanilla jobs have a frozen mathematical worst cost of:

```text
8 × $0.625 = $5.00
```

After the core, trace-reconciled cost-cap headroom was `$3.047201`; conservative
headroom was `$2.652400`. Neither can guarantee completion of all eight Vanilla
jobs under the `$6` cap, so the optional extension was not launched.
