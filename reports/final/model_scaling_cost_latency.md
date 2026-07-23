# Model Scaling Cost and Latency

## Preregistered formal upper bound

The frozen per-run limits are 60,000 total tokens and 3,000 output tokens. The
cost-maximizing split is therefore 57,000 uncached input and 3,000 output.

| Matrix slice | Jobs | Per-run upper bound | Slice upper bound |
| --- | ---: | ---: | ---: |
| gpt-5.4-nano Bare + Standard | 24 | $0.01515 | $0.36360 |
| gpt-5.5 Bare + Standard + Vanilla | 36 | $0.37500 | $13.50000 |
| Core total | 60 | — | **$13.86360** |

The mathematical core upper bound passes the `$15` cap. The optional mid-model
matrix was not considered because the core did not start.

## Smoke cost

The four weak Smoke jobs have three known token records. Applying the frozen
weak-model prices to uncached input, cached input, and output yields a known
artifact-estimated total of `$0.01056473`. The fourth weak timeout has unknown
usage.

All four strong jobs timed out before terminal provider usage was published.
Their mathematical combined upper bound is `$1.50`; their exact monitored cost
is unknown. The full eight-job Smoke upper bound is `$1.56060`.

| Model | System | Known usage runs | Mean wall | Timeout |
| --- | --- | ---: | ---: | ---: |
| gpt-5.4-nano | Bare | 2/2 | 465.29s | 0/2 |
| gpt-5.4-nano | Standard | 1/2 | 494.12s | 1/2 |
| gpt-5.5 | Bare | 0/2 | 605.05s | 2/2 |
| gpt-5.5 | Standard | 0/2 | 605.05s | 2/2 |

The cost cap was not exceeded, but the Smoke requirement is broader than a
mathematical cap: it also requires usable telemetry for live monitoring. With
zero known strong-model usage records, the gate fails closed.
