# Cost and Latency

## Formal benchmark ceilings

The 48-run manifest uses 60,000 total tokens and 3,000 maximum output tokens per
run. At GPT-5.5 prices of `$5/M` uncached input, `$0.50/M` cached input, and
`$30/M` output, the exact cost-maximizing allocation is 57,000 uncached input,
zero cached input, and 3,000 output tokens.

```text
per-run upper bound: $0.375
48-run upper bound: $18.00
fault-injection API cost: $0.00
```

## Observed provider usage

| System | Usage-known runs | Total tokens | Mean tokens | Cached input | Artifact-estimated cost | Mean wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare Simple ReAct | 15/16 | 475,413 known | 31,694.2 known | 202,368 | $2.213684 known | 139.60s |
| Vanilla DeepAgents | 16/16 | 662,428 | 41,401.8 | 419,712 | $1.839861 | 84.64s |
| TongAgent Standard | 16/16 | 545,088 | 34,068.0 | 282,496 | $2.042708 | 96.88s |

Known artifact cost totals `$6.096253`. This is not presented as the exact full
experiment bill because the Bare runner-error artifact has no token usage. The
pre-registered mathematical upper bound remains `$18.00`, so the `$20` cap is
respected without imputing missing usage.

On runs with known telemetry, TongAgent Standard uses `1.075×` Bare's mean
tokens, within the 1.10 target. Its mean wall time is `0.694×` Bare's reported
all-run mean. Because the missing Bare usage prevents a canonical 16-vs-16
token mean, the overall performance non-inferiority flag remains fail-closed.
