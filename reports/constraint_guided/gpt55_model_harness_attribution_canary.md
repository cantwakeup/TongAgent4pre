# GPT-5.5 Model-Harness Attribution Canary

## Decision

The canary supports a **model effect**, not a TongAgent Harness advantage. GPT-5.5 lifts Structured Context, Simple ReAct, and Vanilla DeepAgents to the same 2/4 exact-match set. Structured Context gains one EM over Long-ReAct v1, but does not beat either baseline.

| System | EM | Answers | Exhaustion | Tokens | Search/Fetch | Wall | Conservative cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long-ReAct v1 | 1/4 | 3/4 | 0 | 156,772 | 6/10 | 383.2s | $0.92761 |
| Simple ReAct | 2/4 | 4/4 | 0 | 105,386 | 6/10 | 288.3s | $0.627455 |
| Vanilla DeepAgents | 2/4 | 4/4 | 2 | 229,020 | 12/15 | 337.4s | $1.248175 |
| Structured Context (frozen prior run) | 2/4 | 3/4 | 0 | 82,330 | 13/11 | 747.9s | $0.56605 |

All 12 new task/system pairs produced terminal artifacts. There were no timeouts or runner errors. Vanilla DeepAgents exhausted the search budget on 0069 and the token budget on 0637; those are retained as formal algorithm outcomes.

## Per-task attribution

| Task | Long-ReAct v1 | Simple ReAct | Vanilla DeepAgents | Structured Context | EM attribution |
|---|---|---|---|---|---|
| 0069 | `6` | `7 years` | `7 years (about...)` | `7 years` | No EM. Simple and Structured found the correct semantic candidate, but the frozen conservative metric retains the period in gold `7 years.` |
| 0612 | ABSTAIN | Selayar | Selayar | ABSTAIN | Shared relation/selection failure; gold is Makassarese. |
| 0615 | `Summer Magic` | `Summer Magic` | `Summer Magic` | `Summer Magic` | Model effect: all four GPT-5.5 systems match. |
| 0637 | `143` | `143 years` | `143 years` | `143 years` | Structured Context repairs v1's missing unit, but both baselines do too. |

The correctness sets are not complementary: Long-ReAct v1 matches only 0615, while both baselines and Structured Context match the same pair, 0615 and 0637. There is therefore no baseline-only correct trajectory on 0069 or 0612 to transplant into TongAgent from this canary.

## Interpretation

This result rejects both the strong-positive and medium-positive scenarios. It falls into **model effect / still not above baseline**:

- Structured Context versus Long-ReAct v1: 2/4 versus 1/4.
- Structured Context versus best baseline: 2/4 versus 2/4.
- Structured Context's incremental EM is a formatting/unit repair on 0637, not an exclusive multi-hop retrieval success.
- Simple ReAct has the strongest current Pareto profile: the same 2/4 EM, 4/4 answer rate, no exhaustion, lower wall time, fewer searches, and substantially fewer tokens than Vanilla DeepAgents or Long-ReAct v1. Structured Context uses fewer tokens but has lower answer rate, more searches, and much higher wall time.

The monitored usage across the Structured Context GPT-5.5 canary and this 12-run attribution canary was 553,438 input tokens and 20,070 output tokens across 84 requests. Charging all input at the uncached GPT-5.5 rate yields a conservative **$3.36929**, leaving **$26.63071** of the $30 cap. This is a snapshot-relative estimate because the endpoint does not expose daily or per-model spend.

```text
MODEL_EFFECT_CONFIRMED = YES
STRUCTURED_CONTEXT_BEATS_V1 = YES
TONGAGENT_BEATS_BASELINE = NO
DEV12_TONGAGENT_MECHANISM_EXPANSION_SUPPORTED = NO
```
