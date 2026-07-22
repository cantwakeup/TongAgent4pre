# Structured Context Canary Decision

## Outcome

The preregistered same-model Structured Context canary failed. It produced no exact-match gain on either mechanism task, exhausted the search budget once, and exceeded the frozen search/fetch ceilings. Full dev12 expansion is not allowed.

| Variant | Model | EM | Mechanism EM gain | Legal | Exhaustion | Search/Fetch | Mean tokens | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Frozen v1 | `gpt-5.4-nano` | — | 0 | 4/4 | 0 | 9/9 | 50,583.5 | Baseline |
| Structured Context | `gpt-5.4-nano` | 0/4 | 0 | 4/4 artifacts | 1 | 12/11 | 27,782.5 | Fail |
| Model-capacity follow-up | `gpt-5.5` | 2/4 | 1 | 4/4 | 0 | 13/11 | 20,582.5 | Diagnostic only |

The `gpt-5.5` follow-up confirms that model capacity materially affects answer selection: 0615 changed from a malformed non-match to `Summer Magic`, and 0637 changed from `168 years` to the matching `143 years`. It does not validate Structured Context because the model changed and the 9/9 external-tool ceilings were exceeded.

The endpoint usage delta for the strong-model run was 76,154 input tokens and 6,176 output tokens across 26 requests. Charging all input at the uncached `gpt-5.5` rate gives a conservative estimate of **$0.56605**, well below the $30 session cap. The endpoint exposes cumulative tokens but no daily or per-model cost breakdown, so this estimate covers usage since the recorded pre-run snapshot rather than all activity earlier in the calendar day.

```text
STRUCTURED_CONTEXT_CANARY_PASSED = NO
MECHANISM_TASK_EM_GAIN = 0
DEV12_EXPANSION_ALLOWED = NO
```
