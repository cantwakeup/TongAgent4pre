# Stage D deterministic offline fixture smoke

> Deterministic offline fixture smoke test; not a formal benchmark result.

- Experiment: `stage-d-offline-smoke-final-20260719-a8`
- Source Git SHA: `6d51f71715fca01fdb60ac749395953dec2da267`
- Dataset digest: `sha256:b0c8a12630cb71416009128d2f6bd1703a5418ef7926e1807ac73dc71a245487`
- Fixture revision: `6f09f8f239d324b03ff8f948e8dfb29dfeacdf3e24c20d146514a06b742b99e9`
- Fairness fingerprint: `sha256:d0609f6df2bfa994785e85f525398268037f3d2c1f6a7537685f48c499cec4b8`
- Fresh execution: 18 executed, 0 skipped
- Resume execution: 0 executed, 18 skipped
- Result hashes after resume: unchanged across all 18 terminal records
- Targeted rerun: 1 new `simple_react/fixture-fetch-retry` attempt
- Strict validation: passed

## System comparison

| System | Runs | Completed | Partial | Budget exhausted | Tool calls | Mean tools | Wall time (s) | Mean wall time (s) | Failure distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| simple_react | 6 | 5 | 0 | 1 | 22 | 3.66667 | 4.07723 | 0.679538 | budget_exhausted: 1 |
| vanilla_deepagents | 6 | 5 | 0 | 1 | 22 | 3.66667 | 4.45898 | 0.743163 | budget_exhausted: 1 |
| tongagent | 6 | 4 | 1 | 1 | 41 | 6.83333 | 10.7148 | 1.78579 | budget_exhausted: 1 |

## Selected task results

| System | Task | Status | Failure | Search | Fetch | Relevant | Tools | Wall time (s) | Evidence | Structural coverage |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| simple_react | fixture-budget-resume | budget_exhausted | budget_exhausted | 4 | 0 | 4 | 5 | 0.678865 | N/A | N/A |
| simple_react | fixture-conflict | completed | — | 2 | 3 | 2 | 5 | 0.683590 | N/A | N/A |
| simple_react | fixture-fetch-retry | completed | — | 1 | 3 | 1 | 4 | 0.682232 | N/A | N/A |
| simple_react | fixture-nonempty-irrelevant | completed | — | 1 | 0 | 0 | 1 | 0.662138 | N/A | N/A |
| simple_react | fixture-one-hop | completed | — | 1 | 2 | 1 | 3 | 0.682937 | N/A | N/A |
| simple_react | fixture-two-source | completed | — | 2 | 2 | 2 | 4 | 0.687468 | N/A | N/A |
| vanilla_deepagents | fixture-budget-resume | budget_exhausted | budget_exhausted | 4 | 0 | 4 | 5 | 0.753278 | N/A | N/A |
| vanilla_deepagents | fixture-conflict | completed | — | 2 | 3 | 2 | 5 | 0.747697 | N/A | N/A |
| vanilla_deepagents | fixture-fetch-retry | completed | — | 1 | 3 | 1 | 4 | 0.746753 | N/A | N/A |
| vanilla_deepagents | fixture-nonempty-irrelevant | completed | — | 1 | 0 | 0 | 1 | 0.729406 | N/A | N/A |
| vanilla_deepagents | fixture-one-hop | completed | — | 1 | 2 | 1 | 3 | 0.742794 | N/A | N/A |
| vanilla_deepagents | fixture-two-source | completed | — | 2 | 2 | 2 | 4 | 0.739050 | N/A | N/A |
| tongagent | fixture-budget-resume | budget_exhausted | budget_exhausted | 4 | 0 | 1 | 6 | 1.863594 | 0 | 0.0 |
| tongagent | fixture-conflict | completed | — | 2 | 3 | 2 | 10 | 2.459304 | 3 | 1.0 |
| tongagent | fixture-fetch-retry | completed | — | 1 | 3 | 1 | 8 | 1.693963 | 2 | 1.0 |
| tongagent | fixture-nonempty-irrelevant | partial | — | 1 | 0 | 0 | 2 | 1.245436 | 0 | 0.0 |
| tongagent | fixture-one-hop | completed | — | 1 | 2 | 1 | 7 | 1.788324 | 2 | 1.0 |
| tongagent | fixture-two-source | completed | — | 2 | 2 | 2 | 8 | 1.664137 | 2 | 1.0 |
