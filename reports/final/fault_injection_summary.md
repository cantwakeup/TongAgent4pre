# Fault-Injection Summary

## Experiment

```text
experiment: final-transparent-harness-v2-fault-20260722T175204Z
faults: fetch first-attempt failure, temporary model connection failure,
        process interruption and resume, near-budget exhaustion
tasks per fault: 2
systems: bare_simple_react, tongagent_standard
runs: 16
model/retrieval backend: local fixture
API cost: $0.00
```

## Aggregate results

| System | Recovery | Valid terminal | Trace complete |
| --- | ---: | ---: | ---: |
| Bare Simple ReAct | 4/8 (50%) | 6/8 (75%) | 6/8 (75%) |
| TongAgent Standard | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |

## By fault

| Fault | Bare recovery | Standard recovery | Standard behavior |
| --- | ---: | ---: | --- |
| Fetch first attempt | 0/2 | 2/2 | one explicit failed-provider retry per case |
| Model connection | 0/2 | 2/2 | one transparent model retry per case |
| Process interruption/resume | 2/2 | 2/2 | checkpoint restored 2/2; no duplicate successful fetch |
| Near budget | 2/2 | 2/2 | valid terminal result without bypassing budget |

The fetch retry records one additional provider fetch attempt, as expected,
but process resume records zero duplicate fetches for TongAgent Standard. Bare
restart repeats one successful fetch in each resume case.

TongAgent Standard added 605 fixture tokens across the two fetch-retry cases,
zero additional tokens in the resume and near-budget cases, and about 4.59
seconds total additional wall time across all eight controlled cases. Model
connection failures do not expose a comparable failed-call token count.

All preregistered recovery targets pass:

```text
recovery success >= 80%: YES
trace completeness = 100%: YES
resume avoids duplicate successful fetch: YES
valid terminal rate > Bare: YES

OPERATIONAL_RELIABILITY_SUPERIORITY = YES
RECOVERY_SUPERIOR = YES
```
