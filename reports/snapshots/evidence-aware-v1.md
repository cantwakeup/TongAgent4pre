# Evidence-aware TongAgent V1 snapshot

- Frozen implementation commit: `ab6350d770db2d6cf1d86fb3dfd7dd3549571dc5`
- Snapshot scope: retrieval backend, guarded public fetch, shared budgets, strict FSM,
  permissive workflow, post-hoc evidence verification, typed deterministic synthesis,
  and optional Fact-Gap retrieval.
- Primary fixed FRAMES set: `0016`, `0123`, `0191`, `0664`, `0718` (seed 17).

## Recorded experiments

| Experiment | Systems | Main observed result |
| --- | --- | --- |
| `frames-permissive-second-20260721-215544` | simple_react, vanilla_deepagents, TongAgent permissive | Raw normalized EM: 0/5, 2/5, 0/5. TongAgent completed 1/5 and partial 4/5; it created research/evidence artifacts but conservative finalization often abstained. |
| `frames-fact-gap-retry-20260722-004211` | TongAgent permissive + Fact-Gap | 5/5 partial, raw normalized EM 0/5. Fact-Gap increased retrieval use but did not establish an answer-rate gain. |

The original first benchmark and pilot outputs remain under `learning/search-agent/output/evaluations/`; this snapshot does not replace or reinterpret them.

## Artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| `frames-permissive-second-20260721-215544/summary.json` | `9906002dd2a47ef81df83a8dc3c80f1cb67f37ed80afe7601b0aeb8afce964da` |
| `frames-fact-gap-retry-20260722-004211/summary.json` | `3519d9ba82a60fdb39be99b0a4e5e9f7fbc7e8645b8ec46308ae2c1fd16125ee` |
| `frames-fact-gap-retry-20260722-004211/summary.md` | `f7bcfb285d938c0876c900989732e5b95b88d65ce52547badd8cae0f267b1f80` |

## Current conclusion and limitation

Evidence-aware V1 has a working, auditable retrieval-to-claim path but is not yet a
demonstrably better answering system: strict quote/registration gates and Fact-Gap
repairs can turn useful partial research into abstentions.  The next isolated branch
therefore evaluates candidate-answer revision and fixed selective-release policies;
it must not tune on FRAMES references or overwrite these artifacts.
