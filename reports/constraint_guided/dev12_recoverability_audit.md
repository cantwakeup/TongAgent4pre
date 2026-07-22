# Constraint-Guided dev12 Recoverability Audit

## Outcome

This is a pure offline audit. It made no Agent/runtime changes, called no model or network service, and launched no new benchmark. Selective Heavy Mode and a second trajectory remain disabled.

The primary evidence is the completed `constraint-guided-dev12-iter1-20260722T102000Z` artifact set. The frozen v1 experiment is used only to identify stable failure patterns. Reference answers were opened only after the runs for error classification.

The audit selected **Structured Context as the only proposed Iteration 2 mechanism**. It was subsequently implemented, tested on the frozen canary, and rejected by the preregistered hard gate. The code-only mechanism change was then reverted; the audit and experiment artifacts remain preserved.

## Artifact boundary

The stored artifacts contain full queries, ranked search results/snippets, tool-call decisions, fetch status/source metadata, Python arguments/results, context telemetry, and final answers. Model-message bodies and fetched-page bodies are intentionally persisted as hashes plus character counts. Therefore, a fact that may exist only inside a fetched page but is absent from saved snippets is marked conservatively as not auditable.

## Aggregate diagnosis

| Primary class | Tasks | Count |
|---|---|---:|
| Query/retrieval | 0299, 0612, 0637, 0750 | 4 |
| Context/operation/selection | 0069, 0132, 0360, 0615 | 4 |
| Fetch | 0087 | 1 |
| Format | 0163, 0473 | 2 |
| No benchmark failure | 0452 | 1 |

Additional recoverability signals:

- Explicit correct candidate appeared in 4/12 traces.
- The persisted fact chain is auditable in 6/12 traces; 0132 and 0360 are excluded because the required fetched passage is redacted or an operand is missing.
- The final answer satisfies normalized whole-string EM in only 1/12 traces.
- Runtime completed without timeout, budget exhaustion, or runner error in 8/12 traces.
- All required facts appeared in 6/12 traces, and a final operation was possible from saved trace facts in 6/12 traces.
- `stronger_model_or_retrieval` is recommended for 4/12, below the stop threshold of more than 6.
- Fetch is the unique primary class for only 1 task, below the two-task fetch-recovery trigger.

## Per-task audit

| Task | Runtime failure | Candidate seen | Final EM | Fact chain auditable | First divergence | Primary class |
|---|---|---:|---:|---:|---:|---|
| 0069 | token exhaustion | No | No | Yes | 7 | operation |
| 0087 | token exhaustion | No | No | Yes | 2 | fetch |
| 0132 | none | No | No | No | 6 | operation |
| 0163 | none | Yes | No | Yes | — | format |
| 0299 | token exhaustion | No | No | No | 1 | query |
| 0360 | none | Not auditable | No | No | 4 | context |
| 0452 | none | Yes | Yes | Yes | — | none |
| 0473 | none | Yes | No | Yes | — | format |
| 0612 | none | No | No | No | 1 | query |
| 0615 | none | Yes | No | Yes | 6 | selection |
| 0637 | none | No | No | No | 1 | query |
| 0750 | token exhaustion | No | No | No | 5 | query |

Key trace-level findings:

- **0069:** SDP/1899 and Parliament/1906 both appear, but the model first computes two dates inside 1906 and exhausts the token budget immediately after finding 1899.
- **0087:** search snippets contain Straneo, Bonelli, and both nationalities, but every canonical fetch fails; the terminal selection reuses a noisy result for another Lesticus species.
- **0132:** relevant dates/pages are partially present, but the Python calls are malformed or compute only the first interval. The exact 79 CE date is not retained in an auditable passage.
- **0163:** the 712 candidate and its operands are auditable and runtime succeeds, but the terminal short answer omits “years”, so whole-string EM is false.
- **0299:** the first query copies the multi-hop question, then the trace diverges to Omelas/New Dimensions rather than isolating the early world-building stories and their magazine.
- **0360:** the exact year table is successfully fetched, but the body is redacted in the artifact and the following model turn abstains.
- **0473:** the candidate 5 and supporting chain are auditable and runtime succeeds, but the short answer does not whole-string match the reference sentence.
- **0615:** `Summer Magic` is explicitly present after the Parent Trap/Hayley Mills chain, but the final answer emits the intermediate actress.
- **0637:** both runs misread “preceded” as a historical predecessor and never surface the required 1862 opening operand.
- **0750:** the trace asks for Alan Menken's lifetime Grammy total rather than the cumulative total at Tom Hanks's first Oscar.

## Single-mechanism decision

The two largest buckets are tied at four tasks. The deterministic tie-break is whether the existing trace already contains an executable fact chain:

- Selection/context/operation: 3 of 4 have an already-present fact chain that is lost, misapplied, or not selected.
- Query/retrieval: 0 of 4 contain the missing required fact.

Therefore the only preregistered Iteration 2 modification is **Structured Context**. It may retain bounded confirmed facts, operands, intermediate/final roles, and unresolved relations. It may not add fetch recovery, query rewriting, a final candidate selector, Heavy Mode, or a second trajectory in the same iteration.

## Frozen four-task canary

The canary is fixed from failure class before implementation:

| Role | Task | Failure-class basis |
|---|---|---|
| Mechanism | 0069 | operation failure; operands already present |
| Mechanism | 0615 | selection failure; intermediate and final candidates present |
| Control | 0612 | query failure; required fact absent |
| Control | 0637 | query/relation failure; required operand absent |

The controls are not chosen from reference-answer difficulty. They are two tasks outside the selected mechanism's class.

The frozen-v1 values on these four tasks are: 4/4 legal results, zero timeout/runner error/token exhaustion, 9 search calls, 9 fetch calls, and 50,583.5 mean tokens.

Expansion requires **all** of the following: at least one new EM on 0069 or 0615, 4/4 legal results, zero timeout and runner error, zero token exhaustion, no more than 9 searches and 9 fetches, and mean tokens no greater than 55,641.85. Repairing two fact chains without an EM gain is diagnostic only and cannot authorize dev12.

The same-model Structured Context canary `constraint-guided-structured-context-canary-20260722T130936Z` produced 4 result artifacts, 0/4 EM, zero timeout/runner error, one search-budget exhaustion, 12 searches, 11 fetches, and 27,782.5 mean tokens. Neither mechanism task gained EM, so the canary failed and dev12 expansion was prohibited.

A separate model-capacity follow-up used `gpt-5.5` because the original model was `gpt-5.4-nano`. It produced 2/4 EM and repaired mechanism task 0615, with 4/4 legal results and no exhaustion. It nevertheless used 13 searches and 11 fetches, above the frozen 9/9 ceilings. This changed-model diagnostic supports model capacity as an important bottleneck, but it does not retroactively validate Structured Context or authorize dev12.
