# Constraint-Guided dev12 Recoverability Audit

## Outcome

This is a pure offline audit. It made no Agent/runtime changes, called no model or network service, and launched no new benchmark. Selective Heavy Mode and a second trajectory remain disabled.

The primary evidence is the completed `constraint-guided-dev12-iter1-20260722T102000Z` artifact set. The frozen v1 experiment is used only to identify stable failure patterns. Reference answers were opened only after the runs for error classification.

The audit selects **Structured Context as the only proposed Iteration 2 mechanism**, but does not implement or run it in this audit.

## Artifact boundary

The stored artifacts contain full queries, ranked search results/snippets, tool-call decisions, fetch status/source metadata, Python arguments/results, context telemetry, and final answers. Model-message bodies and fetched-page bodies are intentionally persisted as hashes plus character counts. Therefore, a fact that may exist only inside a fetched page but is absent from saved snippets is marked conservatively as not auditable.

## Aggregate diagnosis

| Primary class | Tasks | Count |
|---|---|---:|
| Query/retrieval | 0299, 0612, 0637, 0750 | 4 |
| Context/operation/selection | 0069, 0132, 0360, 0615 | 4 |
| Fetch | 0087 | 1 |
| No failure | 0163, 0452, 0473 | 3 |

Additional recoverability signals:

- Explicit correct candidate appeared in 4/12 traces.
- All required facts appeared in 6/12 traces.
- A final operation was possible from saved trace facts in 6/12 traces.
- `stronger_model_or_retrieval` is recommended for 4/12, below the stop threshold of more than 6.
- Fetch is the unique primary class for only 1 task, below the two-task fetch-recovery trigger.

## Per-task audit

| Task | Candidate seen | Required facts | Operation possible | First divergence | Primary class | Main recoverability |
|---|---:|---:|---:|---:|---|---|
| 0069 | No | Yes | Yes | 7 | operation | structured context |
| 0087 | No | Yes | Yes | 2 | fetch | fetch recovery; structured context |
| 0132 | No | No | No | 6 | operation | structured context; stronger capability |
| 0163 | Yes | Yes | Yes | — | none | already correct |
| 0299 | No | No | No | 1 | query | query reformulation |
| 0360 | Not auditable | Not auditable | No | 4 | context | structured context |
| 0452 | Yes | Yes | Yes | — | none | already correct |
| 0473 | Yes | Yes | Yes | — | none | already correct |
| 0612 | No | No | No | 1 | query | query reformulation |
| 0615 | Yes | Yes | Yes | 6 | selection | structured context; candidate selection |
| 0637 | No | No | No | 1 | query | relation-aware query reformulation |
| 0750 | No | No | No | 5 | query | temporal query reformulation |

Key trace-level findings:

- **0069:** SDP/1899 and Parliament/1906 both appear, but the model first computes two dates inside 1906 and exhausts the token budget immediately after finding 1899.
- **0087:** search snippets contain Straneo, Bonelli, and both nationalities, but every canonical fetch fails; the terminal selection reuses a noisy result for another Lesticus species.
- **0132:** relevant dates/pages are partially present, but the Python calls are malformed or compute only the first interval. The exact 79 CE date is not retained in an auditable passage.
- **0299:** the first query copies the multi-hop question, then the trace diverges to Omelas/New Dimensions rather than isolating the early world-building stories and their magazine.
- **0360:** the exact year table is successfully fetched, but the body is redacted in the artifact and the following model turn abstains.
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

No canary has been run. A future four-task run may expand to dev12 only if it gains at least one EM **or** repairs at least two preregistered fact chains, has zero timeout and runner errors, and does not exceed the frozen-v1 token-exhaustion count for these same four tasks. Otherwise the one mechanism must be reverted and dev12 must not run.
