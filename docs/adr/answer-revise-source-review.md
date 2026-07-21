# ADR: Answer-Revise source review

Date: 2026-07-22

## Decision

Implement a local, source-bound Atomic Fact adapter and a deterministic
post-hoc support/revision layer.  We do not add a heavyweight external runtime
dependency and do not copy code from repositories without a clear license.

| Work | Reusable component | Integration choice | License | Copy/adapt/reimplement |
| --- | --- | --- | --- | --- |
| [FActScore](https://github.com/shmsw25/FActScore) | Atomic-fact decomposition and per-fact factuality framing | Use its public conceptual interface for short-answer atomization; retain source-bound local adapter | MIT | Reimplement locally; no vendored code in this change |
| [SAFE / long-form-factuality](https://github.com/google-deepmind/long-form-factuality) | Self-contained facts and search-grounded support checking | Use the self-contained-fact and post-hoc verification pattern | Apache-2.0 for `common/`, `eval/`, `longfact/`, and `main/`; bundled `third_party/factscore` is MIT | Reimplement locally; no dependency or copied code |
| [RARR](https://github.com/anthonywchen/RARR) | Question/evidence/agreement/revision sequence | Use only the high-level revise-or-remove principle | No repository license found during review | Algorithm only; no code copied |
| [CRAG](https://github.com/HuskyInSalt/CRAG) | Correct/ambiguous/incorrect retrieval triage and corrective fallback | Map post-hoc support outcomes to fixed retrieval-quality labels; retain existing bounded retrieval | No repository license found during review | Concept only; no code copied |

## Consequences

- Existing Strict, Permissive, and Fact-Gap modes remain unchanged.
- `answer_revise` is evaluated as a new runtime mode and bypasses Fact Slot / Fact
  Gap retrieval by default.
- Exact quote registration remains canonical Evidence Graph evidence; it is a
  post-hoc record and not a research-time blocking gate.
- All release policies are fixed before FRAMES replay or live scoring.  Reference
  answers are scoring-only and never enter planning, retrieval, revision, or policy
  decisions.
