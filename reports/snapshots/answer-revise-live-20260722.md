# Answer-Revise live run snapshot (2026-07-22)

- Commit: `5f48b9440f7361dd74933a00506496590bb6e8a7`
- Experiment: `frames-answer-revise-20260722-092749`
- Runtime: `answer_revise`
- Terminal results: 5/5; incomplete attempts: 0
- Outcomes: four `deadline_exceeded` subprocess terminals and one normal
  `partial`/`abstain` terminal (`frames-test-0718`)
- Raw normalized exact match: 0/5
- Mean wall time: 601.257 seconds; total wall time: 3006.286 seconds

This is a recoverability snapshot, not evidence of an algorithmic win.  The only
run that reached selective finalization produced a candidate but abstained because
the deterministic answer execution did not complete.  Raw outputs remain under
`learning/search-agent/output/evaluations/frames-answer-revise-20260722-092749`.

## Integrity

| Artifact | SHA-256 |
| --- | --- |
| `summary.json` | `81dc79acbe69a2ab9265a401172792e8a8568d1396348878e358ff5d1c5b5f7e` |
| `final-summary.json` | `81dc79acbe69a2ab9265a401172792e8a8568d1396348878e358ff5d1c5b5f7e` |
| `answer_revise_launcher_manifest.json` | `51ca53f1ee64ae8880128c869c2914096df6da5bc58e3186fc6614d25a56bac9` |
