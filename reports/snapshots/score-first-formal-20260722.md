# Score-First formal holdout snapshot

- Experiment: `score-first-formal-holdout-20260722T041353Z`
- Frozen commit: `6ed6ce8ee2012e03200eda8e41fb186826b339f1`
- Fairness fingerprint: `sha256:a339208589ba26501a480ada8eb47c460ca71c9ed91e0cd87fded81254af65ac`
- Dataset digest: `sha256:b461b502332f6a9a4a041abc35b78cafbdf278bd181afe59911c2cb90c7ae4c8`
- Model/seed/watchdog: `gpt-5.4-nano` / `17` / `600s`
- Result integrity: 15 selected results, 0 incomplete attempts, 0 runner errors, 0 timeouts

| System | Raw EM | Normalized EM | Answer rate | Abstain | Budget exhausted | Total tokens | Search / Fetch | Mean wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_react | 1/5 | 1/5 | 4/5 | 1 | 0 | 119,650 | 7 / 10 | 63.8s |
| vanilla_deepagents | 0/5 | 0/5 | 4/5 | 1 | 1 token | 249,066 | 7 / 14 | 73.0s |
| tongagent score_first | 0/5 | 0/5 | 1/5 | 4 | 0 | 22,788 | 14 / 21 | 113.2s |

| Task | simple_react | vanilla_deepagents | score_first TongAgent |
|---|---|---|---|
| 0109 | Gran Canaria (wrong) | Tenerife with explanation (whole-string EM false) | ABSTAIN |
| 0235 | Aaron Dessner, July 21 1976 (wrong) | March 31 1984 (wrong) | ABSTAIN |
| 0326 | 12 years (exact) | 12 years older (whole-string EM false) | ABSTAIN |
| 0485 | ABSTAIN | token-budget ABSTAIN | ABSTAIN |
| 0504 | Barbara Kingsolver with explanation (whole-string EM false) | Barbara Kingsolver with explanation (whole-string EM false) | 1892 (wrong) |

Score-First produced successful fetched sources and Research Notes on all five tasks, but only one final answer. Its sole answer had complete source-ID mapping, yet the mapped value was an intermediate fact rather than the requested entity. The non-blocking post-hoc artifact therefore reports source-mapping coverage, not semantic claim verification. The principal remaining failure is Notes-to-final-answer synthesis/operation selection, not runner, timeout, or token exhaustion.

Final decisions: `SCORE_FIRST_STAGE_COMPLETED=YES`, `SCORE_FIRST_PIPELINE_WORKING=NO`, `TONGAGENT_BEATS_BASELINE=NO`.
