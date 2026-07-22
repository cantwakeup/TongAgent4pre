# Score-First development iterations

Development set: `0016`, `0191`, `0718`. The set and formal holdout were pre-registered in commit `a8fb01f6b28deba3a623c6e89136dc817a05d7d5` before implementation. Reference answers were used only by post-run scoring.

| Iteration | Hypothesis / category | Commit | Answers (0016 / 0191 / 0718) | Raw EM | Normalized EM | Answer rate | Runner errors | Hard timeouts | Tokens (known) | Search / Fetch (known) | Mean wall | Decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Minimal Score-First path can replace online evidence gates / initial implementation | `27ec967` | timeout / ABSTAIN / `95` | 0/3 | 0/3 | 1/3 | 0 | 1 | 7,120 | 7 / 10 | 495.3s | Keep as working base; no early success |
| 2 | Earlier SQs monopolize fetch budget; cap candidate acquisition at two attempts per SQ / source selection | `22e0b09` | `25` / ABSTAIN / `95` | 0/3 | 0/3 | 2/3 | 0 | 0 | 14,215 | 9 / 12 | 118.4s | Keep: +1 answered task, -1 timeout, 76% lower mean wall |
| 3 | Treat singular `which` queries correctly, canonicalize statehood terms, and remove redundant combination SQs / query | `dd3f1fa` | `326.1 m` / ABSTAIN / `1873` | 0/3 | 0/3 | 2/3 | 0 | 0 | 13,004 | 8 / 13 | 117.9s | Revert: no score/answer gain and <20% efficiency gain |

Iteration 3 was reverted by `bec3b7133815f93f9c357861311fd288e3d9091a`. Development stopped after the pre-registered maximum of three iterations and nine TongAgent runs. The frozen behavior for the formal holdout is therefore the retained iteration-2 architecture.
