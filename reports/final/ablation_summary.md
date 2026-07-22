# Architecture and Ablation Summary

The project converged through several frozen research modes:

| Variant | Role in the final story | Final status |
| --- | --- | --- |
| Score-First | reliability-first answer pipeline | frozen historical ablation |
| Long-ReAct v1 | performance-first long tool loop | frozen historical ablation |
| Structured Context | structured-state mechanism experiment | retained as ablation, not default |
| GPT-5.5 attribution canary | separated model strength from Harness effects | model effect confirmed; no baseline accuracy lead |
| TongAgent Standard | transparent operational Harness around Bare ReAct | final default Harness |

The attribution canary showed that the stronger model lifted multiple systems,
while Structured Context did not establish a general accuracy advantage over
the baselines. The final architecture therefore removed accuracy-altering
research control from the default path.

The formal 16-task result is consistent with that attribution:

```text
Bare Simple ReAct standard EM: 6/16
TongAgent Standard standard EM: 5/16
Vanilla DeepAgents standard EM: 5/16
```

TongAgent Standard does not beat the Bare policy on accuracy. Its demonstrated
contribution is operational: deterministic accounting, complete trace,
checkpoint/resume, retry, and non-blocking post-hoc audit.

The final claim is therefore limited to Harness value, not a stronger reasoning
algorithm.
