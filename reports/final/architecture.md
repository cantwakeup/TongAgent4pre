# Final Transparent Harness Architecture

## Frozen design

The final architecture separates Agent policy from operational Harness
services.

```text
question
  -> shared transparent ReAct policy
     -> web_search / fetch_url
     -> raw FINAL_ANSWER
  -> non-blocking Harness side effects
     -> budget telemetry
     -> structured trace
     -> checkpoint/resume
     -> retry and cache
     -> Source Ledger and post-hoc audit
```

`bare_simple_react` and `tongagent_standard` share the same explicit prompt,
model-facing tools, graph builder, model parameters, and ReAct loop. Their
prompt SHA-256 is:

```text
73a4e7cc90c0ed0b2f5f743de28f35747647fc33c80e1bb6af706b1e5ada6eef
```

TongAgent Standard adds only checkpoint, retry, cache, telemetry, trace, and
post-hoc provenance services. It does not perform answer synthesis or evidence
gating. In the formal benchmark, all 16 TongAgent Standard artifacts satisfy:

```text
raw_model_answer == final_answer
```

All 16 also contain a checkpoint database, structured trace, and post-hoc
audit.

`vanilla_deepagents` remains the repository-pinned mature Harness baseline. Its
native planning/delegation prompt is part of the compared system and has hash:

```text
8bf254b3638c8fdf41adbf65a7477aadb94060414bdc55798115c7198c79ab03
```

It shares resources with the other systems but is not included in policy-level
prompt parity.

## Shared evaluation resources

The three systems share GPT-5.5, seed 17, the 16-task dataset, Unified
Retrieval Backend, search/fetch limits `4/6`, 60,000 total tokens, 3,000 maximum
output tokens, a 600-second watchdog, question-only input, and the same scorer.
The common fairness fingerprint is:

```text
sha256:516c790a52a97ee393c990af08ec8229658469529e7264dad2fa77feeb487233
```

Evidence Graph and citation mapping remain available as post-hoc audit data;
they cannot reject or rewrite the model answer in TongAgent Standard.

## Frozen commit

```text
branch: exp/final-transparent-harness
commit: 7a94dd2a583fc9770586a5de5fbad1e37a2639c1
```

No Agent behavior changed after this commit.
