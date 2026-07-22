# Reproduction

## Frozen inputs

```text
branch: exp/final-transparent-harness
commit: 7a94dd2a583fc9770586a5de5fbad1e37a2639c1
manifest: learning/search-agent/evaluation/manifests/final_harness_benchmark_v2.json
dataset: learning/search-agent/evaluation/datasets/final_harness_benchmark_v2.jsonl
config: learning/search-agent/evaluation/configs/final_harness_gpt55_seed17_v2.live.json
```

The credential file is `/tmp/tongagent-frames-pilot.env`. Check only its
existence, owner, mode, and that `OPENAI_API_KEY` is non-empty. Never print the
value. The formal run used a mode-600, current-user-owned file.

## Offline validation

From `learning/search-agent`:

```bash
uv run --group test ruff format --check .
uv run --group test ruff check .
uv run --group test python -m compileall -q evaluation tests scripts *.py
PYTHONPATH=. uv run --group test python -m pytest --disable-socket -q
```

Observed result: `405 passed, 78 subtests passed`.

## Dry-run

```bash
env -u OPENAI_API_KEY PYTHONPATH=. uv run python -m evaluation.cli run \
  --systems bare_simple_react vanilla_deepagents tongagent_standard \
  --dataset evaluation/datasets/final_harness_benchmark_v2.jsonl \
  --seed 17 \
  --output output/evaluations \
  --experiment final-transparent-harness-v2-preflight \
  --config evaluation/configs/final_harness_gpt55_seed17_v2.live.json \
  --dry-run
```

Expected schedule: `16 tasks × 3 systems = 48 jobs`.

## Formal benchmark

The completed immutable experiment is:

```text
output/evaluations/final-transparent-harness-v2-formal-20260722T162330Z
```

It was launched once in a foreground shell after safely sourcing the credential
file. It produced 48 `result.json` files, no `attempt-0002`, and canonical
`summary.json`, `summary.csv`, and `summary.md` files.

Do not rerun it. Algorithm failures and the Bare runner error are part of the
formal result.

## Fault injection

```bash
env -u OPENAI_API_KEY PYTHONPATH=. uv run python -m evaluation.fault_injection \
  --manifest evaluation/manifests/final_fault_injection.json \
  --output output/evaluations/final-transparent-harness-v2-fault-20260722T175204Z
```

This evaluation is fixture-only and makes no model or public-network call.

## Final interpretation

```text
ACCURACY_SUPERIORITY = NO
PERFORMANCE_NON_INFERIORITY = NO
OPERATIONAL_RELIABILITY_SUPERIORITY = YES

FINAL_HARNESS_FROZEN = YES
PERFORMANCE_NON_INFERIOR = NO
RECOVERY_SUPERIOR = YES
TONGAGENT_BEATS_BASELINE_ON_ACCURACY = NO
TONGAGENT_PROVIDES_HARNESS_VALUE = YES
```

`PERFORMANCE_NON_INFERIOR` is fail-closed because one Bare artifact lacks token
usage, even though every observable non-inferiority comparison passes. The
Harness-value conclusion rests on the preregistered controlled recovery
experiment, not on an accuracy claim.
