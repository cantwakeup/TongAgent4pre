#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="/home/huiwei/sy/TongAgent/langchain-deepagents/learning/search-agent"
REPO_ROOT="/home/huiwei/sy/TongAgent/langchain-deepagents"
ENV_FILE="/tmp/tongagent-frames-pilot.env"
CURRENT_EXP_FILE="/tmp/current-browsecomp-high-budget-exp.txt"

resume=0
dry_run=0
experiment_id=""

while (($#)); do
  case "$1" in
    --resume)
      resume=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --experiment-id)
      experiment_id="${2:?--experiment-id requires a value}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"
head_sha="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing launch: worktree is not clean" >&2
  exit 2
fi

if ((resume)); then
  if [[ -z "$experiment_id" ]]; then
    if [[ ! -s "$CURRENT_EXP_FILE" ]]; then
      echo "resume requires --experiment-id or $CURRENT_EXP_FILE" >&2
      exit 2
    fi
    experiment_id="$(<"$CURRENT_EXP_FILE")"
  fi
  manifest_artifact="$APP_ROOT/output/evaluations/$experiment_id/exploratory_manifest.json"
  if [[ ! -f "$manifest_artifact" ]]; then
    echo "resume manifest is missing: $manifest_artifact" >&2
    exit 2
  fi
  expected_head="$(jq -r '.git_sha' "$manifest_artifact")"
  if [[ "$head_sha" != "$expected_head" ]]; then
    echo "refusing resume: HEAD differs from original experiment" >&2
    exit 2
  fi
else
  if [[ -z "$experiment_id" ]]; then
    experiment_id="browsecomp-high-budget-resource-envelope-v1-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  expected_head="$head_sha"
fi

if ((dry_run)); then
  cd "$APP_ROOT"
  exec uv run python -m scripts.run_browsecomp_high_budget \
    --dataset evaluation/datasets/final_harness_benchmark_v2.jsonl \
    --manifest evaluation/manifests/browsecomp_high_budget_8.json \
    --config evaluation/configs/browsecomp_high_budget_gpt55.live.json \
    --experiment-id "$experiment_id" \
    --expected-head "$expected_head" \
    --dry-run
fi

if [[ ! -r "$ENV_FILE" ]]; then
  echo "credential file is missing or unreadable: $ENV_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY was not loaded" >&2
  exit 2
fi

umask 077
printf '%s\n' "$experiment_id" >"$CURRENT_EXP_FILE"

cd "$APP_ROOT"
args=(
  --dataset evaluation/datasets/final_harness_benchmark_v2.jsonl
  --manifest evaluation/manifests/browsecomp_high_budget_8.json
  --config evaluation/configs/browsecomp_high_budget_gpt55.live.json
  --experiment-id "$experiment_id"
  --expected-head "$expected_head"
)
if ((resume)); then
  args+=(--resume)
fi

echo "experiment_id=$experiment_id"
echo "study=exploratory resource-envelope study"
echo "schedule=8 tasks x 2 systems = 16 jobs, concurrency=1"
echo "cost_upper_bound_usd=11.60"
exec uv run python -m scripts.run_browsecomp_high_budget "${args[@]}"
