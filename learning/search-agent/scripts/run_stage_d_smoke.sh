#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
EXPERIMENT_ID="${1:-stage-d-offline-smoke}"
DATASET="${PROJECT_ROOT}/evaluation/datasets/stage_d_offline_smoke.jsonl"
FIXTURES="${PROJECT_ROOT}/evaluation/fixtures"
OUTPUT_ROOT="${PROJECT_ROOT}/output/evaluations"
EXPERIMENT_DIRECTORY="${OUTPUT_ROOT}/${EXPERIMENT_ID}"

if [[ ! "${EXPERIMENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "invalid experiment id: ${EXPERIMENT_ID}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -e "${EXPERIMENT_DIRECTORY}" ]]; then
  echo "refusing to reuse an existing experiment: ${EXPERIMENT_DIRECTORY}" >&2
  echo "pass a new experiment id; this script never deletes prior results" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m evaluation.validate_stage_d \
  --dataset "${DATASET}" \
  --fixtures "${FIXTURES}"

fresh_output="$(
  "${PYTHON_BIN}" -m evaluation.cli run \
    --systems simple_react vanilla_deepagents tongagent \
    --dataset "${DATASET}" \
    --seed 0 \
    --output "${OUTPUT_ROOT}" \
    --experiment "${EXPERIMENT_ID}" \
    --no-resume
)"
printf '%s\n' "${fresh_output}" >"${EXPERIMENT_DIRECTORY}/fresh-run.json"

(
  cd "${EXPERIMENT_DIRECTORY}"
  find . -type f -path '*/attempt-*/result.json' -print0 \
    | sort -z \
    | xargs -0 -r sha256sum
) >"${EXPERIMENT_DIRECTORY}/fresh-results.sha256"

resume_output="$(
  "${PYTHON_BIN}" -m evaluation.cli run \
    --systems simple_react vanilla_deepagents tongagent \
    --dataset "${DATASET}" \
    --seed 0 \
    --output "${OUTPUT_ROOT}" \
    --experiment "${EXPERIMENT_ID}" \
    --resume
)"
printf '%s\n' "${resume_output}" >"${EXPERIMENT_DIRECTORY}/resume-run.json"

(
  cd "${EXPERIMENT_DIRECTORY}"
  find . -type f -path '*/attempt-*/result.json' -print0 \
    | sort -z \
    | xargs -0 -r sha256sum
) >"${EXPERIMENT_DIRECTORY}/resume-results.sha256"
cmp \
  "${EXPERIMENT_DIRECTORY}/fresh-results.sha256" \
  "${EXPERIMENT_DIRECTORY}/resume-results.sha256"

rerun_output="$(
  "${PYTHON_BIN}" -m evaluation.cli run \
    --systems simple_react \
    --dataset "${DATASET}" \
    --limit 1 \
    --seed 0 \
    --output "${OUTPUT_ROOT}" \
    --experiment "${EXPERIMENT_ID}" \
    --rerun
)"
printf '%s\n' "${rerun_output}" >"${EXPERIMENT_DIRECTORY}/rerun-run.json"

validation_output="$(
  "${PYTHON_BIN}" -m evaluation.validate_stage_d \
    --dataset "${DATASET}" \
    --fixtures "${FIXTURES}" \
    --experiment-directory "${EXPERIMENT_DIRECTORY}" \
    --require-rerun
)"
printf '%s\n' "${validation_output}" >"${EXPERIMENT_DIRECTORY}/validation.json"
printf '%s\n' "${validation_output}"

echo "Stage D deterministic fixture smoke passed: ${EXPERIMENT_DIRECTORY}"
