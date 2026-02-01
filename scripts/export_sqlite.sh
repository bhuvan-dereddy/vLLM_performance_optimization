#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_profile.sh"

IN_REP="${REPO_ROOT}/artifacts/vllm_capture.nsys-rep"
OUT_SQLITE="${REPO_ROOT}/artifacts/vllm_capture.sqlite"

if [[ ! -f "${IN_REP}" ]]; then
  echo "ERROR: missing ${IN_REP}"
  exit 1
fi

# Important:
# Use `nsys stats` to generate the SQLite with raw trace tables
# (OSRT_API, CUPTI_ACTIVITY_KIND_KERNEL, etc).
"${NSYS}" stats \
  --force-overwrite=true \
  --output="${REPO_ROOT}/artifacts/vllm_capture" \
  "${IN_REP}" >/dev/null

# nsys stats writes: <output>.sqlite
if [[ ! -f "${OUT_SQLITE}" ]]; then
  echo "ERROR: expected sqlite not found: ${OUT_SQLITE}"
  echo "Check nsys version/output naming."
  exit 1
fi

echo "Wrote: ${OUT_SQLITE}"
