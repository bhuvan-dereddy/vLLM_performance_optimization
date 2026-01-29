#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"
source "${REPO_ROOT}/scripts/env_profile.sh"

mkdir -p "${REPO_ROOT}/artifacts"
OUT_BASE="${REPO_ROOT}/artifacts/vllm_capture"

"${NSYS}" profile \
  --force-overwrite=true \
  --stats=true \
  --trace=osrt,cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o "${OUT_BASE}" \
  python "${REPO_ROOT}/scripts/run_vllm.py"

echo "✅ Wrote: ${OUT_BASE}.nsys-rep"
