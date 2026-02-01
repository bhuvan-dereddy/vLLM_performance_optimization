#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"
source "${REPO_ROOT}/scripts/env_profile.sh"

mkdir -p "${REPO_ROOT}/artifacts"
OUT_BASE="${REPO_ROOT}/artifacts/p2_imagenette_baseline"

"${NSYS}" profile \
  --force-overwrite=true \
  --stats=true \
  --trace=osrt,cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o "${OUT_BASE}" \
  python "${REPO_ROOT}/scripts/run_imagenette.py" \
    --run_name p2_baseline \
    --dataset_size 320px \
    --batch_size 64 \
    --num_workers 0 \
    --warmup_steps 10 \
    --profile_steps 50

echo " Wrote: ${OUT_BASE}.nsys-rep"
