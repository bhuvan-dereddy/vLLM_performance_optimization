#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"

source "${REPO_ROOT}/venv/bin/activate"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export NSYS="/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys"

if [ ! -x "${NSYS}" ]; then
  echo "ERROR: NSYS not found at ${NSYS}"
  exit 1
fi
