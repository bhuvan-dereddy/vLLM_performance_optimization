#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"

# venv
source "${REPO_ROOT}/venv/bin/activate"

# vLLM profiling stability: create worker as clean process
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Nsight Systems binary
export NSYS="/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys"
