#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"
source "${REPO_ROOT}/scripts/env_profile.sh"

IN_REP="${REPO_ROOT}/artifacts/p2_imagenette_opt4.nsys-rep"
OUT_SQLITE="${REPO_ROOT}/artifacts/p2_imagenette_opt4.sqlite"

"${NSYS}" export \
  --type sqlite \
  --force-overwrite=true \
  -o "${OUT_SQLITE}" \
  "${IN_REP}"

echo "✅ Wrote: ${OUT_SQLITE}"
