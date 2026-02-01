#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"
cd "${REPO_ROOT}"

# --- Activate venv so `python` exists ---
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/venv/bin/activate"
else
  echo "ERROR: venv not found at ${REPO_ROOT}/venv. Create it first."
  exit 1
fi

echo "=== ENV ==="
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_profile.sh"

mkdir -p "${REPO_ROOT}/artifacts"

echo "=== VLLM SETUP ==="
python "${REPO_ROOT}/scripts/download_model.py"
# download once; your script should skip if already present

echo "=== VLLM PROFILE (NSYS -> .nsys-rep) ==="
bash "${REPO_ROOT}/scripts/profile_vllm_nsys.sh"

echo "=== SQLITE EXPORT (RAW TABLES via nsys stats) ==="
bash "${REPO_ROOT}/scripts/export_sqlite.sh"

echo "=== PARSE SQLITE -> JSON SUMMARY ==="
python "${REPO_ROOT}/parse/parse_sqlite_baseline.py" \
  "${REPO_ROOT}/artifacts/vllm_capture.sqlite" \
  --out "${REPO_ROOT}/artifacts/vllm_capture_summary.json"

echo "=== DONE ==="
ls -lh "${REPO_ROOT}/artifacts/vllm_capture."*
echo "Wrote JSON: ${REPO_ROOT}/artifacts/vllm_capture_summary.json"
