#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/gpu-profiling"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/venv/bin/activate"

echo "=== IMAGENETTE BASELINE ==="

bash scripts/profile_imagenette_baseline.sh
# profile baseline → artifacts/p2_imagenette_baseline.nsys-rep

bash scripts/export_imagenette_baseline_sqlite.sh
# export baseline nsys-rep → sqlite

python parse/parse_sqlite_baseline.py artifacts/p2_imagenette_baseline.sqlite \
  --out artifacts/p2_imagenette_baseline_summary.json
# parse baseline sqlite → JSON summary


echo "=== IMAGENETTE OPT4 ==="

bash scripts/profile_imagenette_opt4.sh
# profile opt4 → artifacts/p2_imagenette_opt4.nsys-rep

bash scripts/export_imagenette_opt4_sqlite.sh
# export opt4 nsys-rep → sqlite

python parse/parse_sqlite_baseline.py artifacts/p2_imagenette_opt4.sqlite \
  --out artifacts/p2_imagenette_opt4_summary.json
# parse opt4 sqlite → JSON summary


echo "=== DIFF ==="

python parse/parse_diff_imagenette.py
# baseline vs opt4 diff → artifacts/p2_imagenette_diff.json


echo "=== DONE ==="
ls -lh artifacts/p2_imagenette_*