from __future__ import annotations

import argparse
import csv
import itertools
import os
import shlex
import signal
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import request

from gpuprof.bruteforce import BruteForceSearch
from gpuprof.constants import RESULTS_ROOT
from gpuprof.evaluate import evaluate
from gpuprof.formatting import format_num, format_delta
from gpuprof.io_utils import load_json, write_json
from gpuprof.process import http_get, wait_for_ready, terminate_process_group, wait_for_report_finalized
from gpuprof.search import choose_best, print_search_output, write_summary_csv
from gpuprof.server_cmd import get_knob_flags


def run_baseline(cfg: Dict[str, Any]) -> None:
    baseline_dir = RESULTS_ROOT / "baseline"
    mask = "0" * len(cfg.get("knobs", []))
    _, flags, assignment = get_knob_flags(cfg.get("knobs", []), [0] * len(cfg.get("knobs", [])))
    result = evaluate(cfg, baseline_dir, flags, assignment, profile=True)

    config_path = baseline_dir / "config.json"
    merged_config = load_json(config_path) if config_path.exists() else {}
    merged_config.update(
        {
            "knob_mask": mask,
            "assignment": assignment,
            "flags": flags,
            "resolved_server_cmd": result.resolved_cmd,
            "resolved_env": {k: v for k, v in result.resolved_env.items() if k in (cfg.get("env") or {}) or k in ((cfg.get("server") or {}).get("env") or {})},
        }
    )
    write_json(
        config_path,
        merged_config,
    )

    if not (baseline_dir / "trace.nsys-rep").exists():
        raise SystemExit("Baseline profiling failed: trace.nsys-rep missing")

    print("BASELINE_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="optimizations.json")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists() and args.config == "optimizations.json":
        fallback = Path("configs/optimizations.json")
        if fallback.exists():
            cfg_path = fallback

    cfg = load_json(cfg_path)

    baseline_dir = RESULTS_ROOT / "baseline"
    if not baseline_dir.exists():
        run_baseline(cfg)
        return

    search_dir = Path("results/search/latest")
    search_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_json(baseline_dir / "client_metrics.json")
    BruteForceSearch(cfg, search_dir).run(baseline_metrics)


if __name__ == "__main__":
    main()
