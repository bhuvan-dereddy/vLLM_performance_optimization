from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from gpuprof.bruteforce import BruteForceSearch
from gpuprof.evaluate import evaluate
from gpuprof.io_utils import load_json, write_json
from gpuprof.paths import baseline_dir as get_baseline_dir, search_dir_latest
from gpuprof.server_cmd import get_knob_flags


def run_baseline(cfg: Dict[str, Any]) -> None:
    baseline_dir = get_baseline_dir()
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

    baseline_dir = get_baseline_dir()
    if not baseline_dir.exists():
        run_baseline(cfg)
        return

    search_dir = search_dir_latest()
    search_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_json(baseline_dir / "client_metrics.json")
    BruteForceSearch(cfg, search_dir).run(baseline_metrics)
