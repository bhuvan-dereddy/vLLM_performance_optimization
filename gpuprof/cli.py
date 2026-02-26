from __future__ import annotations

import argparse
import re
import secrets
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from gpuprof.bruteforce import BruteForceSearch, build_sweep_definition
from gpuprof.evaluate import evaluate, resolve_workload_params
from gpuprof.io_utils import load_json, write_json
from gpuprof.reporting import print_search_output
from gpuprof.server_cmd import resolve_server

RUNS_ROOT = Path("runs")


def generate_run_id() -> str:
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{prefix}_{suffix}"


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise SystemExit("Invalid --run-id. Use only letters, numbers, underscore, dash, or dot.")


def create_run_dir(run_id: str) -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def collect_runtime_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "vllm_version": None,
        "driver_version": None,
        "cuda_version": None,
        "git_commit_hash": None,
    }

    try:
        from importlib import metadata as importlib_metadata

        info["vllm_version"] = importlib_metadata.version("vllm")
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            driver_match = re.search(r"Driver Version:\s*([0-9.]+)", proc.stdout)
            cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", proc.stdout)
            if driver_match:
                info["driver_version"] = driver_match.group(1)
            if cuda_match:
                info["cuda_version"] = cuda_match.group(1)
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            if commit:
                info["git_commit_hash"] = commit
    except Exception:
        pass

    return info


def write_config_resolved(cfg: Dict[str, Any], cfg_path: Path, run_dir: Path) -> None:
    serve_cmd, _, _, _, _ = resolve_server(cfg)
    workload = resolve_workload_params(cfg, serve_cmd)
    sweep_definition = build_sweep_definition(cfg)

    write_json(
        run_dir / "config_resolved.json",
        {
            "original_config_path": str(cfg_path.resolve()),
            "resolved_server_command_argv": serve_cmd,
            "resolved_workload": {
                "input_token_count": workload["input_token_count"],
                "output_token_count": workload["output_token_count"],
            },
            "resolved_sweep_definition": sweep_definition,
            "runtime": collect_runtime_info(),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run strict baseline->sweep->best experiment pipeline.")
    ap.add_argument("--config", required=True, help="Path to experiment config JSON.")
    ap.add_argument("--run-id", default=None, help="Optional run ID; default is YYYYMMDD_HHMMSS_<6char>.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")
    cfg = load_json(cfg_path)

    run_id = args.run_id or generate_run_id()
    validate_run_id(run_id)
    run_dir = create_run_dir(run_id)

    baseline_dir = run_dir / "baseline"
    sweep_dir = run_dir / "sweep"
    best_dir = run_dir / "best"
    baseline_dir.mkdir(parents=True, exist_ok=False)
    sweep_dir.mkdir(parents=True, exist_ok=False)
    best_dir.mkdir(parents=True, exist_ok=False)

    write_config_resolved(cfg, cfg_path, run_dir)

    baseline_result = evaluate(cfg, baseline_dir, [], {}, profile=True)
    if not (baseline_dir / "trace.nsys-rep").exists():
        raise SystemExit("Baseline profiling failed: trace.nsys-rep missing")

    sweep_definition = build_sweep_definition(cfg)
    sweep_result = BruteForceSearch(cfg, sweep_dir, sweep_definition).run(baseline_result.metrics)
    best_trial = sweep_result.best_trial
    best_assignment = {str(k): v for k, v in (best_trial.get("assignment") or {}).items()}
    best_flags = [str(x) for x in (best_trial.get("flags") or [])]

    best_result = evaluate(cfg, best_dir, best_flags, best_assignment, profile=True)
    if not (best_dir / "trace.nsys-rep").exists():
        raise SystemExit("Best profiling failed: trace.nsys-rep missing")

    baseline_p95_total_ms = baseline_result.metrics.get("p95_latency_ms")
    best_p95_total_ms = best_result.metrics.get("p95_latency_ms")
    absolute_reduction_ms: Any = None
    percent_improvement: Any = None
    if baseline_p95_total_ms is not None and best_p95_total_ms is not None:
        absolute_reduction_ms = float(baseline_p95_total_ms) - float(best_p95_total_ms)
        if float(baseline_p95_total_ms) != 0.0:
            percent_improvement = (absolute_reduction_ms / float(baseline_p95_total_ms)) * 100.0

    best_config = {
        "trial": best_trial.get("trial"),
        "knob_mask": best_trial.get("knob_mask"),
        "choice_indices": best_trial.get("choice_indices"),
        "assignment": best_assignment,
        "flags": best_flags,
        "knob_values": best_trial.get("knob_values"),
        "sweep_metrics": best_trial.get("metrics"),
        "profile_metrics": best_result.metrics,
        "resolved_server_cmd": best_result.resolved_cmd,
    }
    write_json(best_dir / "best_config.json", best_config)
    write_json(
        best_dir / "summary.json",
        {
            "baseline_p95_total_ms": baseline_p95_total_ms,
            "best_p95_total_ms": best_p95_total_ms,
            "absolute_reduction_ms": absolute_reduction_ms,
            "percent_improvement": percent_improvement,
            "best_trial": best_config,
            "profile_trace_summary": best_result.trace_summary,
            "profile_nvtx_phases": best_result.nvtx_phases,
            "baseline_metrics": baseline_result.metrics,
            "baseline_trace_summary": baseline_result.trace_summary,
        },
    )

    print_search_output(
        rows=sweep_result.rows,
        baseline_metrics=baseline_result.metrics,
        best_row=best_result.metrics | {"knob_mask": str(best_trial.get("knob_mask"))},
        baseline_trace=baseline_result.trace_summary,
        best_trace=best_result.trace_summary,
        baseline_nvtx=baseline_result.nvtx_phases,
        best_nvtx=best_result.nvtx_phases,
    )
