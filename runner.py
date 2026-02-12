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

from gpuprof.evaluate import evaluate
from gpuprof.server_cmd import get_knob_flags
from gpuprof.utils import load_json, write_json, format_num, format_delta


def choose_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        r
        for r in records
        if r.get("error_rate") == 0.0
        and r.get("p95_latency_ms") is not None
        and r.get("p99_latency_ms") is not None
        and r.get("p95_ttft_ms") is not None
        and r.get("p50_latency_ms") is not None
        and r.get("chunks_per_s") is not None
    ]
    if not valid:
        raise SystemExit("No valid combinations with error_rate == 0")

    def key_fn(r: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        p95 = float(r.get("p95_latency_ms"))
        p99 = float(r.get("p99_latency_ms"))
        p95_ttft = float(r.get("p95_ttft_ms"))
        p50 = float(r.get("p50_latency_ms"))
        tps = float(r.get("chunks_per_s")) if r.get("chunks_per_s") is not None else float("-inf")
        return (p95, p99, p95_ttft, p50, -tps)

    return sorted(valid, key=key_fn)[0]


def print_search_output(
    rows: List[Dict[str, Any]],
    baseline_metrics: Dict[str, Optional[float]],
    best_row: Dict[str, Any],
    baseline_trace: Dict[str, Any],
    best_trace: Dict[str, Any],
    baseline_nvtx: Dict[str, Any],
    best_nvtx: Dict[str, Any],
) -> None:
    print("trial,knob_mask,p95_latency_ms,p99_latency_ms,p95_ttft_ms,p50_latency_ms,chunks_per_s,error_rate,delta_p50_ms")
    b_p50 = baseline_metrics.get("p50_latency_ms")
    for r in rows:
        print(
            ",".join(
                [
                    str(r["trial"]),
                    str(r["knob_mask"]),
                    format_num(r.get("p95_latency_ms")),
                    format_num(r.get("p99_latency_ms")),
                    format_num(r.get("p95_ttft_ms")),
                    format_num(r.get("p50_latency_ms")),
                    format_num(r.get("chunks_per_s")),
                    format_num(r.get("error_rate")),
                    format_delta(b_p50, r.get("p50_latency_ms")),
                ]
            )
        )

    print(
        "BEST,knob_mask={mask},p50_latency_ms={p50},delta_p50_ms={delta}".format(
            mask=best_row["knob_mask"],
            p50=format_num(best_row.get("p50_latency_ms")),
            delta=format_delta(b_p50, best_row.get("p50_latency_ms")),
        )
    )

    def line(label: str, b: Any, v: Any) -> str:
        return f"{label}: {format_num(b)} -> {format_num(v)} ({format_delta(b, v)})"

    print("BEST_VS_BASELINE")
    print(line("p95_latency_ms", baseline_metrics.get("p95_latency_ms"), best_row.get("p95_latency_ms")))
    print(line("p99_latency_ms", baseline_metrics.get("p99_latency_ms"), best_row.get("p99_latency_ms")))
    print(line("p95_ttft_ms", baseline_metrics.get("p95_ttft_ms"), best_row.get("p95_ttft_ms")))
    print(line("p50_latency_ms", baseline_metrics.get("p50_latency_ms"), best_row.get("p50_latency_ms")))
    print(line("chunks_per_s", baseline_metrics.get("chunks_per_s"), best_row.get("chunks_per_s")))
    print("")
    print(line("gpu_compute_ms", baseline_trace.get("gpu_compute_ms"), best_trace.get("gpu_compute_ms")))
    print(line("memcpy_ms", baseline_trace.get("memcpy_ms"), best_trace.get("memcpy_ms")))
    print(line("osrt_wait_ms", baseline_trace.get("osrt_wait_ms"), best_trace.get("osrt_wait_ms")))
    print("")
    print(line("nvtx_get_batch_ms", baseline_nvtx.get("get_batch_ms"), best_nvtx.get("get_batch_ms")))
    print(line("nvtx_prefill_ms", baseline_nvtx.get("prefill_ms"), best_nvtx.get("prefill_ms")))
    print(line("nvtx_decode_ms", baseline_nvtx.get("decode_ms"), best_nvtx.get("decode_ms")))
    print("")

    print("TOP_KERNELS_BASELINE")
    b_top = baseline_trace.get("top_kernels", [])
    for i in range(5):
        if i < len(b_top):
            item = b_top[i]
            print(f"{i+1}) {item.get('name','na')},{format_num(item.get('total_ms'))},{format_num(item.get('percent'))}")
        else:
            print(f"{i+1}) na,na,na")

    print("")
    print("TOP_KERNELS_BEST")
    v_top = best_trace.get("top_kernels", [])
    for i in range(5):
        if i < len(v_top):
            item = v_top[i]
            print(f"{i+1}) {item.get('name','na')},{format_num(item.get('total_ms'))},{format_num(item.get('percent'))}")
        else:
            print(f"{i+1}) na,na,na")


def write_summary_csv(path: Path, rows: List[Dict[str, Any]], baseline_p50: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "knob_mask", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "p95_ttft_ms", "chunks_per_s", "error_rate", "delta_p50_ms"])
        for r in rows:
            w.writerow(
                [
                    r["trial"],
                    r["knob_mask"],
                    r.get("p50_latency_ms"),
                    r.get("p95_latency_ms"),
                    r.get("p99_latency_ms"),
                    r.get("p95_ttft_ms"),
                    r.get("chunks_per_s"),
                    r.get("error_rate"),
                    (None if baseline_p50 is None or r.get("p50_latency_ms") is None else (float(r["p50_latency_ms"]) - float(baseline_p50))),
                ]
            )


class BruteForceSearch:
    def __init__(self, cfg: Dict[str, Any], base_dir: Path) -> None:
        self.cfg = cfg
        self.base_dir = base_dir

    def run(self, baseline_metrics: Dict[str, Optional[float]]) -> None:
        knobs = self.cfg.get("knobs", [])
        n = len(knobs)
        rows: List[Dict[str, Any]] = []
        all_trials: List[Dict[str, Any]] = []

        for trial, bits in enumerate(itertools.product([0, 1], repeat=n)):
            mask, flags, assignment = get_knob_flags(knobs, bits)
            trial_dir = self.base_dir / "trials" / f"trial_{trial:04d}_{mask}"
            result = evaluate(self.cfg, trial_dir, flags, assignment, profile=False)
            row = {
                "trial": trial,
                "knob_mask": mask,
                "assignment": assignment,
                "flags": flags,
                "p50_latency_ms": result.metrics.get("p50_latency_ms"),
                "p95_latency_ms": result.metrics.get("p95_latency_ms"),
                "p99_latency_ms": result.metrics.get("p99_latency_ms"),
                "p50_ttft_ms": result.metrics.get("p50_ttft_ms"),
                "p95_ttft_ms": result.metrics.get("p95_ttft_ms"),
                "p99_ttft_ms": result.metrics.get("p99_ttft_ms"),
                "chunks_per_s": result.metrics.get("chunks_per_s"),
                "error_rate": result.metrics.get("error_rate"),
            }
            rows.append(row)
            all_trials.append(
                {
                    "trial": trial,
                    "knob_mask": mask,
                    "assignment": assignment,
                    "flags": flags,
                    "metrics": result.metrics,
                    "run_dir": str(trial_dir),
                    "resolved_server_cmd": result.resolved_cmd,
                    "resolved_env": result.resolved_env,
                }
            )

        best_row = choose_best(rows)
        best_trial = next((trial for trial in all_trials if trial.get("trial") == best_row.get("trial")), None)
        if best_trial is None:
            raise SystemExit("Failed to resolve best trial metadata")

        best_metrics = best_trial["metrics"]
        best_run_dir = best_trial["run_dir"]
        best_flags = best_trial["flags"]
        best_assignment = best_trial["assignment"]

        best_config = {
            "knob_mask": best_row["knob_mask"],
            "assignment": best_assignment,
            "flags": best_flags,
            "metrics": best_metrics,
            "resolved_server_cmd": best_trial.get("resolved_server_cmd", []),
            "resolved_env": {
                k: v
                for k, v in (best_trial.get("resolved_env", {}) or {}).items()
                if k in (self.cfg.get("env") or {}) or k in ((self.cfg.get("server") or {}).get("env") or {})
            },
        }

        import shutil

        results_root = Path("results")
        baseline_dir = results_root / "baseline"
        best_dir = results_root / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(Path(best_run_dir) / "client_raw.json", best_dir / "client_raw.json")
        shutil.copy(Path(best_run_dir) / "client_metrics.json", best_dir / "client_metrics.json")
        best_profile_dir = best_dir / "profile_tmp"
        evaluate(self.cfg, best_profile_dir, best_flags, best_assignment, profile=True)
        shutil.copy(best_profile_dir / "trace_summary.json", best_dir / "trace_summary.json")
        shutil.copy(best_profile_dir / "nvtx_phases.json", best_dir / "nvtx_phases.json")

        write_summary_csv(self.base_dir / "summary.csv", rows, baseline_metrics.get("p50_latency_ms"))
        write_json(
            self.base_dir / "summary.json",
            {
                "objective": {"primary": "p95_latency_ms", "tie_breakers": ["p99_latency_ms", "p95_ttft_ms", "p50_latency_ms", "chunks_per_s"]},
                "trials": all_trials,
                "best": best_config,
            },
        )
        write_json(self.base_dir / "best_config.json", best_config)
        (self.base_dir / "best_command.sh").write_text(
            "#!/usr/bin/env bash\n"
            + " ".join(shlex.quote(x) for x in best_config["resolved_server_cmd"])
            + "\n"
        )

        baseline_metrics_live = load_json(baseline_dir / "client_metrics.json")
        baseline_trace = load_json(baseline_dir / "trace_summary.json") if (baseline_dir / "trace_summary.json").exists() else {"gpu_compute_ms": None, "memcpy_ms": None, "osrt_wait_ms": None, "top_kernels": []}
        baseline_nvtx = load_json(baseline_dir / "nvtx_phases.json") if (baseline_dir / "nvtx_phases.json").exists() else {"get_batch_ms": None, "prefill_ms": None, "decode_ms": None}
        best_trace = load_json(best_dir / "trace_summary.json") if (best_dir / "trace_summary.json").exists() else {"gpu_compute_ms": None, "memcpy_ms": None, "osrt_wait_ms": None, "top_kernels": []}
        best_nvtx = load_json(best_dir / "nvtx_phases.json") if (best_dir / "nvtx_phases.json").exists() else {"get_batch_ms": None, "prefill_ms": None, "decode_ms": None}

        print_search_output(
            rows=rows,
            baseline_metrics=baseline_metrics_live,
            best_row=best_metrics | {"knob_mask": best_row["knob_mask"]},
            baseline_trace=baseline_trace,
            best_trace=best_trace,
            baseline_nvtx=baseline_nvtx,
            best_nvtx=best_nvtx,
        )


def run_baseline(cfg: Dict[str, Any]) -> None:
    results_root = Path("results")
    baseline_dir = results_root / "baseline"
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

    results_root = Path("results")
    baseline_dir = results_root / "baseline"
    if not baseline_dir.exists():
        run_baseline(cfg)
        return

    search_dir = Path("results/search/latest")
    search_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_json(baseline_dir / "client_metrics.json")
    BruteForceSearch(cfg, search_dir).run(baseline_metrics)


if __name__ == "__main__":
    main()
