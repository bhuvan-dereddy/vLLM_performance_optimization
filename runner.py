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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import request

from gpuprof.process import http_get, wait_for_ready, terminate_process_group, wait_for_report_finalized
from gpuprof.trace_parse import find_nsys, parse_trace_sqlite
from gpuprof.utils import load_json, write_json, format_num, format_delta


def resolve_server(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], str, int, float]:
    if "serve_cmd" in cfg:
        raw = cfg["serve_cmd"]
        if isinstance(raw, str):
            serve_cmd = shlex.split(raw)
        elif isinstance(raw, list):
            serve_cmd = [str(x) for x in raw]
        else:
            raise SystemExit("serve_cmd must be string or list")
    else:
        server = cfg.get("server", {})
        serve_cmd = [
            "vllm",
            "serve",
            str(server["model_dir"]),
            "--host",
            str(server.get("host", "127.0.0.1")),
            "--port",
            str(server.get("port", 8000)),
            "--dtype",
            str(server.get("dtype", "float16")),
            "--gpu-memory-utilization",
            str(server.get("gpu_memory_utilization", 0.9)),
            "--max-model-len",
            str(server.get("max_model_len", 2048)),
        ]

    env = dict(os.environ)
    env_cfg = cfg.get("env", {})
    if not env_cfg and isinstance(cfg.get("server"), dict):
        env_cfg = cfg["server"].get("env", {})
    for k, v in (env_cfg or {}).items():
        env[str(k)] = str(v)

    host = cfg.get("workload", {}).get("host")
    port = cfg.get("workload", {}).get("port")
    timeout_s = cfg.get("server", {}).get("startup_timeout_s", 240)

    if host is None:
        host = _arg_value(serve_cmd, "--host", default="127.0.0.1")
    if port is None:
        port = int(_arg_value(serve_cmd, "--port", default="8000"))

    return serve_cmd, env, str(host), int(port), float(timeout_s)


def _arg_value(cmd: Sequence[str], key: str, default: str) -> str:
    for i, tok in enumerate(cmd):
        if tok == key and i + 1 < len(cmd):
            return cmd[i + 1]
    return default


def resolve_model_name(cmd: Sequence[str], cfg: Dict[str, Any]) -> str:
    if len(cmd) >= 3 and cmd[0] == "vllm" and cmd[1] == "serve":
        return str(cmd[2])
    if "--model" in cmd:
        return _arg_value(cmd, "--model", default="models/tinyllama")
    server = cfg.get("server", {})
    if "model_dir" in server:
        return str(server["model_dir"])
    return "models/tinyllama"


def get_knob_flags(knobs: List[Dict[str, Any]], bits: Sequence[int]) -> Tuple[str, List[str], Dict[str, bool]]:
    mask = "".join(str(int(b)) for b in bits)
    flags: List[str] = []
    assignment: Dict[str, bool] = {}
    for b, knob in zip(bits, knobs):
        name = str(knob["name"])
        on_flags = [str(x) for x in knob.get("on_flags", [])]
        off_flags = [str(x) for x in knob.get("off_flags", [])]
        if not on_flags and "flag" in knob and "on_value" in knob:
            on_flags = [str(knob["flag"]), str(knob["on_value"])]
        elif not on_flags and "flag" in knob and knob.get("type") == "bool":
            on_flags = [str(knob["flag"])]
        assignment[name] = bool(b)
        flags.extend(on_flags if b else off_flags)
    return mask, flags, assignment


def parse_client_metrics(client_raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    stats = client_raw.get("stats") or {}
    total = int(client_raw.get("num_requests", 0) or 0)
    failed = int(client_raw.get("failed_requests", 0) or 0)
    error_rate: Optional[float]
    if total > 0:
        error_rate = failed / float(total)
    else:
        error_rate = None

    def as_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    return {
        "p50_latency_ms": as_float(stats.get("p50_total_ms")),
        "p95_latency_ms": as_float(stats.get("p95_total_ms")),
        "p99_latency_ms": as_float(stats.get("p99_total_ms")),
        "p50_ttft_ms": as_float(stats.get("p50_ttft_ms")),
        "p95_ttft_ms": as_float(stats.get("p95_ttft_ms")),
        "p99_ttft_ms": as_float(stats.get("p99_ttft_ms")),
        "chunks_per_s": as_float(stats.get("throughput_chunks_s")),
        "error_rate": error_rate,
    }


@dataclass
class EvaluationResult:
    run_dir: Path
    ok: bool
    metrics: Dict[str, Optional[float]]
    nvtx_phases: Dict[str, Optional[float]]
    trace_summary: Dict[str, Any]
    resolved_cmd: List[str]
    resolved_env: Dict[str, str]


def evaluate(
    cfg: Dict[str, Any],
    run_dir: Path,
    knob_flags: List[str],
    assignment: Dict[str, bool],
    profile: bool,
) -> EvaluationResult:
    run_dir.mkdir(parents=True, exist_ok=True)

    serve_cmd, env, host, port, startup_timeout_s = resolve_server(cfg)
    resolved_cmd = serve_cmd + knob_flags

    logs_path = run_dir / "logs.txt"
    client_raw_path = run_dir / "client_raw.json"

    nsys_cmd_prefix: List[str] = []
    if profile:
        nsys = find_nsys()
        trace = cfg.get("nsys", {}).get("trace", "cuda,nvtx,osrt")
        trace_prefix = run_dir / "trace"
        nsys_cmd_prefix = [
            nsys,
            "profile",
            "--force-overwrite=true",
            f"--trace={trace}",
            "--sample=none",
            "--cpuctxsw=none",
            "-o",
            str(trace_prefix),
        ]

    proc: Optional[subprocess.Popen] = None
    ok = False
    metrics = {
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "p50_ttft_ms": None,
        "p95_ttft_ms": None,
        "p99_ttft_ms": None,
        "chunks_per_s": None,
        "error_rate": 1.0,
    }
    nvtx_phases = {"get_batch_ms": None, "prefill_ms": None, "decode_ms": None}
    trace_summary = {
        "gpu_compute_ms": None,
        "memcpy_ms": None,
        "osrt_wait_ms": None,
        "top_kernels": [],
    }

    with logs_path.open("wb") as logs_f:
        proc = subprocess.Popen(
            nsys_cmd_prefix + resolved_cmd,
            stdout=logs_f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        workload = cfg.get("workload", {})
        if "--max-model-len" in resolved_cmd:
            client_max_model_len = int(_arg_value(resolved_cmd, "--max-model-len", "2048"))
        else:
            client_max_model_len = int(cfg.get("server", {}).get("max_model_len", 2048))
        client_max_new_tokens = int(workload.get("max_new_tokens", 64))
        if "max_input_tokens" in workload:
            client_max_input_tokens = int(workload["max_input_tokens"])
        else:
            client_max_input_tokens = max(1, client_max_model_len - client_max_new_tokens - 32)

        try:
            if not wait_for_ready(host, port, startup_timeout_s):
                raise RuntimeError("server_not_ready")

            dataset = cfg.get("dataset", {})
            prompts = dataset.get("path") or dataset.get("prompts_jsonl")
            if not prompts:
                raise RuntimeError("dataset.path or dataset.prompts_jsonl missing")

            endpoint = workload.get("endpoint", "/v1/chat/completions")
            num_requests = int(workload.get("request_count", workload.get("num_requests", 0)))
            if num_requests <= 0:
                num_requests = int(cfg.get("experiment", {}).get("final_requests", 0))
            if num_requests <= 0:
                num_requests = int(cfg.get("experiment", {}).get("screening_requests", 20))

            client_cmd = [
                "python",
                "scripts/bench_client.py",
                "--host",
                host,
                "--port",
                str(port),
                "--endpoint",
                str(endpoint),
                "--model",
                resolve_model_name(resolved_cmd, cfg),
                "--prompts",
                str(prompts),
                "--num-requests",
                str(num_requests),
                "--concurrency",
                str(workload.get("concurrency", 4)),
                "--max-new-tokens",
                str(client_max_new_tokens),
                "--max-model-len",
                str(client_max_model_len),
                "--max-input-tokens",
                str(client_max_input_tokens),
                "--temperature",
                str(workload.get("temperature", 0.0)),
                "--timeout-s",
                str(workload.get("timeout_s", 180.0)),
                "--out",
                str(client_raw_path),
            ]

            subprocess.run(client_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            raw = load_json(client_raw_path)
            metrics = parse_client_metrics(raw)
            ok = (metrics.get("error_rate") == 0.0)

        except Exception:
            ok = False
        finally:
            if proc is not None:
                terminate_process_group(proc)

    write_json(
        run_dir / "client_metrics.json",
        {
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "p99_latency_ms": metrics.get("p99_latency_ms"),
            "p50_ttft_ms": metrics.get("p50_ttft_ms"),
            "p95_ttft_ms": metrics.get("p95_ttft_ms"),
            "p99_ttft_ms": metrics.get("p99_ttft_ms"),
            "chunks_per_s": metrics.get("chunks_per_s"),
            "error_rate": metrics.get("error_rate"),
        },
    )

    write_json(
        run_dir / "config.json",
        {
            "resolved_server_cmd": resolved_cmd,
            "resolved_env": {k: v for k, v in env.items() if k in (cfg.get("env") or {}) or k in ((cfg.get("server") or {}).get("env") or {})},
            "assignment": assignment,
            "profile": profile,
            "client_effective": {
                "max_model_len": client_max_model_len,
                "max_new_tokens": client_max_new_tokens,
                "max_input_tokens": client_max_input_tokens,
                "safety_buffer_tokens": 32,
            },
        },
    )

    if profile:
        trace_rep = run_dir / "trace.nsys-rep"
        wait_for_report_finalized(trace_rep)

        nsys = find_nsys()
        sqlite_path = run_dir / "trace.sqlite"
        backoffs = [1, 2, 4, 8, 8, 8]
        export_ok = False
        for delay_s in backoffs:
            proc = subprocess.run(
                [nsys, "export", "--type", "sqlite", "--output", str(sqlite_path), str(trace_rep)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            with logs_path.open("ab") as logs_f:
                if proc.stdout:
                    logs_f.write(proc.stdout)
                if proc.stderr:
                    logs_f.write(proc.stderr)

            if proc.returncode == 0 and sqlite_path.exists():
                export_ok = True
                break
            time.sleep(delay_s)

        if not export_ok:
            raise SystemExit(f"Failed to export sqlite for {trace_rep}")

        nvtx_phases, trace_summary = parse_trace_sqlite(sqlite_path)
        write_json(run_dir / "nvtx_phases.json", nvtx_phases)
        write_json(run_dir / "trace_summary.json", trace_summary)

    return EvaluationResult(
        run_dir=run_dir,
        ok=ok,
        metrics=metrics,
        nvtx_phases=nvtx_phases,
        trace_summary=trace_summary,
        resolved_cmd=resolved_cmd,
        resolved_env=env,
    )


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
