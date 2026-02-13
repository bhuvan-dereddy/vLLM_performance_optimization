from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import subprocess
import time

from gpuprof.config import get_concurrency, get_endpoint, get_num_requests, get_prompts_path, get_temperature, get_timeout_s
from gpuprof.process import wait_for_ready, terminate_process_group, wait_for_report_finalized
from gpuprof.server_cmd import resolve_server, _arg_value, resolve_model_name
from gpuprof.trace_parse import find_nsys, parse_trace_sqlite
from gpuprof.utils import load_json, write_json


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

            prompts = get_prompts_path(cfg)
            endpoint = get_endpoint(cfg)
            num_requests = get_num_requests(cfg)

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
                str(get_concurrency(cfg)),
                "--max-new-tokens",
                str(client_max_new_tokens),
                "--max-model-len",
                str(client_max_model_len),
                "--max-input-tokens",
                str(client_max_input_tokens),
                "--temperature",
                str(get_temperature(cfg)),
                "--timeout-s",
                str(get_timeout_s(cfg)),
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
