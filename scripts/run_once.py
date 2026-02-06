from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def http_get(url: str, timeout_s: float = 2.0) -> Tuple[int, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            code = resp.getcode()
            body = resp.read(4096).decode("utf-8", errors="ignore")
            return code, body
    except Exception as e:
        return 0, str(e)


def wait_for_ready(host: str, port: int, timeout_s: float) -> Tuple[bool, str]:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    last = ""
    while time.time() < deadline:
        code, body = http_get(url, timeout_s=2.0)
        last = f"code={code} body={body[:200]}"
        if code == 200:
            return True, last
        time.sleep(0.5)
    return False, last


def terminate_process_tree(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


@dataclass
class RunResult:
    ok: bool
    reason: str
    server_cmd: List[str]
    server_stdout_tail: str
    server_stderr_tail: str
    client_stats: Dict[str, Any]


def tail_text(path: Path, max_bytes: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="ignore")
    return data[-max_bytes:].decode("utf-8", errors="ignore")


def run_nsight_profile(
    run_dir: Path,
    model_dir: str,
    host: str,
    port: int,
    dtype: str,
    gpu_mem_util: float,
    max_model_len: int,
    prompts: str,
    num_requests: int,
    concurrency: int,
    max_new_tokens: int,
    temperature: float,
    extra_flags: List[str],
) -> Tuple[bool, str]:
    """
    Run inference with Nsight Systems profiling and extract NVTX data
    Returns: (success, error_message)
    """
    nsys_path = "/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys"
    
    if not Path(nsys_path).exists():
        return False, f"Nsight Systems not found at {nsys_path}"
    
    # Create a simple profiling script
    profile_script = run_dir / "profile_run.py"
    profile_script.write_text(f'''
from __future__ import annotations
import time
from pathlib import Path
from contextlib import contextmanager
import ctypes
from vllm import LLM, SamplingParams

try:
    import nvtx
except Exception:
    nvtx = None

@contextmanager
def nvtx_range(name: str):
    if nvtx is None:
        yield
        return
    nvtx.push_range(name)
    try:
        yield
    finally:
        nvtx.pop_range()

def _load_cudart():
    candidates = ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None

_CUDART = _load_cudart()

def cuda_profiler_start():
    if _CUDART is None:
        return
    try:
        _CUDART.cudaProfilerStart()
    except Exception:
        pass

def cuda_profiler_stop():
    if _CUDART is None:
        return
    try:
        _CUDART.cudaProfilerStop()
    except Exception:
        pass

def main():
    model_dir = "{model_dir}"
    
    with nvtx_range("phase:model_init"):
        llm = LLM(
            model=str(model_dir),
            dtype="{dtype}",
            tensor_parallel_size=1,
            gpu_memory_utilization={gpu_mem_util},
            max_model_len={max_model_len},
            trust_remote_code=False,
        )
    
    warmup_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    run_params = SamplingParams(
        temperature={temperature},
        top_p=1.0,
        max_tokens={max_new_tokens}
    )
    
    # Load prompts
    import json
    prompts = []
    with open("{prompts}", "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            prompts.append(obj["prompt"])
            if len(prompts) >= {num_requests}:
                break
    
    with nvtx_range("phase:warmup"):
        _ = llm.generate(["Warmup"], warmup_params)
    
    cuda_profiler_start()
    try:
        with nvtx_range("phase:inference"):
            t0 = time.time()
            outputs = llm.generate(prompts, run_params)
            t1 = time.time()
    finally:
        cuda_profiler_stop()
    
    print(f"Inference time: {{(t1-t0):.3f}}s")

if __name__ == "__main__":
    main()
''')
    
    nsys_out = run_dir / "profile_capture"
    
    # Run nsys profile
    nsys_cmd = [
        nsys_path, "profile",
        "--force-overwrite=true",
        "--trace=osrt,cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o", str(nsys_out),
        "python", str(profile_script),
    ]
    
    nsys_log = run_dir / "nsys_profile.log"
    
    try:
        with nsys_log.open("w") as f:
            result = subprocess.run(
                nsys_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=600,  # 10 min timeout
                env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn"),
            )
        
        if result.returncode != 0:
            return False, f"nsys profile failed with code {result.returncode}"
        
        # Export to SQLite
        nsys_rep = Path(str(nsys_out) + ".nsys-rep")
        if not nsys_rep.exists():
            return False, f"Expected .nsys-rep not found: {nsys_rep}"
        
        sqlite_out = run_dir / "profile_capture.sqlite"
        export_cmd = [
            nsys_path, "stats",
            "--force-overwrite=true",
            f"--output={run_dir}/profile_capture",
            str(nsys_rep),
        ]
        
        result = subprocess.run(
            export_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        
        if result.returncode != 0 or not sqlite_out.exists():
            return False, "SQLite export failed"
        
        # Parse SQLite to extract NVTX summary
        parse_cmd = [
            "python", "parse/parse_sqlite_baseline.py",
            str(sqlite_out),
            "--out", str(run_dir / "nvtx_summary.json"),
        ]
        
        result = subprocess.run(
            parse_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        
        if result.returncode != 0:
            return False, "NVTX parsing failed"
        
        return True, "profiling_complete"
        
    except subprocess.TimeoutExpired:
        return False, "profiling_timeout"
    except Exception as e:
        return False, f"profiling_error: {str(e)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)

    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)

    ap.add_argument("--prompts", required=True)
    ap.add_argument("--num-requests", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", dest="timeout_s", type=float, default=180.0)

    ap.add_argument("--server-startup-timeout-s", type=float, default=240.0)
    ap.add_argument("--server-extra", default="[]")
    ap.add_argument("--enable-nsight-profiling", action="store_true")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    server_stdout = run_dir / "server_stdout.log"
    server_stderr = run_dir / "server_stderr.log"
    client_out = run_dir / "client_raw.json"
    client_summary = run_dir / "client_summary.json"

    try:
        extra_flags = json.loads(args.server_extra)
        if not isinstance(extra_flags, list):
            raise ValueError("server-extra must be a JSON list")
        extra_flags = [str(x) for x in extra_flags]
    except Exception as e:
        raise SystemExit(f"Bad --server-extra: {e}")

    ok = False
    reason = ""
    client_stats: Dict[str, Any] = {}

    # If profiling is enabled, use the profiling path (no separate server)
    if args.enable_nsight_profiling:
        prof_ok, prof_msg = run_nsight_profile(
            run_dir=run_dir,
            model_dir=args.model_dir,
            host=args.host,
            port=args.port,
            dtype=args.dtype,
            gpu_mem_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            prompts=args.prompts,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            extra_flags=extra_flags,
        )
        
        if prof_ok:
            # Profiling completed, but we still need client metrics
            # For now, mark as successful - in production you'd run bench_client too
            ok = True
            reason = prof_msg
            # Set dummy client stats (in real scenario, run client benchmark)
            client_stats = {
                "p50_total_ms": 0.0,
                "p95_total_ms": 0.0,
                "p50_ttft_ms": 0.0,
                "p95_ttft_ms": 0.0,
                "throughput_tok_s": 0.0,
                "wall_s": 0.0,
                "note": "profiling_mode_no_client_benchmark"
            }
        else:
            ok = False
            reason = prof_msg
        
        summary = RunResult(
            ok=ok,
            reason=reason,
            server_cmd=["nsight_profiling_mode"],
            server_stdout_tail="",
            server_stderr_tail="",
            client_stats=client_stats,
        )
        client_summary.write_text(json.dumps(summary.__dict__, indent=2))
        return

    # Normal mode: Start server and run client benchmark
    server_cmd = [
        "vllm", "serve", args.model_dir,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--max-model-len", str(args.max_model_len),
    ] + extra_flags

    with server_stdout.open("wb") as out_f, server_stderr.open("wb") as err_f:
        proc = subprocess.Popen(
            server_cmd,
            stdout=out_f,
            stderr=err_f,
            preexec_fn=os.setsid,
        )

    try:
        ready, ready_msg = wait_for_ready(args.host, args.port, args.server_startup_timeout_s)
        if not ready:
            reason = f"server_not_ready: {ready_msg}"
            return

        cmd = [
            "python", "scripts/bench_client.py",
            "--host", args.host,
            "--port", str(args.port),
            "--endpoint", "/v1/chat/completions",
            "--model", args.model_dir,
            "--prompts", args.prompts,
            "--num-requests", str(args.num_requests),
            "--concurrency", str(args.concurrency),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--timeout-s", str(args.timeout_s),
            "--out", str(client_out),
        ]
        subprocess.check_call(cmd)

        raw = json.loads(client_out.read_text())
        client_stats = raw.get("stats", {})
        runmeta = raw.get("run", {})
        ok_requests = runmeta.get("ok_requests", 0)

        if ok_requests and ok_requests > 0:
            ok = True
            reason = "ok"
        else:
            ok = False
            reason = f"client_no_ok_requests: {raw.get('failures_sample', [])[:3]}"

    finally:
        terminate_process_tree(proc, grace_s=12.0)

        summary = RunResult(
            ok=ok,
            reason=reason,
            server_cmd=server_cmd,
            server_stdout_tail=tail_text(server_stdout),
            server_stderr_tail=tail_text(server_stderr),
            client_stats=client_stats,
        )
        client_summary.write_text(json.dumps(summary.__dict__, indent=2))

        print(f"✅ Wrote {client_summary}")
        print(json.dumps(client_stats, indent=2))


if __name__ == "__main__":
    main()
