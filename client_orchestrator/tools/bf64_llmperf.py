#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

# ============ USER CONFIG ============
GPU_SSH_HOST = "gpu-profiling-ec2"   # ssh alias on cheap EC2
GPU_PUBLIC_IP = "100.31.137.86"
PORT = 8000

# llmperf (cheap EC2)
HOME = Path.home()
LLMPERF_DIR = HOME / "llmperf"
CONFIG_PATH = HOME / "optimizations_6knobs.json"
RESULTS_ROOT = HOME / "bf64_results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

BASE_URL = f"http://{GPU_PUBLIC_IP}:{PORT}/v1"

# Deterministic workload
SEED = 42
CONCURRENCY = 1
REQS = 2000
TIMEOUT = 1800
MEAN_IN = 512
STD_IN = 0
MEAN_OUT = 256
STD_OUT = 0
TEMPERATURE = 0.0
# =====================================

def run(cmd, capture=False, env=None):
    if capture:
        return subprocess.check_output(cmd, text=True, env=env)
    subprocess.check_call(cmd, env=env)
    return ""

def ssh(args_list, capture=False):
    # Always arg-list to avoid quoting issues
    cmd = ["ssh", GPU_SSH_HOST] + args_list
    return run(cmd, capture=capture)

def wait_for_server(timeout_s=240):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.check_output(
                ["bash", "-lc", f"curl -s {BASE_URL}/models | head -c 200"],
                text=True,
            )
            if "models/tinyllama" in out or '"data"' in out:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def load_config():
    return json.loads(CONFIG_PATH.read_text())

def mask_to_flags(mask, knobs):
    flags = []
    for i, knob in enumerate(knobs):
        if (mask >> i) & 1:
            flags.extend(knob.get("on_flags", []))
        else:
            flags.extend(knob.get("off_flags", []))
    return flags

def restart_vllm(server_cfg, extra_flags, label, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)

    # Start via remote helper script (reliable detach)
    # We pass model_dir exactly as in config (models/tinyllama) and flags after.
    out = ssh(
        ["/home/ubuntu/start_vllm.sh", server_cfg["model_dir"], str(server_cfg.get("port", PORT))] + extra_flags,
        capture=True,
    ).strip()

    # Script prints log path
    (run_dir / "server_log_path.txt").write_text(out + "\n")
    (run_dir / "server_flags.txt").write_text(" ".join(extra_flags) + "\n")

    if not wait_for_server(server_cfg.get("startup_timeout_s", 240)):
        # Pull tail of server log for debugging
        tail = ssh(["bash", "-lc", f"tail -n 200 '{out}' || true"], capture=True)
        (run_dir / "server_log_tail.txt").write_text(tail)
        raise RuntimeError(f"vLLM failed to become ready for {label}. See {run_dir}/server_log_tail.txt")

def run_llmperf(run_dir: Path, label: str):
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = BASE_URL
    env["OPENAI_API_KEY"] = "EMPTY"
    env["PYTHONHASHSEED"] = "0"

    cmd = [
        "python3", str(LLMPERF_DIR / "token_benchmark_ray.py"),
        "--llm-api", "openai",
        "--model", "models/tinyllama",
        "--seed", str(SEED),
        "--num-concurrent-requests", str(CONCURRENCY),
        "--max-num-completed-requests", str(REQS),
        "--timeout", str(TIMEOUT),
        "--mean-input-tokens", str(MEAN_IN),
        "--stddev-input-tokens", str(STD_IN),
        "--mean-output-tokens", str(MEAN_OUT),
        "--stddev-output-tokens", str(STD_OUT),
        "--additional-sampling-params", f'{{"temperature":{TEMPERATURE}}}',
        "--metadata", f"label={label}",
        "--results-dir", str(run_dir),
    ]

    with open(run_dir / "client.log", "w") as f:
        subprocess.check_call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

def read_summary(run_dir: Path):
    summaries = list(run_dir.glob("models-*_summary.json"))
    if not summaries:
        raise RuntimeError(f"No summary json in {run_dir}")
    s = json.loads(summaries[0].read_text())

    # llmperf summary schema: ttft_s / overall output throughput is printed as "Overall Output Throughput"
    # Your earlier file had: models-tinyllama_550_150_summary.json with ttft_s
    return {
        "ttft_p95": s["ttft_s"]["p95"],
        "ttft_p50": s["ttft_s"]["p50"],
        "itl_p50": s["inter_token_latency_s"]["p50"],
        "e2e_p95": s["end_to_end_latency_s"]["p95"],
        "throughput": s.get("overall_output_throughput_token_per_s", s.get("overall_output_throughput", 0.0)),
        "errors": s.get("num_errored_requests", s.get("Number Of Errored Requests", 0)),
        "summary_path": str(summaries[0]),
    }

def main():
    cfg = load_config()
    server_cfg = cfg["server"]
    knobs = cfg["knobs"]

    results = []
    baseline = None

    for mask in range(64):
        mask_str = format(mask, "06b")
        label = f"mask_{mask_str}"
        run_dir = RESULTS_ROOT / label
        run_dir.mkdir(parents=True, exist_ok=True)

        flags = mask_to_flags(mask, knobs)
        print(f"\n=== TRIAL {label} ===")
        print("  flags:", " ".join(flags) if flags else "(none)")

        restart_vllm(server_cfg, flags, label, run_dir)
        run_llmperf(run_dir, label)

        metrics = read_summary(run_dir)
        metrics["mask"] = mask_str
        metrics["flags"] = flags
        results.append(metrics)

        print(f"  p95 TTFT: {metrics['ttft_p95']:.6f}s  throughput: {metrics['throughput']}")

        if mask == 0:
            baseline = metrics

    valid = [r for r in results if int(r.get("errors", 0)) == 0]
    valid.sort(key=lambda r: (r["ttft_p95"], -float(r["throughput"])))

    best = valid[0] if valid else None

    (RESULTS_ROOT / "summary_all.json").write_text(
        json.dumps({"baseline": baseline, "best": best, "all": results}, indent=2)
    )

    with open(RESULTS_ROOT / "summary.csv", "w") as f:
        f.write("mask,ttft_p95,ttft_p50,e2e_p95,itl_p50,throughput,errors,summary_path\n")
        for r in results:
            f.write(
                f"{r['mask']},{r['ttft_p95']},{r['ttft_p50']},{r['e2e_p95']},{r['itl_p50']},{r['throughput']},{r['errors']},{r['summary_path']}\n"
            )

    print("\n===== BEST =====")
    print(best)

    if baseline and best:
        print("\nΔ p95 TTFT (baseline - best):", baseline["ttft_p95"] - best["ttft_p95"])
        print("Best flags:", " ".join(best["flags"]) if best["flags"] else "(none)")

if __name__ == "__main__":
    main()
