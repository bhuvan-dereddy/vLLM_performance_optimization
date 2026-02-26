import json, os, subprocess
from pathlib import Path

RUN_ID = os.environ.get("RUN_ID", "").strip()
if not RUN_ID:
    raise SystemExit("Set RUN_ID env var, e.g. RUN_ID=20260224_092432_hi3qo8")

base = Path("runs") / RUN_ID
cfg = json.load(open(base / "config_resolved.json"))
sweep_def = cfg["resolved_sweep_definition"]

PROMPTS = os.environ.get("PROMPTS", "data/sharegpt_prompts_prefill_1850_1900.jsonl")
MODEL_DIR = os.environ.get("MODEL_DIR", "models/tinyllama")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
DTYPE = os.environ.get("DTYPE", "float16")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "25"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "180"))
STARTUP_TIMEOUT_S = int(os.environ.get("STARTUP_TIMEOUT_S", "240"))
START_TRIAL = int(os.environ.get("START_TRIAL", "0"))

def server_extra_from_assignment(a: dict):
    extra = ["--attention-backend", "FLASHINFER"]  # stable on T4

    if a.get("enable_chunked_prefill") is True:
        extra += ["--enable-chunked-prefill"]
    elif a.get("enable_chunked_prefill") is False:
        extra += ["--no-enable-chunked-prefill"]

    if a.get("enforce_eager") is True:
        extra += ["--enforce-eager"]

    if "max_num_batched_tokens" in a:
        extra += ["--max-num-batched-tokens", str(a["max_num_batched_tokens"])]

    if a.get("enable_prefix_caching") is True:
        extra += ["--enable-prefix-caching"]
    elif a.get("enable_prefix_caching") is False:
        extra += ["--no-enable-prefix-caching"]

    if "block_size" in a:
        extra += ["--block-size", str(a["block_size"])]

    if a.get("disable_log_requests") is True:
        extra += ["--disable-log-requests"]

    if a.get("disable_log_stats") is True:
        extra += ["--disable-log-stats"]

    return extra

def run_one(trial_id: int, assignment: dict):
    trial_dir = base / "sweep" / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    metrics = trial_dir / "client_raw.json"
    if metrics.exists():
        return "skip_done"

    extra_json = json.dumps(server_extra_from_assignment(assignment))
    gpu_mem = float(assignment.get("gpu_memory_utilization", 0.9))

    cmd = [
        "python", "scripts/run_once.py",
        "--run-dir", str(trial_dir),
        "--model-dir", MODEL_DIR,
        "--host", HOST,
        "--port", str(PORT),
        "--dtype", DTYPE,
        "--gpu-mem-util", str(gpu_mem),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--prompts", PROMPTS,
        "--num-requests", str(NUM_REQUESTS),
        "--concurrency", str(CONCURRENCY),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--temperature", "0.0",
        "--timeout-s", str(TIMEOUT_S),
        "--server-startup-timeout-s", str(STARTUP_TIMEOUT_S),
        "--server-extra", extra_json,
    ]

    print(f"\n=== TRIAL {trial_id:03d} gpu_mem={gpu_mem} ===")
    print("server_extra:", json.loads(extra_json))
    subprocess.run(cmd, check=False)
    return "ran"

total = len(sweep_def)
print("total trials in resolved_sweep_definition:", total)
print("starting from trial:", START_TRIAL)

ran = skipped = 0
for i in range(START_TRIAL, total):
    assignment = sweep_def[i].get("assignment", {})
    r = run_one(i, assignment)
    if r == "skip_done":
        skipped += 1
    else:
        ran += 1

print("\nDONE. skipped_done=", skipped, "ran_now=", ran)
