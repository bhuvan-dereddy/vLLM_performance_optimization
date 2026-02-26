SPEC.md
Goal
Build a tool that:
1. First run creates a baseline profile + metrics.
2. Second run brute-forces all knob combinations (fast, no nsys), prints a single CSV-like results table, chooses the best config, then profiles only the best with nsys and prints a concise best-vs-baseline diagnostic block.
This is an “automatic optimization chooser” for vLLM serving workloads (no custom CUDA kernels required).

0. Inputs
0.1 Config file
A single JSON config file (default optimizations.json) that defines:
* vLLM server launch command (serve_cmd) and env vars
* workload replay configuration (dataset path, concurrency, request_count, max_new_tokens, endpoint)
* a list of N knobs (N=3 or 4 for now), each boolean “OFF/ON”
* nsys tracing config (trace=cuda,nvtx,osrt)
* objective and constraints
0.2 Knob model (boolean)
Each knob is boolean:
* OFF: knob not applied
* ON: knob applied (either flag OR key/value set to a fixed “on_value”)
Total combinations = 2^N. Include “all OFF” in table.

1. Run Modes (automatic)
1.1 Baseline Mode (first run)
Trigger: results/baseline/ does not exist.
Actions:
* Start vLLM server with baseline config (no knobs).
* Run workload once, wrapped in nsys profile.
* Export sqlite from the .nsys-rep.
* Collect client metrics.
* Parse sqlite to extract:
    * NVTX phase timings (if present): get_batch, prefill, decode
    * GPU summary: gpu_compute_ms, memcpy_ms, osrt_wait_ms
    * Top kernels: top 5 by total time
Artifacts saved to:
results/baseline/
  config.json
  client_metrics.json
  trace.nsys-rep
  trace.sqlite
  nvtx_phases.json
  trace_summary.json
STDOUT (baseline mode):
* Must print exactly one line (no extra text):
BASELINE_DONE
Then exit with code 0.

1.2 Search Mode (second run)
Trigger: results/baseline/ exists.
This mode has two stages.
Stage A — Sweep (no nsys)
Actions:
* Enumerate all 2^N knob combinations.
* For each combination:
    * Start vLLM server with baseline + knob args
    * Run workload WITHOUT nsys
    * Collect client metrics:
        * p50_latency_ms
        * p95_latency_ms (optional but recommended)
        * tokens_per_s (optional but recommended)
        * error_rate (required; must be 0 for valid result)
    * Stop server cleanly
* Save sweep summary as CSV and JSON.
Stage B — Profile best only (nsys)
Actions:
* Choose the best valid combination:
    * Reject any combo where error_rate > 0
    * Primary objective: minimize p50_latency_ms
    * Tie-breakers (in order):
        1. lower p95_latency_ms
        2. higher tokens_per_s
* Re-run ONLY the best config with nsys profile
* Export sqlite
* Parse sqlite for:
    * NVTX phase timings (if present): get_batch, prefill, decode
    * GPU summary: gpu_compute_ms, memcpy_ms, osrt_wait_ms
    * Top kernels: top 5 by total time
Artifacts saved to:
results/search/latest/
  summary.csv
  summary.json
  best_config.json
  best_command.sh
  best/
    config.json
    client_metrics.json
    trace.nsys-rep
    trace.sqlite
    nvtx_phases.json
    trace_summary.json

2. Output Format Requirements (STRICT)
2.1 Search Mode STDOUT must contain ONLY:
1. One CSV-like table header + rows, exactly this schema:
trial,knob_mask,p50_latency_ms,p95_latency_ms,tokens_per_s,error_rate,delta_p50_ms
* trial: integer starting at 0
* knob_mask: N-bit string (e.g., for N=4: 0110)
* delta_p50_ms: p50_latency_ms - baseline_p50_latency_ms
1. One BEST line:
BEST,knob_mask=<mask>,p50_latency_ms=<value>,delta_p50_ms=<value>
1. One BEST_VS_BASELINE block with exactly these sections/keys:
BEST_VS_BASELINE
p50_latency_ms: <baseline> -> <best> (<delta>)
p95_latency_ms: <baseline> -> <best> (<delta>)
tokens_per_s: <baseline> -> <best> (<delta>)

gpu_compute_ms: <baseline> -> <best> (<delta>)
memcpy_ms: <baseline> -> <best> (<delta>)
osrt_wait_ms: <baseline> -> <best> (<delta>)

nvtx_get_batch_ms: <baseline_or_na> -> <best_or_na> (<delta_or_na>)
nvtx_prefill_ms: <baseline_or_na> -> <best_or_na> (<delta_or_na>)
nvtx_decode_ms: <baseline_or_na> -> <best_or_na> (<delta_or_na>)

TOP_KERNELS_BASELINE
1) <name>,<total_ms>,<percent>
2) ...
3) ...
4) ...
5) ...

TOP_KERNELS_BEST
1) <name>,<total_ms>,<percent>
2) ...
3) ...
4) ...
5) ...
2.2 No other stdout allowed
* No progress bars
* No [i/N]
* No pretty tables
* No logging except the required outputs above

3. Nsys File Requirements (STRICT)
* Generate .nsys-rep only for:
    1. baseline run
    2. best-chosen run
* Do NOT produce .nsys-rep for sweep trials.
* Sweep trials must NOT call nsys profile.

4. Trace Parsing Requirements
Must export sqlite:
nsys export --type sqlite --output <path>.sqlite <path>.nsys-rep
The code must parse sqlite robustly:
* If NVTX ranges exist, compute mean (or total) time for get_batch, prefill, decode
* Always compute:
    * gpu_compute_ms from CUDA kernel durations
    * memcpy_ms from memcpy ops (H2D/D2H/device memcpy)
    * osrt_wait_ms from OS runtime wait/poll (if present)
    * Top 5 kernels by total kernel time
If a metric is unavailable due to missing tables, set it to null and print na in stdout.

5. Reproducibility Requirements
* Workload must be deterministic:
    * fixed dataset file
    * fixed request_count and concurrency
* Include the resolved server cmd + env in each config.json
* Save raw logs for baseline and best:
    * results/.../logs.txt

6. Extensibility for Agentic Selection (future)
Code must implement a stable interface:
* evaluate(config, profile: bool) -> ResultSearch strategy is a module:
* BruteForceSearch (now)
* future AgentSearch (later) reuses evaluate() and stored history
Store sweep history:
* results/search/latest/summary.json must include all trial configs + metrics.

What it does later (agentic roadmap)
After this brute-force version works:
* Replace BruteForceSearch with:
    * Two-stage successive halving (screen with small N, confirm with larger N)
    * Greedy coordinate descent
    * Bayesian optimization
* Keep the same evaluate() function, output formats, trace parsing, and baseline/best profiling rules.

How to use (for README)
* First run:
    * python runner.py --config optimizations.json
    * produces baseline and prints BASELINE_DONE
* Second run:
    * python runner.py --config optimizations.json
    * prints the CSV table + BEST line + BEST_VS_BASELINE block
    * produces best config and best .nsys-rep
