import itertools
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpuprof.evaluate import evaluate
from gpuprof.io_utils import load_json, write_json
from gpuprof.paths import baseline_dir as get_baseline_dir, best_dir as get_best_dir
from gpuprof.search import choose_best, print_search_output, write_summary_csv
from gpuprof.server_cmd import get_knob_flags


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

        baseline_dir = get_baseline_dir()
        best_dir = get_best_dir()
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
