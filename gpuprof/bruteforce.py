import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpuprof.evaluate import evaluate
from gpuprof.io_utils import write_json
from gpuprof.reporting import write_summary_csv
from gpuprof.search import choose_best
from gpuprof.server_cmd import knob_options


@dataclass
class SweepRunResult:
    rows: List[Dict[str, Any]]
    trials: List[Dict[str, Any]]
    best_row: Dict[str, Any]
    best_trial: Dict[str, Any]


def build_sweep_definition(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    knobs = cfg.get("knobs", [])
    definition: List[Dict[str, Any]] = []
    option_matrix = [knob_options(knob) for knob in knobs]
    expected_trials = 1
    for opts in option_matrix:
        expected_trials *= len(opts)

    for trial, selected_indices in enumerate(itertools.product(*[range(len(opts)) for opts in option_matrix])):
        flags: List[str] = []
        assignment: Dict[str, Any] = {}
        knob_values: List[Dict[str, Any]] = []
        mask_parts: List[str] = []

        for idx, knob in enumerate(knobs):
            name = str(knob.get("name", f"knob_{idx}"))
            choice_index = int(selected_indices[idx])
            choice = option_matrix[idx][choice_index]
            mask_parts.append(str(choice_index))
            assignment[name] = choice["value"]
            flags.extend([str(x) for x in choice["flags"]])
            knob_values.append(
                {
                    "name": name,
                    "choice_index": choice_index,
                    "value": choice["value"],
                    "flags": [str(x) for x in choice["flags"]],
                }
            )

        knob_mask = "".join(mask_parts)
        definition.append(
            {
                "trial": trial,
                "knob_mask": knob_mask,
                "choice_indices": [int(x) for x in selected_indices],
                "flags": flags,
                "assignment": assignment,
                "knob_values": knob_values,
            }
        )

    if len(definition) != expected_trials:
        raise SystemExit(
            f"Sweep trial generation mismatch: expected {expected_trials}, got {len(definition)}"
        )
    return definition


class BruteForceSearch:
    def __init__(self, cfg: Dict[str, Any], sweep_dir: Path, sweep_definition: List[Dict[str, Any]]) -> None:
        self.cfg = cfg
        self.sweep_dir = sweep_dir
        self.sweep_definition = sweep_definition

    def run(self, baseline_metrics: Dict[str, Optional[float]]) -> SweepRunResult:
        rows: List[Dict[str, Any]] = []
        trials: List[Dict[str, Any]] = []
        print(f"Sweep trials: {len(self.sweep_definition)}")

        for trial_def in self.sweep_definition:
            trial = int(trial_def["trial"])
            knob_mask = str(trial_def["knob_mask"])
            flags = [str(x) for x in trial_def["flags"]]
            assignment = {str(k): v for k, v in (trial_def["assignment"] or {}).items()}

            trial_dir = self.sweep_dir / f"trial_{trial:03d}"
            result = evaluate(self.cfg, trial_dir, flags, assignment, profile=False)
            p95_total_ms = result.metrics.get("p95_latency_ms")
            p95_ttft_ms = result.metrics.get("p95_ttft_ms")
            chunks_per_s = result.metrics.get("chunks_per_s")
            error_rate = result.metrics.get("error_rate")

            row = {
                "trial": trial,
                "knob_mask": knob_mask,
                "assignment": assignment,
                "flags": flags,
                "choice_indices": trial_def["choice_indices"],
                "trial_dir": trial_dir.name,
                "p95_total_ms": p95_total_ms,
                "p95_ttft_ms": p95_ttft_ms,
                "chunks_per_s": chunks_per_s,
                "error_rate": error_rate,
                "p50_latency_ms": result.metrics.get("p50_latency_ms"),
                "p95_latency_ms": result.metrics.get("p95_latency_ms"),
                "p99_latency_ms": result.metrics.get("p99_latency_ms"),
                "p50_ttft_ms": result.metrics.get("p50_ttft_ms"),
                "p99_ttft_ms": result.metrics.get("p99_ttft_ms"),
            }
            rows.append(row)

            trial_record = {
                "trial": trial,
                "knob_mask": knob_mask,
                "choice_indices": trial_def["choice_indices"],
                "assignment": assignment,
                "flags": flags,
                "knob_values": trial_def["knob_values"],
                "metrics": result.metrics,
                "run_dir": str(trial_dir),
                "resolved_server_cmd": result.resolved_cmd,
                "resolved_env": result.resolved_env,
            }
            trials.append(trial_record)
            write_json(trial_dir / "config_trial.json", trial_record)

        best_row = choose_best(rows)
        best_trial = next((trial for trial in trials if int(trial["trial"]) == int(best_row["trial"])), None)
        if best_trial is None:
            raise SystemExit("Failed to resolve best trial metadata")

        best_config = {
            "trial": best_trial["trial"],
            "knob_mask": best_trial["knob_mask"],
            "choice_indices": best_trial["choice_indices"],
            "assignment": best_trial["assignment"],
            "flags": best_trial["flags"],
            "knob_values": best_trial["knob_values"],
            "metrics": best_trial["metrics"],
            "resolved_server_cmd": best_trial["resolved_server_cmd"],
            "resolved_env": best_trial["resolved_env"],
        }

        write_summary_csv(self.sweep_dir / "summary.csv", rows, baseline_metrics.get("p50_latency_ms"))
        summary_trials = [
            {
                "trial": int(r["trial"]),
                "trial_dir": str(r["trial_dir"]),
                "knob_mask": str(r["knob_mask"]),
                "choice_indices": list(r["choice_indices"]),
                "assignment": dict(r["assignment"]),
                "error_rate": r.get("error_rate"),
                "p95_total_ms": r.get("p95_total_ms"),
                "p95_ttft_ms": r.get("p95_ttft_ms"),
                "chunks_per_s": r.get("chunks_per_s"),
            }
            for r in rows
        ]
        write_json(
            self.sweep_dir / "summary.json",
            {
                "objective": {"primary": "p95_total_ms", "tie_breakers": ["p95_ttft_ms", "chunks_per_s"]},
                "trials": summary_trials,
                "best": best_config,
            },
        )

        return SweepRunResult(
            rows=rows,
            trials=trials,
            best_row=best_row,
            best_trial=best_trial,
        )
