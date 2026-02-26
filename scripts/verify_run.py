from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text()
    except Exception as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return obj


def as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if math.isfinite(x):
            return x
    return None


def as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def resolve_p95_total_ms(metrics: Dict[str, Any]) -> Optional[float]:
    for key in ("p95_total_ms", "p95_e2e_latency_ms"):
        x = as_number(metrics.get(key))
        if x is not None:
            return x
    return None


def verify_client_metrics(
    label: str,
    metrics_path: Path,
    errors: List[str],
    warnings: List[str],
) -> Optional[float]:
    try:
        metrics = load_json(metrics_path)
    except RuntimeError as exc:
        errors.append(f"{label}: {exc}")
        return None

    warmup_requests = as_int(metrics.get("warmup_requests"))
    if warmup_requests is None or warmup_requests < 0:
        errors.append(f"{label}: warmup_requests must be an integer >= 0")

    num_requests = as_int(metrics.get("num_requests"))
    if num_requests is None or num_requests <= 0:
        errors.append(f"{label}: num_requests must be an integer > 0")

    total_sent = as_int(metrics.get("total_sent"))
    if total_sent is None:
        errors.append(f"{label}: total_sent must be an integer")
    elif warmup_requests is not None and num_requests is not None:
        expected_total = warmup_requests + num_requests
        if total_sent != expected_total:
            errors.append(
                f"{label}: total_sent mismatch: expected {expected_total}, got {total_sent}"
            )

    if "error_rate" not in metrics:
        errors.append(f"{label}: missing error_rate")
    elif as_number(metrics.get("error_rate")) is None:
        errors.append(f"{label}: error_rate must be numeric")

    p95_total_ms = resolve_p95_total_ms(metrics)
    if p95_total_ms is None:
        errors.append(
            f"{label}: missing numeric p95_total_ms (or alias p95_e2e_latency_ms)"
        )

    if "p95_ttft_ms" not in metrics:
        warnings.append(f"{label}: p95_ttft_ms missing")
    elif as_number(metrics.get("p95_ttft_ms")) is None:
        warnings.append(f"{label}: p95_ttft_ms is non-numeric")

    return p95_total_ms


def find_best_trial_info(summary: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    trials = summary.get("trials")
    if not isinstance(trials, list):
        return None, None, None

    best = summary.get("best")
    if not isinstance(best, dict):
        return None, None, None

    trial_id_raw = best.get("trial")
    trial_id = as_int(trial_id_raw)

    best_p95 = None
    if isinstance(best.get("metrics"), dict):
        best_p95 = resolve_p95_total_ms(best["metrics"])
        if best_p95 is None:
            best_p95 = as_number(best["metrics"].get("p95_latency_ms"))
    if best_p95 is None:
        best_p95 = resolve_p95_total_ms(best)
    if best_p95 is None:
        best_p95 = as_number(best.get("p95_latency_ms"))

    trial_dir: Optional[str] = None
    if trial_id is not None:
        for t in trials:
            if isinstance(t, dict) and as_int(t.get("trial")) == trial_id:
                td = t.get("trial_dir")
                if isinstance(td, str):
                    trial_dir = td
                break

    return trial_id, trial_dir, best_p95


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify artifacts and metrics for a run directory.")
    ap.add_argument("--run-id", required=True, help="Run ID under runs/<run_id>/")
    args = ap.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    run_dir = Path("runs") / args.run_id
    required_files = [
        run_dir / "config_resolved.json",
        run_dir / "baseline" / "client_metrics.json",
        run_dir / "baseline" / "trace.nsys-rep",
        run_dir / "sweep" / "summary.json",
        run_dir / "sweep" / "summary.csv",
        run_dir / "best" / "best_config.json",
        run_dir / "best" / "client_metrics.json",
        run_dir / "best" / "trace.nsys-rep",
        run_dir / "best" / "summary.json",
    ]

    if not run_dir.exists() or not run_dir.is_dir():
        errors.append(f"missing run directory: {run_dir}")
    for path in required_files:
        if not path.exists():
            errors.append(f"missing required file: {path}")

    baseline_p95: Optional[float] = None
    best_p95: Optional[float] = None
    if not errors:
        baseline_p95 = verify_client_metrics(
            "baseline/client_metrics.json",
            run_dir / "baseline" / "client_metrics.json",
            errors,
            warnings,
        )
        best_p95 = verify_client_metrics(
            "best/client_metrics.json",
            run_dir / "best" / "client_metrics.json",
            errors,
            warnings,
        )

        sweep_summary_path = run_dir / "sweep" / "summary.json"
        try:
            sweep_summary = load_json(sweep_summary_path)
        except RuntimeError as exc:
            errors.append(str(exc))
            sweep_summary = {}

        trials = sweep_summary.get("trials")
        if not isinstance(trials, list):
            errors.append("sweep/summary.json: trials must be a list")
            trials = []

        ok_trials = 0
        for t in trials:
            if not isinstance(t, dict):
                continue
            er = as_number(t.get("error_rate"))
            if er is not None and er == 0.0:
                ok_trials += 1

        best_trial_id, best_trial_dir, best_trial_p95 = find_best_trial_info(sweep_summary)
        if best_trial_id is None:
            errors.append("sweep/summary.json: missing best trial id")
        if best_trial_p95 is None:
            errors.append("sweep/summary.json: missing best trial p95_total_ms")

        print(f"Run: {run_dir}")
        print(f"Sweep trials recorded: {len(trials)}")
        print(f"Sweep trials with error_rate == 0: {ok_trials}")
        if best_trial_id is not None:
            best_dir_msg = best_trial_dir if best_trial_dir is not None else "<unknown>"
            p95_msg = "na" if best_trial_p95 is None else f"{best_trial_p95:.6f}"
            print(f"Best trial: id={best_trial_id}, dir={best_dir_msg}, p95_total_ms={p95_msg}")

        if baseline_p95 is not None and best_p95 is not None:
            abs_delta = baseline_p95 - best_p95
            pct = (abs_delta / baseline_p95 * 100.0) if baseline_p95 != 0.0 else None
            pct_msg = "na" if pct is None else f"{pct:.4f}%"
            print(
                "Baseline vs best p95_total_ms: "
                f"baseline={baseline_p95:.6f}, best={best_p95:.6f}, "
                f"delta_ms={abs_delta:.6f}, percent={pct_msg}"
            )

    for msg in warnings:
        print(f"WARNING: {msg}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}")
        print(f"FAILED: {len(errors)} error(s)")
        return 1

    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
