from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def powerset(knobs: List[Dict[str, Any]]) -> List[Tuple[str, List[str], Dict[str, Any]]]:
    """
    All 2^N combos. combo_id is bitstring in knob order.
    Returns: (combo_id, flags_list, assignment)
    """
    combos = []
    n = len(knobs)
    for bits in itertools.product([0, 1], repeat=n):
        combo_id = "".join(str(b) for b in bits)
        flags: List[str] = []
        assignment: Dict[str, Any] = {}
        for b, k in zip(bits, knobs):
            name = k["name"]
            if b == 0:
                assignment[name] = False
                flags += k.get("off_flags", [])
            else:
                assignment[name] = True
                flags += k.get("on_flags", [])
        combos.append((combo_id, flags, assignment))
    return combos


def score(record: Dict[str, Any], primary: str) -> float:
    """
    Lower is better.
    - latency metrics like *_ms => minimize
    - throughput like *_tok_s => maximize (negate)
    Invalid runs => +inf
    """
    if not record.get("ok"):
        return float("inf")
    stats = record.get("client_stats") or {}
    v = stats.get(primary)
    if v is None:
        return float("inf")
    try:
        fv = float(v)
    except Exception:
        return float("inf")

    if primary.endswith("_tok_s"):
        return -fv
    return fv


def run_combo_with_profiling(
    out_dir: Path,
    cfg: Dict[str, Any],
    combo_id: str,
    flags: List[str],
    assignment: Dict[str, Any],
    num_requests: int,
    stage: str,
    enable_profiling: bool = False,
) -> Dict[str, Any]:
    """
    Enhanced version that optionally runs Nsight profiling
    """
    run_dir = out_dir / stage / f"combo_{combo_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/run_once_with_profile.py",
        "--run-dir", str(run_dir),
        "--model-dir", cfg["server"]["model_dir"],
        "--host", cfg["server"]["host"],
        "--port", str(cfg["server"]["port"]),
        "--dtype", cfg["server"]["dtype"],
        "--gpu-mem-util", str(cfg["server"]["gpu_memory_utilization"]),
        "--max-model-len", str(cfg["server"]["max_model_len"]),
        "--server-startup-timeout-s", str(cfg["server"].get("startup_timeout_s", 240)),
        "--prompts", cfg["dataset"]["prompts_jsonl"],
        "--num-requests", str(num_requests),
        "--concurrency", str(cfg["workload"]["concurrency"]),
        "--max-new-tokens", str(cfg["workload"]["max_new_tokens"]),
        "--temperature", str(cfg["workload"]["temperature"]),
        "--timeout-s", str(cfg["workload"]["timeout_s"]),
        "--server-extra", json.dumps(flags),
    ]
    
    if enable_profiling:
        cmd.append("--enable-nsight-profiling")

    ok = True
    err = None
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        ok = False
        err = f"run_once_failed: {e}"

    # Read run_once summary
    summary_path = run_dir / "client_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        ok = bool(summary.get("ok")) and ok
        client_stats = summary.get("client_stats", {})
        reason = summary.get("reason", "")
        server_cmd = summary.get("server_cmd", [])
        stderr_tail = summary.get("server_stderr_tail", "")
    else:
        client_stats = {}
        reason = "missing_client_summary"
        server_cmd = []
        stderr_tail = ""

    # If profiling was enabled, also load NVTX data
    nvtx_summary = {}
    if enable_profiling:
        nvtx_path = run_dir / "nvtx_summary.json"
        if nvtx_path.exists():
            nvtx_summary = load_json(nvtx_path)

    rec = {
        "stage": stage,
        "combo_id": combo_id,
        "assignment": assignment,
        "flags": flags,
        "run_dir": str(run_dir),
        "ok": ok,
        "reason": reason if reason else err,
        "client_stats": client_stats,
        "nvtx_summary": nvtx_summary if enable_profiling else None,
        "server_cmd": server_cmd,
        "server_stderr_tail": stderr_tail,
    }
    return rec


def compute_delta(baseline_stats: Dict[str, Any], best_stats: Dict[str, Any]) -> Dict[str, Any]:
    def pct_improve(b: Any, v: Any, higher_better: bool) -> Any:
        try:
            b = float(b)
            v = float(v)
        except Exception:
            return None
        if b == 0:
            return None
        if higher_better:
            return round((v - b) / b * 100.0, 3)
        return round((b - v) / b * 100.0, 3)

    keys = [
        "p50_total_ms", "p95_total_ms",
        "p50_ttft_ms", "p95_ttft_ms",
        "throughput_tok_s", "wall_s",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        b = baseline_stats.get(k)
        v = best_stats.get(k)
        out[k] = {"baseline": b, "best": v}

    out["improvement_pct"] = {
        "p95_total_ms": pct_improve(baseline_stats.get("p95_total_ms"), best_stats.get("p95_total_ms"), higher_better=False),
        "p95_ttft_ms": pct_improve(baseline_stats.get("p95_ttft_ms"), best_stats.get("p95_ttft_ms"), higher_better=False),
        "throughput_tok_s": pct_improve(baseline_stats.get("throughput_tok_s"), best_stats.get("throughput_tok_s"), higher_better=True),
    }
    return out


def compute_nvtx_delta(baseline_nvtx: Dict[str, Any], best_nvtx: Dict[str, Any]) -> Dict[str, Any]:
    """Compare NVTX phase timings"""
    if not baseline_nvtx or not best_nvtx:
        return {}
    
    baseline_phases = {p["name"]: p for p in baseline_nvtx.get("phases", [])}
    best_phases = {p["name"]: p for p in best_nvtx.get("phases", [])}
    
    comparison = {}
    for phase_name in baseline_phases:
        if phase_name in best_phases:
            b_time = baseline_phases[phase_name]["total_s"]
            v_time = best_phases[phase_name]["total_s"]
            
            if b_time > 0:
                improvement = round((b_time - v_time) / b_time * 100.0, 3)
                comparison[phase_name] = {
                    "baseline_s": b_time,
                    "best_s": v_time,
                    "improvement_pct": improvement
                }
    
    return comparison


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_comparison_table(baseline_stats: Dict, best_stats: Dict, delta: Dict, nvtx_delta: Dict) -> None:
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║              PERFORMANCE COMPARISON (vs Baseline)                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ CLIENT METRICS                                                      │")
    print("├──────────────────────┬──────────────┬──────────────┬────────────────┤")
    print("│ Metric               │ Baseline     │ Best         │ Improvement    │")
    print("├──────────────────────┬──────────────┬──────────────┬────────────────┤")
    
    metrics = [
        ("p95_total_ms", "P95 Latency (ms)"),
        ("p95_ttft_ms", "P95 TTFT (ms)"),
        ("throughput_tok_s", "Throughput (tok/s)"),
    ]
    
    for key, label in metrics:
        b_val = baseline_stats.get(key, 0)
        v_val = best_stats.get(key, 0)
        imp = delta.get("improvement_pct", {}).get(key)
        
        if imp is not None:
            sign = "✅" if imp > 0 else "⚠️" if imp < 0 else "➡️"
            print(f"│ {label:20s} │ {b_val:12.2f} │ {v_val:12.2f} │ {imp:+6.2f}% {sign:3s} │")
        else:
            print(f"│ {label:20s} │ {b_val:12.2f} │ {v_val:12.2f} │ N/A            │")
    
    print("└──────────────────────┴──────────────┴──────────────┴────────────────┘")
    
    if nvtx_delta:
        print()
        print("┌─────────────────────────────────────────────────────────────────────┐")
        print("│ NVTX PHASE BREAKDOWN                                                │")
        print("├──────────────────────┬──────────────┬──────────────┬────────────────┤")
        print("│ Phase                │ Baseline (s) │ Best (s)     │ Improvement    │")
        print("├──────────────────────┬──────────────┬──────────────┬────────────────┤")
        
        for phase_name, phase_data in sorted(nvtx_delta.items()):
            b_time = phase_data["baseline_s"]
            v_time = phase_data["best_s"]
            imp = phase_data["improvement_pct"]
            
            sign = "✅" if imp > 0 else "⚠️" if imp < 0 else "➡️"
            short_name = phase_name.replace("phase:", "")
            print(f"│ {short_name:20s} │ {b_time:12.3f} │ {v_time:12.3f} │ {imp:+6.2f}% {sign:3s} │")
        
        print("└──────────────────────┴──────────────┴──────────────┴────────────────┘")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimizations.json")
    ap.add_argument("--out-dir", default="artifacts/autotune")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = cfg["metrics"]["primary"]
    screening_n = int(cfg["experiment"]["screening_requests"])
    final_n = int(cfg["experiment"]["final_requests"])
    top_k = int(cfg["experiment"]["top_k_final"])

    knobs = cfg["knobs"]
    combos = powerset(knobs)

    baseline_id = "0" * len(knobs)
    
    print_banner("GPU INFERENCE AUTOTUNING WITH NVTX PROFILING")
    print(f"Primary metric: {primary}")
    print(f"Total combinations: {len(combos)} (2^{len(knobs)} knobs)")
    print(f"Baseline combo: {baseline_id} (all knobs OFF)")

    history_path = out_dir / "history.jsonl"

    # ============================================================
    # STAGE 1: Quick screening (client metrics only, no profiling)
    # ============================================================
    print_banner("STAGE 1: SCREENING ALL COMBINATIONS (Quick)")
    print(f"Running {len(combos)} combinations with {screening_n} requests each...")
    
    stage1: List[Dict[str, Any]] = []
    for i, (combo_id, flags, assignment) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] Testing combo_{combo_id}...", end=" ", flush=True)
        rec = run_combo_with_profiling(
            out_dir=out_dir,
            cfg=cfg,
            combo_id=combo_id,
            flags=flags,
            assignment=assignment,
            num_requests=screening_n,
            stage="stage1",
            enable_profiling=False,  # No profiling in stage 1
        )
        rec["score"] = score(rec, primary)
        stage1.append(rec)
        
        if rec.get("ok"):
            print(f"✅ {primary}={rec['client_stats'].get(primary, 'N/A')}")
        else:
            print(f"❌ Failed: {rec.get('reason', 'unknown')}")
        
        with history_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    stage1_sorted = sorted(stage1, key=lambda r: r["score"])
    top_valid = [r for r in stage1_sorted if r.get("ok")]
    
    print(f"\n✅ Stage 1 complete: {len(top_valid)}/{len(combos)} valid configurations")
    
    # Ensure baseline is included
    baseline_rec_stage1 = next((r for r in stage1 if r["combo_id"] == baseline_id), None)
    if baseline_rec_stage1 is None:
        raise SystemExit("Baseline combo not found in stage1 (unexpected).")
    
    # Select top K candidates
    top_candidates = top_valid[:top_k]
    
    print(f"\nTop {len(top_candidates)} candidates for deep profiling:")
    for i, r in enumerate(top_candidates, 1):
        print(f"  {i}. combo_{r['combo_id']}: {primary}={r['client_stats'].get(primary, 'N/A')}")

    # ============================================================
    # STAGE 2: Deep profiling with NVTX
    # ============================================================
    print_banner("STAGE 2: DEEP PROFILING WITH NVTX")
    print(f"Profiling baseline + top {len(top_candidates)} candidates with {final_n} requests each")
    print("This will capture GPU kernels, memory copies, and NVTX phase timings...\n")
    
    stage2: List[Dict[str, Any]] = []

    # 2A) Baseline with profiling
    print(f"🔬 [1/{len(top_candidates)+1}] Profiling BASELINE (combo_{baseline_id})...")
    b2 = run_combo_with_profiling(
        out_dir=out_dir,
        cfg=cfg,
        combo_id=baseline_id,
        flags=baseline_rec_stage1["flags"],
        assignment=baseline_rec_stage1["assignment"],
        num_requests=final_n,
        stage="stage2",
        enable_profiling=True,  # ENABLE PROFILING
    )
    b2["score"] = score(b2, primary)
    stage2.append(b2)
    with history_path.open("a") as f:
        f.write(json.dumps(b2) + "\n")
    
    if b2.get("ok"):
        print(f"   ✅ Client: {primary}={b2['client_stats'].get(primary, 'N/A')}")
        if b2.get("nvtx_summary"):
            phases = b2["nvtx_summary"].get("phases", [])
            if phases:
                print(f"   ✅ NVTX phases captured: {len(phases)}")
                for p in phases[:3]:  # Show top 3
                    print(f"      - {p['name']}: {p['total_s']:.3f}s")

    # 2B) Top candidates with profiling
    for idx, r1 in enumerate(top_candidates, 2):
        if r1["combo_id"] == baseline_id:
            continue
        
        print(f"\n🔬 [{idx}/{len(top_candidates)+1}] Profiling combo_{r1['combo_id']}...")
        r2 = run_combo_with_profiling(
            out_dir=out_dir,
            cfg=cfg,
            combo_id=r1["combo_id"],
            flags=r1["flags"],
            assignment=r1["assignment"],
            num_requests=final_n,
            stage="stage2",
            enable_profiling=True,  # ENABLE PROFILING
        )
        r2["score"] = score(r2, primary)
        stage2.append(r2)
        with history_path.open("a") as f:
            f.write(json.dumps(r2) + "\n")
        
        if r2.get("ok"):
            print(f"   ✅ Client: {primary}={r2['client_stats'].get(primary, 'N/A')}")
            if r2.get("nvtx_summary"):
                phases = r2["nvtx_summary"].get("phases", [])
                if phases:
                    print(f"   ✅ NVTX phases captured: {len(phases)}")

    stage2_sorted = sorted(stage2, key=lambda r: r["score"])
    best = stage2_sorted[0] if stage2_sorted else None
    baseline_final = b2

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print_banner("🏆 AUTOTUNING COMPLETE - FINAL REPORT")
    
    if best:
        print(f"\nWINNER: combo_{best['combo_id']}")
        print(f"Configuration: {json.dumps(best['assignment'], indent=2)}")
        print(f"Flags: {' '.join(best['flags'])}")
    
    baseline_stats = (baseline_final.get("client_stats") or {})
    best_stats = ((best or {}).get("client_stats") or {})
    delta = compute_delta(baseline_stats, best_stats)
    
    baseline_nvtx = baseline_final.get("nvtx_summary", {})
    best_nvtx = (best or {}).get("nvtx_summary", {})
    nvtx_delta = compute_nvtx_delta(baseline_nvtx, best_nvtx)
    
    print_comparison_table(baseline_stats, best_stats, delta, nvtx_delta)

    # Save reports
    write_json(out_dir / "baseline_final.json", baseline_final)
    write_json(out_dir / "best_final.json", best if best else {})

    report = {
        "primary_metric": primary,
        "baseline_combo_id": baseline_id,
        "baseline_flags": baseline_final.get("flags", []),
        "best_combo_id": (best or {}).get("combo_id"),
        "best_flags": (best or {}).get("flags", []),
        "baseline_stats": baseline_stats,
        "best_stats": best_stats,
        "delta": delta,
        "baseline_nvtx": baseline_nvtx,
        "best_nvtx": best_nvtx,
        "nvtx_delta": nvtx_delta,
        "notes": [
            "Stage 1: All combos screened with client metrics only (fast)",
            "Stage 2: Top K + baseline profiled with Nsight NVTX (detailed)",
            "NVTX phases show GPU kernel execution breakdown by operation type"
        ],
    }
    write_json(out_dir / "report.json", report)

    leaderboard = {
        "primary_metric": primary,
        "stage1_total_combos": len(stage1),
        "stage1_valid": sum(1 for r in stage1 if r.get("ok")),
        "stage1_top": stage1_sorted[: min(10, len(stage1_sorted))],
        "stage2_rerun_count": len(stage2),
        "stage2_results": stage2_sorted,
        "baseline_final": baseline_final,
        "best": best,
    }
    write_json(out_dir / "leaderboard.json", leaderboard)
    write_json(out_dir / "best_config.json", best if best else {})

    print("\n📁 Files written:")
    print(f"   ✅ {out_dir}/report.json")
    print(f"   ✅ {out_dir}/baseline_final.json")
    print(f"   ✅ {out_dir}/best_final.json")
    print(f"   ✅ {out_dir}/leaderboard.json")
    print(f"   ✅ {out_dir}/best_config.json")
    
    if best and best.get("ok"):
        print("\n🎯 To use the best configuration:")
        print(f"   vllm serve {cfg['server']['model_dir']} \\")
        for flag in best['flags']:
            print(f"     {flag} \\")
        print()


if __name__ == "__main__":
    main()
