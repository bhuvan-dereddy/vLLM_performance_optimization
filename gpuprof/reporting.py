import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpuprof.formatting import format_num, format_delta


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
