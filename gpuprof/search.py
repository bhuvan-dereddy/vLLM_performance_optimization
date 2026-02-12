from typing import Any, Dict, List, Tuple


def choose_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        r
        for r in records
        if r.get("error_rate") == 0.0
        and r.get("p95_latency_ms") is not None
        and r.get("p99_latency_ms") is not None
        and r.get("p95_ttft_ms") is not None
        and r.get("p50_latency_ms") is not None
        and r.get("chunks_per_s") is not None
    ]
    if not valid:
        raise SystemExit("No valid combinations with error_rate == 0")

    def key_fn(r: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        p95 = float(r.get("p95_latency_ms"))
        p99 = float(r.get("p99_latency_ms"))
        p95_ttft = float(r.get("p95_ttft_ms"))
        p50 = float(r.get("p50_latency_ms"))
        tps = float(r.get("chunks_per_s")) if r.get("chunks_per_s") is not None else float("-inf")
        return (p95, p99, p95_ttft, p50, -tps)

    return sorted(valid, key=key_fn)[0]

