from typing import Any, Dict, List, Tuple


def choose_top_k(records: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    def as_float(v: Any) -> float:
        return float(v)

    def first_present(r: Dict[str, Any], keys: List[str]) -> Any:
        for key in keys:
            if r.get(key) is not None:
                return r.get(key)
        return None

    valid: List[Dict[str, Any]] = []
    for r in records:
        error_count = r.get("error_count")
        if error_count is not None and as_float(error_count) > 0.0:
            continue
        error_rate = r.get("error_rate")
        if error_rate is not None and as_float(error_rate) > 0.0:
            continue

        p95_total = first_present(r, ["p95_total_ms", "p95_latency_ms"])
        p95_ttft = first_present(r, ["p95_ttft_ms"])
        chunks_per_s = first_present(r, ["chunks_per_s"])
        if p95_total is None or p95_ttft is None or chunks_per_s is None:
            continue
        valid.append(r)

    if not valid:
        raise SystemExit("No valid combinations after filtering errors")

    def key_fn(r: Dict[str, Any]) -> Tuple[float, float, float]:
        p95_total = as_float(first_present(r, ["p95_total_ms", "p95_latency_ms"]))
        p95_ttft = as_float(first_present(r, ["p95_ttft_ms"]))
        tps = as_float(first_present(r, ["chunks_per_s"]))
        return (p95_total, p95_ttft, -tps)

    return sorted(valid, key=key_fn)[:k]


def choose_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return choose_top_k(records, 1)[0]
