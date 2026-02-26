from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple


def ns_to_s(ns: int) -> float:
    return float(ns) / 1e9


def bytes_to_mb(b: int) -> float:
    return float(b) / (1024 * 1024)


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (name,)
    )
    return cur.fetchone() is not None


def get_table_info(cur: sqlite3.Cursor, table: str) -> List[Tuple[Any, ...]]:
    cur.execute(f"PRAGMA table_info({table});")
    return cur.fetchall()


def get_columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    return [r[1] for r in get_table_info(cur, table)]


def fetch_scalar(cur: sqlite3.Cursor, q: str, params: Tuple[Any, ...] = ()) -> Any:
    cur.execute(q, params)
    row = cur.fetchone()
    if not row:
        return None
    return row[0]


def compute_capture_window_ns(cur: sqlite3.Cursor) -> int:
    if table_exists(cur, "OSRT_API"):
        start_ns, end_ns = cur.execute(
            "SELECT MIN(start), MAX(end) FROM OSRT_API;"
        ).fetchone()
    elif table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        start_ns, end_ns = cur.execute(
            "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL;"
        ).fetchone()
    else:
        raise SystemExit(
            "No OSRT_API and no CUPTI_ACTIVITY_KIND_KERNEL, cannot compute capture window."
        )

    if start_ns is None or end_ns is None:
        raise SystemExit("Capture window timestamps missing.")
    return int(end_ns - start_ns)


def load_nvtx_phases(cur: sqlite3.Cursor, capture_window_ns: int) -> List[Dict[str, Any]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []

    has_stringids = table_exists(cur, "StringIds")

    if has_stringids:
        rows = cur.execute(
            """
            SELECT COALESCE(n.text, s.value) AS label,
                   SUM(n.end - n.start) AS total_ns,
                   COUNT(*) AS n_ranges
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON s.id = n.textId
            WHERE COALESCE(n.text, s.value) LIKE 'phase:%'
              AND n.end IS NOT NULL
              AND n.end > n.start
            GROUP BY label
            ORDER BY total_ns DESC;
            """
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT n.text AS label,
                   SUM(n.end - n.start) AS total_ns,
                   COUNT(*) AS n_ranges
            FROM NVTX_EVENTS n
            WHERE n.text LIKE 'phase:%'
              AND n.end IS NOT NULL
              AND n.end > n.start
            GROUP BY label
            ORDER BY total_ns DESC;
            """
        ).fetchall()

    phases: List[Dict[str, Any]] = []
    for label, total_ns, n_ranges in rows:
        total_ns_i = int(total_ns or 0)
        pct = (total_ns_i / capture_window_ns * 100.0) if capture_window_ns > 0 else 0.0
        phases.append(
            {
                "name": str(label),
                "total_s": round(ns_to_s(total_ns_i), 6),
                "pct_of_capture_window": round(pct, 2),
                "ranges": int(n_ranges or 0),
            }
        )
    return phases


def kernel_name_expr(cur: sqlite3.Cursor) -> Tuple[str, str, str]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    cols = set(get_columns(cur, table))

    # Common Nsight SQLite schema: demangledName exists but is sometimes a StringIds id
    if "demangledName" in cols:
        if table_exists(cur, "StringIds"):
            sample = fetch_scalar(cur, "SELECT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1;")
            if sample is not None and str(sample).isdigit():
                return "s.value", "JOIN StringIds s ON s.id = CAST(k.demangledName AS INTEGER)", "StringIds(CAST(demangledName))"
        return "k.demangledName", "", "demangledName"

    if "nameId" in cols and table_exists(cur, "StringIds"):
        return "s.value", "JOIN StringIds s ON s.id = k.nameId", "StringIds(nameId)"

    return "'<unknown_kernel_name>'", "", "unknown"


def load_gpu_kernel_summary(cur: sqlite3.Cursor, top_n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": False}

    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return out

    out["present"] = True
    total_ns = fetch_scalar(cur, "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL;") or 0
    out["total_s"] = round(ns_to_s(int(total_ns)), 6)

    name_expr, join_clause, source = kernel_name_expr(cur)
    out["name_source"] = source

    rows = cur.execute(
        f"""
        SELECT {name_expr} AS name,
               SUM(k.end-k.start) AS total_ns,
               COUNT(*) AS instances
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        {join_clause}
        GROUP BY name
        ORDER BY total_ns DESC
        LIMIT ?;
        """,
        (top_n,),
    ).fetchall()

    out["top"] = [
        {
            "name": str(name),
            "total_s": round(ns_to_s(int(total_ns or 0)), 6),
            "count": int(instances or 0),
        }
        for (name, total_ns, instances) in rows
    ]
    return out


# Minimal mapping (common values seen in your output)
MEMCPY_KIND_MAP = {
    1: "Host-to-Device",
    2: "Device-to-Host",
    8: "Device-to-Device",
}


def load_gpu_memcpy_summary(cur: sqlite3.Cursor) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": False}

    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return out

    out["present"] = True

    cols = set(get_columns(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"))
    bytes_col = "bytes" if "bytes" in cols else None
    group_field = "copyKind" if "copyKind" in cols else ("kind" if "kind" in cols else None)

    total_ns = fetch_scalar(cur, "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY;") or 0
    out["total_s"] = round(ns_to_s(int(total_ns)), 6)

    if bytes_col:
        total_bytes = fetch_scalar(cur, f"SELECT SUM({bytes_col}) FROM CUPTI_ACTIVITY_KIND_MEMCPY;") or 0
        out["total_mb"] = round(bytes_to_mb(int(total_bytes)), 6)
    else:
        out["total_mb"] = None

    by_op: List[Dict[str, Any]] = []
    if group_field:
        q = f"""
        SELECT {group_field} AS kind,
               SUM(end-start) AS total_ns,
               COUNT(*) AS count
               {', SUM(' + bytes_col + ') AS total_bytes' if bytes_col else ''}
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY {group_field}
        ORDER BY total_ns DESC;
        """
        for row in cur.execute(q).fetchall():
            kind = int(row[0]) if row[0] is not None else -1
            total_ns_i = int(row[1] or 0)
            count = int(row[2] or 0)
            item: Dict[str, Any] = {
                "op": MEMCPY_KIND_MAP.get(kind, f"kind_{kind}"),
                "kind": kind,
                "total_s": round(ns_to_s(total_ns_i), 6),
                "count": count,
            }
            if bytes_col:
                total_bytes = int(row[3] or 0)
                item["total_mb"] = round(bytes_to_mb(total_bytes), 6)
            by_op.append(item)

    out["by_op"] = by_op
    out["notes"] = [
        "Memcpy kinds are mapped using a small common mapping (H2D/D2H/D2D).",
        "If you see kind_* values, the enum differs in this Nsight version; totals are still correct.",
    ]
    return out


def load_cpu_waits(cur: sqlite3.Cursor, capture_window_ns: int, top_n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": False}
    if not table_exists(cur, "OSRT_API"):
        return out

    out["present"] = True
    rows = cur.execute(
        """
        SELECT s.value AS name,
               SUM(a.end-a.start) AS total_ns,
               COUNT(*) AS calls
        FROM OSRT_API a
        JOIN StringIds s ON s.id = a.nameId
        GROUP BY s.value
        ORDER BY total_ns DESC
        LIMIT ?;
        """,
        (top_n,),
    ).fetchall()

    items: List[Dict[str, Any]] = []
    for name, total_ns, calls in rows:
        total_ns_i = int(total_ns or 0)
        pct_thread_summed = (total_ns_i / capture_window_ns * 100.0) if capture_window_ns > 0 else 0.0
        items.append(
            {
                "name": str(name),
                "total_s": round(ns_to_s(total_ns_i), 6),
                "calls": int(calls or 0),
                "pct_thread_summed_vs_window": round(pct_thread_summed, 2),
            }
        )

    out["top"] = items
    out["notes"] = [
        "Percent can be >100% because OSRT time is summed across many threads.",
        "High epoll_wait/poll/pthread_cond_* means CPU threads are mostly waiting while GPU work runs asynchronously.",
    ]
    return out


def ui_checks(cur: sqlite3.Cursor, phases: List[Dict[str, Any]], kernels: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces numbers that line up with Nsight CLI reports:
      - nvtx_sum -> your NVTX phase totals
      - cuda_gpu_kern_sum -> top kernel total + instances
      - cuda_gpu_mem_time_sum / mem_size_sum -> memcpy + memset totals if present
    """
    out: Dict[str, Any] = {}

    # --- nvtx_sum equivalent ---
    nvtx_sum_items = []
    for p in phases:
        nvtx_sum_items.append(
            {
                "range": p["name"],
                "total_s": p["total_s"],
                "ranges": p["ranges"],
            }
        )
    out["nvtx_sum"] = nvtx_sum_items

    # --- cuda_gpu_kern_sum equivalent (top kernel line) ---
    top_kernel = None
    if kernels.get("present") and kernels.get("top"):
        top_kernel = kernels["top"][0]
    out["cuda_gpu_kern_sum_top"] = (
        {
            "name": top_kernel["name"],
            "total_s": top_kernel["total_s"],
            "instances": top_kernel["count"],
        }
        if top_kernel
        else None
    )

    # --- cuda_gpu_mem_time_sum + cuda_gpu_mem_size_sum equivalent ---
    mem_rows: List[Dict[str, Any]] = []

    # MEMSET (Nsight shows it separately in your CLI report)
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMSET"):
        cols = set(get_columns(cur, "CUPTI_ACTIVITY_KIND_MEMSET"))
        # bytes is common, but not guaranteed
        bytes_col = "bytes" if "bytes" in cols else None

        total_ns = fetch_scalar(cur, "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMSET;") or 0
        count = fetch_scalar(cur, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMSET;") or 0
        item = {
            "operation": "[CUDA memset]",
            "total_s": round(ns_to_s(int(total_ns)), 6),
            "count": int(count),
        }
        if bytes_col:
            total_bytes = fetch_scalar(cur, f"SELECT SUM({bytes_col}) FROM CUPTI_ACTIVITY_KIND_MEMSET;") or 0
            item["total_mb"] = round(bytes_to_mb(int(total_bytes)), 6)
        mem_rows.append(item)

    # MEMCPY grouped
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        cols = set(get_columns(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"))
        bytes_col = "bytes" if "bytes" in cols else None
        group_field = "copyKind" if "copyKind" in cols else ("kind" if "kind" in cols else None)

        if group_field:
            q = f"""
            SELECT {group_field} AS kind,
                   SUM(end-start) AS total_ns,
                   COUNT(*) AS count
                   {', SUM(' + bytes_col + ') AS total_bytes' if bytes_col else ''}
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY {group_field}
            ORDER BY total_ns DESC;
            """
            for row in cur.execute(q).fetchall():
                kind = int(row[0]) if row[0] is not None else -1
                total_ns_i = int(row[1] or 0)
                count = int(row[2] or 0)
                op = MEMCPY_KIND_MAP.get(kind, f"kind_{kind}")
                item = {
                    "operation": f"[CUDA memcpy {op}]",
                    "kind": kind,
                    "total_s": round(ns_to_s(total_ns_i), 6),
                    "count": count,
                }
                if bytes_col:
                    total_bytes = int(row[3] or 0)
                    item["total_mb"] = round(bytes_to_mb(total_bytes), 6)
                mem_rows.append(item)

    # Sort like Nsight report (largest time first)
    mem_rows.sort(key=lambda r: r.get("total_s", 0.0), reverse=True)
    out["cuda_gpu_mem_sum"] = mem_rows

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path", help="Path to .sqlite exported by nsys")
    ap.add_argument("--out", default="artifacts/summary.json", help="Output JSON path")
    ap.add_argument("--top-osrt", type=int, default=10, help="Top N OSRT calls")
    ap.add_argument("--top-kernels", type=int, default=15, help="Top N GPU kernels")
    args = ap.parse_args()

    db = args.sqlite_path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db)
    cur = conn.cursor()

    capture_window_ns = compute_capture_window_ns(cur)

    phases = load_nvtx_phases(cur, capture_window_ns)
    kernels = load_gpu_kernel_summary(cur, args.top_kernels)
    memcpy = load_gpu_memcpy_summary(cur)
    cpu_waits = load_cpu_waits(cur, capture_window_ns, args.top_osrt)

    checks = ui_checks(cur, phases, kernels)

    conn.close()

    summary: Dict[str, Any] = {
        "run": {
            "type": "nsight_systems_baseline_summary",
            "sqlite": db,
            "capture_window_s": round(ns_to_s(capture_window_ns), 6),
        },
        "phases": phases,
        "gpu": {
            "kernels": kernels,
            "memcpy": memcpy,
        },
        "cpu": {
            "wait_calls": cpu_waits,
        },
        "ui_checks": checks,
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
