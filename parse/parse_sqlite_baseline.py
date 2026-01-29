from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def ns_to_s(ns: int) -> float:
    return float(ns) / 1e9


def bytes_to_mb(b: int) -> float:
    return float(b) / (1024 * 1024)


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (name,))
    return cur.fetchone() is not None


def get_columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]


def fetch_scalar(cur: sqlite3.Cursor, q: str, params: Tuple[Any, ...] = ()) -> Any:
    cur.execute(q, params)
    row = cur.fetchone()
    if not row:
        return None
    return row[0]


def pick_kernel_name_expr(kernel_cols: set) -> Tuple[str, str]:
    """
    Returns:
      - select_expr: SQL expression for kernel name (e.g., "s.value" or "k.demangledName")
      - join_clause: either "" or "JOIN StringIds s ON s.id = k.nameId"
    """
    # Most informative first
    if "demangledName" in kernel_cols:
        return "k.demangledName", ""
    if "name" in kernel_cols:
        return "k.name", ""
    if "shortName" in kernel_cols:
        return "k.shortName", ""
    if "nameId" in kernel_cols:
        return "s.value", "JOIN StringIds s ON s.id = k.nameId"

    # Fallback: no known name column
    return "'<unknown_kernel_name>'", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path", help="Path to .sqlite exported by nsys")
    ap.add_argument("--out", default="artifacts/summary.json", help="Output JSON path")
    ap.add_argument("--top-osrt", type=int, default=10, help="Top N OSRT calls")
    ap.add_argument("--top-kernels", type=int, default=15, help="Top N GPU kernels")
    args = ap.parse_args()

    db = args.sqlite_path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # ----------------------------
    # 1) Capture window
    # Prefer OSRT_API (CPU timeline), else fallback to kernel timeline
    # ----------------------------
    capture_start_ns = None
    capture_end_ns = None

    if table_exists(cur, "OSRT_API"):
        cur.execute("SELECT MIN(start), MAX(end) FROM OSRT_API;")
        capture_start_ns, capture_end_ns = cur.fetchone()
    elif table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL;")
        capture_start_ns, capture_end_ns = cur.fetchone()

    if capture_start_ns is None or capture_end_ns is None:
        raise SystemExit("Could not determine capture window (no OSRT_API and no CUPTI kernel rows).")

    capture_window_ns = int(capture_end_ns - capture_start_ns)

    # ----------------------------
    # 2) OSRT summary
    # ----------------------------
    osrt_summary: Dict[str, Any] = {"present": False}
    if table_exists(cur, "OSRT_API"):
        osrt_summary["present"] = True
        rows = cur.execute(
            """
            SELECT s.value AS name,
                   SUM(a.end - a.start) AS total_ns,
                   COUNT(*) AS calls
            FROM OSRT_API a
            JOIN StringIds s ON s.id = a.nameId
            GROUP BY s.value
            ORDER BY total_ns DESC
            LIMIT ?;
            """,
            (args.top_osrt,),
        ).fetchall()

        top_calls = []
        for name, total_ns, calls in rows:
            total_ns = int(total_ns or 0)
            calls = int(calls or 0)
            pct = (total_ns / capture_window_ns * 100.0) if capture_window_ns > 0 else 0.0
            top_calls.append(
                {
                    "name": name,
                    "total_s": round(ns_to_s(total_ns), 6),
                    "calls": calls,
                    "pct_of_capture_window": round(pct, 3),
                }
            )

        osrt_summary["top_calls"] = top_calls
        osrt_summary["notes"] = [
            "OSRT time is summed across threads, so totals can exceed capture window.",
            "poll/epoll_wait/pthread_cond_wait usually mean CPU threads are waiting for work or synchronization.",
        ]

    # ----------------------------
    # 3) GPU kernel summary (schema-aware)
    # ----------------------------
    gpu_summary: Dict[str, Any] = {"present": False}
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        gpu_summary["present"] = True

        kernel_cols = set(get_columns(cur, "CUPTI_ACTIVITY_KIND_KERNEL"))
        name_expr, join_clause = pick_kernel_name_expr(kernel_cols)

        total_kernel_ns = fetch_scalar(cur, "SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL;") or 0
        gpu_summary["total_kernel_s"] = round(ns_to_s(int(total_kernel_ns)), 6)

        # Top kernels by summed duration
        q = f"""
            SELECT {name_expr} AS name,
                   SUM(k.end - k.start) AS total_ns,
                   COUNT(*) AS instances
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            {join_clause}
            GROUP BY name
            ORDER BY total_ns DESC
            LIMIT ?;
        """
        kernel_rows = cur.execute(q, (args.top_kernels,)).fetchall()

        top_kernels = []
        for name, total_ns, inst in kernel_rows:
            top_kernels.append(
                {
                    "name": str(name),
                    "total_s": round(ns_to_s(int(total_ns or 0)), 6),
                    "instances": int(inst or 0),
                }
            )

        gpu_summary["top_kernels"] = top_kernels
        gpu_summary["kernel_name_source"] = (
            "demangledName" if "demangledName" in kernel_cols else
            "name" if "name" in kernel_cols else
            "shortName" if "shortName" in kernel_cols else
            "StringIds(nameId)" if "nameId" in kernel_cols else
            "unknown"
        )

    # ----------------------------
    # 4) GPU memcpy summary (keep simple + robust)
    # ----------------------------
    memcpy_summary: Dict[str, Any] = {"present": False}
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        memcpy_summary["present"] = True

        memcpy_cols = set(get_columns(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"))
        bytes_col = "bytes" if "bytes" in memcpy_cols else None

        # Prefer copyKind (common), else kind, else no grouping
        if "copyKind" in memcpy_cols:
            group_field = "copyKind"
        elif "kind" in memcpy_cols:
            group_field = "kind"
        else:
            group_field = None

        total_memcpy_ns = fetch_scalar(cur, "SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY;") or 0
        memcpy_summary["total_memcpy_s"] = round(ns_to_s(int(total_memcpy_ns)), 6)

        if bytes_col:
            total_bytes = fetch_scalar(cur, f"SELECT SUM({bytes_col}) FROM CUPTI_ACTIVITY_KIND_MEMCPY;") or 0
            memcpy_summary["total_memcpy_mb"] = round(bytes_to_mb(int(total_bytes)), 6)

        by_op: List[Dict[str, Any]] = []
        if group_field:
            q = f"""
                SELECT {group_field} AS op,
                       SUM(end - start) AS total_ns,
                       COUNT(*) AS count
                       {', SUM(' + bytes_col + ') AS total_bytes' if bytes_col else ''}
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                GROUP BY {group_field}
                ORDER BY total_ns DESC;
            """
            for row in cur.execute(q).fetchall():
                op = row[0]
                total_ns = int(row[1] or 0)
                count = int(row[2] or 0)
                item = {
                    "op": op,
                    "total_s": round(ns_to_s(total_ns), 6),
                    "count": count,
                }
                if bytes_col:
                    total_bytes = int(row[3] or 0)
                    item["total_mb"] = round(bytes_to_mb(total_bytes), 6)
                by_op.append(item)

        memcpy_summary["by_op"] = by_op
        memcpy_summary["notes"] = [
            "The op field is an enum value; mapping to H2D/D2H/D2D is optional for baseline.",
            "Totals are still correct baseline evidence even before enum decoding.",
        ]

    # ----------------------------
    # 5) CUDA graph node events (optional)
    # ----------------------------
    graph_summary: Dict[str, Any] = {"present": False}
    if table_exists(cur, "CUDA_GRAPH_NODE_EVENTS"):
        graph_summary["present"] = True
        n = fetch_scalar(cur, "SELECT COUNT(*) FROM CUDA_GRAPH_NODE_EVENTS;") or 0
        graph_summary["node_events_rows"] = int(n)

    conn.close()

    summary: Dict[str, Any] = {
        "type": "nsight_systems_baseline_summary",
        "sqlite": db,
        "capture_window_s": round(ns_to_s(capture_window_ns), 6),
        "cpu_osrt": osrt_summary,
        "gpu_kernels": gpu_summary,
        "gpu_memcpy": memcpy_summary,
        "cuda_graph": graph_summary,
        "baseline_notes": [
            "Evidence-first baseline profiling (not optimization).",
            "GPU kernel/memcpy tables present => child/worker process tracing worked.",
        ],
    }

    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
