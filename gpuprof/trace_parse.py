from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlite3
import subprocess


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    return cur.fetchone() is not None


def column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def find_nsys() -> str:
    candidates = [
        "nsys",
        "/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys",
        "/opt/nvidia/nsight-systems/2025.5.1/target-linux-x64/nsys",
    ]
    for c in candidates:
        try:
            p = subprocess.run([c, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p.returncode == 0:
                return c
        except Exception:
            continue
    raise SystemExit("nsys not found")


def read_phase_ms(cur: sqlite3.Cursor, phase_name: str) -> Optional[float]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return None
    has_stringids = table_exists(cur, "StringIds")
    if has_stringids:
        row = cur.execute(
            """
            SELECT SUM(n.end - n.start)
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON s.id = n.textId
            WHERE n.end IS NOT NULL AND n.end > n.start
              AND (n.text = ? OR s.value = ?)
            """,
            (phase_name, phase_name),
        ).fetchone()
    else:
        row = cur.execute(
            """
            SELECT SUM(n.end - n.start)
            FROM NVTX_EVENTS n
            WHERE n.end IS NOT NULL AND n.end > n.start
              AND n.text = ?
            """,
            (phase_name,),
        ).fetchone()
    total_ns = row[0] if row else None
    if total_ns is None:
        return None
    return float(total_ns) / 1e6


def parse_trace_sqlite(sqlite_path: Path) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()

    gpu_compute_ms: Optional[float] = None
    memcpy_ms: Optional[float] = None
    osrt_wait_ms: Optional[float] = None

    top_kernels: List[Dict[str, Any]] = []

    if table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        row = cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        total_kernel_ns = row[0] if row else None
        if total_kernel_ns is not None:
            gpu_compute_ms = float(total_kernel_ns) / 1e6

        name_expr = "'<unknown_kernel_name>'"
        join_clause = ""
        if column_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"):
            if table_exists(cur, "StringIds"):
                sample = cur.execute("SELECT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1").fetchone()
                if sample and sample[0] is not None and str(sample[0]).isdigit():
                    name_expr = "s.value"
                    join_clause = "LEFT JOIN StringIds s ON s.id = CAST(k.demangledName AS INTEGER)"
                else:
                    name_expr = "k.demangledName"
            else:
                name_expr = "k.demangledName"
        elif column_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL", "nameId") and table_exists(cur, "StringIds"):
            name_expr = "s.value"
            join_clause = "LEFT JOIN StringIds s ON s.id = k.nameId"

        rows = cur.execute(
            f"""
            SELECT {name_expr} AS name, SUM(k.end - k.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            {join_clause}
            GROUP BY name
            ORDER BY total_ns DESC
            LIMIT 5
            """
        ).fetchall()

        total_ns_for_pct = float(total_kernel_ns or 0.0)
        for name, total_ns in rows:
            total_ms = float(total_ns or 0.0) / 1e6
            pct = (float(total_ns or 0.0) / total_ns_for_pct * 100.0) if total_ns_for_pct > 0 else None
            top_kernels.append(
                {
                    "name": str(name) if name is not None else "<unknown_kernel_name>",
                    "total_ms": total_ms,
                    "percent": pct,
                }
            )

    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        row = cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()
        if row and row[0] is not None:
            memcpy_ms = float(row[0]) / 1e6

    if table_exists(cur, "OSRT_API") and table_exists(cur, "StringIds"):
        row = cur.execute(
            """
            SELECT SUM(a.end - a.start)
            FROM OSRT_API a
            JOIN StringIds s ON s.id = a.nameId
            WHERE LOWER(s.value) LIKE '%wait%'
               OR LOWER(s.value) LIKE '%poll%'
               OR LOWER(s.value) LIKE '%sleep%'
               OR LOWER(s.value) LIKE '%futex%'
               OR LOWER(s.value) LIKE '%cond%'
               OR LOWER(s.value) LIKE '%select%'
            """
        ).fetchone()
        if row and row[0] is not None:
            osrt_wait_ms = float(row[0]) / 1e6

    nvtx = {
        "get_batch_ms": read_phase_ms(cur, "get_batch"),
        "prefill_ms": read_phase_ms(cur, "prefill"),
        "decode_ms": read_phase_ms(cur, "decode"),
    }

    conn.close()

    summary = {
        "gpu_compute_ms": gpu_compute_ms,
        "memcpy_ms": memcpy_ms,
        "osrt_wait_ms": osrt_wait_ms,
        "top_kernels": top_kernels,
    }
    return nvtx, summary
