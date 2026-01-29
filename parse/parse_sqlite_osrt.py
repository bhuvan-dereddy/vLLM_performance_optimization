import argparse
import json
import sqlite3
from pathlib import Path

def ns_to_s(ns: int) -> float:
    return float(ns) / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path", help="Path to .sqlite exported by nsys")
    ap.add_argument("--out", default="artifacts/summary.json", help="Output JSON path")
    ap.add_argument("--top", type=int, default=10, help="Top N OSRT calls to keep")
    args = ap.parse_args()

    db = args.sqlite_path
    out = args.out
    top_n = args.top

    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # Capture window (matches UI timeline extent)
    cur.execute("SELECT MIN(start), MAX(end) FROM OSRT_API")
    start_ns, end_ns = cur.fetchone()
    if start_ns is None or end_ns is None:
        raise SystemExit("OSRT_API has no rows; did you capture --trace=osrt ?")

    window_ns = int(end_ns - start_ns)
    if window_ns <= 0:
        raise SystemExit("Capture window is non-positive; unexpected trace timestamps")

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
        (top_n,)
    ).fetchall()

    conn.close()

    top_calls = []
    for name, total_ns, calls in rows:
        total_ns = int(total_ns or 0)
        calls = int(calls or 0)
        pct = (total_ns / window_ns) * 100.0
        top_calls.append({
            "name": name,
            "total_s": round(ns_to_s(total_ns), 6),
            "calls": calls,
            "pct_of_capture_window": round(pct, 3),
        })

    summary = {
        "type": "nsight_systems_osrt_summary",
        "sqlite": db,
        "capture_window_s": round(ns_to_s(window_ns), 6),
        "top_calls": top_calls,
        "notes": [
            "OSRT time is summed across threads, so totals can exceed capture window.",
            "poll/epoll_wait/pthread_cond_wait usually mean CPU threads are waiting for work or synchronization."
        ],
    }

    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
