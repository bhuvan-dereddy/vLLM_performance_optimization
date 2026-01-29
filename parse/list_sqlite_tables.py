import sqlite3
from pathlib import Path

DB = Path("/home/ubuntu/gpu-profiling/artifacts/vllm_capture.sqlite")

def main():
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()

    out = Path("/home/ubuntu/gpu-profiling/artifacts/sqlite_tables.txt")
    out.write_text("\n".join(tables) + "\n")
    print(f"Wrote {out} ({len(tables)} tables)")

if __name__ == "__main__":
    main()
