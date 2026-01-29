import sqlite3

DB = "artifacts/vllm_capture.sqlite"

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS' LIMIT 1;")
    if cur.fetchone() is None:
        print("NVTX_EVENTS table not found (NVTX data missing).")
        conn.close()
        return

    cur.execute("PRAGMA table_info(NVTX_EVENTS);")
    cols = cur.fetchall()
    conn.close()

    for c in cols:
        print(c)

if __name__ == "__main__":
    main()
