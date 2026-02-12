from pathlib import Path
from typing import Optional
from urllib import request

import os
import signal
import subprocess
import time


def http_get(url: str, timeout_s: float = 2.0) -> int:
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        return resp.getcode()


def wait_for_ready(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    while time.time() < deadline:
        try:
            if http_get(url) == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def terminate_process_group(proc: subprocess.Popen, grace_s: float = 12.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def wait_for_report_finalized(trace_rep: Path, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last_size: Optional[int] = None
    stable_for = 0.0

    while time.time() < deadline:
        if not trace_rep.exists():
            time.sleep(0.5)
            continue

        size = trace_rep.stat().st_size
        if last_size is None or size != last_size:
            last_size = size
            stable_for = 0.0
        else:
            stable_for += 0.5
            if stable_for >= 2.0:
                return
        time.sleep(0.5)

    raise SystemExit(f"Timed out waiting for finalized report: {trace_rep}")
