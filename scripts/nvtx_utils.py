from __future__ import annotations

from contextlib import contextmanager

try:
    import nvtx  # pip package: nvtx
except Exception:  # pragma: no cover
    nvtx = None


@contextmanager
def nvtx_range(name: str):
    """
    Safe NVTX range context manager.

    - If nvtx is installed, you get visible markers in Nsight Systems.
    - If not installed, it becomes a no-op (your script still runs).
    """
    if nvtx is None:
        yield
        return

    nvtx.push_range(name)
    try:
        yield
    finally:
        nvtx.pop_range()