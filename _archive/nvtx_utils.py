from __future__ import annotations
from contextlib import contextmanager

try:
    import nvtx
except Exception:
    nvtx = None


@contextmanager
def nvtx_range(name: str):
    if nvtx is None:
        yield
        return
    nvtx.push_range(name)
    try:
        yield
    finally:
        nvtx.pop_range()
