from typing import Any


def format_num(value: Any) -> str:
    if value is None:
        return "na"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "na"


def format_delta(base: Any, value: Any) -> str:
    try:
        b = float(base)
        v = float(value)
        return f"{(v - b):.6f}"
    except Exception:
        return "na"
