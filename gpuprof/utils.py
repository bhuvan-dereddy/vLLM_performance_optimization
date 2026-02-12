from pathlib import Path
from typing import Any

import json


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


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
