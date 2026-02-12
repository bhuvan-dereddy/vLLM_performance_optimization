import os
import shlex
from typing import Any, Dict, List, Sequence, Tuple


def resolve_server(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], str, int, float]:
    if "serve_cmd" in cfg:
        raw = cfg["serve_cmd"]
        if isinstance(raw, str):
            serve_cmd = shlex.split(raw)
        elif isinstance(raw, list):
            serve_cmd = [str(x) for x in raw]
        else:
            raise SystemExit("serve_cmd must be string or list")
    else:
        server = cfg.get("server", {})
        serve_cmd = [
            "vllm",
            "serve",
            str(server["model_dir"]),
            "--host",
            str(server.get("host", "127.0.0.1")),
            "--port",
            str(server.get("port", 8000)),
            "--dtype",
            str(server.get("dtype", "float16")),
            "--gpu-memory-utilization",
            str(server.get("gpu_memory_utilization", 0.9)),
            "--max-model-len",
            str(server.get("max_model_len", 2048)),
        ]

    env = dict(os.environ)
    env_cfg = cfg.get("env", {})
    if not env_cfg and isinstance(cfg.get("server"), dict):
        env_cfg = cfg["server"].get("env", {})
    for k, v in (env_cfg or {}).items():
        env[str(k)] = str(v)

    host = cfg.get("workload", {}).get("host")
    port = cfg.get("workload", {}).get("port")
    timeout_s = cfg.get("server", {}).get("startup_timeout_s", 240)

    if host is None:
        host = _arg_value(serve_cmd, "--host", default="127.0.0.1")
    if port is None:
        port = int(_arg_value(serve_cmd, "--port", default="8000"))

    return serve_cmd, env, str(host), int(port), float(timeout_s)


def _arg_value(cmd: Sequence[str], key: str, default: str) -> str:
    for i, tok in enumerate(cmd):
        if tok == key and i + 1 < len(cmd):
            return cmd[i + 1]
    return default


def resolve_model_name(cmd: Sequence[str], cfg: Dict[str, Any]) -> str:
    if len(cmd) >= 3 and cmd[0] == "vllm" and cmd[1] == "serve":
        return str(cmd[2])
    if "--model" in cmd:
        return _arg_value(cmd, "--model", default="models/tinyllama")
    server = cfg.get("server", {})
    if "model_dir" in server:
        return str(server["model_dir"])
    return "models/tinyllama"


def get_knob_flags(knobs: List[Dict[str, Any]], bits: Sequence[int]) -> Tuple[str, List[str], Dict[str, bool]]:
    mask = "".join(str(int(b)) for b in bits)
    flags: List[str] = []
    assignment: Dict[str, bool] = {}
    for b, knob in zip(bits, knobs):
        name = str(knob["name"])
        on_flags = [str(x) for x in knob.get("on_flags", [])]
        off_flags = [str(x) for x in knob.get("off_flags", [])]
        if not on_flags and "flag" in knob and "on_value" in knob:
            on_flags = [str(knob["flag"]), str(knob["on_value"])]
        elif not on_flags and "flag" in knob and knob.get("type") == "bool":
            on_flags = [str(knob["flag"])]
        assignment[name] = bool(b)
        flags.extend(on_flags if b else off_flags)
    return mask, flags, assignment
