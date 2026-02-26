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
        scalar_flag_map = {
            "max_num_seqs": "--max-num-seqs",
            "swap_space": "--swap-space",
            "block_size": "--block-size",
            "kv_cache_dtype": "--kv-cache-dtype",
        }
        for key, flag in scalar_flag_map.items():
            if key in server and server[key] is not None:
                serve_cmd.extend([flag, str(server[key])])

        bool_flag_map = {
            "enable_prefix_caching": "--enable-prefix-caching",
            "disable_log_requests": "--disable-log-requests",
            "enable_chunked_prefill": "--enable-chunked-prefill",
            "enforce_eager": "--enforce-eager",
        }
        for key, flag in bool_flag_map.items():
            if bool(server.get(key, False)):
                serve_cmd.append(flag)

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


def _arg_int(cmd: Sequence[str], key: str) -> int | None:
    for i, tok in enumerate(cmd):
        if tok == key:
            if i + 1 >= len(cmd):
                raise RuntimeError(f"{key} is present but has no value")
            return int(cmd[i + 1])
    return None


def validate_server_cmd(cmd: Sequence[str]) -> None:
    max_num_batched_tokens = _arg_int(cmd, "--max-num-batched-tokens")
    if max_num_batched_tokens is None:
        return
    max_model_len = _arg_int(cmd, "--max-model-len")
    if max_model_len is None:
        raise RuntimeError("--max-model-len must be present when --max-num-batched-tokens is set")
    if max_num_batched_tokens < max_model_len:
        raise RuntimeError(
            f"max_num_batched_tokens ({max_num_batched_tokens}) must be >= max_model_len ({max_model_len})"
        )


def assemble_server_cmd(serve_cmd: Sequence[str], knob_flags: Sequence[str]) -> List[str]:
    cmd = [str(x) for x in serve_cmd] + [str(x) for x in knob_flags]
    validate_server_cmd(cmd)
    return cmd


def resolve_model_name(cmd: Sequence[str], cfg: Dict[str, Any]) -> str:
    if len(cmd) >= 3 and cmd[0] == "vllm" and cmd[1] == "serve":
        return str(cmd[2])
    if "--model" in cmd:
        return _arg_value(cmd, "--model", default="models/tinyllama")
    server = cfg.get("server", {})
    if "model_dir" in server:
        return str(server["model_dir"])
    return "models/tinyllama"


def knob_options(knob: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = str(knob.get("name", "knob"))
    off_flags = [str(x) for x in knob.get("off_flags", [])]
    if "values" in knob:
        values = knob.get("values")
        if not isinstance(values, list) or not values:
            raise SystemExit(f"knob {name}: values must be a non-empty list")
        options: List[Dict[str, Any]] = []
        for idx, value in enumerate(values):
            if isinstance(value, dict):
                if "value" not in value:
                    raise SystemExit(f"knob {name}: values[{idx}] missing value")
                option_value = value["value"]
                if "flags" in value:
                    flags = [str(x) for x in value.get("flags", [])]
                elif isinstance(option_value, bool):
                    flag = knob.get("flag")
                    if flag is None:
                        raise SystemExit(f"knob {name}: flag is required for boolean values")
                    flags = [str(flag)] if option_value else off_flags
                else:
                    flags = []
            else:
                option_value = value
                flag = knob.get("flag")
                if flag is None:
                    raise SystemExit(f"knob {name}: flag is required for scalar values")
                if isinstance(option_value, bool):
                    # OFF must be explicit if off_flags are provided.
                    flags = [str(flag)] if option_value else off_flags
                else:
                    flags = [str(flag), str(option_value)]
            options.append(
                {
                    "choice_index": idx,
                    "value": option_value,
                    "flags": flags,
                }
            )
        return options

    on_flags = [str(x) for x in knob.get("on_flags", [])]
    if not on_flags and "flag" in knob and "on_value" in knob:
        on_flags = [str(knob["flag"]), str(knob["on_value"])]
    elif not on_flags and "flag" in knob and knob.get("type") == "bool":
        on_flags = [str(knob["flag"])]

    return [
        {"choice_index": 0, "value": False, "flags": off_flags},
        {"choice_index": 1, "value": True, "flags": on_flags},
    ]


def get_knob_flags(knobs: List[Dict[str, Any]], bits: Sequence[int]) -> Tuple[str, List[str], Dict[str, Any]]:
    mask = "".join(str(int(b)) for b in bits)
    flags: List[str] = []
    assignment: Dict[str, Any] = {}
    for b, knob in zip(bits, knobs):
        name = str(knob["name"])
        options = knob_options(knob)
        idx = int(b)
        if idx < 0 or idx >= len(options):
            raise SystemExit(f"knob {name}: invalid choice index {idx}")
        choice = options[idx]
        assignment[name] = choice["value"]
        flags.extend([str(x) for x in choice["flags"]])
    return mask, flags, assignment
