#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, parse_float=Decimal)


def parse_mask(mask: str, expected_len: int) -> List[int]:
    if any(ch not in "01" for ch in mask):
        raise SystemExit("--mask must contain only 0/1")
    if len(mask) != expected_len:
        raise SystemExit(f"--mask length must be {expected_len}, got {len(mask)}")
    return [int(ch) for ch in mask]


def base_serve_cmd(cfg: Dict[str, Any]) -> List[str]:
    """
    Build a vLLM OpenAI-compatible server command that is safe for
    non-interactive SSH execution (no PATH reliance).
    """
    server = cfg.get("server", {})

    # Hard-bind to the repo venv python so PATH never matters.
    # If you move the repo, update this one constant.
    repo_dir = "/home/ubuntu/gpu-profiling"
    venv_python = f"{repo_dir}/venv/bin/python"

    model_dir = str(server.get("model_dir", "models/tinyllama"))
    host = str(server.get("host", "127.0.0.1"))
    port = str(server.get("port", 8000))
    dtype = str(server.get("dtype", "float16"))
    gpu_mem = server.get("gpu_memory_utilization", Decimal("0.90"))
    max_model_len = str(server.get("max_model_len", 2048))

    gpu_mem_str = str(gpu_mem)

    # Use module invocation to avoid relying on a `vllm` CLI being on PATH.
    # This runs the OpenAI-compatible server.
    return [
        venv_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        gpu_mem_str,
        "--max-model-len",
        max_model_len,
    ]


def get_knob_flags(
    knobs: Sequence[Dict[str, Any]],
    bits: Sequence[int],
    enabled_only: bool,
) -> List[str]:
    flags: List[str] = []
    for bit, knob in zip(bits, knobs):
        on_flags = [str(x) for x in knob.get("on_flags", [])]
        off_flags = [str(x) for x in knob.get("off_flags", [])]

        # Back-compat mappings (keep semantics)
        if not on_flags and "flag" in knob and "on_value" in knob:
            on_flags = [str(knob["flag"]), str(knob["on_value"])]
        elif not on_flags and "flag" in knob and knob.get("type") == "bool":
            on_flags = [str(knob["flag"])]

        if bit:
            flags.extend(on_flags)
        elif not enabled_only:
            flags.extend(off_flags)

    return flags


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "configs" / "optimizations_6knobs.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(default_config))
    ap.add_argument("--mask", required=True, help="Bitstring in knob order, e.g. 010101")
    ap.add_argument("--print-flags", action="store_true", help="Print enabled flags only")
    ap.add_argument("--run", action="store_true", default=False, help="Exec vLLM serve command")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    knobs = cfg.get("knobs", [])
    bits = parse_mask(args.mask, len(knobs))

    if args.print_flags:
        enabled_flags = get_knob_flags(knobs, bits, enabled_only=True)
        print(shell_join(enabled_flags))
        return

    cmd = base_serve_cmd(cfg)
    cmd.extend(get_knob_flags(knobs, bits, enabled_only=False))

    if args.run:
        print(f"Starting vLLM for mask {args.mask}", file=sys.stderr, flush=True)
        os.execvp(cmd[0], cmd)

    print(shell_join(cmd))


if __name__ == "__main__":
    main()