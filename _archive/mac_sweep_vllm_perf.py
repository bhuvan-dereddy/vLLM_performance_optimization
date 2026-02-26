#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence


MASK_BITS = 6
MIN_VALID_JSON_BYTES = 200
BENCH_ENDPOINT = "/v1/chat/completions"
BENCH_MAX_NEW_TOKENS = 64


@dataclass
class Settings:
    ec2_host: str
    ec2_user: str
    ssh_key: str
    ec2_repo_dir: str
    local_port: int
    bench_client: Path
    dataset_path: Path
    remote_out_dir_raw: str
    out_dir: Path
    num_prompts: int
    concurrency: int


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def env_int(name: str, default_str: str) -> int:
    raw = os.environ.get(name, default_str).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {raw}") from exc


def load_settings() -> Settings:
    out_dir_raw = os.environ.get("OUT_DIR", "~/out_vllmperf")
    return Settings(
        ec2_host=env_required("EC2_HOST"),
        ec2_user=os.environ.get("EC2_USER", "ubuntu"),
        ssh_key=env_required("SSH_KEY"),
        ec2_repo_dir=os.environ.get("EC2_REPO_DIR", "/home/ubuntu/gpu-profiling"),
        local_port=env_int("LOCAL_PORT", "8001"),
        bench_client=Path(os.path.expanduser(os.environ.get("BENCH_CLIENT", "~/bench_client.py"))),
        dataset_path=Path(os.path.expanduser(os.environ.get("DATASET_PATH", "~/sharegpt_prompts.jsonl"))),
        remote_out_dir_raw=out_dir_raw,
        out_dir=Path(os.path.expanduser(out_dir_raw)),
        num_prompts=env_int("NUM_PROMPTS", "100"),
        concurrency=env_int("CONCURRENCY", "1"),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Run only one mask, e.g. 010101")
    ap.add_argument("--start", help="Start mask (inclusive), e.g. 000000")
    ap.add_argument("--end", help="End mask (inclusive), e.g. 111111")
    return ap.parse_args()


def validate_mask(mask: str) -> str:
    if len(mask) != MASK_BITS or any(ch not in "01" for ch in mask):
        raise SystemExit(f"Invalid mask: {mask}. Expected {MASK_BITS} bits of 0/1.")
    return mask


def resolve_masks(args: argparse.Namespace) -> List[str]:
    if args.only:
        if args.start or args.end:
            raise SystemExit("--only cannot be combined with --start/--end")
        return [validate_mask(args.only)]

    if args.start or args.end:
        if not args.start or not args.end:
            raise SystemExit("Provide both --start and --end")
        start = int(validate_mask(args.start), 2)
        end = int(validate_mask(args.end), 2)
        if start > end:
            raise SystemExit("--start must be <= --end")
        return [format(i, f"0{MASK_BITS}b") for i in range(start, end + 1)]

    return [format(i, f"0{MASK_BITS}b") for i in range(2 ** MASK_BITS)]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_valid_json_file(path: Path, min_bytes: int = MIN_VALID_JSON_BYTES) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size <= min_bytes:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def ssh_base(settings: Settings) -> List[str]:
    target = f"{settings.ec2_user}@{settings.ec2_host}"
    return ["ssh", "-i", settings.ssh_key, target]


def run_ssh(settings: Settings, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ssh_base(settings) + [f"bash -lc {shlex.quote(remote_cmd)}"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            f"SSH command failed ({proc.returncode}): {remote_cmd}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return proc


def append_csv_row(
    results_csv: Path,
    row: Sequence[str],
) -> None:
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "mask",
                    "status",
                    "raw_json",
                    "remote_log",
                    "client_log",
                    "started_at_iso",
                ]
            )
        writer.writerow(row)


def stop_remote_server(settings: Settings, mask: str) -> None:
    remote_cmd = "\n".join(
        [
            "set +e",
            f'pid_file="/tmp/vllm_sweep_{mask}.pid"',
            'if [ -f "$pid_file" ]; then',
            '  pid="$(cat "$pid_file")"',
            '  kill "$pid" 2>/dev/null || true',
            "  sleep 2",
            '  kill -9 "$pid" 2>/dev/null || true',
            '  rm -f "$pid_file"',
            "fi",
            "lsof -t -iTCP:8000 -sTCP:LISTEN | xargs -r kill -9 || true",
        ]
    )
    run_ssh(settings, remote_cmd, check=False)


def start_remote_server(settings: Settings, mask: str) -> str:
    remote_log = f"{settings.remote_out_dir_raw.rstrip('/')}/server_logs/{mask}.server.log"
    remote_cmd = "\n".join(
        [
            "set -e",
            f"cd {shlex.quote(settings.ec2_repo_dir)}",
            f"OUT_DIR_RAW={shlex.quote(settings.remote_out_dir_raw)}",
            'if [ "$OUT_DIR_RAW" = "~" ]; then',
            '  OUT_DIR="$HOME"',
            'elif [[ "$OUT_DIR_RAW" == ~/* ]]; then',
            '  OUT_DIR="$HOME/${OUT_DIR_RAW#~/}"',
            "else",
            '  OUT_DIR="$OUT_DIR_RAW"',
            "fi",
            'mkdir -p "$OUT_DIR/server_logs"',
            "lsof -t -iTCP:8000 -sTCP:LISTEN | xargs -r kill -9 || true",
            f'nohup python3 scripts/serve_cmd_for_mask.py --mask {shlex.quote(mask)} --run > "$OUT_DIR/server_logs/{mask}.server.log" 2>&1 &',
            f'echo $! > /tmp/vllm_sweep_{mask}.pid',
        ]
    )
    run_ssh(settings, remote_cmd, check=True)
    return remote_log


def wait_ready(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_err = "unknown error"
    while time.time() < deadline:
        proc = subprocess.run(["curl", "-sSf", url], text=True, capture_output=True)
        if proc.returncode == 0:
            return
        last_err = (proc.stderr or proc.stdout or f"curl exited with {proc.returncode}").strip()
        time.sleep(0.5)
    raise TimeoutError(
        f"Timed out after {int(timeout_s)}s waiting for {url}. Last curl error: {last_err}"
    )


def run_local_logged(cmd: Sequence[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"\n$ {' '.join(shlex.quote(x) for x in cmd)}\n")
        log_f.flush()
        proc = subprocess.run(cmd, text=True, stdout=log_f, stderr=subprocess.STDOUT)
        log_f.write(f"[exit_code={proc.returncode}]\n")
        return int(proc.returncode)


def run_bench_client(
    settings: Settings,
    raw_json: Path,
    client_log: Path,
) -> None:
    cmd = [
        "python3",
        str(settings.bench_client),
        "--skip-tokenizer",
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.local_port),
        "--endpoint",
        BENCH_ENDPOINT,
        "--prompts",
        str(settings.dataset_path),
        "--num-requests",
        str(settings.num_prompts),
        "--concurrency",
        str(settings.concurrency),
        "--max-new-tokens",
        str(BENCH_MAX_NEW_TOKENS),
        "--temperature",
        "0.0",
        "--timeout-s",
        "180",
        "--out",
        str(raw_json),
    ]

    raw_json.parent.mkdir(parents=True, exist_ok=True)
    if raw_json.exists():
        raw_json.unlink()

    rc = run_local_logged(cmd, client_log)
    if rc != 0:
        raise RuntimeError(f"bench_client failed with exit code {rc}")
    if not is_valid_json_file(raw_json):
        raise RuntimeError(f"Invalid benchmark output JSON: {raw_json}")


def add_output_meta(
    raw_json: Path,
    mask: str,
    settings: Settings,
    endpoint: str = BENCH_ENDPOINT,
    max_new_tokens: int = BENCH_MAX_NEW_TOKENS,
) -> None:
    with raw_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected top-level JSON object in {raw_json}")

    meta_obj = payload.get("_meta")
    if not isinstance(meta_obj, dict):
        meta_obj = {}
    meta_obj.update(
        {
            "mask": mask,
            "local_port": settings.local_port,
            "num_requests": settings.num_prompts,
            "concurrency": settings.concurrency,
            "max_new_tokens": max_new_tokens,
            "endpoint": endpoint,
            "timestamp_epoch": int(time.time()),
        }
    )
    payload["_meta"] = meta_obj

    raw_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = load_settings()
    masks = resolve_masks(args)

    out_dir = settings.out_dir
    raw_dir = out_dir / "raw"
    logs_dir = out_dir / "logs"
    server_logs_dir = out_dir / "server_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    server_logs_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"

    for mask in masks:
        started_at = now_iso()
        raw_json = raw_dir / f"{mask}.json"
        client_log = logs_dir / f"{mask}.client.log"
        remote_log = f"{settings.remote_out_dir_raw.rstrip('/')}/server_logs/{mask}.server.log"
        status = "fail"

        if is_valid_json_file(raw_json):
            status = "skip"
            print(f"[{mask}] skip (existing valid JSON): {raw_json}")
            append_csv_row(
                results_csv,
                [mask, status, str(raw_json), remote_log, str(client_log), started_at],
            )
            continue

        try:
            print(f"[{mask}] launching remote server")
            remote_log = start_remote_server(settings, mask)
            ready_url = f"http://127.0.0.1:{settings.local_port}/v1/models"
            print(f"[{mask}] waiting for tunnel readiness on {ready_url}")
            wait_ready(ready_url)
            print(f"[{mask}] running local bench_client")
            run_bench_client(settings=settings, raw_json=raw_json, client_log=client_log)
            add_output_meta(raw_json=raw_json, mask=mask, settings=settings)
            status = "ok"
            print(f"[{mask}] ok")
        except Exception as exc:
            print(f"[{mask}] fail: {exc}", file=sys.stderr)
            client_log.parent.mkdir(parents=True, exist_ok=True)
            with client_log.open("a", encoding="utf-8") as f:
                f.write(f"\nERROR: {exc}\n")
        finally:
            try:
                stop_remote_server(settings, mask)
            except Exception as stop_exc:
                with client_log.open("a", encoding="utf-8") as f:
                    f.write(f"ERROR during remote stop: {stop_exc}\n")
            append_csv_row(
                results_csv,
                [mask, status, str(raw_json), remote_log, str(client_log), started_at],
            )


if __name__ == "__main__":
    main()
