from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from transformers import AutoTokenizer


# -------------------------- utils --------------------------

def pctl(values: List[float], q: float) -> Optional[float]:
    """Percentile without numpy. q in [0,100]."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    # linear interpolation
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def read_prompts_jsonl(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("prompt")
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
    return prompts


def default_max_model_len() -> int:
    for key in ("MAX_MODEL_LEN", "VLLM_MAX_MODEL_LEN"):
        val = os.environ.get(key)
        if val:
            try:
                return int(val)
            except Exception:
                continue
    return 2048


def select_prompts(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    max_input_tokens: int,
    num_requests: int,
) -> List[str]:
    ok = []
    for p in prompts:
        try:
            input_tokens = len(tokenizer.encode(p, add_special_tokens=False))
        except Exception:
            continue
        if input_tokens <= max_input_tokens:
            ok.append(p)
    if not ok:
        raise SystemExit("No prompts within max_input_tokens; adjust MAX_INPUT_TOKENS or dataset.")
    if len(ok) >= num_requests:
        return ok[:num_requests]
    # Repeat prompts deterministically if we need more than the filtered pool.
    out: List[str] = []
    idx = 0
    while len(out) < num_requests:
        out.append(ok[idx % len(ok)])
        idx += 1
    return out


# -------------------------- SSE streaming --------------------------

async def sse_stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Tuple[int, Optional[float], int, Optional[str]]:
    """
    Robust SSE parser:
      - aiohttp yields arbitrary byte chunks, not line-aligned
      - we buffer and split on '\\n'
    Returns:
      (http_status, first_token_ts, content_chunks, error)
    """
    first_token_ts: Optional[float] = None
    content_chunks = 0
    buf = ""

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            status = resp.status

            if status != 200:
                txt = await resp.text()
                return status, None, 0, f"HTTP {status}: {txt[:200]}"

            async for chunk in resp.content.iter_chunked(4096):
                if not chunk:
                    continue
                buf += chunk.decode("utf-8", errors="ignore")

                # Process complete lines
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()
                    if not data:
                        continue

                    if data == "[DONE]":
                        return status, first_token_ts, content_chunks, None

                    if first_token_ts is None:
                        first_token_ts = time.time()

                    # Each SSE event is JSON
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    # Count only actual content deltas (ignore role-only events)
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            content_chunks += 1

            # Stream ended without [DONE]
            return status, first_token_ts, content_chunks, None

    except Exception as e:
        return 0, None, 0, str(e)


# -------------------------- benchmark core --------------------------

@dataclass
class RequestResult:
    idx: int
    prompt_len: int
    request_sent_ts: float
    first_token_ts: Optional[float]
    last_token_ts: float
    http_status: int
    output_tokens: int  # we use "content chunk count" as a proxy
    error: Optional[str]


async def run_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_s: float,
    idx: int,
) -> RequestResult:
    async with sem:
        request_sent_ts = time.time()

        payload = {
            "model": model,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        status, first_token_ts, chunks, err = await sse_stream_chat(
            session=session,
            url=url,
            payload=payload,
            timeout_s=timeout_s,
        )

        last_token_ts = time.time()

        return RequestResult(
            idx=idx,
            prompt_len=len(prompt),
            request_sent_ts=request_sent_ts,
            first_token_ts=first_token_ts,
            last_token_ts=last_token_ts,
            http_status=status,
            output_tokens=chunks,
            error=err,
        )


async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    prompts_path = Path(args.prompts)
    prompts = read_prompts_jsonl(prompts_path)
    if not prompts:
        raise SystemExit(f"No prompts loaded from {prompts_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    selected = select_prompts(
        prompts=prompts,
        tokenizer=tokenizer,
        max_input_tokens=args.max_input_tokens,
        num_requests=args.num_requests,
    )

    url = f"http://{args.host}:{args.port}{args.endpoint}"

    sem = asyncio.Semaphore(args.concurrency)

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)

    t0 = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, p in enumerate(selected):
            tasks.append(
                asyncio.create_task(
                    run_one(
                        sem=sem,
                        session=session,
                        url=url,
                        model=args.model,
                        prompt=p,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        timeout_s=args.timeout_s,
                        idx=i,
                    )
                )
            )

        results: List[RequestResult] = await asyncio.gather(*tasks)
    t1 = time.time()

    # Only successful requests should count for stats
    ok_results = [r for r in results if r.http_status == 200 and r.error is None]

    total_ms: List[float] = []
    ttft_ms: List[float] = []
    total_chunks = 0

    for r in ok_results:
        total_ms.append((r.last_token_ts - r.request_sent_ts) * 1000.0)
        if r.first_token_ts is not None:
            ttft_ms.append((r.first_token_ts - r.request_sent_ts) * 1000.0)
        total_chunks += int(r.output_tokens or 0)

    # Define benchmark wall time as [first send .. last token] over successful requests
    wall_s = (
        (max(r.last_token_ts for r in ok_results) - min(r.request_sent_ts for r in ok_results))
        if ok_results
        else 0.0
    )

    throughput_tok_s = (total_chunks / wall_s) if wall_s > 0 else 0.0

    stats = {
        "p50_total_ms": pctl(total_ms, 50),
        "p95_total_ms": pctl(total_ms, 95),
        "p50_ttft_ms": pctl(ttft_ms, 50),
        "p95_ttft_ms": pctl(ttft_ms, 95),
        "throughput_tok_s": round(float(throughput_tok_s), 6),
        "wall_s": round(float(wall_s), 6),
    }

    run = {
        "host": args.host,
        "port": args.port,
        "endpoint": args.endpoint,
        "model": args.model,
        "prompts_file": str(prompts_path),
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "timeout_s": args.timeout_s,
        "ok_requests": len(ok_results),
        "failed_requests": len(results) - len(ok_results),
        "total_elapsed_s": round(t1 - t0, 6),
    }

    # Save a small sample of failures (helps debugging)
    failures = []
    for r in results:
        if r.http_status != 200 or r.error is not None:
            failures.append(
                {
                    "idx": r.idx,
                    "http_status": r.http_status,
                    "error": r.error,
                }
            )
        if len(failures) >= 10:
            break

    out = {
        "run": run,
        "stats": stats,
        "failures_sample": failures,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--model", default="models/tinyllama")

    ap.add_argument("--prompts", required=True)
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)

    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", dest="timeout_s", type=float, default=180.0)
    ap.add_argument("--max-model-len", type=int, default=default_max_model_len())
    ap.add_argument("--max-input-tokens", type=int, default=None)

    ap.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.max_input_tokens is None:
        args.max_input_tokens = max(1, int(args.max_model_len) - int(args.max_new_tokens) - 32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asyncio.run(main_async(args))
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"✅ Wrote {out_path}")
    print(json.dumps(payload["stats"], indent=2))


if __name__ == "__main__":
    main()
