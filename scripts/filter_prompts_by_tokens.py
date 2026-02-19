#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="output_path", required=True, help="Output JSONL path")
    ap.add_argument("--model", required=True, help="Tokenizer model path/name")
    ap.add_argument("--max-input-tokens", type=int, required=True)
    return ap.parse_args()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def extract_prompt_text(obj: dict[str, Any]) -> str:
    prompt = obj.get("prompt")
    if isinstance(prompt, str):
        return prompt

    messages = obj.get("messages")
    if not isinstance(messages, list):
        return ""

    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        role_text = role if isinstance(role, str) else ""
        content_text = content_to_text(msg.get("content"))
        if role_text and content_text:
            lines.append(f"{role_text}: {content_text}")
        elif content_text:
            lines.append(content_text)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    total = 0
    kept = 0
    dropped = 0
    max_tokens_observed = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as in_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for raw_line in in_f:
            line = raw_line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except Exception:
                dropped += 1
                continue

            if not isinstance(obj, dict):
                dropped += 1
                continue

            prompt_text = extract_prompt_text(obj)
            if not prompt_text:
                dropped += 1
                continue

            token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if token_count > max_tokens_observed:
                max_tokens_observed = token_count

            if token_count <= args.max_input_tokens:
                out_f.write(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")
                kept += 1
            else:
                dropped += 1

    print(f"total={total}")
    print(f"kept={kept}")
    print(f"dropped={dropped}")
    print(f"max_tokens_observed={max_tokens_observed}")


if __name__ == "__main__":
    main()
