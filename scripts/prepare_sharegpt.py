from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

def iter_sharegpt_prompts(obj: Any) -> Iterable[str]:
    """
    Tries to extract a single "user prompt" string from each record.

    ShareGPT/Vicuna-style datasets commonly store conversations like:
      item["conversations"] = [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]

    We'll take the first human/user message as the prompt.
    """
    if isinstance(obj, list):
        for item in obj:
            yield from iter_sharegpt_prompts(item)
        return

    if not isinstance(obj, dict):
        return

    conv = obj.get("conversations")
    if isinstance(conv, list):
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            who = (turn.get("from") or turn.get("role") or "").lower()
            txt = turn.get("value") or turn.get("content")
            if isinstance(txt, str) and txt.strip():
                if who in ("human", "user"):
                    yield txt.strip()
                    return

    # Some variants use "messages"
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for turn in msgs:
            if not isinstance(turn, dict):
                continue
            who = (turn.get("role") or "").lower()
            txt = turn.get("content")
            if isinstance(txt, str) and txt.strip() and who == "user":
                yield txt.strip()
                return

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to ShareGPT raw JSON file")
    ap.add_argument("--out", dest="out", required=True, help="Output prompts .jsonl")
    ap.add_argument("--max-prompts", type=int, default=20000)
    ap.add_argument("--min-chars", type=int, default=40)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(inp.read_text())
    prompts: List[str] = []
    seen = set()

    for p in iter_sharegpt_prompts(raw):
        if len(prompts) >= args.max_prompts:
            break
        if len(p) < args.min_chars:
            continue
        # de-dup exact matches
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)

    with out.open("w") as f:
        for i, p in enumerate(prompts):
            rec = {"id": i, "prompt": p}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f" Wrote {out} with {len(prompts)} prompts")

if __name__ == "__main__":
    main()
