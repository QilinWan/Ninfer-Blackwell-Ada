#!/usr/bin/env python3
"""Benchmark prompts against a running ninfer-serve, reading its request log for exact metrics."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = {
    "essay": ("Write a detailed essay about the history of Rome. Write at least 400 words.", 512),
    "count": ("List the numbers from 1 to 400, separated by commas. No other text.", 512),
    "explain": ("Explain how a transformer attention layer works, step by step, in detail.", 512),
    "image": ("Describe this image in detail: colors, shapes, and what it might be used for.", 128),
}


def post(base_url, model, prompt, max_tokens, image=None, timeout=1800.0):
    content = []
    if image is not None:
        import base64
        data = image.read_bytes()
        url = "data:image/x-portable-pixmap;base64," + base64.b64encode(data).decode()
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
    content.append({"type": "text", "text": prompt})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "enable_thinking": False,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--label", default="bench")
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="s11")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompts", default="essay,count,explain")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0)
    args = parser.parse_args()

    start = args.log.stat().st_size if args.log.exists() else 0
    names = [n.strip() for n in args.prompts.split(",") if n.strip()]
    for _ in range(args.repeat):
        for name in names:
            prompt, cap = PROMPTS[name]
            if args.max_tokens:
                cap = args.max_tokens
            image = args.image if name == "image" else None
            began = time.monotonic()
            status = post(args.base_url, args.model, prompt, cap, image)
            print(f"{name}: HTTP {status} wall {time.monotonic() - began:.2f}s")

    records = []
    with args.log.open(encoding="utf-8") as handle:
        handle.seek(start)
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "request_done":
                records.append(record)
    if not records:
        print("no request_done records", file=sys.stderr)
        return 1
    total_decode = 0
    total_tokens = 0
    total_drafted = 0
    total_accepted = 0
    for index, record in enumerate(records):
        result = record["result"]
        timings = record["timings_seconds"]
        tokens = result["completion_tokens"]
        decode = timings["decode"]
        spec = record.get("speculative") or {}
        drafted = spec.get("drafted_tokens", 0)
        accepted = spec.get("accepted_tokens", 0)
        rate = (100.0 * accepted / drafted) if drafted else 0.0
        tok_s = (tokens / decode) if decode else 0.0
        total_decode += decode
        total_tokens += tokens
        total_drafted += drafted
        total_accepted += accepted
        print(f"req{index}: tokens={tokens} decode={decode:.3f}s tok/s={tok_s:.2f} "
              f"drafted={drafted} accepted={accepted} accept={rate:.1f}%")
    print(f"{args.label}: total tokens={total_tokens} decode={total_decode:.3f}s "
          f"tok/s={total_tokens / total_decode if total_decode else 0:.2f} "
          f"accept={100.0 * total_accepted / total_drafted if total_drafted else 0:.1f}% "
          f"({total_accepted}/{total_drafted})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
