#!/usr/bin/env python3
"""Run the 60-prompt greedy text set against a running ninfer-serve; save token ids from its log."""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> int:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read()[:400]!r}", file=sys.stderr)
        return error.code
    except OSError as error:
        print(f"request failed: {error}", file=sys.stderr)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("/data/s11/tools/text-prompts-60.json"))
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="s11")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
    start_size = args.log.stat().st_size if args.log.exists() else 0
    failures = 0
    for index, prompt in enumerate(prompts):
        status = post(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
        if status != 200:
            failures += 1
            print(f"prompt {index}: HTTP {status}", file=sys.stderr)
    if failures:
        print(f"{failures} requests failed", file=sys.stderr)
        return 1

    records = []
    with args.log.open("r", encoding="utf-8") as handle:
        handle.seek(start_size)
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "request_done":
                records.append(record["result"]["generated_token_ids"])
    if len(records) != len(prompts):
        print(f"log has {len(records)} completion records, expected {len(prompts)}", file=sys.stderr)
        return 1
    args.output.write_text(json.dumps({"prompts": prompts, "token_ids": records}, indent=1))
    print(f"wrote {len(records)} token-id rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
