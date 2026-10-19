#!/usr/bin/env python3
"""Ask trivial and small questions with thinking on/off and report reasoning-token counts."""
import argparse
import json
import time
import urllib.error
import urllib.request

PROMPTS = {
    "add": "What is 2+2? Answer with one number.",
    "mul": "What is 17*23? Answer with the number only.",
    "hello": "Say hello.",
    "verbatim": "Reply with exactly: ok",
    "date": "What is the capital of France? Answer with one word.",
    "code": "Write a Python one-liner that reverses a string.",
}


def ask(base_url, model, prompt, thinking, max_tokens, timeout=900.0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": thinking,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    began = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - began


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="s11")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompts", default="add,mul,hello,verbatim,date,code")
    args = parser.parse_args()
    names = [n.strip() for n in args.prompts.split(",") if n.strip()]
    print(f"{'prompt':9} {'think':5} {'total':>5} {'reason':>6} {'content':>7} {'sec':>6}  answer")
    for name in names:
        prompt = PROMPTS[name]
        for thinking in (False, True):
            payload, seconds = ask(args.base_url, args.model, prompt, thinking, args.max_tokens)
            usage = payload.get("usage", {})
            details = usage.get("completion_tokens_details", {}) or {}
            reasoning = details.get("reasoning_tokens", 0) or 0
            total = usage.get("completion_tokens", 0)
            message = payload.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            answer = message.strip().replace("\n", " ")[:40]
            print(f"{name:9} {str(thinking):5} {total:5} {reasoning:6} {total - reasoning:7} "
                  f"{seconds:6.2f}  {answer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
