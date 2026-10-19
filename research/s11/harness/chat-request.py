#!/usr/bin/env python3
"""Send one chat completion (optionally with an image) to ninfer-serve; print status + JSON."""
import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="s11")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    content = []
    if args.image is not None:
        payload = args.image.read_bytes()
        media_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
        url = f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
    content.append({"type": "text", "text": args.prompt})
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "enable_thinking": False,
        "stream": False,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    except OSError as error:
        print(f"request failed: {error}", file=sys.stderr)
        return 2
    text = raw.decode("utf-8", errors="replace")
    print(f"HTTP {status}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(text[:4000])
        return 0 if 200 <= status < 300 else 1
    if 200 <= status < 300:
        message = parsed.get("choices", [{}])[0].get("message", {}).get("content")
        print(f"answer: {message!r}")
        print(f"usage: {parsed.get('usage')}")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2)[:4000])
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
