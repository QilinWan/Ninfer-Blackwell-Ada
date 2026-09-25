"""One synthetic long-context retrieval check against an already running NInfer server.

Uses a local tokenizer, never downloads weights, and reports the server's actual token count.
This checks five exact facts, not general long-context quality.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--model", default="qwen3.8-27b-uncensored")
parser.add_argument("--tokens", type=int, default=389000)
parser.add_argument("--output", required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
paragraph = (
    "Operations journal: an image processing service accepts uploads, validates their dimensions, "
    "and places work in a bounded queue. The scheduler balances interactive requests with background "
    "jobs. Workers decode inputs, reuse immutable buffers where possible, and release temporary "
    "storage after processing. Engineers investigate latency using measured queue times, CPU usage, "
    "memory pressure, and device activity rather than assumptions. A request identifier remains "
    "stable across retries. Snapshot files are written atomically and retained until the new state "
    "has been verified. This paragraph is routine background documentation, not an audit record.\n"
)
base_ids = tokenizer.encode(paragraph, add_special_tokens=False)
ids = (base_ids * (args.tokens // len(base_ids) + 1))[:args.tokens]
expected = {"ORCHID": "PX7-ALDER-9214", "MAPLE": "M9V-EMBER-4731", "COPPER": "K2R-FROST-6850",
            "AMBER": "W8C-RIVER-3026", "VERTEX": "J4N-CEDAR-8175"}
parts = ["Read this operations archive and recover the five audit verification phrases.\n"]
last = 0
for fraction, (name, code) in zip((.1, .3, .5, .7, .9), expected.items()):
    cut = int(len(ids) * fraction)
    parts.append(tokenizer.decode(ids[last:cut]))
    parts.append(f"\nAUDIT RECORD: The verification phrase for {name} is {code}. END AUDIT RECORD.\n")
    last = cut
parts.append(tokenizer.decode(ids[last:]))
parts.append("\nReturn only the five verification phrases, one line per name: ORCHID, MAPLE, COPPER, AMBER, VERTEX. Do not invent missing values.\n")
content = "".join(parts)
print(json.dumps({"prepared_body_tokens": args.tokens, "characters": len(content)}), flush=True)
payload = {"model": args.model, "messages": [{"role": "user", "content": content}],
           "reasoning_effort": "none", "temperature": 0, "max_tokens": 128}
request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
start = time.monotonic()
with opener.open(request, timeout=1800) as response:
    result = json.load(response)
answer = result["choices"][0]["message"]["content"]
checks = {name: code in answer for name, code in expected.items()}
report = {"wall_seconds": round(time.monotonic() - start, 3), "expected": expected,
          "checks": checks, "correct": sum(checks.values()), "answer": answer,
          "usage": result["usage"], "timings": result.get("timings"),
          "finish_reason": result["choices"][0]["finish_reason"]}
Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
assert result["usage"]["prompt_tokens"] > 262144, "The check did not exceed the native window"
assert all(checks.values()), "Some audit records were not retrieved"
