#!/usr/bin/env python3
"""GSM8K 跑分底座（S9）· 客户端。

对 OpenAI 兼容的 /v1/chat/completions 逐题取答，落 JSONL，交给 grade.py 判分。

写死的口径（D3 判定用，不得中途更改）：
  * 子集：test.jsonl 的**文件顺序前 N 题**（默认 N=500，全量 1319）。子集用
    `--count` 选，落盘时写 `subset_digest`（选中题的 question+gold 串起来取 SHA256），
    两次跑分母/分子必须同 digest 才可比。
  * 解码：temperature=0（贪心）、`--max-tokens`（默认 512）、stop = ["\n\nQ:"]。
  * 提示：standard GSM8K k-shot（默认 8，取自 train.jsonl 的前 k 题），
    system + 少量 few-shot 示例 + 目标题；`--shot 0` = zero-shot。
  * 每题记录：index/question/gold/output/elapsed_ms/finish_reason/prompt_tokens/
    completion_tokens/engine（来自 /v1/models 的 meta），以及 `--engine-tag` 自由文本
    （**分母必须写明 `--tp 1` 与构建尖，分子写明 `--tp 2` 与同一构建尖**）。
  * 并发：线程池，默认 1（判分对比要求逐题可复现，并发>1 时写 `concurrency` 字段）。

用法（例）：
  python3 run_gsm8k.py --base-url http://127.0.0.1:8083 --model qwen \
      --count 500 --out preds_tp1.jsonl --engine-tag "v-follow-81c8ce0-r0 --tp 1"
  python3 grade.py --pred preds_tp1.jsonl --gold data/gsm8k_test.jsonl --report report_tp1.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request

GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)\s*$")


def load_rows(path: str, count: int):
    rows = []
    for index, line in enumerate(open(path, encoding="utf-8")):
        row = json.loads(line)
        gold = None
        for ln in row["answer"].splitlines():
            m = GOLD_RE.search(ln.strip())
            if m:
                gold = m.group(1)
        rows.append({"index": index, "question": row["question"], "gold": gold})
        if count and len(rows) >= count:
            break
    return rows


def subset_digest(rows) -> str:
    blob = "\n".join(f"{r['index']}|{r['question']}|{r['gold']}" for r in rows)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def few_shot(train_path: str, k: int):
    shots = []
    if not k or not train_path or not os.path.exists(train_path):
        return shots
    for line in open(train_path, encoding="utf-8"):
        row = json.loads(line)
        gold = None
        for ln in row["answer"].splitlines():
            m = GOLD_RE.search(ln.strip())
            if m:
                gold = m.group(1)
        solution = re.split(r"####", row["answer"])[0].strip()
        shots.append(
            {
                "question": row["question"],
                "content": f"{solution}\nThe answer is {gold}.",
            }
        )
        if len(shots) >= k:
            break
    return shots


def build_messages(row, shots):
    messages = [{"role": "system", "content": "You solve grade-school math problems. "
                 "Show your reasoning, then end with 'The answer is X'."}]
    for shot in shots:
        messages.append({"role": "user", "content": shot["question"]})
        messages.append({"role": "assistant", "content": shot["content"]})
    messages.append({"role": "user", "content": row["question"]})
    return messages


def post(base_url: str, payload: dict, api_key: str | None, timeout: float, insecure: bool):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
        return json.loads(response.read().decode("utf-8"))


def engine_meta(base_url: str, timeout: float, insecure: bool):
    try:
        ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=timeout,
                                    context=ctx) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - meta 只作记录，不作判据
        return {"unavailable": repr(error)}


def ask(row, args, shots):
    payload = {
        "model": args.model,
        "messages": build_messages(row, shots),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stop": ["\n\nQ:"],
        "stream": False,
    }
    if args.extra_body:
        payload.update(json.loads(args.extra_body))
    last = None
    for attempt in range(args.retries + 1):
        started = time.perf_counter()
        try:
            body = post(args.base_url, payload, args.api_key, args.request_timeout, args.insecure)
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = body.get("usage") or {}
            return {
                "index": row["index"],
                "question": row["question"],
                "gold": row["gold"],
                "output": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason"),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "attempt": attempt,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            last = repr(error)
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    return {"index": row["index"], "question": row["question"], "gold": row["gold"],
            "output": "", "error": last, "attempt": args.retries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="local")
    ap.add_argument("--data", default="data/gsm8k_test.jsonl")
    ap.add_argument("--train", default="data/gsm8k_train.jsonl")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--shot", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--extra-body", help="追加到请求体的 JSON，例如引擎开关")
    ap.add_argument("--engine-tag", required=True,
                    help="构建尖 + --tp 1 / --tp 2 + 服务开关，分母分子都必须写全")
    ap.add_argument("--out", required=True)
    ap.add_argument("--insecure", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.data, args.count)
    shots = few_shot(args.train, args.shot)
    meta = {
        "kind": "ninfer-gsm8k-run",
        "engine_tag": args.engine_tag,
        "base_url": args.base_url,
        "model": args.model,
        "n": len(rows),
        "subset": "file-order-first-N",
        "subset_digest": subset_digest(rows),
        "shot": args.shot,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stop": ["\n\nQ:"],
        "concurrency": args.concurrency,
        "engine_meta": engine_meta(args.base_url, args.request_timeout, args.insecure),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results, failures = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(ask, row, args, shots) for row in rows]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            if "error" in record:
                failures += 1
            results.append(record)
            if done % 50 == 0:
                print(f"  … {done}/{len(rows)} (errors {failures})", file=sys.stderr)
    results.sort(key=lambda r: r["index"])
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    answered = sum(1 for r in results if r.get("output"))
    print(json.dumps({"wrote": args.out, "n": len(results), "answered": answered,
                      "request_errors": failures, "subset_digest": meta["subset_digest"]},
                     ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
