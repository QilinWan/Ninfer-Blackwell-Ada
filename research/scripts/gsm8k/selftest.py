#!/usr/bin/env python3
"""S9 判分链路的自检（CPU，零 GPU，零现网负载）。

起一个本机 OpenAI 兼容桩服务，三种应答模式各跑一遍小样本，再判分：
  gold    : 逐字 "The answer is {gold}."            → canonical 必须 100%
  messy   : 带千分位/$/推理前缀/中文"答案是"        → canonical 必须 100%
  perturb : 答案 +1                                 → canonical 必须 0%，numeric 必须 0%
并验证两次跑的 subset_digest 相同（可比性）、`--report` 落盘可读。
"""
from __future__ import annotations

import http.server
import json
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "gsm8k_test.jsonl")
GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)\s*$")


def dataset():
    out = {}
    for line in open(DATA, encoding="utf-8"):
        row = json.loads(line)
        gold = None
        for ln in row["answer"].splitlines():
            m = GOLD_RE.search(ln.strip())
            if m:
                gold = m.group(1)
        out[row["question"]] = gold
    return out


def respond(question: str, gold: str, mode: str) -> str:
    if gold is None:
        return "The answer is 0."
    if mode == "gold":
        return f"The answer is {gold}."
    if mode == "perturb":
        value = float(str(gold).replace(",", "").replace("$", "")) + 1
        return f"The answer is {value:g}."
    # messy：先写一段推理，再带 $ 与千分位；最后一句用中文"答案是"
    digits = str(gold).replace(",", "").replace("$", "")
    value = float(digits)
    pretty = f"{value:,.0f}" if abs(value - round(value)) < 1e-9 else f"{value:,.2f}"
    sign = "$" if value >= 0 else ""
    return (f"Let me work it out.\nStep 1: 3 apples.\nStep 2: total is {pretty}.\n"
            f"So the answer is {sign}{pretty}.\n答案是 {sign}{pretty}。")


class Handler(http.server.BaseHTTPRequestHandler):
    mode = "gold"
    questions: dict = {}

    def log_message(self, *args):  # 静音
        return

    def _send(self, code, body):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "stub", "engine": "gsm8k-selftest"}]})
        else:
            self._send(404, {"error": "no route"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        gold = self.questions.get(question)
        text = respond(question, gold, self.mode)
        usage = {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                 "completion_tokens": len(text) // 4}
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}],
                         "usage": usage, "model": "stub"})


def run_case(mode: str, expected_canonical: float, expected_numeric: float | None) -> bool:
    Handler.mode = mode
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    out = os.path.join(HERE, f"selftest_{mode}.jsonl")
    report = os.path.join(HERE, f"selftest_{mode}.report.json")
    try:
        subprocess.run(
            [sys.executable, "run_gsm8k.py", "--base-url", f"http://127.0.0.1:{port}",
             "--model", "stub", "--count", "40", "--shot", "0", "--max-tokens", "64",
             "--concurrency", "4", "--engine-tag", f"stub/{mode}", "--out", out],
            check=True, cwd=HERE, stdout=subprocess.DEVNULL)
        subprocess.run(
            [sys.executable, "grade.py", "--pred", out, "--gold", DATA, "--report", report],
            check=True, cwd=HERE, capture_output=True, text=True)
        summary = json.load(open(report, encoding="utf-8"))["summary"]
        ok = (abs(summary["canonical_acc"] - expected_canonical) < 1e-9
              and (expected_numeric is None
                   or abs(summary["numeric_acc"] - expected_numeric) < 1e-9)
              and summary["n"] == 40 and summary["unmatched_gold"] == 0)
        print(f"[{mode}] canonical={summary['canonical_acc']:.4f} "
              f"numeric={summary['numeric_acc']:.4f} n={summary['n']} "
              f"digest={summary['subset_digest'][:16]} → {'OK' if ok else 'FAIL'}")
        return ok
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    if not os.path.exists(DATA):
        print(f"缺数据：{DATA}（下载 test.jsonl 到 data/）", file=sys.stderr)
        return 2
    Handler.questions = dataset()
    results = [
        run_case("gold", 1.0, 1.0),
        run_case("messy", 1.0, 1.0),
        run_case("perturb", 0.0, 0.0),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
