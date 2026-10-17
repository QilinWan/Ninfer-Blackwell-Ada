#!/usr/bin/env python3
"""GSM8K 判分底座（S9）· 判分核。

规范口径（写死，勿自创）：
  gold     : test.jsonl 每行 answer 字段 "#### <数字>" 之后的数字。
  predicted: 模型输出文本里，先取最后一次 "The answer is X" / "答案是 X"（X 允许 $ , 空格 与结尾句点），
             取不到则取全文最后一个数字串（含负号与小数）。
  比较     : (1) canonical 字符串比较：去掉 '$'、','、结尾 '.' 后逐字符相等；
             (2) 数值比较：两侧解析成数，int/float，相对误差 1e-6。
             两个口径都报；canonical 是主判据（与 openai/grade-school-math 的 eval.py 一致）。

  python3 grade.py --pred <preds.jsonl> --gold data/gsm8k_test.jsonl [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys

GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)\s*$")
# 主提取：最后一次 "The answer is X"（大小写不敏感，允许 'the final answer is'）
ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([-+]?\$?[\d,]*\.?\d+)", re.IGNORECASE
)
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def gold_of(answer_text: str) -> str | None:
    for line in answer_text.splitlines():
        m = GOLD_RE.search(line.strip())
        if m:
            return m.group(1)
    return None


def _clean(number: str) -> str:
    n = number.replace("$", "").replace(",", "").strip()
    n = n.rstrip(".")
    if n in ("+", "-"):
        return ""
    return n


def extract_predicted(text: str) -> str:
    if text is None:
        return ""
    hits = ANSWER_IS_RE.findall(text)
    if hits:
        return _clean(hits[-1])
    nums = NUMBER_RE.findall(text)
    return _clean(nums[-1]) if nums else ""


def as_number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def is_equiv_canonical(pred: str, gold: str) -> bool:
    return _clean(pred) == _clean(gold)


def is_equiv_numeric(pred: str, gold: str) -> bool:
    a, b = as_number(_clean(pred)), as_number(_clean(gold))
    if a is None or b is None:
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= 1e-6


def grade(preds_path: str, gold_path: str):
    golds, gold_by_question = {}, {}
    for index, line in enumerate(open(gold_path, encoding="utf-8")):
        row = json.loads(line)
        golds[index] = (gold_of(row["answer"]), row)
        gold_by_question[row["question"]] = gold_of(row["answer"])
    rows, n_canon, n_num, n_missing = [], 0, 0, 0
    run_meta = None
    for line in open(preds_path, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("kind") == "ninfer-gsm8k-run":
            run_meta = rec
            continue
        index = rec.get("index")
        if index is None and "question" in rec:
            gold = gold_by_question.get(rec["question"])
        else:
            gold = golds[index][0] if index in golds else rec.get("gold")
        text = rec.get("output") or rec.get("text") or rec.get("completion") or ""
        if gold is None:
            n_missing += 1
            continue
        pred = extract_predicted(text)
        canon, num = is_equiv_canonical(pred, gold), is_equiv_numeric(pred, gold)
        n_canon += canon
        n_num += num
        rows.append(
            {"index": index, "gold": gold, "extracted": pred, "canonical": canon, "numeric": num}
        )
    total = len(rows)
    summary = {
        "engine_tag": (run_meta or {}).get("engine_tag"),
        "subset_digest": (run_meta or {}).get("subset_digest"),
        "n": total,
        "canonical_correct": n_canon,
        "canonical_acc": (n_canon / total) if total else None,
        "numeric_correct": n_num,
        "numeric_acc": (n_num / total) if total else None,
        "unmatched_gold": n_missing,
    }
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="data/gsm8k_test.jsonl")
    ap.add_argument("--report")
    args = ap.parse_args()
    summary, rows = grade(args.pred, args.gold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        if not row["canonical"]:
            print(f"  miss idx={row['index']} gold={row['gold']!r} extracted={row['extracted']!r}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False)
    return 0 if summary["n"] and summary["unmatched_gold"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
