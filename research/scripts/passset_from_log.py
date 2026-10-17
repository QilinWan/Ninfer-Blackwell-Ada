#!/usr/bin/env python3
"""passset_from_log.py — 把 ctest 输出解析成分母 TSV。

用法: passset_from_log.py <ctest-log> <out.tsv> [备注行…]
行格式: 编号<TAB>用例名<TAB>状态<TAB>秒；表头以 # 起。
"""
import re
import sys

LINE = re.compile(
    r"^\s*(\d+)/\d+\s+Test\s+#\d+:\s+(\S+)\s+[\s.]*\**\s*"
    r"(Passed|Failed|Skipped|Timeout|Not Run)\s*(?:\([^)]*\)\s*)?([0-9.]+)\s*sec"
)
SUB = re.compile(r"^\s*Test project|^\s*The following tests (did not run|FAILED)|^\s*\d+ - ")


def main() -> int:
    log, out = sys.argv[1], sys.argv[2]
    notes = sys.argv[3:]
    rows = {}
    # ctest 用 \r 刷新进度行 ⇒ 多条结果会落在同一物理行内，先按 \r 拆成逻辑行。
    text = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    for line in text.split("\n"):
        m = LINE.match(line.strip())
        if m:
            rows[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# 权威设备分母 · sm_120a @ GPU1 (RTX 5060 Ti, CC 12.0, 36 SM)\n")
        for note in notes:
            fh.write(f"# {note}\n")
        fh.write("# 列: 编号\t用例名\t状态\t秒\n")
        for index in sorted(rows):
            name, status, seconds = rows[index]
            fh.write(f"{index}\t{name}\t{status}\t{seconds}\n")
    passed = sum(1 for v in rows.values() if v[1] == "Passed")
    skipped = sum(1 for v in rows.values() if v[1] == "Skipped")
    other = [name for _, (name, st, _) in sorted(rows.items()) if st not in ("Passed", "Skipped")]
    total = sum(float(v[2]) for v in rows.values())
    print(f"解析 {len(rows)} 条：Passed {passed} / Skipped {skipped} / 其余 {len(other)}；耗时合计 {total:.2f} s")
    if other:
        print("非通过项:", ", ".join(other))
    return 0


if __name__ == "__main__":
    sys.exit(main())
