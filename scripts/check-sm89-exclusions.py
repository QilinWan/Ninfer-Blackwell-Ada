#!/usr/bin/env python3
"""check-sm89-exclusions.py — 断言 Ada(sm_89) 剔除表仍然命中真实源文件。

为什么需要：`cmake/Sm89Compat.cmake` 用路径正则把 Blackwell-only 的翻译单元从 `ninfer_ops`
里剔除（FP4 w4a4 / K8V4 追加与因果档）。**正则失配不会报错**，只会“不再剔除”，于是 sm_89
构建一路编到 ptxas 才炸，或在他人机器上悄悄产出含非法指令的库。上游 dev 正在做 `src/ops`
家族大搬迁（`90d25ed` +417 文件、`6a4104f` +478 文件），这类静默失配是**跟随主线的首要风险**。

用法：scripts/check-sm89-exclusions.py [--verbose]
退出码：0 = 每条模式都命中且存在 stub；1 = 有模式失配（表过期）或缺 stub；2 = 解析不到剔除表。
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

CFG = "cmake/Sm89Compat.cmake"
STUB_DIRS = ("src/ops/sm89/", "src/ops/sm120a/")
BLACKWELL_PTX = re.compile(
    r"kind::mxf8f6f4|e2m1x2|cp\.async\.bulk\.tensor|setmaxnreg|elect\.sync|tcgen05")


def git_lines(*args):
    out = subprocess.run(["git"] + list(args), capture_output=True, text=True).stdout
    return [x for x in out.split("\n") if x]


def norm(cmake_string):
    """CMake 字符串里的两个反斜杠即正则里的一个：把连续反斜杠压成一个。"""
    return re.sub(r"\\+", r"\\", cmake_string)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip()
    if not root:
        print("不在 git 仓库内", file=sys.stderr)
        return 2
    cfg = Path(root) / CFG
    if not cfg.exists():
        print("找不到 %s（非 Ada 树？）" % CFG, file=sys.stderr)
        return 2

    m = re.search(r"set\(NINFER_SM89_DROPPED_PATTERNS(.*?)\n\)", cfg.read_text(), re.S)
    if not m:
        print("解析不到 NINFER_SM89_DROPPED_PATTERNS 块，%s 格式已变动" % CFG, file=sys.stderr)
        return 2

    pats = []
    for raw in re.findall(r'"([^"]+)"', m.group(1)):
        p = norm(raw)
        if ".cu" not in p:
            continue
        pats.append((p, re.compile(p)))
    if not pats:
        print("剔除表为空——上游删了整条 FP4/K8V4 路线？人工确认后修表", file=sys.stderr)
        return 2

    files = git_lines("ls-files", "src/ops/*.cu")
    print("%-52s %4s  %s" % ("模式", "命中", "样例"))
    broken = []
    for pat, rx in pats:
        hit = [f for f in files if rx.search(f)]
        if not hit:
            broken.append(pat)
        note = "" if hit else "   🔴 失配：表已过期，Ada 不再剔除该路由"
        print("%-52s %4d  %s%s" % (pat, len(hit), (hit[0][:58] if hit else "-"), note))
        if args.verbose:
            for f in hit[1:]:
                print("%-52s      %s" % ("", f))

    dropped = set()
    for pat, rx in pats:
        dropped |= {f for f in files if rx.search(f)}
    stubs = [f for f in git_lines("ls-files", "src/ops") if f.startswith(STUB_DIRS)]
    print("\n剔除表 %d 条模式 → 命中 %d 个 TU（去重）｜ 占 src/ops/*.cu %d/%d = %d%%"
          % (len(pats), len(dropped), len(dropped), len(files),
             100 * len(dropped) // max(1, len(files))))
    print("stub 文件 %d 个：%s" % (len(stubs), ", ".join(Path(s).name for s in stubs[:6]) or "无"))

    rc = 0
    if broken:
        print("🔴 %d 条模式失配 -> 修 %s" % (len(broken), CFG))
        rc = 1
    if dropped and not stubs:
        print("🔴 剔除了 TU 但树内没有 stub：被删路由运行期没有显式 unsupported 答复（UB 风险）")
        rc = 1

    sentinel = []
    for f in files:
        if f in dropped:
            continue
        try:
            if BLACKWELL_PTX.search((Path(root) / f).read_text(errors="ignore")):
                sentinel.append(f)
        except OSError:
            pass
    if sentinel:
        print("🟡 %d 个未剔除文件含 Blackwell 专属指令，需确认 Ada 可编：" % len(sentinel))
        for f in sentinel[:15]:
            print("      %s" % f)
    print("✅ 剔除表与当前树一致" if rc == 0 else "❌ 见上")
    return rc


if __name__ == "__main__":
    sys.exit(main())
