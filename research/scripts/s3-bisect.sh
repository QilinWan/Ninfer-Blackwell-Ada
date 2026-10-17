#!/usr/bin/env bash
# s3-bisect.sh — 找 S3 `case-3-long`（6144＝六块 prefill＋INT8 KV）发散引入的那一步。
# 谓词＝只比 case-3-long.ids 与参考（记录在 $TREE 里的 tools/golden/recorded/2026-09-30/fork-f19e6821）：
#   一致＝good(0)，不一致＝bad(1)，构建/产件/参考缺失＝跳过(125)。
#
# 夜里跑法（**不要在主树 bisect**：主树的 build-120a-cu13 是门链共用的热面，会被踩）：
#   cd /home/ai-agent/DSHW/ninfer-watch/ada-tree
#   git worktree add /tank/ninfer-window/bisect 7d762ec5
#   cd /tank/ninfer-window/bisect && bash ../../relay/s3-bisect.sh --pin            # 先在基线面钉合成件
#   git bisect start 7d762ec5 03726c08~ -- && S3_BISECT_CARD=<空卡> git bisect run ../../relay/s3-bisect.sh
#   git bisect reset; git worktree remove --force /tank/ninfer-window/bisect
#
# 两个本轮定下的口径（别退回去）：
#   1) 合成件必须由**基线面产一次并钉住**（S3_SYNTH_DIR），否则 bisect 的每一步同时换了
#      "被测实现"和"被测输入"：生成器 ninfer_qwen3_5_text_context_tp2_test 在层段里也被改过，
#      它的漂移会冒充发散＝混杂因子。
#   2) 判据只比 .ids（`diff -r` 会把 record.sh 的 *.out/*.log 记成 Only in＝任何面都 FAIL）。
# 磁盘：根分区只剩约 25 G，构建目录实测 44 G ⇒ worktree 与构建目录都必须落在 /tank。
set -uo pipefail
R=/home/ai-agent/DSHW/ninfer-watch/relay
TREE="${BISECT_TREE:-/tank/ninfer-window/bisect}"
SYNTH="${S3_SYNTH_DIR:-/tank/ninfer-window/synthetic-bisect}"
OUT="$R/s3-bisect"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=/home/ai-agent/DSHW/ninfer-watch/deps-ada/lib
CARD="${S3_BISECT_CARD:-1}"
REF="$TREE/tools/golden/recorded/2026-09-30/fork-f19e6821"
CASE="${S3_CASE:-case-3-long}"
sha=$(git -C "$TREE" rev-parse --short HEAD 2>/dev/null || echo notree)
mkdir -p "$OUT"
log="$OUT/$sha.log"; : > "$log"
say() { echo "$*" >> "$log"; }
[ -d "$TREE/.git" ] || { say "no worktree at $TREE (skip)"; exit 125; }
[ -f "$REF/$CASE.ids" ] || { say "no reference $REF/$CASE.ids at this face (skip)"; exit 125; }
JOBS=14 nice -n 12 cmake --build "$TREE/build-120a-cu13" --target ninfer-tp1-golden \
    ninfer_qwen3_5_text_context_tp2_test >> "$log" 2>&1 \
  || { say "build rc=$? (skip)"; exit 125; }
# --pin＝在基线面产合成件并落钉；常规步只用钉好的那份，绝不重写。
if [ "${1:-}" = "--pin" ] || [ ! -f "$SYNTH/model.ninfer" ]; then
    if [ "${1:-}" != "--pin" ] && [ ! -f "$SYNTH/model.ninfer" ]; then
        say "no pinned synthetic artifact and not the pinning step (skip)"; exit 125
    fi
    rm -rf "$SYNTH"; mkdir -p "$SYNTH"
    CUDA_VISIBLE_DEVICES=$CARD nice -n 12 "$TREE/build-120a-cu13/tests/ninfer_qwen3_5_text_context_tp2_test" \
        write "$SYNTH" >> "$log" 2>&1
    [ -f "$SYNTH/model.ninfer" ] || { say "synthetic write failed (skip)"; exit 125; }
    say "PINNED synthetic $SYNTH/model.ninfer sha=$(sha256sum "$SYNTH/model.ninfer" | cut -c1-16) size=$(stat -c %s "$SYNTH/model.ninfer") at $sha"
    [ "${1:-}" = "--pin" ] && { say "pin only, no verdict"; exit 125; }
fi
run=$(mktemp -d /tank/ninfer-window/gold.XXXXXX)
CUDA_VISIBLE_DEVICES=$CARD nice -n 12 "$TREE/tools/golden/record.sh" \
    "$TREE/build-120a-cu13/apps/ninfer-tp1-golden" "$SYNTH/model.ninfer" "$run" >> "$log" 2>&1 \
  || { say "record rc=$? (skip)"; rm -rf "$run"; exit 125; }
if diff "$REF/$CASE.ids" "$run/$CASE.ids" >> "$log" 2>&1; then
    say "GOOD: $CASE identical to the reference"; rm -rf "$run"; exit 0
fi
say "BAD: $CASE diverges: $(diff "$REF/$CASE.ids" "$run/$CASE.ids" | grep -c '^[<>]') lines"
for c in case-1-short case-2-medium; do
    say "side $c difflines=$(diff "$REF/$c.ids" "$run/$c.ids" 2>/dev/null | wc -l)"
done
rm -rf "$run"; exit 1
