#!/usr/bin/env bash
# window-d-setup.sh — 双卡窗口的 gate.sh 环境量与首轮参考（开窗前把参数定死，窗口里只做测量）。
# 用法：
#   记录我方基线（首轮，必须空目录）：MODE=record FACE=int8  bash relay/window-d-setup.sh
#   之后对照：                        MODE=compare FACE=int8 bash relay/window-d-setup.sh
#   nvfp4 KV 面（我方本轮开闸的）＝FACE=nvfp4；两面各有自己的参考目录（坑③：别拿 int8 参考判 nvfp4）
# 需要两张空闲卡（GATE_DEVICES 0,1）＋gate-72 或主树的 build 目录（BUILD=…）。
set -uo pipefail
W=/tank/ninfer-window                      # 临时件与参考都落 /tank（根分区 90% 已用）
SRC=/home/ai-agent/DSHW/ninfer-watch/ada-tree
BUILD="${BUILD:-$SRC/build-120a-cu13}"
FACE="${FACE:-int8}"
MODE="${MODE:-compare}"
NF4=/tank/models/mrwan0410-lynn-nf4-ninfer/qwen3.8-27b-lynn-nf4.ninfer
W4A4=/tank/models/lynn/NVFP4-NInfer/W4A4/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer
ART="${ARTIFACT:-$NF4}"
REF="$W/reference-$FACE"
OUT="$W/run-$FACE-$(date +%Y%m%d-%H%M)"
mkdir -p "$W" "$OUT"
export GATE_SRC="$SRC"
export GATE_ARTIFACT="$ART" GATE_DF2_ARTIFACT="$ART" GATE_TEST_ARTIFACT="$ART"
export GATE_TEST_SCRATCH="$W/scratch"
export GATE_GOLDEN_MODEL="${GATE_GOLDEN_MODEL:-/tmp/synthetic-tp1/model.ninfer}"  # 合成两层件（write 模式产）
export GATE_PROMPTS="$SRC/tools/tp2/reference/prompts.jsonl"   # 60 题 GSM8K 形（--record 硬要求）
export GATE_DEVICES="${GATE_DEVICES:-0,1}"
export GATE_PORT="${GATE_PORT:-8090}"
export GATE_PPL_PAIRED="${GATE_PPL_PAIRED:-1}"                 # 改 bits 的提交用配对判据
export GATE_CTEST_EXCLUDE='ninfer_qwen3_5_(prefix|score|moe|dflash_prefill)_real_test|ninfer_qwen3_5_vision_workspace_test'
# 关键修正：默认串里的 `dflash` 会连 `*_dflash2_*_test` 一起排除；上面显式只排 dflash_prefill 与单卡跑不动的件。
# 产线旗标面（与 tools/tp2/gate.sh:43 同形，KV 档按 FACE 切）：
export GATE_SERVE_FLAGS="--tp 2 --devices $GATE_DEVICES --kv-dtype $FACE --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft --vision --vision-device 0 --max-vision-tokens 4096"
export GATE_DF2_FLAGS="--tp 2 --devices $GATE_DEVICES --kv-dtype $FACE --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --spec dflash2 --draft-tokens 7 --lm-head-draft"
if [ "$MODE" = record ]; then mkdir -p "$REF"; rm -rf "$REF"; mkdir -p "$REF"; fi
if [ ! -s "$REF/prompts.jsonl" ]; then cp -f "$GATE_PROMPTS" "$REF/prompts.jsonl"; fi
echo "== face=$FACE mode=$MODE build=$BUILD ref=$REF out=$OUT"
echo "== serve flags: $GATE_SERVE_FLAGS"
echo "== df2   flags: $GATE_DF2_FLAGS"
echo "== golden model: $GATE_GOLDEN_MODEL  (存在？$([ -f "$GATE_GOLDEN_MODEL" ] && echo yes || echo 'no：先跑 write 模式'))"
echo "== 跑法（在 gpu-lease 里）："
echo "   $MODE=record GATE_DEVICES=0,1 bash relay/window-d-setup.sh   # 首轮建基线（空目录）"
echo "   $MODE=compare GATE_DEVICES=0,1 bash relay/window-d-setup.sh"
echo "   然后：gpu-lease run ricerca -- bash $SRC/tools/tp2/gate.sh ${MODE:+--} $( [ "$MODE" = record ] && echo --record ) --stages shards,ctest,golden,ppl,greedy,dflash2 --out "$OUT" "$BUILD" "$REF""
if [ "${RUN:-0}" = 1 ]; then
  args=(--stages shards,ctest,golden,ppl,greedy,dflash2 --out "$OUT" "$BUILD" "$REF")
  [ "$MODE" = record ] && args=(--record "${args[@]}")
  bash "$SRC/tools/tp2/gate.sh" "${args[@]}" 2>&1 | tee "$OUT/gate.log"
fi
