#!/usr/bin/env bash
# gate-steps.sh — 全构建门链（worktree 复用）：reset → 贴补丁 → 构建 → 家族子集 → 簇门 → passset
# 用法：bash relay/gate-steps.sh <tree-sha> <tag-label>   （全部绝对路径，中途换目录也不写错地方）
set -uo pipefail
R="/home/ai-agent/DSHW/ninfer-watch/relay"
G="/home/ai-agent/DSHW/ninfer-watch/gate-72"
SHA="${1:?tree-sha}"; LABEL="${2:?label}"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$G/../deps-ada/lib"
CHAIN="$R/gate-$LABEL-chain.log"
: > "$CHAIN"
cd "$G" || exit 9
git checkout -- . >>"$CHAIN" 2>&1
git reset --hard "$SHA" >>"$CHAIN" 2>&1
# GATE_PATCHES：S6 能力门（gates-r72.patch）自 47 回合起已落进主树 ⇒ 新面只贴构建面补丁：
#   GATE_PATCHES="$R/vision-on-build-ada.patch" bash relay/gate-steps.sh <sha> <label>
# 旧面（<=3528e19e 等）仍要两份 ⇒ 默认保持两份。
PATCHES="${GATE_PATCHES:-$R/vision-on-build-ada.patch $R/gates-r72.patch}"
# 幂等贴补丁：已应用（--reverse --check 过）＝跳过并报 present；可正向贴＝贴；都不行＝FAIL 并列文件。
ok=1; report=""
for pt in $PATCHES; do
  b=$(basename "$pt")
  if git apply --reverse --check "$pt" >/dev/null 2>&1; then report="$report present:$b"
  elif git apply --check "$pt" >/dev/null 2>&1 && git apply "$pt"; then report="$report applied:$b"
  else ok=0; report="$report FAILED:$b($(git apply --check "$pt" 2>&1 | head -1 | cut -c1-40))"; fi
done
echo "patches:$report" >>"$CHAIN"
[ $ok -eq 1 ] || echo "patch stage incomplete: continuing on the face as it stands" >>"$CHAIN"
JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests > "$R/gate-$LABEL-build.log" 2>&1
rc=$?; echo "build rc=$rc $(date -u +%H:%M:%S)" >>"$CHAIN"; [ $rc -ne 0 ] && exit $rc
cd "$G/build-120a-cu13" || exit 9
CUDA_VISIBLE_DEVICES=1 nice -n 12 ctest -j1 --timeout 900 -R "linear|fp8|nvfp4|bf16" > "$R/gate-$LABEL-sub.log" 2>&1
echo "sub rc=$? $(grep -E '% tests passed' "$R/gate-$LABEL-sub.log" | tail -1) $(grep -E 'Total Test time' "$R/gate-$LABEL-sub.log" | tail -1)" >>"$CHAIN"
CUDA_VISIBLE_DEVICES=1 nice -n 12 ctest -j1 --timeout 900 -E "softmax_attention" > "$R/gate-$LABEL-full.log" 2>&1
echo "full rc=$? $(grep -E '% tests passed' "$R/gate-$LABEL-full.log" | tail -1) $(grep -E 'Total Test time' "$R/gate-$LABEL-full.log" | tail -1)" >>"$CHAIN"
python3 "$R/passset_from_log.py" "$R/gate-$LABEL-sub.log" "$R/passset-sm120-tp2steps-$LABEL-sub.tsv" "steps $LABEL family subset" >>"$CHAIN" 2>&1
python3 "$R/passset_from_log.py" "$R/gate-$LABEL-full.log" "$R/passset-sm120-tp2steps-$LABEL.tsv" "steps $LABEL cluster gate -E softmax_attention" >>"$CHAIN" 2>&1
echo "done $(date -u +%H:%M:%S)" >>"$CHAIN"
