#!/usr/bin/env bash
# finish-everything.sh — 把开窗前的活一路跑完：audit49b 资格面→tag；gate.sh golden 实跑（验 ids-only 修复）；
# audit50 面（SM 数相等闸）链→资格面→tag；S3 的 case-3-long bisect；最后剥离演练（CPU，独占构建）。
# 全程只用测试卡（GPU1，现网 strata 在 GPU0＝PID 3013603，不碰不 kill）；不同时跑两份门链。
set -uo pipefail
R=/home/ai-agent/DSHW/ninfer-watch/relay
B=/home/ai-agent/DSHW/ninfer-watch/ada-tree
G=/home/ai-agent/DSHW/ninfer-watch/gate-72
W=/tank/ninfer-window
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$R/../deps-ada/lib"
CARD=1
L="$R/finish.log"; : > "$L"
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$L"; }
tag_from_chain(){ # sha label tagname  从链日志抓数，全绿才打 tag
  local sha=$1 label=$2 name=$3 cl="$R/gate-$label-chain.log"
  local b=$(grep -m1 "^build rc=" "$cl"); local su=$(grep -m1 "^sub rc=" "$cl")
  local fu=$(grep -m1 "^full rc=" "$cl"); local q="$R/fin-$label-qual-ctest.log"
  local qs="qual ctest: $(grep -m1 'tests passed' "$q" | tr -s ' ') $(grep -m1 'Total Test time' "$q")"
  local dt=$(grep '^dtype' "$R/fin-$label-qual.log" 2>/dev/null | tr '\n' ';')
  if ! { echo "$b" | grep -q "rc=0"; } || ! { echo "$su" | grep -q "rc=0"; } || ! { echo "$fu" | grep -q "rc=0"; } \
     || ! { echo "$su" | grep -q "100%"; } || ! { echo "$fu" | grep -q "100%"; } \
     || ! grep -q "tests passed out of" "$q"; then
    say "NOT TAGGING $label (a face is red): $b | $su | $fu | $qs"; return 1
  fi
  ( cd "$B" && git tag -a "$name" "$sha" -m "gate at $sha (label $label, patch set: vision-on-build-ada.patch only; S6 is in the tree): $b; $su; $fu; $qs; per-dtype $dt; two-device suites skip with code 77 on one card = not two-card evidence." ) \
    && say "TAGGED $name @ $sha" || say "tag failed for $label"
}
qual_face(){ # label  在 gate-72 面上跑 12/2 资格面（簇门 -E 排除它）
  cd "$G/build-120a-cu13" || return 9
  CUDA_VISIBLE_DEVICES=$CARD nice -n 12 ctest -j1 --timeout 1800 -R softmax_attention > "$R/fin-$1-qual-ctest.log" 2>&1
  echo "qual ctest rc=$? $(grep -m1 'tests passed' "$R/fin-$1-qual-ctest.log") $(grep -m1 'Total Test time' "$R/fin-$1-qual-ctest.log")" >> "$L"
  : > "$R/fin-$1-qual.log"
  for dt in nvfp4 int8 bf16; do
    CUDA_VISIBLE_DEVICES=$CARD nice -n 12 ./tests/ninfer_softmax_attention_test --kv-dtype "$dt" > "$R/fin-$1-qual-$dt.log" 2>&1
    echo "dtype $dt rc=$?" >> "$R/fin-$1-qual.log"
  done
  cat "$R/fin-$1-qual.log" >> "$L"
}
say "waiting for audit49b chain"
while ! grep -q "^done " "$R/gate-audit49b-chain.log" 2>/dev/null; do sleep 30; done
say "audit49b: $(grep -E '^(build|sub|full) rc=' "$R/gate-audit49b-chain.log" | tr '\n' '|')"
qual_face audit49b
tag_from_chain 035403d6 audit49b v-tp2-audit49-per-rank-arena-and-oracle-gates-r0

say "gate.sh golden run (validates the ids-only comparison fix)"
mkdir -p "$W/ref-synthetic/golden" "$W/out-golden"
cp -f "$B/tools/golden/recorded/2026-09-30/fork-f19e6821/"*.ids "$W/ref-synthetic/golden/" 2>/dev/null
cp -f "$B/tools/tp2/reference/prompts.jsonl" "$W/ref-synthetic/" 2>/dev/null
( cd "$B" && GATE_DEVICES="1,1" GATE_GOLDEN_MODEL=/tmp/synthetic-tp1/model.ninfer \
    GATE_SRC="$B" timeout 900 bash tools/tp2/gate.sh --stages golden --out "$W/out-golden" \
    "$B/build-120a-cu13" "$W/ref-synthetic" > "$R/fin-gate-golden.log" 2>&1 )
say "gate.sh golden rc=$?  $(grep -E '^golden' "$W/out-golden/verdicts.txt" 2>/dev/null | tr '\n' '|')"

say "chain for bd1be0ec (label audit50, SM-count equality gate)"
GATE_PATCHES="$R/vision-on-build-ada.patch" nice -n 12 bash "$R/gate-steps.sh" bd1be0ec audit50 >> "$L" 2>&1
say "audit50: $(grep -E '^(build|sub|full) rc=' "$R/gate-audit50-chain.log" | tr '\n' '|')"
qual_face audit50
tag_from_chain bd1be0ec audit50 v-tp2-sm-count-equality-and-followups-r0

say "S3 bisect (case-3-long: six-chunk INT8 KV) on the main tree"
( cd "$B" && git bisect start 7d762ec5 03726c08~ -- >/dev/null 2>&1 && \
  S3_BISECT_CARD=$CARD RUN=1 timeout 5400 git bisect run "$R/s3-bisect.sh" > "$R/fin-s3-bisect.log" 2>&1; \
  echo "bisect rc=$?" >> "$R/fin-s3-bisect.log"; git bisect reset >> "$R/fin-s3-bisect.log" 2>&1 )
say "S3 bisect done: $(grep -m1 'first bad commit' "$R/fin-s3-bisect.log" | cut -c1-90)"

say "peel drill (S10 stage 1: bounded newest 25 commits, own build dir on /tank)"
rm -rf /tank/peel-drill /tank/peel-build
( cd "$B" && git worktree add -f /tank/peel-drill HEAD > "$R/fin-peel.log" 2>&1 && cd /tank/peel-drill && \
  git revert --no-commit HEAD~25..HEAD >> "$R/fin-peel.log" 2>&1; echo "revert rc=$?" >> "$R/fin-peel.log"; \
  git -c core.editor=true commit -m "drill: strip a bounded tp 2 segment" >> "$R/fin-peel.log" 2>&1; \
  JOBS=12 nice -n 19 bash scripts/build-ada.sh 120a --tests >> "$R/fin-peel-build.log" 2>&1; \
  echo "peel build rc=$?" >> "$R/fin-peel.log"; \
  git grep -c "tensor_parallel\|TpExecution\|PeerRuntime\|peer_mailbox" -- src | head -3 >> "$R/fin-peel.log" 2>&1; \
  cd "$B" && git worktree remove --force /tank/peel-drill >> "$R/fin-peel.log" 2>&1 )
say "peel drill: $(grep -E 'revert rc=|peel build rc=' "$R/fin-peel.log" | tr '\n' '|')"
say "ALL DONE"
