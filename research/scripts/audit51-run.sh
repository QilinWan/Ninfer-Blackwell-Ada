#!/usr/bin/env bash
# audit51-run.sh - 开窗前置：树尖 d01dbcf7 全构建门链（含 bd1be0ec/340bd4c0，线性祖先）+ 12/2 资格面。
# 纪律：gate.sh 不抓卡；本脚本固定 CUDA_VISIBLE_DEVICES=1；不同时跑两份门链。
set -uo pipefail
R=/home/ai-agent/DSHW/ninfer-watch/relay
G=/home/ai-agent/DSHW/ninfer-watch/gate-72
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$R/../deps-ada/lib
CARD=1
L=$R/audit51-run.log
: > $L
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
say "chain start for d01dbcf7 label audit51"
GATE_PATCHES=$R/vision-on-build-ada.patch nice -n 12 bash $R/gate-steps.sh d01dbcf7 audit51 >> $L 2>&1
say "chain rc=$? $(grep -E '^(build|sub|full) rc=' $R/gate-audit51-chain.log | tr '\n' '|')"
say "qual face start"
cd $G/build-120a-cu13 || exit 9
CUDA_VISIBLE_DEVICES=$CARD nice -n 12 ctest -j1 --timeout 1800 -R softmax_attention > $R/audit51-qual-ctest.log 2>&1
say "qual ctest rc=$? $(grep -m1 'tests passed' $R/audit51-qual-ctest.log) $(grep -m1 'Total Test time' $R/audit51-qual-ctest.log)"
: > $R/audit51-qual.log
for dt in nvfp4 int8 bf16; do
  CUDA_VISIBLE_DEVICES=$CARD nice -n 12 ./tests/ninfer_softmax_attention_test --kv-dtype $dt > $R/audit51-qual-$dt.log 2>&1
  echo "dtype $dt rc=$? last=$(tail -1 $R/audit51-qual-$dt.log | cut -c1-70)" >> $R/audit51-qual.log
done
say "all closed"
