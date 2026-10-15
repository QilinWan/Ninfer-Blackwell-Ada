#!/bin/bash
# Find the largest --max-context that starts on this machine, for a given feature set.
# Usage: probe-max-context.sh <engine> <artifact> "<extra flags>" ctx1 ctx2 ...
# Detects OOM, capacity rejection and process death, and exits each probe early instead of
# waiting out the timeout.  Bare ceiling on a 32 GB Ada card: ~820K tokens; with
# --vision --spec dflash2 --max-concurrency 2: ~544K.
set -uo pipefail
ENGINE=$1; ARTIFACT=$2; EXTRA=${3:-}; shift 3
for CTX in "$@"; do
  pkill -x ninfer-serve 2>/dev/null; sleep 3
  nohup "$ENGINE" "$ARTIFACT" --host 127.0.0.1 --port 8099 \
    --max-context "$CTX" --kv-capacity auto --prefill-chunk 8192 \
    --kv-dtype rk4v4-e8 --rope-yarn-factor 4 --rope-original-max-position 262144 \
    $EXTRA > /tmp/probe_$CTX.log 2>&1 &
  PID=$!; OK=0
  for _ in $(seq 1 40); do
    sleep 3
    grep -qa "listening on" /tmp/probe_$CTX.log && { OK=1; break; }
    grep -qaiE "FATAL|out of memory|cudaErrorMemoryAllocation|bad_alloc" /tmp/probe_$CTX.log && break
    kill -0 $PID 2>/dev/null || break
  done
  if [ "$OK" = 1 ]; then
    printf "  %-9s OK   %s  VRAM %s\n" "$CTX" "$(grep -aoE 'KV [0-9,]+ tokens' /tmp/probe_$CTX.log | tail -1)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  else
    printf "  %-9s FAIL %s\n" "$CTX" "$(grep -aiE 'FATAL' /tmp/probe_$CTX.log | tail -1 | cut -c1-120)"
  fi
done
pkill -x ninfer-serve 2>/dev/null
