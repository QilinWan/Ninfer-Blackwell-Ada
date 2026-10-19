#!/usr/bin/env bash
# usage: serve-start.sh <label> <build-dir> [extra ninfer-serve args...]
set -euo pipefail
LABEL=$1; BUILD=$2; shift 2
mkdir -p /data/s11/logs
MODEL=${S11_MODEL:-/tank/models/Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/qwen3_8_27b_quasar_nvfp4.ninfer}
LOG=/data/s11/logs/serve-$LABEL.log
REQ=/data/s11/logs/req-$LABEL.jsonl
: > "$REQ"
# free the local Qwen stack's VRAM without touching its files
curl -s -m 120 -X POST -H 'Origin: http://127.0.0.1:8084' -H 'Content-Type: application/json' \
  -d '{}' 'http://127.0.0.1:8084/unload?wait=10' || true
echo
export LD_LIBRARY_PATH=/data/s11/deps-ada/lib
setsid nohup "$BUILD/apps/ninfer-serve" "$MODEL" --host 127.0.0.1 --port 8091 --model-id s11 \
  --tp 2 --devices 0,1 --vision --vision-device 0 \
  --kv-dtype nvfp4 --max-context "${S11_MAX_CONTEXT:-32768}" --kv-capacity "${S11_MAX_CONTEXT:-32768}" --max-concurrency "${S11_CONCURRENCY:-1}" \
  --prefill-chunk 1024 --max-vision-tokens "${S11_MAX_VISION:-1024}" --greedy \
  --request-log-jsonl "$REQ" "$@" > "$LOG" 2>&1 &
PID=$!
echo "pid=$PID log=$LOG req=$REQ"
for i in $(seq 1 240); do
  if ! kill -0 "$PID" 2>/dev/null; then echo "server exited early"; tail -20 "$LOG"; exit 1; fi
  if curl -s -m 2 -o /dev/null http://127.0.0.1:8091/v1/models; then echo "ready after ${i}s"; exit 0; fi
  sleep 1
done
echo "server did not become ready"; tail -20 "$LOG"; exit 1
