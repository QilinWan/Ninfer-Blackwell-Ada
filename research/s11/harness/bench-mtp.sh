#!/usr/bin/env bash
set -uo pipefail
export S11_MAX_CONTEXT=262144 S11_MAX_VISION=1024
for D in 3 4 5; do
  pkill -x ninfer-serve 2>/dev/null; sleep 2
  bash tools/serve-start.sh bench-d$D ./build-120a-cu13 --spec mtp --draft-tokens $D >/dev/null 2>&1 || { echo "draft=$D start failed"; continue; }
  python3 tools/bench.py --log logs/req-bench-d$D.jsonl --label "draft=$D" --repeat 2
  pkill -x ninfer-serve 2>/dev/null; sleep 2
done
