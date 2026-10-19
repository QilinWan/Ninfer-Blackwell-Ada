#!/usr/bin/env bash
set -uo pipefail
export S11_MAX_CONTEXT=262144 S11_MAX_VISION=1024
for D in 0 3 4 5 6; do
  pkill -x ninfer-serve 2>/dev/null; sleep 2
  if [ "$D" = 0 ]; then
    bash tools/serve-start.sh bench-nospec ./build-120a-cu13 >/dev/null 2>&1
    python3 tools/bench.py --log logs/req-bench-nospec.jsonl --label "nospec" --repeat 2
  else
    bash tools/serve-start.sh bench-d$D ./build-120a-cu13 --spec mtp --draft-tokens $D >/dev/null 2>&1
    python3 tools/bench.py --log logs/req-bench-d$D.jsonl --label "draft=$D" --repeat 2
  fi
  pkill -x ninfer-serve 2>/dev/null; sleep 2
done
