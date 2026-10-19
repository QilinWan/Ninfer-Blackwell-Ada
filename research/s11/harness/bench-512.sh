#!/usr/bin/env bash
set -uo pipefail
export S11_MAX_CONTEXT=524288 S11_MAX_VISION=1024
for cfg in "4 1" "5 1" "5 2" "5 4"; do
  set -- $cfg; D=$1; C=$2
  pkill -x ninfer-serve 2>/dev/null; sleep 3
  export S11_CONCURRENCY=$C
  if ! bash tools/serve-start.sh b512-d$D-c$C ./build-120a-cu13 --spec mtp --draft-tokens $D --rope-yarn-factor 2 --rope-original-max-position 262144 >/tmp/start-$D-$C.txt 2>&1; then
    echo "draft=$D conc=$C: START FAILED $(tail -1 /tmp/start-$D-$C.txt | cut -c1-140)"; continue
  fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' '/')
  python3 tools/bench.py --log logs/req-b512-d$D-c$C.jsonl --label "512K d$D c$C mem=$mem" --repeat 2 | tail -1
  pkill -x ninfer-serve 2>/dev/null; sleep 2
done
