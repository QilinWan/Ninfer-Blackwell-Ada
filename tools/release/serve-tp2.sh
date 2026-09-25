#!/usr/bin/env bash
# Two-GPU launcher for the release tarball: serve-tp2.sh <artifact.ninfer> [ninfer-serve flags...]
# The production profile of the README (two 16 GB sm_120 boards, no P2P): INT8 KV, 196,608-token
# context, MTP with 3 draft tokens, Vision on device 0. Environment overrides:
#   NINFER_HOST (127.0.0.1)  NINFER_PORT (8080)  NINFER_DEVICES (0,1)  NINFER_MAX_CONTEXT (196608)
#   NINFER_VISION=0 drops the Vision flags (artifacts without a vision component).
# Extra flags are appended and win over the profile. Run it under a supervisor that restarts on a
# non-zero exit: ninfer-serve exits with status 2 once the Engine fails (docs/serving.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 <artifact.ninfer> [ninfer-serve flags...]" >&2; exit 2; }
artifact=$1; shift
context=${NINFER_MAX_CONTEXT:-196608}
vision=(--vision --vision-device 0 --max-vision-tokens 4096)
[ "${NINFER_VISION:-1}" = 0 ] && vision=()
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/ninfer-serve" "$artifact" \
  --host "${NINFER_HOST:-127.0.0.1}" --port "${NINFER_PORT:-8080}" \
  --tp 2 --devices "${NINFER_DEVICES:-0,1}" --kv-dtype int8 \
  --max-context "$context" --kv-capacity "$context" --device-state-slots 4 --max-concurrency 1 \
  --spec mtp --draft-tokens 3 "${vision[@]}" "$@"
