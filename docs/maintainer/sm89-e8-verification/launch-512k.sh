#!/bin/bash
# NInfer 512K / 2-concurrent / vision / speculative launcher for the Ada (sm_89) build.
# Validated 2026-09-25 on a 32 GB Ada card with the 27B artifacts: KV 544,128 tokens,
# 30.3 GiB VRAM, decode 83-92 tok/s, speculative acceptance 36-42% (DFlash2) or 63-67% (MTP),
# image request answered correctly.
#
# Usage:  launch-512k.sh <engine-binary> <artifact.ninfer> [dflash2|mtp] [extra args...]
#   e.g.  launch-512k.sh ./build-sm89-full/apps/ninfer-serve /models/qwen3.8-27b-lynn-df2.ninfer dflash2
set -euo pipefail
ENGINE=${1:?engine binary}; ARTIFACT=${2:?artifact}; SPEC=${3:-dflash2}; shift 3 || shift $#
case "$SPEC" in
  mtp)     DRAFT=3 ;;
  dflash2) DRAFT=7 ;;
  *) echo "spec must be mtp or dflash2" >&2; exit 2 ;;
esac
exec "$ENGINE" "$ARTIFACT" \
  --host 0.0.0.0 --port 8080 \
  --max-context 512000 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --kv-dtype rk4v4-e8 \
  --rope-yarn-factor 4 \
  --rope-original-max-position 262144 \
  --vision \
  --spec "$SPEC" --draft-tokens "$DRAFT" \
  --prefill-chunk 8192 \
  --max-request-mib 64 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --default-max-tokens 8192 \
  "$@"
