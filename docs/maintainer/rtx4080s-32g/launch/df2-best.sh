#!/usr/bin/env bash
# ============================================================================
#  qwen3.8-27b-lynn-df2.ninfer  ·  NInfer (Ada sm_89)  ·  RTX 4080 SUPER 32GB
#  Tuned 2026-09-24 on autodl-container-84a24c9a34-7083e900 (driver 595.71.05)
#
#  Measured single-stream decode (greedy, 512-1024 new tokens, code prompt):
#      --spec dflash2 --draft-tokens 7 --lm-head-draft   90-101 tok/s
#      --spec dflash2 --draft-tokens 3 --lm-head-draft   ~96 tok/s (most stable)
#      no speculation                                    ~40 tok/s
#  Prefill (8K-token prompt): ~1.4-1.5k tok/s (chunk has almost no effect on Ada)
#
#  Usage:
#    ./df2-best.sh                      # interactive-ish one-shot, 1024 tokens
#    ./df2-best.sh ask "your question"
#    ./df2-best.sh serve                # OpenAI/Anthropic HTTP server on :8080
#    MODE=quality ./df2-best.sh ask "hi"   # model-preset sampling instead of greedy
#    CTX=131072 ./df2-best.sh ask "hi"     # smaller KV pool
#
#  Env knobs: CTX KVCAP KV SPEC DRAFT LMD CHUNK VISION MODE NEW PORT LANES
# ============================================================================
set -uo pipefail
[ -f /root/ninfer/env.sh ] && . /root/ninfer/env.sh

ENGINE=${ENGINE:-/root/ninfer/bin/ninfer}
SERVE=${SERVE:-/root/ninfer/bin/ninfer-serve}
ARTIFACT=${ARTIFACT:-/root/autodl-tmp/models/qwen3.8-27b-lynn-df2.ninfer}

# ---- speed-tuned defaults (measured, see README) ---------------------------
CTX=${CTX:-262144}      # per-sequence context ceiling (native 262144)
KVCAP=${KVCAP:-262144}  # shared Main Text KV pool (single user may use all of it)
KV=${KV:-int8}          # int8 beats fp8 by ~10% on this card; nvfp4/k8v4 are Blackwell-only
SPEC=${SPEC:-dflash2}   # this artifact carries the DFlash2 draft head
DRAFT=${DRAFT:-7}       # K=7 measured best; K=3 is the steadier alternative
LMD=${LMD:-1}           # --lm-head-draft: +4..5% at K=3, +11% at K=7
CHUNK=${CHUNK:-2048}    # prefill chunk (multiple of 128); 2048 is the safe default
VISION=${VISION:-0}     # 1 enables image/video input; costs VRAM, zero text-speed effect
MODE=${MODE:-fast}      # fast = exact argmax (max tok/s, deterministic)

FLAGS=(--max-context "$CTX" --kv-capacity "$KVCAP" --kv-dtype "$KV"
       --prefill-chunk "$CHUNK")
[ "$SPEC" != none ] && FLAGS+=(--spec "$SPEC" --draft-tokens "$DRAFT")
[ "$LMD" = 1 ] && [ "$SPEC" != none ] && FLAGS+=(--lm-head-draft)
[ "$VISION" = 1 ] && FLAGS+=(--vision)

# ---- sampling --------------------------------------------------------------
# fast    : argmax. Highest speculative acceptance -> highest tok/s.
# quality : Qwen3.8 registered presets (docs/cli.md). Costs a few % of tok/s.
case "$MODE" in
  fast)  FLAGS+=(--greedy) ;;
  think) FLAGS+=(--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0) ;;
  nothink) FLAGS+=(--no-thinking --temperature 0.7 --top-p 0.80 --top-k 20 \
                   --min-p 0 --presence-penalty 1.5) ;;
  *)     echo "MODE must be fast|think|nothink" >&2; exit 2 ;;
esac

case "${1:-ask}" in
  serve) shift; exec "$SERVE" "$ARTIFACT" "${FLAGS[@]}" --port "${PORT:-8080}" \
                              --max-concurrency "${LANES:-1}" "$@" ;;
  ask)   shift; exec "$ENGINE" "$ARTIFACT" --prompt "$*" --max-new "${NEW:-1024}" "${FLAGS[@]}" ;;
  raw)   shift; exec "$ENGINE" "$ARTIFACT" "$@" "${FLAGS[@]}" ;;
  *)     exec "$ENGINE" "$ARTIFACT" --prompt "Reply with one short sentence." \
                            --max-new "${NEW:-1024}" "${FLAGS[@]}" ;;
esac
