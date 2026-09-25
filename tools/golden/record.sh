#!/usr/bin/env bash
# Records the tp 1 golden cases of one build: record.sh <ninfer-tp1-golden> <artifact> <out-dir>.
# Three greedy cases over three prefill regimes (one chunk, two chunks, six chunks with INT8 KV),
# 128 new tokens each, prompt seed 7. Compare two builds with `diff -r <out-a> <out-b>`.
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <ninfer-tp1-golden> <artifact> <out-dir>" >&2; exit 2; }
runner=$1; artifact=$2; out=$3
mkdir -p "$out"
run() { # name, args...
  local name=$1; shift
  if ! "$runner" "$artifact" --seed 7 --max-new 128 "$@" > "$out/$name.out" 2> "$out/$name.log"; then
    echo "$name: runner failed, see $out/$name.log" >&2; tail -3 "$out/$name.log" >&2; exit 1
  fi
  grep -E '^(prompt_tokens|generated_tokens|finish_reason|ids) ' "$out/$name.out" > "$out/$name.ids"
  echo "$name: $(grep '^generated_tokens' "$out/$name.ids")"
}
run case-1-short  --prompt-tokens 32   --max-context 4096
run case-2-medium --prompt-tokens 2048 --max-context 4096
run case-3-long   --prompt-tokens 6144 --max-context 8192 --kv-dtype int8
