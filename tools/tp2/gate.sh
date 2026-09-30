#!/usr/bin/env bash
# Behavior-preservation gate for the two-GPU layer: one command, PASS/FAIL per stage.
#
#   tools/tp2/gate.sh [--record] [--stages LIST] [--attention] [--out DIR] <build-dir> <reference-dir>
#
# --record writes <reference-dir> from this build instead of comparing against it. Stages, in order
# (default: all but build):
#   build    cmake --build <build-dir> (no GPU)
#   ctest    the unit, Op and two-device tests that a tp2 change can reach, plus the tp2 *_real tests
#            (NINFER_TEST_ARTIFACT); --attention adds ninfer_softmax_attention_test (~12 min)
#   golden   tools/golden/record.sh on the synthetic tp1 model; ids must equal the reference's
#   ppl      ninfer-perplexity --tp 2 --quick on the QUASAR artifact, 65536/32768 and 4096/2048,
#            INT8 KV; every printed digit of the per-source table must equal the reference's
#   greedy   ninfer-serve with the production flags (MTP3, --lm-head-draft, C=1), the reference's
#            60 prompts at T=0 / 128 tokens; texts must equal the reference's; ms/round (median of
#            decode seconds / MTP rounds) is reported against it (PASS within +-1 %, else WARN)
#   dflash2  a short DFlash2 K=7 --lm-head-draft run on the _df2 artifact (10 prompts), same checks
#
# The gate does not take a GPU lease; wrap it (e.g. gpu-lease run ricerca -- tools/tp2/gate.sh ...).
# Paths default to the development workstation; override them with the GATE_* variables below.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=${GATE_SRC:-$(cd "$here/../.." && pwd)}   # source tree: golden record.sh, perplexity corpus

ARTIFACT=${GATE_ARTIFACT:-/home/feyd/ninfer-artifacts/qwen3_8_27b_quasar_nvfp4.ninfer}
DF2_ARTIFACT=${GATE_DF2_ARTIFACT:-/home/feyd/ninfer-artifacts/qwen3_8_27b_quasar_nvfp4_df2.ninfer}
TEST_ARTIFACT=${GATE_TEST_ARTIFACT:-$DF2_ARTIFACT}
TEST_SCRATCH=${GATE_TEST_SCRATCH:-/home/feyd/ninfer-artifacts/tmp}
GOLDEN_MODEL=${GATE_GOLDEN_MODEL:-/home/feyd/ninfer-artifacts/tmp/golden-sync/synthetic/model.ninfer}
PROMPTS=${GATE_PROMPTS:-}           # --record only: JSONL of {"id","text"}; copied into the reference
DEVICES=${GATE_DEVICES:-0,1}
PORT=${GATE_PORT:-8090}
CORPUS=${GATE_CORPUS:-$src/eval/corpora/perplexity-1m/manifest.json}
SERVE_FLAGS=${GATE_SERVE_FLAGS:---tp 2 --devices $DEVICES --kv-dtype int8 --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft --vision --vision-device 0 --max-vision-tokens 4096}
DF2_FLAGS=${GATE_DF2_FLAGS:---tp 2 --devices $DEVICES --kv-dtype int8 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --spec dflash2 --draft-tokens 7 --lm-head-draft}
DF2_PROMPTS=${GATE_DF2_PROMPTS:-10}
# Tests that cannot run on one 16 GB board (tp1 27B artifacts) or need a different artifact.
CTEST_EXCLUDE=${GATE_CTEST_EXCLUDE:-ninfer_qwen3_5_(prefix|score|moe|dflash|dflash2|dflash_prefill)_real_test|ninfer_qwen3_5_vision_workspace_test}

record=0; stages="ctest,golden,ppl,greedy,dflash2"; attention=0; out=""
while [ $# -gt 0 ]; do
    case $1 in
        --record) record=1; shift ;;
        --stages) stages=$2; shift 2 ;;
        --attention) attention=1; shift ;;
        --out) out=$2; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -eq 2 ] || { echo "usage: $0 [--record] [--stages LIST] [--attention] [--out DIR] <build-dir> <reference-dir>" >&2; exit 2; }
build=$(cd "$1" && pwd); ref=$2
mkdir -p "$ref"; ref=$(cd "$ref" && pwd)
[ -n "$out" ] || out=$build/gate/$(date +%Y%m%d-%H%M%S)
[ $record = 1 ] && out=$ref
mkdir -p "$out"; out=$(cd "$out" && pwd)
summary=$out/summary.txt; : > "$summary"
failed=0

say() { echo "[$(date '+%F %T')] $*" | tee -a "$summary"; }
verdict() { # stage status detail
    say "$(printf '%-8s %-5s %s' "$1" "$2" "$3")"
    [ "$2" = FAIL ] && failed=1
    return 0
}
has() { [[ ",$stages," == *",$1,"* ]]; }
gpu_busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }

say "gate $([ $record = 1 ] && echo RECORD || echo CHECK) | build $build | ref $ref | out $out | src $(git -C "$src" rev-parse --short=8 HEAD 2>/dev/null) dirty=$(git -C "$src" status --porcelain 2>/dev/null | wc -l) | stages $stages"
say "clocks $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | tr '\n' ' ')"

# ---------------------------------------------------------------- build
if has build; then
    t0=$(date +%s)
    if cmake --build "$build" -j"${GATE_JOBS:-12}" > "$out/build.log" 2>&1; then
        verdict build PASS "$(( $(date +%s) - t0 )) s, $(grep -c warning "$out/build.log") warnings"
    else
        verdict build FAIL "see $out/build.log"; say "RESULT FAIL"; exit 1
    fi
fi

# ---------------------------------------------------------------- ctest
if has ctest; then
    t0=$(date +%s)
    exclude=$CTEST_EXCLUDE
    [ $attention = 1 ] || exclude="$exclude|ninfer_softmax_attention_test"
    NINFER_TEST_ARTIFACT=$TEST_ARTIFACT NINFER_TEST_SCRATCH_DIR=$TEST_SCRATCH \
        ctest --test-dir "$build" -j1 --timeout 900 --output-on-failure -E "$exclude" > "$out/ctest.log" 2>&1
    rc=$?
    line=$(grep -E 'tests passed|tests failed' "$out/ctest.log" | tail -1)
    tp2=$(grep -cE 'tp2.*Passed' "$out/ctest.log")
    if [ $rc = 0 ]; then verdict ctest PASS "$line; tp2 tests passed: $tp2; $(( $(date +%s) - t0 )) s"
    else verdict ctest FAIL "$line; $(grep -E '\*\*\*|Failed' "$out/ctest.log" | grep -oE 'ninfer_[a-z0-9_]+' | sort -u | tr '\n' ' ')"; fi
fi

# ---------------------------------------------------------------- golden
if has golden; then
    g=$out/golden; rm -rf "$g"
    if CUDA_VISIBLE_DEVICES=${DEVICES%%,*} "$src/tools/golden/record.sh" "$build/apps/ninfer-tp1-golden" "$GOLDEN_MODEL" "$g" > "$out/golden.log" 2>&1; then
        if [ $record = 1 ]; then verdict golden REC "$(tr '\n' ' ' < "$out/golden.log")"
        elif diff -r "$ref/golden" "$g" > "$out/golden.diff" 2>&1; then
            verdict golden PASS "3/3 identical ($(cat "$g"/*.ids | md5sum | cut -c1-8))"
        else verdict golden FAIL "ids differ, see $out/golden.diff"; fi
    else verdict golden FAIL "runner failed, see $out/golden.log"; fi
fi

# ---------------------------------------------------------------- ppl
if has ppl; then
    for p in "64k 65536 32768" "4k 4096 2048"; do
        read -r tag ctx stride <<< "$p"
        d=$out/ppl-$tag; rm -rf "$d"; mkdir -p "$d"
        t0=$(date +%s)
        "$build/apps/ninfer-perplexity" "$ARTIFACT" --corpus "$CORPUS" --quick --context "$ctx" --stride "$stride" \
            --tp 2 --devices "$DEVICES" --kv-dtype int8 --output "$d" > "$d.log" 2>&1
        rc=$?
        # The per-source table and the overall line, without the rate and paths.
        awk 'f{print} /^domain/{f=1} /^overall/{exit}' "$d.log" > "$d.table"
        ov=$(grep -E '^overall' "$d.log" | awk '{print $NF}')
        if [ $rc != 0 ] || [ -z "$ov" ]; then verdict "ppl-$tag" FAIL "rc=$rc, see $d.log"
        elif [ $record = 1 ]; then verdict "ppl-$tag" REC "$ov ($(( $(date +%s) - t0 )) s)"
        elif diff "$ref/ppl-$tag.table" "$d.table" > /dev/null; then
            verdict "ppl-$tag" PASS "$ov, $(wc -l < "$d.table") lines identical ($(( $(date +%s) - t0 )) s)"
        else verdict "ppl-$tag" FAIL "$ov vs $(grep -E '^overall' "$ref/ppl-$tag.table" | awk '{print $NF}')"; fi
    done
fi

# ---------------------------------------------------------------- serve stages
serve_pid=""
serve_down() {
    [ -n "$serve_pid" ] || return 0
    kill "$serve_pid" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$serve_pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$serve_pid" 2>/dev/null; wait "$serve_pid" 2>/dev/null; serve_pid=""; sleep 2
}
trap serve_down EXIT
serve_up() { # tag artifact flags...
    local tag=$1 art=$2; shift 2
    if gpu_busy; then say "$tag: GPU not empty: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')"; return 1; fi
    rm -f "$out/req-$tag.jsonl"
    "$build/apps/ninfer-serve" "$art" --host 127.0.0.1 --port "$PORT" --model-id gate "$@" \
        --request-log-jsonl "$out/req-$tag.jsonl" > "$out/serve-$tag.log" 2>&1 &
    serve_pid=$!
    for _ in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "127.0.0.1:$PORT/health")" = 200 ] && return 0
        kill -0 "$serve_pid" 2>/dev/null || { say "$tag: server died, see $out/serve-$tag.log"; serve_pid=""; return 1; }
        sleep 0.5
    done
    say "$tag: no health"; serve_down; return 1
}
serve_stage() { # tag artifact limit flags...
    local tag=$1 art=$2 limit=$3; shift 3
    local t0; t0=$(date +%s)
    if ! serve_up "$tag" "$art" "$@"; then verdict "$tag" FAIL "server did not start"; return; fi
    python3 "$here/gate_client.py" run "127.0.0.1:$PORT" "$ref/prompts.jsonl" "$limit" "$out/$tag.jsonl" > "$out/$tag.client.log" 2>&1
    local rc=$?
    serve_down
    if [ $rc != 0 ]; then verdict "$tag" FAIL "client rc=$rc, see $out/$tag.client.log"; return; fi
    local res
    if [ $record = 1 ]; then
        res=$(python3 "$here/gate_client.py" summarize "$out/$tag.jsonl" "$out/req-$tag.jsonl")
        verdict "$tag" REC "$res ($(( $(date +%s) - t0 )) s)"
    else
        res=$(python3 "$here/gate_client.py" compare "$ref/$tag.jsonl" "$ref/req-$tag.jsonl" "$out/$tag.jsonl" "$out/req-$tag.jsonl")
        local st=${res%% *}
        verdict "$tag" "$st" "${res#* } ($(( $(date +%s) - t0 )) s)"
    fi
}
if [ $record = 1 ] && { has greedy || has dflash2; }; then
    [ -n "$PROMPTS" ] || { say "--record needs GATE_PROMPTS"; exit 2; }
    cp "$PROMPTS" "$ref/prompts.jsonl"
fi
# shellcheck disable=SC2086
has greedy && serve_stage greedy "$ARTIFACT" 0 $SERVE_FLAGS
# shellcheck disable=SC2086
has dflash2 && serve_stage dflash2 "$DF2_ARTIFACT" "$DF2_PROMPTS" $DF2_FLAGS

if [ $record = 1 ]; then say "RESULT RECORDED"; exit $failed; fi
[ $failed = 0 ] && say "RESULT PASS" || say "RESULT FAIL"
exit $failed
