#!/usr/bin/env bash
#
# Ada (sm_89) acceptance harness.
#
# Runs the checks that a sm_89 build of this repository must pass on Ada hardware, in the order
# that fails fastest, and writes one machine-readable result line per check. Every stage is
# idempotent and re-runnable; nothing here deletes a build tree or a downloaded artifact.
#
#   ARTIFACT=/root/models/qwen3.8-27b-lynn-df2.ninfer ./tools/verify_sm89.sh all
#   ./tools/verify_sm89.sh build ops              # narrow run
#
# Stages: env build ops load ppl capacity perf summary
#   env       record GPU, driver, CUDA, CMake, host compiler
#   build     configure + compile both shared-window routes (vision auto-detected)
#   ops       ctest over the Op families this port changed (FP8 encoding, Q8 shared window, PDL)
#   load      load the artifact and generate; compare greedy prefixes across speculation modes
#   ppl       fixed-corpus perplexity, BF16 KV vs INT8 KV
#   capacity  start the engine at the full native context on the selected KV dtype
#   perf      one prefill and one decode sample, recorded without performance claims
#
# Requires: an sm_89 GPU for every stage except env/build, and a v3 .ninfer artifact for
# load/ppl/capacity/perf (groupwise-int recipe; NVFP4 artifacts need a Blackwell build).
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ARCH=${ARCH:-89}
BUILD_DIR=${BUILD_DIR:-$REPO/build-sm$ARCH}
OUT=${OUT:-$REPO/out/verify-sm$ARCH}
ARTIFACT=${ARTIFACT:-}
CORPUS=${CORPUS:-$REPO/eval/corpora/perplexity-1m/manifest.json}
CTX=${CTX:-262144}
KV=${KV:-int8}
JOBS=${JOBS:-$(nproc)}
RESULTS=$OUT/results.tsv
mkdir -p "$OUT"
: > "$RESULTS"

log()  { printf '%-9s %s\n' "[$1]" "${*:2}"; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# record <stage> <name> <PASS|FAIL|SKIP> <detail>
record() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$(stamp)" "$1" "$2" "$3" "${4:-}" | tee -a "$RESULTS"
}

need_artifact() {
    if [[ -z "$ARTIFACT" || ! -f "$ARTIFACT" ]]; then
        record "$1" "artifact" SKIP "set ARTIFACT=/path/to/model.ninfer"
        return 1
    fi
    return 0
}

stage_env() {
    log env "collecting"
    {
        echo "date=$(stamp)"
        echo "git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)-$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader ||
            record env gpu FAIL "nvidia-smi unavailable"
        nvcc --version | grep -oE "release [0-9.]+" || record env nvcc FAIL "nvcc not on PATH"
        cmake --version | head -1
        c++ --version | head -1
        echo "uname=$(uname -a)"
    } > "$OUT/env.txt" 2>&1
    sed 's/^/    /' "$OUT/env.txt"
    record env collected PASS "$OUT/env.txt"
}

stage_build() {
    log build "configuring sm_$ARCH into $BUILD_DIR"
    local vision=ON
    pkg-config --exists 'libavcodec>=60' 2>/dev/null || vision=OFF
    [[ $vision == OFF ]] && log build "FFmpeg >= 6 not found; configuring with vision OFF"
    cmake -S "$REPO" -B "$BUILD_DIR" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_LINK_SMOKE=ON -DNINFER_ENABLE_VISION=$vision \
        -DBUILD_TESTING=ON > "$OUT/configure.log" 2>&1
    if [[ $? -ne 0 ]]; then
        record build configure FAIL "$(grep -m1 'CMake Error' -A3 "$OUT/configure.log" | tr '\n' ' ')"
        return 1
    fi
    record build configure PASS "vision=$vision"
    cmake --build "$BUILD_DIR" -j "$JOBS" > "$OUT/build.log" 2>&1
    if [[ $? -ne 0 ]]; then
        record build compile FAIL "$(grep -m3 -E 'error:' "$OUT/build.log" | tr '\n' ' ')"
        return 1
    fi
    record build compile PASS "$(wc -l < "$OUT/build.log") log lines"
    "$BUILD_DIR/tools/link_smoke/ninfer-link-smoke" > "$OUT/linksmoke.txt" 2>&1 &&
        record build link-smoke PASS "$(cat "$OUT/linksmoke.txt")" ||
        record build link-smoke FAIL "$(cat "$OUT/linksmoke.txt")"
}

stage_ops() {
    log ops "running the Op families this port changed"
    # FP8 encoding, Q8 dynamic shared window, PDL fallback, INT8/FP8 KV, real-model routes.
    local filter='linear_fp8|linear_add_fp8|linear_swiglu_fp8|attn_input_fp8|gdn_input_proj|linear_q8|linear_add_q8|linear_pair_q8|linear_swiglu_q8|attn_input_proj|softmax_attention|sliding_window|kv_cache_append|context_kv_materialize|sparse_moe|gdn_gating|gdn_replay|dynamic_grouped_conv|qwen3_5'
    ( cd "$BUILD_DIR" && ctest -R "$filter" --output-on-failure --timeout 900 ) \
        > "$OUT/ops.log" 2>&1
    local rc=$?
    local summary; summary=$(grep -E "tests passed|tests failed" "$OUT/ops.log" | tail -2 | tr '\n' ' ')
    log ops "$summary"
    grep -E "^\s+[0-9]+/[0-9]+ Test.*Not Run|\*\*\*Failed" "$OUT/ops.log" | head -10 | sed 's/^/    /'
    [[ $rc -eq 0 ]] && record ops ctest PASS "$summary" || record ops ctest FAIL "$summary"
    return $rc
}

# generate <label> <extra args...>
generate() {
    local label=$1; shift
    "$BUILD_DIR/apps/ninfer" "$ARTIFACT" \
        --prompt "Explain in three short bullet points why a memory-bound decoder favours large L2 cache and high bandwidth over extra CUDA cores." \
        --max-context 16384 --max-new 192 --kv-dtype "$KV" "$@" \
        > "$OUT/$label.txt" 2> "$OUT/$label.err"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        record load "$label" FAIL "exit $rc: $(tail -2 "$OUT/$label.err" | tr '\n' ' ')"
        return 1
    fi
    record load "$label" PASS "$(wc -c < "$OUT/$label.txt") chars"
    grep -oE "decode[^|]*tok/s|generated [0-9]+ tokens" "$OUT/$label.err" | tail -2 | sed 's/^/      /'
    return 0
}

stage_load() {
    need_artifact load || return 0
    log load "artifact: $ARTIFACT"
    generate no-spec --spec none
    generate mtp --spec mtp --draft-tokens 3
    # Speculation must not change greedy output: compare the accepted prefixes.
    if [[ -s "$OUT/no-spec.txt" && -s "$OUT/mtp.txt" ]]; then
        if head -c 400 "$OUT/no-spec.txt" | sed 's/[[:space:]]//g' | grep -qF "$(head -c 200 "$OUT/mtp.txt" | sed 's/[[:space:]]//g' | cut -c1-60)"; then
            record load greedy-agreement PASS "first 60 chars of the no-spec answer contain the mtp answer"
        else
            record load greedy-agreement FAIL "prefixes differ; inspect $OUT/{no-spec,mtp}.txt"
        fi
    fi
    grep -iE "nan|illegal memory|CUDA error|unsupported" "$OUT"/*.err 2>/dev/null | head -3 |
        sed 's/^/    /' && record load cleanliness PASS "no NaN or CUDA error in run logs"
}

stage_ppl() {
    need_artifact ppl || return 0
    [[ -f "$CORPUS" ]] || { record ppl corpus SKIP "missing $CORPUS"; return 0; }
    for dtype in bf16 "$KV"; do
        log ppl "quick corpus with $dtype KV"
        "$BUILD_DIR/apps/ninfer-perplexity" "$ARTIFACT" --corpus "$CORPUS" --quick \
            --kv-dtype "$dtype" > "$OUT/ppl-$dtype.txt" 2> "$OUT/ppl-$dtype.err"
        local line; line=$(grep -iE "overall|perplexity" "$OUT/ppl-$dtype.txt" | tail -1)
        if [[ $? -eq 0 && -n "$line" ]]; then record ppl "$dtype" PASS "$line"
        else record ppl "$dtype" FAIL "$(tail -2 "$OUT/ppl-$dtype.err" | tr '\n' ' ')"; fi
    done
}

stage_capacity() {
    need_artifact capacity || return 0
    log capacity "starting the engine at $CTX tokens with $KV KV"
    "$BUILD_DIR/apps/ninfer" "$ARTIFACT" --prompt "Reply with the single word: ready" \
        --max-context "$CTX" --max-new 8 --kv-dtype "$KV" \
        > "$OUT/capacity.txt" 2> "$OUT/capacity.err"
    local rc=$?
    local report; report=$(grep -oiE "kv[^\n]*(capacity|tokens)[^\n]*" "$OUT/capacity.err" | tail -1)
    if [[ $rc -eq 0 ]]; then record capacity "$CTX/$KV" PASS "${report:-started}"
    else record capacity "$CTX/$KV" FAIL "${report:-$(tail -2 "$OUT/capacity.err" | tr '\n' ' ')}"; fi
}

stage_perf() {
    need_artifact perf || return 0
    log perf "sampling decode and prefill (no performance claim, just order of magnitude)"
    "$BUILD_DIR/apps/ninfer" "$ARTIFACT" --prompt "Count from 1 to 200, one number per line." \
        --max-context 32768 --max-new 1024 --kv-dtype "$KV" --spec mtp --draft-tokens 3 \
        > "$OUT/perf-decode.txt" 2> "$OUT/perf-decode.err"
    grep -oiE "(decode|prefill|throughput|ttft)[^|]*[0-9.]+[^|]*(tok/s|ms|s)\b" \
        "$OUT/perf-decode.err" | tail -6 | sed 's/^/    /' | tee "$OUT/perf-sample.txt"
    record perf sample PASS "$(tr '\n' ';' < "$OUT/perf-sample.txt")"
}

stage_summary() {
    log summary "$RESULTS"
    column -t -s $'\t' "$RESULTS" | sed 's/^/  /'
    local fails; fails=$(awk -F'\t' '$4=="FAIL"' "$RESULTS" | wc -l)
    echo "  ---- FAIL rows: $fails"
    [[ $fails -eq 0 ]]
}

main() {
    local stages=("$@")
    [[ ${#stages[@]} -eq 0 || "${stages[0]}" == all ]] &&
        stages=(env build ops load ppl capacity perf summary)
    for s in "${stages[@]}"; do
        case "$s" in
            env) stage_env ;; build) stage_build ;; ops) stage_ops ;;
            load) stage_load ;; ppl) stage_ppl ;; capacity) stage_capacity ;;
            perf) stage_perf ;; summary) stage_summary ;;
            *) echo "unknown stage: $s" >&2; exit 2 ;;
        esac
    done
}

main "$@"
