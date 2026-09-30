#!/usr/bin/env bash
# The fork's footprint inside upstream's files: tools/tp2/surface.sh [<upstream-ref> [<ref>]]
# (defaults up/master and HEAD). Prints, for the whole tree and for src/+include/, the upstream
# files the fork modifies, the lines added/removed inside them, the diff hunks inside them (-U0 and
# git's default context) and the fork's own files; then the files with the most hunks. Compare the
# numbers before and after a merge or a refactor of the tp2 layer (docs/maintainer/upstream-merge.md).
set -euo pipefail
base=${1:-up/master}; ref=${2:-HEAD}
row() { # scope...
    local files lines h0 h3 own
    files=$(git diff --no-renames --diff-filter=M --name-only "$base" "$ref" -- "$@" | wc -l)
    lines=$(git diff --no-renames --diff-filter=M --numstat "$base" "$ref" -- "$@" | awk '{a+=$1; d+=$2} END{printf "+%d/-%d", a, d}')
    h0=$(git diff --no-renames --diff-filter=M -U0 "$base" "$ref" -- "$@" | grep -c '^@@' || true)
    h3=$(git diff --no-renames --diff-filter=M "$base" "$ref" -- "$@" | grep -c '^@@' || true)
    own=$(git diff --no-renames --diff-filter=A --name-only "$base" "$ref" -- "$@" | wc -l)
    printf '%-14s upstream files modified %4d | lines %-14s | hunks -U0 %5d, default %4d | own files %4d\n' \
        "$*" "$files" "$lines" "$h0" "$h3" "$own"
}
echo "$(git rev-parse --short=8 "$base") .. $(git rev-parse --short=8 "$ref")"
row .
row src include
echo "most hunks (-U0) inside upstream files:"
git diff --no-renames --diff-filter=M -U0 "$base" "$ref" \
    | awk '/^\+\+\+ b\//{f=substr($0,7)} /^@@/{h[f]++} END{for (k in h) print h[k], k}' | sort -rn | head -"${TOP:-15}"
