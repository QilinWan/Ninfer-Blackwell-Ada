#!/usr/bin/env bash
# Linux entry point of the agentic A/B suite: checks the workload corpus and the tools, then runs
# linux.py (two ninfer-serve builds under systemd user units, ABBA over the seeds).
#
# Usage: run_linux.sh --arm-a BIN --arm-b BIN --model ARTIFACT [linux.py options, see --help]
# Environment:
#   AB_PYTHON         interpreter (default python3; the suite uses the standard library only)
#   AB_CORPUS_REPO    git repository holding AB_CORPUS_COMMIT (default: this checkout)
#   AB_CORPUS_COMMIT  corpus commit of the observations (default e48a0d28, Wallawalla47/ninfer-custom)
# A long run belongs in its own unit, e.g.
#   systemd-run --user --unit=agab-run --collect -p WorkingDirectory=$PWD \
#     bash -c 'run_linux.sh ... > run.log 2>&1'
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${AB_PYTHON:-python3}
repo=${AB_CORPUS_REPO:-$(cd "$here/../.." && pwd)}
commit=${AB_CORPUS_COMMIT:-e48a0d28}
if ! git -C "$repo" cat-file -e "${commit}^{commit}" 2>/dev/null; then
  echo "corpus commit $commit is not in $repo; fetch it, e.g.:" >&2
  echo "  git clone --bare https://github.com/Wallawalla47/ninfer-custom DIR && export AB_CORPUS_REPO=DIR" >&2
  exit 2
fi
for tool in systemd-run systemctl nvidia-smi; do
  command -v "$tool" >/dev/null || { echo "$tool not found" >&2; exit 2; }
done
export AB_CORPUS_REPO=$repo AB_CORPUS_COMMIT=$commit
exec "$py" "$here/linux.py" "$@"
