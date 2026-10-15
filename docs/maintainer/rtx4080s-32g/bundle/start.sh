#!/bin/sh
# NInfer engine launcher for the prebuilt Ada (sm_89) bundle.
#
# Resolves its own directory, so the archive can be extracted anywhere, and puts the
# bundled lib/ first on the loader path. The bundle therefore needs no CUDA Toolkit and
# no compiler on the host, and it wins over an older libstdc++ placed on LD_LIBRARY_PATH
# by an activated conda environment.

set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <model.ninfer> [engine options...]" >&2
  echo "       $0 --help" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/ninfer-serve" "$@"
