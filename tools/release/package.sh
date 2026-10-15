#!/usr/bin/env bash
# Packages a release tarball from a Release build tree: package.sh <build-dir> <version> [out-dir]
# Contents: bin/{ninfer,ninfer-serve,ninfer-perplexity} (stripped), lib/libcudart.so.13 (the CUDA
# runtime the binaries link, redistributable), serve-tp2.sh, README.md, COMMIT, SHA256SUMS.
# The tarball is named after the target: linux x86_64, sm_120a, the distribution it was built on.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <build-dir> <version> [out-dir]" >&2; exit 2; }
build=$1; version=$2; out=${3:-dist}
here=$(cd "$(dirname "$0")" && pwd)
distro=$(. /etc/os-release && echo "${ID}${VERSION_ID}")
name="ninfer-tp2-${version}-linux-x86_64-sm120a-${distro}"
stage="$out/$name"
rm -rf "$stage"; mkdir -p "$stage/bin" "$stage/lib"
for app in ninfer ninfer-serve ninfer-perplexity; do
  cp "$build/apps/$app" "$stage/bin/$app"
  strip --strip-unneeded "$stage/bin/$app"
done
cudart=$(ldd "$build/apps/ninfer-serve" | awk '/libcudart\.so/ {print $3}')
[ -n "$cudart" ] || { echo "libcudart not found in ldd output" >&2; exit 1; }
cp -L "$cudart" "$stage/lib/$(basename "$cudart")"
cp "$here/serve-tp2.sh" "$stage/serve-tp2.sh"
cp "$here/README-release.md" "$stage/README.md"
{ echo "version $version"; echo "commit $(git -C "$here" rev-parse HEAD)"; echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ) on $distro"; } > "$stage/COMMIT"
( cd "$stage" && sha256sum bin/* lib/* serve-tp2.sh README.md COMMIT > SHA256SUMS )
tar -C "$out" -czf "$out/$name.tar.gz" "$name"
( cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256" )
echo "$out/$name.tar.gz"
