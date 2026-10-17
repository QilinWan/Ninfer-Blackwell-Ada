#!/usr/bin/env bash
# build-ada.sh — 本树（Blackwell sm_120a，单架构）的可复现构建入口。
# 文件名是历史名（本仓库曾支持 Ada sm_89；该架构已于 2026-10-18 移除），只构建 120a。
#
# 为什么要有这个脚本：本机环境有三个静默坑，踩过一次就要花半天再踩回来 ——
#   1) PATH 里 /usr/bin/cmake 是 3.22.1，而 CMakeLists 要求 >=3.28，真正的 4.4.3 在 ~/.local/bin；
#   2) libcurl 要求 >=7.85，Ubuntu 22.04 只有 7.81，需本地私有 prefix（同级 ../deps-ada）；
#   3) FFmpeg 需 >=60（6.x），本机没有 → 文本引擎用 -DNINFER_ENABLE_VISION=OFF，
#      该开关是我们自己加的（src/media/decode/stub.cpp 提供空实现）。
# 目录名固定为 build-<arch>-<cuda>，绝不复用旧的 build120/（那里的 CMakeCache 写死了
# CMAKE_CUDA_ARCHITECTURES=120a 与已删除路径的 CMAKE_HOME_DIRECTORY，会静默覆盖命令行）。
#
# 用法：
#   scripts/build-ada.sh               # Blackwell，文本引擎
#   scripts/build-ada.sh --vision      # 需要 FFmpeg>=6 的完整功能（本机在 ../deps-ada）
#   scripts/build-ada.sh --tests       # 附带 tests/bench（BUILD_TESTING/NINFER_BUILD_BENCHMARKS）
#   scripts/build-ada.sh --clean       # 先删构建目录
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ARCH=120a
case "${1:-}" in
  120a) shift ;;
  ''|--*) : ;;
  *) echo "本树只构建 sm_120a；忽略/拒绝架构参数 '$1'" >&2; exit 2 ;;
esac
VISION=OFF; TESTS=OFF; CLEAN=0
for a in "$@"; do
  case "$a" in
    --vision) VISION=ON ;;
    --tests)  TESTS=ON ;;
    --clean)  CLEAN=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
  esac
done

# --- 工具链 ---
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
CMAKE="$(command -v cmake)"; NINJA="$(command -v ninja || true)"
case "$CMAKE" in */home/*) : ;; *) echo "⚠ cmake 不是用户态新版：$CMAKE（$($CMAKE --version|head -1)）；期望 ~/.local/bin/cmake 4.x" >&2; exit 1 ;; esac
CUDA_V="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"

# --- 依赖 prefix（curl>=7.85）---
DEPS="${NINFER_DEPS_PREFIX:-$PWD/../deps-ada}"
if [ -x "$DEPS/bin/curl-config" ]; then
  export PKG_CONFIG_PATH="$DEPS/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  export LD_LIBRARY_PATH="$DEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CURLV="$("$DEPS/bin/curl-config" --version | awk '{print $2}')"
else
  CURLV="$(pkg-config --modversion libcurl 2>/dev/null || echo none)"
  echo "🔴 没有可用的 libcurl 开发包（找到 $CURLV，需 >=7.85）。"
  echo "   建一个私有 prefix（无需 root、无需 GPU）："
  echo "     PREFIX=$DEPS; mkdir -p \$PREFIX/src && cd \$PREFIX/src \\"
  echo "     && curl -LJO https://github.com/curl/curl/releases/download/curl-8_11_0/curl-8.11.0.tar.xz \\"
  echo "     && tar xf curl-8.11.0.tar.xz && cd curl-8.11.0 && mkdir b && cd b \\"
  echo "     && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=\$PREFIX -DBUILD_SHARED_LIBS=ON \\"
  echo "          -DBUILD_CURL_EXE=OFF -DCURL_USE_OPENSSL=OFF -DCURL_USE_LIBPSL=OFF -DBUILD_TESTING=OFF \\"
  echo "     && cmake --build . -j && cmake --install ."
  exit 1
fi

BUILD="build-${ARCH}-cu${CUDA_V%.*}"
[ "$CLEAN" = 1 ] && rm -rf "$BUILD"
GEN=(); [ -n "$NINJA" ] && GEN=(-G Ninja)
echo "== configure: arch=sm_$ARCH cuda=$CUDA_V vision=$VISION tests=$TESTS -> $BUILD"
"$CMAKE" -S . -B "$BUILD" "${GEN[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DNINFER_ENABLE_VISION="$VISION" \
  -DBUILD_TESTING="$TESTS" \
  -DNINFER_BUILD_BENCHMARKS="$TESTS" \
  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON \
  -DCMAKE_PREFIX_PATH="$DEPS"

J="${JOBS:-$(nproc)}"
echo "== build -j$J"
"$CMAKE" --build "$BUILD" -j"$J"
echo "== OK  产物：$BUILD/apps/  $BUILD/src/"
file "$BUILD/apps/ninfer-serve" 2>/dev/null | cut -c1-100 || true
