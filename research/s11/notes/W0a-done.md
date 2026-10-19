# W0a deps — done（DeepSeek 接手）

产物：`/data/s11/deps-ada`
- libcurl **8.11.0**（cmake，`-DBUILD_CURL_EXE=OFF -DCURL_USE_OPENSSL=OFF`；系统无 openssl 头，
  与 `scripts/build-ada.sh` 的历史配方一致；该前缀只服务本地构建/下载）。
- FFmpeg **6.1.1**（`--enable-shared --disable-static --disable-doc --disable-programs
  --enable-pic --disable-x86asm`）。nasm 缺失、nasm.us 经代理 SSL 失败 → C 回退；
  **功能完整**（libavcodec 60.31.102 / libavformat 60.16.101 / libavutil 58.29.100 / libswscale 7.5.100）。
- 校验：`PKG_CONFIG_PATH=deps-ada/lib/pkgconfig pkg-config --modversion libavcodec` = 60.31.102 ≥ 60；
  cmake configure 输出 `Found libavformat/libavcodec/libavutil/libswscale`，`NINFER_ENABLE_VISION=ON` 生效。

构建：`/data/s11/build-120a-cu13`，全量 996 目标 rc=0（日志 `/data/s11/logs-build.txt`），
`apps/ninfer-serve` 已产出。首次全量用 `JOBS=14 nice -n 12`。

日志：`/data/s11/logs-deps-{curl-configure,ffmpeg-configure,ffmpeg-build}.txt`。

下一步：等 W0b 制品（用户另一会话）→ W1 复现/基线。
