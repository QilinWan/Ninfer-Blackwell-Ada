# NInfer（`blackwell-only` 分支说明）

这个分支只构建 **Blackwell `sm_120a`**。此前本仓库维护的 Ada（`sm_89`）构建、兼容层
（`cmake/Sm89Compat.cmake`、`src/ops/sm89/`）、工具（`tools/verify_sm89.sh`、
`scripts/check-sm89-exclusions.py`）、镜像与文档已在 2026-10-18 从本分支移除。

- 引擎与双卡 `--tp 2` 的研究记录、实测数据、失败原因与复现方法：见
  [`research/README.md`](research/README.md)、[`research/replay-log-20261016.md`](research/replay-log-20261016.md)、
  [`research/s7-seed-5060ti.md`](research/s7-seed-5060ti.md)。
- Ada 时代的完整树与提交历史保留在 `tp2/raw` 分支（以及 GitHub 上已发布的 `engine-sm89-*` 归档）。
- 英文说明：见 [README.md](README.md)。

构建：`cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON`，
CUDA >= 13.0，FFmpeg >= 6 与 libcurl >= 7.85。
