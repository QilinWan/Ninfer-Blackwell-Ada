# W13 交付：分支 / tag / 文档

- 分支：`s11-multimodal-tp2`（基于 `blackwell-only` @ b55da868）
  提交 `e6977697 feat(qwen3_5): run multimodal prefill at tensor-parallel width 2`
- tag：`s11-multimodal-tp2-green`（门链绿后才打）
- 远端：`origin` = QilinWan/Ninfer-Blackwell-Ada，分支与 tag 均已推送成功。
- 文档：
  - 新增 `docs/maintainer/s11-notes.md`（机制、实测数字、限制、既有缺陷），并登记进 `docs/README.md`；
  - `docs/cli.md` Two GPUs 段、`docs/maintainer/tensor-parallel.md` §8 改为"已支持（handoff）"；
  - `include/ninfer/types.h` / `apps/cli/options.cpp` / `src/serve/serve_options.cpp` /
    `model_instance.cpp` / `startup.cpp` 的拒绝文案与注释同步；
  - 测试：`test_engine_vision_tp2_real.cpp` 改为真跑（单图/双图/非方形/贪心确定性/MTP 图像），
    两处 DFlash2 套件在制品缺组件时 skip 77。
- 证据：`/data/s11/logs/`（gate-sub/full/qual、ctest-tp2-final、vision-*.txt、serve-*.log、
  ids-base/new.json）与 `/data/s11/notes/W*.md`；基线 worktree `/data/s11/tree-base`（b55da868）
  与 `build-base` 保留。
