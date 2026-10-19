# W11 二卡套件（ctest -R tp2，本次分母 10）

命令：`CUDA_VISIBLE_DEVICES=0,1 NINFER_TEST_ARTIFACT=<tank artifact> ctest -R tp2 -j1 --timeout 900`
结果：**100% tests passed, 0 failed out of 10**，总 304.69 s（`logs/ctest-tp2.txt`）。

通过：sharded_materialization_tp2、text_context_tp2、text_context_tp2_real、engine_tp2_real、
engine_mtp_tp2_real、engine_mtp_tp2_optimized_real、**engine_vision_tp2_real**（parity，含多图/
宽图/确定性）、**engine_vision_tp2_mtp_real**。

跳过（77，非失败）：engine_dflash2_tp2_real、engine_vision_tp2_dflash2_real ——
正确原因：制品不含 DFlash2 组件（计划书 §12 明确该制品只有 text+vision+MTP）。

单测显式的双卡分母：tp2 相关 10/10 计入，2 skip 由制品组件决定；不把 skip 当证据。
