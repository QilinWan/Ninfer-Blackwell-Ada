# W12 门链（单卡，当次分母）

构建：`/data/s11/build-120a-cu13`（全量 996 目标 rc=0；第二次增量 rc=0）。
执行：无 `NINFER_TEST_ARTIFACT`（real 测试按设计 skip 77），`CUDA_VISIBLE_DEVICES=1`，`nice -n 10`。

| 面 | 命令 | 结果 |
|---|---|---|
| 家族子集 | `ctest -j1 --timeout 900 -R 'linear\|fp8\|nvfp4\|bf16'` | **28/28 passed，0 failed**，419.64 s |
| 簇门 | `ctest -j1 --timeout 900 -E 'softmax_attention'` | **174/174（0 failed）**，102 passed / 72 skipped，1055.72 s |
| 资格面 | `ctest -j1 -R '^ninfer_softmax_attention_test$'` | **1/1 passed**，756.95 s（研究记录 757.47 s） |
| 资格面 dtype | `--kv-dtype bf16 / int8 / nvfp4` | 三个都 rc=0：93.39 s / 132.65 s / 157.38 s |

skip 的来源是单卡无 27B 制品（real/双卡套件）与 split 变体，不是失败；tp2 双卡面另见 W11。
日志：`logs/gate-sub.txt`、`logs/gate-full.txt`、`logs/gate-qual.txt`。
