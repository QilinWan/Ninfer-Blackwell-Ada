# Ops inventory — capy `include/ninfer/ops/tp2/*` vs ours `include/ninfer/ops/*`

P0.3 产出。capy=`/data/DSH-work/capy-flash`，ours=`/data/DSH-work/df2-split`。
**命名空间一致**：capy tp2 头全部 `namespace ninfer::ops`（capy `tp2/linear_topk.h:11`），`tp2/` 只是布局分离，不是子空间 → 移植头文件放 ours flat 目录即可，无 using 改造。
capy 13 头（430 行 API 面），ours 50 头（flat）。

## A. 两侧已同构（capy tp2 = ours flat，同名同签名）

| 头 | ours 证据（include/ninfer/ops/） | capy 证据（include/ninfer/ops/tp2/） | 差异 |
|---|---|---|---|
| argmax.h | `kArgmaxSplitMaxRowsPerRank=65536` :31、`kArgmaxSplitCandidateRows=8` :34、`argmax_split_pack` :46、`argmax_split_select` :57 | :21/:24/:36/:47 | 无（逐符号一致） |
| attn_input_proj.h | `…_column_parallel_workspace_capacity_bytes` :121、`attn_input_proj_column_parallel` :142/:150 | :34/:55/:63 | capy 多 `attn_input_proj_head_block` :79 → 表 B |
| gdn_gating_proj.h | :112/:116/:124 | :27/:31/:39 | 无 |
| gdn_input_proj.h | column_parallel :268-311（含 conv_snapshot/conv_record） | :40-102 | 无 |
| linear_add.h | `linear_add_row_parallel_workspace_capacity_bytes` :90、`linear_add_row_parallel` :131/:139 | :26/:88 | capy 多 `linear_add_row_parallel_begin` :80（split 层循环 begin/finish 拆分用，P3.3 评估） |
| linear.h | `linear_column_parallel` :185/:192、`linear_row_parallel` :225/:232 | :59/:66/:99/:106 | 无 |
| linear_swiglu.h | `linear_swiglu_column_parallel` :120/:129 | :49/:58 | 无 |

## B. ours 缺失（capy tp2 独有，P2 移植对象）

| capy 头:line | 符号 | 语义 | 任务 |
|---|---|---|---|
| linear_topk.h:29/:32/:39/:51/:63/:72 | `kLinearTopKSplitRows=65536`、`kTopKSplitCandidateRows=2*16*8`、`linear_topk_split_workspace_capacity_bytes`、`linear_topk_split`、`topk_split_pack`、`topk_split_merge` | 词表行拆候选排名；每 rank 16×64bit key base-256 打包 → 1×allreduce_sum → 两侧合并 | **P2.1** |
| sliding_window_attention.h:18/:27 | `sliding_window_attention_head_block` + workspace | 每 rank 16Q/4KV 头块注意力 | **P2.2**（先试通用 op 传本地 head 数） |
| rmsnorm_rope.h:16/:40 | `rmsnorm_rope_head_block`、`rmsnorm_rope_text_shard` | 头块 norm+RoPE / 文本分片 | **P2.2/P3.3** |
| context_kv_materialize.h:11/:20/:25 | `kContextKVMaterializeBlockKVHeads=4`、`context_kv_materialize_head_block` + workspace | drafter context KV 头块物化 | **P2.3** |
| dynamic_grouped_conv.h:23 | `dynamic_grouped_conv_finish_add` | conv finish 在两 rank 相同 sum 上各自施加（draft_split.h:36-37 注释） | **P2.3** |
| hyper_connection.h:29/:68/:84 | `hyper_connection_mix_split`、`hyper_connection_combine_mix_split` | Qwen4Exp MoE hyper-connection | **不移植**（非目标，plan §1） |

ours 对应头现状：`linear_topk.h`/`context_kv_materialize.h`/`sliding_window_attention.h`/`dynamic_grouped_conv.h`/`rmsnorm_rope.h` 均 0 个 split 符号（grep 实测；rmsnorm_rope.h:10 与 dynamic_grouped_conv.h:51 的命中是注释词）。

## C. 实现布局映射

- capy：每 op 子目录 `src/ops/<op>/tp2/*.cu|cpp|inc`（42 个文件，实测 find）+ `src/ops/tp2/sources.cmake`；`src/ops/CMakeLists.txt:16-20` 逐 op `include(<op>/sources.cmake)` 挂载。
- ours：split 实现平铺——`src/ops/launcher/argmax_split.cu`、`src/ops/wrapper/argmax.cpp`、`src/ops/linear/linear.cpp`、`src/ops/wrapper/attn_input_proj.cpp`（grep 实测）；无 `tp2/` 子目录。
- **移植约定**：新实现按 ours 布局放 `src/ops/<op>/`（capy 的 `fp8_*_shard_a8.cu`/`nvfp4_*_a4.cu` 内核族可按制品 dtype 需要挑选），头放 `include/ninfer/ops/` flat；不引入 capy 的 `tp2/sources.cmake` 挂载链。

## D. 数值契约（P2.4 对拍基准）

capy `draft_split.h:83-84`：`linear_topk_split` 的每块行与整 head 行 **逐 bit 相等**，key 携带全局 token id，合并结果 = 整 head `ops::linear_topk`。→ P2.1/P2.4 验收按逐 bit，不是容差。
capy `draft_tp2.inc` 消费点：`dflash2_candidates_split`（:94 签名，P2.1 头 → P3.4 调用）。

## E. 数量小结

13 capy tp2 头：7 已同构 + 1 部分多符号（attn_input_proj、linear_add 记 B）+ 5 全缺（表 B 前 5 行）+ 1 不移植。新增 ops 工作量集中在 **linear_topk_split 族（P2.1，1.5d）** 与 **head-block 三件套（P2.2/P2.3，2d）**。
