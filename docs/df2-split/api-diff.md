# API diff — capy-flash (`60a4153b`) vs df2-split (`e0a5470c`)

P0.2 产出。capy 路径前缀 `capy:` = `/data/DSH-work/capy-flash`，ours 前缀 `ours:` = `/data/DSH-work/df2-split`。
结论：**核心数据结构两侧高度同源；缺口集中在 `TpExecution` 的 rank1 drafter 成员、`execution/tp2/` 整个 round 子系统与 `ops` 的 top-k split**。适配面小，签名冲突在 `DFlashBatchContext` 两字段。

## 1. DFlashPersistentState — 完全一致

`capy:src/models/qwen3_5/program/storage/draft_context.h` = `ours:.../draft_context.h`（`diff -u` 为空）。P3.1/P3.5 可直接在两侧复用同一类型。

## 2. DFlashBatchContext — 字段集冲突

- capy: `capy:src/models/qwen3_5/program/context.h:99`
- ours: `ours:src/models/qwen3_5/program/execution_context.h:96`（文件不同名）
- ours 多两字段（capy 无）：`std::span<TokenId> host_drafts; CudaCompletionEvent& drafts_ready;`（ours:104-105）
- 其余逐字段一致，含 `peer_frame`/`peer_continuation_hidden_store`（两侧都有，rank1 decode frame 的挂载点已就位）。

**移植含义**：`draft_tp2.inc` 的 round 若直接引用 `state.host_drafts/drafts_ready` 会编译失败；capy 的 `dflash2_share_drafts_tp2`（capy draft_tp2.inc:265）需适配 ours 的 handoff 机制（`ours:program/program_impl.h:477 DFlashDraftHandoff`）。

## 3. TpExecution — capy 多 3 个成员

`capy:execution/tp.h:105` vs `ours:execution/tp.h:97`：

| 成员 | capy | ours | 处理 |
|---|---|---|---|
| `Tensor rope_delta` | 有（rank1 副本，prefill 用） | 无 | P3.6 planner 需镜像；非 split 必需 |
| `const Qwen4Runtime* qwen4` | 有 | 无 | 不移植（MoE 非目标） |
| `DFlashPeerDrafter* dflash` | 有（tp.h:132） | 无 | **P3.1 核心新增** |
| `complete()` | 含 `rope_delta.data` | 不含 | 随新增成员调整 |

其余成员（execution/events/parameters/work/linear_attention/text_cache/text_kv_table_row/ordinary/replay_records/mtp_*/prefill_hidden/staged_draft_collectives）两侧逐行一致。

## 4. execution/tp2/ 子系统 — 整树缺失

ours 无 `src/models/qwen3_5/execution/tp2/`。capy 19 文件：`draft_split.{h,cpp}`、`draft_tp2.inc`(430行)、`attention_split`、`ffn_split`、`gdn_split`、`linear_split.h`、`mtp_split`、`exchange_prefetch`、`split.h`、`workspace_split.h`、`text_{tp2.inc,context_private.inc,context_public.inc}`、`sources.cmake`。

`capy:execution/tp2/draft_split.h` API 面（12 项，全部 ours 无）：

- `struct DFlashPeerDrafter { ExecutionCore execution; DFlashPersistentState& dflash; }`（:24）
- `dflash2_drafter_split(const Parameters&)`（:31）
- `dflash2_draft_round_tp2(DFlashBatchContext&, batch, k, DFlashEnvelopes)`（:41）
- `dflash2_split_mlp_fused(const LinearParameters&)`（:48，NVFP4 gate/up → fused SwiGLU）
- `dflash2_selector_rank(const Parameters& rank0)`（:54）
- `dflash2_share_drafts_tp2(DFlashBatchContext&, batch)`（:60）
- `dflash_append_context_peer(DFlashPeerDrafter&, DFlashDecodeIngress&, batch, draft_window, envelope)`（:65）
- `dflash_peer_batch_sink(...)` / `dflash_peer_prefill_sink(...)`（:72/:76，返回 `DFlashFeatureSink`）
- `dflash2_candidates_split(hidden[2], head[2], broadcast, ids, scores, workspace[2], execution, events, peer_ids, peer_scores)`（:94）
- `dflash2_candidates_split_workspace_bytes(...)`（:102）
- `dflash2_split_candidates(rank0, rank1)`（:109）

Round 入口映射：ours 单卡 `propose_dflash2_batch`（`ours:execution/draft.cpp:270`）、`dflash_decode_batch_body`（:573）、`dflash_decode_batch`/`capture_dflash_decode_batch`（:809/:818）↔ capy `propose_dflash2_batch_tp2`（capy draft_tp2.inc:50）+ `dflash2_draft_round_tp2`（:326）。capy primary 模式也走 tp2.inc（`broadcast=true` 分支，draft_split.h:86-89），这是 §11 “round 重写本身带收益”的结构依据。

## 5. sharding — capy 多 split 规则与两个 env 旋钮

`capy:load/sharding.cpp`（615 行）vs ours（377 行）：

- capy 有：`NINFER_TP_DRAFTER`（:209）、`split_dflash2_drafter()`（:238）、`NINFER_TP_SELECTOR`（:246）、`selector_rank(LoadOptions&)`（:258）、`draft_block_shard()`（:264）、dflash2 组件分派（:464）
- ours 有：`NINFER_TP_DRAFT_HEAD`（:197）；dflash2/* 全走 `PrimaryOnly`（:231）
- 两侧 `ShardAxis` 枚举一致（`{artifact/slices.h}` capy:59 / ours:58：Replicated/Rows/Columns/PrimaryOnly/SingleDevice/HostMapped）——ours **已有 `SingleDevice`**，selector 落位无需扩枚举
- 两侧 `load/dflash2.cpp:24-28` 相同（`extra.selector_rank` 已进加载绑定）；ours `draft.cpp:382,388` 已按 `config.dflash2->selector_rank` 索引——**selector 单卡落位的下游接线两侧已存在**，P1.1/P1.2 只补 sharding 上游。

## 6. Parameters — drafter 结构逐行一致

`capy:execution/parameters.h:103-145` ≡ `ours:.../parameters.h:101-143`（diff 仅注释）：`DynamicConvParameters`/`DraftBlockParameters`/`SelectorParameters`/`DraftParameters`/`ProposalParameters` 字段一致；`ProposalParameters::split()`（按词表行拆，含 `token_ids` 行映射）ours 已有。capy 多出的是 Qwen4*/HyperConnection/Qsa（非目标，不移植）。

## 7. ops：top-k split 全缺

- capy `include/ninfer/ops/tp2/linear_topk.h`：namespace `ninfer::ops`（与通用 ops 同空间）；`kLinearTopKSplitRows=65536`（:29）、`linear_topk_split`（:51）、`topk_split_pack`、`topk_split_merge`
- ours `include/ninfer/ops/` 无任何 `linear_topk_split|topk_split_*`（grep 0 命中）→ P2.1 新增
- capy 其余 tp2 ops 头（`include/ninfer/ops/tp2/`13 个：attn_input_proj/sliding_window_attention/context_kv_materialize/dynamic_grouped_conv/linear/linear_add/linear_swiglu/rmsnorm_rope/argmax/gdn_*_proj/hyper_connection）：同名通用版 ours 都有（flat 目录），差异在 split 变体 → P0.3 清单。

## 8. 风险确认

1. `DFlashBatchContext` 两字段差异（§2）是 P3 唯一已知的签名硬冲突点；适配方向：round 用 ours 的 `DFlashDraftHandoff` 替换 capy 的 share 路径，不改结构体字段。
2. capy `text_tp2.inc`/`text_context_*.inc`（target 侧 rank1 驱动）ours 以 `PeerRuntime`（`ours:program/program_impl.h:385`）+ `execution/tp.cpp` 实现，命名体系不同——P3 接线逐点核对，不按文件搬。
3. ours 的 MTP 双验/embedding-host/512K/YaRN 修复均在 `tp.h`/`parameters.h`/`sharding.cpp` 相同锚点上，不与上述缺口重叠。
