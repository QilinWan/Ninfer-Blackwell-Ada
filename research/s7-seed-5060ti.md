# S7 seed · 5060 Ti 调参数据与 D 桶基准（从参考分支取，2026-10-16 16:5x）

用途：**S7 不必等全序 168**——数据先落表（`relay/`），168 落地时按供体结构写我方行；D2/D4 的预算与目标也据此校准。
来源：`forks/baiyouqing-tp2-5060ti`（2× RTX 5060 Ti 实测 fork，blobless ⇒ 只 `git show`）、`forks/ValerioDolci-ninfer-tp2`（供体主线，`src/core/tp2/device_tuning.h` 在 tip）。

## 1. 调参表的形状（供体主线 `src/core/tp2/device_tuning.h`，我方需照此写 5060 Ti 行）
- header-only，**行按 (compute_capability, sm_count) 匹配，首个命中生效**；末行 = **2× RTX 5070 Ti 实测**（所有设备在表存在之前的默认值）⇒ **5060 Ti 行插在它上面**；加新 GPU 只加行（`docs/maintainer/upstream-merge.md` "Adding a new GPU"）。
- 我方 5060 Ti 的键：`compute_capability = 120`（major*10+minor，sm_120）、`sm_count = 36`、`attention_sm_count = 0`（⇒ 用设备自身 SM 数）。
- 表内字段（S7 要供的四个量）：
  | 字段 | 含义 |
  |---|---|
  | `nvfp4_attn_input_shard_first_tiled256_tokens` | NVFP4 `[7168,5120]` 分片 A4 TMA 路：从哪个 T 起读 256-token scale tile（父问题在 T=1024 切） |
  | `tp2_ordinary_graph_allowance_bytes` | ordinary 每 batch size（一类）的 graph allowance |
  | `tp2_mtp_graph_class_allowance_bytes` | MTP 每拓扑类 × batch 的 allowance（供体 38 自陈"**每类 24 MiB × 2，未实测**" ⇒ 我方必须自己标） |
  | `tp2_dflash2_graph_class_allowance_bytes` | DFlash2 同上 |
- **不进表**（编译期 schedule 选择，全设备共享，改了会动上游选择器）：NVFP4 A16 sliced-K 实例与 MLP-down SIMT MinBlocksPerSm、INT8 attention split 规则本身（它**读表里的 SM 数**）、mailbox 轮询自旋上限 `kPeerSpinLimit`。

## 2. 5060 Ti 实测（baiyouqing，官方 `nvfp4` 制品 @ `--tp 2`）
- 每卡权重 **10.08 GiB**；**MTP 关 = 38 tok/s**；**`--spec mtp --draft-tokens 3 --lm-head-draft` = 76–86 tok/s**。
- **`int8` KV：单槽吃满 262,144 token 原生上限，每卡仍余 842 MiB**；`ninfer-serve` 与 CLI 余量**完全一样**（视觉关闭时不预留媒体/响应缓冲）⇒ **D2 配额判据可直接引此量级**。
- **W4A4 形态：每卡权重少 16%，同 int8 档 decode = 106 tok/s** ⇒ 与我方制品（Merkyor Coder390 W4A4 V3，22.141 GiB ⇒ **11.07 GiB/卡**）同族。

## 3. 对我方目标口径的三条修正（写进 D4 判据前必读）
1. **【2026-10-16 21:5x 更正，原判定作废】76–86 tok/s 是我方**可达**目标**：原判"制品 `proposal/head=q4_g64` ⇒ `--lm-head-draft` 不可用"＝**误推**。实测依据＝① `src/models/qwen3_5/execution/parameters.h:75` "the optimized proposal head is **PrimaryOnly**, and its proposal runs on rank 0"（＋`model.h:43`）⇒ 双卡下草稿头**整块在 rank 0、根本不分裂**，所以"q4_g64 无 TP2 半形状"这条约束（§制品报告第 43 行说的正是**分裂路径**）不适用；② 我方 ops 有该编码的单卡路由：`src/ops/linear/q4/`（gemv/mma/simt/sliced_k）＋ **`src/ops/linear_topk/q4_m64.cu`**（DFlash2 选择器要的 top-k 融合 linear）。⇒ D4 判据恢复为线：**`--spec mtp --draft-tokens 3 --lm-head-draft`** 对标 76–86，W4A4 同 int8 档 106 tok/s 为参照；**唯一被政策拒的是 `--spec dflash2` 不带 `--lm-head-draft`**（`load.cpp:61`、`model_instance.cpp:79`：全文表 head 是词表分裂的，`linear_topk` 无分裂形）。MTP 在双卡**不依赖** lm-head-draft（39 验收＝MTP 头像 Text 层分裂；q8_g32 半形状在 35 已进我方 A16 sliced-K）。权威落点＝台账 §31（证据链）＋ 薄卡制品段。

2. **int8 KV 是 tp2 唯一有实测余量的档**（+bf16）；我方制品 22.141 GiB ⇒ 每卡 11.07 GiB 权重 + 每卡 KV + 激活 + graph allowance + **rank1 全词表 logits 帧** ⇒ D2 首件事 = 复现"单槽 262144 + 余量"在 W4A4/11.07 GiB 下是否成立。
3. 参考分支有 **`k16i8` KV 档（BF16 keys + INT8 values）**（fork-only，commit 5655968/0815877/2782021）：它**不在供体主线**，且 S6 能力门目前只放行 bf16/int8 ⇒ 若 D2 显存不够，`k16i8` 是一个候选出路，但**要过我方 KV 布局门/oracle**（`causal_cache/e8` 的先例），不是白拿。

## 4. 参考分支的其它可迁移项（逐条判定，不照搬）
| 项 | 出处 | 判定 |
|---|---|---|
| **pinned-host peer mailbox**（无 P2P 时经 pinned host 的 peer 传输） | baiyouqing 6403f96 | **相关**：我方 5060 Ti 对**实测无 P2P**（§5 机器：P2P=0、pinned 13.2/13.35 GB/s、跨卡 6.6 GB/s）⇒ S8/D1 的降级路径正好是这条；到 59/111 时按我方 mailbox 面落，不搬其实现 |
| **`runtime: scope plan-validation failures to one request (stop engine self-restart)`** | baiyouqing c50ccc9 | **建议采纳其语义**（tp2 真机期稳健性）：计划校验失败只拒当前请求，不炸整引擎。落点＝我方 `Engine`/`Scheduler` 的失败面（S8/S10 期间做，单独 commit） |
| **vision-aware MTP 输入嵌入 parity with tp1 / vision chunk 在 workspace roots 之前编码** | 7872795 / 572a64c | 到 62（Vision on one rank）时对照；vision 已 ON ⇒ 有真门 |
| **MSVC/Windows 双卡构建** | 839877d | **不做**（我方目标 Linux） |
| `parallelno-dflash2-tp2`（prefill_boost） | 087dca4 | DFlash2 块（43–46）撞车时对照 |
| `wamansou-tp2-1m`（YaRN 1M） | 6a355d5 | **不承诺**：我方制品原生 262144，供体 TP2 无 YaRN 1M |

## 5. 门与验收（S7 的判据不变，来源＝执行计划 §1 S7 行 + 台账 §13 更正 ②）
- 验收点 = **全序 168 `8ef949fe`**（`ops/tp2` 每设备调参表进树）；在此之前任何"launch plan SM 数"判据都是空谈。
- 168 落地后：加 5060 Ti 行（CC 120 / SM 36 / attention_sm_count 0），日志里 **launch plan 的 SM 数 = 36**（P0.5 同判据）；三档 graph allowance **先用保守值（供体 24 MiB/类 ×2）占位，D2 窗口内实测标定**。

## 调参待办（本次撤除的 70-SM 步，36 SM 实测后再定）
- 全序 130 `2a596191`：[7168,5120] attn shard 的 TMA tile 选择在 T=1024 用 128-token（供体 5070 Ti 实测 −6.4%；wave 依据＝70 SM，我方 36 SM 需重测）。我方取我方后残留 hunk 引用 `first_tiled256_tokens(weight.n)`＝该函数不存在 ⇒ 移植时要连函数一起写。
- 全序 142 `91582814`：NVFP4 A16 sliced-K 与 down SIMT 的 70-SM 调参＋家族改名（`Nvfp4SlicedInstance` 被净应用的 `nvfp4_schedule.cuh` 变成非模板别名 ⇒ 我方 13 个调用点不合规）。移植顺序＝先 36 SM 测 tile/波次，再统一改名到 13 点（attn/gdn/linear_add/swiglu/8 shapes）。
- 两处都必须在 D4 的 36 SM 实测窗口里量化，才谈采纳（口径：76–86/106 tok/s 只作参考）。

## 6. 本机实测（2x RTX 5060 Ti 16 GB · sm_120a · 36 SM · PCIe 无 P2P）· 2026-10-18 D 窗口

构建＝`build-120a-cu13`（CUDA 13，vision ON，commit `d01dbcf7`＋`d4050f1e`＋`95884c20`）；KV 全部 int8 除注明；证据＝`/tank/ninfer-window/{d1,d2,d3,d4}`、`relay/gate-audit51-*`。

### 6.1 两件制品

| 件 | 字节 | 每卡权重（serve, tp2） | tp2 可用性 |
|---|---|---|---|
| NF4 `qwen3.8-27b-lynn-nf4` | 19,345,900,804（sha `faf551e2…c53268`，队内已核） | —（加载即失败） | **不可**：`text/output_head`＝`q6_g64 [248320,5120]`，q6 无半词表 `[124160,5120]` 形状 ⇒ `q6 linear: unsupported shape`（需新形状＋oracle 资格面） |
| W4A4 `…W4A4-MTP-DFlash2` | 23,773,675,264（本机 sha256 `3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808`） | r0 **10.7 GiB**（sharded 9.27/replicated 1.19/local 282.5 MiB）、r1 **10.5 GiB** | 可用（D1–D4 全部实测落此件） |

打分相（perplexity）两 rank 对称：各 **10.1 GiB**（sharded 8.90＋replicated 1.19）；两卡占用差 **728 MiB**＝打分相 arena 不对称（rank0 全词表 gather＋两份 head 半）。

### 6.2 容量（W4A4 · int8 KV · vision ON · MTP3 · 4 state slots）

| 容量 | 结果 |
|---|---|
| 32,768 | pages 512/512，runtime 1.74 GiB，free **2.87 GiB** |
| **196,608** | pages 3,072/3,072，runtime 4.48 GiB，free **132.7 MiB** ⇒ **本机 tp2 可交付上限** |
| 262,144（原生）· **int8** KV | 起不来：请求 5,984,344,833 B，weights 之后只有 4,947,902,464 B |
| **262,144（原生）· nvfp4 KV + vision + MTP3** | **起得来**：pages 4,096/4,096，runtime 3.58 GiB，free **1.03 GiB** |
| **262,144（原生）· nvfp4 KV + 无 vision + MTP5** | **起得来**：pages 4,096/4,096，runtime 2.87 GiB，free **2.01 GiB** |

⇒ **原生 262,144（256K）在主口径（NVFP4 权重 + nvfp4 KV）下可达**；此前"262,144 起不来"只对 int8 KV 成立（KV 体积约为 nvfp4 的 2 倍）。

### 6.3 graph allowance（实测 observed / 当前常量）

| 类 | rank0 / rank1 observed | allowance | 与占位（8/8/11 MiB/类） |
|---|---|---|---|
| ordinary（无 spec） | 0.0 / 0.0 MiB | 8.0 MiB | 成立（3x observed 落 8 MiB 地板） |
| MTP3（K=3） | 2.0 / 2.0 MiB | 8.0 MiB | 成立（3x2=6 → 地板 8） |
| DFlash2（K=7） | 22.0 / 12.0 MiB（6 拓扑类合计） | 66.0 MiB＝6x11 | 成立（每类 3x3.67→11） |

### 6.4 速度（60 题 gate 集 / T=0 / 128 tok；与供体口径同形）

| 面 | decode | accept 率 | ms/round | 备注 |
|---|---|---|---|---|
| ordinary（无 spec，48 tok 样本） | **39.0 tok/s** | — | — | vision OFF 单测；D4 另测 |
| MTP3＋`--lm-head-draft`（int8，196,608 ctx） | **51.9 tok/s** | **72.43 %** | **60.04 ms** | 60/60 逐字可复现（复跑 −0.09 %） |
| DFlash2 K=7＋`--lm-head-draft`（int8，32k ctx） | **133.9 tok/s** | **59.20 %** | **37.82 ms** | 10/10 逐字可复现（复跑 +0.09 %） |

供体口径（76–86 MTP3 / 106 W4A4-int8）**只作参考、未在本机复现**：本机 MTP3 实测 51.9 tok/s（accept 已 72 %，慢在每轮固定成本：无 P2P 的 host-staged all-reduce＋rank0 独担 drafter/proposal/vision），DFlash2 反而到 133.9（K=7、轮成本摊薄）。

### 6.5 KV 档代价（perplexity，quick 4k/2048 stride，261,167 tok）

| KV | PPL | mean_nll | score rate |
|---|---|---|---|
| int8 | **4.259926** | 1.449252 | 1580.7 tok/s |
| nvfp4 | **4.267108** | 1.450936 | 1580.7 tok/s |

⇒ nvfp4 KV 的 PPL 代价 **+0.007182（+0.169 %）**，打分速率**无差**（1580.7）。

### 6.6 未测/登记

- **shard TMA 界 1024**：本树**没有 shard bench**（`bench/ops/attn_input_proj_bench.cu` 固定造 14336 行父形，从不读 `nvfp4_attn_input_shard_first_tiled256_tokens`）⇒ 保留 1024 占位；要测必须先补一个 shard 几何 bench（记为 D4 债）。
- **NVFP4 12/2 row 组六 / key shift 7-8**：结构面（`src/ops/softmax_attention/dense/causal_cache/nvfp4/plan.cpp:24-38`，按设备 SM 数算），无字段可写回；本机 36 SM 的 splits 需 causal bench 才可见（未跑）。
- **GSM8K 500**：见 §6.7。

### 6.7 GSM8K 500（thinking off · 8-shot · max 512 · T=0）

| KV | canonical | numeric | request_errors | unmatched_gold | subset_digest |
|---|---|---|---|---|---|
| int8 | **484/500＝96.8 %** | 484/500 | 0 | 0 | `245dfab7c79d880c…` |
| **nvfp4**（主口径） | **482/500＝96.4 %** | 482/500 | 0 | 0 | 同上（可比） |

两跑同 digest ⇒ 分母可比；本表是 **tp2 分子**。27B 的 `--tp 1` 分母在本地结构性不可得（两件制品均 >16 GB 单卡），口径按 §3/§5 旧裁定保留。

### 6.8 NVFP4 面（主口径）逐字与扫参（nvfp4 KV）

- 逐字：`--record`→`compare` **PASS** — greedy **60/60 逐字一致**（52.17 tok/s·accept 72.18 %·ms/round 59.853）、dflash2 **10/10 逐字一致**（140.38 tok/s·accept 62.09 %·ms/round 37.531）；基线 `/tank/ninfer-window/reference-nvfp4/`。
- 扫参（60 题，T=0，nvfp4 KV）：

| draft-tokens K | decode | accept | ms/round |
|---|---|---|---|
| 2 | 45.10 tok/s | 80.69 % | 57.376 |
| 3 | 52.39 tok/s | 72.18 % | 59.594 |
| 4 | 54.85 tok/s | 64.76 % | 64.252 |
| **5（最快）** | **56.96 tok/s** | 57.09 % | 65.767 |

`--spec mtp` 的 `--draft-tokens` **硬上限＝5**（6/7 被 usage 拒绝）；DFlash2 用 K=7。
- nvfp4 KV vs int8（同 K/同题集）：greedy +0.9 %、dflash2 +4.6 %、DFlash2 accept 59.20→62.09 %、PPL +0.169 % ⇒ **nvfp4 KV 在本机不付速度代价**。

### 6.9 上下文 × 并发矩阵（nvfp4 KV，W4A4，tp2，state slots 8）

| spec | ctx | C | 结果 |
|---|---|---|---|
| MTP3 | 262,144 | 1 | ✅ vision ON：runtime 3.58 GiB，free **1.03 GiB**；无 vision：2.87 GiB，free **2.01 GiB** |
| MTP3 | 262,144 | 2 | ✅ runtime 3.24 GiB，free 1.64 GiB |
| MTP3 | 262,144 | 4 | ✅ runtime 3.42 GiB，free 1.47 GiB |
| MTP3 | 262,144 | 8 | ✅ runtime 3.76 GiB，free 1.12 GiB |
| DFlash2 K7 | 262,144 | 1 | ❌ 需 3,740,238,336 B／只有 3,008,036,864 B（drafter 抬高 rank0 权重） |
| DFlash2 K7 | 196,608 | 1 | ❌ 需 3,136,254,464 B |
| DFlash2 K7 | 180,224 | 1 | ✅ free **66.7 MiB**（极限） |
| DFlash2 K7 | 163,840 | 1 | ✅ free 210.7 MiB |
| DFlash2 K7 | 163,840 | 4 | ❌ 需 3,471,893,248 B |
| DFlash2 K7 | 32,768 | 4 | ✅ runtime 2.07 GiB，free 916.7 MiB |
| DFlash2 K7 | 32,768 | 8 | ❌ 需 3,021,768,448 B（差 13 MiB） |

- 每加一条 lane 的运行时预留 ≈ **+0.21 GiB**（MTP 262k：C1 3.58 → C2 3.24(无 vision 基准) → C8 3.76）。
- **并发实测**（MTP3，262,144，C=8，8 条并发、每条 64 tok）：wall 2.35 s，总 512 tok ⇒ **aggregate 217.5 tok/s**，每流 27–35 tok/s（单流 52 t/s）。
- 结论：**多用户＝MTP + C≤8 + 256K**；**单用户极速＝DFlash2 + ≤163,840**（DFlash2 长上下文只能 C=1，32k 可 C=4）。
