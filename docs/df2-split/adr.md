# ADR：DFlash2 TP2 split 移植策略 — adapt，不 vendor，不 merge

P0.4 产出。依据：plan §0/§1/§3、P0.2 `api-diff.md`、P0.3 `ops-inventory.md`、交接 `README.md` 规则 2/5。
状态：已采纳（待主代理复核）。日期：2026-10-22。基线：`df2-split`@`e0a5470c`；参考：capy `60a4153b`。

## 决策

按 **API 面适配移植**（adapt）capy 的 TP2 split 子系统：符号级对照 → 在我们的文件布局与类型系统内实现等价物。不 vendor capy 文件原样入库，不整仓 merge `capy/main`（+22,508 行/96 文件）。

## 证据（为什么 merge/vendor 不成立）

1. 签名硬冲突：`DFlashBatchContext` 我们多 `host_drafts`/`drafts_ready`（`program/execution_context.h:104-105`），capy 无（`program/context.h:99`）；capy `dflash2_share_drafts_tp2` 直接引用其 draft 共享路径，搬入即编译失败（P0.2 §2）。
2. 结构体命名体系不同：capy `program/context.h` + 19 文件 `execution/tp2/`；我们 `program/execution_context.h` + `PeerRuntime`（`program_impl.h:385`）+ 单文件 `execution/draft.cpp`。
3. 布局约定不同：capy ops 实现 42 文件在 `src/ops/<op>/tp2/` + `tp2/sources.cmake`；我们平铺（`src/ops/launcher/argmax_split.cu` 等，P0.3 §C）。namespace 两侧同为 `ninfer::ops`，头文件按我们 flat 目录落位。

## 保留项与移植面重叠（改这些锚点必须回归检查）

| 保留项 | 锚点（实测） | 与移植重叠 |
|---|---|---|
| MTP 双验修复 | `0d820ed4` → `program/speculative/mtp.cpp:9`（每 round 一次 target 验证） | 无直接重叠（MTP 后端不动）；P3.4 候选共享改动后须复验 |
| 多模态 MTP bridge 修复 | `e0a5470c` → `execution_context.h`、`prefill.cpp`、`transactions/replay.cpp`、`test_engine_vision_tp2_real.cpp` | **P3.1 改 `execution_context.h`、P3.2 改 `prefill.cpp`** — 同一文件，改动必须保持 rank1 retained hidden 语义 |
| `--embedding-host` | `7af63f2e` → `src/models/load_options.h` | **P1.1 加旋钮就在 `load_options.h`** — 同文件 |
| 512K/YaRN | `runtime/contract/yarn.h`、`program/planning/startup.cpp`、`parameters.h`、`ops/wrapper/rope.cpp` | **P3.6 改 `planning/startup.cpp`**；capy 无 YaRN（plan §5.3），rank1 drafter 的 `rope_delta`/位置偏移必须走我们的 YaRN 系数，不能抄 capy 的 `max-context` 限制 |
| Vision/multimodal | `--vision` rank0 编码→rank1 交接（`--help` 实测在 usage 内） | selector 落 rank1 时与 vision 的 PrimaryOnly 交互按 capy `sharding.h:44` 注释（Vision 上 selector 回 rank0） |

## 范围裁定

- 移植：sharding split 规则 + 2 个 env 旋钮（P1）、`execution/tp2/` round 子系统 + peer drafter + feature sink（P3）、top-k split ops 族 + head-block 三件套（P2）、planner/graph/StateImage 接线（P3.5-3.7）、测试骨架（P1.4/P2.4）。
- 不移植：capy Qwen4*/MoE/PLE/HyperConnection/CPU expert offload/elastic KV/v0.6 优化（plan §1 非目标；`ops-inventory.md` §B 表末行；`parameters.h` 的 Qwen4 段）。
- 默认不变：`NINFER_TP_DRAFTER=primary` 保持现状默认，split 为 opt-in（plan §1）；运行期回退 = 设回 env，不改二进制（plan §6）。
- Selector 语义以 capy 代码为准：**默认 rank1**，`NINFER_TP_SELECTOR=primary` 才到 rank0（README 规则 5；参考文档勘误见 plan §3）。

## 数值与验证契约

- 逐 bit 判据：split 行块输出 = 整 head `ops::linear_topk`（capy `draft_split.h:83-84`），P2.4 harness 按逐 bit，非容差。
- 门禁沿用 plan §6 G0–G7；改动 token 产出路径（kernels/round/共享）按本仓 AGENTS.md 需 GSM8K/MMLU-Pro 级质量检查（主代理在 P4 执行）。
- 生产 8081 二进制与 `model.sh` 全程不动；commit 仅在主代理要求时做，Conventional Commit 小写主题（本仓 AGENTS.md）。

## 后果

正面：改动面收敛在 ~15 个已知锚点文件；保留项语义可逐 commit 回归；split 与 primary 共用同一 round 重写路径（capy primary 也走 `draft_tp2.inc`，plan §11），移植 P3 后 primary 模式有望同收益。
代价：`draft_tp2.inc` 430 行需逐点适配 `DFlashBatchContext`/`PeerRuntime` 命名体系，不能整文件照搬；API 对不上时按交接规则标 `[!]` 交回主代理。
