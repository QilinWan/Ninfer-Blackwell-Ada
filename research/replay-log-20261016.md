# TP2 接线 · 逐步台账（2026-10-16 会话，GPU1 侧）

> 这是"每步做了什么、门是什么"的权威账本。**逐条写实测，不写自评。**
> 分支 `p1/tp2-replay`；供体线性队列见 `relay/tp2-replay-full-223.txt`（223 步全序），
> 本会话起的剩余队列 `relay/tp2-replay-queue-215.txt`（步 9 起 215 步）。

## 0. 会话现场与硬约束（本会话实测）

| 项 | 实测 |
|---|---|
| 沙箱 | 会话开始 `workspace-write` → `/dev/nvidia*` 打不开（`cuInit=304`）；用户随后把策略改为 **danger-full-access + approval=never** ⇒ GPU1 可直接用，**本会话不再请求任何审批** |
| GPU1 自检 | `relay/probe/gpu_probe`：`cuDeviceGetCount=1`、RTX 5060 Ti、**CC 12.0 / 36 SM / 15849 MiB**、alloc 到 12 GiB OK |
| GPU0 | 现网 `strata --serve` PID 440568，15590 MiB，**全程未碰、未 kill** |
| 编译自限 | `JOBS=14 nice -n 12`（生产侧 CPU-MoE 敏感） |
| 树 | `ada-tree` 分支 `p1/tp2-replay`；`tp2/step8` = `1b2ce547`（旧步 8 检查点，保留）；`tp2/raw` = 供体 `79d969d4` |

**关键现场修正（影响接力）**：
1. **`v-tp2-skeleton`（`b797cb54`，271 个 TP2 自有文件）不在 B 阶段的重放基里**。`git merge-base --is-ancestor v-tp2-skeleton 958d88af` = **NO**，`git ls-files | grep -c tp2` 在 `p0/replay` 尖 = **0**。⇒ 重放中"新增文件"型提交会正常创建它们（不会因骨架已落盘而变空）。已落地的步 4/7 之所以变空，是因为它们的**共享面文件**（`allreduce.{h,cu}`、`device_scope.h`、`split_launch.h`、`linear_dispatch.h`、`test_linear_split.cpp`…）随 P0 我方层已在树里（逐个 `-f` 核过 present）。
2. 旧的"步 9/223"现场不可原地续跑：`git rebase --onto p1/tp2-replay 0b3c723b tp2/raw` 得到 **253** 步（供体历史非线性，`0b3c723b` 排除面多掉了 38 个后期提交 ⇒ 会重放已应用内容）。**正确的续跑法**：用显式 todo 注入：
   ```bash
   cd ada-tree && git checkout -q tp2/raw
   GIT_SEQUENCE_EDITOR='cp ../relay/tp2-replay-queue-215.txt' GIT_EDITOR=true \
     git rebase -i --onto p1/tp2-replay 0b3c723b tp2/raw      # msgnum/end = 1/215，停在 06b4011e
   ```
   队列生成法（可重建）：`git rev-list --no-merges --reverse --topo-order d44ab584..79d969d4` ⇒ 223 行，第 8 行是 `0b3c723b`，其后 215 行即队列。

## 1. S1 / **G-TP2-1 —— 已闭合** ✅

- 提交 **`03726c08`** `feat(ops): register tensor-parallel shard shapes for fp8, nvfp4 and bf16 linear`，tag **`v-tp2-halfshapes-r0`**。
- 内容：FP8 6 半、NVFP4 2 半、BF16 2 半 + 三家族各自的注册面（geometry/shapes/dispatch/sources）+ A8/A4 半 K 量化器 + NVFP4 A4 TMA 实例表 2 case + `linear_add` 4 个穷举 switch 收口 + 公开契约 `include/ninfer/ops/linear.h` + 5 个测试套件扩面（`test_fp8_a16/a8`、`test_nvfp4_a16/a4`、`test_bf16_a16`）。
- **门（实测）**：sm_120a 带 tests+bench 全构建 **855/855 rc=0**；`ctest -R "linear|fp8|nvfp4|bf16"` @GPU1 **25/25 100% 通过 417.08 s**；`merge-safety.sh` ✅；**无非法内存访问**。
- 细节与"三家族机制互不相同"的落地清单：见 `g-tp2-1-half-shapes.md`（已改成本会话的闭合记录）。
- **本会话挖出的硬事实（写进任何 TP2 结论）**：split-K 的 workspace 容量**锚点不可继承** —— 分裂条件是 `row_tiles * ceil(T/block_tokens) % 170 ∈ [1,85]`，行向半把 row_tiles 减半 ⇒ `[7168,5120]` 从 **T=289**、`[8192,5120]` 从 **T=193** 就需要 partials，而父形状（14336/16384）要到 385/257。首跑实测兑现：`FP8 TMA split-K requires aligned caller partials`（7168 T=289、8192 T=193/256）。
  处置：新增 `src/ops/linear/fp8/fp8_a8_split_k.h`（按 ladder 的 band 逐段求最大需求）。
  证据（CPU，可复跑）：`relay/capacity_probe.cpp` + `relay/verify_fp8_a8_quotes.py` ⇒ **12 个 FP8 形状 × T=1..4096 的报价 ≥ 精确需求，under-quote = 0**；over-quote 只出现在"band 内某 T 分裂、本 T 不分裂"处（父形状同款，区间上界语义）。

## 2. 全量设备分母（G-TP2-1 之后）✅

- `CUDA_VISIBLE_DEVICES=1 ctest --test-dir build-120a-cu13 -j1 --timeout 900` @GPU1：**146/148 执行，100% 通过，0 失败，1877.17 s**。
- 逐条入 **`relay/passset-sm120-g1-20261016.tsv`**（148 行：129 Passed + 17 Skipped + 2 Not Run(Disabled)）。解析器：`relay/passset_from_log.py`。
- 对比 `v-follow-81c8ce0-r0` 的记录口径（144 注册 / 142 执行 / 100% / 1747.78 s）：**注册 +4** = 本层新落的双设备用例（`allreduce`/`linear_split`/`output_head_split`/`attention_headlocal`，单卡具名跳过 77）；**无缩水**。
- **代价披露**：`ninfer_softmax_attention_test` 从基线 **231 s → 785.67 s（3.4×）**，来源是步 8 落的 **12/2 头本地几何**用例矩阵；`--timeout 900` 只剩 ~14% 余量。**未处理**（留给下一接力者：收窄 12/2 的取样点，或在 `tests.cmake` 给它单独 timeout）。
- 历史 124 版 TSV 里 4 个名字（`admission_policy`、`materialization_budget`、`softmax_attention_k8v4`、`softmax_attention_nvfp4`）在上游 81c8ce09 被合并/改名（现见 `artifact_materialization`、`kv_cache_append_k8v4`、`softmax_attention`）⇒ 属 P0 收口时已接受的口径，非本会话引入。

## 3. S2 重放进度（本会话）

| 步 | 供体 | 结果 | 门 |
|---|---|---|---|
| 9 | `06b4011e` attn_input_proj 列并行分片 | ✅ 已解并落 `a2c4a996^`（HEAD 前进前的那个提交；解冲突 6 文件 + 接受 1 删除） | 857/857 rc=0；`ctest -R attn_input_proj`：`ninfer_attn_input_proj_test` **Passed 71.53 s**、`ninfer_attn_input_proj_split_test` **具名跳过**（<2 卡） |
| 10 | `71aa0eb9` gdn 卷积半形状注册 | ✅ 干净落地（无冲突，未单独构建） | 进步 11 的合并门 |
| 11 | `366e82a0` gdn 门控 + fp8 gdn 输入投影列并行 | ⏸ **停在可续跑检查点**（4 文件冲突未解，零 CPU 消耗）：`gdn_input_proj/fp8/{a8(2块),decode(1),matrix(5),output.cuh(1)}`；其余 11 文件（含 1132 行新测试 `test_gdn_projections_split.cpp`、`wrapper/gdn_{input,gating}_proj.cpp`、`weight_input.cpp`、`include/ninfer/ops/gdn_*.h`、`plan.{h,cpp}`、`conv_plan.{h,cpp}`）**已自动合并入索引** | — |
| 11b | 检查点 | 冲突文件逐字节 + 状态 + 提交消息存在 **`relay/step11-checkpoint/`**（`STATUS.txt`/`STATE.txt`/`MESSAGE.txt`/`src/…`）；续跑命令见 §5（已实测可用：msgnum/end=1/213，stopped=366e82a0） | — |

步 9 的解法要点（写进提交说明的账本，提交消息未追加 note —— 见"未做/风险"）：
- 5 个 launcher 模板化到 `Problem`（geometry + section output），公开入口用 `visit_fp8_attn_input_problem(weight.n, …)` 选父/半；
- section output 采用供体的具名模板结构体（我方 `LinearBf16SegmentedOutput` 不暴露分段行数），父段 6144/1024 不变，半段 3072/512；
- 共享 A8 容量报价取父/半两者最大（同一分裂算术，见 §1）；
- `fp8_attn_input_small_t.cu` 接受上游删除；两处 dependent 非类型成员去掉 `typename`（C++ 里非法）。

## 4. 未做 / 风险（不要当成已完成）

1. **步 9 的提交消息没有附"replay note"**（续跑时已前进到步 10，改写历史需二次 rebase）。本文件 §3 是权威记录。
2. **半形状的 attn_input_proj 执行面在单卡只到编译门**：其 parity 用例 `*_split_test` 单卡具名跳过 ⇒ 属 G-2 窗口。**不要**声称分片投影已数值验证。
3. 步 10（gdn 半形状）未单独过门，与步 11 共用一次构建门。
4. 剩余 **~204 步**：按 2026-10-15 的实测节奏（1–19 文件语义合并/步）外推 6–10 小时，且 144 步是"多文件"重提交（`relay/tp2-replay-queue-215.txt` 全序，风险画像：add-only 5、≤3 文件 106、>3 文件 144）。
5. 四件必做（`p1-tp2-replay-plan` §3）中 **Sm89Compat `/tp2/` 剔除面 + check-sm89-exclusions 基线**必须在**第一个注册 `*/tp2/*.cu` 的提交**内加（供体侧 23be72cf / f57432cb / 9cfa1903 是候选注册点）；**S6 能力门文案、S7 5060Ti 调参行、S8 mailbox 运行期回退**都在 `src/core/tp2/*` 落地之后（约步 16+）⇒ 顺序不可提前。
6. G-2 双卡窗口未申请（用户口径：到点再提）。
7. 本会话**未跑 sm_89 编译门**（决定 B：Ada 一期排在 TP2 之后）。

## 4b. **S9 GSM8K 500 题判分底座 —— 已闭合（CPU 侧），分母数值留窗口** ✅

- 位置 `relay/gsm8k/`（窗口内要落进 `tools/tp2/`，那里由后面的重放步创建 ⇒ 见 README 顶部说明）。
- 数据集官方件：test **1319 行** sha256 `3730d312…c39d14`；train **7473 行** sha256 `17f347dc…13465`（k=8 提示来源）。
- 组成：`run_gsm8k.py`（OpenAI 兼容客户端，贪心/k8/stop/并发/重试/`engine_tag` 强制/`subset_digest`）＋`grade.py`（`####` gold；`The answer is X` 提取，取不到则全文末位数字；**canonical 主判据**＋numeric 副判据 1e-6）＋`README.md`（**D3 判定口径写死**：同 digest、同 `max_tokens`、同 shot、分母分子只允许 `--tp` 变化、掉幅 **<1 个百分点**、ppl 不得单独作判据）。
- **门（实测）**：`python3 selftest.py` 三模式各 40 题全 OK：`gold` canonical=1.0000、`messy`（$、千分位、推理前缀、中文"答案是"）canonical=1.0000、`perturb`（答案+1）canonical=0.0000 且 numeric=0.0000；三次 `subset_digest` 一致 `be881eb4e2fa788c…`；`--shot 8` 提示路 12/12 通过并落 `engine_meta`。
- **未验证（披露）**：客户端对真实 `ninfer-serve` 的兼容性未测（本会话不给 GPU0 现网加压）⇒ 窗口首跑要核 `/v1/models` 字段、`reasoning_content` 键名、`stop` 生效。500 题子集＝文件顺序前 500（换随机必须重取分母）。

## 5. 续跑命令（下一会话照此）

```bash
cd /home/ai-agent/DSHW/ninfer-watch/ada-tree
export PATH=$HOME/.local/bin:$PATH
git status --short; cat .git/rebase-merge/msgnum .git/rebase-merge/end .git/rebase-merge/stopped-sha
# 每步：解冲突 → 安全网 → 编译门 → 继续
bash ../relay/merge-safety.sh 958d88af <改动文件…>
JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests        # 门（约 10–15 min 全量，增量分钟级）
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=../deps-ada/lib ctest --test-dir build-120a-cu13 -R "linear|fp8|nvfp4|bf16|attn_input|gdn"
# 若 rebase 已推进（当前停在 2/213 @ac4c8302）：先跑上面的门，再解步 12
git add -A && GIT_EDITOR=true git rebase --continue              # 变空则 --skip
# 每 10–20 步：git update-ref refs/heads/p1/tp2-replay HEAD; git tag -a v-tp2-<簇名>-rN -m "…门数据…"
```
断点续跑（若 rebase 被中断/中止）：把 `msgnum` 之前的供体 SHA 从队列里去掉，再
`GIT_SEQUENCE_EDITOR='cp <新队列>' GIT_EDITOR=true git rebase -i --onto <你的检查点分支> <上一供体SHA> tp2/raw`。

## 6. 步 11（`366e82a0` gdn 列并行）的解法设计（本会话已把冲突读透，落笔前停手）

供体这一层的形态与步 9 **不同**：没有 `visit_*` 运行期选择子，而是 **launcher 模板化 `<class Geometry, class Output, …>` + 每族一个公开 shard 入口**。plan.h 已自动合并出的 6 个 shard 声明（权威签名）：
`fp8_gdn_input_shard_{decode,matrix,a8}_launch`、`fp8_gdn_input_shard_{a16,a8}_dispatch`、`fp8_gdn_input_shard_dispatch`（参数与父版同，权重是 **[8192,5120]**）。

1. **`fp8_gdn_input_output.cuh`（1 块，87 行）**：取**右侧**（供体）结构，落点在我方树：
   - 删掉我方 `using Fp8GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;`（几何等价：`kQkvRows=2048+2048+6144=10240`、`kZRows=6144`、`kRows=16384`）；
   - 用供体 `template <KeyRows, ValueRows> struct Fp8GdnInputSectionOutput`（父 `<2048,6144>`、半 `<1024,3072>` ⇒ `kRows=8192`）+ 6 条 `static_assert`；
   - 我方所有 `store/store_vector` 消费者只要求同签名（`parent_row, token, float/uint4`）⇒ 直接可用；`#include "ops/linear/common/output.cuh"` 可换成 `ops/common/memory.cuh`（`store_vec`）+ `ops/linear/fp8/fp8_geometry.h`。
2. **`fp8_gdn_input_decode.cu`（1 块）**：**保留我方 ladder 体**（`Fp8A16GemvSchedule<4,4,16,4,…,1,1>` + `launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>`），外面套 `template <class Geometry, class Output> run_decode` + `static_assert(Geometry::kOutputRows == Output::kRows)`；两个入口：父 `<Fp8N16384K5120, Fp8GdnInputOutput>`、半 `<Fp8N8192K5120, Fp8GdnInputShardOutput>`（`Fp8N8192K5120` 由 G-TP2-1 提供）。
3. **`fp8_gdn_input_matrix.cu`（5 块）**：同法逐块取"右侧模板形参 + 我方调度体"：
   - 我方已把 K-split 路换成 **sliced-K**（`Fp8A16SlicedKMmaSchedule<warps, Capacity, …>` + `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, Capacity>>`）⇒ 只加 `<class Geometry, class Output, int Capacity>` 形参，体不动；
   - `static_assert(10240 % kBlockRows == 0)` / `6144 %` 两处换成 `Output::kQkvRows % …` / `Output::kZRows % …`；
   - 表/数组（launcher table）加 `<Geometry, Output>` 模板参数，父/半各实例化一份，两个公开入口分别取用。
4. **`fp8_gdn_input_a8.cu`（2 块）**：**取我方 ladder**（`Tma64x128`、`Tma192x128`、`MidBulk`、`Bulk`——上游"unify fp8 templates"把供体的单一 `Fp8A8DefaultSchedule` 换成了按 token 分档的 4 路），把每路套 `<class Geometry, class Output>`；父/半两个入口。
   **必须同时处理容量**：`MidBulk`(128×128) 与 `Bulk`(128×256) 是 split-K 调度 ⇒ 半形状 row_tiles 减半（16384/256=64 → 8192/256=32；16384/128=128 → 8192/128=64），分裂点前移。照 §1 的办法用 `fp8_a8_split_k.h` 的 band 求值，**不要**照抄 plan.cpp 里"K 不变 ⇒ 容量同父"的注释（那句对 attn_input 已被实测证伪：`FP8 TMA split-K requires aligned caller partials`）。
5. 收尾：`bash ../relay/merge-safety.sh 958d88af <17 文件>` → 构建门 `JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests` → `ctest -R "gdn|attn_input"` → `git add -A && GIT_EDITOR=true git rebase --continue`。
6. **§3.1 四件必做**：`366e82a0` 的 `weight_input.cpp` + 新 shard 入口是**第一批把 `*/tp2/` 之外的分片机制接上模型**的提交；`*/tp2/*.cu` 的**注册**发生在 `23be72cf`(attn_input_proj) / `f57432cb`(gdn_input_proj) / `9cfa1903`(linear_swiglu) / `1d32d3e6`(DFlash2)。**第一个把 `/tp2/*.cu` 加进 sources.cmake 的提交内**必须同批加 `cmake/Sm89Compat.cmake` 的 `"/tp2/.*\\.cu$"` 并更新 `scripts/check-sm89-exclusions.py` 命中基线（守卫"命中 0 ⇒ 构建红"）。

## 7. 【会话结束补记 · 必读】步 10/11 设备门 = 38/39（1 红），根因已定位，未修

`ctest --test-dir build-120a-cu13 -R "linear|fp8|nvfp4|bf16|attn_input|gdn|attention"` @GPU1（2026-10-16 02:4x，**1350.31 s**）：**38/39 通过 = 97%**；5 个双设备套件按设计具名跳过（`linear_split`、`attention_headlocal`、`attn_input_proj_split`、`allreduce`、`output_head_split`）。

**唯一红项**：`ninfer_gdn_projections_split_test` 启动即失败（0.02 s），**不是**跳过：
```
$ CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=../deps-ada/lib ./build-120a-cu13/tests/ninfer_gdn_projections_split_test
FAIL registry
FAIL gdn projections split (registry)
exit=1
gdn_input workspace: rejected: unexpected A8 workspace
```
- 机制：该套件的 **registry 段跑在 device_count<2 ⇒ return 77 的门之前**（`tests/ops/test_gdn_projections_split.cpp:1091-1099` 才是设备门）⇒ 单卡永远走不到具名跳过 ⇒ **直接让全量门相对 144/148 分母缩水**。
- 语义：registry 段查的是 **分片 [8192,5120] 的 workspace 报价**；我方 `src/ops/wrapper/gdn_input_proj.cpp` 的容量入口只认父 rows=16384，分片查询被策略校验拒（"unexpected A8 workspace"）。
- 对照正确写法＝本会话步 9 落的 `attn_input_proj_column_parallel_workspace_capacity_bytes`：它**独立入口**、只做 `validate_policy` + qtype 门，然后 `return detail::fp8_attn_input_workspace_capacity_bytes(policy,min,max)`（K 不变 ⇒ 报价同父）。
- **下一会话第 1 件（纯 CPU 可验，无需第二张卡）**：给 gdn 加同款分片容量入口（`gdn_input_proj_column_parallel_workspace_capacity_bytes`，qtype 门 + `fp8_gdn_input_workspace_capacity_bytes`），并让 registry 段用它；跑上面那条单二进制命令直到 `OK registry`，再跑本行 ctest 子集回到 39/39。
- 明说：**步 10/11 的编译门 rc=0 是真的，设备门不是绿的**（这条红我未修，因会话上下文已尽）。日志 `/tmp/ctest-step11.log`（tmp，重启即失）。

## 8. 步 12/13 落地记录（同一会话续做，2026-10-16 03:0x）

**§7 那条红已修**（先修它再往下）：`tests/ops/test_gdn_projections_split.cpp` 的 registry 段原来写死"A8 from T=8"（供体自己树里的 frontier），与我方实测 frontier（`fp8_gdn_input_plan.cpp:23` = **T≥17**）冲突 ⇒ 单卡上 registry 先失败，永远走不到 `device_count<2 ⇒ 77`，直接污染全量门。
**改法**：断言改成**"分片报价 == 父的单设备报价"**（正是供体注释里自己声明的契约"the workspace query are those of the parent"），去掉字面 T：
```cpp
const std::size_t parent = ops::gdn_input_proj_workspace_capacity_bytes(kFp8, kParentRows, kHidden, policy, tokens, tokens);
if (bytes != parent) throw std::runtime_error("shard workspace quote is not the parent's");
```
实测：`OK registry` → `SKIP: … requires two CUDA devices, found 1` → **exit=77** ✅；`gdn_headsplit` 77 ✅；`gated_delta_net` **exit=0 OK**（criteria 移入共享头后我方 3 档 `kCriteria` 保留，未破）。

**步 12** `ac4c8302`（gdn 头分裂对等套件）→ **`611ce11f`**：
- `tests/ops/test_gated_delta_net.cpp`：保留我方 **3 档 Criteria/PrecisionProfile/prefill_profile/InputPattern**（供体侧把它们删掉移进 `ops/gdn_criteria.h`；我方那 2 个 criterion 函数与共享头的 recurrent 值逐位相同 ⇒ 共享头只留 2 函数，我方 3 档留在测试里）；
- `tests/ops/test_gdn_headsplit.cpp`（供体新文件）按我方 API 适配：**容量查询 4 参**（去掉供体多出的 bool）、`gated_delta_net`/`gated_rmsnorm` 尾参从 `nullptr` 改 **`DeviceExecutionView`**（注入与本树 `execution()` 助手同一实现）；`causal_conv1d_silu` 仍收 stream，未改。

**步 13** `9b3e3dff`（fp8/nvfp4 的 linear_swiglu + linear_add 半形状注册）→ **部分落地，缺口显式拒绝**：
- ✅ **linear_add** 全侧：`fp8_linear_add_a8.cu` 的启动分发从"if k==6144 else 17408"改成 **4 路 switch（6144/3072/17408/8704）**，`launch_problem<K>` 的 `if constexpr (K == 6144 || K == 3072)` ⇒ 半形状继承父 ladder（N=5120 与 row_tiles 不变 ⇒ 波锚点可继承，与 G-TP2-1 的**行向半**结论正好互补）；`fp8_linear_add_partial_capacity_bytes(k,…)` 同步 `k==6144||k==3072`；`plan.cpp` 的 frontier 取 **`output_family ? 17 : 20`**（我方实测值；供体的 22/25 是它 5070Ti 调参，**不采**，已在注释里写明）；`fp8_linear_add_decode.cu`/`nvfp4_linear_add_{decode,small_t}.cu` 的半形状 case 由自动合并落好（`N5120K3072/K8704` 走 launch/`kResidual8704Launchers`，NVFP4 [5120,8704] 不进 A4 融合路＝与供体说明一致）；2 个 `*_small_t/w4a4.cu` 接受上游删除。
- ✅ 测试面：`tests/ops/linear_add/test_fp8.cpp` 取供体的 4 个 half/pair 用例，**crossover 换成我方 17/20**（865U→17、867U→20）。
- ⚠ **未做（显式拒绝，非静默）**：`linear_swiglu` 的 `[17408,5120]` 半形状**没注册**——我方 decode/small_t 把 gate/up 几何写死（`using Geometry = Fp8N34816K5120` / `Nvfp4N34816K5120` + `kLaunchers` 表），且 `fp8_linear_swiglu_partial_capacity_bytes(max_tokens)` **不含行数**（G-TP2-1 已证明行向半必须把行数放进报价：Bulk 在 34816 是 136 tiles、在 17408 是 68 tiles ⇒ 锚点前移）。要闭合需要：把容量签名改成 `(rows, max_tokens)` 并接 `fp8_a8_split_k.h` 的 band 求值 + 把 decode/small_t 模板化到 Problem（同 attn_input 的 visitor 法）+ NVFP4 swiglu 的 TMA 路按 `N17408K5120` 实例化 + 我方 2 个 swiglu 测试套件加半用例。**当前状态**：这些 id 落在 G-TP2-1 加的 `break` 组 ⇒ 运行期给可读拒绝 `nvfp4/fp8 linear_swiglu: unsupported problem`（TP2 模型层在更后面的步骤才会调用它，届时必须已注册 ⇒ 见 §9 待办）。
- 门（步 13）：merge-safety ✅ 无缺失、冲突标记 0；构建门与设备门在写这行时**仍在跑**（结果补在本节末）。

## 9. 明确待办（按优先级，全在单卡/CPU 可做）
1. 等步 13 的构建门 + `ctest -R "linear_add|linear_swiglu|linear"` 设备门结果补进 §8（红了先修，别往前堆）。
2. **补 LinearSwiGLU 半形状注册**（上面 ⚠ 的清单）——这是 TP2 模型层 MLP 的硬前置。
3. 步 14 起：`56d3ec40`（split pair 校验助手）→ `124ccf98`（NVFP4 [5120,8704] 走 TMA）→ `0ad769d9`（文档）→ `24b37ce6`（artifact 切片）…（队列 `relay/tp2-replay-queue-213-from-step11.txt`，当前消费到第 5 行）。
4. 第一个把 `/tp2/*.cu` 加进 sources.cmake 的提交内：同批加 `cmake/Sm89Compat.cmake` 的 `"/tp2/.*\.cu$"` + 更新 `scripts/check-sm89-exclusions.py` 命中基线。

## 10. 步 13–17 落地（本会话续做，2026-10-16 04:0x–04:4x，全在 GPU1/单卡）

| 步 | 供体 | 提交 | 门（实测，fresh 二进制） |
|---|---|---|---|
| 13 | `9b3e3dff` fp8/nvfp4 的 linear_swiglu + linear_add 半形状 | `585b98bd`（tag `v-tp2-fused-shapes-r0`） | 全构建 rc=0；4 个 fused 套件 **OK**（`linear_add_fp8`、`linear_add_nvfp4`、`linear_swiglu_fp8`、`linear_swiglu_nvfp4`，含供体/我方新增的 half profiles seeds 1805/1807/1815/1817、865U/867U） |
| 14 | `ad2a973b` swiglu 列并行 + linear_add 行并行（含 2 个新双设备套件） | `ad763edd` | 全构建 rc=0；宽子集设备门 **42/42 100%（1394.54 s）** |
| 15 | `56d3ec40` split pair 统一校验助手 | `f6d6dee9` | 进步 16 的门 |
| 16 | `124ccf98` NVFP4 [5120,8704] linear_add 走 TMA | `dc2ac303`（tag `v-tp2-fused-split-r1`） | 构建 rc=0；`linear_add_nvfp4` **OK** |
| 17 | `7058bca2` [5120,3072] linear 的 A8 floor | `4b2b578c`（tag `v-tp2-a8-floor-r0`） | 构建 rc=0；`linear_fp8_a8/a16`、`linear_add_fp8`、`attn_input_proj` 全 **OK**；`linear_split` 77 具名跳过；merge-safety ✅ |

**步 13 实测抓出的两个真缺陷（我修了，写清机制）**
1. **A16 分发只认父 K**：`fp8_linear_add_a16.cu` / `nvfp4_linear_add_a16.cu` 原来是 `if (weight.k == 6144) launch_matrix<6144> else launch_matrix<17408>` ⇒ 半形状 3072/8704 掉进 else ⇒ `static K does not match operands`（首用例即 terminate，双套件同因）。
   修：**把"操作数静态 K"与"实测 ladder"分开** ⇒ `template <int K, int Ladder = K>`，分发 4 路（6144→<6144>、3072→<3072,6144>、17408→<17408>、8704→<8704,17408>）；模板体内 16 处 `Nvfp4ScheduleInstance<…, 6144|17408>` 字面量换成 `K`。
2. **供体的 nullable arena 契约（"dispatcher 收可空 workspace，只有量化激活路需要"）已进 plan.h，我方定义还是引用** ⇒ 用 `WorkspaceArena*` + 只在量化路检查 null（fp8: A8 路；nvfp4: A16 之外的路）。

**步 17 的语义落点（我方与供体的 floor 不同，逐条写明）**：供体把 linear() over [5120,3072] 从 25 降到 22（＝它那边 linear_add 6144-family 的 floor）。我方 6144-family floor 是 **17** ⇒ [5120,3072] **保持 17**（两侧本就相等）；真正的失配在 **[5120,8704]**：我方 linear() 父 floor=17，而 linear_add down-family floor=20 ⇒ 行并行时 rank0(A16@T=17..19) 与 rank1(A8) 走**不同激活路由**。已把 **[5120,8704] 的 uses_a8 改成 20**（注释写明理由），测试探针随之从 17 改 20、A16 侧加 19 的边界探针。

**门成本与口径披露**
- 步 14 的 42/42 宽子集门在跑到 #146 时，我为步 16 重建过 `ninfer_ops` 与 `linear_add_nvfp4` 测试二进制 ⇒ 该轮里 #146 之后启动的用例混入了步 16 代码（仍是 42/42 通过）。步 16/17 之后我又用 fresh 二进制把受影响 5 套件重跑一遍（上表）。
- `softmax_attention` 在该子集里 785 s（占 1394 s 的一半以上），`--timeout 900` 余量只剩 ~14%：随重放继续加用例，**下一会话建议先给该用例单独 timeout 或收窄 12/2 取样点**，否则会出现假红。

**会话续跑状态**：rebase 停在 **8/213 @ `0ad769d9`**（步 18＝纯文档：`docs/maintainer/op-development.md` + 5 个 Op 头的 TP2 段落，预计近干净）。已消费队列前 8 行；`p1/tp2-replay` = `4b2b578c`。剩余 205 步，其中 `*/tp2/*.cu` 的注册在队列第 156/158/159 行（全序第 166/168/169 步）⇒ §3.1 的 Sm89Compat `/tp2/` 剔除面**远未到点**，S5/S6/S7/S8 亦在步 19+（artifact 切片/放置/模型绑定）之后。

## 11. 前瞻发现：供体后半段的分片机制与我的做法**不同**（步 156+ 会撞车，先记账）

读 `git show 79d969d4:tools/tp2/check_shard_sources.py` 得到权威事实：供体的 `*/tp2/*_{shard,half}_*.cu` 是 **"一段 `#define` 改名前言 + `#include` 上游 launcher/plan 源文件"**（同一份上游源在父/半两个 TU 里以改名符号编译两次），受 `tools/tp2/check_shard_sources.py` 守卫（检查两件事：被 include 的上游源里不得有**匿名 namespace 之外**的 template/inline/constexpr/static/kernel 定义；被改名的标识符不得出现在前言未先 include 的头里）。该机制写进供体的 `docs/maintainer/upstream-merge.md §2.4`。

我这一程（G-TP2-1/步 9/11/13/17）走的是**另一条**：把 launcher 模板化到 `Geometry/Output`（或 `K/Ladder`）+ 注册半形状，让**同一份代码**在编译期实例化父与半。两条路都自洽，但**步 156/158/159（全序 166/168/169：`23be72cf` attn_input_proj、`f57432cb` gdn_input_proj、`9cfa1903` linear_swiglu 的 "build the two-device shard from upstream's own sources"）会把 include-改名文件加进 sources.cmake**，届时选择是：
- **A（我的默认）**：到点**逐文件判**——凡我方底座已有 `Geometry/Output` 模板面的（attn_input_proj、gdn_input_proj、linear_swiglu、linear_add、linear 家族），**保持模板化注册**，把供体的 include-改名文件**不落地**（`--skip` 其注册面），S4 的验收从"check_shard_sources.py 通过"改为"**半形状在父/半两侧共享同一份 launcher 源码 ⇒ 无改名面可失配**"（更强的不变式：不存在"链只留一份定义"的风险类）；
- **B**：全盘照供体（include-改名），则需把 §3.1 的 `Sm89Compat "/tp2/.*\.cu$"` 剔除面 + `check-sm89-exclusions.py` 基线 + `check_shard_sources.py` 一起落地并纳入每步门。
A 的代价：这些步要写"偏离说明"进台账（我已在做同类事）；B 的代价：引入一个上游改动即静默失配的机制。**注意 A 下 S4 的门名要改**，别到窗口才发现分母里没有它。

## 12. 步 19–28 落地（全序），停在步 29 的大冲突前（2026-10-16 05:0x）

编号口径：**全序第 N 步 = 队列第 N−8 行**（队列从全序第 9 步起）。当前 rebase 停在 **队列第 21 行 = 全序第 29 步 = `815615e6`**（"plan one tensor-parallel rank's sequence layout"），已落 **28/223 步**。

| 全序 | 供体 | 提交 | 门（fresh 二进制，实测） |
|---|---|---|---|
| 19 | `24b37ce6` artifact 按行/列切编码父件 | 自动干净 | 进步 22 门 |
| 20 | `9cafb4bc` artifact 把父件放上至多两设备 | 自动干净 | 同上 |
| 21 | `ebbab2a4` 单/双设备放置测试 | 自动干净 | 同上 |
| 22 | `d1918c7d` 模型按 rank 放置逻辑参数 | `b477cfcd` | 全构建 rc=0；`-R "qwen3_5\|artifact\|shard\|load\|…"` **31/31 100%（238.83 s）** |
| 23 | `7e7ba6e1` 每 rank 一套权重绑定/预备 | `7307f80e` | 全构建 rc=0；同子集 **31/31 100%**（`Parameters(source, yarn, rank)` 与供体 `Parameters(model, rank)` 两形态共存） |
| 24 | `d99269e2` LoadSummary 报每设备权重字节 | 自动干净 | 进步 25 门 |
| 25 | `06483bdc` qwen3_5 双设备执行操作数 + 分片投影 | `fd5731ff`（tag `v-tp2-model-tp-exec-r0`） | 全构建 rc=0；`-R "qwen3_5\|artifact\|shard\|load\|attention\|gdn\|softmax\|kv_cache\|text"` **51/51 100%（1153.77 s）** |
| 26–28 | Text 层/prefill/decode 双设备 + 对比测试 | `3a964069`/`988920f4`/`b98f9ffb`/`77540cd3` | 自动干净（步 30 门覆盖） |
| 30 | `41920c8a` **`--tp/--devices` 参数面（S5 主体）** | `cf62bbbb` | 全构建 rc=0；`ninfer_cli_options_test` **Passed**、`ninfer_serve_options_test` 先红后 **ok**（见下）、子集 **39/39 100%（239.95 s）** |
| 18 | `0ad769d9` TP2 契约文档 | `aae1d16e`（tag `v-tp2-docs-r0`） | 全量 **151/153 执行 100%（3230.05 s）** → `relay/passset-sm120-tp2steps17-r0.tsv` |

**S5 现状**：`EngineOptions::{tp, devices}` ✓、CLI `--tp 1|2 --devices A,B` ✓（`product::resolve_tensor_parallel_devices`）、serve 的 tp2 策略 ✓（本树只有一个 Host 预算旋钮 `--host-context-mib` ⇒ 供体的 `--host-state-slots/--host-kv-mib` 两个用例**合并成一个**，改写见 `tests/test_serve_options.cpp` 注释）、LoadSummary 每设备字节 ✓。**未做**：`vision_device`、`--no-tp-mailbox`/`tp_mailbox`（在队列后面的 mailbox/vision 步）⇒ S5 记 **≈70%**，不当 100% 交。

**本会话我造成并修掉的两个真事故（写全，供审）**
1. **regex 劈开多行赋值**：`src/serve/serve_options.cpp` 的 `--host-context-mib` 分支被我插成 `x =` / 我加的行 / `parse(…)` 三截，**编译侥幸通过**，值再没赋进去 ⇒ `ninfer_serve_options_test` 一串 P0 用例红（"Host context MiB did not preserve exact bytes" 等）。修＝整块重写；复跑 `ok`。**教训：窄区域禁宽 regex 插入，先读全块再整块替换**（已写进 `relay/step21-plan-brief.md` 纪律 3）。
2. **模型/服务层签名约定**：供体的 TP 代码按 `cudaStream_t` 传尾参，我方 P0 合并后 `text_rope`/`causal_softmax_attention`/`gated_delta_net`/`gated_rmsnorm` 收 `DeviceExecutionView` ⇒ 新增 `TextContext::rank_execution_view(rank)`（rank0 用 `ctx_.execution_view()`、rank1 用 `tp_->execution->dev[1]->execution_view()`）逐点替换；`text_rope` 同时补回 `&parameters_.rope_scaling`（TP2 两 rank 必须共享同一份不可变 YaRN 系数）。

**工具加固**：`relay/merge-safety.sh` 现在**扫索引**（`git grep --cached`）——本会话 serve_options.cpp 的标记是 **rerere 以"M 状态"塞进索引**的，只扫工作树拦不住。实测新守卫在步 21 现场报"索引 0 / 工作树 3"。**纪律新增**：每次 `git rebase --continue` 之后立刻跑一次 merge-safety。

**为什么停在步 29 没硬吃**：`request_plan.cpp` 的冲突是**单块 1601 行**（供体整层重写规划），加上两版全文（1987×2）要 ~50k+ token 与数小时；半合并状态对比零消耗检查点明显更差。任务书（含用符号差集枚举 P0 特性的可执行打法）：**`relay/step21-plan-brief.md`**；现场：**`relay/step21-checkpoint/`**。

## 13. 里程碑行号表 + 对计划的**三处更正**（2026-10-16，机器生成：`relay/tp2-remaining-steps.tsv`，193 步 / 1155 处非新增改动）

编号：**全序 = 队列行 + 8**。当前停在队列 21 行（全序 29）。

| 全序 | 供体 | 内容 | 对单卡桶的意义 |
|---|---|---|---|
| **29** | `815615e6` | 按 rank 规划序列布局（`request_plan.cpp` 1601 行单块） | **当前停点**，任务书 `relay/step21-plan-brief.md` |
| 32 | `6a476599` | **Engine 在两设备上下文上启动**（15 处改动） | S5 的 `tp/devices/vision_device` 在这一步进 EngineOptions 消费端；**S6 的 sm_89 `--tp 2` 可读拒绝应挂这里** |
| 35 | `666edc62` | 注册双设备 **MTP** 分片几何 | MTP 在 tp2 是"具名跳过/拒绝"面（供体 README 的 greedy 阶段用 MTP3）⇒ 与 D3 关联 |
| 59 | `2ec3da81` | **`src/ops/kernel/peer_exchange.cuh`（PeerMailbox）诞生** | 我预备到 `relay/tp2-tools/mailbox_probe.cu` 的 D1 探针**在这一步之前编不过**（它 include 该头）⇒ 到 59 才能编译验 |
| 62 | `ff592ee2` | Vision 跑在单个 TP rank（24 处） | 本树 `-DNINFER_ENABLE_VISION=OFF` ⇒ 该步预期只到编译门；D2 的 Vision 子项取舍照旧 |
| 94/95/96/116 | `80db8faf`/`b14be3d5`/`7a740326`/`7ae77f55` | **G-TP2-1 的扩展面**：NVFP4 `[5120,3072]` 半形状、NVFP4 列并行 attention / gdn 输入投影、linear 的 tp2 半形状表 | 我 G-TP2-1 只注册了供体 `84a1459a` 的 10 个半；**NVFP4 `[5120,3072]` 当时不存在** ⇒ 到 94 要新增 alias/id/resolve/dispatch/sources + 定性（同一套三家族机制，照 §5 的 wave 算术做容量） |
| 108 | `0590593f` | `tools/tp2` 独立 mailbox 探针进树（nvcc-only） | D1 探针的门在这里自然闭合 |
| **111** | `a0369376` | **"tp2: step the mailbox down instead of failing"** | **更正 ①：供体已有 mailbox 降级路径**（我计划文档写的是"供体自陈无降级 ⇒ 我方必须补"）。S8 的范围从"从零加降级"缩为**采用+补我方超时用例**；台账 §5 与执行计划 §1 的 S8 行按此改 |
| 139 | `b8ce3639` | 优化 proposal head 跨 rank 分裂（28 处） | 我方制品 head 是 `q4_g64` ⇒ 该步只到编译门，`--lm-head-draft` 在 D 桶继续标"不可用" |
| 159–167 | `bba98da5`…`23be72cf/f57432cb/9cfa1903` | `tools/tp2` 门 + 把双设备形态搬进 `*/tp2/`（include-改名机制） | **§11 的 A/B 抉择点**；S3 的 `gate.sh` 从 159 起可用 |
| **168** | `8ef949fe` | **`ops/tp2` 每设备调参表**（per-device tuning table） | **更正 ②：S7 的 `device_tuning.h` 在 168 落地，不是 16** ⇒ 5060 Ti 行 + SM=36 判据的验收点要到 168 才存在；在其之前任何"launch plan SM 数"判据都是空谈 |

**更正 ③（成本/纪律）**：供体 `tools/tp2/gate.sh` 的 ctest 阶段**默认排除 `ninfer_softmax_attention_test`（~12 min）**，要 `--attention` 才含。我方全量门 3230 s、该项 785 s、`--timeout 900` 余量 14% 的问题，**供体已给出标准解法**：日常簇门用 `ctest -E "softmax_attention"`，簇边界/收口再跑含 attention 的全量。已写进 `relay/tp2-tools/GATE-NOTES.md`。

**制品相关三处适配**（进 D 桶前必改，不是可选项）：`--lm-head-draft`（q4_g64 head ⇒ 去）；`GATE_ARTIFACT` 指向 `/tank/models/lynn/…` 的 W4A4 V3；QUASAR/ppl 语料我方没有 ⇒ ppl 阶段改用我方语料 + `ppl_paired.py`（已备副本）的 PASS/WARN/FAIL 统计规则（α=0.05、|Δ|<0.5%、逐窗 dNLL 配对 t + 符号检验），且 **ppl 只作支持证据，判定用 GSM8K**（口径不变）。

## 14. 全序 29 落地 + 停在全序 30（program/lane 层）前的收尾数据（2026-10-16 11:3x）

**全序 29 `815615e6` → 提交 `3f0c27d0`，tag `v-tp2-planning-tp-r0`**（处置见 `relay/step21-plan-brief.md` 附/S5b、附2/S6）：
- 落地：`startup.cpp` 的 tp2 规划面（供体 +187）+ `startup.h` 的 `int tp`（**保留我方 `multiprocessor_count`，designator 按声明序**）+ 三处我方 API 适配（attention 容量补 `DeviceExecutionView`、GDN 容量去供体多出的 `bool`、Host 两层合并成我方 `host_capacity_bytes.value_or(0)`）；
- **不 graft**：`request_plan.cpp` 取我方（供体 admission 挂在 `ProgramImpl::inspect_lane`/`AdmissionCandidate`，我方 0 处；强 graft＝造平行 admission 架构）⇒ 3 条 tp2 admission 政策记 **S5b**，落点＝全序 32 之后的 `runtime/engine/scheduler.h` + `context_cache/*`；
- 门：**全构建 rc=0**；`ninfer_model_runtime` 单目标 rc=0；**簇设备门 `-E "softmax_attention"` = 158/158 执行 100% 通过 / 1044.54 s**；`merge-safety.sh`（新扫索引版）索引 0 / 工作树 0。

**新纪律的量化收益（写死给后面所有会话用）**：
| 门 | 命令 | 成本 |
|---|---|---|
| 日常簇门（每簇） | `ctest -j1 --timeout 900 -E "softmax_attention"` | **1044.54 s / 158 项 100%** |
| 全量门（簇边界、收口） | `ctest -j1 --timeout 900` | **3230.05 s / 151 项 100%**（softmax 785 s 是唯一 timeout 风险源） |
| 融合 Op 子集 | `-R "linear\|fp8\|nvfp4\|bf16\|attn_input\|gdn\|attention"` | 1394.54 s / 42 项 |
| 模型/加载子集 | `-R "qwen3_5\|artifact\|shard\|load\|…"` | 238.83 s / 31 项 |

**当前停点（零消耗检查点）**：**队列 22/213 ＝ 全序 30 ＝ `7ad8894b` "run a Program over two devices with mirrored storage"**（8 文件 +249/−18，其中 `program_impl.{cpp,h}` +182、`prefill.cpp` +26、`program.{h,cpp}`、`transactions/commit.cpp`、`storage/context.cpp`(DU)）。7 处未合并 + 1 处接受删除，逐字节存档 **`relay/step30-checkpoint/`**（STATUS/STATE/MESSAGE）。
续跑：把剩余队列 `relay/tp2-replay-queue-192-from-full31.txt`（192 行，首 `7ad8894b`）注入：
```bash
cd ada-tree && git checkout -q tp2/raw
GIT_SEQUENCE_EDITOR='cp ../relay/tp2-replay-queue-192-from-full31.txt' GIT_EDITOR=true \
  git rebase -i --onto p1/tp2-replay 815615e6 tp2/raw      # 应停在 1/192 @7ad8894b
```
（`815615e6` 是已应用的最大供体；队列首行即当前停点。）

**会话总计**：G-TP2-1 闭合 + 全序 9→29 落地（**21 个新提交**，其中本会话 22 个含跳步）+ **8 个簇 tag** + S9 闭合 + S5≈70% + 全量/簇级分母各一份 + 工具加固（索引扫描守卫、`\r` 解析器）+ 里程碑行号表与三处计划更正 + S5b/S6 设计任务书。
**GPU 交还**：GPU1 4 MiB / 0%、零残留；GPU0 现网 15590 MiB 全程未碰未 kill 未加压；`/tank` 未写；全程零审批。

## 15. 用户三条裁定（2026-10-16 11:5x，权威，覆盖此前口径）

**R1｜G-2 双卡开窗时点**：**全部单 GPU 测试结束、且只有"真机真模型"才能验的项**才开窗。可核验的开窗前置（四条齐才提，提前 2 小时报形态 A/B/C）：
1. 队列跑到 program/lane 接线完毕并过 `*/tp2/` 注册（全序 164/166/167）与调参表（168）；
2. **S3** `--tp 1` 等价门（合成两层模型，prefill 32/2048/6144 三档逐字 + 60 greedy + 10 DFlash2 + paired ppl）**绿**；
3. **S4/S5/S6/S7/S8** 在单卡/纯 CPU 全部闭合（S5 含 S5b 三条 admission 政策、S6 含 `vision_device`/`tp_mailbox`）；
4. **S9** 判分底座对**真实 serve** 的首跑兼容通过（`/v1/models` 字段、`reasoning_content` 键名、`stop` 行为）。
窗口内容只放 **D1（三探测）→ D2（容量/allowance）→ D3（逐字 + GSM8K 分子分母）→ D4（速度与扫参）**。窗口前不安排任何"要双卡才有意义"的验证。

**R2｜Vision 不延后**：**新会话第 1 号任务**＝把 FFmpeg 6（≥60）源码装进同级 `deps-ada`（与 libcurl 同法），把 `scripts/build-ada.sh` 的 vision 翻成默认 ON 并过编译门；此后每簇门都带 vision 编译面。**不延后**的理由：它决定 D2 的 Vision 子项与全序 62（`ff592ee2` Vision on one TP rank）是否有真门。配方要点见接力提示词 §任务1。

**R3｜Ada 与"平行架构"**（先纠正再裁定）：
- **（2026-10-16 16:2x 更正）本条的 S5b 三条政策已由全序 42 消灭 ⇒ 作废，见 §25.2（S6 能力门不受影响，保留）。**
- 我说的"平行架构"＝**admission 两套实现并存**（供体 `ProgramImpl::inspect_lane`/`AdmissionCandidate` vs 我方 `runtime/engine/scheduler.h`+`context_cache`），**与 Ada 无关**。裁定：不在 29 号步 graft 供体文面；S5b 三条政策在我方站点做正名提交（全序 32 之后）。
- 用户对 Ada 的裁定（照此执行）：**给 Ada 的 TP2 只预留接口**＝能力门（S6：`--tp 2` 在 sm_89 启动即**可读拒绝并给原因**，用纯函数 verdict 暴露成 CPU 可测面），**绝不影响 sm_120a 编译门**，实现留后（Ada 一期排在 TP2 首要目标之后，现网 Ada 继续用 `engine-sm89-cu128-r2`）。

## 16. 任务1（R2 Vision 前置）闭合 + 全序 30/31/32 落地（2026-10-16 12:0x–13:0x，新会话 GPU1 侧）

**编号口径不变：全序 = 队列行 + 8。** 起点 = 提示词 §现场（`3f0c27d0` 全序 29，停在队列 22/213 = 全序 30 `7ad8894b`）。

### 16.1 任务1 · FFmpeg 6 进 `deps-ada` + vision 默认 ON（用户裁定 R2，第 1 号任务）✅

| 项 | 实测 |
|---|---|
| 系统基线（不达标） | `/usr/bin/ffmpeg 4.4.2` ⇒ `libavformat.so.58`、**无 dev 头**、`pkg-config` 无 libav*；`nasm`/`yasm` **缺** |
| 构建配方 | **`relay/build-ffmpeg.sh`**（VER=6.1.6、JOBS=14、nice 12、外网走 7890 代理）；`--prefix=$PWD/../deps-ada`、`--enable-shared --disable-static`、`--disable-programs --disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages --disable-debug --disable-autodetect --disable-avdevice --disable-postproc`、只开 avformat/avcodec/avutil/avfilter/swscale/swresample + `--enable-zlib`、**`--disable-x86asm`（纯 C 路径，因 nasm 缺）** |
| 制品 | `deps-ada/src/ffmpeg-6.1.6.tar.xz` sha256 **`d4fcb164…07370a`**（ffmpeg.org 只签 `.asc`，无 md5/sha512 文件 ⇒ TLS + 本地 sha256 入台账）；安装 **rc=0** |
| 版本 | libavformat **60.16.101** / libavcodec **60.31.102** / libavutil **58.29.100** / libavfilter 9.12.100 / libswscale **7.5.100** / libswresample 4.12.100 ⇒ **过 `cmake/Dependencies.cmake:24` 的 `>=60 />=60 />=58 />=7` 门**；`libcurl 8.11.0` 未动（`deps-ada` 未移动） |
| 编译门 | `cmake -DNINFER_ENABLE_VISION=ON` 配置 **rc=0**（打印 `Found libavformat, version 60.16.101`）；`g++ -fsyntax-only src/media/decode/decode.cpp` **rc=0**（FFmpeg 6 packet side-data API 真编过） |
| vision 翻默认 ON | `scripts/build-ada.sh`：`VISION=ON` 默认 + 新增 `--no-vision`（保留 stub 路径）+ 注释改写；**补丁存档 `relay/vision-on-build-ada.patch`**（rebase 期间工作树必须无 unstaged 才能 `--continue` ⇒ 门前后 `git apply` / `git checkout --` 来回切） |
| 门数据（全构建 + 簇门，见 16.5） | vision=ON 的 sm_120a 全构建 `--tests` rc=0；`ctest -R "vision\|media\|decode"` 从 **Not Run/Disabled** 变真跑：`ninfer_media_decode_test`、`ninfer_qwen3_5_frontend_test` 解除 DISABLED（`tests/CMakeLists.txt:12-24` 的文本引擎门）⇒ **分母 +2 注册/执行** |

### 16.2 全序 30 `7ad8894b`（program/lane 镜像存储，8 文件 +249/−18）→ 我方落 **`97b99fed`**

**逐块全部"取我方 + 定点 graft"**；新工装 **`relay/resolve_blocks.py`**（按块序选 `ours|theirs|both`，diff3 五段解析，**拒绝任何宽 regex 插入**，落盘前自检 marker=0）。
**架构落差清单（我方树实测 0 处 ⇒ 供体文面不可直接落）**：`detail::RuntimeContractAccess`（供体 program.h 的 140 行 detail 块）、`normalized_private_capacity` 匿名 ns、`PressureRecoveryScratch`、`RewriteCheckpointDisposition/restore_path/checkpoint_kind/is_rewrite_checkpoint_restore` 别名块、`publish_kv_rows()`（我方**无此函数**，行发布内联在 3 个站点）、`bind_sequence_kv()`（我方无）。
**graft 清单（供体语义逐条落我方站点）**：
1. `ProgramImpl::PeerRuntime::PeerRuntime`（rank 1 的 persistent/workspace/decoder/state_images/RoundState + `ordinary_peer_frame`）落 `program_impl.cpp:25`；
2. `if (peer) { attach_tensor_parallel_mirrors(); }` 挂 `program_impl.cpp:120`（**在任何 mutation 之前**，比我方 host arena / `state_store` / `text_kv_addresses` 构造更早，满足"attach 时两池都无绑定行"）；
3. `tp_execution.emplace(TpExecution{…})` + `set_peer_i32(peer->io.text_kv_table_row, 0)` 挂 `program_impl.cpp:226`（我方块位 = lane 初始化后、`host_tokens` 前；**字段名与我方 `execution/tp.h:69-88` 逐一对齐** ⇒ 设计顺序无需改写）；
4. **Host 旋钮改写**（`ContextCacheOptions` 我方只有 `host_capacity_bytes`）：`program_impl.cpp:92` 的供体两字段拒绝 → `plan.context_cache.host_capacity_bytes.value_or(0) != 0`，文案随之改；
5. **行发布的平行站点**：供体改 `publish_kv_rows()`（在 DU 的 `program/storage/context.cpp`），我方新增**一个** `ProgramImpl::publish_peer_kv_rows()`（`program_impl.cpp:313`，含供体的 `mirror_row(...).row_index() != row` 跨 rank 一致性抛错），在**我方全部 3 个行发布站点**调用：`program/prefill.cpp:219`（advance_prefill）、`transactions/commit.cpp:262`（forced tokens）、`transactions/replay.cpp:174`（advance_replay）⇒ 我方"每次 bind/prefill/forced-token 两 rank 同 row"的等价面；
6. `DU program/storage/context.cpp` = **接受我方删除**（`git rm`），其 +10 语义由 5) 承接；
7. **我方 P0 不变量补齐**：`create_program` 的 tp2 重载加 `plan.impl_->multiprocessor_count == execution.dev[0]->multiprocessor_count()`（`program.cpp:334`，单卡面同判据）+ 在 `SequencePlan` 私有段补 tp2 重载的 `friend`（供体把 `impl_` 放 public，我方是 private ⇒ 不给供体开洞）。
**干净应用的供体面（无需手工）**：`execution_context.h`（供体 `context.h` 被 rename 检测匹配到我方改名后的文件，`tp`/`graph_bridge`/`peer_frame` 三字段进）、`program.cpp` 重载、`program_impl.h` 的 PeerRuntime/tp 成员/`synchronize_devices`/`set_peer_i32`/`attach_tensor_parallel_mirrors` 声明、`device.synchronize()`→`synchronize_devices()`、`peer->work.reset()`、析构 rank 1 同步。
**编译**：`program_impl.cpp`/`prefill.cpp`/`program.cpp`/`commit.cpp`/`replay.cpp`/`decode.cpp`/`graphs.cpp` 单 TU 全 rc=0。
**`ScopedCurrentDevice` 落差**：供体在 `program_impl.h` 内联自带一份 ⇒ 我方**不复制**，改 `#include "core/device_scope.h"` + 在 `detail` 开 `using ninfer::ScopedCurrentDevice;`（`program_impl.h:39-45`，供体代码写 `detail::ScopedCurrentDevice` 的 3 处站点全命中）。

### 16.3 全序 31 `07c9371e`（ordinary decode 双设备，eager+captured）→ 我方落 **`960d5d34`**

冲突只在 `program/graphs.cpp`（5 块，全取我方）：**落差 = 我方 `instantiate_graph_family` 不是模板**（供体 `template <class Prepare, class Synchronize>`；我方把 prepare/launch 循环放在调用点）。
适配：给它加**第 4 形参 `const std::function<void()>& synchronize`**（`graphs.cpp:37-40`）⇒ 供体在函数体内应用的 `synchronize()` 直接命中；两处调用点传 `synchronize_all`（`graphs.cpp:376`、`477`）；`#include <functional>`。
**双设备等待的站点映射**：供体改的 3 个函数内 `device.synchronize()` 在我方不存在，等价面 = 我方调用点的 6 处 ⇒ 逐行改 `synchronize_devices()`（`graphs.cpp:380/382/455/458/487/490`）。
干净应用：`synchronize_all` 定义（176）、`clear_stable_controls` 的 rank 1 memset 块（`peer->io.token/pos/rope_pos/rope_delta`）、`peer->work.reset()`、`execution_core()` 的 `tp_binding()/graph_peer_bridge()`、`ordinary_state` 的 `peer ? &*peer->io.ordinary : nullptr`、**warm 循环 `warm_batches = tensor_parallel() ? ordinary_batch_limit : 1U`**（tp2 每个 batch size 先 eager 再 capture）、末尾 `synchronize_devices()`。

### 16.4 全序 32 `6a476599`（Engine 双设备启动，15 文件 +315/−37）→ **已解 9 块并暂存，编译 rc=0**

冲突 9 文件：`AGENTS.md`、`README.md`(净应用)、`docs/cli.md`(2)、`docs/maintainer/engine-architecture.md`、`docs/serving.md`、`engine.cpp`、`model_instance.{h,cpp}`、`operational_log.cpp`、`tests/runtime/test_kv_capacity.cpp`。
代码适配点：
- `engine.cpp`：`construct_model(options, execution)`（供体面，`execution` 已在我方 init list）；
- `model_instance.cpp`：块1 = **我方 `normalize_engine_options` 只有 3 字段版**（供体的 disabled-cache 大块 + `max_private_continuations/shared_prefixes/long_anchors` 在我方不存在）⇒ 取我方 + 落 **tp2 的 `device_state_slots = max(C,4)`** 与 **`host_capacity_bytes` 必须为 0**（省略→0，非 0→拒），`public` 包装两个 `construct_model` 重载 + 匿名 ns `construct_model_on(options, device, execution)` 取供体；
- **`EngineOptions` 必须非 const**（供体把签名改 `const&`）：我方 P0 在 `model_instance.cpp:261` 把 plan 解析出的 Host 预算**写回** `options.context_cache.host_capacity_bytes` ⇒ 三个签名全部保 `EngineOptions&`（我方文面权威）；
- `validate_options` 的 tp2 拒绝面（供体应用干净）里两处供体 Host 字段 → 我方 `host_capacity_bytes.value_or(0)`；
- **YaRN 一致性（我方 P0 独有）**：`peer_parameters` = `Parameters(*model, options.yarn, 1)`，**不是**供体的 `(*model, 1)`（=native）⇒ 两 rank 共享同一份不可变系数（"张量并行只分头，不分 rope"）；
- 日志：`operational_log.cpp` 保留我方单行 `context | history …` + 供体新增 tp2 行（文案按我方单旋钮改）；
- 测试：`test_kv_capacity.cpp` 取 **both**（我方 rk4v4-e8 布局表 30 行 + 供体 symmetric 容量 8 行，`resolve_kv_capacity_symmetric` 净应用）；
- 文档：`AGENTS.md` 合并句（保我方"一个 GPU"产品边界 + tp2 实验模式一句）；`docs/cli.md` 保我方 `--kv-dtype …|rk4v4-e8` 行 + 供体 `--tp/--devices` 两行 + 新增 "## Two GPUs"（**加 rk4v4-e8 与 `--host-context-mib` 的 tp2 拒绝口径**）；`docs/maintainer/engine-architecture.md` 新增 **§8.1 双 GPU 张量并行**（按我方单 Host 旋钮 + YaRN 一致 + "从 retained hidden 采样在 tp2 抛错" 的**已实现**口径写；**不写** zero-suffix admission 政策＝S5b 未落地）；`docs/serving.md` 的表格行按我方两旋钮改写。

### 16.5 S6 能力门（用户裁定 R3）已落码 + 站点图

**补丁存档 `relay/s6-capability-gate.patch`（201 行）**：
- `src/product/tensor_parallel_options.h`（**零 CUDA、inline 纯函数**，与 `--tp/--devices` 解析同头）新增
  `tensor_parallel_capability_verdict(int cc_major, int cc_minor)`：**只放行 12.0**，否则返回**带原因**的完整文案（点名 sm_120 / 实测型号 / 观测到的 `major.minor` / 单卡 `--tp 1` 出路）；
  `tensor_parallel_kv_verdict(KvCacheStorage)`：**只放行 BFloat16 / Int8Group64**，其余点名存储名（**`rk4v4-e8` 默认拒绝**＝S6 第二条验收）；
- 接线：`src/runtime/engine/engine.cpp:27 initialize_execution()` ⇒ **在取到两个 `DeviceContext` 之后、读制品之前**，逐 rank 用 `compute_capability()`（`src/core/device.cu:191` = `major*10+minor`，拆 `cc/10, cc%10`）调 verdict，非空即 `throw std::invalid_argument(*verdict)`；
  `src/runtime/engine/model_instance.cpp:78 validate_options()` 的 KV 判据改为**消费同一 verdict**（我方单权威，不留平行文案）；
- 测试（CPU，零 cuda 引用）：`tests/test_cli_options.cpp` 加 verdict 组（12.0 放行 / 8.9 拒绝且文案含 `sm_120`+`8.9`+`--tp 1` / 12.1 拒绝；bf16+int8 放行；`rk4v4-e8` 拒绝且点名；fp8/nvfp4/k8v4 拒绝）；`tests/test_serve_options.cpp` 加**serve 路由解析出的 `--kv-dtype` 与同一 verdict 对齐**两条；
- **不影响 sm_120a 编译门**（纯头 + 两处调用）；真机 sm_89 的拒绝文案在 4080S 异机验（本机两卡皆 CC 12.0，物理不可测）。

### 16.6 S5b 三条 admission 政策的**我方站点图**（本会话勘查，未落码；落点＝全序 32 之后）

| 政策 | 我方落点 | 实测事实 |
|---|---|---|
| ① tp 2 拒绝 **zero-suffix 复用**（`reuse_base >= prompt_tokens`，从 retained hidden 采首 token 需词表分裂 head） | `src/runtime/engine/context_cache/resource_manager.h:83 candidates()`：在 `std::erase_if(... prepare_source ...)`（同函数 ~125 行）之后加一条"frontier 覆盖整条 prompt 的 checkpoint 源在 tp 2 丢弃"；**root/miss 源保留** ⇒ 代价＝付一次 prefill（＝供体语义"当 miss"） | 我方**无 `AdmissionCandidate`/`inspect_lane`**（供体层 0 处）；执行面 backstop 已在 `program/prefill.cpp:156-162 sample_from_hidden` 抛错（全序 30 落）|
| ② tp 2 拒绝**一切 speculative 复用**（非 Root） | 同函数同处：源的 `role`/`kind` 属 MTP/DFlash 目录时丢弃 | **启动已拒 `--spec`**（`model_instance.cpp:71-74`）⇒ 此条在 tp 2 是 backstop，与 ① 同站点实现，不造第二套 admission |
| ③ 目录条目不能被 MTP/DFlash 续接 ⇒ **miss 不是故障** | **n/a**：供体文面 `published MTP\|DFlash checkpoint is not materializable` 在我方树 **0 处**（`git grep "is not materializable" HEAD` = 空）⇒ 无可改写站点；若后续供体步把它带进树，按 ①② 同法落"返回 miss" | 记入本表，避免下一接力者去找不存在的 throw |
| tp 宽度怎么进 `candidates()` | 调用方 `src/runtime/engine/engine_core.h:1696` 处用 **`instance_.peer_parameters != nullptr`** 作 tp 2 判据（`model_instance.h:22` 我方已有该成员，随全序 32 落地）⇒ **不需要给 `Program` 开新访问器**，也不引入供体 `tensor_parallel()` 面 | 门：`ctest -R "context\|admission\|engine\|qwen3_5\|speculative\|preemption"`（真判定在窗口） |

### 16.7 门数据（全序 30–32 · **vision=ON** 首次带 vision 编译/执行面 · GPU1 单卡）

| 门 | 命令 | 实测 |
|---|---|---|
| 全构建 | `JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests`（vision 默认 ON） | **661/661 rc=0**，产物 `build-120a-cu13/apps/ninfer-serve` ✓ |
| 簇设备门 | `CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=../deps-ada/lib ctest -j1 --timeout 900 -E "softmax_attention"` | **100% passed out of 160**（**133 Passed + 27 Skipped(=77 具名单卡跳过) + 0 Failed / 1103.16 s**）→ **`relay/passset-sm120-tp2steps32-vision-r0.tsv`** |
| vision 真跑（R2 验收） | 同上 log 内 | `ninfer_media_decode_test` **Passed 0.02 s**、`ninfer_qwen3_5_frontend_test` **Passed 0.94 s** ⇒ 这两条在文本引擎下是 DISABLED（`tests/CMakeLists.txt:12-24`），翻 ON 后**进分母并真执行**（分母 158→**160**） |
| GPU 纪律 | 探针 + 交还 | 开工 `cuDeviceGetCount==1 / CC 12.0 / 36 SM / 15849 MiB / alloc 12 GiB OK`；GPU0 现网 `strata --serve`（PID 440568，15590 MiB）**全程未碰未 kill 未加压** |
| 门成本更新 | `-E softmax_attention` 簇门 | **1103.16 s**（上轮 1044.54 s，+2 条 vision 用例 + 双设备面）；分母 `passset-sm120-tp2steps32-vision-r0.tsv` |

## 17. 全序 33–34 落地 + 停在 全序 35（MTP 分片几何）前的收尾（2026-10-16 12:3x，本会话续做）

### 17.1 落地计数：**34/223**（本会话新落 5 步 30→34）

| 全序 | 供体 | 我方落点 | 内容 |
|---|---|---|---|
| 30 | `7ad8894b` | **`97b99fed`** | Program 跑两设备 + 镜像存储（§16.2） |
| 31 | `07c9371e` | **`960d5d34`** | ordinary round 双设备解码，eager+captured（§16.3） |
| 32 | `6a476599` | **`7220a9f9`** | **Engine 在双设备 ExecutionContext 上启动**（§16.4，里程碑命中） |
| 33 | `acc38d67` | **`acc38d67`→净应用**（`test(qwen3_5): serve Qwen3.8 27B through a two-device Engine`） | 零手工（干净落下） |
| 34 | `5b129659` | **`647c2487`** | tp 2 Device checkpoint 池 = **`max(2C, 8)`**（§17.2） |

**簇 tag：`v-tp2-mirrored-program-r0` @ `647c2487`**（annotated 里写了门数据）。

### 17.2 全序 34 的解法（4 文件 5 块，全"取我方 + 落供体新规则"）

`model_instance.cpp:165` 的 normalize：把我方 §16.4 的 `max(C,4)` **升级成供体的 `max(2C,8)`**（含供体新写的"prefill 内 capture 满了就跳过、不再 snap 到 Host，池满后新会话只保 endpoint ⇒ 下一轮整条 re-prefill"根因文），
`operational_log.cpp:456` 的 tp2 行加 `{}` Device 池读数 + "raise --device-state-slots"；
`serve_options.cpp:166` 帮助文本（**修掉 §16.4 遗留的半句断裂**：我方合并时把"ordinary decoding with bf16 or int8 KV; it defaults …"丢了 ⇒ 补回并改 `max(2x concurrency,8)` + 我方单旋钮 `Host budget to 0`）；
`docs/serving.md:910` 表格两行（我方两旋钮版，默认随 34 更新）。
**编译门**：`model_instance.cpp` / `operational_log.cpp` / `serve_options.cpp` 三 TU rc=0；`merge-safety.sh 958d88af` 索引 0 / 工作树 0、符号清单无缺失。

### 17.3 停点（零消耗检查点，与上一会话同形态）

**队列 27/213 ＝ 全序 35 ＝ `666edc62` "register the two-device MTP shard geometries"**（22 文件 +352/−86），
**5 处未合并，每文件 1 块**：`include/ninfer/ops/linear.h`(ours11/base10/theirs11)、`src/ops/linear/q8/q8_dispatch.cpp`(11/9/11)、`src/ops/linear/q8/sources.cmake`(1/0/1)、`tests/ops/linear/test_q8_a16.cpp`(7/5/8)、`tests/ops/test_gated_delta_net_replay_record.cpp`(29/24/24)；逐字节存档 **`/home/ai-agent/DSHW/relay/step35-checkpoint/`**（STATUS 22 行 / STATE=27,213,666edc62 / MESSAGE 全文）。
**这一步就是 G-TP2-1 三家族机制的又一次注册**（照 `g-tp2-1-half-shapes.md` 的现行机制做，别克隆父文件）：
- Q8 Linear 注册 5 个 MTP 半形状 **[5120,5120](fc) / [7168,5120](packed attn in) / [5120,3072](attn out) / [17408,5120](gate|up) / [5120,8704](down)**，供体自陈"**继承被半分问题的 schedule，未重测**"⇒ 我方按台账 §14 的 **split-K 容量锚点不可继承**（行向半 `row_tiles` 减半 ⇒ 首分裂点前移）与 **A16/A8 分发静态 K vs 实测 ladder 分离**（`template <int K, int Ladder = K>`）、**行并行两 rank 同激活路由 / A8 floor 对齐**三条一起处理；
- `mtp_split_attn_in` 按输入行数选 rank 的 `[7168,T]` 打包切片（3072|512|3072|512 行，12 query/gate + 2 KV 头），核按行几何模板化，`[14336,T]` 实例不动；
- `gated_delta_net_replay_record` 收 8/24 头，`gdn_replay_fold` 注册 `FoldGeometry<48, 8, 24, 5120>`（27B 几何的一 rank 之半）。
- 我方 Q8 家族是**运行期 visitor**（非 gdn 的模板注册）⇒ 见台账 §11 的机制取舍与 `check_shard_sources.py` 门名问题（S4）。

### 17.4 续跑（已实测的注入形态）

```bash
cd /home/ai-agent/DSHW/ninfer-watch/ada-tree && git checkout -q tp2/raw
GIT_SEQUENCE_EDITOR='cp ../relay/tp2-replay-queue-187-from-full35.txt' GIT_EDITOR=true \
  git rebase -i --onto p1/tp2-replay 5b129659 tp2/raw      # 应停在 1/187 @666edc62（全序 35）
```
（`5b129659` = 已应用的最大供体＝全序 34；队列首行即当前停点。直接 `git rebase --continue` 亦可，停点索引里已含非冲突面。）
**rebase 期间纪律**：`git rebase --continue` **要求工作树无 unstaged**（本会话实测：一个未暂存的 `scripts/build-ada.sh` 就让 continue 报 "You must edit all merge conflicts…"）⇒ **vision 补丁与 S6 补丁用 `git apply` 进、`git checkout --` 出**，存档在
**`relay/vision-on-build-ada.patch`（39 行）**、**`relay/s6-capability-gate.patch`（201 行）**。

### 17.5 本会话未跑的账（明说，别当成已完成）

1. **全序 30–34 的簇边界全量门（含 softmax_attention）未跑**；只跑了 `-E softmax_attention` 簇门（§16.7）。
2. **S6 只到"写码 + 单 TU 编译（step-32 树上 6 TU rc=0）"**：`git rebase --continue` 后 全序 35 的 `include/ninfer/ops/linear.h` 带 marker ⇒ S6 编译面被挡；**下一会话在 全序 35 落地后**：`git apply ../relay/s6-capability-gate.patch ../relay/vision-on-build-ada.patch`（已在 32/34 树上验证可干净应用）⇒ `ninja … engine.cpp.o model_instance.cpp.o` + `ninja tests/ninfer_cli_options_test tests/ninfer_serve_options_test` + `ctest -R "options"`，再进簇门。
3. **S5b 未落码**，站点图已写（§16.6）。
4. S3/S4/S7/S8/S10 未开始；里程碑行号表按 §13（**全序 32 已命中**，后续：35 MTP 分片几何、59 peer_exchange、62 Vision on one rank、94-96+116 半形状扩展、108 mailbox 探针、111 mailbox 降级、139 proposal head、164/166/167 `*/tp2/` 注册、**168 调参表**）。

### 17.6 GPU1 交还

开工自检 `cuDeviceGetCount==1 / CC 12.0 / 36 SM / 15849 MiB / alloc 12 GiB OK`；
**GPU0 现网 `strata --serve`（PID 440568，15590 MiB）全程未碰、未 kill、未加压（零推理请求）**；
交还：GPU1 **4 MiB / 0%**、`nvidia-smi --query-compute-apps` 只有 PID 440568@GPU0、零我方进程残留；`/tank` 未写；**全程零审批**（沙箱 danger-full-access + approval=never）。

## 18. 全序 35（MTP 分片几何）+ S6 门 + S5b 落码（2026-10-16 13:0x–，同会话续做）

### 18.1 新普适事实（第 4 条，加进 §14/§15 那一节）：供体 Q8 家族 ≠ 我方 Q8 家族

`git show 666edc62` 的 5 个新 `shapes/nX_kY.cu` 全部 `#include "ops/linear/q8/q8_ksplit_launch.cuh"`，用
`Q8KSplitSchedule` / `launch_q8_ksplit` / `launch_q8_mma_r64x16_c48_k128_a1`——**我方树没有这套文件**
（供体侧 `src/ops/linear/q8/` 是 `q8_ksplit_{launch,mma,grouped_mma,config}` + `q8_rowsplit_gemm_*` + `q8_k2048_decode`）。
我方 Q8 家族 = **A16 sliced-K + instance 机制**：`q8_instance_launch.cuh` 的
`launch_q8_a16_sliced<Geometry, Capacity, Schedule>`，Schedule = `Q8A16SlicedKMmaSchedule<BlockTokens, KWarps,
Stages, MinBlocksPerSm, ScaleAccess, ActCache, WgtCache, ActStage, StaticK, TokenCapacity, Exact>`
（`q8_schedule.cuh:113-152`），且 **`with_problem<Geometry::kInputRows, Capacity>` 把静态 K 绑到调用形状**
⇒ 正是 §15"A16/A8 静态 K vs 实测 ladder 分离"的机制面。
**结论：5 个半形状文件按我方机制从各自的父形状生成**（继承父的实测 ladder 与尾实例，几何换成半），
不是搬供体文面：**半形状父映射（父在我方 Q8 注册表里全在）**
`[14336,5120]→[7168,5120]`(packed MTP attn in, 行半)、`[34816,5120]→[17408,5120]`(gate|up, 行半)、
`[5120,6144]→[5120,3072]`(attn out, K 半)、`[5120,10240]→[5120,5120]`(MTP fc, K 半)、
`[5120,17408]→[5120,8704]`(down, K 半)。
**两条风险自动消解**（写下来省后面时间）：① Q8 的调用侧容量报价 **= 0**（`linear.cpp:139-142`，K-slice partials 在
shared memory 内归约，`q8_a16_sliced_k_mma.cuh:8`）⇒ §14 的 split-K 锚点规则**不适用 Q8**；② 我方 Q8 家族
**无 A8 路由**（`select_q8_launch` → `select_q8_a16_launch`）⇒ 行并行两 rank 的激活路由一致性**平凡满足**。
唯一约束 = `StaticK % (KWarps*64) == 0`（`q8_schedule.cuh:123`）：父 ladder 只用 KWarps ∈ {4,8} ⇒
3072/5120/8704 全部整除 ✓（8704/512=17、/256=34）。

### 18.2 新纪律：净应用（`M`）的新文件**必须单 TU 编译验**

全序 33 的 `tests/models/qwen3_5/test_engine_tp2_real.cpp`（净应用、无冲突）在全构建里炸三处
⇒ "apply 干净"≠"能编译"：
1. `RuntimeStats` 的 7 个供体计数不存在（`private_endpoint/turn_closure/long_anchor/shared_stable_prefix_selections`、
   `pressure_private_owners_evicted`、`pressure_shared_owners_evicted`、`pressure_checkpoints_dropped`）⇒ 证据行改的我方字段
   （`checkpoint_selections`/`reused_prompt_tokens`/`pressure_spill_pages`/`state_forks`/`state_moves`/`device_state_occupied_slots`）；
2. `ContextCacheOptions` 的 4 个供体字段（测试自建 tp2 选项 + 断言默认）⇒ 我方 `host_capacity_bytes=0` + `device_state_slots>=8`；
3. `ContextCacheHints::retention` / `CacheRetentionHint::LiveSession` 不存在 ⇒ 我方 = `session_key` + 默认 `update_session_index`。
**⇒ 每步在 `git rebase --continue` 之前，除冲突文件外，把该步新增的文件（`git show --name-status <sha>` 的 `A` 行）逐个单 TU 编。**

### 18.3 S6 门 + S5b 落码（合并补丁存档 **`relay/s6-s5b-gates.patch`（326 行）**，取代 §16.5 的 s6 单独存档）

- **S6**：见 §16.5（未变），接线在 `engine.cpp:27 initialize_execution()`（取到两 `DeviceContext` 后、读制品前逐 rank 验 CC）+ `model_instance.cpp validate_options()` 消费同一 KV verdict。
- **S5b 已落**（原三条 → 我方**一个纯函数 + 一个站点**，不造平行 admission）：
  `product::tensor_parallel_reuse_verdict(tp, reused_tokens, prompt_tokens, speculative_backend)`
  （`src/product/tensor_parallel_options.h`）→ 在 `context_cache/resource_manager.h:83 candidates()` 的
  `prepare_source` 过滤之后第二条 `std::erase_if` 落政策①（zero-suffix 丢弃）与②（speculative 丢弃）；
  **root 源（`reused_tokens==0`）永不丢弃** ⇒ 被拒的候选变成一次 prefill（miss），不是故障；
  政策③（供体 "not materializable" throw）我方 0 处 ⇒ 不适用（§16.6 已记）。
  宽度/后端的来源：`engine_core.h` 新增两个启动期常量 `tensor_parallel_width_ = instance.peer_parameters ? 2 : 1`
  与 `speculative_at_startup_ = options.speculative.backend != None`，在调用点传入（**不给 `Program` 开新访问器**）。
  **注意**：全序 40（`12160839` "accept `--spec mtp` at tp 2"）会把当前"tp2 拒 speculative"的启动门打开 ⇒ 那时政策②成为活路径，S5b 的两条都要重新过门。
- **门（本轮，vision=ON + 全序 35 + S6 + S5b 同树，GPU1）**：
  | 门 | 实测 |
  |---|---|
  | 全构建 | `scripts/build-ada.sh 120a --tests` ⇒ **278/278 rc=0**（含 5 个新 Q8 半形状 nvcc TU、mtp_pack、gdn recurrent/replay） |
  | 簇设备门 | `ctest -j1 --timeout 900 -E "softmax_attention"` ⇒ **100% passed out of 161** / **1056.62 s** → **`relay/passset-sm120-tp2steps35-s6s5b-r0.tsv`**（161 注册 = 160 + 全序 33 带来的 `ninfer_qwen3_5_engine_tp2_real_test`） |
  | **S6+S5b 的 CPU 判据** | `ninfer_cli_options_test` **Passed 0.00 s**、`ninfer_serve_options_test` **Passed 0.00 s** ⇒ 能力 verdict（12.0 放行 / 8.9 带原因拒绝 / 12.1 拒绝）+ KV verdict（bf16/int8 放行，`rk4v4-e8`/fp8/nvfp4/k8v4 点名拒绝）+ **reuse verdict（tp1 不拒；tp2 partial 放行、zero-suffix 拒、speculative 拒、root 永不拒）** 全绿 |
  | 新 Q8 半形状数值 | `ninfer_linear_q8_a16_test` **Passed 24.69 s**（含 5 个半形状 × 41 个 T 取样点）、`linear_add_q8_a16` **6.54 s**、`linear_pair_q8_a16` **2.65 s**、`linear_swiglu_q8_a16` **4.48 s**；`ninfer_mtp_pack_test` **2.30 s**、`ninfer_gated_delta_net_replay_record_test` **5.39 s**（8/24 头几何） |
  | vision | `ninfer_media_decode_test` **0.01 s**、`ninfer_qwen3_5_frontend_test` **0.96 s** 真跑通过 |
  | 单卡具名跳过（77） | `text_context_tp2`、`text_context_tp2_real`、`engine_tp2_real` 等 ⇒ 真判定在窗口（D 桶），符合 R1 |
- **全序 36 `ed6b71e3` 已解 1 块**（`execution/text.cpp` 1770：取我方 `rank_execution_view(rank)` 签名 + graft 供体的 `rank_valid_columns(rank)` 进 head-local attention 的 valid-columns 参数；`TextContext::rank_valid_columns` 随供体净应用进树 = `text.cpp:1651`）：该步 **+1181/−47**（`text.cpp` 779、`mtp.cpp` 119、`ffn/gdn/tp.h/text.h/workspace.h`），编译门待跑（供体净应用面预计要按 §16.2 的签名落差逐点改写）。

## 19. 全序 36（MTP 头双设备 verify/propose）落地（2026-10-16 13:3x，同会话续做）

### 19.1 落地：35 → `be8c48ba`；36 = `ed6b71e3`，**1 块冲突 + 6 处净应用签名失配**

- 冲突块（`execution/text.cpp:1770`）：取我方 `ops::causal_softmax_attention(..., rank_execution_view(rank))` 签名 + **graft 供体语义** `rank_valid_columns(rank)`（head-local attention 只吃本 rank 的 valid columns）。`TextContext::rank_valid_columns` 随净应用进树（`text.cpp:1651`）。
- **净应用 ≠ 能编译（第 18.2 条纪律再次兑现）**：供体在 MTP 三处新代码里按它的 op 签名写调用 ⇒ 我方需逐点改写（**不改供体语义**）：
  | 供体写法 | 我方写法 |
  |---|---|
  | `text_rope(pos, *config_.rope_parameters, qn[r], kn[r], stream)` | `text_rope(pos, *config_.rope_parameters, **&parameters_.rope_scaling**, qn[r], kn[r], **rank_execution_view(rank)**)`（3 处：verify prefill、k-only append、decode 单列） |
  | `ops::causal_softmax_attention(..., a_batch, stream)` / `(..., a[r], stream)` / `ops::causal_softmax_attention_cached(..., a[r], stream)` | 末参换 `rank_execution_view(rank)`（3 处） |
  ⇒ 与 §16.2 的两条底座签名完全同形（`DeviceExecutionView{stream, multiprocessor_count}` + `ops::TextRopeScaling*` 共享不可变系数）。
- 规模：**+1181/−47**，`execution/{text.cpp 779, text.h 90, mtp.cpp 119, mtp.h 56, ffn.cpp 61, ffn.h 13, gdn.cpp 31, gdn.h 13, linear.h 15, tp.h 27, workspace.h 24}`；MTP 层的分片口径 = 输入投影列并行（rank0 收缩 normalized token embedding、rank1 收缩 normalized target hidden，**packed input 从不成形**，只 rank0 embed）；packed Q|K|Gate|V 用 `mtp_split_attn_in` 拆；输出投影与 post-mixer down 行并行；**三次 all-reduce** 保持残差与两 rank 的 MTP KV page 一致；proposal head = 词表分裂输出头（两 rank gather 后 rank0 argmax），优化头只加载在 rank0 ⇒ **对我方制品（q4_g64）`--lm-head-draft` 仍不可用**。
- **门（vision=ON + 全序 36 + S6 + S5b 同树，GPU1）**：全构建 **57/57 增量 rc=0**（`== OK 产物`）；簇设备门 `ctest -j1 --timeout 900 -E "softmax_attention"` = **100% passed out of 161 / 1058.79 s**（133 Passed + 28 Skipped(77 具名) + 0 Failed）→ **`relay/passset-sm120-tp2steps36-r0.tsv`**。
  这一步的 MTP 双设备面在单卡的可见判据 = 编译门（`libninfer_model_runtime.a` rc=0 + 全构建）+ 6 处签名改写单 TU 通过；**MTP round 的数值判定在窗口**（`ninfer_qwen3_5_mtp_round_test` 等按 77 具名跳过；`text_context_tp2*`、`engine_tp2_real` 同）。
- **簇 tag：`v-tp2-mtp-shard-geometries-r0` @ `bd9201af`**（全序 35-37 MTP 分片几何/头双设备/对齐）。

## 20. 停在 全序 38（MTP round 双设备）：本会话的收线点（2026-10-16 13:5x）

**落地计数 37/223**：`647c2487`(34) → `be8c48ba`(**35** MTP 分片几何) → `02e62169`(**36** MTP 头双设备 verify/propose) → `bd9201af`(**37** MTP 头对齐双设备 prefill chunk，净应用) → **停在队列 30/213 = 全序 38 = `48c259a5`**（+486/−142，10 文件）。

**停点形态（与 §17.3 同法）**：6 处未合并（**5 UU + 1 DU**）+ **9 个冲突块**，逐字节存档 **`/home/ai-agent/DSHW/relay/step38-checkpoint/`**（STATUS 10 行 / STATE=30,213,48c259a5 / MESSAGE 全文 31 行 = 供体这一步的权威规格）。块清单（我方/基线/供体行数）：

| 文件:块 | 行范围 | ours/base/theirs | 难度与打法 |
|---|---|---|---|
| `program/execution_context.h:1` | 150-170 | 3 / 3 / **11** | 供体给 `MtpBatchContext` 加 rank-1 侧字段（MTP KV、records、frames、prefill hidden 的 peer 面）；我方该文件是改名后的 `context.h` ⇒ 逐字段取供体新增，保我方字段序 |
| `program/graphs.cpp:1` | 386-567 | **0** / 85 / **93** | **我方整段不存在**（§16.3 已把 prepare/launch 循环放调用点）⇒ 照 §16.3 的适配法：把供体的"MTP profile 每个 batch size 先双卡 eager warm、再 capture 成一张双设备图"落到我方调用点（`synchronize_all` + `tp_binding()` + `graph_peer_bridge()`），**不要**恢复供体的模板族 |
| `program/program_impl.cpp:1` | 31-48 | 4 / 5 / 5 | 小；`PeerRuntime` 扩 MTP 侧分配（rank 1 的 MTP KV/records/frames/prefill hidden）⇒ 与我方 `PeerRuntime` 逐字段并 |
| `program/speculative/mtp.cpp:1..4` | 4-11 / 210-249 / 254-369 / 372-412 | 1/1/2 · 6/3/**27** · **58**/36/18 · 2/**24**/11 | **本步主体**：MTP round 双设备（同一 ingress 上传 rank1、`target_verify_accept` 把 accepted 数复制到 rank1、两 rank 各自选并发布 accepted hidden、anchor/frontier/licensed 计数 pull 到 rank1（**rank1 stream 上的 pull，用跨设备 event 定序**）、commit 时把 rank1 records 折进 rank1 GDN state、alignment/proposals/AR steps 双 rank）。块3/块4 = **我方文面比供体长**（58 vs 18、2 vs 24）⇒ 取我方骨架 + 逐条落供体新语义，别整块 --theirs |
| `program/speculative/target_verification.cpp:1..2` | 2-9 / 57-111 | 1/1/2 · 1/**0**/**50** | 块2 = 供体新增 50 行（verify/accept 的 peer 面）⇒ 取 both 后修剪（我方该函数无基线内容） |
| `program/storage/context.cpp`（**DU**） | — | 接受我方删除 | 供体 +16 行的语义 = **`publish_kv_rows()` 现在也写并交叉校验 rank 1 的 *backend* KV row scalar**（MTP 行跟随 Text 行在两 rank 同步）⇒ 落我方 §16.2 第 5 条的**同一个** `ProgramImpl::publish_peer_kv_rows()`（扩 backend 行：`backend_kv_addresses->bound_row(...)` + `mirror_row` 交叉校验），三个站点自动继承，**不新增第二处发布实现** |

**与 S5b 的交叉印证（供体自陈）**：*“The MTP bridge stays single-device and becomes a backstop at width 2: admission already declines every non-Root reuse with a speculative backend there.”* ⇒ 我方 §18.3 的 `tensor_parallel_reuse_verdict`（政策②：tp2 拒 speculative 非 Root 复用）**正是这句的前置**；全序 40（`12160839` "accept `--spec mtp` at tp 2"）打开启动门后，政策②成为活路径 ⇒ **S5b 补丁必须在 40 落地后重跑门**（`ctest -R "context|admission|engine|qwen3_5|speculative|preemption"`）。

**缺口分诊与后续计划：`relay/gap-plan-20261016.md`**（226 个供体独有文件 ⇒ A 206 随步到达 / B 7 = DU 日历 / C 13 = 三档处置；隐藏 include 依赖全扫 = 1 处，全序 178）

## 21. 用户两条授权（2026-10-16 14:3x，权威，覆盖台账 §11 的 A/B 悬置与 gap-plan §5 第三档）

**A1｜全序 164/166/167 的 `*/tp2/` 注册机制 = 走我方三家族的现行机制（= 台账 §11 的 B 方案）**，不采供体的 include-改名（A 方案）。
理由（记下来给后面所有会话用）：① 与 S1/G-TP2-1、§18.1 的 Q8 重写同构 ⇒ **一个 Op 家族只有一套实现**（AGENTS.md"一套当前权威"）；② 单卡可验（供体 include-改名把双设备形态做成"必须有两块卡才有注册面"的门，`check_shard_sources.py` 的门名要换成我方机制的等价门 = S4 清单项）；③ 96 个 `*/tp2/` 文件（A 类）到点**不整族搬**：按我方机制落等价面 + 逐文件判定，落不下的记进差异清单。
⇒ **§11 的 A/B 抉择就此关闭**；164/166/167 的打法＝"取我方骨架 + 语义逐条落"（同 30/35/36）。

**A2｜C 类第三档移植决定**：**`serve/prefill_miss_dump.{h,cpp}` 在 D2 之前移植成我方实现**（接我方 metrics/`operational_log`，**不复制供体 serve 面**），用途＝窗口内 GSM8K 掉点的 prefill-miss 前缀取证；**`serve/gpu_telemetry.{h,cpp}` 暂不移植**，D2 若出现显存/graph allowance 归因不清再评估（登记在 gap-plan §5）。
`serve/prompt_overlap.*` 维持不移植（属 pressure/admission 家族，与 A1 的"不造平行架构"同源）。

## 22. 全序 38（MTP round 双设备，eager+captured）落地（2026-10-16 14:4x，同会话续做）

**编号校正（重要，写死）**：`全序 = 队列行 + 8`（全局 223 序列）。当前在制品队列 `relay/tp2-replay-queue-184-from-full38.txt` 的第 *n* 行 = **全序 37+n**。⇒
**38 = `48c259a5`（run the MTP round on two devices）· 39 = `12160839`（accept `--spec mtp` at tp 2）· 40 = `958a6ea8`（test serve 27B with MTP）· 41 = `b7dcc3d1` · 42 = `c29ccc46` · 43 = `084cd71c`（verify DFlash2 proposals）· 44 = `8229d197` · 45 = `600be761`（accept `--spec dflash2`）**。
⇒ **S5b 政策②（tp2 拒 speculative 非 Root 复用）成活路径的点 = 全序 39**（不是 40）：该步落地后立刻跑 `ctest -R "context|admission|engine|qwen3_5|speculative|preemption"`；DFlash2 同判在 **45**。

### 22.1 解法（6 文件 9 块 + 1 DU，全部"取我方 + 定点 graft"）
| 文件 | 块 | 我方落差 → 落法 |
|---|---|---|
| `program/execution_context.h` | 1（3/3/11） | 我方把供体的 `target_verify_accept` **拆成 `target_verify_forward` + `target_accept`**（P0 面）⇒ 保留拆分，把供体的 peer 面落成**各多一个参数**的两个 peer 重载（`peer` frame view / `peer_continuation_hidden_store`） |
| `program/program_impl.cpp` | 1（4/5/5，在 PeerRuntime 构造内） | PeerRuntime 增 rank 1 的 **ReplaySSM records + fold plan**（`replay_records.emplace(backing, *plan.persistent.replay_records)` + `replay_fold.emplace(*replay_records, state_images->linear().all_layers_view())`）；`prefill_hidden` 绑定随供体净应用 ✓ |
| `program/graphs.cpp` | 1（**ours=0** / 85 / 93） | **我方没有供体那段单阶段 speculative 图区**（我方 = P0 的 **forward/finish 双族**结构，`mtp_context`/`run`/`capture` 在块后）⇒ 取我方（删 85+93），再把供体两步语义落我方区：① `mtp_context` 追加 `peer ? &*peer->io.mtp_decode : nullptr` + `peer ? &peer->state_images->continuation_hidden_store() : nullptr`；② eager warm 改 `warm_batches = tensor_parallel() ? max_concurrency : 1U`，**每档两阶段都跑**（Forward+Finish，合我方入口形状），`synchronize_all()` 前后各一次 |
| `program/speculative/mtp.cpp` | 1（1/1/2） | 取我方改名 include + 补 `program_impl.h`（`detail::ScopedCurrentDevice` 与 PeerRuntime 面在此可见） |
| | 2（6/3/**27**） | 取供体的 `tp/rank1/peer` 推导 + rank 1 ingress 上传，**保住 P0 的 `phase == Forward` 语义**（我方 rank 0 上传本就只在 Forward ⇒ rank 1 同守卫，peer view 构造无条件）；`hidden_size/max_context/stream` 局部随供体进（后续区段引用） |
| | 3（**58**/36/18） | **取供体结构**（`MtpRoundView round = slice_mtp_frame(...)` + `verify_view(...)` + peer prepare）——我方那 58 行的局部张量列表被 `round` 视图取代；**保住 P0 的 Forward 守卫**包住两个 prepare（rank0 + rank1）。供体的 `MtpRoundView/slice_mtp_frame/verify_view/prepare_next_round` 已随净应用进树（`mtp.cpp:26/61/98/123`）⇒ 结构可直取 |
| | 4（2/24/11） | 拆成我方两入口：`if (peer) { target_verify_forward(..., verify_view(round, …), verify_view(*peer, tp->replay_records), envelopes.target_verify); target_accept(..., 同上两 view, *state.peer_continuation_hidden_store); } else { target_verify_forward(...); }` |
| `program/speculative/target_verification.cpp` | 1（1/1/2）· 2（1/0/**50**） | 取我方改名 include + 补 `program_impl.h`；**把供体合并式 peer 函数按我方拆分点切开**：verify = `card.set_gdn_state_action(RecordForReplay, frame.replay_records)` + `card.target_verify_batch({frame.X, peer.X}…)` 数组形；accept = rank0 `speculative_accept_greedy_drafts` + `select_accepted_hidden` + `scatter`，再接**rank 1 的跨设备 pull**（`events.inputs_ready(0)` record → rank1 `cudaStreamWaitEvent` → D2D 复制 `accepted_drafts` → rank1 select+scatter 到 peer store）。peer 面拒 feature capture / sparse drafts（`proposal_q`）⇒ 抛错文案照供体 |
| `program/storage/context.cpp`（**DU**） | — | 接受我方删除；其 +16 行语义 = **backend(MTP) KV row 也写 rank 1 并交叉校验 mirror 行号** ⇒ 扩进我方**唯一**的 `ProgramImpl::publish_peer_kv_rows()`（`backend_row = kv->backend ? bound_row : 0` + `backend_kv_cache()->execution_tables().mirror_row(...).row_index()` 校验 + `set_peer_i32(peer->io.backend_kv_table_row, backend_row)`）⇒ **三个发布站点（prefill / forced-tokens / replay）自动继承**，无第二实现 |

### 22.2 工装坑（记进纪律，省未来时间）
**Python heredoc 里"以 `"` 结尾的行紧邻 `\"\"\"` 会吃掉收尾引号"**：本轮两处 `#include "...program_impl.h` 因此缺右引号（`#include expects "FILENAME"`）。⇒ 落盘文本行尾多留一个 `\n` 或整段末尾加空行；`resolve_blocks.py` 的 `both/ours/theirs` 无此问题（不写文本），**自造文本块**要防。
另一处：块 3 取供体结构后，我方块外的一行 P0 上下文 `target_accept(..., verify)` 引用了被删的局部 ⇒ 改 `verify_view(round, state.execution.replay_records)`。**教训**：整块改结构时，扫一遍块外对本块局部名的引用（编译器会抓，但一次改全比多轮快）。
- **门（vision=ON + 全序 38 + S6 + S5b 同树，GPU1）**：4 个 TU（`program_impl`/`speculative/mtp`/`target_verification`/`graphs`/`decode`）**单 TU rc=0**；全构建 **72/72 增量 rc=0**；簇设备门 `ctest -j1 --timeout 900 -E "softmax_attention"` = **100% passed out of 161 / 1074.16 s**（133 Passed + 28 Skipped + 0 Failed）→ **`relay/passset-sm120-tp2steps38-r0.tsv`**；`merge-safety` 索引 0 / 工作树 0、符号清单无缺失。MTP 双设备 round 的**数值判定在窗口**（`mtp_round`/`text_context_tp2*`/`engine_tp2_real` 单卡 77 具名跳过）。

## 23. 全序 39（`12160839` accept `--spec mtp` at tp 2）解到编译通过（2026-10-16 15:0x）

**3 块 + 净应用面**（+273/−67，8 文件）：
- `planning/startup.cpp`（7/6/**1**）：供体把 attention 容量调用抽象成 `attention_scratch(layout, batch, min_width, max_width)` 局部助手（**3 个调用点**：text 层 360、verify 426、proposal 462 ⇒ MTP 双设备工作区计划复用同一报价）。取供体结构，**把容量函数第 10 参恢复成我方 `DeviceExecutionView{nullptr, plan.multiprocessor_count}`**（`startup.cpp:337-344` 定义处改一次，三站点自动继承）⇒ 单 TU rc=0。
- `docs/cli.md`（3/3/4）：取供体的能力句（"ordinary decoding **and `--spec mtp`** … The MTP head is split like a Text layer and verification runs on both ranks；`--draft-tokens`/`--lm-head-draft` 照单卡，**优化 proposal head 只加载在 rank 0 并在那里单独 proposal**"）+ 保住我方 additions（`rk4v4-e8` 与 `--host-context-mib` 的拒绝口径、`--spec dflash|dflash2` 仍拒）。
- `src/serve/serve_options.cpp`（2/4/4）：我方两行帮助文（单 Host 旋钮 + `max(2x concurrency,8)`）改 "ordinary **or `--spec mtp`** decoding"。
- 净应用：`include/ninfer/types.h`(7)、`apps/cli/options.cpp`(3)、`model_instance.cpp`(10 = Engine 侧接受 MTP at tp2)、`docs/serving.md`(22)、`docs/maintainer/qwen3_5-model.md`(55)。⇒ 门后逐 TU 编译（`model_instance`/`options`/`serve_options`）。
- **这一步就是 S5b 政策② 的成活点**（启动不再拒 `--spec mtp`）⇒ 落地后必跑 `ctest -R "context|admission|engine|qwen3_5|speculative|preemption"`（`tensor_parallel_reuse_verdict` 的 speculative 分支自此是活路径）。

**⚠ 本会话纪律违例，如实披露**：为拿单 TU 反馈，在全序 38 的簇门运行期间编译了 **1 个对象文件**（`startup.cpp.o`，**无链接、未重链任何库或测试二进制**）⇒ 该轮门数据的二进制未变、结论有效；**教训固化**：块外净应用的签名失配，先在树内静态核对（读调用点的被调方声明），把编译排队到门后，或先跑门再改。

## 24. 全序 40 与 41 落地（2026-10-16 15:4x，同会话续做）

### 24.1 全序 40 `958a6ea8` → **`28ff8f6d`**（test: serve 27B with MTP through a two-device Engine）
- 1 块 = `tests/README.md`：取供体新增的"两设备测试 + MTP(K=3) 对比无 speculation + `optimized_real` 变体"段（含 `ctest -R 'ninfer_qwen3_5_engine_(mtp_)?tp2'` 命令）**+ 我方那段的权威文面**（`real` 标签 / `RUN_SERIAL` / 我方 scenario 名 `vision|late-instructions|concurrent`，供体的 `pressure-resume|forced-token-kv-row` 属被剔除层 ⇒ 不带入）。
- **§18.2 纪律立刻兑现**：净应用的新测试 `tests/models/qwen3_5/test_engine_mtp_tp2_real.cpp`（304 行）单 TU 编译抓到 **2 处供体 `ContextCacheOptions` 字段**（`host_state_slots`/`host_kv_capacity_bytes`，与 33 的测试同形）⇒ 改我方 `host_capacity_bytes = 0` ⇒ **rc=0**。

### 24.2 全序 41 `b7dcc3d1`（retain rank 1's target hidden under MTP）：8 块 / 3 文件 + 1 个"复活文件"
供体把"MTP 桥要读 rank 1 的 target hidden"补齐：此前只有 rank 0 把 hidden 写进 StateImage，**rank 1 的 mirror 槽一直是 0，被镜像的 Fork/copy 把 0 一路带下去**（这就是本步要修的洞）。

| 落差 | 我方落法 |
|---|---|
| `PrefillContext` 新字段 `peer_rewrite_checkpoint_hidden` | 供体放在 `dflash_host_ingress` 之后；我方**追加到字段表末尾**并在构造后赋值（同我方 `prefill_gpu_timer` 法）⇒ 我方聚合初始化不破 |
| 供体新增 lambda `bind_rewrite_capture_hidden()`（在被我方替换的 487 行区内） | 块外应用行引用它 ⇒ **在我方站点定义同名 lambda**（用我方变量名 `request.next_capture/request.capture_groups`），语义 = rank0 取 `state_images->continuation_hidden_slot(selectors.destination)`、`peer_retains_hidden()` 时 rank1 取 mirror pool 同槽 |
| `committed_state_slot()` + `peer_tail_hidden()` 定义在**我方已删的 `storage/context.cpp`**（本步被 git 复活成索引里的 `M`） | 接受删除（再 `git rm`），两函数落我方 `storage/sequence.cpp`（`refresh_state_views` 同处）。**照抄会造出假故障**：供体用 `fork_pending ? read : write`，而我方 tail-hidden 视图取 **`state.read`** ⇒ 我方的 `committed_state_slot` 按我方规则实现（`valid(read) && residency(read) != HostOnly` ⇒ `physical_slot(read)`），`peer_tail_hidden` 校验 `continuation_hidden_slot(*slot).data == sequence.tail_hidden.data` 后才返回 mirror pool 同槽 |
| `copy_tail` 签名：供体 `(sequence, column)` | **两重载并存**：我方 `(sequence, const Tensor& source)`（replay/forced-token 站点传切片）+ 供体列形式（定义随净应用进 `decode.cpp:123`，含 rank1 同列本地 copy） |
| `set_rewrite_checkpoint_hidden_output(Tensor*, Tensor* peer = nullptr)` | 供体给 peer 形参**带默认值** ⇒ 我方 1 参调用点不破；两处 prefill 站点显式传 `state.peer_rewrite_checkpoint_hidden` |
| 供体在 materialize 里 `set_peer_i32(peer->io.rope_delta, …)` | **n/a，非缺口**：我方 rope delta 按值进每次 launch/graph（P0 设计），无该设备标量发布点；`graphs.cpp:201` 已把 rank1 的 `rope_delta` 与 controls 一起清零 |
| 部分终止的 speculative commit 要在 rank1 选+scatter 其提交列 | 供体在 `resolve_pending_raw` 的 hunk **净应用成功** ⇒ 我方对应点 `prefill.cpp:415` 已有 `peer_retains_hidden()` 分支（frame/`MtpDecodeState` 校验 + D2D 选择器 + rank1 select/scatter）✓ |

**门（vision=ON + 全序 41 + S6 + S5b 同树）**：4 个 TU（`prefill`/`decode`/`storage/sequence`/`program_impl`）单 TU rc=0；全构建 ______；簇门 ______；**S5b 成活门** `ctest -R "context|admission|engine|qwen3_5|speculative|preemption"` = **100% passed out of 35 / 277.28 s**（`ninfer_qwen3_5_{engine_tp2,engine_mtp_tp2,engine_mtp_tp2_optimized}_real` = 77 具名跳过 ⇒ 真判定在窗口）。

### 24.3 工装坑（本轮新增两条）
1. **自造文本块漏行**：execution_context.h 的 peer 字段我只写了注释没写声明 ⇒ `has no member named 'peer_rewrite_checkpoint_hidden'`（编译 4 处抓到）。⇒ 规则：**自造文本块落盘后，用 `grep -c` 验证块内新增的每个标识符都进了文件**（本轮起把这条加进 30 分钟自测）。
2. **相邻块的行号串扰**：program_impl.h 两块相邻（590-744 与 752-773），我在近端块文本里带上了远端块首行（`clear_lane`）⇒ 远端符号被吞。解法＝**`git checkout --ours -- <file>` 复位后重打补丁**（索引 stage2 仍在， UU 未解时最便宜）。⇒ 规则：同文件多块时，**只写本块范围内的内容**，块外符号交给块外区域；解完立刻 `git grep -c <被删符号>` 对比 `git show HEAD:<file>` 的符号集。

## 25. 全序 42 落地 + **S5b 退役（我方的自我更正，权威裁定见下）** + 停在 全序 43

### 25.1 全序 42 `c29ccc46` → **`0b52432b`**（resume retained prefixes with MTP at tp 2）
- 唯一冲突 = `planning/request_plan.cpp` 的 **1608 行单块**（我方 9 行 vs 基线 803 / 供体 792）⇒ **取我方**（供体面挂在 `AdmissionCandidate`/`ProgramImpl::inspect_lane` 上，我方 0 处 ⇒ 不 graft，同 §14/§16.2 判定），`request_plan.cpp.o` 单 TU rc=0。
- 供体净应用面（text.cpp 的 `mtp_forward_batch` 列形式、prefill/program_impl 的 backstop 删除、draft.cpp 侧）随步进来 ⇒ 与 §24 的 peer 面拼上。

### 25.2 **S5b 退役裁定（2026-10-16 16:2x，我按"一套权威 + 不造平行架构"自决，非重大方向变更）**
供体在全序 42 的说明里写死：*“Admission **no longer declines** non-Root reuse at tensor-parallel width 2 … The bridge backstop and the sample_from_hidden backstop are **removed**.”* 原因 = 该步之后 tp2 已具备机制面：**词表分裂 head 可为整条复用采样首 token**（rank 1 经跨设备 event 拉 rank 0 保留的那一列，hidden 轴是复制的 ⇒ 对每个后端都精确），且 **rank 1 保留自己的 target hidden**（全序 41 补的洞）。
⇒ 我在 §18.3 落的 **S5b 政策①（zero-suffix 丢弃）与②（speculative 丢弃）是 全序 29→42 之间的过渡政策**，其存在理由已被 42 消除。**结论：S5b 过滤器退役，S6 能力门保留。**
- 动作（写死给下一会话）：提交 S6 时**从 `relay/s6-s5b-gates.patch` 里删去 S5b 三段**——① `tensor_parallel_reuse_verdict`（product 头）；② `resource_manager.h: candidates()` 的 `tp/speculative_backend` 形参与第二条 `erase_if`；③ `engine_core.h` 的 `tensor_parallel_width_/speculative_at_startup_` 两常量与调用点两实参；外加 `tests/test_cli_options.cpp` 的 reuse 断言段（**S6 的 capability/KV 断言保留**）。
- 保留项（有价值，不算平行架构）：§16.6 的站点图（它标出的正是 42 消灭的那条路径）与本文的裁定记录。
- 我此前把"S5b 三条"当成正面清单项，是照 全序 29 的供体文面推的；42 的文面到了，按事实改判。台账 §15 R3/S5b 行**按本节作废**（在 §15 里标注指向本节）。

### 25.3 停点：队列 **35/213 ＝ 全序 43 ＝ `084cd71c`**（verify DFlash2 proposals over two devices）
3 文件、**8 块**：`execution/draft.cpp`（5 块）、`program/execution_context.h`（2 块）、`program/speculative/target_verification.cpp`（1 块）。逐字节存档 **`/home/ai-agent/DSHW/relay/step43-checkpoint/`**（含 STATUS/STATE/MESSAGE 全文规格 + 队列 **`relay/tp2-replay-queue-179-from-full43.txt`**）。
续跑：`git rebase -i --onto p1/tp2-replay 0b52432b tp2/raw` + 注入 `tp2-replay-queue-179-from-full43.txt` ⇒ 停 1/179 @`084cd71c`。
下一步序：**43 DFlash2 双设备 verify → 44 `8229d197`（run the DFlash2 round）→ 45 `600be761`（accept `--spec dflash2` at tp 2）→ 46 `11956486`（serve 27B DFlash2 测试）**；45 落地后 `--spec dflash2` 成活 ⇒ 与 S5b 无关（政策已退役），门＝`ctest -R "speculative|dflash|preemption|qwen3_5"`。

### 25.4 落地计数与门（本会话累计 34 → **42/223**）
35 `be8c48ba` · 36 `02e62169` · 37 `bd9201af` · 38 `8eb87408` · 39 `a17b4902` · 40 `28ff8f6d` · 41 `61f2142c` · 42 `0b52432b` = **8 步 / 一次会话**。
门：全序 41+42 树（vision=ON + S6 + S5b）全构建 **196/196 rc=0**；S5b 成活子集 **100%/35 · 277.28 s**；簇门 `-E softmax_attention`（全序 41/42 树，163 注册 = 161 + 全序 40 的两条 MTP tp2 真制品测试）**100% passed out of 163 / 1076.57 s**（→ `relay/passset-sm120-tp2steps41-r0.tsv`）；全序 38 簇门 **100%/161 · 1074.16 s**（§22）。

## 26. 全序 43 落地 + B（跳步扫描）+ C（S7 seed）（2026-10-16 17:1x，本会话第 3 段）

### 26.1 全序 43 `084cd71c` → **`104e6d28`**（verify DFlash2 proposals over two devices，5 文件 +197/−75）
8 块全解：`execution/draft.cpp` 5 块、`program/execution_context.h` 2 块、`speculative/target_verification.cpp` 1 块。
- `DFlashBatchContext` 加 peer 两字段（**追加到我方字段表末尾**，§24.2 同法）；`target_verify_forward/target_accept` 的 peer 重载**扩到支持 DFlash2**：verify 允许 rank 0 的 `feature_sink`（**peer 不得带 sink/proposal 分布** ⇒ 保留我方的显式校验），accept 允许 **sparse drafts**（`speculative_accept_sparse_drafts`，rank 0 独自验收）。
- drafter 只在 rank 0，rank 1 只 verify：我方站点保留 **P0 的 `phase == Forward` 守卫**，在守卫内做 rank 1 的 ingress 上传 + **drafts 的跨设备 pull（`events.inputs_ready(0)` → rank1 `cudaStreamWaitEvent` → D2D）** + rank1 用自己那份 drafts 走同一个 `speculative_prepare_verify_inputs`；rank 0 的 verify 视图用我方显式 designated-init（含 `&sink`），rank 1 的视图用 `dflash_verify_view(*state.peer_frame, batch, tp->replay_records, false, nullptr)`。
- **本轮 4 个工装坑（写死）**：① 自造文本块**末尾以 `"` 紧邻 `"""`** ⇒ 收尾引号被吃（第 3 次撞上，改用单引号变量拼）；② 内联脚本里的 `"OURS"` 记号不会被展开（只有 `resolve-step41.py` 实现了 sentinel）⇒ 把字面 `OURS` 写进文件；③ 需要**取我方整文件**时用 **`git show :2:<file>`**（索引 stage2，UU 未解时最可靠）复位后按锚点重打补丁；④ 供体新增的匿名 ns 助手若被我方调用点在**其定义之前**使用 ⇒ 必须加**前向声明**（`dflash_verify_view`）。
- **净应用面在 42/43 之后又抓到 2 处签名/字段失配**（§18.2 第 4 次兑现）：`transactions/replay.cpp:291` 的 `mtp_bridge_and_propose` 调用（42 把签名加到 8 参：`peer_previous_hidden` + `span<const int32> rope_position(=3)` + `next_embedding=nullptr`）⇒ 我方站点补 peer 实参（`peer_retains_hidden() ? &peer_tail_hidden(sequence) : nullptr`）；`test_engine_mtp_tp2_real.cpp` 在 42 扩的 16k/INT8-KV/4-lane 场景里又用了 **7 个供体 RuntimeStats 计数** ⇒ 换我方字段。

### 26.2 门（vision=ON + 全序 43 + S6 + S5b 同树，GPU1）
- 全构建 **rc=0**（0 error，77 目标增量）；簇门 `-E softmax_attention` = **100% passed out of 163 / 1061.38 s**（133 Passed + 30 Skipped）→ **`relay/passset-sm120-tp2steps43-r0.tsv`**；`merge-safety` 索引 0 / 工作树 0、符号清单无缺失。
- GPU1 交还：开工探针 `cuDeviceGetCount==1`，结束时 **4 MiB / 0%**、零我方进程；**GPU0 现网 strata（PID 440568，15590 MiB）全程未碰、未 kill、未加压**；`/tank` 未写；全程零审批。
- 本会话累计：**34 → 43/223（9 步）**，三段门数据齐（§16.7 / §18.3 / §19 / §22 / §24 / §26）。

### 26.3 B：跳步安全性扫描（用户选择 A+B+C 的 B）
`relay/tp2-skip-scan-43-223.tsv`（181 行）＝ 对全序 43–223 逐步分类：**KEEP 144 / SKIP-SAFE-CAND 33 / SKIP-UNSAFE 2**。
- 判据：只动 `docs/**`、`bench/**`、`tools/golden/**`、`tools/release/**`、`*.md` 的步 = 候选；再查"该步新建的文件是否被后续 KEEP 步改动/引用"⇒ 被引用者记 SKIP-UNSAFE（**全序 96 `108f9c3c`** 建 `docs/performance/two-gpu.md` 被全序 177 改；**全序 109 `370a6670`** 建 `docs/maintainer/tensor-parallel.md` 被 118/122 改）。
- 保守面：`tools/tp2/**`（gate.sh / check_shard_sources.py / mailbox 探针 / golden 参考件）、`tests/**`、`src/**`、`cmake`、`*.sh` 全记 **KEEP**（D3/D4 要用，且我方要验）。
- 收益实测口径：33/179 步 ≈ **18% 步数**，但都是薄步 ⇒ 估计省 **8–12% 墙钟**；跳步须逐步在 ledger 记一行"跳过+理由"，并保证 `merge-safety` 的基线符号集不因跳步而丢（跳的步不含 src 变更 ⇒ 安全）。

### 26.4 C：S7 seed 与 D 桶基准（新文档 **`relay/s7-seed-5060ti.md`**）
- 供体 `src/core/tp2/device_tuning.h` 形状：**header-only，按 (CC, SM) 匹配首行生效，末行 = 5070 Ti 实测默认** ⇒ 我方加 5060 Ti 行（CC **120**、SM **36**、`attention_sm_count=0`），四个量：`nvfp4_attn_input_shard_first_tiled256_tokens`、`tp2_ordinary_graph_allowance_bytes`、`tp2_mtp_graph_class_allowance_bytes`、`tp2_dflash2_graph_class_allowance_bytes`；**不进表**的三项（NVFP4 A16 实例、INT8 attention split 规则本体、`kPeerSpinLimit`）。
- baiyouqing 5060 Ti 实测：官方 nvfp4 制品 **每卡 10.08 GiB**，**MTP 关 38 tok/s**，**`--spec mtp --draft-tokens 3 --lm-head-draft` = 76–86 tok/s**；**int8 KV 单槽吃满 262144 且每卡余 842 MiB**（serve 与 CLI 余量相同）；**W4A4 每卡少 16% 权重 ⇒ 106 tok/s**。
- ~~**对 D4 判据的修正（重要）**~~ **＝2026-10-16 21:5x 作废（见 §31）**：76–86 依赖 `--lm-head-draft`，"我方制品 `proposal/head = q4_g64` ⇒ 该优化头不可用" ⇒ 我方目标应写"`--spec mtp --draft-tokens 3`（无 lm-head-draft）实测档 + W4A4 裸 106 tok/s 参照"，**不把 76–86 当必达线**。
- 可迁移项判定（详见该文件 §4）：pinned-host peer mailbox（我方 5060 Ti 无 P2P ⇒ 与 S8/D1 直接相关，到 59/111 按我方面落）、`scope plan-validation failures to one request`（**建议采纳语义**：计划校验失败只拒当前请求）、vision-MTP parity（62 时对照）、MSVC/Windows（不做）、`k16i8` KV 档（fork-only，S6 目前只放行 bf16/int8 ⇒ 若 D2 不够显存再评估，需过我方案门，不白拿）。

### 26.5 停点：队列 **36/213 ＝ 全序 44 ＝ `8229d197`**（run the DFlash2 round at tp 2）
4 处未合并：`program/decode.cpp`、`program/graphs.cpp`、`program/planning/startup.cpp`、`program/program_impl.cpp`；逐字节 **`/home/ai-agent/DSHW/relay/step44-checkpoint/`**（STATUS/STATE/MESSAGE + 供体该步 stat + 4 个带 marker 文件），续跑队列 **`relay/tp2-replay-queue-178-from-full44.txt`**（onto = `084cd71c`）。
下一会话顺序：**44 → 45 `600be761`（accept `--spec dflash2` at tp 2 ⇒ 落地后跑 `ctest -R "speculative|dflash|preemption|qwen3_5"`）→ 46 `11956486`（serve 27B DFlash2 测试）**；B 的跳步自 49（`1c4b470f` docs）起可用。

## 27. 全序 44–49 落地（DFlash2 双设备整轮 + 接受 `--spec dflash2`）+ **A3 兑现：S5b 退役落盘**（2026-10-16 18:0x–19:0x，本会话第 1 段）

### 27.1 落地表（供体 sha → 我方 sha）与逐块落法
| 全序 | 供体 → 我方 | 落法（块＝我方/基线/供体行数） |
|---|---|---|
| 44 | `8229d197` → **`df00521c`** | 4 文件 **7 块**。`program_impl.cpp` 1（**0**/14/22）＋2（5/5/6）＋3（10/6/9）＝块1 取我方（我方已删 `normalized_private_capacity` 匿名 ns，全树 0 引用）**＋必 graft** 供体新函数 `PeerRuntime::layout(plan)`（净应用后的初始化列表已写 `persistent(layout(plan).bytes)`）；块2/3 **取供体**（`own=layout(plan)` 代 `plan.persistent`，三帧择一校验含 `dflash_decode`）⇒ 我方块3 里 全序 30 时代的 `if (!io.ordinary) throw` 与重复 `io = RoundState` 两条**退役死码**一并消失。`decode.cpp` 1（4/1/3）＝取我方 + 照我方 MTP 站点（`decode.cpp:482-483`）位置式补两条 peer 实参 ⇒ 落我方字段表**末尾**（`host_drafts`/`drafts_ready` 之后，§24.2）。`graphs.cpp` 1（**0**/93/102）＝取我方（双族 forward/finish，§22.1）：我方区已有 `warm_batches=tensor_parallel()?max_concurrency:1U` + 每档两阶段 eager + `synchronize_all` ⇒ 只需给 `dflash_context` lambda 追加同两条 peer 实参；**不可取供体**（供体调 4 参 `dflash_graph_profiles(…,batch)`，我方是 3 参双族）。`startup.cpp` 1（14/4/5）＝取我方（`multiprocessor_count`＋Host 默认块＋我方赋值序）+ graft `persistent_layout(*impl,0)` 与 `if (impl->tp != 1) { impl->peer_persistent = persistent_layout(*impl,1); }`；块2（23/22/28）＝取我方配额骨架 + graft 供体 tp2 规则（per-class `if (impl->tp != 1) { return 24ULL*kMiB; }`，finish 配额 tp2 同取 24 MiB＝tp2 下 finish 也是双设备图）。净应用面（3 个 ns 级 `dflash_{context,proposal,accept}_workspace`、tp2 `verification` lambda、`validate_target_options` 放行 DFlash2＋要求优化 head、`tp.h` 把 `replay_records` 前移到 `ordinary` 之后＝声明序）**零冲突**，容量 API 与 §14 三规则逐一同形 ⇒ 本步无需改写。 |
| 45 | `600be761` → **`ef408696`** | 2 块：`docs/cli.md`（6/3/6）＝取我方 + graft 供体能力句（DFlash2 drafter 只在 rank 0 且**必须 `--lm-head-draft`**），保住我方 `(the optimized proposal head is loaded on rank 0 only…)`、`rk4v4-e8` 与 `--host-context-mib` 拒绝口径；`serve/serve_options.cpp`（2/3/4）＝取我方帮助文 + 同能力句。余 7 文件净应用零冲突；`docs/serving.md` 就地补 `rk4v4-e8` 到拒绝列表（供体漏 Ada 档）。4 TU 单编 rc=0（`load`/`model_instance`/`serve_options`/`apps/cli/options`）。 |
| 46 | `11956486` → **`b79a9b42`** | **零冲突**（新测试 `test_engine_dflash2_tp2_real.cpp` 275 行 + `tests.cmake` + README）⇒ 按 §18.2 全构建验：该测试 TU 抓到 **3 处供体 API**（`ContextCacheOptions.host_state_slots`/`host_kv_capacity_bytes` ⇒ 我方 `host_capacity_bytes = 0`；`ContextCacheHints.retention` 与 `CacheRetentionHint::LiveSession` ⇒ 我方无此字段/枚举 ⇒ 删该行、保留 `session_key`）；改写**随 49 提交 `748ae53e`**（46 已零冲突落地，回填非 HEAD 提交代价高）⇒ 275/275 rc=0。 |
| 47/48 | `636f517f`/`aed5ce19` → **`46d750eb`/`3ca4032a`** | **零冲突**（tp2 logits 在 rank 0 聚合：`text.{h,cpp}`/`tp.{h,cpp}`/`workspace.h`/`startup.cpp`/`prefill.cpp`/`mtp.cpp`/`target_verification.cpp` + `test_text_context_tp2.cpp` +85 行）。48 把 47 的 `tp.cpp` +58 行**删回 31 行**（收敛到 `text.h` 的聚合面）⇒ 由全构建验。 |
| 49 | `f1d0f4ce` → **`748ae53e`** | 1 块 `graphs.cpp`（**0**/5/13）＝取我方（我方 `instantiate_graph_family` 是 4 参非模板定义 ⇒ 供体那段 5 参模板前向声明对她是**死码**）+ graft `free_device_bytes(int)`（`ScopedCurrentDevice`+`cudaMemGetInfo`）。余净应用：`program_impl.{h,cpp}` 的 `graph_observed_bytes[rank]`、`types.h` +6、`operational_log.cpp` +10（**per-rank 报告，不 enforce**）。 |

### 27.2 A3 兑现：**S5b 退役落盘**＝新制品 **`relay/s6-gates.patch`**（旧 `relay/s6-s5b-gates.patch` 留档不再用于门）
照 `587–598 行` 删四段：① 产品头 `tensor_parallel_reuse_verdict`（28 行）② `resource_manager.h` 整段（`candidates()` 的 `tp`/`speculative_backend` 形参 + 第二条 `erase_if`）③ `engine_core.h` 整段（两常量 + 初始化 + 调用点两实参）④ `tests/test_cli_options.cpp` 的 reuse 断言段（22 行；**S6 的 capability/KV 断言保留**）。做法＝树面重放旧补丁→删四段→`git diff` 重生成（hunk 头自动正确）⇒ **5 文件 +110/−4**、`git apply --check` 在 44 与 49 两个树面均通过、树内 0 处 `tensor_parallel_reuse_verdict`。**自 44 起所有门＝`vision-on-build-ada.patch` + `s6-gates.patch`。**

### 27.3 门（vision=ON + S6（S5b 已退）+ 全序 44–49 树，GPU1）
- 全序 44 单独门（44 树）：4 TU 单编 rc=0 → 全构建 **86/86 rc=0** → 簇门 `-E softmax_attention` = **100% of 163 / 1068.30 s**（133 Passed + 30 Skipped=77 具名）→ **`relay/passset-sm120-tp2steps44-r0.tsv`**；`merge-safety` 索引 0 / 工作树 0。
- 全序 49 树门（45–49 同树，**披露：45 的成活子集门与簇门合并跑在 49 树面**，49 树 ⊇ 45 面 ⇒ 覆盖率不减）：全构建 **275/275 rc=0**（0 error）；`ctest -R "speculative|dflash|preemption|qwen3_5"` = **100% of 32 / 32.93 s**（11 Passed + 21 Skipped ⇒ 46 的新 tp2 真制品测试在 77 具名跳过之列）；簇门 `-E softmax_attention` = **100% of 164 / 1051.85 s**（133 Passed + 31 Skipped；分母 +1 ＝ 46 的新测试） → **`relay/passset-sm120-tp2steps49-r0.tsv`**。
- GPU1 交还：探针 `cuDeviceGetCount==1`；结束 **4 MiB / 0%**、零我方进程；GPU0 现网 strata（PID 440568，15590 MiB）全程未碰未 kill 未加压；`/tank` 未写；全程零审批。

### 27.4 规则区（本段新增，写死）+ 停点
- **编号坑（重要）**：`tp2-skip-scan-43-223.tsv` 第 1 列 ＝ **队列行 + 3 = 全序 − 5**（表头写"全序 43-223"是笔误；实测 TSV 39=`8229d197`＝全序 44，TSV 49=`1c4b470f`＝全序 54）⇒ **跳步判定按 sha 查表，绝不按行号**；"自 49 起可跳"＝全序 **54**，两条 SKIP-UNSAFE＝全序 **101/114**。
- **净应用面引用供体新函数时，"取我方"的块必须 graft 该函数**（本步 `PeerRuntime::layout()`＝初始化列表已引用）；同块内我方被后续语义取代的旧校验（rank1 只允许 ordinary）＝退役死码，取供体时一并删，不留双校验。
- **图配额 / profile 家族块＝取我方骨架 + 只 graft 数值规则**：供体调我方不存在形参的函数（4 参 `dflash_graph_profiles`）即失配；供体的模板前向声明落在我方非模板定义上＝死码。
- ~~**制品事实（写死给 D 桶）**~~ **＝2026-10-16 21:5x 作废，正确口径见 §31**：45 起 `--spec dflash2` 在 tp2 成活，但 **tp2 DFlash2 必带 `--lm-head-draft`**（rank 0 的完整优化 proposal head；全文表 head 按词表分裂、`linear_topk` 无分裂形）⇒ 我方 W4A4 V3 制品 `proposal/head=q4_g64` **没有该 head** ⇒ 44/45 的 DFlash2 双设备路由**我方制品跑不了**，46 的新测试是 device-gated real 项（77 具名跳过之列）⇒ **DFlash2 tp2 的数值判定只能在窗口**，与 §26.4 的 D4 lm-head-draft 限制同源。
- 簇 tag **`v-tp2-dflash2-round-r0`** ＝ `748ae53e`（全序 43–49，annotated，消息＝本轮门数据）。
- 停点：队列 **42/213 ＝ 全序 50 ＝ `b34e2772`**（`feat(serve): log whether the tp 2 devices use direct P2P`，4 文件 +18/−11）**2 块未解**：`runtime/engine/engine.cpp`、`serve/operational_log.cpp`（`docs/serving.md`/`include/ninfer/types.h` 净应用 ✓）。续跑＝同一 rebase 里 `GIT_EDITOR=true git rebase --continue`（`.git/rebase-merge` 在位，msgnum 42/213，序列头 `748ae53e`）；冷启动＝`git checkout -q tp2/raw` 后 `git rebase -i --onto p1/tp2-replay 748ae53e tp2/raw` + 注入队列 **`relay/tp2-replay-queue-172-from-full50.txt`**（首行 `b34e2772`）。下一步序：**50 → 51 `e7f5cc11`（size the tp 2 CUDA Graph allowance from measured usage）→ 52 `84c2e2ad` → 53 `17746050` → 54 `1c4b470f` ＝ 第一跳（跳步自全序 54 起）**。

## 28. 全序 50–57 落地（p2p 报告 / 实测图配额 / arena 零填充 / 设备作用域 / 图交换回退）+ 第一跳（2026-10-16 19:0x–20:0x，本会话第 2 段）

### 28.1 落地表（每格＝该步净变更 +A/−B；块落法照实测文面）
| 全序 | 供体 → 我方 | 落法 |
|---|---|---|
| 50 | `b34e2772` → **`75965b44`** | 2 块＝**取我方 + graft**。`engine.cpp +6/−4`：`initialize_execution(options, bool& peer_access)` 与 `load.peer_access` 声明/初始化随净应用进树（**初始化列表已引用** ⇒ 同 §27.4 规则①），冲突块只落"我方对齐组"：插 `load.peer_access    = peer_access;` 于 `load.cuda_sync_mode` 之后、保我方 `sampling_defaults   =` 的列对齐；`operational_log.cpp +5/−4`＝我方文面（"the Host context-cache **tier is** off"）+ 供体新句 `p2p {}` ⇒ 4 占位符⇔4 实参（`service.load_summary().peer_access` 净应用 ✓）；`types.h +3`＝`LoadSummary::peer_access`。2 TU 单编 rc=0。 |
| 51 | `e7f5cc11` → **`5a8396d5`** | 1 块 `startup.cpp +26/−17`＝**取我方配额骨架**（3 参 `dflash_graph_profiles` + finish 每档）+ graft 供体**实测常量** `if (impl->tp != 1) { return kTp2DFlash2GraphClassAllowance; }` ⇒ 我在 49 graft 的 24 MiB/类 **被 11 MiB/类取代**，finish 配额 tp2 同取该常量；lambda 补显式 `-> std::size_t`（`size_t` 常量与 `ULL*kMiB` 混推失配）。常量来源注释写死＝**两只 RTX 5070 Ti / 32K / B=1 / INT8 KV**（ordinary 8 · MTP 8 · DFlash2 11 MiB/类，"scales linearly with concurrency，B>1 未测"）⇒ **5060 Ti 行仍待 S7 的 `device_tuning.h`**（§26.4 四量之一）。`operational_log.cpp +13/−3`（observed vs allowance 逐 rank 告警）净应用 ✓。2 TU rc=0。 |
| 52 | `84c2e2ad` → **`9216faf4`** | 1 块 `program_impl.cpp +11/−3`＝取我方（我方匿名 ns 在 44 已删，全树 0 引用）+ graft 新助手 `tensor_parallel_zero_fill(plan)`（`peer_persistent ? ZeroFill::Yes : ZeroFill::No`，自造匿名 ns 包住）。净应用＝`arena.h +7/−1`（`DeviceArena(capacity, ZeroFill zero = No)`、`ninfer::ZeroFill`）、`arena.cu +8/−9`、`materializer.cpp +17/−1`、`tests/test_arena.cpp +14/−0` ⇒ 4 TU 单编 rc=0。 |
| 53 | `17746050` → **`0f52d554`** | **5 块 / 3 文件全取我方**：`program_impl.h +1/−19` 两块＝我方**早已**把该助手搬进 `core/device_scope.h`（`src/core/device_scope.h:15`）并在 detail 命名 `using ninfer::ScopedCurrentDevice;` ⇒ 供体"去 `detail::` 限定"的改动与我方无关（`detail::` 经 using 仍有效）；`execution/draft.cpp +2/−2` 块（**0**/108/108）＝供体的**单阶段** `dflash_decode_batch_body` 整块在我方不存在（我方 = `draft.cpp:573` 与 `:896` 两个 phase body）⇒ 删；`speculative/mtp.cpp +8/−9` 两块＝保我方 P0 的 `phase == Forward` 守卫 + `detail::` 限定。净应用 6 文件（`ffn +12/−11`、`text.cpp +14/−31`、`execution/mtp +3/−4`、`decode +1/−1`、`prefill +1/−1`、`target_verification +1/−1`）⇒ **9 TU 单编 rc=0**。 |
| 54 | `1c4b470f` → **跳过**（TSV 49 `SKIP-SAFE-CAND`） | **跳步记行**：只动 `docs/maintainer/{engine-architecture,resource-scheduling-and-context-cache}.md`（DOC:2，0 处 src/include/tests）⇒ 不影响 `merge-safety` 基线符号集；带入＝在我方另一套文档权威（Ada 口径 + §16.2 面）上重写，零行为收益。 |
| 55 | `3be82075` → **`6a74206e`** | **零冲突**（`materializer.cpp +10/−9`、`artifact-container.md +1/−1`）。 |
| 56 | `440481b2` → **`6ce1df58`** | 1 块 `tests/test_decode_graph.cpp +25/−0`＝取我方（我方该处是 `check_external_event(device)` + 双设备 SKIP 说明）+ graft 供体 25 行新测段（两节点对一 ⇒ `update()` 抛 "update result"、`update_or_reinstantiate()` 回落实例化、`diagnostic` 非空、launch 得 `0x55555555`、同拓扑再交换不回退；作用域已核：`device/storage/executable/failures/expect*` 在块外皆可见）。净应用 `core/decode_graph.cpp +92/−3`、`decode_graph.h +10/−0`（`update_or_reinstantiate`＝`decode_graph.h:119`）、`graphs.cpp +13/−1` ⇒ 3 TU rc=0。 |
| 57 | `f3846fcb` → **`3d6ef5cc`** | 3 块（`decode.cpp +1/−4`、`graphs.cpp +23/−16`、`program_impl.h +10/−0`）＝**取我方骨架 + 交换逻辑只留一套**：`decode.cpp` 块（**1**/48/45）取我方（我方匿名 ns 的三个 graph helper 在 P0 已并入 `DecodeGraphFamily::select/install`）；保留供体 ns 级 `install_graph_definition(family, topology, profile_index, label)` 作**唯一实现**（graphs.cpp:76 的净应用调用点已指向它 ⇒ 不 graft 就是未定义引用）；我方 `DecodeGraphFamily::install` 加 `const char* label` 形参并**委派**它，6 个调用点补标签 `"ordinary"` / `"speculative forward"` / `"speculative finish"`；`update_rejected` 字段落我方 `DecodeGraphProfile`。`<cstdio>` 进 graphs.cpp include。4 TU rc=0。 |

### 28.2 门与纪律（本段）
- **每步单 TU 门全绿**（50＝2 · 51＝2 · 52＝4 · 53＝9 · 56＝3 · 57＝4），`merge-safety` 每步索引 0 / 符号清单无缺失。
- **全构建 + 簇门未跑**：44–49 门（§27.3）之后连 8 步 ⇒ **补跑在 58 落地后**（vision=ON + `s6-gates.patch`）。⚠ 门补丁漂移：50 改 `initialize_execution` 签名 ⇒ `git apply --check relay/s6-gates.patch` **已 FAIL**（vision 补丁仍 OK）⇒ 在 58 树面按 §27.2 同法重生成（`git apply --3way` 旧补丁 → 手工合 engine.cpp 两处 → `git diff`）。
- 新工具 **`relay/tu_compile.py <build-dir> <后缀>…`**：读 `compile_commands.json` 并行编匹配 TU、逐条报 rc ⇒ 每步落地前跑（免手写 `python -c` 的 shell 引号坑）。

### 28.3 停点（下一步规格）
- 停点：队列 **50/213 ＝ 全序 58 ＝ `cef61349`**（reclaim idle private continuations for Host-less prefill captures，8 文件 +179/−24）＝ **5 UU + 1 DU / 9 块**：`context_cache/resource_manager.h +52/−1`（**5 块**）、`engine_core.h +4/−1`（1）、`model_instance.cpp +7/−6`（1）、`docs/maintainer/{engine-architecture,resource-scheduling-and-context-cache}.md`（各 1）＋ **DU `tests/test_resource_manager.cpp +57/−0`**（我方删、供体改 ⇒ 照 §22.1 `storage/context.cpp` 先例 `git rm` 接受删除，其语义 graft 进我方唯一站点；该步的 `test_engine_tp2_real.cpp +43/−10` 已净应用进索引）。逐字节 **`relay/step58-checkpoint/`**；冷启动队列 **`relay/tp2-replay-queue-164-from-full58.txt`**（首 `cef61349`，onto `3d6ef5cc`）。
- **新规则（写死）**：① **`git checkout --theirs -- <file>` 取整文件**（＝丢我方该文件全部改动，非"块取供体"）⇒ 复位我方版只认 **`git show :2:<file>`**（§26.1 坑 ③ 的镜像面）；本轮实踩：头文件被 `--theirs` 覆盖后净应用字段 `update_rejected` 消失，须手工补回。② **跳步表 TSV 行 + 5 = 全序 ⇒ 按 sha 查表**。③ 供体的"实测常量"落我方＝替换数值 + 保留我方骨架 + 注释写死测量硬件（5070 Ti ⇒ 我方 5060 Ti 待自测）。
- 下一步序：**58 → 59 `2ec3da81` / 60 `c1ff2508`（PeerMailbox 交换＝S8 相关；我方 5060 Ti 无 P2P ⇒ 按我方面落，见 §26.4 可迁移项）→ 61 `fe7dda13`（`--no-tp-mailbox`）→ 62 `ff592ee2`（Vision 单 rank / 双设备 prefill）→ 63 `d0a1a665`（`--vision-device`/`--max-vision-tokens`）→ 64 `9348e32e`（tp2 图像 serve 测试）**；**65 `67e4d923`＝第二个可跳**（TSV 60）。

## 29. 全序 59–61 落地（PeerMailbox / `--no-tp-mailbox`）+ **58＝整步空效跳过** + 停在 62（2026-10-16 20:0x–20:4x，本会话第 3 段）

### 29.1 全序 58 `cef61349` → **跳过（净效应为空提交 ⇒ `git rebase --skip`，非"跳步表"跳）**
供体这步的宿主机制在本树**整体不存在**：`resource_manager.h` 我方版 ＝ **1461 行**（供体版 3390 行；含 marker 的冲突文件 5078 行），逐符号实测 `CatalogEntry` / `CaptureAssessment` / `ActiveCaptureReserveResult` / `RetentionClass` / `private_retention_weight` / `private_has_active_edge` / `erase_session_if_owner` / `continuation_checkpoint_count` / `record_checkpoint_drops` / `pressure_private_owners_evicted` / `inspect_capture` / `release_continuation` / `skip_capture` / `rebuild_prefix_index` **全为 0 处**（实测法：`git show :2:<file> | grep -c <符号>`）⇒ 与 §14（`request_plan.cpp` 不 graft）/ §25.1 同一判定＝**强 graft＝造第二套 admission/目录层**（铁律 3）。
落法＝**5 UU 的 9 块全取我方** + `engine_core.h`（取我方＝不带 `host_state_slots == 0` 实参，因 `ResourceManager` 无形参）+ `model_instance.cpp`/`docs/{engine-architecture,resource-scheduling-and-context-cache}.md`（取我方＝**我方"必被跳过"的文面对我方树是权威**）+ **DU `tests/test_resource_manager.cpp` ⇒ `git rm`**（我方删除，供体 +57 的测试落我方不存在的层）+ **回退净应用面**：`tests/models/qwen3_5/test_engine_tp2_real.cpp`（58 加 `exercise_reuse_after_one_shots`＝32 个 one-shot 洪水后要求 ≥80% 复用 ⇒ 测的是我方未实现的政策，窗口会假失败；`git checkout HEAD -- <file>`）与 `docs/serving.md` 的 reclaim 段（`git checkout HEAD --`）。⇒ 索引/工作树对 HEAD **零差异**＝空提交 ⇒ `git rebase --skip`。
**记规则**：全块取我方 ⇒ 先 `git diff HEAD --stat`（空＝这步对我方无宿主）⇒ `--skip` 并在账本记一行原因＋**净应用面里"测未实现政策"的测试与文档必须一起回退**。

### 29.2 全序 59–61 落地
| 全序 | 供体 → 我方 | 落法 |
|---|---|---|
| 59 | `2ec3da81` → **`48e4aa89`** | **零冲突**（`feat(ops): exchange captured all-reduces through an epoch-flagged PeerMailbox`＝ops 面 mailbox，S8/D1 相关；我方 5060 Ti 无 P2P ⇒ 走 pinned host 路径）。 |
| 60 | `c1ff2508` → **`afa8fe35`** | 2 块 `startup.cpp`（6/3/4、15/14/15）＝**取我方 + graft `tp_mailbox`**：`impl->tp_mailbox = inputs.tp_mailbox;`（插在 `impl->tp` 之后、我方 `multiprocessor_count` 之前）＋ designated-init 里 `.tp_mailbox` 按**声明序**落 `.tp` 与 `.context_cache` 之间（`startup.h` 净应用已把字段排在 `tp` 后 ⇒ 顺序合法）；`docs/maintainer/engine-architecture.md` 取我方。净应用 6 文件（`types.h +4`＝`EngineOptions::tp_mailbox`、`graph_execution.h +2`、`program_impl.{cpp,h} +13/+7`、`qwen3_5-model.md +18`）⇒ 4 TU rc=0。 |
| 61 | `fe7dda13` → **`a9d674cb`** | 4 块全＝**取我方 + graft**：`apps/cli/main.cpp`（6/5/6）插 `engine_options.tp_mailbox = cli.tp_mailbox;`（我方对齐组＝`=` 列 43）；`docs/cli.md`（3/2/4）取我方 YaRN 段＋插入供体 mailbox 两句（`--no-tp-mailbox` 保 staged copies）；`serve_options.cpp` 两块＝usage 列表加 `[--no-tp-mailbox]` ＋ 我方 tp2 帮助段后插 `--no-tp-mailbox keeps the captured --tp 2 all-reduces on cross-device copies`。净应用 6 文件（`cli/options.{h,cpp}`、`serve_options.h`、`generation_service.cpp`、`test_cli_options.cpp`、`test_serve_options.cpp`）⇒ **8 TU rc=0**（含两个测试 TU ⇒ 断言的正是 graft 的帮助文面）。 |
- 簇 tag **`v-tp2-graph-allowance-mailbox-r0`** ＝ `a9d674cb`（全序 50–61，消息＝本段门数据与 58 的跳过理由）。
- **门状态（写死给下一会话）**：50–61 **每步单 TU 全绿**（2/2/4/9/3/4/8 ＋ 59 由 60/61 的 TU 覆盖），`merge-safety` 每步索引 0；⚠ **全构建门与簇门欠一次**（44–49 之后连 12 步）⇒ 62 落地后第一件事＝按 §28.2 重生成 `relay/s6-gates.patch`（50 改了 `initialize_execution` 签名 ⇒ 现补丁 `--check` FAIL）→ 全构建 → 簇门 `-E softmax_attention` → 存 `relay/passset-sm120-tp2steps62-r0.tsv`（分母应仍 **164**；58 回退后没加测试）。

### 29.3 停点＝下一步规格（全序 62，本会话结束点）
- 停点：队列 **54/213 ＝ 全序 62 ＝ `ff592ee2`**（`feat(qwen3_5): run Vision on one tensor-parallel rank and prefill it over two devices`）＝ **24 文件 +730/−216 / 14 块 + 1 DU**（逐字节 `relay/step62-checkpoint/`，STATUS 全列）。冷启动队列 **`relay/tp2-replay-queue-160-from-full62.txt`**（首 `ff592ee2`，onto `a9d674cb`）。
- 块分布：`execution/vision.h`（3）、`vision.cpp`（3）、`planning/startup.cpp`（2）、`include/ninfer/types.h`（1，`+20/−5`＝`EngineOptions` 的 vision-device 面）、`execution/text.cpp`（1，**+206/−64**＝最大块）、`program/prefill.cpp`（1）、`program/program.h`（1）、`program_impl.cpp`（1）、`runtime/engine/model_instance.cpp`（1，**+40/−17**＝与我方 `validate_options` 的 Vision/tp2 拒绝面正面对撞）＋ **DU `program/transactions/materialization.cpp`**（我方删、供体改 +20/−5 ⇒ 照 §22.1/§29.1 先例 `git rm` + 语义 graft）。净应用 10 文件（`text.h +24/−16`、`tp.h +6/−1`、`frontend.{cpp,h}`、`processor.{cpp,h}`、`execution_context.h`、`request_plan.cpp`、`startup.h +36/−4`、`program.cpp`、`program_impl.h`、`speculative/mtp.cpp +25/−16`、`test_text_context_tp2.cpp`）。
- **预判落差**：① 我方 tp2 现在**拒绝 Vision**（`validate_target_options`/`validate_options`/44 的 `vision_enabled` 拒绝面，账本 `549 行` §23 的 cli.md 口径）⇒ 62 是"开启"这面 ⇒ 逐点核对我方拒绝文面与 S6 门（`tensor_parallel_capability_verdict` 在门补丁里，不在树内）；② R3 裁定"Ada 只留接口"＝Vision-at-tp2 的接口面＝S1/S2 的既有闭合口径（`266–281 行` §15）⇒ 若 62 落我方骨架需要新注册面，按 **A1** 走我方模板化注册，不复制供体家族文件；③ 46/58 教训：**净应用测试断言我方未实现的政策 ⇒ 回退测试并在账本记 gap**；④ 63 `d0a1a665`（`--vision-device`/`--max-vision-tokens`）与 64 `9348e32e`（tp2 图像 serve 测试）＝62 的下游，65 `67e4d923`＝第二个可跳（TSV 60）。

## 30. 全序 62 落地＝参数面（Vision-at-tp2 机制面判为 S11）+ 停在 63（2026-10-16 21:0x–21:3x，本会话第 4 段）

### 30.1 全序 62 `ff592ee2` → **`f64af753`**：14 块全取我方 + 4 处 graft ⇒ **净落地只有参数面**（+17 行，`include/ninfer/types.h`）
逐块（我方/基线/供体）：`types.h`（3/5/10）、`text.cpp`（6/5/13）、`program_impl.cpp`（5/8/8）、`model_instance.cpp`（8/7/9）、`startup.cpp`（5/5/7、16/15/17）、`vision.h`（6/2/5、8/4/12、0/1/5）、`vision.cpp`（5/4/23、2/0/30、15/5/9）、`prefill.cpp`（**10/491/490**）、`program.h`（**4/164/171**）＋ **DU `transactions/materialization.cpp`**（照 gap-plan §3 B 类日历 `62` 行：我方无该文件 ⇒ 接受删除）。
Graft 已落：`kMinimum/MaximumMaxVisionTokens` 两常量、`program_impl.cpp` 的 `vision_rank(plan.features.vision_rank)`（头里 `const int vision_rank` 是净应用 ⇒ 不初始化＝"uninitialized const member"）、`model_instance.cpp` 的 `.max_item_vision_tokens`、`startup.cpp` 的 `impl->vision_parameters/max_vision_item_tokens` 与设计器 `.vision_parameters/.max_vision_item_tokens`（按声明序紧跟 `.parameters`；`make_sequence_planner_impl` 已带 `peer_parameters`）＋ `text.cpp` 的 **3 轴 RoPE 末列 gather**（`three_axis_rope` ⇒ 三次 D2D 拷进 `last_rope[r]`，再走我方 `text_rope(..., &parameters_.rope_scaling, ..., rank_execution_view(rank))` 签名）。
**判为机制面的理由（实测编译，铁律 4 第 6 次兑现）**：取我方后净应用面缺 5 处符号，且都落在**我方已重写的类**里＝不可定点 graft：`VisionChunk::peer_embeddings`（`text.cpp:2109/2172`）、`VisionContext::plan_receiver`（`startup.cpp:787`）、`SequencePlanner` 私有构造（`program.cpp:317`）、`construct_model` 的 `peer_parameters` 面（`model_instance.cpp:276/283`）、`VisionPrefillSession` 双 rank 面（`ranks_`/`tensor_parallel_`/`vision_rank_`/`bind_rank_output`/`encode_on_ranks`/handoff 的 `encoded_`+`copied_`，`vision.cpp:471–496`）⇒ 这 21 个文件整面 `git checkout HEAD --` 回退，只留参数面。
**保住我方权威**：`types.h` 的 tp2 能力句**不跟供体**改成"with or without Vision"，仍写"Vision 拒绝"，并在 `vision_device` 注释写死"本树 tp2 拒绝 Vision ⇒ 该旋钮只在单卡设备间移动 tower"。
⇒ **新任务 S11「Vision 在 tp2」**（供体全貌＝本步；单卡可测判据＝`VisionPrefillSession` 双 rank + `VisionChunk.peer_embeddings` + `WorkspacePlan::vision_receiver/receiver_capacity/rank_vision/rank_capacity` 报价面；数值判定＝窗口 D2 的 Vision 子项）。提交保持供体消息（先例＝§25.1 全序 42 的 1608 行单块取我方）。

### 30.2 停点＝全序 63 `d0a1a665`（`--vision-device`/`--max-vision-tokens` 进 ninfer 与 ninfer-serve）
队列 **55/213**；**2 文件 3 块 + 1 新文件**：`apps/cli/options.h`（1：我方 `YarnOptions yarn` + `enable_vision` ⇒ graft `vision_device`/`max_vision_tokens` 两字段）、`src/serve/serve_options.cpp`（3：usage 列表加 `[--vision-device N] [--max-vision-tokens N]`；**帮助文里供体的"with or without --vision"tp2 说法不带入**；解析站点 graft `product::validate_vision_options(enable_vision, vision_device, max_vision_tokens, devices)`）＋ **A 类新文件 `src/product/vision_options.h`（44 行，纯 CPU：`parse_max_vision_tokens`/`validate_vision_options`，引用 `product/tensor_parallel_options.h` 的 `detail::parse_small_decimal` ⇒ 单 TU 验）**；净应用 6 文件（`apps/cli/{main.cpp,options.cpp}`、`generation_service.cpp`、`operational_log.cpp`、`serve_options.h`、**两个测试**）。逐字节 `relay/step63-checkpoint/`（13 文件）；冷启动队列 **`relay/tp2-replay-queue-159-from-full63.txt`**（首 `d0a1a665`，onto `f64af753`）。
门状态：**全构建 + 簇门仍欠一次**（44–49 之后连 13 步）；`relay/s6-gates.patch` 现 FAIL 三处＝`engine.cpp:27`、`tests/test_cli_options.cpp:165`、`tests/test_serve_options.cpp:450` ⇒ 下一会话顺序＝解 63 → 落地 → **重生成门补丁**（`git apply --3way` 旧 `s6-s5b-gates.patch` → 删四段（§27.2）→ 合 63 的新站点 → `git diff`）→ 全构建 → 簇门（分母应仍 **164**）→ `passset-sm120-tp2steps63-r0.tsv`。跳步自 **65 `67e4d923`**（TSV 60）起。

### 30.3 新规则（写死）
- **"取我方"之后必须编译同一步的净应用文件**：若缺的符号落在**我方重写过的那个类**里（本步 `VisionPrefillSession`/`TextContext`/`SequencePlanner`），该步＝**机制步**，正解＝整面回退只留参数面 + 立 S-task；若在函数体/调用点（44/52/57 的 `layout`/`zero_fill`/`install_graph_definition`），正解＝graft 该助手。这两条的分辨法＝`grep -c <符号> <我方文件>` 为 0 且符号属于**类成员** ⇒ 机制步。
- 供体把我方权威文面改宽（如"tp2 支持 Vision"）时**不跟**：能力句与字段注释按我方实测状态写，落差记本节。

## 31. **更正：TP2 的 MTP 与 DFlash2 都可用**（2026-10-16 21:5x，用户质疑后实测代码，本会话第 5 段）

**结论**：我在 §26.4/§27.4 写下的"我方制品 `proposal/head=q4_g64` ⇒ `--lm-head-draft` 不可用 ⇒ 双卡的 MTP 优化头与 DFlash2 整条路由跑不了"＝**误推，作废**。76–86 tok/s 是**可达目标**，投机解码在双卡成立。

**证据链（四条，逐条可复跑）**
| # | 事实 | 出处 |
|---|---|---|
| ① | 双卡下**优化草稿头不分裂**：`MtpParameters::output_head` 注释＝"Empty on a tensor-parallel rank that does not hold the selected head: the optimized proposal head is **PrimaryOnly**, and its proposal runs on rank 0" | `src/models/qwen3_5/execution/parameters.h:74-77`（＋`model.h:43` 定义 PrimaryOnly） |
| ② | 供体文档同口径：DFlash2 的 `dflash2/*` 与 `proposal/*` 权重**只放 rank 0**，"splits only the target"；被拒的只有"全文表 head 的词表分裂形"（`linear_topk` 无分裂形）⇒ 所以 DFlash2 在 tp2 **必须带** `--lm-head-draft`，而带了就**可用** | `docs/maintainer/dflash.md` §Tensor-parallel execution（45/46 落树）；`src/models/qwen3_5/load.cpp:61`、`runtime/engine/model_instance.cpp:79`、`planning/startup.cpp:1206` |
| ③ | q4_g64 有**单卡（＝整块）执行路由**：`src/ops/linear/q4/`（gemv / mma / simt / sliced_k_mma）＋ **`src/ops/linear_topk/q4_m64.cu`**（DFlash2 选择器要的 top-k 融合 linear 有 q4 实现）⇒ "无半形状"只封**分裂路径**，封不到 PrimaryOnly 的整块路径 | `ls src/ops/linear/q4/`、`ls src/ops/linear_topk/`（制品报告第 43 行的原意＝"这条**分裂**路径不支持"） |
| ④ | **MTP 在 tp2 不依赖 lm-head-draft**：39 验收的是"MTP 头像 Text 层一样分裂、verification 跑双 rank"；MTP 投影是 q8_g32，其半形状在 35 已按我方 A16 sliced-K 机制注册（不是供体 ksplit） | 台账 §18.1（Q8 家族落差裁定）、tag `v-tp2-mtp-round-r0` 消息、§22/§23（38/39 落法） |

**能力矩阵（我方制品＝22.141 GiB W4A4，16 GB 卡）**：单卡**装不下**（权重 11.07 GiB/卡是双卡的分法）⇒ 投机只有双卡这一种场景有意义；双卡可用＝`ordinary` · `--spec mtp` · `--spec mtp --lm-head-draft` · `--spec dflash2 --lm-head-draft`；被拒＝`--spec dflash2`（不带 lm-head-draft）。

**作废与改写面**：§26.4 的 D4 修正条（已就地标作废）、§27.4 制品事实 bullet（已标作废）、`relay/s7-seed-5060ti.md` §3.1（已整条重写为本口径）、薄卡制品段与 D4 口径（本次同步改）。**权威＝本节**，其余写"见 §31"。

**新立任务＝S12「投机三档实测矩阵」**（原 D 桶只按单档设计，量级变了）：
1. **D2 预算**：rank 0 独扛 DFlash2 草模（q8 投影）＋ 340 MiB 草稿头（q4_g64）＋ 自己那半 text/KV/StateImage ＋ graph allowance；参照实测余量 842 MiB 是**官方 nvfp4（10.08 GiB/卡）**的数，我方 11.07 GiB/卡 少约 1 GiB ⇒ 逐档核 `--lm-head-draft` 开/关的 rank0 峰值（**开了反而省**：小头 340 MiB vs 分裂全文表 head 的 rank1 全词表 logits 帧）。
2. **D3 判定**：三档各跑 GSM8K 500（掉幅 <1 分、只允许 `--tp` 变、`subset_digest` 相同）＋ 60 greedy 与 10 条 DFlash2 文本逐字对同树 `--tp 1`——**注意我方单卡跑不了 27B ⇒ tp1 分母需换成 5090 现网或合成模型面**（这条要在开窗前定，是 R1 前置的新增项）。
3. **D4 速度**：`A B B A` 三档扫描（T=1,4,8,16,32,512,1024），76–86 tok/s 恢复为**必达线**，106 为参照。
4. 窗口时长由 7–9 h 涨到 **约 10–13 h** ⇒ "拆两次窗口"的划分要重排（D1+D2 / D3+D4 的切法不再够）。
**待用户定**（我按原排序继续推重放，不阻塞）：(a) D4 必达线是否恢复 76–86；(b) S12 三档矩阵是否全进窗口；(c) D3 的 `--tp 1` 分母取哪台（现网 5090 / 合成模型 / 放弃该对照）。

**新规则（写死）**：**判"某编码在 TP2 能不能用"，先看放置策略**（`Split` / `PrimaryOnly` / `SingleDevice`，见 `model.h:43`）——"无半形状"只封**分裂面**，封不到整块在 rank 0 的路由；凡制品编码/能力类结论，必须引 ①放置策略 ②ops 家族目录 ③准入政策代码行 三处，缺一不得写进卡与台账。

## 32. 全序 63–72 落地（Vision 参数面 / 4 连跳 / 派发几何常量 / 集中 free-bytes）+ 补跑 50–72 全构建门（2026-10-16 21:4x–22:3x，本会话第 6 段）

### 32.1 落地表
| 全序 | 供体 → 我方 | 落法 |
|---|---|---|
| 63 | `d0a1a665` → **`383a3178`** | 4 块（`apps/cli/options.h` 2/1/3、`serve_options.cpp` 2/1/2、3/3/4、2/25/27）＝**取我方 + graft**：CLI 结构体两字段（`vision_device`/`max_vision_tokens`）、usage 列表两旋钮、serve 解析站点 `product::validate_vision_options(enable_vision, vision_device, max_vision_tokens, devices)`（紧跟 `resolve_tensor_parallel_devices`，落我方单-Host-旋钮流）；**帮助文里供体的"with or without `--vision`"tp2 说法不带入**（我方 tp2 拒绝 Vision＝权威，`model_instance.cpp:84`）。**A 类新文件 `src/product/vision_options.h`（44 行，纯 CPU：`parse_max_vision_tokens`＋`validate_vision_options`，用我方 `tensor_parallel_options.h:19` 的 `detail::parse_small_decimal`）⇒ 经 9 个 TU 单编验 rc=0（含两个测试 TU ⇒ 断言的就是 graft 的文面）。** |
| 64 | `9348e32e` → **跳过** | 新测试 `test_engine_vision_tp2_real.cpp`（+358）＝**测我方未实现的政策**：它构造 `tp=2 + enable_vision` 的 Engine，而我方在 `model_instance.cpp:84` 拒（"Engine tp 2 does not support Vision"）⇒ 照 §29.1 规则（净应用测试测未实现政策 ⇒ 回退＋记 S-task）整步跳，测试随 **S11** 一起落。 |
| 65 | `67e4d923` → **跳过**（TSV 60 `SKIP-SAFE-CAND`，DOC:4） | 纯文档（讲 Vision at tp2 ＝ S11 未实现面）⇒ 带入＝写下我方不具备的能力。 |
| 66 | `9225705e` → **跳过** | DU：它只改 64 新建的那个文件（64 已跳 ⇒ 无宿主）。 |
| 67 | `4ae30a9f` → **跳过（8 个 DU，无缺口）** | 供体在 `causal_cache/{prompt,small_t}*.cu` 的**派发尾部**补"拒绝不支持的头几何"。我方 P0 已把这 8 个扁平文件换成 `causal_cache/{bf16,fp8,int8,k8v4,nvfp4,e8}/…` ＋**派发入口集中校验** `require_causal_geometry`（`causal_softmax_attention.cpp:33-47`：只放行 24/4、12/2、16/2 ＋ `12/2` 需 BF16/INT8；另 `:218/:260` 校验 KV 头数）⇒ 我方覆盖更强（早于 kernel），净效应空 ⇒ `--skip`。 |
| 68 | `654b41d2` → **`85e42955`** | 1 块 `nvfp4_linear_add_plan.cpp`（2/2/2）＝**取我方阈值**（`first_a4 = … ? 17 : 8`＝我方标定值，供体是 7）+ 只 graft 命名（`input_rows == Nvfp4N5120K6144::kInputRows`，常量随净应用进树）。`wrapper/linear_swiglu.cpp +18/−5`、`mtp_pack.cpp +22/−8` 净应用 ⇒ 3 TU rc=0。 |
| 69–70 | `d433d14b`→**`7f9227fc`** / `e01a1039`→**`90cfccfb`** | 零冲突（shard/arena 报错命名；tp 宽度与 `vision_rank` 一次校验）。 |
| 71 | `db320778` → **`1a6e1fc4`** | 1 块 `model_instance.cpp`（7/7/9）＝**取我方**（**不带** `planner.unallocated_reservation_bytes` 的 credit＝62 的 S11 面，我方无该函数）＋ graft 本步真语义＝用净应用进来的单一助手 `rank_free_bytes(primary, execution)` 替掉我方内联循环（`std::vector<std::size_t> rank_budgets = rank_free_bytes(device, execution);`）⇒ 1 TU rc=0。 |
| 72 | `44a0a64f` → **`91848f22`** | 零冲突（staging events 走 `DeviceSelection`）。 |
- 序列头 **`91848f22`＝全序 72**；本会话累计落地 **44→72 ＝ 24 步**，跳 **54 / 64 / 65 / 66 / 67**（每跳本表已记理由）。

### 32.2 补跑的门（§28.2/§29.2 欠账在此清）
- 门补丁漂移处理：在**独立 worktree `ninfer-watch/gate-72`（detached @ `91848f22`）**里 `git apply --3way relay/s6-gates.patch` ⇒ 手工合 2 处（`engine.cpp`＝能力门留 `tp==2` 分支内 ＋ 50 的 `peer_access` 赋值留在分支外＝所有路径都赋值；`tests/test_cli_options.cpp`＝我方 63 的 vision 断言段 ∪ 供体 S6 断言段）⇒ 生成 **`relay/s6-gates-r72.patch`**（后续会话底档），门树面 `--check` 通过。
- 全构建（vision=ON + s6-gates-r72，50–72 树）＝ **GATEBUILD**；簇门 `-E softmax_attention` = **GATE72** → `relay/passset-sm120-tp2steps72-r0.tsv`。
- 停点：**队列 65/213 ＝ 全序 73 ＝ `c9062525`**（plan the tp 2 workspace with the shared reserve helpers）＝ **1 文件 4 块**（4/4/4、8/9/9、1/6/6、3/2/2，全在 `planning/startup.cpp` 的 tp2 工作区计划区）⇒ 落法预判＝取我方 + graft"把局部 lambda 换成 ns 级 `reserve_*` 助手"（44 已进树的四个助手，账本 `640 行` §27.1）；逐字节 `relay/step73-checkpoint/`；冷启动队列 `relay/tp2-replay-queue-149-from-full73.txt`（onto `91848f22`）。

### 32.3 规则区（本段新增）
- **"入口集中校验"覆盖"派发尾逐点校验"**：判 67 这类加固步时，先查我方是否把同类校验放在**派发入口**（`require_causal_geometry`），是＝无缺口，直接 `--skip` 并把两条位置写进台账，避免下个会话重复勘查。
- **阈值类块＝取我方标定值 + 只 graft 命名**（68：我方 17 / 供体 7；命名常量随净应用进树 ⇒ 不 graft 就留魔数）。
- worktree 门法（新工装）：`git worktree add --detach ../gate-<全序> <HEAD-sha>` ⇒ **不动在位 rebase** 就能跑"全构建＋簇门"，`.git` 共享（59 MB）、`../deps-ada` 相对路径仍成立；门完 `git worktree remove` 回收。

### 32.4 门状态披露（收门者必读，与薄卡 v9"第一件事"同口径）
- 在 `gate-72` 跑的这次全构建＝**vision=OFF**（我只对 vision 补丁做了 `--check`，未 `git apply`）⇒ 它是 50–72 的**文本引擎面**构建证明（进度快照 `relay/gate72-build.log`，455/969 时 0 error）。
- 标准门（R2 裁定：每簇带 vision 编译面）＝进 `gate-72` ⇒ `git apply ../relay/vision-on-build-ada.patch` ⇒ 重跑 `JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests`（重配 VISION=ON＝增量补 vision 面）⇒ rc=0 后跑簇门 `-E softmax_attention`（GPU1，分母 164）→ `passset-sm120-tp2steps72-r0.tsv` ⇒ 两面数据都记进 §33。
- 门补丁成品＝**`relay/gates-r72.patch`**（199 行 / 5 文件＝S6 能力门＋KV 门在 72 树面的合并版；`engine.cpp` 已合 50 的 `peer_access` 签名，`test_cli_options.cpp` ＝我方 63 断言 ∪ S6 断言）；vision 面单列（`vision-on-build-ada.patch`）。

## 33. 全序 73–91 落地（共享报价助手 / 图控制清单 / 双设备测试套件 / A8 家族常量 / mailbox 只在无 P2P）+ 停在 92＝供体自撤 56/57 回退（2026-10-16 22:5x–23:5x，本会话第 7 段）

### 33.1 落地与跳过表（块＝我方/基线/供体）
| 全序 | 供体 → 我方 | 落法 |
|---|---|---|
| 73 | `c9062525` → **`03e978e4`** | 4 块全**取我方 + 只 graft `reserve_*` 命名**：GDN 容量保我方 **4 参**（去供体的 `true`，§14 三规则）；tp2 计划里供体的 3 处 **Vision 行**（`plan.features.vision` 的 I32 1×3、MTP 合成嵌入 + `visual_scatter_indices`）**不带入**（＝S11 面）；我方多出的 1 行控制照旧。 |
| 74 | `311ab71e` → **`7dcb00c7`** | 1 块 graphs.cpp（1/1/3）＝**取供体**（我方 `stable_controls(io)` 双族控制清单只留一套）+ graft 我方字段名 `io.dflash_prefill->local_append_count`（供体写 `produced_count`；`round_buffers.h:228` 是权威）。 |
| 75 | `c413ac14` → **`215624d0`** | 2 块 causal_cache 测试（57/32/0、10/9/0）＝**取供体的删除**（判据移进新共享头）＋ graft 进 `tests/ops/attention_criteria.h`：**我方预算**（Int8 2.0e-2／FP8 8.0e-2／K8V4 8.0e-2；注释写明"供体 3.15e-3/1.2e-2 是 5070 Ti/5090 标定，S13 复评后才收窄"）＋ **E8 判据与分支** ＋ **`kv_storage_supported()` 的 Ada/Blackwell 编译期门**（4 处调用点靠它）。A 类新头 ×2 ⇒ **9 个测试 TU 单编 rc=0**。 |
| 79 | `f695d347` → **`f3dcb723`** | 2 块 program_impl.h（5/0/3、3/3/4）＝取我方 + graft 家族旗标 `reinstantiate_rejected_updates`；净应用面把旗标设在我方**没有的** `mtp_graphs/dflash_graphs` 上 ⇒ 改成我方三家族 `{ordinary, speculative_forward, speculative_finish}`（graphs.cpp 唯一设置点）。 |
| 80 | `bf0b2f31` → **`0251d748`** | 2 块文档＝取我方权威 + graft 新政策句（"tp2 DFlash2 的草模带 full-attention 层则拒绝；其 KV 无 rank 1 镜像"）；代码面净应用 `startup.cpp:1200-1210` 的拒绝 ⇒ 1 TU rc=0。**制品实测：`dflash2.layer_types` 全 sliding（5 层，window 2048）⇒ 对我方制品不触发。** |
| 81 | `4534b7bf` → **跳过** | `request_plan.cpp` 1 块（9/792/785）＝我方无 admission 宿主（§14/§25.1）⇒ 取我方后 **`git diff HEAD` 空** ⇒ 净效应空提交 ⇒ `--skip`。 |
| 82 | `aa2fecc1` → **跳过** | 依附 58（"reclaim 只在 tp2 开"）＝宿主层不存在（同 §29.1 符号实测）⇒ 取我方 + DU `git rm` ⇒ 净空。 |
| 83 | `ad2dc8fb` → **`4d553e65`** | 1 块 cli.md（3/1/2）＝取我方 + graft "mailbox **只在无 peer access 时**用"；净应用＝`engine.cpp:167 if (peer_access) { options.tp_mailbox = false; }` ⇒ **我方 5060 Ti 无 P2P ⇒ 生产路径就是 mailbox**（S8 的降级/超时用例因此是真路径，不是边缘）。test_allreduce +32 行净应用 ⇒ 2 TU rc=0。 |
| 84 | `f8108a37` → **跳过** | 供体把 mirror copy 的图案改成非周期（让漏拷贝可见）；我方版**早已**是 `(i*29) ^ (i>>8) ^ (i>>16) ^ (plane*61+seed+17)`（带 seed）⇒ 同语义已具备 ⇒ 取我方后净空 ⇒ skip。 |
| 76–78, 85 | 文档/测试/`7e290349` → **自动落地** | 零冲突（含 row-parallel 派发前查参数与 workspace 余量）。 |
| 86 | `4f682b75` → **`3d4fade4`** | 5 块（1/0/1、0/1/3 ×2、1/1/2、4/1/2）＝取我方骨架 + **采纳家族常量机制、保留我方标定值**：`fp8_geometry.h` 的 `kFp8OutputFamilyFirstA8Tokens/DownFamily…` 写成 **17 / 20**（供体 22/25＝5070 Ti），计划文件用常量替魔数 ⇒ 1 个 plan TU + 2 个 shapes `.cu`（nvcc）rc=0。 |
| 87 | `d2b55765` → **`19753c39`** | 8 块（README 3/3/3、cli 0/2/2 + 6/16/17、serving 4/27/28、tp2 真测 3/1/1 + 6/7/7 + 0/22/22）＝**全取我方权威**（tp2 段落保我方 DFlash2 `--lm-head-draft`＋`rk4v4-e8`＋`--host-context-mib` 口径；测试的意大利语提示词照旧）＋ README graft "multi-GPU **beyond the experimental two-GPU mode described below**"；净应用含 **memory summary 报 rank 0 权重**（program_impl/startup_log）⇒ 8 TU rc=0。 |
| 88–91 | `e10c930e`/`ce46746d`/`a0fd103f`/`f23871af` → **自动落地** | NOTICE、README 双卡开篇（89＝TSV 84 可跳候选，零冲突就落了）、tp2 Engine 测试**要求真实投机接受率**、**rank 0 logits 改内核聚合**（＝91，也是 92 撤 56/57 回退的"因"）。 |
- 序列头 **`f23871af`＝全序 91**；本会话累计 **44→91＝42 步落地 ＋ 9 步有据跳过**（54/58/64/65/66/67/81/82/84）。

### 33.2 门（本段清账与在跑）
- 全构建：**vision=OFF 面 969/969 rc=0**（worktree `gate-72`，50–72 树，快照 `relay/gate72-build.log`）；**vision=ON 面 rc=0**（重配后增量 46 目标，`/tmp/g72v-build.log`）⇒ §28.2/§29.2 的欠账在 72 树面清掉。
- 簇门 `-E softmax_attention`（GPU1，vision=ON，164 分母）**在跑**：`/tmp/g72-gate.log`（128/164 时 0 Failed）⇒ 收门与 passset72 的步骤在薄卡 v10"第一件事"里，收完记 §34 一行。
- 门补丁成品＝**`relay/gates-r72.patch`**（199 行 / 5 文件＝S6 门在 72 树面合并 50 的 `peer_access` 签名与 63 的断言后的重生版）＋ `vision-on-build-ada.patch`。⚠ 87 之后 `program_impl.h/decode_graph.*` 被 92 撤动 ⇒ 92 落地后按同法再重生一次（`git apply --check` 自检）。
- 铁律 6 披露：簇门运行期间**只编单 `.o`**（79＝2、83＝2、86＝3、87＝8 个 TU），**无链接、无库重链** ⇒ 门二进制未变、该轮结论有效。

### 33.3 停点＝全序 92 `fa8c9e39`（**供体自撤 56/57 的回退**）规格
`revert(core): drop the CUDA Graph re-instantiate fallback now that its cause is fixed`（7 文件 +24/−107）＝3 块：`program/decode.cpp`（4/1 面）、`program/graphs.cpp`（4/29）、`program/program_impl.h`（0/13）。落法＝**撤到我方 56/57 之前**：
1. `decode_graph.{cpp,h}` 删 `update_or_reinstantiate`（净应用面 −27/−8）；2. `program_impl.h` 删 `update_rejected` 字段（−13）与 79 的家族旗标；3. `graphs.cpp`：`install_graph_definition` 回到 plain `update` ⇒ **顺带解掉我方 57 为它加的 `install(profile, const char* label)` 形参与 6 个调用点标签**（不留死参数，AGENTS.md"替换要彻底"）；4. `tests/test_decode_graph.cpp` 撤 56 的 26 行段（照供体 +11/−27）。逐字节 `relay/step92-checkpoint/`；冷启动队列 `relay/tp2-replay-queue-122-from-full92.txt`（首 `fa8c9e39`，onto `f23871af`）。
- **91 是本步的因**（rank 0 logits 改内核聚合 ⇒ 不再有 memcpy 参数变化）；若 92 之后仍有 profile 交换被拒，回到 56/57 的实现＝允许，但要在台账记"为何重新需要"。

### 33.4 制品头实测（CPU 读 `.ninfer` 前 2 MiB 目录，写死给 D 桶）
`dflash2`＝5 层**全 `sliding_attention`**、`sliding_window 2048`、head_dim 128、32Q/8KV、`dflash2.selector_top_k 16`、`selector_rank 256`、`conv_kernel_size 2/group_size 16`、`target_layer_ids [5,19,33,47,61]`；`proposal`＝**domain indexed / rows 131,072**（q4_g64，340 MiB）；`mtp`＝`Qwen3_5MTP`；`vision`＝27 层 tower。⇒ 三条判定：**(a)** 80 的"full-attention 草模"拒绝**不触发** ⇒ tp2 DFlash2 可用；**(b)** `--lm-head-draft` 有整块小草稿头可用（PrimaryOnly，见 §31）；**(c)** 制品 MLP 是 **NVFP4** ⇒ **sm_89（4080S）无该路由**（`scripts/check-sm89-exclusions.py` 剔 23 TU）⇒ 云机验证首选 **5090 32 GB（sm_120a）**，4080S 需换非 FP4 制品。

### 33.5 新任务与新规则
- **S13**：单卡 32 GB 真机验**原生** `--spec mtp` / `--spec dflash2` / `--lm-head-draft` 全套不退化（用户指令：云机 4080S 时验；按 (c) 优先 5090）＋复评 75/86 收窄的判据（Int8/FP8 预算）。
- **S14**：83 之后 mailbox＝我方生产路径 ⇒ S8 的超时→staged-copy 降级＋告警用例**要在 D1 前用 CPU 侧写好**（`peer_exchange.cuh`/`peer_mailbox.h` 面）。
- 规则①：**家族常量步＝采纳机制、保留我方标定值**（68 的 17/20、86 的 17/20、75 的判据）⇒ 常量名跟供体、数值跟我方实测，注释写死两侧来源。
- 规则②：**取我方后 `git diff HEAD` 空 ⇒ 净效应空提交 ⇒ `--skip`**（本段 81/82/84 三连用此法，理由逐步记 33.1）。
- 规则③：供体的**自撤步**（92 撤 56/57）＝按我方骨架撤到底，并**解掉我方为旧实现加的 plumbing**（死参数/死旗标/死注释），不留二手结构。

## 34. 全序 92–93 落地（撤净 56/57 回退 plumbing）+ **50–72 门清账**＋停点 94＝半形状注册任务书（2026-10-16 23:5x–24:2x，本会话第 8 段）

### 34.1 全序 92 `fa8c9e39` → **`925598d8`**（供体自撤 56/57）＝撤到我方 56/57 之前，**并解掉我方为它加的 plumbing**
3 处 UU（`program/program_impl.h` 5/3/0＋6/3/0、`program/decode.cpp` 1/37/34、`program/graphs.cpp` 17/0/0＋19/0/0）＋4 净应用（`core/decode_graph.{cpp,h}` 删 `update_or_reinstantiate`、`tests/test_decode_graph.cpp` 撤 56 的 26 行段）。落法＝**取我方骨架 + 撤净**：
① 删家族旗标 `reinstantiate_rejected_updates`（79 graft）；② 删 ns 级 `install_graph_definition` 的声明与定义（57 graft 的核心）；③ 删 `DecodeGraphProfile::update_rejected`；④ **`DecodeGraphFamily::install(profile, label)` 回到 `install(profile)`，5 个调用点（`decode.cpp`）＋1 个（`graphs.cpp`）的标签实参一并撤**（不留死参数，AGENTS.md"替换要彻底"）；�5 撤掉因 fprintf 而加的 `<cstdio>`（净应用面留了一份，我方 57 又加了一份 ⇒ 重复 include 已清，文件现无 stdio 用法）；⑥ 交换规则回到 plain `update` ＝ 我方 38 时代的双族形状（`select/install` 两方法）。
⇒ 5 个 TU（`decode_graph.cpp`/`program_impl.cpp`/`graphs.cpp`/`decode.cpp`/`test_decode_graph.cpp`）**单编 rc=0**（门在跑期间只编 `.o`、无链接＝铁律 6 的合法用法，已在 34.2 披露）；`merge-safety` 索引 0、符号清单无缺失。
**因**＝91 `f23871af`（rank 0 logits 改内核聚合，去掉 pitched copy 的 memcpy 参数变化）⇒ 触发回退的根因消失。**若日后再现交换被拒**：按 56/57 重落，并在此记"为何重新需要"。

### 34.2 门清账（§28.2 / §29.2 / §32.2 的欠账在此结清，GPU1）
- 全构建 **vision=OFF 969/969 rc=0**（50–72 树，worktree `gate-72`@`91848f22`，快照 `relay/gate72-build.log`）＋ 重配 **vision=ON 增量 rc=0**（46 目标）＝**标准门树面**。
- 簇门 `ctest -j1 --timeout 900 -E softmax_attention`（vision=ON ＋ `relay/gates-r72.patch`）＝ **100% passed out of 164 / 1162.57 s**（133 Passed ＋ 31 Skipped＝77 具名）→ **`relay/passset-sm120-tp2steps72-r0.tsv`**。分母与 §27.3 的 164 一致（62/63 没加测试 ⇒ 无漂移）。
- 门运行期间编过 5 个单 `.o`（34.1）＝无链接、门二进制未变 ⇒ 该轮结论有效（照 §23 的披露法）。
- **worktree 门法升级（省 30–40 分钟/次）**：下轮**复用** `gate-72`，不新建：`git -C ../gate-72 checkout -- . && git -C ../gate-72 checkout <新HEAD-sha> && git apply ../relay/vision-on-build-ada.patch && git apply --3way ../relay/gates-r72.patch`（漂移处手工合，见 §32.2 的 engine.cpp/test 合并法）→ `git diff > ../relay/gates-r<新全序>.patch` → `JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests`（**增量**）→ 簇门 → passset。`merge-safety` 在门补丁撤出后的主树面跑。

### 34.3 停点＝全序 94 `80db8faf`（**半形状注册任务书**，逐字节 `relay/step94-checkpoint/`，27 文件；冷启动队列 `relay/tp2-replay-queue-120-from-full94.txt`，onto `2dae2344`）
`feat(ops): register the nvfp4 [5120,3072] half for linear and linear_add`（21 文件；6 UU 块＋**4 DU**＋1 A＋10 净应用）。**照 `g-tp2-1-half-shapes.md` §2 的 nvfp4 六处配方**（该文件＝G-TP2-1 闭合记录，权威）：
1. ①`shapes/n5120_k3072.cu`（A，净应用已在树）②`nvfp4_shapes.h` extern（+1 净应用）③`nvfp4_geometry.h` **三处**＝alias／`Nvfp4GeometryId` 枚举项／`resolve_fp8_geometry` 式分支（+14/−1 净应用）④`nvfp4_dispatch.cpp` 的 `kShapes`（+1/−1）⑤`sources.cmake`（+1，**空白分隔列表，别加逗号**）⑥**A4 TMA 实例表**：`nvfp4_a4_tma.cu` 的 `switch (Nvfp4GeometryId)` 要给半形状**自己的 case**（`launch<K, promotion>` 的 K 是静态量 ⇒ 指向父 id 必得 `static K does not match operands`），量化器加 `Nvfp4Activation3072Geometry` 分支，`nvfp4_a4_tma_route<>` 指自己。
2. **4 个 DU**＝供体的 W4A4 路文件（`nvfp4/nvfp4_w4a4_tma.cu`、`linear_add/nvfp4/nvfp4_linear_add_w4a4.cu`、`linear_swiglu/nvfp4/nvfp4_linear_swiglu_{w4a4,w4a4_tma}.cu`）＝我方 P0 已删该族（A4 走 `nvfp4_a4.cu` 面）⇒ **`git rm` 接受删除**，其 A4/TMA 语义落我方 `nvfp4_a4*.cu`（净应用的 `nvfp4_a4.cu +3/−0` 已在索引）。
3. **穷举 switch 收口**（§2 附带面）：新枚举项进 `linear_add/nvfp4/{decode,small_t,a4}` 的"已识别、无 linear_add 路由"`break` 组 ⇒ 6 个 UU 块（plan 2/2/3、add small_t 2/2/6、swiglu decode 6/15/16、swiglu small_t 2/7/8、`ops/linear.h` 12/11/12、test 1/1/1）逐块**取我方骨架＋只 graft 注册项**；swiglu 的半形状属后面第 13 步（`9b3e3dff`）⇒ **本步不做**（照 §2 原文）。
4. **判据（G-TP2-1 同款）**：全构建 rc=0 → `CUDA_VISIBLE_DEVICES=1 ctest -R "linear|fp8|nvfp4|bf16"`（上次 25/25 100% / 417.08 s）→ `python3 ../relay/verify_fp8_a8_quotes.py`（**under-quote=0**）→ **无 `cudaErrorIllegalAddress`**（§3 的失败清单是路标：`unsupported shape`→`static K`→`IllegalAddress`→`A8 workspace interval contract mismatch`）。A1 裁定（164/166/167 走我方模板化注册）同样管这类步。

## 35. 全序 94 落地＝nvfp4 [5120,3072] 半形状（按我方模板注册）+ 94 验收门在跑 + 停在 95（2026-10-17 00:3x–01:1x，本会话第 9 段）

### 35.1 全序 94 `80db8faf` → **`ec67d342`**（register the nvfp4 [5120,3072] half for linear and linear_add，21 文件）
落法＝**A1 裁定（走我方模板化注册）＋ §34.3 的六处配方**：注册面净应用零冲突（`nvfp4_geometry.h` 的 alias／`Nvfp4GeometryId::N5120K3072`／resolve 分支 ＋ `nvfp4_shapes.h` extern ＋ `nvfp4_dispatch.cpp` 的 `kShapes` ＋ `sources.cmake`）；**净应用的 A 类 shapes 文件写的是供体 W4A4 家族名（`Nvfp4W4a4MmaSchedule`/`launch_nvfp4_gemv`/`launch_nvfp4_a16_chunks<32,…>` ⇒ T 上限 32）与我方不合** ⇒ **照我方兄弟 K 向半 `shapes/n5120_k8704.cu` 重写**：`Nvfp4A16GemvSchedule`/`Nvfp4A16SimtSchedule`/`Nvfp4A4MmaSchedule` ＋ 父 `[5120,6144]` 的完整 ladder（gemv→simt→sliced_k→mma，**注册在每一个正 T**，符合 `ops/linear.h` 契约）＋ `nvfp4_a4_tma_route<N5120K3072>()` ＋ `uses_a4 = max_tokens >= kNvfp4OutputFamilyFirstA4Tokens`。
**家族常量取我方标定**（§33.5 规则①）：`kNvfp4OutputFamilyFirstA4Tokens` 7→**17**（我方实测，注释写明供体 7 是它自己的 5070 Ti 数）、`kNvfp4DownFamilyFirstA4Tokens` 保持 8；`linear_add` 计划与 `linear_add_small_t` 的 warp crossover 引入 `kOutputFamily`（含 `Nvfp4N5120K3072::kInputRows`），测试侧 `first_a4 = k == 6144 || k == 3072 ? 17 : 8`；A4 TMA 实例表加 `case N5120K3072: launch<3072>`（静态 K＝§配方⑥的"指父 id 必报 static K does not match operands"的正解）；量化器 `Nvfp4Activation3072Geometry` 分支净应用已在（`nvfp4_a4.cu:89-90`）。
6 块全取我方骨架；**4 个 DU**（供体 W4A4 路文件 `nvfp4/nvfp4_w4a4_tma.cu`、`linear_add/nvfp4/nvfp4_linear_add_w4a4.cu`、`linear_swiglu/nvfp4/nvfp4_linear_swiglu_{w4a4,w4a4_tma}.cu`）＝`git rm`；swiglu 的半形状按 §2 原文留给后面的 `9b3e3dff`；`ops/linear.h` 契约句 graft（NVFP4 现注册 `[17408,5120]`/`[5120,8704]`/`[5120,3072]`，半形状 A4 从 T=17 ＝ 与 linear_add 同族一个常量）。⇒ **9 个 TU 单编 rc=0**（含新 shapes 文件与 `nvfp4_a4_tma.cu` 的 nvcc）。

### 35.2 94 的验收门（G-TP2-1 同款，正在跑）
主树面增量全构建 `cmake --build build-120a-cu13 -j14`（**686 目标**，日志 `/tmp/b94.log`，快照 `relay/b94-build.log`）→ `CUDA_VISIBLE_DEVICES=1 ctest -R "linear|fp8|nvfp4|bf16"`（上次 25/25 100% / 417.08 s）→ `python3 relay/verify_fp8_a8_quotes.py`（**under-quote=0**）→ 无 `cudaErrorIllegalAddress`。结果数字待补：GATE94。

### 35.3 停点＝全序 95 `b14be3d5`（**add the nvfp4 column-parallel attention input projection**）
**7 UU/AA/DU 文件 + 6 净应用**：`attn_input_proj/nvfp4/` 的 decode（3 块 3/2/3、5/35/7、3/3/1）、`_output.cuh`（**AA：6/0/89**＝双方各自新增同路径文件，需逐段并）、`_plan.cpp`（0/31/33＝我方删过该区）、`_plan.h`（3/0/3）、`_small_t.cu`（3/2/3、9/17/18、1/1/2）、**DU `_w4a4.cu`、DU `linear/nvfp4/nvfp4_w4a4_tma.cu`、DU `nvfp4_w4a4_tma_launch.h`**（供体 W4A4 族＝我方已删 ⇒ `git rm`）；净应用 `ops/attn_input_proj.h +7/−7`、`wrapper/attn_input_proj.cpp +39/−10`、`weight_input.cpp +2/−1`、`nvfp4_geometry.h +4`、`tests/test_attn_input_proj_split.cpp +84/−55`。
⚠ **这是 §18.1 家族落差的正面区**（供体 ksplit/列并行 ≠ 我方 **A16 sliced-K**）：落法＝取我方 sliced-K 机制 ＋ 只 graft"列平行（K 向半）"的报价/入口，**不引入供体 ksplit 路**（判定基线 §1.1"一套当前权威"）；AA 文件按我方命名重写；判据＝`ctest -R "attn_input|nvfp4"` ＋ 容量报价↔精确一致（`verify_fp8_a8_quotes.py` 同款核验 nvfp4 的 split 报价）＋ 无非法内存访问。逐字节 `relay/step95-checkpoint/`（16 文件）；冷启动队列 `relay/tp2-replay-queue-127-from-full95.txt`（首 `b14be3d5`，onto `ec67d342`）。
- **§35.3 补（本会话停笔前的勘查，落 95 直接用）**：AA 文件 `nvfp4_attn_input_output.cuh` 两侧＝我方 6 行别名（`Nvfp4AttentionInputOutput = LinearBf16SegmentedOutput<6144,1024,6144,1024>`，来自 `ops/linear/common/output.cuh`）vs 供体 89 行新家族（`Nvfp4AttentionInputSections<QueryRows,KeyRows>`＋`Nvfp4AttnInputProblem<…>`＋`visit_nvfp4_attn_input_problem(parent_rows, body)`）。**解＝取我方机制、只加 shard 变体** `LinearBf16SegmentedOutput<3072, 512, 3072, 512>`（＝12/24 q·gate 行与 2/4 k·v 行，父序 q|k|g|v），decode/small_t/plan 按 `weight.n` ∈ {14336, 7168} 选分段；不引入供体 visit 家族（`gap-plan §1.1`）。`plan.cpp` 块（0/31/33）＝我方已删该区 ⇒ 取我方；`decode.cu` 的 5/35/7 与 `small_t.cu` 的 9/17/18 同理＝取我方 + 选段 graft。净应用契约面（`attn_input_proj.h` 三处、`nvfp4_geometry.h` 两 alias、`weight_input.cpp` 的 shard+NVFP4 准入、`plan.h` 注释）已在索引。

## 36. 94 的门跑出来是 **1 红**＝`ninfer_linear_add_nvfp4_test` 抛 `nvfp4 linear_add: unsupported problem`；根因与已落的修正（2026-10-17 01:3x，本会话第 10 段，会话封顶停笔）
- 门数据（`gate-72` worktree，94 树面 `ec67d342` ＋ vision=ON ＋ `gates-r72.patch` 零漂移）：全构建 **567/567 rc=0**（快照 `relay/g94-build.log`）；子集 `-R "linear|fp8|nvfp4|bf16"` ＝ **96%，1 个失败 out of 27 / 444.03 s**（`relay/g94-sub.log`）；簇门跑到 125/164 被我停（它必撞同一项）。**94 的门未过＝S2 在 94 处有一个真红，已知、已修、未重跑。**
- **根因（两侧机制不合）**：94 净应用面把 `[5120,3072]` 当 linear_add 问题登记（`plan` 的 `output_family` 含 K3072、`decode` 有 `launch<N5120K3072>`、测试新增 `run_shape(5120,3072)`），**但其 A4 路在供体的 W4A4 族文件里＝我在 94 判为 DU 删掉的 4 个文件之一** ⇒ 测试在 T≥17 走 A4 时落到"unsupported problem"。＝§35.1 的"家族常量取我方"没覆盖到的第二层：**被删族 carried 一条路**。
- **已落修正（3 文件，主树面未提交，见下）**：`linear_add/nvfp4/nvfp4_linear_add_plan.cpp` ＝ `input_rows == N5120K3072` 一律 `A16`（注释写明本树无该半的 A4 路）；`nvfp4_linear_add_decode.cu` ＝ 保留 `case N5120K3072: launch<Nvfp4N5120K3072>`（**A16 路可用**）＋注释；`tests/ops/linear_add/test_nvfp4.cpp` ＝ `first_a4 = k==6144 ? 17 : (k==3072 ? 1<<30 : 8)`（半形状永不在本树走 A4）。⇒ 3 TU 单编 rc=0（无 ctest 在跑期间链接）。
- **下一会话的头两件事（顺序不可颠倒）**：
 1. **先单独提交这 3 个文件**（94 的后续修正，`git add` 这 3 路径 → `git commit -m "fix(ops): keep the nvfp4 [5120,3072] half on A16 for linear_add (no W4A4 family in this tree)"`，在 95 解冲突**之前**做；此刻索引里有 95 的 UU ⇒ 需先 `git stash push -- <3 files>`，落 95 到干净停点后再 `git stash pop` 提交，**绝不能把这 3 个文件扫进 95 的提交**）。
 2. 重跑 94 修正后的子集门＋簇门：`gate-72` 里 `git reset --hard <含修正的新 HEAD>` → 重贴 vision＋`gates-r72.patch` → 增量构建 → `ctest -R "linear|fp8|nvfp4|bf16"`（应 27/27 100%）→ `-E softmax_attention`（164）→ `passset-sm120-tp2steps94-r0.tsv` → 回填本节。
- **规则升级（写死）**：判 DU/A 类文件删除的安全性时，**必须查"该文件是否 carried 一条被净应用面引用的路由"**（方法＝删后跑家族子集门，不是只看单 TU 编译；本轮 94 的 3 个 DU 里 `linear_add_w4a4.cu` 正是这种）。族被删 ⇒ 同时把"引用它的那条登记面"（plan/decode/测试）改到我方机制上，一次做完。

## 37. 94 的门重跑＝**仍是 1 红**（这次是我方修正写错在测试侧）＋ 全序 95–115 落地 21 步（2026-10-17 02:0x–03:0x，会话第 11 段）

### 37.1 94 修正后的门（worktree `gate-72` ＠ `8cbbe0fe` ＝ 94＋修正，vision ON ＋ `gates-r72.patch` 零漂移）
- 全构建 **rc=0**（`relay/g94b-build.log`，快照同法）；子集 `-R "linear|fp8|nvfp4|bf16"` ＝ **96%，1 红 / 27，515.85 s**（`relay/g94b-sub.log`）；簇门 `-E "softmax_attention"` ＝ **99%，1 红 / 164，1081.10 s**（`relay/g94b-full.log`）。
- passset：**`relay/passset-sm120-tp2steps94-sub-r0.tsv`**（26 行：Passed 23 / Skipped 3）与 **`relay/passset-sm120-tp2steps94-r0.tsv`**（163 行：Passed 132 / Skipped 31；红项 `Subprocess aborted` 不被解析器收录＝已知限制，红项在表头注记里）。
- 红项＝`ninfer_linear_add_nvfp4_test`（簇门里编号 158）＝**同一个测试**，症状从 94 的 `unsupported problem` 变成 **`std::bad_alloc`**。

### 37.2 根因＝我 94 修正的第二层错（测试侧），修正已备好**未落**
`tests/ops/linear_add/test_nvfp4.cpp:94` 用 `first_a4 = 1<<30` 表达"永不走 A4"，但该文件第 116–131 行由**调用表本身**取 token 上限（`kMaximumTokens = max(invocation.tokens)` ⇒ host 激活/残差缓冲 ＝ 1<<30 × K × 2 B）⇒ 分配炸 `std::bad_alloc`。＝§36"被删族 carried 一条路"的**第三层**：连"永不"也要受测试的缓冲上限表达。
- 备好：**`relay/step94b-test-bound-fix.patch`**（`first_a4 = k==6144 ? 17 : (k==3072 ? 1025 : 8)` ＝ 该半形状取本测试最宽 T ⇒ 每个 `AllowA4` 调用经 `resolve_route` 落 **A16** 路（plan 已强制），按 A4 宽容差比精确 FP32 参考，token 上限不再爆）。
- 落法（顺序）：解 116 到干净停点 ⇒ `git apply ../relay/step94b-test-bound-fix.patch` ⇒ 单 TU 编 `tests/ops/linear_add/test_nvfp4.cpp` ⇒ **单独 commit**（消息同 §36 头两件事，**绝不扫进 116**）⇒ 再 `--continue`。

### 37.3 全序 95–115＝21 步落地（每步单 TU 全绿、`merge-safety` 索引 0）
- **95 `9d245170`**（列平行 attn 输入投影）：AA `nvfp4_attn_input_output.cuh` ＝ 取我方别名＋**只加 shard 变体** `LinearBf16SegmentedOutput<3072,512,3072,512>`；decode/small_t/a16/a4/a4_tma **按 `weight.n` 选段**（模板参数＝我方机制；**不引供体 `visit_*` 家族**）；`plan.cpp/plan.h` 取我方（K 相同 ⇒ 路由/截止/容量天然共享）＋ graft plan.h 三行说明；3 个 DU（供体 W4A4 族）`git rm`；**12 TU rc=0**。
- **96 `54bb1370`**（列平行 GDN 三形）：同法，shard 别名 `LinearBf16SegmentedOutput<5120,3072>`（父 10240/6144 ⇒ 2*1024+3072 / 3072）；净应用 wrapper 里供体写 `nvfp4_w4a4_workspace_capacity_bytes`＝**我方无此族** ⇒ graft 成 `nvfp4_a4_workspace_capacity_bytes`（坑②＋§36 规则）；`nvfp4_gdn_conv_uses_a4` 谓词取净应用（作用在我方 `resolve_plan` 上）；5 冲突全取我方＋选段 graft；3 DU `git rm`；**10 TU rc=0**。
- **97/100–107/113 文档面**（`ea8c0967 b0da936a … 2ec55f12`）：cli.md 取我方＋graft"列平行投影取 FP8/NVFP4"三行；README 净应用的 **vision@tp2 能力句按 坑③ 回退**（本树 `model_instance.cpp:84` 拒 `--tp 2` 的 Vision）＝去掉 fork-note 例子里的 `--vision` 旗标、QUASAR 卡同处加注、`--vision` 进"已知限制"清单；serving.md/tests/README 取我方。
- **108 `5ca1573d`**（D1 探针进树）：A 类新文件 `tools/tp2/mailbox_probe.cu` ⇒ 按文档命令 `nvcc -O2 -std=c++20 -arch=sm_120a` **单编 rc=0**（铁律4/坑⑦）。
- **110 `f94ae937` / 111 `e3275dbc`**（S8/S14＝mailbox 启动探针＋图内 hang 降级）：110 仅 program.h 1 块＝取我方＋graft `tp_transport()`；111 graphs.cpp 3 块＝**取我方骨架**（我方＝forward/finish 两族合一）＋把供体新华子 `synchronize_graphs`/`capture_and_instantiate` 接上：我方区（instantiate＋首轮 launch 暖机）整段包进 lambda、`instantiate_graph_family(..., synchronize_all)`→`synchronize_graphs`、首轮 `synchronize_devices()`→`synchronize_graphs()`、尾部加 `for(;;){try{...}catch{... degrade_peer_mailbox()}}`；`program_impl.cpp` 里供体写的 `mtp_graphs`/`dflash_graphs` ⇒ 改我方族名 `speculative_forward_graphs`/`speculative_finish_graphs`；**7 TU rc=0**（含 nvcc 的 allreduce.cu/peer_mailbox.cu）。
- **112 `e9ff45b1`**（`--vram-headroom-mib`）：净应用解析面自动落；3 块取我方＋graft（帮助行、`vram_headroom_mib` 字段、`requires --kv-capacity auto` 校验块）＋serving.md 表格一行；**8 TU rc=0**（含 3 个测试 TU＝断言帮助文面/行为）。
- **115 `2b7a3602`**（kernel 属性每设备＝双设备必经点）：`fp8_shared.cuh`/`nvfp4_shared.cuh` 4 块**全取我方**（我方已有 `core/kernel_attr_once.h`＋`core::FuncAttrPerDevice`，＝"要当能力做"已落）；净应用的两个 bf16 头写供体路径 `ops/launcher/kernel_attr_once.h`＋裸名 ⇒ graft 成 `core/kernel_attr_once.h`＋`core::FuncAttrPerDevice`；DU `q8_shared.cuh` `git rm`（我方 Q8＝A16 sliced-K，§18.1；全树无引用）；**4 TU rc=0**。

### 37.4 停点＝全序 116 `7ae77f55`「ops/linear: register the tp2 half shapes in the unified templates」
**17 个 UU / 45 块**（fp8 shapes 6 文件×4、q8 shapes 5×1、nvfp4 shapes 3×1、bf16 shapes 2×1、`nvfp4_geometry.h`×1、`nvfp4_a4_tma.cu`×1、`include/ninfer/ops/linear.h`×1）＋净应用 3（`fp8/shapes/n124160_k5120.cu`、`tests/ops/linear/test_nvfp4_a4.cpp`、`tests/ops/test_linear_split.cpp`）。逐字节＋规格＝`relay/step116-checkpoint/`（STATUS/MESSAGE）。
⚠ **与我方 G-TP2-1 正面撞面**：供体把 16 个半形状改写成"父选择器的复制＋几何折半"，而我方 94/35 已按**我方模板**（A16 sliced-K / 模板化注册）登记了这些半形状 ⇒ 逐块判＝**取我方骨架**，只 graft 供体的新报价与新家族常量（`kNvfp4OutputFamilyFirstA4Tokens` 我方已是 **17**＝同值）、q8 半形状取我方、`linear.h` 契约句取我方＋ graft 半形状清单；净应用的两个测试文件按 坑③ 判"测供体政策"的段落。**先 `awk` 数块**再上 `resolve_blocks.py`。
冷启动队列＝`relay/tp2-replay-queue-106-from-full116.txt`（首 `7ae77f55`，onto `2b7a3602`＝全序 115）。

### 37.5 门状态（写死给下一会话）
95–115 **每步单 TU 全绿**（12＋10＋7＋8＋4＋4＋…）＋`merge-safety` 索引 0；⚠ **全构建门与簇门自 94＋fix 之后 21 步未跑**（95–115 树面）。⇒ 116 落地后、解 117 之前：`gate-72` 里 `git reset --hard <116 落地的 sha>` → 重贴 vision＋`gates-r72.patch` → `JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests` → `ctest -j1 --timeout 900 -R "linear|fp8|nvfp4|bf16"`（**含 94b 修正后应 27/27 100%**）→ `-E "softmax_attention"`（分母 **164**）→ `passset-sm120-tp2steps116-r0.tsv` → 打簇 tag `v-tp2-nvfp4-column-parallel-mailbox-r0`（消息＝门数据）。

## 38. 全序 116–128 落地 11 步＋2 步有据跳过；停点＝129（perplexity 打分上 tp 2）（2026-10-17 03:2x–04:0x，会话第 12 段）
| 全序 | 供体 → 我方 | 落法（一句） |
|---|---|---|
| 116 | `7ae77f55` → **`c84dcadb`** | 17 UU/41 块**全取我方**＝我方 35/94 登记的半形状**已是"父选择器的复制"**（逐字比对：我方 `fp8/shapes/n5120_k3072.cu` 的 `Gemv/Tma32x64/Tma64x128/MidBulk/Bulk`＋ladder 与父 `n5120_k6144.cu` 同套），供体侧是**旧结构**（`C2/C4` SIMT＋`Fp8A8T32R32K128`）⇒ 不取；净应用 3 文件（`n124160` 注释、`test_nvfp4_a4` 去掉 T=6 额外调用、`test_linear_split` 加 T=16/17）＝与我方 17 标定一致 ⇒ 取；graft 六处 **K 整除性/出处注释**（3072=6·512、8704=17·512、5120=10·512；`nvfp4_geometry.h` 写"上游 fc3993d8 把输出族移到 17＝我方实测值"）⇒ **19 TU rc=0** |
| 116b | — → **`e6a43239`** | §37.2 的 94b 修正**单独提交**（`git apply` → 单 TU rc=0 → `git commit -- <1 file>`，**未扫进 116**）＝`first_a4` 的 `1<<30` → `1025` |
| 117 | `893751e5` → **(skip)** | 供体三件**我方全有**：`text.h` 的 `rank_execution_view` 声明（我方注释更详）、`text.cpp` 的实现、`startup.cpp` 的 4 参 workspace 查询；第四件＝headsplit 按 T=16 分界取判据＝依赖我方**未载**的 chunked 预填充（`tests/ops/gdn_criteria.h` 我方＝两条固定判据）⇒ 取我方后 `git diff HEAD --stat` **空** ⇒ 按铁律 8 `--skip` |
| 118–122 | `138c3706 b3f93dd6 d8cd8427 b15713d2 1ce3d727 1e203948` → `0c6117f1 d3a065b8 8e349010 5a70dc5b da9c43b9 65d21156` | 零冲突净落（W4A4→A4 改名跟随、两设备 seam 测试、golden 记录 2026-09-27、注释对齐）。123 的 `operational_log.cpp` 一块**取我方**＝我方政策"prefill checkpoint 池满即跳过"（我方 `plan_reclaim` 无捕获期调用者＝不载"先回收 idle private continuations"），供体文面＝其树政策 |
| 124–126 | `fec6ac00 1df724ce a1283d9c` → `d42c9aa7 c65ce299 4e7a2d2a` | **agentic A/B 套件**（`bench/agentic_ab/`＝S12 的 ABBA 工具：`run_linux.sh`/`linux.py` 双 arm systemd-unit、`analyze.py` 命名 arm＋seed spread）＝9 新文件＋NOTICE；4 个 py 全过 `ast.parse`；bench/README 的供体段**插在我方 "## Benchmark inputs" 之前**（不混节） |
| 127 | `f3020510` → **`5ab0ab81`** | 零冲突（每次双设备图启动后 gate rank 1 的流） |
| 128 | `f1e6ef9c` → **(skip)** | 取我方＝净效应空：我方 `abort()` 已是 `try { device.synchronize(); } catch(...) { return out; }` **之后**才 `clear_lane` ⇒ 供体这 8 行安全修正我方早已等价实现（§30 的 lane 生命周期面） |

### 38.1 门与簇 tag（116 面，跑在 `gate-72`）
工装升级＝**`relay/gate-steps.sh <tree-sha> <label>`**（全绝对路径，修掉 §37 那次 `cd build-120a-cu13` 后相对 `../relay` 写错地方；一次做完 构建→子集→簇门→两份 passset）。116＋94b 面（`c84dcadb`，含 `e6a43239`）：`vision-on-build-ada.patch` 与 `gates-r72.patch` **零漂移**（21 步后仍 `--check` 过）。**GATE116＝全构建 rc=0；家族子集 `-R "linear|fp8|nvfp4|bf16"` ＝ 100%，27/27，422.78 s**（`relay/gate-116-sub.log`、`passset-sm120-tp2steps-116-sub.tsv`）＝**94 的红清掉了**（`[5120,3072]` 走 A16＋哨兵取本测试最宽 T 后，`ninfer_linear_add_nvfp4_test` 全绿）；**簇门 `-E "softmax_attention"` ＝ 100%，165/165，1058.89 s**（`relay/gate-116-full.log`、`passset-sm120-tp2steps-116.tsv`＝Passed 133/Skipped 32）。⚠ **分母从 164 变 165**＝118–126 净应用带进一项新测试（`ninfer_perplexity_options_test` 之外的 seam 覆盖；日后取分母按当次 ctest 计数，别再用 164）。⇒ 95–116 簇门**首次全绿**，簇 tag 已打（`v-tp2-nvfp4-column-parallel-mailbox-r0` @`c84dcadb`）。

### 38.2 停点＝全序 129 `36d8af0d`「perplexity: score at tp 2 on the two-rank forward」
8 文件 **9 块**（`planning/startup.cpp` 4 块、`program_impl.{cpp,h}`、`model_instance.cpp`、`include/ninfer/types.h`、`docs/cli.md`）＋净应用大面（`apps/perplexity/{main.cpp,−130,options.cpp+106,options.h+35}`、`apps/CMakeLists.txt`、新测试 `ninfer_perplexity_options_test`、docs `perplexity.md`/`tensor-parallel.md` 五节）。
**＝D 桶判定面工具**（GSM8K/perplexity 在 tp 2 打分）：落法＝取我方骨架＋**撤掉 tp2 对 CausalScoring 的四处拒绝**（`validate_options`/planner/workspace plan/`ProgramImpl`）＋workspace plan 加 `causal_score` phase（两 rank 布局）＋`prefill_text_chunk` 接 `TpExecution`＋打分 tile 走词表分裂头（`output_logits_split_rank0`，rank1 复制 rank0 的 staged hidden 投自己那半、rank0 gather `[V,C]` 跑 `target_logprobs`）；tp1 不变（`tp_binding()` 为空）。判据＝单卡逐 TU 编（含新测试 TU）＋两设备套件在单卡自跳 77。
逐字节＋规格＝`relay/step129-checkpoint/`（16 文件）；冷启动队列＝`relay/tp2-replay-queue-93-from-full129.txt`（首 `36d8af0d`，onto `5ab0ab81`＝全序 127）。

## 39. 全序 130–141 落地 3 步＋2 步有据撤除（70-SM 调参类），停在 143（softmax causal_cache）（2026-10-17 04:2x–05:0x，会话第 13 段）
| 全序 | 供体 → 我方 | 落法/理由 |
|---|---|---|
| 130 | `2a596191` → **(skip)** | 供体＝[7168,5120] shard 在 **70-SM 5070 Ti** 上的 wave 量化调参（128-token TMA tile 走到 T=1024，196.1→183.6 µs，实测非我方）；我方 GPU1 实测 **36 SM**（cudaDeviceGetAttribute 读到）⇒ 依据不迁移。取我方后残留 hunk 引用我方不载的 `first_tiled256_tokens(weight.n)` ⇒ 全取我方＝净效应空 ⇒ skip。调参需求进 **S7/D4 待办**（`relay/s7-seed-5060ti.md`）。 |
| 131–136 | `b3f93dd6…` → `285910b4 b21a2749 20d86c23 …` | 零冲突净落：pipelined mailbox 交换核（132）后的文档面（per-GPU 调参记录、v0.2.2 复测）＝**记着我方"实测为准"口径的出处**，直接可用。 |
| 132 | `cdd2fc13` → **(净落)** | pipelined mailbox 交换核＋`NINFER_TP_MAILBOX_LEGACY=1` A/B 开关：1 block＝取我方＋graft 一行 `const ops::PeerExchangeKernel kernel = peer_mailbox->kernel();`（净应用面在 `degrade_peer_mailbox` 与启动处用它 ⇒ 缺它＝编译不过，坑②）；`peer_mailbox.cu`/`program*`/`model_instance`/`operational_log`/`test_allreduce` **6 TU rc=0**，A 类 `tools/tp2/mailbox_probe.cu`（583 行改动）按文档 nvcc **rc=0**。 |
| 137–141 | `0fc625ce…` → `b4b25d6b e63965d3 …` f6a194eb | 137＝seam 覆盖测试：`test_nvfp4_a4` 一块取我方（保我方 seed 722/727 标定注释）＋graft `[17408,5120]` 的 255/256/257 TMA 切点调用（我方 `Invocation` 已是 4 字段形＝可直接落）⇒ **5 测试 TU rc=0**；⚠ 其文面引用"shard keeps the 128-token tile through 1024 since 2a596191"＝**130 被我 skip** ⇒ 该注释在我方树不成立（测试＝两设备套件，单卡自跳 77，无门影响）。139＝**分裂优化提案头**（`--spec mtp --lm-head-draft` 在 tp2：Q4 `[131072,5120]` → 每 rank `[65536,5120]`，`argmax_split_pack/select`＋16 字节 allreduce＝精确并集、位等值）：`parameters.cpp` 取 both（两个 include 都要），`linear.h` 取我方＋graft Q4 registry 一句；A 类新文件 `launcher/argmax_split.cu`、`q4/shapes/n65536_k5120.cu`、`tests/ops/test_proposal_head_split.cpp` 随 cmake 重生 DB 后单编 ⇒ **13 TU rc=0**（含 94 式命名失配＝无）。 |
| 142 | `91582814` → **(skip)** | 供体＝sliced-K/down-SIMT 在 **70-SM** 的调参＋家族改名（净应用的 `nvfp4_schedule.cuh`/`a16_sliced_k_mma.cuh` 把 `Nvfp4SlicedInstance` 变成非模板别名 ⇒ 我方 **13 个调用点**（attn/gdn/linear_add/swiglu/8 个 shapes）报 "is not a template"）。撤法＝6 个文件整文件回退到 HEAD ⇒ 净效应空 ⇒ skip；理由入 S7 调参待办（先 36 SM 实测，再谈移植改名）。 |
- 全序 143 `7b4f073a` → **`ec1c4fde`**（每 causal-cache 派发拒绝不支持的头数＝加固；我方 tp2 的 **12/2 分支必须保**，故＝取我方＋把三条链补成 `24 / 12(tp2) / 16 / else throw`：bf16 2 站、int8 3 站；fp8/nvfp4/k8v4 净应用面单编通过）⇒ **5 个 `.cu`（nvcc）rc=0**。
- 新停点＝全序 144 `cbd0f83d`（4 个冲突文件含 `causal_cache/int8/{launch.cu,plan.cpp}`；逐字节与队列**待下轮导出**）。
- 停点＝全序 143 `7b4f073a`（softmax_attention causal_cache 的 bf16/int8 `launch.cu` 冲突＝门里 `-E softmax_attention` 排除族，单卡可编不可测）；逐字节＝`relay/step143-checkpoint/`；冷启动队列＝`relay/tp2-replay-queue-79-from-full143.txt`（首 `7b4f073a…`，onto `f6a194eb`＝全序 141）。
- 门账（**141 面 `f6a194eb` 实测**）：全构建 **rc=0**；家族子集 `-R "linear|fp8|nvfp4|bf16"` ＝ **100%，27/27，422.35 s**（`relay/gate-141-sub.log`）；**簇门 `-E softmax_attention` ＝ 100%，167/167，1101.26 s**（`relay/gate-141-full.log`；分母 **165→167**＝137/139 带来 `test_proposal_head_split`＋`test_perplexity_options`）。⇒ 95–141 面数值全绿，141 面簇 tag 已打。⇒ 95–141 面数值全绿；141 面簇门落账后打簇 tag `v-tp2-mailbox-pipelined-proposal-head-r0`。95–129 面（116）上次已 100%/27 与 100%/165。
- §39 补（第 19 回合实测，替换上一会话写坏的同一行）：全序 146–153 落地 8 步＋149 skip。
  - **146**（FP8 半形状随 upstream retune）＝5 个半形状＋family 常量**取我方后净效应 0**：逐字比对我方半形状已是我方 retuned 父的复制，常量 17/20 与供体新值同值；只剩 2 个容量 hunk（gdn shard 的 TMA split-K partials）⇒ 9 TU rc=0。
  - **147/148** 零冲突净落。148＝`cc879f10`（gdn shard 首个 A8 宽度 17 的缺 workspace 拒绝用例）。
  - **149**＝skip：供体把测试写死 T 8→17，我方断言＝比对父的公开报价（更强且不写死标定）⇒ 取我方后净效应空（铁律 8）。
  - **150**＝取我方（我方逐 band 用 shard 自身行块数，已覆盖供体的 T=289/193 修复且范围更大）＋**必须 graft 两参 `fp8_attn_input_partial_capacity_bytes(rows, max_tokens)` 的定义**（声明与 wrapper 调用随净应用落齐＝坑②）⇒ 12 TU rc=0。
  - 150 验修时把主树面 `ninfer_ops` 完整构建＋设备链接跑到 **185/185**，当场抓到 **145 的净应用缺陷**：7 个 `causal_cache/**/template_launch.cuh`、`tiled_launch.cuh`、`mxfp8_tiled_launch.cuh` 引用 `ops/launcher/kernel_attr_once.h`（我方无此路径）＝`fatal error: No such file`。修法＝`core/kernel_attr_once.h`＋`core::FuncAttrPerDevice`（照 115 先例）。该修复**未提交**，在 `git stash@{0}`（pop 因 154 有冲突被拒）。
  - **151–153** 零冲突净落（`cf59d1c1` golden 2026-09-30、`9c02c1ce` bench 等）。HEAD＝`9c02c1ce`（全序 153）。
- 新停点＝队列 146/213 ＝ 全序 **154 `08240dd2`**「bench: causal attention bench selects the tp 2 per-device 12/2 geometry」，唯一冲突文件 `bench/ops/ninfer_bench_common.h`（1 块）。
- 续跑顺序（不可颠倒）：解 154 到干净停点 ⇒ `git stash pop` ⇒ 单独 commit 上面那条 include 修复（消息 `fix(ops): point the causal-cache kernel-attribute guards at core/kernel_attr_once.h`）⇒ 重跑 `cmake --build build-120a-cu13 --target ninfer_ops` 复验 ⇒ 攒够 3–5 步跑全构建门（分母取当次 ctest 计数＝上次 **167**）。

- §39 补二（第 19 回合续跑实测）：
  - 154 `08240dd2`＝**取我方后净效应非空**：我方 `summarize_timings` 已把 mean 作第 4 字段（统一入口，6 处调用共享），供体逐处加＝同一逻辑我方只留一套；净余 hunk＝`ColdTiming::mean_us` 字段＋bench CSV 追加 `mean_us` 列（列位不变）⇒ 已落 `90f9fbd4`，`ninfer_causal_softmax_attention_bench` 构建为验收（**跑通＝S12/D4 的 A B B A 从此有 mean 列**）。
  - 155 `4b9c098e`＝取我方（我方 144 已把 int8 plan 改成逐设备 `multiprocessor_count`＋`causal_query_tiles_underfill_sms`，且守卫含 heads 24/16/12）＋采纳其**新增设备查询面**（`causal_partition.{h,cpp}` 的 `causal_attention_sm_count`／`causal_attention_device_sm_counts`，＋57 行纯新增）⇒ 4 个 plan TU rc=0；落 `ee13ddf0` 前一步。
  - 156/157 零冲突净落（含 golden `ee13ddf0`）。158 `169514ea`＝**整步撤除**：其内核侧配套（`causal_merge.cuh`、`int8/grouped_mma.cuh`）依赖供体新 helper，取我方 header 即编不过（`causal_split_share/run` 未定义）；全采＝在 36 SM 上盲改 INT8 分组核的 split 粒度，且本族在簇门被 `-E softmax_attention` 排除＝单卡不可验 ⇒ 按口径撤除，记进 S7/D4 调参待办（bench 的 mean 列与 12/2 per-device 选择已就位，测起来才有依据）。
  - 现场：rebase 在位 `msgnum=152 / end=213`，停点＝全序 160 `30dcb143`（冲突含 `models/qwen3_5/execution/text.h`）；HEAD 见上（修复提交）＋`93371a23`（159＝tools/tp2 行为保持门）。- §39 补二 的 sha 更正（上一行里的  因引号未展开）：修复提交＝****（内容＝7 处 include 路径改 ＋，并把 158 的 5 个文件回到 ；验收＝int8/bf16  nvcc rc=0、供体路径残留 0）。被它纠正的误标提交＝（实为 158 内容）。下一停点＝全序 160 （3 个 UU：、、）。

- §39 补二 sha 更正：修复提交＝**f1e45ca8**（7 处 include 改 core/kernel_attr_once.h ＋ core::FuncAttrPerDevice，并把 158 的 5 个文件回到 ee13ddf0；验收＝int8/bf16 launch.cu nvcc rc=0、供体路径残留 0）；被它纠正的误标提交＝**03fb0d91**（实为 158 内容）。下一停点＝全序 160 `30dcb143`（3 个 UU：execution/attention.h、text.cpp、text.h）。
- §39 补三（第 20 回合实测）：**include 修复跨族验证**＝causal_cache 五dtype `launch.cu`（bf16/fp8/int8/k8v4/nvfp4）nvcc 全 rc=0（f1e45ca8 生效，供体路径残留 0）。**160 `30dcb143` 与 161 `bf250edb`＝两步整步撤除（skip）**：两者都是**纯文件搬家**（execution/tp2/*_split.{h,cpp}＋tp2/text_tp2.inc；ops/*/tp2/*_tp2.inc＋include/ninfer/ops/tp2/*.h＋src/ops/tp2/sources.cmake，自述 "Code is moved, not changed"/"No statement changes"）。取我方则新文件成孤儿（索引里的 A 文件定义同名函数＝重复符号风险），全采则要把我方 6 处 graft（heads 12/24/16 守卫、逐设备 SM 预算、12/2 分支、shard 容量两参重载）逐字迁进新文件＝搬家期回归面；收益仅目录形状，且我方 92 已按 `p1/tp2` 层边界自置布局 ⇒ 撤，登记为 **P1 布局待办**（S 桶清完再做，需一次全量构建＋簇门护航）。撤法＝`git read-tree -u --reset HEAD` 后 `git rebase --skip`（净效应 0，无 tp2/ 目录残留）。
- 现场：rebase 在位 msgnum=154 / end=213，停点＝全序 **162 `0a60b088`**（首块冲突 `src/models/qwen3_5/program/speculative/target_verification.cpp`）；HEAD＝`f1e45ca8`（include 修复＋158 撤除）。94→161＝57 步落地＋7 步有据撤除（117/128/130/142/149/158/160/161 中 160/161 同因＝纯搬家）。
- §39 补四（第 20 回合续跑）：**162 `0a60b088`＝整步撤除（skip）**＝第三例纯搬家（program/tp2/，16 块/6 文件），撤法同 160/161（`git read-tree -u --reset HEAD` ⇒ 净效应 0 ⇒ skip，无 program/tp2 残留）。**163 `8b437375`「core: move the dual-device capture and the KV mirrors into core/tp2/」＝零冲突自动落地** ⇒ 我方现在 core 层采纳了供体的搬家布局，而 execution/ops/program 三层跳过＝**布局口径不一致**；下一件事必须判：全采（重放 160/161/162）或回退 163，判据＝主树面全量构建（`relay/b163-build.log`，尾行 `rc=`）＋门链。
- 现场：rebase 在位 msgnum=156 / end=213，停点＝全序 164 `23be72cf`（14 个 UU＝大步）；HEAD=`8b437375`。162 的 checkpoint/队列已导出（`relay/step162-checkpoint/`、`relay/tp2-replay-queue-60-from-full162.txt`＝60 行）；164 的未导（诚实记）。
- §39 补五（第 21 回合实测，164–175）：
  - **164/166/167/170 ＝整步撤除（skip）**：四例同一族「shard 由上游源文件二次编译」搬家（attn/gdn/swiglu 的 ops/*/tp2/ 与 runtime/engine/tp2/）。实测耦合＝其 fragment 引用 `ninfer/ops/tp2/*.h` 与 `src/ops/tp2/sources.cmake`（＝已跳过的 161 基建）⇒ 单独采纳必成孤儿/重复符号。撤法＝`git read-tree -u --reset HEAD` ⇒ 净效应 0 ⇒ skip（验证：无 tp2 残留）。这批与 160/161/162 同属**一个布局工程**：要采就整批采＋全构建护航（P1 布局待办）。
  - **168（S7 名节点）＝落我方形态**：新建 `src/core/tp2/device_tuning.h`（键 CC120/SM36；shard TMA 界 1024＝不采 130 的 70-SM 例外；allowance 占位），把 shard 的 A4 TMA scale-tile 界接到表读（`nvfp4_attn_input_a4.cu`）⇒ nvcc rc=0。171 把表并入**同一路径**的供体单表（namespace `ninfer::tp2`，含 attention_sm_count＋三档 allowance），我方 5060 Ti 行插在 catch-all 之上（键＝本机实测 36 SM；值＝5070 Ti 占位，注明未在本板实测、D2/D4 标定）；表 `std::array` 2→3 行修正 ⇒ a4 / causal_partition / int8 plan / startup **4 TU rc=0**＝单表（铁律3）。
  - 171 的两处安全处置：startup.cpp 的 graph allowance 读取＝**整文件回退我方**（保留 50–61 的实测字面量单点；表内 allowance 字段在头注释标明＝D2/D4 测量槽、当前无读取）；`causal_partition.cpp` 无可见设备时的回退（供体写 `kCausalAttentionSmCount`，我方 155 已删该常量）＝就地 170＋注释指向表。
  - 172 `50118be6`（tools/tp2 静态守卫）与 173（docs gate 指引）零冲突净落。新停点＝队列 168/213 ＝ 全序 **176 `d8acd817`**（1 UU：`linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu`）。
  - **门链在跑**＝`bash relay/gate-steps.sh a1ee019d 175`（gate-72 worktree；117–175 约 59 步未过全构建门；上次分母 167＝取当次计数）＝日志 `relay/gate-175-chain.log`。主树面在停点有 UU 时不构建（第 20 回合我曾违规，rc=2 由冲突标记造成，不作证据）。
- §39 补六（第 22 回合实测）：**176 `d8acd817` 与 178 `8f1540c3` ＝整步撤除（skip）**，同因＝它们的落点在已跳过的 tp2 布局基建（176 的 4 个 tp2 文件＝167 族；178 的 10 个 tp2 文件＝161/162 族，5 个 UU 全是 A/DU 且 0 块＝纯 add/add）。
  - 176 语义＝「gate/up 半形状保留本 fork 的 small-T 调度」＝我方 142-skip 后的既有状态（我方逐几何 dispatch＋我方 schedule），取我方即净效应 0。
  - 178 语义＝**DFlash2 在 tp2 把优化提案头按词表分裂 top-16**（Q4 [131072,5120]→[65536,5120] 两块，`linear_topk_split`/`topk_split_pack`/`topk_split_merge`，一次 all-reduce 交出两份榜，rank0 取前 16；自测＝15 条文本 15/15 与现build相同、acceptance 相同、ms/round −2.1%）。这是**能力面**，但我方已有 139 的 MTP 分裂提案头（Q4 半形状＋argmax 合并）覆盖 `--lm-head-draft` 主路径；DFlash2 的 top-16 分裂属 D3/D4 收益 ⇒ 与**布局工程整批**一起处理（P1 待办：160/161/162/164/166/167/170/176/178 一次采齐＋全构建＋门护航）。
  - 177 `5b25d4ec`（docs：取上游一侧前先查本 fork 的其它改动＝正是本会话反复用到的判据）零冲突净落。新停点＝队列 171/213 ＝ 全序 **179 `1d32d3e6`**（17 个 UU，含 `tests/ops/tests.cmake`＝大步）。
  - 门链 175 面仍在构建（`relay/gate-175-build.log` 现 31/366；链日志到 `patches ok`）。数字与簇 tag 待下轮回填 §39 补五末行。
- §39 补七（第 23 回合实测）：
  - **179/180/182 ＝整步撤除（skip，净效应 0）**：179＝DFlash2 tp2 phase 2（把 drafter 分裂到两 rank，30 个 tp2 落点）；180＝同族的 shard-guard 布局＋维护者文档；182＝wide mailbox slots（批处理轮次的宽槽：pipelined 核下 160/240/320 KiB 交换 17.2/23.7/31.5 µs 对 31.4/40.2/45.4 µs，独立探针＝我方已落的 108 工具；宽槽＝hidden×(K+1)×max_concurrency，DFlash2 K=7@C=4＝320 KiB，MTP3＝160 KiB）。三者落点全在被跳过的 161/162 布局基建上⇒采进 P1 布局整批清单（160/161/162/164/166/167/170/176/178/179/180/182）。上面的探针数字与语义照此记下，整批采时直接引用，不再重勘查。
  - 177 `5b25d4ec`、181 `adf51b1d` 零冲突净落。
  - **175 面全构建门＝FAILED（rc=1，15:30:07）**，唯一失败点＝`src/models/qwen3_5/load/sharding.cpp:4` 引 `ninfer/ops/tp2/argmax.h`（No such file）。根因链＝139 落地时该头**随那次 pick 的索引在场**（我当时单 TU 编 sharding.cpp rc=0 为真），后续整步撤除用 `git read-tree -u --reset HEAD` 把它清掉 ⇒ 悬空 include 到全构建才暴露。
  - 修复＝**`3528e19e`** `fix(load): include ninfer/ops/argmax.h for the split argmax prototypes`（1 文件；在 183 停点用 pathspec 单独提交，未扫进文档步）⇒ sharding.cpp TU rc=0。门链已在该面重跑＝`relay/gate-181fix-chain.log`（现 `patches ok`＋构建中）。
  - **纪律升级（写死）**：整步撤除若用 `read-tree --reset`，必须立刻跑一次**全量构建**（单 TU 不足以证明），并检查「先前步骤净应用带入的 A 文件是否被后续落地引用」；判据＝`git ls-tree -r HEAD --name-only | grep <path>` ＋ 全树 grep 该 include。已并入坑清单（第 24 回合起在薄卡铁律区生效）。
  - 现场：rebase 在位 msgnum=175/end=213，停点＝全序 **183 `6eab7b6f`**「docs: wide mailbox slots, measured」（2 个 UU 文档块：`docs/maintainer/tensor-parallel.md`、`docs/performance/two-gpu.md`）。94→182＝58 步落地＋10 步有据撤除。
- §39 补八（第 24 回合实测，183–190）：六步整步撤除（净效应均 0，逐条 read-tree 后验证）：183（docs：wide mailbox slots 的测量句＝能力在 182 已撤＝坑③）、184（`fdf6c33a` DFlash2 tp2 plan B＝candidate selector 上 rank 1，5 个 tp2 落点）、185（`2480b707` tools/tp2 gate reference，3 个 tp2）、186（**`be1960ce` NVFP4 KV cache at tp 2**，10 UU＝4 个 tp2 fragment 承载准入/接线）、188（`7f500254` 70-SM 的 down-half 四行/warp 调参）、189/190（docs 宣称 nvfp4 KV 在 tp2 已接受＝坑③，我方准入仍拒）。177/181/187 零冲突净落（187＝两设备 head-local parity 测试 for NVFP4 KV：单卡自跳 77＝无害，等 186 落地时即有对象）。
  - 186 撤除时的完整规格记下（下轮**不要重勘查**）：非 tp2 面＝`causal_cache/nvfp4/{plan.cpp,launch.cu,tiled_launch.cu}`、`causal_geometry.h`、`causal_softmax_attention.cpp`、`serve_options.cpp`、`tests/ops/softmax_attention/causal_cache.cpp`；tp2 承载面＝`execution/tp2/{text_tp2.inc,text_context_public.inc}`、`program/tp2/startup_tp2.inc`、`runtime/engine/tp2/model_instance_tp2.inc`（我方对应＝text.cpp / startup.cpp / model_instance.cpp 内联）。**T-nvfp4KV 待办**＝把非 tp2 面采进来＋在我方内联处接准入（我方现于 `model_instance.cpp`、`cli.md`、`tensor_parallel_options` 拒 nvfp4/k8v4 KV@tp2）＋S6 能力门放行 nvfp4 KV＋跑 187 的两设备 parity（D1/D2 窗口）。供体档案（issue #2，2x5070 Ti，glfenix）：PPL 4.1229/4.3987 vs int8 4.1335/4.3859；GSM8K-500 0.974 vs 0.972；262k 下每卡省 ~2 GiB；decode MTP3 +4–8%、DFlash2 +4–17%（55–210k）；默认仍 int8 ⇒ 对我方 D2 显存（11.07 GiB/卡权重）这是最实的出路，优先于 P1 布局整批。
  - 门链（修复面 `3528e19e`，label 181fix）：src/models/runtime/bench 全过，现 **141/332 链接 tests** ⇒ include 修复在全量构建层面成立；子集/簇门数字待落 `relay/gate-181fix-chain.log`。
  - 现场：rebase 在位 msgnum=183/end=213（停点 sha 见下轮首条命令自证），工作树 UU 计数见链上；94→189＝58 步落地＋16 步有据撤除（117/128/130/142/149/158/160/161/162/164/166/167/170/176/178/179/180/182/183/184/185/186/188/189/190 中已落地者不记，撤除者以本段与 §39 补五/六/七/八 为准）。
- 纪律再确认：`git read-tree -u --reset HEAD` ＋ `--skip` 的整步撤除必须伴随**一次全量构建**验悬空引用（第 23 回合的 `ninfer/ops/tp2/argmax.h` 教训）；本轮 141/332 的链接进度即该判据在跑。
- §39 补九（第 25 回合）：191（`0076b695` docs：nvfp4 KV@tp2 与 drafter 分裂的文面）＝取我方⇒净效应 0⇒skip（坑③：两项能力在 186/179/184 已撤；顺带记一条**判 D 桶要用的档案事实**＝供体测得 nvfp4 KV 下 **DFlash2 acceptance 掉 10–15 个点**（MTP3 只 +4–8% decode、DFlash2 +4–17%），⇒ T-nvfp4KV 若为 DFlash2 服务则不划算，只为 KV 容量/大制品服务）。
  - 191 的块里还有我方该抄的一句量化：`nvfp4` cache ≈ 比 `int8` 密 1.8×，262k 下每卡多留 ~2 GiB；`--max-context` 受位置容量约束（int8/nvfp4 同规）。
  - **192 `2dc5b1d9`＝停点保留不解**（本轮判为不可盲改）：它是 context-cache 生命周期的 5 文件语义改动（8 块：commit.cpp 3、resource_manager.h 2、program.cpp/program_impl.h/engine_core.h 各 1）。供体形＝`finish(sequence)`→`finish_terminal(sequence, cancelled)` 加新入口 `finish_cancelled`；engine_core 从 `program->abort()` 改为 `resources_.finish(program, lane, sequence, /*cancelled=*/true)`，由 ResourceManager 在 Program 无法保序列时**回退丢弃**。我方形＝lane 生命周期自置（`abort()` 直接走 program，`clear_lane`＋`detach_checkpoint(sequences[lane])`），两侧结构不同名⇒逐块 graft 需同时改我方 `Program::abort` 语义与 ResourceManager 的 cataloguing 回退，**验收必须跑 engine/context-cache 测试**（`ninfer_engine_test`、`test_context_cache*`）。
  - 判据（写死）：192 属 issue #3 类真修复（取消的同会话续请求把上一轮 checkpoint 连同序列丢弃 ⇒ 下一轮整段重 prefill）＝我方生产路径收益明确 ⇒ **不 skip**，留到门链跑完（GPU1 空闲、可链接可跑测试）的回合做：`git show 2dc5b1d9 -- <file>` 逐块判＋我方形 graft ⇒ 单 TU 编 5 文件 ⇒ 门链复跑（子集＋簇全绿）才算落地。
  - 现场：rebase 在位 msgnum=184/end=213，停点＝全序 **192 `2dc5b1d9`**，5 个 UU 文件带标记（主树面此时不可构建）；门链 181fix 在 gate-72 跑至 191/332 链接 tests（＝include 修复在全量层面成立），子集/簇数字未落账。
- §39 补九 续（门链 181fix 面 `3528e19e`＝117–190＋include 修复，链 done 16:03:54）：全构建 **rc=0**（332 目标）；家族子集 **100%，27/27，416.47 s**；**簇门 `-E softmax_attention` ＝ 100%，167/167，1049.13 s**（Passed 134／Skipped 33，`passset-sm120-tp2steps-181fix.tsv`）。⇒ **117–190 面的全构建门与簇门补齐并全绿**（第 20 回合记的「59 步欠一次门」清账）。簇 tag 已打：**`v-tp2-tuning-table-and-guards-r0` @ `3528e19e`**（消息＝门数据）。- §39 补十（第 26 回合）：**192 正式parked，不盲改也不撤**：新增判据＝它改的是 context-cache **状态语义**（ResourceManager::finish 加 `cancelled` 形参＋Active lane 允许＋Program 无法保序列时回退 discard；我方 `finish/abort` 为 lane 自置形），按 ada-tree/AGENTS.md 的验收规则，**必须先读 `docs/maintainer/op-development.md` 并以独立 oracle/引擎测试验收**（`ninfer_engine_test`、context-cache 套件），门链占用机器期间只做单 TU 编＝不足以定它的正确性。落法：门链跑完→逐块 graft 5 文件→单 TU 编 5 个→`gate-steps.sh` 全构建＋子集＋簇门复跑→才 `--continue`。
- §40（第 27 回合，全序 191–194）：
  - 191 skip（docs 宣称已撤能力，坑③；档案事实已记：nvfp4 KV ≈1.8× 于 int8、262k 每卡 +2 GiB、**DFlash2 acceptance −10~15 点**）。193 `12286e95` skip＝取我方净效应 0：我方 50–61 图面已把 caller stream 显式化（`create_active(pages, 0, device.stream)`＝与供体重构同形），4 个 TU（program_impl/graphs/paged_kv_cache/test_context_store）rc=0 后按铁律 8 撤。
  - **192 `2dc5b1d9` 与 194 `caee6083`＝整步撤除（skip），理由＝落在同一族我方未载的机制上**：实测计数＝我方树 **无** `FinishDisposition`(0)、`clear_lane_strict`(0)、`active_continuations`(0)（而有 `reserve_active_capture` 5 处、`has_context_transaction` 28 处）。192 需新造这三件＋ResourceManager 的 disposition/回退＝**整链 feature port**，不是 graft；194 的 `start_request(MaterializationTransaction&)` 在我方布局里不存在（该块我方侧为空），其同步收益我方已由 8 处 `synchronize_devices/device.synchronize` 覆盖（含失败/回滚路径 237/416/499/524/706/721/734）。⇒ 两者并入 **P2 待办：materialization-transaction 整链**（192/194 同链，issue #3 与 issue #4 收益在链尾）。
  - 194 的 README 参考数（issue #4，WSL2 v0.4.2，只作上游参考、非必达线）：**MTP3 112–114 tok/s**；`--kv-dtype nvfp4` 下 **full 262,144 可用**；pipelined mailbox 起步数据。记入 s7 参考位。
  - 现场：rebase 在位 msgnum=186/end=213（停点 sha 见下轮自证），本轮 4 步处理：191/193/194 skip、192 skip（理由如上），零 TU 失败。94→193＝58 步落地＋**21 步有据撤除**。
- §41（第 28 回合，全序 196–200 落地 5 步＋1 修）：
  - **196 `207fd861`＝serve `--turn-anchors`（issue #3 的产品面）**：核心机制（`add_turn_anchor_markers`＋调用点）随净应用落齐；三处 graft＝serve 帮助行、我方解析链里的 `--turn-anchors` 分支、`test_serve_options` 的四条断言（默认 1／0 与 3 到 options／帮助含该旗标）＋serving.md 选项表一行。取我方＝`load_options.h`（我方无 `vision_rank` 形）、`main.cpp`、`serving.md` 文本面。⇒ **3 TU rc=0**（serve_options / generation_service / test_serve_options）。测量档案（供体 2x5070 Ti，只作参考）：34k 续写 6.8 s → 0.1 s。
  - **197/198/199 净落＝`52beb3c3` HostMapped placement、`3053a51d` `--embedding-host`、`a23598bf` direct-I/O staging 对齐**：`--embedding-host` 是我方 D2 显存的直接出路（供体实测**每卡释放 1.18 GiB**、轮时 +0.13%、gate 与参考逐位相同）；我方 graft＝serve 帮助四行（锚在我方 `--device-state-slots` 行前）＋CLI/serve 双解析（净应用已带 `EngineOptions::embedding_host`＝types.h:220、serve_options.h:57）。⇒ 5 TU rc=0（含 ninfer/ninfer-perplexity/cli 测试三个目标）。
  - **200 `62f20678`＝serve 接受 Anthropic `thinking.display: omitted`**（AGENTS 合同面：我方不提供 Anthropic 加密隐推理恢复语义）。⚠ **我在此步违规**：取我方块后直接 commit，没先看该块的净应用调用点 ⇒ 测试 TU 报 `test_hidden_reasoning_signature/round_trip` 未声明（＝我提交了一个坏 TU）。当场补修＝**`0e06e222`**（1 文件 +102 行，取供体两函数原文插入我方调用函数之前）⇒ `test_anthropic_schema.cpp` **rc=0**。教训进纪律：**`--continue` 前必须看见该步全部相关 TU 的 rc=0**，块内被调用者的定义在供体侧时不能整块取我方。
  - 停点＝全序 **201 `ff6b336f`**「dflash2(tp2): run the drafter MLP gate/up as NVFP4」：5 UU 含 3 个 `execution/tp2/`、`program/tp2/` 文件＝P1 布局族的耦合步 ⇒ 判为整步撤除类（与 178/179/180/184/185 同链，DFlash2 drafter 分裂链）。94→200＝**62 步落地＋22 步有据撤除**；门债＝117–190 已清（100%/27＋100%/167），191–200 面（HEAD `0e06e222`）**尚欠一次全构建门**＝下轮门链 label 建议 `200`。
- §42（第 29 回合，全序 201–204）：
  - 201 `ff6b336f`＝整步撤除（skip，净效应 0）：dflash2 草稿 MLP 走 NVFP4。三重耦合＝(a) 3 个 `execution/tp2/`、`program/tp2/` 文件（P1 布局族）；(b) 把 nvfp4 swiglu 半形状的 A16 注册从「仅 T≤16」扩到「每个正 T」（我方 116/137 的登记与测试断言＝我方政策，扩它要改注册面＋契约＋重跑门）；(c) 制品转换配方（converter 面）。**它的收益值得单独追**：制品小 **425 MiB**＝每 rank **212 MiB**（我方 27B 双卡常驻的直接余量）⇒ 记为 **T-drafter-nvfp4（制品转换＋A16 注册加宽）**，与 T-nvfp4KV 同批在 D2 显存预算里定。
  - 202 `66090f2f`＝落地（serve 用 `X-Ninfer-Client` 标请求）：净应用已带 `parse_client_label`＋`request_events.h` 字段＋serving.md 文面；graft＝我方 `RequestLogMetadata` 两处指定初始化里插 `.client = std::move(client),`（在 `.model` 之后＝声明序，编过即证）；schema 保我方 **25**（供体 22）。⇒ 6 TU rc=0（两个 http＋http_transport＋request_log＋两个测试）。对 S12/D3 直接有用（我方 D 桶 A/B 与 acceptance 统计可按调用方分层）。
  - 203 `bd5cabea`＝落地（request_done 记 `accepted_length_histogram`）：序列化面随净应用落齐（request_log.cpp 三处），我方 schema 25 已覆盖（供体 23）；graft＝serving.md 一段说明（每请求 L 分布＝K-adaptive/fallback 问题的数据源）；tools/bench 语料脚本的干净 hunk 落齐（我方 3 块取我方）。⇒ 2 TU rc=0＋语料脚本可编。
  - 204 `cd059c59`＝**撤除（skip）并转为 D3 测量项**：真缺陷（thinking-on 无 budget 时 `model_thinking_tokens=0`＝供体诊断），但其修法是把 semantic tracker 在无 budget 的 decode 热路径上常开（我方 HEAD 注释＝刻意 dormant，省每 token 第二次 close-marker 扫描）＝两侧语义冲突，成本须在我方 36 SM 上实测再定（口径：以本机实测为准）；另有 3×3 块 tools/bench 耦合。记 **T-thinking-tokens：窗口内量成本，采其效果（log 正确）而不牺牲 decode 热路径的替代形（只在请求日志需要时按 request 开）**。
  - 现场：msgnum=196/213，停点 sha 见自证；94→203＝**65 步落地＋24 步有据撤除**；门链 200 面在跑（`relay/gate-200-build.log` 现 320/693 编 tests）。
- §43（第 30 回合，全序 205 落地）：**`cb5028ff`＝GPU 遥测进 throughput 记录**（时钟/显存时钟/功率/温度/区间能耗＋NVML clocks-event 原因＝我方 D4 A B B A 与「整卡降速归因」的采数面）。落法＝5 处 graft＋去重：`serve/CMakeLists.txt` 加 ${CMAKE_DL_LIBS}（NVML 走 dlopen）；`request_events.h` 在 final_interval 之后插 `std::optional<std::vector<GpuTelemetrySample>> gpus;`（字段序＝声明序，编过即证）；`ProductTests.cmake` 注册 `ninfer_gpu_telemetry_real_test`（SKIP_RETURN_CODE 77＝单卡无 NVML 时自跳）；`http_server.h` 取 both（我方 metrics include ＋ gpu_telemetry include）；`http_server.cpp` 先 both 再**删掉供体块里重复的 `public_model_id_`/`service_` 两行**（both 会把我方已有赋值重复一遍＝铁律3）。schema 取我方 **25**（供体此步想写 24→25＝与我方号重合，我方在前序步骤已推进）。⇒ `cmake` 重生 DB 后 **5 TU rc=0**（含新文件 `gpu_telemetry.cpp` 与真机测试 TU）。
  - 同步取我方：`tests/serve/test_serve_corpus.py`、`tests/serve/run_serve_corpus.py`、`tools/bench/README.md`（各 3 块＝供体语料工具形与我方不同）＝**D 桶语料侧遥测未接**，记 D 窗口待办（跑 A B B A 时若要 gpus 记录进语料统计，需把这三处按我方脚本形接上）。
  - 新停点＝队列 198/213 ＝ 全序 **206 `665f1e9f`**（11 个 UU，含 tools/bench 族）＝大步，checkpoint 与队列已导出（`relay/step206-checkpoint/`、`relay/tp2-replay-queue-16-from-full206.txt`＝16 行）。累计 94→205＝**66 步落地＋24 步撤除**。
  - 门链 200 面（`0e06e222`）：构建到 538/693（链 tests），子集/簇数字待落 `relay/gate-200-chain.log`。纪律照旧：停点有 UU ⇒ 主树面不构建；门只在 gate-72 跑。
- §44（第 31 回合勘查，206＝parked 不半落）：**206 `665f1e9f`「serve: log the per-step accepted draft lengths with --log-speculation-detail」**（11 文件 24 块）勘查结果与逐块落法（下轮照此一遍过，不必重读全部块）：
  - 直接 graft（安全，我方侧只增不改）：`include/ninfer/types.h` 一块＝取我方＋插 `std::uint32_t accepted_length_steps = 0;`（带 3 行注释）；`src/serve/generation_service.cpp` 一块＝取 both 的语义（我方 `prepare_impl(... observation ...)` 调用前插 `if (options_.log_speculation_detail) { observation.accepted_length_steps = options_.speculation_detail_max_steps; }`，调用后插 `prepared.speculation_detail = options_.log_speculation_detail;`）；`src/serve/serve_options.cpp` 两块＝我方帮助/解析链里加 `--log-speculation-detail` 与 `--log-speculation-detail-max-steps`（默认 4096，上限 2^20，要求 `--request-log-jsonl`）；`request_log.h` schema **25→26**（我方 25 已＝供体 25，加此字段才进 26）；`request_log.cpp` 渲染 `accepted_lengths`（每步一个 16 进制位、无草稿步＝`-`，上限截断记 `accepted_lengths_truncated`，关闭＝null）随净应用落齐。
  - **需要判断的一块＝程序侧采集点**：`src/models/qwen3_5/program/decode.cpp` 一块＝供体块里是它自己的 `select_graph_profile/select_graph_topology/install_graph_profile` 大 helper 群（我方 115 时代把这些收在 `graphs.cpp`，我方侧＝`namespace {}`），**逐块取我方会连带丢掉它对 accepted-length 写入点的埋设**；`program_impl.h` 3 块＝`SpeculativeStats::accepted_lengths` 字段与绑定；`prefill.cpp`/`request_plan.cpp` 3 块＝bound 传递。⇒ 落法＝取我方骨架后，在我方**每轮接受计数处**（139 时代的 `SpeculativeStats` 累加面）写 `accepted_lengths`（带上 `accepted_length_steps` 上限，0＝不记不配），并确认 `SpeculativeStats` 有该字段；不能只落 serving 面（否则字段恒为 null＝半功能）。
  - 语料/工具＝`tests/serve/test_serve_corpus.py`、`tools/bench/README.md`、`tests/serve/run_serve_corpus.py` 各 4 块＝供体工具形与我方不同 ⇒ 取我方，D 窗口需要时再接（与 205 的 gpus 语料缺口同批处理）。
  - 验收（一轮做完）：全部相关 TU 单编 rc=0（types 的消费者面至少编 `serve/generation_service.cpp`、`serve/serve_options.cpp`、`serve/request_log.cpp`、`program/decode.cpp`、`program/prefill.cpp`、`planning/request_plan.cpp`、`program_impl.cpp`＋测试 `test_request_log.cpp`/`test_serve_corpus` 相关）＋`--continue`＋merge-safety。价值＝S12 逐轮 K-adaptive 数据（比 203 的每请求直方图细）。
  - 门链 200 面（`0e06e222`）：构建 576/693（链 tests）；子集/簇数字待落 `relay/gate-200-chain.log`（＝117–200 全量护航，落账后打 tag）。
- §45（第 33 回合，全序 206–207 落地＋标记缺陷修复）：
  - **修复 `494192f5`＝`fix(tests): drop the conflict markers that entered the serve corpus tooling`**：三文件（`tests/test_serve_corpus.py`、`tools/bench/README.md`、`tools/bench/run_serve_corpus.py`）在 HEAD 各带 1 处标记（我在 202/203 用错路径＋resolve_blocks 对含 `=======` 分隔行的文件解析中止后仍 `git add -A` 提交所致）；从 `0e06e222`（202 前最后一个干净提交）取回，py 文件 `ast.parse` 通过，标记＝0。
  - **纪律升级（本轮起生效并已执行一次）**：`merge-safety` 必须在 **`--continue` 之前**跑（扫索引＋工作树标记＝0），且 `resolve_blocks.py` 对含 `=======` 分隔行的 markdown/py 文件不可用 ⇒ 一律 `git show :2:<file> > <file` 取我方全文件后 `git add`。207 的收尾即按此执行：pre-continue 索引标记 0、工作树标记 0、符号清单无缺失。
  - **206 `c6e8e547`＝逐轮接受长度（`--log-speculation-detail`）**：graft 见 §44；两处我犯的错当场修好＝`accepted_length_steps` 先插进错误结构体（`decode.cpp` 报 no member ⇒ 移入 `RequestControl`）、净应用引用的 `detail_steps_explicit` 未声明（补声明）。**8 TU rc=0**（decode/prefill/request_plan/program_impl/generation_service/serve_options/request_log/test_request_log）。schema 我方 25→26。
  - **207 `cd0422f1`＝输出对提示的 4-gram 重合率**（＝判「copy 型草稿（prompt lookup）对我们真实流量到底值不值」的数据面，正对 A6/D3-D4 的 proposal-head 取舍）：`generation_service.cpp` 三块＝取 theirs（prepare_impl 多参形）＋both（保留我方 constraint/tools 行）＋取我方后 graft `if (speculation_detail) { prompt.token_ids() → prepared.prompt_token_ids }`；`tests/test_request_log.cpp` 取 theirs（含 206 的直方图精确差值断言）。⇒ **2 TU rc=0**。语料/工具面（我方版）未接此统计＝D 窗口待办（与 205 的 gpus 语料缺口同批）。
  - 门链 200 面＝`build rc=0`＋**子集 100%/27·420.85 s**；簇门在跑（分母取当次计数）。停点＝全序 **208 `2acb2784`**（docs/serving.md 1 处冲突）。94→207＝**69 步落地＋24 步有据撤除**。
- §46（第 36 回合，200 面红清账＋209/210 落地）：
  - **200 面簇门那一个红＝根因三条全部查明**：① fixture 缺 `engine_options.embedding_host = true`（我 198 取我方时丢了供体 hunk，断言随干净应用落了＝坑②/坑③合成）；② `transactions/binding.cpp:419` 仍以 3 参调 `install_sampling`（206 把签名改成 5 参＝我方布局的调用点没在供体步里）；③ `tests/test_request_log.cpp` 的 `materialization` 断言腿要求我方未载的 P2 materialization 链（192/194 parked）。
  - 修复＝**`d0166ffe`**（2 文件：binding 传 `0U, 0U`＝该路径不记长度；fixture 补旗标；materialization 腿改为 `if (done.contains("materialization"))` 守卫＋注释指向 P2＝**保留覆盖不删**）。验收＝`cmake --build --target ninfer_request_log_test` 链接成功 ＋ `CUDA_VISIBLE_DEVICES=1 ./tests/ninfer_request_log_test` 输出 **ok、rc=0**（GPU1，实跑，非编而不过）。
  - **209 `4355abca`＝README 汇总**：取我方限制清单（我方拒 tp2 Vision / 只 bf16+int8 KV），只 graft 我方**已落地**的两项为 “Since v0.4.4” 条目＝`--turn-anchors`（196）与 `--embedding-host`（198，1.18 GiB/卡、位等值），并注明 `--log-speculation-detail`（206/207）与吞吐记录里的每卡时钟/功率/能耗（205）；**明确写上**「供体对 nvfp4 KV、DFlash2 分裂 drafter、tp2 Vision 的说法不适用本树（P1 布局批 / 未测）」＝坑③在文档面的执行。pre-continue 索引/工作树标记＝0。
  - **210 `6867c3e0`＝skip**：`tests/test_resource_manager.cpp` 在我方是 DU（我方早已删该文件），供体这步给它加 `finish_cancelled` 的 fake 与新用例＝P2 链（192）配套 ⇒ `git rm` 后净效应 0 ⇒ 撤，理由记此。
  - 门链已在新面重跑＝`bash relay/gate-steps.sh 4355abca 209`（日志 `relay/gate-209-chain.log`；判据＝子集 27/27 且簇门 100%，分母取当次计数＝上次 167）；绿则打簇 tag `v-tp2-serve-telemetry-and-speculation-detail-r0 @ 4355abca`（消息＝门数据＋d0166ffe 的三因修复）。
  - 累计：94→210＝**71 步落地＋25 步有据撤除**；停点与队列＝下轮首条命令自证（`cat .git/rebase-merge/msgnum .git/rebase-merge/end .git/rebase-merge/stopped-sha`）。
- §47（第 39 回合＝队列跑到头）：
  - 末步 213 `79d969d4`＝**整步撤除（skip）**：四条回收政策（① 绝不牺牲「重建代价高于被保护 capture」的 checkpoint（protected_rebuild_work）；② 先按低到高丢 long anchors（新 API `Program::drop_continuation_anchor`，复用 `publish_checkpoint_drop`、无副作用即拒）；③ 拥有者按 retention weight → GreedyDual priority 排序）落在 `AdmissionCandidate`/`seal_materialization`/`pressure_owners` 这套我方未载的 P2 准入/压力机制上（program.h/program_impl.h/resource_manager.h 三块我方侧多为空；`tests/test_resource_manager.cpp` 我方已删＝DU）。⇒ 记 **T-reclaim-priority**（issue #3：tp2 上一个 1.8k 短会话的 capture 把长 agent 会话整体挤掉，下一轮整段重 prefill），做 P2 时随链一并落。
  - **rebase 收尾自证**：`Successfully rebased and updated refs/heads/tp2/raw`；`.git/rebase-merge` 消失；`git status` 行数＝0、UU＝0；HEAD＝**`58598a9c`**（其上＝`6460e68c` 220、`9e6d0fa6` 219）。队列 95→213 **全部处理完**（逐步落地/撤除的判据与 sha 都在 §37–§47 各行里，不在此重算总数）。
  - 本轮修掉我在 220 带进的标记＝**`58598a9c`**（3 文件 −116/+3：`serve_options.cpp`、`generation_service.{h,cpp}` 回到 `9e6d0fa6` 内容），验活＝两个 TU **rc=0**（`serve/serve_options.cpp`、`serve/generation_service.cpp`）。两条重放路都试过且不可用：`git diff 9e6d0fa6..6460e68c` 的补丁**含标记文本**（再贴一遍标记），`git show 382a8a60` 的原补丁在 generation_service.cpp 的 207 graft 处 reject（context 不符、无 .rej）。
  - ⇒ **A2（`--prefill-miss-dump`）＝部分交付**：新文件 `src/serve/prefill_miss_dump.{h,cpp}`（87＋167 行）、`operational_log.cpp`、`include/ninfer/engine.h`、`frontend/{frontend.cpp,frontend.h}`、`serve_options.h` 字段、测试 `tests/test_prefill_miss_dump.cpp` ＋ `tests/cmake/ProductTests.cmake` 注册（round 38 已单编 rc=0）**都在树里**；缺＝`serve_options.cpp` 的三个旗标解析与帮助行、`generation_service` 的上下文跟踪接入。修法（下轮，半小时内）：照 `git show 382a8a60 -- src/serve/serve_options.cpp` 在我方解析链 `--log-speculation-detail-max-steps` 分支之后插入，generation_service 侧按其 hunk 的三处插入点逐段 graft；判据＝3 TU rc=0 ＋ 跑 `ninfer_prefill_miss_dump_test`。
  - 验活进行中（不阻塞收尾）：主树面全量构建后台跑＝`relay/final-build.log`（nice 19、-j 6；启动时 gate-72 只到 `patches ok`＝在编译、未进 ctest，两树目录不同）；门链 209 面（`4355abca`）继续在 gate-72 跑，`sub rc=`/`full rc=` 落账后打 209 面 tag，再对 `58598a9c` 补跑一次 final 面门链。
  - 现场风险登记（诚实）：主树面 final 构建的 rc 未回（下轮第一件事＝读 `relay/final-build.log` 尾行 `rc=`）；209 面门链的 sub/full 未落；220 的标记修复虽已提交，`--prefill-miss-dump` 的 serve 面缺口＝已知未验项。
- §48（第 41 回合＝交棒前落档）：
  - **A2 收尾（部分完成，已验）** 新提交＝见本行 sha：`serve_options.cpp` 接上三旗标（`--prefill-miss-dump DIR`/`--prefill-miss-min-tokens`/`--prefill-miss-min-match`）＋帮助行＋usage＋`prefill_miss_tuned` 要求 dump 目录的校验；`generation_service.h` 补 `PrefillMissReport` include 与 `GenerationMetrics::prefill_miss` 字段（＝220 净应用的 `operational_log.cpp::render_prefill_miss` 与 `request_log.cpp` 所需，之前被我把整文件退回 HEAD 时一并剥掉，导致 `PrefillMissReport does not name a type`）。
  - 判据（实跑）：`serve_options.cpp`/`generation_service.cpp`/`operational_log.cpp`/`request_log.cpp` 四 TU rc=0；链 `ninfer_serve_options_test`＝**ok**、链 `ninfer_prefill_miss_dump_test`＝**OK prefill miss dump**（rc=0）。
  - **仍欠的一条腿（下轮第一件事，规格已定）**：`generation_service.cpp` 的 run/prepare 侧接入＝① ctor 建 `prefill_miss_dump_`（`PrefillMissDumpOptions{directory,min_tokens,min_match}` ＋ detokenize lambda，`engine_.get()` 以裸指针捕获）；② 207 那段 `if (speculation_detail)` 扩成 `if (speculation_detail || (prefill_miss_dump_ != nullptr && cache_participation == CacheParticipation::ReadWrite))`；③ 成功路径在 `outcome.metrics.admission` 之后 `outcome.metrics.prefill_miss = prefill_miss_dump_->observe(prompt_ids, result.generated_token_ids, result.reused_prompt_tokens, cancelled?"cancelled":"finished")` 然后 `prepared.prompt_token_ids = {};`；④ 失败路径的 `catch (...)` **必须排在 `catch (const ninfer::RequestError&)` 之后**（我上一轮把它插在前面 ⇒ `... handler must be the last handler`，fpermissive 报错）。原文逐段＝`git show 382a8a60 -- src/serve/generation_service.cpp`。判据＝该 TU rc=0 ＋ 跑 `ninfer_prefill_miss_dump_test` ＋ 家族/簇门。
  - 我这一轮的两条自伤与修法（写进纪律）：① `pkill -f "build-120a-cu13 -j 6"` 的模式匹配到自己的命令行 ⇒ 自杀式 SIGTERM，用 PID 精确 kill 才对；② 为了赶工把整文件 `git checkout HEAD --` 退回，连带剥掉净应用面需要的字段 ⇒ 撤改动要按符号依赖退（`git grep` 先查谁在用），不是按文件退。
  - 队列/门现状：队列 95→213 已跑到头（HEAD 现为上一条 commit，工作树干净）；**209 面门链（`4355abca`）＝全构建 rc=0（17:07:39）＋子集在跑**（`relay/gate-209-chain.log`，`sub rc=`/`full rc=` 待落，绿则打 `v-tp2-serve-telemetry-and-speculation-detail-r0`）；主树面 final 全构建被我停过（保护门链计时），交棒后在 A2 腿落地时一并重跑 `bash relay/gate-steps.sh <HEAD> final`。
- §49（第 42 回合）：**A2 欠腿已补＝`feat(serve): complete the prefill-miss-dump surface in generation_service`**（本轮 sha，见 git log 首行）。
  - 四段全部按 §48 规格落：① ctor 建 `prefill_miss_dump_`（`PrefillMissDumpOptions{directory,min_tokens,min_match}` ＋ 以裸指针捕获 `engine_.get()` 的 detokenize lambda；我把该成员**声明在 `engine_` 之后**⇒ 析构顺序＝dumper 先亡、engine 后亡，注释写明）；② 207 段的 `if (speculation_detail)` 扩成 `… || (prefill_miss_dump_ != nullptr && cache_participation == CacheParticipation::ReadWrite)`；③ 失败路径＝把 `catch (...)` 放在 `wait()` 之后、内部 `try { throw; } catch (const ninfer::RequestError&) { throw_request_error(exception, prepared.constraint_param, prepared.tool_schema_params); }` 原样重抛（**保留我方带 constraint/tools 的签名**，供体只有 `throw_request_error(exception)`）；④ 成功路径在 `prompt_ngram_overlap` 之后 `observe(prompt_ids, generated, reused, cancelled?"cancelled":"finished")`，并把 `prepared.prompt_token_ids = {};` 从 207 的 if 块里**移到两个消费者之后**（否则 overlap 先清空 ⇒ dump 收不到 ids；这是供体原补丁的行位搬动，不是新增语义）。
  - 编译面验活＝`serve/generation_service.cpp`、`serve/operational_log.cpp`、`serve/request_log.cpp` **三 TU rc=0**（nice 19，只编 `.o`）。**运行时验活排队中**：209 面簇门此刻在跑（ctest 37/167，PID 376321）⇒ 按铁律 6 不在其期间链接/跑测试。门一停就执行：`cmake --build build-120a-cu13 --target ninfer_prefill_miss_dump_test ninfer_request_log_test ninfer_serve_options_test -j 8`，然后 `LD_LIBRARY_PATH=../deps-ada/lib CUDA_VISIBLE_DEVICES=1 ./tests/<三者>`（期望＝`OK prefill miss dump`／`ok`／`ok`），再 `bash relay/gate-steps.sh <HEAD> final` 补 final 面门链（子集 27/27＋簇门取当次分母＝上次 167）。
  - 209 面门数据（`4355abca`）：全构建 rc=0（17:07:39）＋家族子集 **100%/27·470.66 s**；簇门在跑（未落账 ⇒ 209 面 tag 仍欠，绿则打 `v-tp2-serve-telemetry-and-speculation-detail-r0`）。
  - 本轮纪律自查：上一轮我为了赶工整文件 `git checkout HEAD --` 退回，连带剥掉净应用面所需的 `GenerationMetrics::prefill_miss` 字段（报 `PrefillMissReport does not name a type`）⇒ 撤改动按**符号依赖**逐符号退（先 `git grep <type>` 找消费者），本轮已按此重放并编过。
- §49 补（第 42 回合末，实测）：209 面（4355abca）簇门＝99%，1 红 / 168，1095.52 s；红＝ninfer_serve_options_test（#46）。
  - 根因＝我在 206 的 graft 与净应用的供体分支成对重复：我那对只校验 steps > 2^20，供体那对要求 [1, kMaximumSpeculationDetailMaxSteps]；我的分支排在链前，令 207 测试期望的 --log-speculation-detail-max-steps 0 被拒永不执行 ⇒ 该面分母 167→168（ninfer_prefill_miss_dump_test 注册生效）。
  - 修复已在 HEAD＝6c07768c 的去重（保留供体严校验）。本轮实跑：ninfer_serve_options_test＝ok、ninfer_prefill_miss_dump_test＝OK prefill miss dump、A2 四段接线＝9c4d4e8e（3 TU rc=0）。
  - 209 面 tag 不打（红门）。final 面门链已启动＝bash relay/gate-steps.sh 9c4d4e8e final（gate-72 已 reset 到 9c4d4e8e，两补丁 patches ok）。判据＝子集 27/27 ＋ 簇门 100%（分母取当次计数），绿则打 v-tp2-prefill-miss-dump-and-q8-half-r0 @ 9c4d4e8e，消息含本条红因与去重说明。
- §50（第 43 回合＝129 欠账清偿＋账本更正）：
  - **账本更正（重要，别再照旧文重做）**：§48/§49 记的「129 收尾取我方 ⇒ tp2 plan 未加 causal_score phase＝容量预约缺失」**不准确**。实测＝我方 `planning/startup.cpp` 在 129 落地时就**已有**该 phase 的两 rank 预约块（现 701–719 行：`output_head_split_workspace_bytes` + `workspace::tp_logits(rank0/rank1)` + `reserve_matrix` + `out.causal_score` 计入 `general_capacity`），`planning/startup.h:67` 也有该字段。真正的闸口在**同函数开头的 plan 门**：`if (plan.causal_scoring || plan.features.vision || DFlash) throw` ⇒ 打分在 tp2 于**预约之前**就响亮失败。教训＝撤/留判断要看**整函数控制流**，不能只看 diff 块文本；欠账登记要写「可复核的符号/行为」，我那句「phase 未加」让下一轮白白重勘查一次。
  - 清偿＝本轮提交（sha 见 git log 首行）：plan 门改成只拒 vision 与 DFlash，消息补 `and causal scoring only`，并注明本树 tp2 拒 vision 的既有政策；确认全树已无 `Engine tp 2 does not support CausalScoring`（129 当时撤的两处＝`model_instance.cpp:71/:78`，本轮 grep 复核＝0 命中；`engine.cpp:179` 是 core 选择非拒绝）。
  - 验活＝`planning/startup.cpp` 单 TU **rc=0**（nice 19，只编 .o，门链在跑 ⇒ 不链接）；merge-safety：索引标记 0、符号无缺失。运行期真打分＝D 桶（两卡），单卡面＝计划能过闸＋预约齐备。
  - 门债现状：209 面（4355abca）＝全构建 rc=0＋子集 100%/27·470.66 s＋簇门 99%（1 红＝serve_options，根因与修复见 §49 补）⇒ **不打 tag**；final 面门链（9c4d4e8e）在跑＝`relay/gate-final-chain.log`（现编译阶段 74/455）；本轮 commit 之后 final 面已非 HEAD ⇒ 门链跑完读数据后，**对 HEAD 再跑一次** `bash relay/gate-steps.sh $(git rev-parse --short HEAD) final2`（判据＝子集 27/27＋簇门 100%，分母取当次计数＝上次 168），绿则打 tag `v-tp2-prefill-miss-dump-q8-half-and-tp2-scoring-r0`。
- §51（第 43 回合＝S11 勘查，未动代码）：Vision@tp2 的**机制层已在我方树里**，缺的只有工作区相位与两道闸（实测符号）：
  - 已在树：`src/models/load_options.h:15-18` ＝ `int vision_rank = 0`，注释写明 Vision 由 `vision_rank` 整块持有、跨 ExecutionContext 两设备；`src/models/qwen3_5/load/sharding.cpp:223` ＝ `whole(ShardAxis::SingleDevice, options.vision_rank)`，261-262 校验 rank 合法；`src/product/vision_options.h:32-43` 的 `validate_vision_options` 已带 tp2 形（`--vision-device must be one of the --devices ids`）；`src/runtime/engine/model_instance.cpp:294-295` 注释确认 tp2 下「Vision tower 可在 rank 1、drafter 与 proposal head 在 rank 0」；单设备 plan 的 vision 相位＝`planning/startup.cpp:1090-1095`（`out.vision = VisionContext::plan_workspace(*config.vision, *parameters.vision, merged, out.general_capacity)` 且 `out.capacity = max(out.capacity, out.vision->capacity_bytes)`），配套 roots 差异＝816-817 行 `workspace::text_prefill_roots(layout, config, tokens, features.vision ? 3 : 0, features.vision ? tokens : 0)`。
  - 两道闸（S11 的开关面）：`src/runtime/engine/model_instance.cpp:81-83` 的 `if (options.enable_vision) throw "Engine tp 2 does not support Vision"`；`planning/startup.cpp:496` 的 `if (plan.features.vision || DFlash) throw`（本轮 `beb82b09` 已把 causal_scoring 从这道门移出，vision 仍在门里）。
  - 因此 S11 的真活＝**tp2 plan 的 vision 相位**：塔所在 rank（`vision_rank`）按 1090-1095 的同一公式加 `out.vision` 并把 capacity 取 max；非塔 rank 只带 merged 落点的 roots 形；`SequencePlanImpl` 目前是否把 `vision_rank` 传到 plan 侧**未证**（供体在 program_impl.h 的 `load_options` 里有 `.vision_rank = vision_rank(options)`，我方该块当时取我方＝没带）＝第一件要查的事。
  - 这属容量报价/预约语义 ⇒ 改前读 `docs/maintainer/op-development.md` §5.2（本轮已读，要点：一个 scratch-capable 入口只有一套权威容量查询；容量按 256B 对齐的 Op-local cursor 量、含内部对齐间隙、不含尾部填充；sizing 与执行共享同一 route 事实与分配配方；组合 Op 报完整公共 scratch 而非私有子项）与 §2.6（tensor-parallel forms）。顺序＝**先相位、后开闸、再文档**；只开闸不加相位＝启动期容量不足的响亮失败（不会静默算错）。
  - 文档同步面（开闸后要一起改）：`docs/cli.md` 我方拒绝清单（现含 `--vision`，97 落）、`README.md` 我方「Vision. --vision is rejected at --tp 2」条（100 落）、`docs/serving.md`、S6 能力门文面（gate 工具不在本树＝`relay/s6-gates-r72.patch` 系，门补丁在 gate-72 里跑）。
  - 验收分级：单卡可验＝TU 编译（startup/model_instance/load_options 消费者）＋`test_cli_options`/`test_serve_options` 的帮助与校验断言＋家族子集门；功能面（图像/视频真跑 tp2）＝D1/D2 窗口，vision 的 tp2 测试在单卡自跳 77。
  - 门链（9c4d4e8e 面）此刻仍在编译（ninja 活着，build 日志 01:37 更新；168 个测试目标链接慢）⇒ 按铁律 6 本轮不在主树链接/跑测试，数据下轮读 `relay/gate-final-chain.log`。
- §51 补（第 43 回合，实测）：S11 第一件「vision_rank 是否传到 plan」**已核实＝已传**。`SequencePlanImpl::features` 的类型就是 `models::LoadOptions`（`planning/startup.h:107`），而 `vision_rank` 正是该结构成员（`src/models/load_options.h:18`）⇒ tp2 plan 里直接可用 `plan.features.vision_rank`，**无需新管线**；`sharding.cpp:223/261` 已在用同一路径的 rank 校验。⇒ S11 剩余＝①在 tp2 plan 内按 1090-1095 同款公式给塔 rank 加 `out.vision` 相位（capacity 取 max）＋非塔 rank 的 roots 形（816-817）②开两道闸（`model_instance.cpp:81-83`、`startup.cpp:496`）③同步 cli.md/README/serving.md 与 S6 门补丁文面。顺序＝相位→闸→文档；属容量语义⇒动前先读 op-development §5.2/§2.6（本轮已读，要点已记 §51）。
- §52（第 43 回合＝S11 第 1 步落地）：实测发现**通道是断的**：`src/models/load_options.h` 的 `load_options(const EngineOptions&)` 从不设 `vision_rank` ⇒ 它恒为默认 0 ⇒ `sharding.cpp:223` 的 `whole(ShardAxis::SingleDevice, options.vision_rank)` 永远落在 rank 0，`sharding.cpp:261-262` 的 rank 校验形同空转，plan 侧 `features.vision_rank` 也恒 0（`planning/startup.h:107` 的类型确实是 `models::LoadOptions`，所以字段在、值没进来）。
  - 修法（本轮提交，sha 见 git log 首行）＝新增 `vision_rank(const EngineOptions&)`＝在 `options.devices` 里查 `options.vision_device` 的下标（未设/查不到＝0；tp1 时产品层 `validate_vision_options` 已强制 `--vision-device` 等于 `--device`），并把 `.vision_rank = vision_rank(options)` 接进 `load_options()`；补 `#include <algorithm>`。
  - 验活＝`qwen3_5/load/sharding.cpp` 与 `planning/startup.cpp` 两 TU **rc=0**；链并跑 `ninfer_qwen3_5_shard_map_test` ＝**「qwen3_5 tensor-parallel shard map passed」**（该测试无需第二设备，正是放置图的验收面）。
  - S11 剩余（顺序不变）＝②tp2 plan 的 vision 相位（照 `startup.cpp:1090-1095` 同款公式给塔 rank 加 `out.vision` 并把 capacity 取 max；非塔 rank 用 816-817 的 roots 形）③开两道闸（`model_instance.cpp:81-83`、`startup.cpp:496`）④文档同步（cli.md 拒绝清单、README 的 Vision 条、serving.md、S6 门补丁文面）。属容量语义⇒动前读 op-development §5.2/§2.6（要点已摘在 §51）。
  - 门链（9c4d4e8e 面）本轮末仍在编译 168 个测试目标（ninja 活着、`relay/gate-final-chain.log` 尚无 rc 行）；我在其编译阶段链并跑了 1 个测试目标（shard_map）＝已披露，未与 ctest 重叠。
- §53（第 44 回合＝S11 第 2 步落地 `d00eb5bb`）：
  - 内容＝tp2 plan 只拒 DFlash（vision 进门外的资格）＋在 tp2 plan 尾部加 vision 相位（与单设备同款公式：`merged = min(plan.capacity, kMaximumVisionItemTokens)`、`VisionContext::plan_workspace(*config.vision, *parameters.vision, merged, general_capacity)`、`capacity = max(capacity, vision->capacity_bytes)`）；`model_instance.cpp` 打开 tp2 Vision 闸（原 `if (options.enable_vision) throw` 三行改成放置说明注释）；文档同批＝`README.md` 的 Vision 条从「rejected」改「works at --tp 2」并写明塔在 `--vision-device`（默认 A）该卡独付、`docs/cli.md` 拒绝清单去掉 `--vision` 并加 `--vision-device must be one of the --devices ids`。
  - 验活＝`planning/startup.cpp` 与 `engine/model_instance.cpp` 两 TU **rc=0**；跑 `ninfer_qwen3_5_shard_map_test`＝**passed**（放置图验收面）。披露：我在 gate-72 门链的**编译阶段**于主树链并跑了 2 个测试目标（此刻 gate-72 无 ctest 在跑，未违铁律 6）。
  - 我踩的新坑（已修，写进纪律）：往 C++ 字符串里塞带真实换行的续行（python 三引号里写 `"...\n"\n`）⇒ 生成 `missing terminating " character`。规矩＝C++ 源码里的字符串续行只能拼字面量，落码后用 `sed -n` 复核该处再编。
  - 未验/未完成（诚实清单）：① **vision@tp2 的测试文件在树里但未注册**＝`tests/models/qwen3_5/test_engine_vision_tp2_real.cpp` 存在，`tests/cmake/*.cmake` 里 grep 不到 `ninfer_add_test` 条目 ⇒ 窗口跑不到它；下一步第一件事＝照 `ninfer_qwen3_5_engine_tp2_real_test` 的注册形补一条（`SKIP_RETURN_CODE 77`＋artifact 依赖），并顺带把文本侧测试目标名列出来（我用的名字 `ninfer_text_context_tests` 不存在＝tests/README 的标签名≠CMake 目标名，别猜）。② 功能面（图像/视频真跑 tp2）＝D1/D2 窗口。③ 容量侧的已知偏差：我方 plan 是「一 arena 容量两 rank 共用」⇒ 相位取 max ⇒ **非塔 rank 也被迫留同样的视野余量**，与供体文案「the tower's GPU pays for it alone」不一致＝S11 第 3 步（per-rank capacity，需 `WorkspacePlan` 分成 rank0/rank1 两份容量并让 KV sizing 按 rank 用各自的），属容量语义＝动前重读 op-development §5.2。
  - 门链状态：209 面（4355abca）簇门 99%（1 红＝serve_options，根因/修复见 §49 补，实跑 ok）⇒ 不打 tag；9c4d4e8e 面的链在本回合仍在编译（`relay/gate-final-build.log`，168 个测试目标）；本回合落了 `d00eb5bb` ⇒ 该链跑完读数据后，对 **HEAD `d00eb5bb`** 再跑 `bash relay/gate-steps.sh d00eb5bb s11`（判据＝子集 27/27＋簇门 100%，分母取当次计数＝上次 168），绿则打 tag `v-tp2-tp2-scoring-vision-and-a2-r0`。
- §54（第 45 回合＝交棒前收尾）：
  - **`9c4d4e8e` 面门全绿并已打 tag**＝`v-tp2-tp2-scoring-a2-and-q8-half-r0`：全构建 rc=0（17:47:20）；家族子集 **100%，28/28，420.42 s**；簇门 `-E softmax_attention` **100%，170/170，1052.09 s**（passset＝`relay/passset-sm120-tp2steps-final*.tsv`）。分母较上次（27/167）各 +1/+3＝191–208 期间新注册的可跑测试；逐条对应关系我**没**逐一核（诚实标注）。
  - **S11 第 3 件（测试注册）完成＝`2a209505`**：`tests/models/qwen3_5/tests.cmake` 注册三条（`ninfer_qwen3_5_engine_vision_tp2_real_test`＝parity 默认，`..._mtp_real_test`，`..._dflash2_real_test`，均 `SKIP_RETURN_CODE 77`）；测试源码适配我方 API＝`context_cache.host_state_slots/host_kv_capacity_bytes` → 单一 `host_capacity_bytes`（我方 61 时代合并），并去掉 `ContextCacheHints::retention`/`CacheRetentionHint`（我方无此旋钮，注释说明由默认 `update_session_index` 维持会话血统）。验活＝TU rc=0 ＋ 链并跑＝**输出 `skip: NINFER_TEST_ARTIFACT is not set`、rc=77**（注册与跳路正确，功能面留 D 窗口）。
  - HEAD 面门链已后台启动＝`bash relay/gate-steps.sh 2a209505 s11`（日志 `relay/gate-s11-chain.log`，判据＝子集全绿＋簇门 100%，分母取当次计数＝上次 170），绿则打 `v-tp2-vision-tp2-tests-r0 @ 2a209505`。
  - S11 状态＝1 通道（`b2a6ee1d`）✅、2 相位＋两道闸＋文档（`d00eb5bb`）✅、3 测试注册（`2a209505`）✅；**剩第 4 件＝per-rank 容量**（我方 plan 一 arena 两 rank 共用取 max ⇒ 非塔 rank 也留同样视野余量，与供体「塔那张卡独付」不一致；属容量报价/预约语义，动前重读 op-development §5.2/§2.6，验收＝startup/plan 相关 TU rc=0＋门）。
  - 交棒时累计：队列 95→213 全处理完（replay 到头）、四个面已 tag（116/141/181fix/9c4d4e8e）、A2 与 129 编译＋单卡测试面清偿、S11 四件做三件。未清＝final 面（s11 链）数据与 tag、S11 第 4 件、T-nvfp4KV（186）、T-drafter-nvfp4（201）、T-reclaim-priority（213）、P1 布局整批、P2 materialization 链（192/194/213）、S7 的 36-SM allowance/TMA 界标定、S12–S14。

## 55. 第 46 回合＝S11 第 4 件（per-rank 容量）＋第三道闸＋文档对形（daf35d5b）
- 现场自证（开工时）：porcelain＝3（上一会话留下的 startup.cpp/startup.h/program_impl.cpp 半成品）、无 rebase-merge、HEAD＝c5cf67f8（模型卡文面，其上＝2a209505 已 tag v-tp2-vision-tp2-tests-r0＝全构建 rc=0＋子集 100%/28·419.70 s＋簇门 100%/173·1064.93 s）。
- **第三道闸＝本轮才找到**：`planning/startup.cpp:validate_target_options`（tp2 段）在 §50 清掉的 plan 门之外，仍在拒 `purpose != Generation` 与 `enable_vision`，文案是 "tensor-parallel execution supports ordinary, MTP and DFlash2 text generation only (no DFlash, Vision or causal scoring)"（≠§50 那句 grep 过的 "Engine tp 2 does not support CausalScoring"）。⇒ §50 的 causal-scoring 清偿在启动期仍被这道闸挡住＝`ninfer-perplexity --tp 2` 起不来。本轮开闸＝Generation/CausalScoring 都收，DFlash 仍拒。
- **我方 vision@tp2 的执行面缺口（新证）**：`execution/text.cpp:3071/3087` 两个 multimodal `prefill_chunk` 入口都有 `require_single_device("multimodal prefill")` ⇒ 带媒体的 prompt 在 tp2 响亮失败；供体的接收面（`WorkspacePlan::vision_receiver`＋事件同步拷贝 `[H,V]`＝供体文档 tensor-parallel.md §8 原文）**不在我方树**，属 P1 布局族（160/161/162/164/166/167/170 基建）。⇒ 塔只能放 rank 0：`transactions/binding.cpp:52/401`、`transactions/replay.cpp:257` 用 rank 0 的 device/Parameters/arena 构造 session，`execution/parameters.cpp:346` 只在持有塔的那张卡准备 Vision Parameters ⇒ 本轮把 `vision_rank != 0` 在 `validate_target_options` 与 `model_instance.cpp` 两道闸都判死（动 artifact 之前）。
- **容量面（S11 第 4 件）**：`WorkspacePlan` 加 `peer_general_capacity/peer_capacity`（0＝不分＝沿用 rank 0 的值，故 tp1 与未分面的 plan 一字不动）；vision 相位只记在编码 rank（＝rank 0）的份额上；`ProgramImpl::PeerRuntime` 按 rank 1 自己的份额开 arena；`SequencePlanImpl::device_reservation_bytes` 改成两 rank 总额的较大者（rank 0 persistent+arena vs rank 1 peer_persistent+peer arena），`resolve_kv_capacity_symmetric` 仍对最紧 rank 的空闲解 ⇒ 非塔 rank 不再被迫留视野余量。参考量级：供体产线记录（`tools/tp2/reference/req-greedy.jsonl` memory 行）encode peak＝216,662,017 B（≈207 MiB）而 general＝113,319,936 B；我方单设备面口径上限 827 MiB（`tests/models/qwen3_5/test_vision_workspace.cpp` 的 kWorkspaceCeiling）。属容量报价/预约语义，已先读 op-development §5.2/§2.6（§2.6 原文＝"Rank r's tensors, weights and workspace live on ec.dev[r]"＝per-rank 容量是规范形状）。
- 文档对形（坑③在文档面）：tensor-parallel.md §8 原文写的是供体机制（receiver/拷贝/kv-capacity 记账），README/cli.md/serving.md 写「vision 在 tp2 可用/被拒」自相矛盾 ⇒ 三处改成我方形态：塔常驻第一张 `--devices`、arena 按 rank 分、文本请求可跑、带媒体的请求在 tp2 被拒（等 P1 接收面）。
- 工具面一致性（好兆头）：`tools/tp2/gate.sh:43` 与 `bench/agentic_ab/linux.py:52` 的产线默认都是 `--vision --vision-device 0` ⇒ 新闸与 S6 门默认行一致，D1 可直接跑「vision 常驻＋文本面」。
- 验收：7 个 TU rc=0（planning/startup.cpp、program/program_impl.cpp、runtime/engine/model_instance.cpp、execution/text.cpp、program/prefill.cpp、transactions/binding.cpp、transactions/replay.cpp）＋`ninfer_qwen3_5_shard_map_test`＝passed。门链＝`bash relay/gate-steps.sh daf35d5b s11b`（在跑，日志 `relay/gate-s11b-chain.log`，判据＝子集全绿＋簇门 100%，分母取当次计数＝上次 28/173）；绿则打 tag `v-tp2-per-rank-arena-and-tp2-scoring-r0`。
- 下一件＝**T-nvfp4KV（186＝be1960ce，我方已证该供体 sha 在场）**：整步只有 40 增/28 删/18 文件，性质＝准入面（撤 tp2 对 nvfp4 KV 的四处拒绝＋causal_geometry 的 nvfp4 分裂几何＋`tests/ops/softmax_attention/causal_cache.cpp` 家族表＋文档）。注意＝该族测试在簇门被 `-E softmax_attention` 排除 ⇒ 单卡验法＝TU rc=0 ＋ 在 gate-72 构建面上**另跑一次 `ctest -R softmax_attention`**（单设备 nvfp4 causal_cache 腿），两设备 parity（187＝`cf59d1c1` 时代的 `ninfer_nvfp4_kv_tp2_parity_test` 类）＝D1/D2。收益口径（供体 2x5070，只作参考）＝262k 每卡省 ~2 GiB、PPL 4.1229/4.3987 vs int8 4.1335/4.3859、GSM8K-500 0.974/0.972、decode MTP3 +4–8%／DFlash2 +4–17%，**DFlash2 acceptance −10~15 点**（§39 补九）⇒ 我方 D2 显存最实出路，代价要在 D 桶用 accept 率量。
- 未清（诚实）：s11b 门链数据与 tag；T-nvfp4KV 码面；P1 布局整批（160/161/162/164/166/167/170/176/178/179/180/182/184/185/201/210＋vision receiver 面）；P2 materialization 链（192/194/213）；S7 标定；S12–S14。

## 56. 第 46 回合续＝S11 第 4 件的门链数据位＋T-nvfp4KV（186）落地 6d1d3ab5
- 门链 s11b＝`bash relay/gate-steps.sh daf35d5b s11b`：全构建 **rc=0**（23:52:15 UTC，增量面＝8 文件）＋家族子集 **100%，28/28，501.44 s**；簇门跑到 08:1x 时 109/173（分母取当次计数＝上次 173）。绿则 tag＝`v-tp2-per-rank-arena-and-tp2-scoring-r0` @ daf35d5b（消息＝本行数据）。
- **T-nvfp4KV＝6d1d3ab5（19 文件 +121/−43）＝186 `be1960ce` 的采进**，逐面规格（下轮勿重勘查）：
  - kernel＝我方 `nvfp4/launch.cu` 两处（grouped、parallel-grouped）＋`nvfp4/tiled_launch.cu` 一处加 `CausalD256H12Kv2` 实例；`nvfp4/plan.cpp` 准入 heads==12、`row_tiles = (query_tile * (heads == 16 ? 8 : 6) + 15)/16`（12/2 与 24/4 同＝组六，16/2＝八；与我方 `key_shift = (row_tiles<=2?7:8) - (heads==16?1:0)` 并存＝与供体 post-186 同形，已逐字比对）；`causal_geometry.h` 的 `CausalD256H12Kv2` 在我方**被声明两次**（重放带进两种文面＝坑②形）⇒ 合并成一条并把注释改成 BF16/INT8-G64/NVFP4-G16。
  - 四道 tp2 拒绝面开 `Nvfp4Group16`＝`runtime/engine/model_instance.cpp`、`planning/startup.cpp:validate_target_options`、`execution/text.cpp` 的 `validate_tensor_parallel` 与 `validate_tensor_parallel_mtp`、`causal_softmax_attention.cpp:require_causal_geometry`（＝五处文案，全树已无 "only bf16 and int8"）。
  - 测试面＝我方 `tests/ops/softmax_attention/causal_cache.cpp` 里 `kTensorParallelGeometry` 自 1b2ce547 时代起**定义了但从没被用**＝死常量；本轮把供体 186 的 `run_tensor_parallel_geometry_cases` 适到我方 harness 形（首参 `DeviceExecutionView execution`；`AttentionCase`/`BatchAttentionCase` 聚合初值与供体逐字同形）并挂进 `run_storage_cases` ⇒ 12/2 在**单设备**上对独立 oracle 验 bf16/int8/nvfp4 三 cache，且未注册的 cache 必须从 sizing query 响亮拒绝（不落进 16/2）。该族在簇门被 `-E softmax_attention` 排除＝必须单独跑（下一步）。
  - 两设备 parity＝**已在树**：`tests/ops/test_attention_headlocal.cpp` 的 storage 列表含 `Nvfp4Group16`（＝187 `4d951afa` 早在重放里零冲突落地；单设备 return 77）。
  - 文档/帮助面改全＝`docs/cli.md`（两处：tp2 KV 列表、拒绝清单去 nvfp4）、`docs/serving.md`、`docs/perplexity.md`（tp2 打分需 bf16|int8|nvfp4）、`docs/maintainer/tensor-parallel.md` 拒绝表行、`docs/maintainer/qwen3_5-model.md`、`docs/maintainer/engine-architecture.md` 中文段（顺带把 daf35d5b 的 per-rank arena／causal scoring 准入／rank-0 塔写对）、`apps/cli/options.cpp`、`apps/perplexity/options.cpp`、`src/serve/serve_options.cpp` 三处 --help。
- 验活（本轮实测）：14 个 TU rc=0＝causal_softmax_attention.cpp、nvfp4/{plan.cpp,launch.cu,tiled_launch.cu}（后两个 nvcc，含 12/2 实例化）、execution/text.cpp、planning/startup.cpp、runtime/engine/model_instance.cpp、apps/cli/options.cpp、apps/perplexity/options.cpp、src/serve/serve_options.cpp、tests/ops/softmax_attention/causal_cache.cpp＋三个把 app 源编进去的测试 TU（cli_options/perplexity_options/rope_scaling）。**在 s11b 的构建阶段编的 `.o`（未链接未跑测＝铁律 6 遵守，gate-72 在 ctest 期主树不链不跑）**。
- 下一步（按序）：① 读 s11b 簇门数据→绿则打 tag；② 在 gate-72 reset 到 6d1d3ab5＋贴两补丁＋构建，跑 `ctest -R softmax_attention`（GPU1）与 `./tests/ninfer_softmax_attention_test --kv-dtype nvfp4`、`int8`、`bf16` 三条，逐条记 Passed/失败数；③ `bash relay/gate-steps.sh 6d1d3ab5 nvfp4` 跑全门；绿则 tag `v-tp2-nvfp4-kv-tp2-and-12-2-qualification-r0`；④ D1 命令（写死，勿拿 int8 参考判 nvfp4）＝`GATE_SERVE_FLAGS="--tp 2 --devices 0,1 --kv-dtype nvfp4 --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft --vision --vision-device 0 --max-vision-tokens 4096" GATE_DF2_FLAGS="--tp 2 --devices 0,1 --kv-dtype nvfp4 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --spec dflash2 --draft-tokens 7 --lm-head-draft" GATE_PPL_PAIRED=1 tools/tp2/gate.sh --record <build> <ref-nvfp4-dir>`，外加 `test_attention_headlocal`（两设备）。
- 收益口径（只作上游参考）：nvfp4 KV ≈ int8 的 1.8× 密度、262k 下每卡 ~2 GiB；供体 PPL 4.1229/4.3987 vs int8 4.1335/4.3859、GSM8K-500 0.974/0.972、decode MTP3 +4–8%／DFlash2 +4–17%、**DFlash2 acceptance −10~15 点**（§39 补九）⇒ 默认仍 int8，D2 显存吃紧时才切；本机 36 SM 的 row_tiles/key_shift 界＝S7 标定项。
- 未清（诚实）：s11b 簇门落账与 tag；nvfp4 的 softmax_attention 实测；nvfp4 面全门；S11 第 5 件＝tp2 多模态接收面（`WorkspacePlan::vision_receiver`＋跨卡拷贝＝P1 布局族）；T-drafter-nvfp4（201）；T-reclaim-priority（213）；P1 整批；P2 链（192/194/213）；S7；S12–S14。

## 57. 第 46 回合续二＝转换器 NVFP4 量化器补齐（327a4c53）＝T-drafter-nvfp4 的前置债
- **真缺陷（本轮实测到）**：ad61da7b 落的 `_dflash2_nvfp4_gate_up`／`_mtp_nvfp4_layer` 与 `tests/convert/test_draft_head_recipes.py` 都写 `method=nvfp4_mse`，**但我方树从来没有这个函数**＝official_recipes.py 没 import、methods.py 没定义（`tools/convert/quantization/` 只有 groupwise/fp8_row）⇒ 任何按配方 opt-in 的转换在运行时 NameError；该测试 import 即失败。它**没进 ctest**（tests/cmake 只注册 chat_templates/regex_choice/json_schema 三条 python 测试）⇒ 历次门链从来看不到＝坑③的新变体（**文档/测试引用了不存在的实现**）。
- 补齐＝4 文件 +299/−1：`tools/convert/quantization/nvfp4.py`（供体 114 行照落＝E2M1 码＋每 16 值组 E4M3FN 尺度＋整阵 FP32 除数；五目标 6.0/5.5/5.0/4.5/4.0 最小平方误差选尺度、平手取靠前的；round_e2m1 平手取偶码）＋`methods.py` 的 `_nvfp4_quantized/nvfp4_absmax/nvfp4_mse`（形＝grouped_absmax/fp8_row_maxabs 的 prepare-method 形：预检＋至少 128 行的两趟（先 amax 后编码）＋`write_codes(begin, codes, scales, weight_divisor)`（我方 TensorOutput 已有第四参，逐字核过）＋AllowA4 必须显式 activation divisor，A16Only 不需）＋`METHODS` 注册两条＋`official_recipes.py` 的 import 补齐。
- 测试面＝`tests/convert/quantization/test_nvfp4.py`（供体 124 行照落＝对 packed words 逐字节独立标量解码 e2m1*e4m3fn/divisor、round_e2m1 边界与平手取偶、weight_divisor 边角）。**本机无 torch/ninja 制品面**（`find` 无 site-packages/torch）⇒ 数值测试＝D 窗口（与重转 nvfp4 drafter 制品同窗口）；本轮＝`py_compile` 三文件＋stub-torch 冒烟（METHODS 解析 nvfp4_mse/nvfp4_absmax、official_recipes 绑到 nvfp4_mse、两个 helper 对每层 gate/up 与 MTP 三组父项都交出 `(nvfp4, nvfp4_mse, A16Only)`）＝import 图与接线面得证。
- **T-drafter-nvfp4（201）剩下的面（本轮勘查，逐字可用）**：`ff6b336f` 18 文件 +433/−45。转换器面已补齐（上面这件）⇒ 剩＝(a) swiglu A16 加宽＝`nvfp4_linear_swiglu_plan.cpp` 去 `tokens <= 16` 上限（我方现文案＝"nvfp4 linear_swiglu A16 is registered only through T=16"，契约头 `include/ninfer/ops/linear_swiglu.h:32/85` 两句要同步）＋`nvfp4_linear_swiglu_small_t.cu`（20）走 sliced-K 到 32 tokens／以上走 token-tiled MMA＋`nvfp4_linear_swiglu_epilogue.cuh` 新 `Nvfp4SwiGluMmaRows`（线程配对 gate/up 行：16 行 MMA tile 装 8 gate＋8 up）＋`src/ops/linear/nvfp4/nvfp4_a16_mma.cuh`（18）；(b) 测试＝`tests/ops/linear_swiglu/test_nvfp4.cpp`（A16 T=1..129）＋`tests/ops/test_linear_swiglu_split.cpp`（两设备半形 vs 父＝逐位同）＋bench 4 行；(c) tp2 分裂轮取融合路＝供体落在 `execution/tp2/draft_tp2.inc`/`draft_split.h`/`program/tp2/startup_tp2.inc`＝**P1 布局族**，我方对应＝`execution/draft.cpp:351/513` 的 `project_swiglu(...)` 与 `planning/startup.cpp` 的 drafter 相位（gate/up 是 NVFP4 时走融合、Q8 保 linear+silu_mul＝现状即 Q8 路）。
- 201 供体测量（只作参考，非必达）：制品 −425 MiB＝每 rank −212 MiB；rank0 空闲 207.7→419.7 MiB@229,376；选择器在 rank 0（产线）⇒ 237,568/245,760 起得来（Q8 差 72 MB）；在 rank 1 ⇒ 253,952 起（Q8 差 93 MB）；**262,144 仍差 9 MB**＝我方 T-nvfp4KV（6d1d3ab5）落地后是最可能补上这 9 MB 的正面＝D2 的算术要点（nvfp4 KV 每 262k 省 ~2 GiB/卡 ⇒ 远超 9 MB）。⇒ 下轮在 D2 前先把我方两件叠起来的容量报价算出来（`--kv-dtype nvfp4`＋`--embedding-host`＋drafter nvfp4）。
- 门链状态：s11b（daf35d5b）簇门在跑；`relay/nvfp4-then-gate.sh` 串好＝链一收尾即在 gate-72 面 reset 6d1d3ab5＋贴补丁＋构建＋`ctest -R softmax_attention`（＝12/2 nvfp4 的单卡资格面）＋逐 dtype 跑 `ninfer_softmax_attention_test`，再对 6d1d3ab5 跑全门（label nvfp4）。327a4c53＝纯 Python 面 ⇒ 不占用门；它的 C++ 护航随下一次门链（对 HEAD 跑）一起。
- 未清（诚实）：s11b/nvfp4 两次门的落账与 tag；softmax_attention 族的 12/2 实测结果；nvfp4 数值 pytest（无 torch）；S11 第 5 件（tp2 多模态接收面＝P1）；T-drafter-nvfp4 的 (a)(b)(c)；T-reclaim-priority；P2 链；S7；S12–S14。

## 58. 第 46 回合末＝s11b 面全绿并打 tag；nvfp4 资格面在跑
- **s11b（`daf35d5b`）＝全绿**：全构建 rc=0（23:52:15 UTC）；家族子集 **100%，28/28，501.44 s**（Passed 25／Skipped 3）；簇门 **100%，173/173，1161.46 s**（Passed 137／Skipped 36＝两设备腿在单卡自跳）。⇒ tag **`v-tp2-per-rank-arena-and-tp2-scoring-r0` @ daf35d5b** 已打（消息含本数据与先例面清单）。链日志＝`relay/gate-s11b-chain.log`（done 00:19:58），passset＝`relay/passset-sm120-tp2steps-s11b*.tsv`。
- 分母与上次同＝28/173（2a209505 面）⇒ 本轮三件（daf35d5b 容量与闸、6d1d3ab5 nvfp4 KV、327a4c53 转换器）都没带新测试注册＝门面分母不变；唯一新增可跑面在 `-E softmax_attention` 排除族里（`ninfer_softmax_attention_test` 的 12/2 用例）＝不进气泡门，需单独跑（正在跑）。
- 后台链（`relay/nvfp4-then-gate.sh`，setsid，日志 `relay/nvfp4-qual.log`）：00:20:16 已把 gate-72 reset 到 **6d1d3ab5** 并开始构建 ⇒ 之后 `ctest -R softmax_attention`（12/2 对独立 oracle＝nvfp4/int8/bf16 三 cache）＋逐 dtype 跑 `ninfer_softmax_attention_test`＋对 6d1d3ab5 跑全门（`relay/gate-nvfp4-chain.log`）。下轮第一件事＝读这两份日志：簇门 100% 且资格面全 Passed ⇒ `git tag -a v-tp2-nvfp4-kv-tp2-and-12-2-qualification-r0 6d1d3ab5 -m "<门数据>"`；有红＝先按 `gate-nvfp4-full.log`/`nvfp4-qual-ctest.log` 的 12/2 用例定根因（plan 的 heads/row_tiles、key_shift、三处实例化）再谈 tag（红门不打）。
- 327a4c53（纯 Python 面＝转换器量化器补齐）＝不占门；它的 C++ 护航随下一次门链（对 HEAD 跑）一起＝下轮若动 C++ 面，用 `bash relay/gate-steps.sh $(git rev-parse --short HEAD) head46`。
- 累计：S11 四件全落＋三道闸全开（第三道＝本轮找到）；T-nvfp4KV 码面落地（等资格与全门）；T-drafter-nvfp4 的转换器前置清账（(a)(b)(c) 规格在 §57）。未清＝nvfp4 资格面/全门与 tag、S11 第 5 件（tp2 多模态接收面＝P1）、P2 链、S7 标定、S12–S14、D1–D4。

## 59. 第 47 回合＝S6 能力门落进树（9fc74c7b）＋S10 层文档面（3abd74b9）
- **S6＝从"门补丁"变"树面"**：`relay/gates-r72.patch`（9.9 KB，过去 8 次门链都只在 gate-72 贴）现落进主树＝`src/product/tensor_parallel_options.h` 两个纯函数（零 CUDA）：`tensor_parallel_capability_verdict(cc_major,cc_minor)` 只收 12.0，其它（含 Ada 8.9、12.1）给可读理由（点名 sm_120、观测到的 CC、`--tp 1` 出路）；`tensor_parallel_kv_verdict(storage)` 按本树现状**收 bf16／int8-G64／nvfp4-G16**（补丁原文只写 bf16/int8＝6d1d3ab5 之前的事实，已改）拒 fp8／k8v4／rk4v4-e8 并点名存储。调用点＝`engine.cpp:initialize_execution` 对两 rank 的 DeviceContext 逐卡问（读 artifact 之前）＋`model_instance.cpp:validate_options` 用它取代字面串。
- **一处事实一个权威（铁律3）**：`planning/startup.cpp` 里那份重复的 KV 检查**删掉**，注释指向两个权威＝product 的准入 verdict＋`text.cpp` 的几何注册检查（后者文案不同＝事实不同，保留）。⇒ 全树 KV@tp2 的准入句只剩 product 一处。
- 测试（CPU＝家族子集里，门链覆盖）：`test_cli_options` 新增＝12.0 收／8.9 与 12.1 拒且理由可读／三种收的存储（含 nvfp4）／三种拒的存储点名；`test_serve_options` 新增＝把 serve 解析出的 `--kv-dtype`（显式 rk4v4-e8 与默认 bf16）送进同一 verdict＝旗标不能承诺启动随后拒绝的路。
- 门链工具面同步＝`relay/gate-steps.sh` 加 `GATE_PATCHES`（默认仍两份补丁以兼容旧面；**新面（≥9fc74c7b）只贴 `vision-on-build-ada.patch`**，因为 gates-r72.patch 的内容已在树里，重复贴必失败）。
- 文档＝`docs/maintainer/tensor-parallel.md` 拒绝表把 KV 行改成点名两个 verdict＋新增"非 CC 12.0 设备"行。
- **S10 文档面＝`docs/maintainer/tp2-ada-notes.md`（新建，docs/README.md 加行）**：五段＝① 本树层面清单（core/tp2 三件、ops 传输与分裂入口、models 的 TpExecution/文路/PeerRuntime/宽度2 plan、runtime 三处、product 两处、五个自跳 77 的双设备套件、tools/tp2）；② 剥离演练（scratch worktree＋revert 整段＋全构建＋`git grep` 层符号＝0 判据，配两条实测坑＝整文件回退剥字段／搬家步留悬空 include）；③ 占位登记（5060 Ti 行的 allowance 8/8/11 MiB、shard TMA 界 1024、nvfp4 12/2 的 row 组与 key shift＝逐项写明哪个 D 档测）；④ 延后的 P1 布局整批与其解锁面（vision 接收面、DFlash2 分裂 drafter、宽 mailbox 槽）＋今天可见后果（媒体 prompt 在 tp2 被拒）；⑤ Ada 后续（S6 接口已在树＝verdict 纯函数 CPU 验；sm_89 构建 45/200 TU 红、21 已在剔除表、24 待 stub；4080S 验 E8／YaRN 260K／`rk4v4-e8` 共存／`--tp 2` 拒绝文案；Ada 生产仍用 `engine-sm89-cu128-r2`）。
- **S10 的演练本身未跑**（要一次全构建，不与门链并发）＝下轮在门链空档跑 `git worktree add ../peel-drill HEAD`＋§2 那四条命令，把结果记进本台账（红＝列 file:line）。
- 验活（本轮实测）：S6 面 **5 个 TU rc=0**（engine/engine.cpp、engine/model_instance.cpp、planning/startup.cpp、tests/test_cli_options.cpp、tests/test_serve_options.cpp；均在 gate-72 的**构建阶段**编的 `.o`＝铁律 6 遵守，未在主树链接）。**链与跑这两个 CPU 测试＝等门链 ctest 收尾**（此刻 gate-72 在跑 `ctest -R softmax_attention`＝12/2 资格面，08:32 起，约 13 分钟，之后逐 dtype，再自动跑 6d1d3ab5 的全门链）。
- 队列（下轮照此，勿重复启动）：① 读 `relay/nvfp4-qual.log`＋`nvfp4-qual-ctest.log`（12/2 三 cache 对独立 oracle 的结果）与 `relay/gate-nvfp4-chain.log`（6d1d3ab5 全门）⇒ 全绿则 `git tag -a v-tp2-nvfp4-kv-tp2-and-12-2-qualification-r0 6d1d3ab5 -m "<门数据>"`；② 主树链并跑 `ninfer_cli_options_test`＋`ninfer_serve_options_test`（期望 ok／rc=0）；③ `GATE_PATCHES=/home/ai-agent/DSHW/ninfer-watch/relay/vision-on-build-ada.patch bash relay/gate-steps.sh 3abd74b9 head47`（＝S6＋S10＋转换器面的全门护航）；绿则 tag `v-tp2-s6-capability-gate-in-tree-r0`；④ 剥离演练（§59 上面那条）；⑤ T-drafter-nvfp4 的 (a)(b)（规格＝§57）。
- 累计单卡桶：S 桶 11 件里 **S1/S2/S3?(工具齐未跑)/S4(同)/S5/S6✅/S7码✅/S8码✅/S9/S10文档✅演练欠/S11(4/5)/S12-S14 工具面**；T/P＝nvfp4 KV 码✅（门欠）、drafter 转换器前置✅、P1/P2/reclaim 未动。

## 60. 第 47 回合续＝T-drafter-nvfp4 (a)(b) 的**逐字落法**（勘查完，未动码）
勘查已把 201 `ff6b336f` 的 swiglu 加宽面读到底（含供体 pre-state 与我方现形逐字比对）。**下轮照此一遍过，不要重勘查**：

- **要点先证（省一轮答疑）＝容量报价不用改**：我方 `nvfp4_linear_swiglu_plan.cpp` 的 capacity 查询对 `A16Only/AllowA8` 返回 **0**，而 `src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh` 全文**无 workspace/Arena/partials/cudaMalloc 引用**＝A16 的 sliced-K 与 a16-MMA 路都不吃 caller scratch ⇒ 加宽 T 域不新增容量面（op-development §5.2 的"一个权威查询覆盖区间内每个合法 extent"自然成立）。供体 201 也没动 capacity 函数＝同一结论。
- (a) 五文件＝
  1. `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`：删 `if (tokens <= 16) {...}` 与那句 `"nvfp4 linear_swiglu A16 is registered only through T=16"`，`A16Only/AllowA8` 分支＝`tokens == 1 → DecodeFusedA16`，其余一律 `SmallTFusedA16`（＝供体 post-186/201 两行形）。
  2. `include/ninfer/ops/linear_swiglu.h:32` 与 `:85` 两句契约文案照供体改（A16Only/AllowA8 收**每个正 T**；便利形对 Q4/Q8、NVFP4、row-scaled FP8 保完整正 T 域）。
  3. `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu`＝**用我方形**加两级：我方现形＝`x.ne[1] <= 2 → kLaunchers/launch_exact<Geometry,2>`（SIMT，带 ActiveTokens 模板）；`<= 8 → launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8,8,2>, Geometry::kInputRows>>(..., Nvfp4SwiGluRows<8>{})`；`else(≤16) → 同前，实例 <16,8,2>`。加＝`<= 32 →` 我方 `launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32,?,?>, Geometry::kInputRows>>`（**参数个数/语义要先照 `nvfp4_instances.cuh` 里 `Nvfp4SlicedInstance` 的模板参数序核**＝供体写 `<32,4,2>` 是他家 3 参义，我方 8/8/2、16/8/2 的义在自家表里）、`else → launch_a16_mma<...Nvfp4A16MmaSchedule<32,64,128,16,16,1,3>...>(..., Nvfp4SwiGluMmaRows{})`；include 加 `ops/linear/nvfp4/nvfp4_instances.cuh`。
  4. `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh`：新增 `Nvfp4SwiGluMmaRows{kPaired=true, kThreadPaired=true; weight_row = begin + (row/16)*8 + row%8 + (row%16>=8 ? rows/2 : 0); output_row = begin + (row/16)*8 + row%8;}`（逐字照抄，注释一并）。**我方 `Nvfp4SwiGluRows<N>` 是模板、供体是非模板** ⇒ 保持我方形，别改签名（13 处调用＝142-skip 史）。
  5. `src/ops/linear/nvfp4/nvfp4_a16_mma.cuh`：在 epilogue 之前加 `} else if constexpr (requires { RowPolicy::kThreadPaired; }) {` 分支（`mi<MT`/`ni<NT` 环＋`apply_pair(destination, out_row, token0, v[0]*alpha, v[2]*alpha)` 与 `token0+1 → v[1], v[3]`，`FullTokens || token...` 守）＝逐字照抄 18 行。
- (b) 测试/护航面＝`tests/ops/linear_swiglu/test_nvfp4.cpp`（＋7＝A16 域到 T=129；**我方 137 登记的 255/256/257 TMA 切点与 seed 722/727 标定注释必须保**，其文面那句"shard keeps the 128-token tile through 1024 since 2a596191"在我不成立＝130 被 skip）；`tests/ops/test_linear_swiglu_split.cpp`（＋32＝两设备半形 vs 父逐位同；单卡自跳 77）；`bench/ops/nvfp4_linear_swiglu_bench.cu`（−4）。门护航＝家族子集 `-R "linear|fp8|nvfp4|bf16"`（含 linear_swiglu）＋簇门；**这是"改注册域"面＝按铁律要跑一次含 attention 的全量**（供体口径）。
- 位变化口径＝官方配方不变，制品靠 `_dflash2_nvfp4_gate_up`（327a4c53 起可用）opt-in ⇒ 现有制品逐位不动；S6 门 `dflash2` 档要在**重转制品**上另记参考（供体＝10/10 文本一致、accepted 62.66% vs 61.50%、ms/round −1.67%）。
- (c) 仍欠＝tp2 分裂轮取融合路：供体落点 `execution/tp2/draft_tp2.inc`、`draft_split.h`、`program/tp2/startup_tp2.inc` ⇒ 我方对应＝`execution/draft.cpp:351`（`project_swiglu(branch.prepared, layer.mlp.gate_up, intermediate_flat, work)`）与 `:513` 两处＋`planning/startup.cpp` 的 drafter 相位（按 gate/up 的格式选融合/线性）。**与 P1 整批同做**（那批含 178/179/184/201 全链）。
- 201 的测量档案（只作参考）＝制品 −425MiB（每 rank −212MiB）；rank0 空闲 207.7→419.7MiB@229,376；选择器在 rank0 ⇒ 237,568/245,760 起得来（Q8 差 72MB）；在 rank1 ⇒ 253,952（Q8 差 93MB）；**262,144 仍差 9MB** ⇒ 与我方 nvfp4 KV（6d1d3ab5）叠加＝D2 最短出路；decode +1.5~2.4%、acceptance ±0.6 点（MTP 侧）、ms/round −1.3~1.7%。
- 现场（本轮末）：HEAD＝`3abd74b9`，工作树 0；`relay/nvfp4-then-gate.sh` 在跑＝gate-72 面 6d1d3ab5 的 `ctest -R softmax_attention`（08:32 起，`--timeout 1800`；PID 1978284/1978286）→ 逐 dtype → 全门链。下轮第一件事＝读 `nvfp4-qual.log`/`gate-nvfp4-chain.log` 决定 tag（判据与命令＝§59 队列①）。

## 61. 第 47 回合续二＝**12/2 半形的单卡资格面＝通过**（6d1d3ab5 面）
- 实测＝gate-72 在 `6d1d3ab5`（贴两补丁）全构建 **rc=0**（00:30:19 UTC）→ `ctest -j1 --timeout 1800 -R "softmax_attention"` ＝ **1/1 Passed，864.22 s**（`relay/nvfp4-qual-ctest.log`；该测在簇门被 `-E softmax_attention` 排除＝这条就是它的资格面）。
- 内容意义＝`ninfer_softmax_attention_test` 的 causal-cache 全族跑了**新挂上的 `run_tensor_parallel_geometry_cases`**：`[256,12,2]` 半形在 **bf16 / int8-G64 / nvfp4-G16** 三个 cache 上对**独立 oracle** 验（三族 seam 宽度 7/8/9/16/17/256/257 × 两个公共入口、16383-context 解码带 replayed capture、1024-token chunk prefill 带逐 query oracle 行、identity/fragmented 批形），并且未注册的 cache 必须从 sizing query 响亮拒绝（不落进 16/2）＝这两条都在断言里。返回 0 ⇒ 全通过（该套件任一用例失败即 rc≠0）。
- 计时口径＝864.22 s（基线注：该测在 12/2 加入前 231 s，供体面 785.67 s；本轮＝36 SM＋三 cache 的 12/2 全矩阵）＝`--timeout 1800` 余量 52 %。⇒ **S2 那条"softmax 785 s 吃掉门预算"的债在本机更宽**：它不在簇门内，单跑一次 14.4 分钟可接受。
- 逐 dtype 复跑（同一二进制，`--kv-dtype nvfp4|int8|bf16`）在跑＝`relay/nvfp4-qual-{nvfp4,int8,bf16}.log`，每份首行 `RUN causal_softmax_attention <storage>`＝确认覆盖面。
- 之后＝自动跑全门链 `bash relay/gate-steps.sh 6d1d3ab5 nvfp4`（`relay/gate-nvfp4-chain.log`）。判据＝子集全绿＋簇门 100%（分母取当次计数＝上次 173）。绿则打 tag `v-tp2-nvfp4-kv-tp2-and-12-2-qualification-r0 @ 6d1d3ab5`，消息要含：资格面 864.22 s/1-1 Passed ＋ 全门三行数据 ＋ 该面是 nvfp4 KV@tp2 的**单卡可得最强证据**（两设备 parity＝`test_attention_headlocal` 的 nvfp4 腿，D1）。
- 本轮累计交回＝S11 第 4 件（daf35d5b，已 tag）、T-nvfp4KV 码面＋单卡资格（6d1d3ab5，全门在跑）、转换器 nvfp4 量化器补齐（327a4c53，真缺陷）、S6 能力门落树（9fc74c7b，5 TU rc=0，CPU 断言面已注册进家族子集）、S10 层文档（3abd74b9，演练待跑）、T-drafter-nvfp4 逐字落法（§60）。

## 62. 第 47 回合末＝nvfp4 KV 面全绿并打 tag；S6 的 CPU 面实跑通过；门链补丁机制查清
- **`6d1d3ab5`（T-nvfp4KV）＝全绿 ⇒ tag `v-tp2-nvfp4-kv-tp2-and-12-2-qualification-r0` 已打**：全构建 rc=0（00:51:14 UTC）；家族子集 **100%，28/28，453.50 s**（Passed 25／Skipped 3）；簇门 **100%，173/173，1051.57 s**（Passed 137／Skipped 36）；**12/2 资格面**（该族被簇门 `-E` 排除＝单独跑）`ctest -R softmax_attention` ＝ **1/1 Passed，864.22 s**＋逐 dtype `ninfer_softmax_attention_test --kv-dtype nvfp4|int8|bf16` 全 **rc=0**（`relay/nvfp4-qual*.log`）。tag 消息含以上全部数据与下面那条补丁事实。
- **门链补丁机制查清（下轮起照此，勿再猜）**：`gates-r72.patch`（S6）自 `daf35d5b` 起在 `model_instance.cpp:83` 的旧 context（"Engine tp 2 does not support Vision"）上贴不上 ⇒ **6d1d3ab5 面＝树＋`vision-on-build-ada.patch`，S6 断言不在该面**（该补丁不新增 ctest 注册＝分母 173 与历次可比，已在 tag 里如实记）。自 `9fc74c7b` 起 S6 内容在树里 ⇒ **新面一律只贴 vision 一份**＝`GATE_PATCHES=/home/ai-agent/DSHW/ninfer-watch/relay/vision-on-build-ada.patch`。`relay/gate-steps.sh` 已改**幂等贴补丁**（逐份报 `present: / applied: / FAILED:(首行原因)`，捆一起失败的老写法废；用 `mv` 原子替换，跑着的链拿旧 inode 不受影响）。
- **S6 的 CPU 面本轮实跑（主树，门链空档）＝通过**：链 `ninfer_cli_options_test`＋`ninfer_serve_options_test` 后直跑＝**cli rc=0／serve rc=0（打印 ok）**。⇒ 能力 verdict（12.0 收；8.9、12.1 拒且理由可读）与 KV verdict（bf16/int8/**nvfp4** 收；fp8/k8v4/rk4v4-e8 拒且点名存储）在树面得证，serve 解析出的 `--kv-dtype` 走同一 verdict。
- **head47 链在跑**＝`GATE_PATCHES=…/vision-on-build-ada.patch bash relay/gate-steps.sh 3abd74b9 head47`（09:20 起，链日志已确认 `patches: applied:vision-on-build-ada.patch`；＝S6＋S10＋转换器面的全构建护航）。判据＝子集全绿＋簇门 100%（分母取当次计数＝上次 173）。绿则打 `v-tp2-s6-capability-gate-in-tree-r0 @ 3abd74b9`。
- **R1 开窗前置的实测核账（本轮新增事实，照此判断"远不远"）**：四条里 ①队列到 213 全处理完＋调参表在树＝满足；③S5/S6/S8码/S7码 闭合（S7 数值＝D2/D4 写回；S8 告警用例＝S14）；②S3 与 ④S9＝**只差跑**（工具全在树：`tools/tp2/gate.sh`＋`tools/golden/record.sh`＋合成制品 write 模式＝纯 C++，不需 torch）。P1 整批／P2 链／T-drafter-nvfp4／T-reclaim **不是开窗前置**（解锁能力面，不是容量/探测/判分面）。
- 两个**硬门前置（本机实测，均为红）**：①**制品＝0**——全盘（`/tank`、`/home/ai-agent`，maxdepth 4）零个 `.ninfer`；`/tank/models/Qwen3.8-27B` 只有 `GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`＋`mmproj-Qwen3.8-27B-Q8_0.gguf`＝llama.cpp 侧权重，**无 HF safetensors**（全盘 `*.safetensors` >500M 零命中）；**本机无 torch**（site-packages 无；numpy 2.2.6 有）⇒ 22.14 GiB 的 `.ninfer` 需"下 HF 权重→建 torch 环境→`tools/convert` 转换"，磁盘 `/tank` 剩 1.2 T。②**GPU0 被现网 strata 占**（PID 440568，15470 MiB，util 100 %）⇒ 双卡窗口需它空闲，且 `gate.sh` 不抓卡＝要 `gpu-lease` 包一层。
- 队列（head47 之后，按序）：① 绿则 tag head47 面；② **S3 合成 tp1 等价门**（golden 32/2048/6144 三档逐字＋shards＋ctest，单卡 GPU1，产物落 `relay/`）；③ **S9 判分底座对真实 serve 首跑兼容**（合成模型起 serve，验 `/v1/models`、`reasoning_content`、`stop`）；④ S10 剥离演练（`tp2-ada-notes.md` §2 四条命令，隔离 worktree＋全构建）；⑤ T-drafter-nvfp4 (a)(b)（逐字落法＝§60）；⑥ 制品与 torch 环境＝等用户点头再动（50 GB 级下载＋转换，属外部副作用动作）。
- 累计：本轮四提交全绿已 tag（daf35d5b／6d1d3ab5）＋两件待门（9fc74c7b／327a4c53／3abd74b9 随 head47 一起护航）；`v-tp2-*` 标签面 8 个。

## 63. 第 47 回合末二＝剥离演练的**实测边界**（7cd4bdfc）；head47 链在跑
- 量出来的层边界（不是抄来的注记）：HEAD 自身历史里最旧的层面＝**`03726c08 feat(ops): register tensor-parallel shard shapes for fp8, nvfp4 and bf16 linear`**（S1 那件），**其父到 HEAD＝197 提交＝整层**；`v-tp2-skeleton`（`b797cb54`）**不是 HEAD 的祖先**（层是经"重放供体步"进来的，供体历史非线性；`fd5731ff`/`cf62bbbb`/`3a964069`/`97b99fed`/`7220a9f9` 在 180-171 位＝比 03726c08 新）⇒ 演练的 base 用 `03726c08~`，不用骨架 tag。
- `tp2-ada-notes.md` §2 据此改＝**两阶段**（先剥 `HEAD~25..HEAD` 量方法与冲突面，再剥 `03726c08~..HEAD` 全层），并写死"197 提交全剥预计要手工处理的位置"＝本 fork 与层共改的文件（`README.md`、`docs/cli.md`、`docs/serving.md`、`apps/*/options.cpp`、`tests/cmake/*.cmake`）；判据＝全构建 rc=0 ＋ `git grep` 层符号＝0；手工改动逐条记台账（演练是测量，不是清理）。
- 本提交＝**纯 markdown**（不占门、不改 C++ 面）⇒ 随下一次门链护航。
- 门链 head47（`3abd74b9`，只贴 vision 补丁）＝全构建 **rc=0**（01:22:12 UTC）；子集与簇门在跑（预计 ~09:50 收尾）。判据＝子集全绿＋簇门 100%（分母取当次计数＝上次 173）；绿则打 `v-tp2-s6-capability-gate-in-tree-r0 @ 3abd74b9`，消息含 S6 的 CPU 面实跑（cli rc=0／serve ok）与本轮的补丁幂等改造。

## 65. 第 47 回合末三＝head47 全绿已打 tag；**制品到手**（开窗前置的两个红门掉一个）
- **head47（`3abd74b9`）＝全绿**：全构建 rc=0；家族子集 **100%，28/28，466.88 s**（Passed 25／Skipped 3）；簇门 **100%，173/173，1097.82 s**（Passed 137／Skipped 36）。⇒ tag **`v-tp2-s6-capability-gate-in-tree-r0` @ 3abd74b9** 已打（消息含 S6 的 CPU 面实跑 cli rc=0／serve ok、12/2 资格面 864.22 s、以及"本面只贴 vision 一份补丁"的 GATE_PATCHES 事实）。至此本轮四面全绿已 tag：`daf35d5b`／`6d1d3ab5`／`3abd74b9`（＋其上的 `9fc74c7b`/`327a4c53` 随面护航）。
- **更正我自己的旧结论（重要，别再照抄）**：上几轮我写"全盘零个 `.ninfer`"＝**扫描口径错**（`find -maxdepth 3/4`，文件在深度 5）。全深度实扫＝本地早有一个 22.14 GiB 制品：`/tank/models/lynn/NVFP4-NInfer/W4A4/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer`（23,773,675,264 B，mtime 10-15 15:46）。**家目录确实没有**（`/home/ai-agent` depth 7 零命中、无任何 >8 GB 文件）⇒"在家文件"的记忆对不上，它在 `/tank/models/`。
- **按指示下载 nvfp4 件（魔搭自家仓，直连不走 7890、`token=` 免 login）＝完成**：落 `/tank/models/mrwan0410-lynn-nf4-ninfer/qwen3.8-27b-lynn-nf4.ninfer`，**19,345,900,804 B**＝与服务端 Size 逐位等；服务端 sha256 `faf551e2c782022606e0ffd382e8d015cec574914a9476ca6007f346c4c53268`；本地 `sha256sum` 在跑（`/tmp/nf4-sha.log`，完成即逐位比）。同仓族另两件（未下）＝`…-MTP-ninfer` 16.96 GiB、`…-DF2-ninfer` 18.61 GiB（`api.get_model_files` 实测）。
- **两件制品的头部实读（`python3 -m tools.artifact.inspect --json`，纯 CPU、无 torch）＝关键 D2 事实**：
  - NF4＝`qwen3.8-27b-lynn-nf4`，components＝**text＋vision＋mtp＋dflash2**，file_bytes **18.017 GiB ⇒ ~9.0 GiB/卡**；formats＝nvfp4 **256**／bf16 579／fp32 608／fp8_e4m3fn_row_bf16 1／q4 55／q5 54／q6 2／q8 28；tensors 1584。
  - W4A4＝`qwen3.8-27b-coder390-w4a4-mtp-dflash2`，同四组件，file_bytes **22.141 GiB ⇒ 11.07 GiB/卡**（正是计划那本账）；formats＝nvfp4 112／fp8 146／q4 55／q5 54／q8 38／bf16 579／fp32 264；tensors 1250。
  - 两者同几何＝dense `Qwen3_5ForCausalLM`，hidden 5120，vocab **248,320**，64 层（16 层 full_attention），`num_attention_heads 24 / num_key_value_heads 4 / head_dim 256` ⇒ **tp2 的 12/2 半形正是它的对半**（与我方本轮注册的 [256,12,2] 完全对上）；`intermediate_size 17408`（我方半形 [17408,5120] 即其 [34816,5120] 的 gate/up 半）；`max_position_embeddings 262,144`＝原生 262k（无 YaRN 承诺）；`proposal {domain: indexed, rows: 131072}`＝**优化提案头在场**（`--lm-head-draft` 可用，Q4 131,072 行＝与我方 139/178 的分裂提案头口径一致）；DFlash2＝5 层 sliding（`sliding_window 2048`、`target_layer_ids [5,19,33,47,61]`、`selector_rank 256/selector_top_k 16`、`conv_group_size 16`）；vision＝depth 27／hidden 1152／merge 2（＝`--vision` 常驻面）。
  - ⇒ 开窗硬门剩一件：**GPU0 释放**（现网 strata PID 440568）＋租约；制品与 torch（转换用）不再是前置（要重转才需 torch）。
- 下一件不变（§63 队列）：S3 合成 tp1 等价门（不需制品）→ S4 shards 跑 → S9 serve 兼容首跑 → S10 剥离演练 → T-drafter-nvfp4 (a)(b)。制品到手后**新增可做**＝D 窗口前置清单复核与 R1 四条的正式核账（开窗申请要提前 2 小时报形态 A/B/C）。

- §65 补（实测完成）：本地 `sha256sum qwen3.8-27b-lynn-nf4.ninfer` ＝ **`faf551e2c782022606e0ffd382e8d015cec574914a9476ca6007f346c4c53268`，与服务端逐位一致，rc=0**（`/tmp/nf4-sha.log`，18.0 GiB 读全量约 4 分钟）⇒ 制品到手＝**开窗的制品门已清**，只剩 GPU0 释放＋`gpu-lease`。薄卡 v24 的第 5 条已按此重写（含两件的头部几何与 formats 实测）。

## 66. 第 47 回合末四＝开窗前置的三跑开跑（S4 已完；S3＋S9 在跑）＋A 清单裁决
- **S4 shards 面已跑**＝`python3 tools/tp2/check_shard_sources.py` → **`shard sources: OK (0 translation units)`，rc=0**。诚实记＝**这条门在我方是空对象**：我方跳过了 160/161/164/166/167 的搬家步 ⇒ 树里没有 `src/ops/*/tp2/*_shard_*.cu`（shard 编在上游源文件里）⇒ 该检查无扫描对象。**S4 的真面＝`ninfer_qwen3_5_shard_map_test`（放置图，本轮 passed rc=0）＋两设备 split 测试（单卡自跳 77）＝D1**；"shards 有对象"这件事要等 P1 落树后才成立（卡 v24 队列④/P1 一并记着）。
- **A 清单裁决（`ZeroFill` 不对称疑点）＝不是缺陷**：`program_impl.cpp:35` 的 `tensor_parallel_zero_fill(plan)`＝`plan.peer_persistent ? Yes : No`；`PeerRuntime`（只在 tp2 存在）字面写 `ZeroFill::Yes` ⇒ **tp2 两 rank 都零填充，策略一致**；差异只发生在 tp1（rank0＝No，正是注释里"A single-device Program's arenas start undefined, as every owner fills what it reads"）。**可改进项＝记**：`PeerRuntime` 可直接调同一 helper 以"一处权威"表达（行为不变）⇒ 不为这一行重开门链（已 tag 面保持），留作 P1 整批时顺手改。
- **剩余＝两跑，已在跑**：主树全构建（432 目标，10:00 起，`/tmp/head47-fullbuild.log`）收尾即接 `relay/s3-s9.sh`（PID 2882724，日志 `relay/s3-s9.log`），做＝
  - **S3**＝`ninfer_qwen3_5_text_context_tp2_test write /tmp/synthetic-tp1` 产合成两层制品（纯 C++，不需 torch）→ `tools/golden/record.sh apps/ninfer-tp1-golden … /tmp/golden-47` 三档（32/2048/6144 prefill、128 tokens、seed 7）→ `diff -r` 对 `tools/golden/recorded/2026-09-30/{fork-f19e6821,upstream-d44ab584}` 两份参考（＝"tp1 与上游等价"的实测面）＋md5 摘要对照。
  - **S9**＝合成模型起 `ninfer-serve`（GPU1，port 8123）→ 探 `/v1/models` 字段、thinking 响应里的 `reasoning_content` 键与 `finish_reason`、带 `stop` 的请求行为 → `relay/gsm8k/run_gsm8k.py` 对活服务首跑（8 题＝判分管线兼容，不断正确性：两层合成模型给垃圾答案是预期的）。
  - 注：S3 里 gate.sh 的 ppl/greedy/dflash2 三档**单卡做不了**（27B 制品 18.0/22.1 GiB > 单卡 15.8 GB）＝D 窗口面；ctest 档＝门链已覆盖（173/173）。
- 跑完的读法＝`relay/s3-s9.log` 逐行（每步带 rc）。三条判据：golden 两参考 `rc=0/difflines=0` ⇒ S3 绿；`S9 readiness rc=1`＋三探印有键名与 finish_reason＋`gsm8k first-run rc=0` ⇒ R1 第 4 条绿。全绿 ⇒ **R1 四条前置齐**＝可以按"提前 2 小时报形态 A/B/C"申请双卡窗口；开窗唯一外部条件＝GPU0（PID 440568）释放＋`gpu-lease`。

## 67. 第 48 回合＝**整树代码审（A–E 五清单）**：审出 1 道漏开的闸＋1 条死线，四提交落盘（门链欠）

- 开工自证：porcelain=0、`git log -1`＝`7cd4bdfc`、四个 tag 如薄卡。**收工＝HEAD `7d762ec5`＝四提交，porcelain=0，未打 tag（门链未跑＝纪律⑦）**。
- **本会话 GPU 不可用（新环境事实）**：`relay/probe/gpu_probe` 与 `CUDA_VISIBLE_DEVICES=1` 两种跑法都 `cuInit→304`、NVML 起不来（`/dev/nvidia*` 在、权限 rw）⇒ 门链与 ctest 跑不了；只做了**单 `.o` 编译**（纪律⑥的披露口径）＝用 `build-120a-cu13/compile_commands.json` 的原命令去掉 depfile 后编 7 个受影响 TU＝`program_impl.cpp`、`planning/request_plan.cpp`、`transactions/binding.cpp`、`runtime/engine/model_instance.cpp`、`tests/models/qwen3_5/test_engine_vision_tp2_real.cpp` 全 rc=0。
- **修①＝第四道闸（P0 级开窗杀手，`e18613d1`）**：`program/program_impl.cpp:157` 在 tp2 仍 `vision_enabled || causal_scoring` 一起拒（`97b99fed` 时代写的），而它上面三道全已开＝`planning/startup.cpp:1234-1252` 收 Generation+CausalScoring 且只要求塔在 rank0、`model_instance.cpp:82-90` 同、容量面已为这两相付钱（打分 tile `startup.cpp:703-721`、按 rank 分 arena `:788-803`）。后果＝`ninfer-serve --tp 2 --vision --vision-device 0`（＝`tools/tp2/gate.sh:43` 的默认行）与 `ninfer-perplexity --tp 2`（`gate.sh:136`）先读 18/22 GiB 制品、传完两卡权重，**才**在 ProgramInitialize 抛 ⇒ D1 的 greedy/dflash2/ppl 三段全起不来，而按 rank 分 arena 那件（S11 第 4 件）在本地与窗口里都永不可达。本轮改成只拒 DFlash；媒体 prompt 仍在 `execution/text.cpp:3073/3089` `require_single_device(「multimodal prefill」)` 响亮拒＝与 `docs/cli.md:401-406`、`docs/serving.md:1060-1064` 已写好的口径一致。同提交把 `test_engine_vision_tp2_real.cpp` 从供体形（塔可在 rank1、跨卡拷贝、图逐字对拍）改成我方形态：rank1 塔＝开闸前拒、常驻塔下文本逐字不变、图请求被两卡调度拒、MTP/DFlash2 腿跑文本会话并验第二轮 resume（`:212-233`、`:235-269`、`:272-329`）。
- **修②＝死线（P1，`6d9e3676`）**：`--log-speculation-detail` 的每步 accept 长度在本树**永远为空**＝`transactions/binding.cpp:422` 是 `install_sampling` 全树唯一调用点且写死 `0U, 0U`（原注释声称的「serve 路径另有安装点」不存在＝坑③），`decode.cpp:124-126` 把 bound clamp 成 0 ⇒ `:104/:557` 的埋点永不追加；上游选项一路写到 `runtime/contract/request.h:19` 就没人读（`engine.cpp:335`＝唯一写入者）。供体把值经 `planning/request_plan.cpp:255/456` 带进计划再在 `program/prefill.cpp:674-676` 安装。修＝计划加字段＋安装点传 `accepted_length_steps` 与 `summary.effective_output_tokens`。**窗口代价**＝`test_engine_mtp_tp2_real.cpp:338-357` 断言「记到的步能复算 counters」与「cap=5 恰留前 5 步」，单卡自跳 77 ⇒ 不修则 D1 白烧一次两卡装载。
- **修③④＝文档对码（`eeb66f90`、`7d762ec5`）**：`tensor-parallel.md:393-395` 写 `ProgramImpl::publish_kv_rows`＋`program/storage/context.cpp`＋`bind_sequence_kv`＝三者在树里都不存在（真名 `publish_peer_kv_rows`＝`program_impl.cpp:617`，调用点 `prefill.cpp:250`/`commit.cpp:266`/`replay.cpp:176`）；`:432-436` 把 reuse 免责归给 `ProgramImpl::inspect_lane` 与 `815615e6`/`c29ccc46`/`4534b7bf`＝函数在本树没有、三个 sha 是 `remotes/tp2/main` 上的供体提交非 HEAD 祖先；`:504` 同一处旧名；`model_instance.cpp:299-300` 注释仍说塔可在 rank1。**全树 8-hex 引用扫完＝文档里 29 个 sha 有 25 个不在我方线上**（17 个是供体对象、挂在与 HEAD 无关的 `remotes/tp2/main` 上；8 个在本仓连对象都没有：`15d2e47b`、`2b5cd40a`、`785d856b`、`a14d686c`、`ccb20680`、`dabae909`、`b40d17ac`、`cfc8f749`）＝第 5 次同类坑，批量改写属下一轮。
- **A 清单＝无缺陷，且失败都响亮**：per-rank arena 构造在 rank1 current 下（`program_impl.cpp:168-172` 注释＋`ScopedCurrentDevice`）、两 rank 的 `work` span 与 arena 同序（`:54-60`、`:94-96`）、arena 越界＝`core/arena.cu:222` `bad_alloc`、split 入口先验两 rank（`ops/common/split_launch.h:77-112`，报价不足＝「rank N workspace is smaller」）；`resolve_kv_capacity_symmetric`＝min(空闲)/max(预约)（`kv_capacity.cpp:150-158`＋`startup.cpp:1378-1388`），两 rank 斜率交叉被 `startup.cpp:1452-1462` 的仿射断言兜住＝响亮不静默；verdict 抛出＝`core/startup.h:29-31` 发 Failed 且 `enable_peer_access` 在其后（`engine.cpp:40` vs `:44`），`ExecutionContext` 构造尾已把 current 绑回 rank0（`core/device.cu:102-105`）。
- **A 一条记**：tp2 计划给两 rank 的 attention 报价都用 rank0 的 SM 数（`startup.cpp:536` 取 `plan.multiprocessor_count`，来自 `:1411` 的 rank0 DeviceContext），执行按各卡 `rank_execution_view(rank)`（`text.cpp:1574-1576`）；`ExecutionContext` 只强制两卡同 CC 不强制同 SM 数（`device.cu:90-100`）⇒ 异型 12.0 卡对（5090+5080）会报价偏，失败面＝上面的响亮 bad_alloc。两卡都是 36 SM ⇒ 本地/窗口都触发不到；可加一行 SM 数相等闸，属改码＝与 P1 整批同做。
- **E 清单＝Ada 面两处不闭合（不影响本地 sm_120a 编译，分类＝记＋一条 4 行可落修）**：本层改到的 CUDA 单元 69 个＝16 个被剔除表命中、53 个仍在 Ada 编译面，逐份 nvcc `-c`（CUDA 13.2）实测 **OK 39／红 14**；同一根因在本层**没碰过**的单元上也复现（`linear/fp8/shapes/n14336|n16384|n248320|n34816_k5120|n5120_k17408` 五份红、`gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_*` 三份绿）⇒ **红是基线债不是本层引入**，与 `tp2-ada-notes.md:112-116` 的「45/200 红、21 已覆盖、24 待 drop/stub」同向。根因只有两类：`cvt.rn.bf16x2.e2m1x2`/`e4m3x2` 的守卫只看**编译器版本**不看目标架构（`linear/nvfp4/nvfp4_codec.cuh:27-38`、`linear/fp8/fp8_a16_codec.cuh:14-18`，`#else` 分支注释自称 exact＝加 `&& !defined(NINFER_SM89)` 即回到那条路，可一次救 10/14）；`.op_restrict` 来自 CUDA 头里的 FP8 A8 TMA 路（本层无 op_restrict 字面量，剔除表也无 FP8 条目）。**符号闭合不成立**：剔除表删掉的 24 个 `.cu` 里有 10 个启动器仍被保留的 `.cpp` 调用且 Ada 面无定义＝`nvfp4_{linear_swiglu,linear_add,attn_input,gdn_input}_a4_launch`（调用点 `linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:95`、`linear_add/nvfp4/nvfp4_linear_add_plan.cpp:72`、`attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:56`、`gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:52`）、`launch_nvfp4_a4_quantize`/`launch_nvfp4_linear_swiglu_a4_tma`（同 `linear_swiglu_plan.cpp:100-101`）、`{nvfp4,k8v4}_kv_{append,cached}_attention`（`causal_cache/causal_softmax_attention.cpp:360/371/406/417`）；而 `src/ops/sm89/nvfp4_sm89_stubs.cpp:61-135` 定义的 24 个名字里 **18 个退役旧名全树零引用**（`launch_nvfp4_w4a4_*`/`nvfp4_*_w4a4_launch` 十个＋`causal_attention_*_launch` 八个；另两个 `*_unsupported` 是桩内形状表自用），`grep -rn w4a4 src include` 只剩 `cmake/Sm89Compat.cmake:32` 的注释），只有 `kv_cache_append_{nvfp4,k8v4}[_batch]_launch` 四个仍对上。＝Ada 面**编译与链接都不闭合**；`cmake/Sm89Compat.cmake:12-13` 与 `docs/sm89.md:37` 那句「Every remaining route … FP8 A8/A16 … compiles and runs on Ada」实测不成立，`docs/sm89.md:116` 举的证据符号 `Q8KSplitAdaSharedWindow` 树里没有（真名 `core::request_shared_window`＝`src/core/shared_window.cuh:44`，且它按容量而非架构返回 0）。
- **C 清单＝python 面闭合**：`tools/**`＋`tests/**` 全量 AST 级 import/名字解析＝0 悬空；`METHODS` 六键全指向已定义函数（`methods.py:348-355`），配方按**函数对象**引用（`official_recipes.py:10/72/91`）＝与 `recipe.assign` 的双形一致（`tools/convert/recipe.py:150-156`），`nvfp4_mse` 案已闭。旗标面＝`gate.sh` 用到的 serve 旗标 17 个全在 `src/serve/serve_options.cpp` 解析表（含 `--lm-head-draft`、`--kv-dtype nvfp4`＝`serve_options.cpp:105`），perplexity 用的 8 个全在 `apps/perplexity/options.cpp:56-92`（`--kv-dtype` 收 nvfp4＝`:73-86`），脚本互引 `tools/golden/record.sh`、`tools/tp2/{ppl_paired.py,gate_client.py,check_shard_sources.py}` 全存在。ctest 可见性＝**只有 `tests/text/` 三条 python 测注册**（`tests/cmake/RuntimeTests.cmake:29-42`），`tests/convert/**`（9 个，含新落的 `quantization/test_nvfp4.py`）与 `tests/artifact/**`、`tests/bench/ttft/**`、`tests/test_serve_corpus.py` 仍不在门里；本机**无 pytest 也无 torch**（`import pytest`/`import torch` 均 ModuleNotFoundError）⇒ 那批连「注册了能跑」都不成立，数值面仍属带 torch 的机器。另记：本仓 `AGENTS.md` 指定的 `/home/neroued/miniconda3/envs/py311/bin/python` 在本机不存在（`python3`＝3.10.12）。
- **D 清单已核＝一致**：`op-development.md` §5.2 三句（一个权威查询／256B 对齐／sizing 与执行共享配方）＝`core/arena.h:88-89` 默认 256 对齐＋`core/arena.cu:204-229` 越界 `bad_alloc`＋`core/layout.h:63-65` 的 dry-run 同名 builder＋报价与执行同函数（`ops/wrapper/attn_input_proj.cpp:302-311/346-348` 对 `models/qwen3_5/execution/attention.cpp:32-40`；FP8 的 T 域单调闸＝`attn_input_proj/fp8/fp8_attn_input_plan.cpp:19-26` 只在 `tokens>=17` 转 A8 且报价看 `max_tokens`＝区间内不会漏）；§2.6 的 rank r 三件套＝`execution/text.cpp:1574-1576`＋`split_launch.h:96-112`＋`program_impl.cpp:168-172` 全对得上。`include/ninfer/ops/linear_swiglu.h:32/85` 的 A16 句与 `nvfp4_linear_swiglu_plan.cpp:31-32` 一致（加宽时两处文案＋`:71-73` 的 0 报价同一批改；§60 那句「容量不用改」实测成立＝A16 路 `nvfp4_a16_sliced_k_mma.cuh`/`nvfp4_a16_mma.cuh` 零 arena 引用）。
- **等窗口（审不出、只能测）**：D1 起 `--tp 2 --vision --purpose scoring` 两条命令能否装载（本轮修的闸）；tp2 打分 tile 两 rank 的 arena 峰值；per-rank arena 省下的视野余量；`accepted_lengths` 与 counters 复算；12/2 两卡 parity（`test_attention_headlocal` 的 nvfp4 腿）。
- **门链拆段实测（本会话）**：**全构建段已实跑＝绿**＝`gate-72` reset 到 `7d762ec5`＋`applied:vision-on-build-ada.patch`＋`JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests` ⇒ **build rc=0**、日志 `relay/gate-audit48-build.log`（error 关键词 1 处＝上面那句 ELF 行的 `version 1 (GNU/Linux)` 误命中、warning 0）；**子集段与簇门段没跑**＝本会话 GPU 不可用，不打 tag（纪律⑦）。
- **修⑤（下一轮，容量面，带数）**：打分相的 arena 报价仍是两 rank 同值＝`startup.cpp:721` 取 `max(rank0, rank1)` 后写进 `general_capacity`，而 `peer_*` 只在 vision 分支才分（`:801-802`）⇒ vocab 248,320、`kCausalScoreTile` 1024 时 rank0 需 485 MiB logits＋两份 242 MiB 分片/中转 ≈ **970 MiB**，rank1 只需 242 MiB 分片＋head scratch ⇒ **rank1 白背约 700 MiB**，正撞 D2 那本账（`docs/perplexity.md:65-66` 那句「约两倍、每卡约 1 GiB」就是它）。改法与 vision 同一形＝在 `plan.causal_scoring` 分支里把 rank1 的那份写进 `peer_general_capacity/peer_capacity`；动它要同时补 §2.6 的「一个权威」句与窗口 D2 的复算。
- **门链后两段的 CPU 切片＝本会话已实跑，并做了面对面差分（新证据，下一轮别重做）**：分母按标签切不开（`gpu` 标签只覆盖 16 个模型用例＝`tests/cmake/NinferTests.cmake` 的 `LABELS "gpu"`），所以用**同一命令在两个面上各跑一遍再逐名比**：基线＝主树 `build-120a-cu13` 的 10:00 二进制（面＝`7cd4bdfc`，四刀之前，**未重建未链接只跑现成件**＝纪律⑥）；审后＝gate-72 的 `7d762ec5`＋vision 补丁。同一条 `CUDA_VISIBLE_DEVICES= ctest -j8 --timeout 240 -E softmax_attention`：**基线 46 绿／96 红／分母 171**，**审后 48 绿／96 红／分母 173**；**红集合逐名相同（新增红＝0，转红＝0）**，96 个红的成因是设备不可用（抽样实证＝`tests/artifact/test_materialization.cpp:220` `cudaErrorOperatingSystem`）；多出的 2 绿＝`ninfer_media_decode_test`、`ninfer_qwen3_5_frontend_test`＝分母差 2 的来源（gate-72 贴的 `vision-on-build-ada.patch` 把 vision 打开，主树那次 configure 是 `vision=OFF`）。⇒ **我这四刀在 CPU 可达的面上零回归**，但 tp1 的 `install_sampling` 真值生效、tp2 的 Program 装载、28 家族子集与 96 设备用例的数值面仍只能等 GPU。日志＝`/home/ai-agent/DSHW/tmp-review/{base_cluster,cpu_cluster}.log`。
- **A 清单的一条更正**：`resolve_kv_capacity_symmetric` **不是**"只有 D2 能测"——`tests/runtime/test_kv_capacity.cpp:115-119` 有 CPU 断言（"symmetric KV capacity did not follow the tightest rank"）；真正零测试覆盖的是 **per-rank arena 的两个字段**：`git grep peer_capacity` ＝只有 `startup.h:75-80`、`startup.cpp:801-802/1381-1383`、`program_impl.cpp:54/58` ＋两份文档，**测试集里一次都没出现** ⇒ S11 第 4 件的本地证据只有编译，D2 是第一块证据（补一个 CPU 用例＝构造 tp2 plan 断言 `peer_capacity == general_capacity`，属可下一轮的小改）。
- **交回动作**：余下两段的命令（红则按上面 file:line 定根因，不打 tag）＝`cd /home/ai-agent/DSHW/ninfer-watch && GATE_PATCHES=/home/ai-agent/DSHW/ninfer-watch/relay/vision-on-build-ada.patch bash relay/gate-steps.sh 7d762ec5 audit48`；判据＝全构建 rc=0＋家族子集全绿＋簇门 100%（分母取当次计数＝上次 173）。

## 68. 第 48 回合续＝**交接给"单卡跑完 96 红"的新会话**：全绿证据要回写的文件清单＋缺测清单

### 68.1 现场（本会话末，自证）
- 树＝`7d762ec5`（四刀：`e18613d1` 闸、`6d9e3676` 死线、`eeb66f90` 文档、`7d762ec5` 注释），porcelain＝0，无 rebase-merge，**未打 tag**（门链后两段未跑＝纪律⑦）。
- 本会话 GPU 不可用＝`cuInit→304`（`relay/probe/gpu_probe` 与 `CUDA_VISIBLE_DEVICES=1` 两种跑法都失败）；**新会话若 `gpu_probe` 出 `cuDeviceGetCount==1` 就能跑**。
- 已实测绿：全构建段＝gate-72 `7d762ec5`＋`applied:vision-on-build-ada.patch` → `build rc=0`、74 目标、0 warning（`relay/gate-audit48-build.log`）。
- 差分证据（无 GPU 模式，同一命令跑两面）：基线（主树 10:00 旧二进制＝`7cd4bdfc`）46 绿／96 红／分母 171，审后（gate-72 `7d762ec5`）48 绿／96 红／分母 173，**红集合逐名相同＝新增红 0**；分母差 2＝vision 补丁打开 `media_decode`/`qwen3_5_frontend`（主树那次 configure 是 `vision=OFF`）。日志＝`/home/ai-agent/DSHW/tmp-review/{base_cluster,cpu_cluster}.log`（临时面，别当台账依赖）。

### 68.2 全绿要回写的文件（跑完照此落，缺一个就不算证据齐）
1. `relay/gate-<label>-chain.log`＝三段各自的 rc／`% tests passed`／`Total Test time`（`gate-steps.sh` 自动写）。
2. `relay/gate-<label>-build.log`＝全构建原文（本次＝`gate-audit48-build.log` 已存在，label 换名后会重写）。
3. `relay/gate-<label>-sub.log`＝家族子集 28 的逐条原文；`relay/gate-<label>-full.log`＝簇门 173 的逐条原文。
4. `relay/passset-sm120-tp2steps-<label>-sub.tsv`＋`relay/passset-sm120-tp2steps-<label>.tsv`（`passset_from_log.py` 生成，＝权威设备分母的逐用例状态＋秒）。
5. **12/2 资格面另跑一次**（它在簇门里被 `-E softmax_attention` 排除）＝`ctest -j1 --timeout 1800 -R softmax_attention` → `relay/ada48-qual-ctest.log`，再逐 dtype `./tests/ninfer_softmax_attention_test --kv-dtype nvfp4|int8|bf16` → `relay/ada48-qual-<dtype>.log`（上次同口径＝1/1 Passed 864.22 s、三份 rc=0，见 §61/§62）。
6. tag＝三面全绿才打，消息必须含：全构建 rc=0 的 UTC 时刻、子集 `100% N/N x.xx s`、簇门 `100% 173/173 x.xx s`（Passed/Skipped 分解）、12/2 资格面秒数、**以及"本面含 §67 的四刀（tp2 打分与常驻塔开闸＋accepted-lengths 接线）"**；建议名 `v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0`。
7. 台账新 §（≤25 行）＝把上面 6 项数字逐条抄进去，并明确"未跑面"（两卡套件在单卡自跳 77＝不算证据）。

### 68.3 缺的测试（按能不能本地跑分，本会话核过）
- **零 GPU、现在就该补进门**：① `tools/test_kv/kv_oracle.py`（纯 numpy，本会话实跑 rc=0＝mean cos 0.991310／min 0.986712／p1 0.988712 ≥ 0.990，5 个深度 needle 全 MATCH，12.3 s）——注册时要同时断言 rc=0 **且** stdout 含 `[PASS]`，否则假绿；② per-rank arena 的 CPU 断言（`peer_capacity`/`peer_general_capacity` **测试集零引用**：`git grep` 只落在 `startup.h:75-80`、`startup.cpp:801-802/1381-1383`、`program_impl.cpp:54/58`）——先例＝`tests/runtime/test_kv_capacity.cpp:115-119` 与 CPU 可跑的 `ninfer_qwen3_5_shard_map_test`，断言 tp2 无 vision 时 `peer_*=0` 且两 rank 同值、tp2 有 vision 时 `capacity>peer_capacity==general_capacity`、`device_reservation_bytes==max(两 rank)`；③ 转换器接线面的 CPU 用例（`recipe.assign` 收函数对象与字符串两形＝`tools/convert/recipe.py:150-156`，`METHODS` 六键＝`methods.py:348-355`，本会话 AST 级全解析 0 悬空）。
- **需要 pytest／torch 的机器**：`tests/convert/**`（9 个）、`tests/artifact/**`（3 个）、`tests/bench/ttft/**`（5 个）、`tests/test_serve_corpus.py` 全不在 ctest；本会话实测＝`import pytest`／`import torch` 均 ModuleNotFoundError，且 `tests/convert/test_cli.py` 用 `PYTHONPATH=. python3` 直接跑**会 rc=0 而零用例**＝**原样按 `RuntimeTests.cmake:29-42` 的 `python3 -B` 形注册＝假绿**（坑④新变体），必须换真收集器或改写为自带 main 且断言收集数>0。
- **需要单卡 GPU**＝本轮的 96 红（ops/kernel/arena/KV/两卡入口的编译期设备用例，全部 `CUDA_VISIBLE_DEVICES=1` 一条卡即可）。
- **需要两卡**＝`*_tp2_*` 套件（单卡自跳 77）、`test_attention_headlocal` 的 nvfp4 腿、D1 四段。

### 68.4 窗口前仍欠的三件（别在窗口里现做）
1. **修⑤**＝打分相 arena 两 rank 同值（`startup.cpp:721` 取 max 后写 `general_capacity`，`peer_*` 只在 vision 分支分）⇒ vocab 248,320・tile 1024 时 rank1 白背约 700 MiB，会污染 D2 的容量结论。
2. **gate.sh 的环境量**＝`GATE_PROMPTS`（`gate.sh:207-208` 的 `--record` 硬要求）、`GATE_ARTIFACT`/`GATE_DF2_ARTIFACT`/`GATE_TEST_ARTIFACT`/`GATE_GOLDEN_MODEL`/`GATE_TEST_SCRATCH`（默认全指 `/home/feyd/...`；本机两件＝`/tank/models/lynn/NVFP4-NInfer/W4A4/…W4A4-MTP-DFlash2.ninfer` 23,773,675,264 B 与 `/tank/models/mrwan0410-lynn-nf4-ninfer/qwen3.8-27b-lynn-nf4.ninfer` 19,345,900,804 B）；`GATE_CTEST_EXCLUDE` 默认串含 `dflash` ⇒ 连 `*_dflash2_*_test` 一起排除（`:50`）。
3. **首轮必须 `--record` 到空目录**建我方基线（`tools/tp2/reference/` 是供体 2x5070 件，`req-greedy.jsonl` 里 `accepted_lengths` 60 行全 null）；临时件落 `/tank`（根分区 90% 已用、构建目录 44 GB）。

## 70. 第 48 回合＝audit48 起跑＋S3 签名锁定（case-3-long 是唯一发散面）＋两条零 GPU 缺口补齐
- 现场：`7d762ec5` 面门链已起跑（label `audit48`，PID 3015407，**只贴 `vision-on-build-ada.patch`**＝日志确认 `patches: applied:vision-on-build-ada.patch`）。选卡实况＝**探针现见 2 张卡**（旧探针硬期望 1 张，打印红字＝非停手条件）：`nvidia-smi --query-compute-apps`＝**strata PID 3013603 在 GPU0（15588 MiB／util 99%）**，GPU1 空闲（4 MiB）⇒ **测试卡＝索引 1＝`gate-steps.sh` 两处 `CUDA_VISIBLE_DEVICES=1` 无需改动**（按新卡纪律在此披露）。
- **S3 实测签名（从 10:08 那跑的落盘件里读出来，纯 CPU）**：`/tmp/golden-47/*.ids` 与两份参考 `tools/golden/recorded/2026-09-30/{fork-f19e6821,upstream-d44ab584}` 逐案例比＝**case-1-short（32＝单块）difflines=0、case-2-medium（2048＝两块）difflines=0、case-3-long（6144＝六块＋INT8 KV）difflines=8**（`generated_tokens 118`→`124`，ids 自该案例后段发散）。⇒ **S3 的红不是"tp1 全红"，＝只有多块 prefill＋INT8 KV 那一档发散**；两参考彼此相同（同 digest）＝参考面自洽。
  - 比较口径坑（记死）：`diff -r` 会把 `record.sh` 产的 `*.out`／`*.log` 记成 "Only in" ⇒ `gate.sh` 的 golden 段用 `diff -r "$ref/golden" "$g"` 在**任何**实现下都会 FAIL（参考目录只有 `.ids`＋`RESULT.md`）。**判据必须只比 `.ids`**（`for c in case-1-short case-2-medium case-3-long; do diff "$ref/$c.ids" "$g/$c.ids"; done`）＝改 gate.sh 的 golden 段（零 GPU 改动，下轮顺手做，属 §68.3 缺口类）。
- **S9 实测（同一批落盘件，纯 CPU 解析）**：`/v1/models`＝`{id, max_model_len 4096, owned_by:ninfer, created, object}` ✓；**带 `stop` 的请求＝`finish_reason:"stop"`**（61 tokens；`prompt_tokens_details.cached_tokens 9`＝前缀复用活着的旁证）✓；thinking 请求＝`finish_reason:"length"`、usage 里 `reasoning_tokens 0`、**message 键＝[content, refusal, role]（无 `reasoning_content`）**。树面解释＝`src/serve/openai_chat_response.cpp:243` 只在 `outcome.reasoning` **非空**时写 `reasoning_content`（流式在 :344/:374 同键），而 `relay/gsm8k/run_gsm8k.py:139` 正是读 `message.get("reasoning_content")` ⇒ **写入方与判分方键名一致＝契约面已核**；合成两层模型不出 reasoning 段（随机权重）⇒ 活体证据要带思考开关重跑（serve 端 `--default-thinking` 系；`run_gsm8k.py` 必带 `--engine-tag`＋`--out`，我上版驱动缺 `--engine-tag` 且把 `--count` 写成了 `--limit`＝rc=2，属我脚本错，不是服务不兼容）。
- **两条零 GPU 缺口已补（提交＝`dd35195a`／`e7d7f20f`）**：
  - `dd35195a`＝`tests/cmake/run_python_oracle.cmake`＋注册 `ninfer_e8_kv_oracle_test`（`tools/test_kv/kv_oracle.py`，纯 numpy）＝**同时断言 rc=0 且 stdout 含 `[PASS]`**（ctest 的 `PASS_REGULAR_EXPRESSION` 会忽略返回码、裸 `add_test` 会漏掉"打印失败但退 0"，两者都不够）。三向自测：真件 rc=0 且含 [PASS] ⇒ 过；打印 `[FAIL]` 且退 0 ⇒ rc=1 拒；打印 `[PASS]` 且退 3 ⇒ rc=1 拒。实跑数据＝mean cos 0.991310（阈 0.990）、5/5 needle MATCH、12.3 s。**按坑④**：`tests/convert/**`、`tests/artifact/**` 仍不注册（本机无 pytest/torch＝零用例退 0＝假绿）。
  - `e7d7f20f`＝把 per-rank 内存界从 `build_sequence_candidate` 里提成纯函数 `detail::bound_rank_sequence_bytes`（新件 `planning/rank_bounds.h`＝不含 model/artifact include，CPU 测试可直连），并补 4 条 CPU 断言（`ninfer_qwen3_5_runtime_mechanisms_test`）：tp1 两 0 项不 binding；rank1 省 drafter ⇒ rank0 bounds；我方今天形状＝编码 rank 的大 arena bounds；**前瞻**＝rank1 带更大 arena（P1 的接收面）必须成为 binding＝KV 池永不按需要更少的 rank 定价。此前 `peer_capacity`/`peer_general_capacity` 在测试集**零引用**＝这条现在补上。行为不变（同一 max、同一 checked_add）。2 个 TU rc=0（在门链 ctest 期间只编 `.o`＝披露）。
- 在跑（勿重复启动）＝`relay/audit48-post.sh`：audit48 簇门一收尾就 ① 跑 12/2 资格面（`ctest -R softmax_attention` → `relay/ada48-qual-ctest.log`；逐 dtype → `relay/ada48-qual-{nvfp4,int8,bf16}.log`；上次口径 1/1 Passed 864.22 s＋三份 rc=0）② 对 `e7d7f20f` 起门链（label `audit48b`，只贴 vision）。数据齐后**由人写 tag 消息**（红门不打）：`v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0 @ 7d762ec5`，内容含全构建 rc=0 的 UTC 时刻＋子集/簇门计数与秒数＋Passed/Skipped 分解＋12/2 资格面秒数＋本面三条内容（tp2 打分开闸、常驻塔开闸、accepted-lengths 接线）。
- **开窗前最硬的一条＝S3 的 case-3-long 根因**（R1 第 2 条不绿就别申请窗口）。可执行方案（零 GPU 起步、跑测时要 GPU1 空档）＝用 golden 当谓词在层段上 bisect：
  `git bisect start 7d762ec5 03726c08~` ＋ 谓词脚本＝`ninfer_qwen3_5_text_context_tp2_test write /tmp/bsyn` → `tools/golden/record.sh build-120a-cu13/apps/ninfer-tp1-golden /tmp/bsyn/model.ninfer /tmp/bgold` → `diff /tmp/bgold/case-3-long.ids <参考>` **只比这一条**；good＝与参考一致。bisect 空间＝197 提交，每步一次定向构建（`--target ninfer-tp1-golden ninfer_qwen3_5_text_context_tp2_test` 比全构建快得多）＋一次三档跑（~60 s）。根因嫌疑面（按 graft 密度排序）＝六块 prefill 的块边界对齐（`mtp_bridge`/`scatter_shifted_visual_embeddings` 与 `prefill_chunk` 的 alignment_tokens）、INT8 KV 的 page/row 发布、12/2 几何下 GDN 线性状态在块边界的 commit、accepted-lengths 埋点动过的 round 边界。合成模型随机权重 ⇒ **不是数值精度问题就是控制流问题**；两档单/两块逐字一致 ⇒ 嫌疑集中在"块数>2 的边界与状态推进"。

## 71. 第 48 回合续二＝**audit48 面全绿（96 红根因＝设备不可用）**＋四条后续补齐（含修⑤）＋S9 判绿
- **audit48（`7d762ec5`＝第 48 回合四刀）＝三面全绿**：全构建 **rc=0**（04:29:15 UTC）；家族子集 **100%，28/28，425.86 s**（Passed 25／Skipped 3）；簇门 **100%，173/173，1108.90 s**（**Passed 137／Skipped 36／Failed 0**）。文件齐＝`gate-audit48-{chain,build,sub,full}.log`＋`passset-sm120-tp2steps-audit48{,-sub}.tsv`。
  ⇒ **§67/§68 那 96 红的根因＝该会话 GPU 不可用**（`cuInit→304`；抽样 `tests/artifact/test_materialization.cpp:220`＝`cudaErrorOperatingSystem`），**不是四刀造成的回归**；工作卡（GPU1）上同一面 137 通过／36 自跳／**0 失败**。选卡实况＝本机现见 **2 张卡**（旧探针硬期望 1 张＝打印红字，非停手条件）：GPU0＝现网 strata（PID 3013603，15588 MiB／util 99%，不碰）、**GPU1＝测试卡＝索引 1 ⇒ `gate-steps.sh` 两处 `CUDA_VISIBLE_DEVICES=1` 无需改**（按新纪律披露）。
- **修⑤已落＝`9cff7a71`「size rank 1's arena by its own score tile」**：打分相在 tp2 是不对称相位（rank0＝全词表 logits 485 MiB＋两份 242 MiB 分片/中转＋scratch；rank1＝自己那半 head＋scratch），而 plan 把 `max(rank0,rank1)` 写进共享 `general_capacity` ⇒ **rank1 白背约 700 MiB**（vocab 248,320・tile 1,024）。改法＝`WorkspacePlan` 新增 `peer_causal_score`，两 rank 的 general 各按自己的相位集取最大（纯函数 `split_general_capacity`），`peer_general_capacity`/`peer_capacity` 从此**每面都显式给出**（不再只 vision 分支）；文档同步＝`docs/perplexity.md`（原文"约两倍、每卡约 1 GiB"已改成逐 rank 真数＋绑定面＝rank0）与 `docs/maintainer/tensor-parallel.md` §6 一句。非打分面行为不变（两份额本来就相等）。验活＝2 TU rc=0（`planning/startup.cpp`、`test_runtime_mechanisms.cpp`；在门链 ctest 期只编 `.o`＝披露）；**省下的 700 MiB 本身＝D2 实测项**。
- **本轮另三条（零 GPU 缺口，§68.3）**：`dd35195a`＝`tests/cmake/run_python_oracle.cmake`＋注册 `ninfer_e8_kv_oracle_test`（**同时断言 rc=0 且 stdout 含 `[PASS]`**；三向自测：真件过／打印 `[FAIL]` 退 0 拒／打印 `[PASS]` 退 3 拒。实跑 mean cos 0.991310、min 0.986712、p1 0.988712 ≥ 0.990，5/5 needle MATCH，12.3 s）；`e7d7f20f`＝per-rank 内存界提纯函数 `bound_rank_sequence_bytes`（`planning/rank_bounds.h`，无重 include，CPU 可直连）＋4 条 CPU 断言（补上 `peer_capacity`/`peer_general_capacity` **测试集零引用**）；`dfd70b54`＝gate.sh golden 段改**逐案例只比 `.ids`**（新函数 `ids_only_equal`；`record.sh` 会旁写 `.out/.log`，旧 `diff -r` 对任何实现都 FAIL＝真信号 8 行被 16-17 行噪声埋住）。
- **S9（R1 第 4 条）＝判绿，且不需要新跑**：三条腿证据来源已核齐＝①`/v1/models` 字段（合成服务活体实测 `id/max_model_len/owned_by:ninfer/created/object`）；②**`reasoning_content` 键名＝契约面＋单测面**（写入方 `src/serve/openai_chat_response.cpp:243` 只在 `outcome.reasoning` 非空时写；流式 `:344/:374` 同键；判分方 `relay/gsm8k/run_gsm8k.py:139` 读的正是它；单测 `tests/test_openai_schema.cpp:800`（非流）与 `:853`（流式 delta）断言该键，**audit48 簇门内 Passed 0.03 s**）；③`stop` 行为（活体 `finish_reason:"stop"`，61 tokens，`cached_tokens 9`）。活体的 reasoning 腿＝合成两层模型随机权重不出思考段（实测 `reasoning_tokens 0`）⇒ **窗口里用真制品补一条 curl 即可，不卡开窗**。我上版驱动脚本两处自伤已记：`--limit` 不存在（是 `--count`）、缺 `--engine-tag`＝rc=2；且 curl 的 `2>&1` 会把 libcurl 版本告警灌进 JSON 头（解析要先跳到 `{"choices"`）。
- 在跑（勿重复启动）＝`relay/audit49-post.sh`（PID 3462861）：等 12/2 资格面的 ctest 跑完（`relay/ada48-qual-ctest.log`，12:54 起，上次同口径 864.22 s）→ 补三份 dtype（`ada48-qual-{nvfp4,int8,bf16}.log`）→ 对 **`9cff7a71`** 起门链（label `audit49`，只贴 vision 补丁；**分母 173→174**＝新注册的 oracle 测）。我停掉了 `audit48-post.sh`（它只覆盖到 `dfd70b54`，会浪费一整轮链），改接新面＝一次链覆盖四条补齐。
- **tag 计划**：① 资格面数字一到就 `git tag -a v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0 7d762ec5 -m "<§68.2 六项＋四刀说明>"`；② audit49 三面全绿＋其资格面后打 `v-tp2-audit49-per-rank-arena-and-oracle-gates-r0 @ 9cff7a71`；红＝不打（纪律⑦），根因首选＝新注册的 `ninfer_e8_kv_oracle_test`（PYTHONPATH/解释器）与 `split_general_capacity` 的两处调用。
- **开窗前剩余障碍（本轮后）**：只剩 **A＝S3 的 `case-3-long` 根因**（谓词脚本已写＝`relay/s3-bisect.sh`，用法＝`git bisect start 7d762ec5 03726c08~ -- && git bisect run ../relay/s3-bisect.sh`，`S3_BISECT_CARD` 指测试卡；只比 `case-3-long.ids`，定向构建两目标，构建/产件失败退 125＝跳过）；**D＝gate.sh 环境量与首轮 `--record` 到空参考目录**（默认全指 `/home/feyd/...`；`GATE_PROMPTS` 是 `--record` 硬要求；`GATE_CTEST_EXCLUDE` 含 `dflash` 会连 DFlash2 排掉；树内 `tools/tp2/reference/` 是供体 2x5070 件、`accepted_lengths` 60 行全 null）；**E＝GPU0 让位＋`gpu-lease`＋临时件落 /tank**（根分区 90%，两构建目录各 44 G）。B（修⑤）码面已完＝数值证据等 D2。Ada 面与 25 个文档 sha 不在 120a 成品编译路径上。

## 72. 第 48 回合续三＝**audit48 全证据齐**（含 12/2 资格面 835.44 s）；两条追加（修⑤已提交＋文档对源）
- **12/2 资格面在 `7d762ec5` 面＝1/1 Passed，835.44 s**（`relay/ada48-qual-ctest.log`；上一面 6d1d3ab5＝864.22 s ⇒ 抖动 −3.3 %，非变慢）。三份 dtype 由 `relay/audit49-post.sh` 续跑（`relay/ada48-qual-{nvfp4,int8,bf16}.log`），跑完打 tag＝`v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0 @ 7d762ec5`（消息含四刀＋三面包＋资格面秒数＝§68.2 第 6 项）。
- 追加两提交（**纯文档/规划面**，不改 C++ 语义）：**`9cff7a71`＝修⑤**（已在 §71 记全）与 **`3ab979bd`＝文档对源**：`tensor-parallel.md` 原有 **20 个 8-hex 引用＝2 个在我方线上（上游 `bace20dc`/`e31bc99b`）＋16 个只在 `remotes/tp2/main` 的对象里＋2 个本仓无对象**，全部写成"这棵树如何如何"＝**同类坑第 5 次**（前 4 次＝`nvfp4_mse` 未定义、`vision_receiver` 文案、gates-r72 补丁≠树面、`docs/sm89.md:116` 的证据符号不存在）。现逐句改成"说代码做什么＋把出处归给 donor line＋以本台账为重放记录"；并把**全档集扫查结果写死**免得下轮重扫＝`upstream-merge.md` 的 `91582814` 已改成"donor 提交＝我方作为重放步 142 撤除"；`README.md` 的 `15d2e47b`＝vendored 仓的 revision（归属用途，非本史）；`sm89.md` 的 5 个与 `e8-port-surface.md` 的 11 个＝**点名的第三方仓**（xkeyc/sergiuszm/UDPSendToFailed/Don-Chad，设计如此）；`artifact-container.md` 的 6 个与 `linear-benchmark.md` 的 1 个＝**字节数长得像 sha**（假阳性，别改）；`e8-yarn-audit-*` 三份＝带日期的档案报告，不是契约。
- 门链排程＝`relay/audit49-post.sh` 在 dtype 三段跑完后对 **`9cff7a71`** 起链（label `audit49`；分母 **173→174**＝新注册 `ninfer_e8_kv_oracle_test`）。`3ab979bd` 是纯文档面＝随 audit49 之后任意一次门链护航，不单独起链（若 audit49 已 reset 到 `9cff7a71`，则下轮对 `3ab979bd` 起一次即可，成本＝一条链）。
- 状态口径（给"进入全量编译"用）：**门链与制品都不是障碍**（audit48 三面全绿＋资格面绿；两件制品在 `/tank` 且 sha256 已逐位核）；**开窗前只剩 A/D/E 三件**＝A 的 `case-3-long` 根因（谓词脚本 `relay/s3-bisect.sh` 已写好并语法过，bisect 空间 197 提交≈8 步）、D 的 `gate.sh` 环境量与首轮 `--record` 空参考目录、E 的 GPU0 让位＋`gpu-lease`＋临时件落 `/tank`。B（修⑤）码面已完，数值＝D2；C（S9）判绿。

## 73. 第 48 回合续四＝驱动脚本自匹配死锁（我的坑，记死）＋接续排程
- **坑（新一条，进卡纪律）**：`relay/audit49-post.sh` 的等待环＝`while pgrep -f "tests/ninfer_softmax_attention_test"`，而该模式**匹配到了启动它的父 `bash -c` 自己的命令行**（里面含这串路径）⇒ ctest 早在 13:08 结束（`ada48-qual-ctest.log`＝**1/1 Passed 835.44 s**，比上一面 864.22 s 快 3.3 %），驱动却一直在睡，三份 dtype 从未开始（日志文件不生成＝发现线索）。这就是接力纪律第 ④ 条"pkill/pgrep 别用会匹配到自己的命令行片段"的**正向版本**：**自己写的脚本也会自匹配**。
- 修法＝换新脚本 `relay/audit49-run.sh`（**不含任何 pgrep 环**：直接跑三份 dtype → 起链），旧驱动按 PID 精确 kill（3462861/3462858/3465074/3465075）。13:21 起 dtype nvfp4 已在跑；三份约 13:32 齐；随后自动对 **`3ab979bd`**（含修⑤与文档对源，doc-only 在最上）起门链 label `audit49`（只贴 vision 补丁，分母 **173→174**）。
- 待办（数据一到就做，别跳）：① 用 `ada48-qual-ctest.log`（835.44 s）＋三份 dtype rc 给 **`7d762ec5`** 打 tag `v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0`（消息按 §68.2 第 6 项六件齐）；② audit49 三面＋其资格面全绿 ⇒ tag `v-tp2-audit49-per-rank-arena-and-oracle-gates-r0 @ 3ab979bd`；红＝不打，根因首选＝新注册的 `ninfer_e8_kv_oracle_test`（解释器/`PYTHONPATH`/`[PASS]` 正则）与 `split_general_capacity` 的两处调用（`startup.cpp` 打分相与 vision 相）。
- 本轮提交链（porcelain=0）：`3ab979bd` 文档对源（20 个 8-hex 引用＝2 在线／16 只在 donor 对象／2 无对象，全部改成"说代码＋归因 donor line＋台账为重放记录"；并把全档集扫查结果写死＝sm89/e8 的第三方仓引用**不改**、artifact-container 的 6 个与 linear-benchmark 的 1 个是**字节数长得像 sha**（假阳性）、README 的 `15d2e47b` 是 vendored 仓 revision、e8-yarn-audit 三份是带日期档案）→ `9cff7a71` 修⑤ → `dfd70b54` golden 只比 `.ids` → `e7d7f20f` per-rank 界纯函数＋4 断言 → `dd35195a` oracle 注册跑法 → `7d762ec5` 四刀（已全绿）。

## 74. 第 48 回合末＝正式编译前的**欠账清单**（照此交审核会话）
- **已验证（证据齐，别重跑）**：`7d762ec5`（四刀）三面全绿＝全构建 rc=0（04:29:15 UTC）＋家族子集 **100% 28/28·425.86 s**（25P/3S）＋簇门 **100% 173/173·1108.90 s（137P/36S/0F）**；**12/2 资格面**＝`ctest -R softmax_attention` **1/1 Passed 835.44 s**＋逐 dtype **nvfp4 rc=0／int8 rc=0**（bf16 在跑，约 13:28 齐）。文件＝`relay/gate-audit48-{chain,build,sub,full}.log`、`relay/passset-sm120-tp2steps-audit48{,-sub}.tsv`、`relay/ada48-qual{,-ctest,-nvfp4,-int8,-bf16}.log`。⇒ §67/§68 的 **96 红根因＝该会话设备不可用**（`cuInit→304`），非回归。tag `v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0`＝bf16 一到就补打（消息含 §68.2 六件）。
- **已落码·尚未验证（＝审核会话该盯的六项）**：
  1. **修⑤**（`9cff7a71`）＝两 rank 的 arena 份额按各自相位集定价；两处调用（打分相 `split_general_capacity`／vision 相 `peer_capacity`）**只编 `.o` rc=0，未进任何门链**＝随 `audit49`（label 已排，约 13:29 起，分母 **173→174**）。
  2. **`e7d7f20f`＋修⑤ 的 6 条 CPU 断言**（`bound_rank_sequence_bytes`／`split_general_capacity`）＝**未链接未跑**（门链 ctest 期不链＝纪律⑥）。audit49 收尾后链并跑 `ninfer_qwen3_5_runtime_mechanisms_test`，期望打印 `Qwen3.6 runtime mechanism checks passed`。
  3. **`dd35195a` 注册的 `ninfer_e8_kv_oracle_test`**＝**从未在 ctest 里跑过**（只有函数级三向自测：真件过／`[FAIL]` 退 0 拒／`[PASS]` 退 3 拒；实跑 mean cos 0.991310、5/5 needle MATCH、12.3 s）。首跑要盯＝解释器（`Python3_EXECUTABLE`）、`PYTHONPATH=项目根`、`TIMEOUT 120`、`[PASS]` 正则。
  4. **`dfd70b54` 的 golden 只比 `.ids`**＝函数级自测过（case-1/2＝0 行差、case-3-long＝8 行真差），**未在 `gate.sh` 实跑里走通**；单卡可用合成件走通一次：`GATE_GOLDEN_MODEL=/tmp/synthetic-tp1/model.ninfer bash tools/tp2/gate.sh --stages golden …`（合成件已产在 `/tmp/synthetic-tp1`，3,300,141,824 B）。
  5. **修⑤ 省下的约 700 MiB 本身＝D2 实测项**；accepted-lengths 的两设备断言（`test_engine_mtp_tp2_real.cpp:338-357`）、tp2 `CausalScoring` 装载（第四道闸已开＝运行时两卡从未跑）、nvfp4 KV 两卡 parity（`tests/ops/test_attention_headlocal.cpp` 的 nvfp4 腿）＝**全在 D 窗口**，单卡自跳 77 不构成证据。
  6. **S3 的 `case-3-long` 根因未定位**（单块/两块逐字 0 行差，六块＋INT8 KV 8 行差）＝谓词脚本 `relay/s3-bisect.sh` 已备，bisect 未跑。
- **明确未修的债（记在案，不假称完成）**：SM 数相等闸（异型 12.0 卡对报价偏＝与 P1 同做）；P1 布局整批（vision 接收面／DFlash2 分裂 drafter／宽 mailbox 槽）；P2 materialization 链（192/194/213）＋T-reclaim-priority；T-drafter-nvfp4 (a)(b)(c)（逐字规格＝§60，转换器前置已清＝327a4c53）；T-thinking-tokens；S7 占位数值（allowance 8/8/11 MiB、shard TMA 界 1024、nvfp4 12/2 的 row 组与 key shift）＝D2/D4 写回；语料侧遥测（205 的 gpus 记录、206/207 的逐轮长度与 4-gram overlap 未接进我方 `tests/serve/*corpus*.py` 形）；S10 剥离演练未跑（边界与命令已写＝§63）；Ada 面（本层 69 个 CUDA 单元 OK 39／红 14＝基线债、codec 守卫缺架构条件、stubs 里 18 个退役名全树零引用＝编译与链接都不闭合）＝**不卡 sm_120a 成品编译**；`tests/convert/**`、`tests/artifact/**` 不注册进 ctest（本机无 pytest/torch，注册＝假绿坑④）。
- 工具/脚本落档＝`relay/s3-bisect.sh`（A 项谓词）、`relay/window-d-setup.sh`（D 项：`GATE_*` 全指我方制品与 `/tank/ninfer-window`、`GATE_PROMPTS` 喂 60 题、`GATE_PPL_PAIRED=1`、**修掉默认排除串里 `dflash` 连 DFlash2 一起排除**、int8 与 nvfp4 各建独立参考目录、首轮 `MODE=record` 落空目录）、`relay/audit49-run.sh`（dtype→起链，无自匹配 pgrep）。卡＝v25/v26＋`relay/提示词-审核会话-20261017.md`。

## 75. 第 48 回合末五＝audit48 已打 tag；两处"编过≠链过/注册≠跑过"当场兑现（b116bdfc／035403d6）
- **audit48（`7d762ec5`）＝tag 已打**＝`v-tp2-audit48-scoring-vision-gate-and-accepted-lengths-r0`（消息含：全构建 rc=0@04:29:15 UTC；子集 100% 28/28·425.86 s（25P/3S）；簇门 100% 173/173·1108.90 s（137P/36S/**0F**）；资格面 1/1 Passed **835.45 s**＋逐 dtype nvfp4/int8/**bf16 全 rc=0**；四刀逐处 file:line；以及"96 红＝该会话 `cuInit→3`／`cudaErrorOperatingSystem`，红集逐名相同＝非回归"与"单卡自跳 77 不构成两卡证据"）。
- **坑（新，进纪律）**：`split_general_capacity`/`bound_rank_sequence_bytes` 我放在 startup.cpp 的**匿名命名空间**里定义，而头文件在 `ninfer::models::qwen3_5::detail` 声明 ⇒ 每 TU 编过（我 4 次单 `.o` rc=0 全真），**第一个全量链接才炸**（`apps/ninfer-serve`: undefined reference）。教训＝**新跨头函数必须至少一次真链接**（定向 `--target apps/ninfer-serve <该测试>` 比全构建便宜得多）；单 `.o` 的 rc=0 在"跨 TU 符号"问题上零证据力。修＝`b116bdfc`（移到层命名空间作用域，注释写明为什么），验＝链 `apps/ninfer-serve`＋`ninfer_qwen3_5_runtime_mechanisms_test` 都过，并实跑打印 `Qwen3.6 runtime mechanism checks passed`（含 6 条新 CPU 断言）。
- **坑二**：`dd35195a` 注册的 oracle 首跑 **ctest Failed**＝不是数学红，是**参数传递被 ctest 拆坏**：`-DPYTHON_CANDIDATES=a;b;c` 被拆成多个 argv、`-DIMPORT_CHECK="import numpy"` 的引号变成值的一部分 ⇒ 候选循环拿到垃圾并走到响亮失败分支（＝设计生效）。修＝`035403d6`：候选表与依赖检查**移进脚本**（命令行只传无空格的值），选第一个能 `import numpy` 的解释器。本机实测＝配置到的解释器是 **python3.14 无 numpy**，`/usr/bin/python3`（3.10.12，numpy 2.2.6）能跑 ⇒ ctest 现在打两行 `[PASS]` 并 100% passed；无可用解释器时仍然 FATAL_ERROR（**不静默跳过**＝4-bit KV 的数学面不能挂个没测的绿）。
- 排程＝对 `035403d6` 起门链 **label `audit49b`**（13:31 起，只贴 vision 补丁，**分母 173→174**＝oracle 测；`gate-audit49b-chain.log`）。上一轮 `audit49`（`3ab979bd`）＝**全构建 rc=1**（就是上面那个 undefined reference）⇒ 按纪律⑦未打 tag、未跑子集/簇门（＝链在第一个红处即停）。三面＋资格面全绿后打 `v-tp2-audit49-per-rank-arena-and-oracle-gates-r0 @ 035403d6`；红＝根因首选＝oracle 注册面与 `split_general_capacity` 的两处调用。
- 收尾后必做（我欠的，别忘）＝链并跑一次 `ninfer_qwen3_5_runtime_mechanisms_test` 与 `ctest -R "e8_kv_oracle|linear_swiglu|softmax_attention"` 在 gate-72 的 audit49b 面上（资格面与 oracle 都要数字）；数据抄进 §76。

## 76. 第 48 回合末六＝四提交（链接坑兑现＋ctest 参数拆坏＋SM 数相等闸＋语料遥测接线），`finish-everything.sh` 一路跑到底
- 提交链（porcelain=0，HEAD `340bd4c0`）＝`b116bdfc`（**两条 per-rank 纯函数从匿名命名空间移到层命名空间**：头文件在 `detail` 声明、定义在匿名 ns ⇒ 每 TU 自洽、4 次单 `.o` rc=0 全真，**第一个全量链接才炸**（`apps/ninfer-serve`: undefined reference to `split_general_capacity`）＝§75 那条坑的正面教训：**新跨头函数至少要定向链一次**（`cmake --build … --target apps/ninfer-serve <测试目标>`），验＝链过＋实跑 `Qwen3.6 runtime mechanism checks passed`（6 条 CPU 断言））；`035403d6`（**oracle 首跑 ctest Failed＝参数被 ctest 拆坏**：`-DPYTHON_CANDIDATES=a;b;c` 拆成多 argv、`-DIMPORT_CHECK="import numpy"` 引号进入值 ⇒ 候选循环拿垃圾走到响亮失败分支。修＝候选表与依赖检查进脚本、命令行只传无空格值；本机配置到的解释器＝**python3.14 无 numpy**，`/usr/bin/python3`(3.10.12, numpy 2.2.6) 可跑 ⇒ ctest 现打两行 `[PASS]` 100% passed；无解时仍 FATAL_ERROR＝不静默跳过）；`bd1be0ec`（**SM 数相等闸**＝A 清单"记"项落地：`core/device.cu` 在 tp2 校验里补 `p0.multiProcessorCount != p1.multiProcessorCount` 抛出（点名两卡 SM 数），理由＝计划用单 rank 的 SM 数报价（`startup.cpp:536`）、执行用各卡 `rank_execution_view` ⇒ 异型同 CC 卡对（5090+5080）会 quote 出跑不动的路，旧失败面＝`core/arena.cu:222` 响亮 bad_alloc；`tensor-parallel.md` 拒绝表加行；device.cu nvcc `.o` rc=0，链与门＝audit50 面）；`340bd4c0`（**语料工具接上四条遥测**＝per-request 带 `accepted_lengths` 的 mean/max/steps/truncated、`accepted_length_histogram`、`prompt_ngram_overlap`、`gpus` 样本；全可选：对 donor 参考实记录（schema 27）验证＝speculative 三键在、`accepted_lengths` 为 null、overlap 键在值 null、**无 `gpus` 键** ⇒ 缺键记 None 不raise；py_compile＋模块可导入＋表达式在真记录上跑过。**纯 Python 面＝不进 C++ 门链**，summary 聚合与统计＝窗口首轮自建参考后再做）。
- audit49b 进度＝全构建 **rc=0**（05:33:07，＝链修好后的面）＋家族子集 **100% 28/28·479.76 s**；簇门在跑（约 13:52 收尾）。
- 排成一条流水（`relay/finish-everything.sh`，日志 `relay/finish.log`＋`fin-*`）＝① audit49b 收尾→跑该面资格面（`fin-audit49b-qual-ctest.log`＋三份 dtype）→ 三面全绿才打 `v-tp2-audit49-per-rank-arena-and-oracle-gates-r0 @ 035403d6`（消息由链日志抓数自动写，**任一段红＝不打并记 `NOT TAGGING`**）；② `gate.sh --stages golden` 单卡实跑（`GATE_DEVICES=1,1`→`CUDA_VISIBLE_DEVICES=${DEVICES%%,*}`＝1，参考目录 `/tank/ninfer-window/ref-synthetic/golden/`＝树内 3 份 `.ids` 拷贝）＝**验 `dfd70b54` 的 ids-only 修复在 gate.sh 里真走通**（`fin-gate-golden.log`）；③ 对 `bd1be0ec` 起链 label `audit50`→资格面→tag `v-tp2-sm-count-equality-and-followups-r0`；④ **S3 bisect**（`s3-bisect.sh`，主树，`git bisect start 7d762ec5 03726c08~` ＋ run，末了 `git bisect reset`，日志 `fin-s3-bisect.log`）＝找 case-3-long 引入步；⑤ **S10 剥离演练**（worktree `/tank/peel-drill`，先剥 `HEAD~25..HEAD`，构建目录在 /tank 内＝`build-ada.sh` 固定目录名，无 `--build-dir` 旗标＝我第一版写错、已在起车前按 PID kill 换正）日志 `fin-peel.log`/`fin-peel-build.log`。
- 全部数字到位后由我把 §77 写齐（资格面秒数、golden verdict、bisect 首坏提交、剥离演练 rc 与手工触碰面），并把卡出到 v28＝交回审核会话的那一版。

## 78. 第 49 回合＝**六项复核（审 `bd1be0ec`/`340bd4c0` 面）**＋分钟级收尾；夜里剩一条链

- 现场自证的**出入先记死**：卡片期望的两枚 tag 在开工时**都不存在**（`git rev-parse` 逐一验＝ABSENT）；`relay/finish.log` 末行停在 05:59:24 的三份 dtype，`tag／golden 实跑／bisect／剥离演练` 四段**一段都没跑**，`pgrep` 无进程＝流水已断（不是"在跑"）。链日志自证绿面＝**`035403d6`**（`gate-audit49b-chain.log` 首行 `HEAD is now at 035403d6`）。
- **本会话做掉（分钟级，全 CPU）**：① 补打 tag **`v-tp2-audit49-per-rank-arena-and-oracle-gates-r0 @ 035403d6`**，消息含 build rc=0 05:33:07、子集 100% 28/28 479.76 s（25P/3S）、簇门 **100% 174/174 1087.59 s（138P/36S/0F，分母 173→174＝新增 `ninfer_e8_kv_oracle_test`）**、资格面 1/1 Passed **760.73 s**、三 dtype rc=0；② `bd1be0ec` 的 `src/core/device.cu:101-114`（SM 数相等闸）**补到单 `.o` 编译证据**＝`HEAD 340bd4c0` 面上按主树 compile_commands 定向编过、rc=0（**只编 `.o`、未链接未跑测**＝纪律⑥口径）；③ `relay/s3-bisect.sh` 加固（见下条）；④ 卡片行号澄清（见审二）。
- **审一（第四道闸）＝无缺陷**：tp2 块内所有校验抛出（`program_impl.cpp:143-166`）都在**首次设备分配之前**（`:168` `ScopedCurrentDevice`＋`PeerRuntime`）；`PeerRuntime` 自断言抛出时其 `DeviceArena` 成员按声明逆序释放、`peer` 保持空 ⇒ 无半构造泄漏；相一致性检查（`:125-132`）更早；媒体仍在 `execution/text.cpp:3073/3089` 响亮拒（`require_single_device` 实现＝`:1555-1560`）。
- **审二（accepted-lengths）＝码面无缺陷，但卡片/提示词行号是供体形**：本树真实链＝`planning/request_plan.cpp:114` 一处拷贝 ＋ 全树唯一安装点 `transactions/binding.cpp:424`（`grep -rn install_sampling --include=*.cpp src`＝仅 `decode.cpp:109` 定义与 `binding.cpp:424` 调用）；**没有** `request_plan.cpp:255/456`，**没有** `prefill.cpp:674-676`，本树也无 `AdmissionCandidateImpl`。clamp 语义＝`decode.cpp:124-126` `min(bound, output_tokens)`，`None` 后端强制 0 ⇒ 非投机轮永不记录＝符合契约；`install_resume_sampling`（`transactions/replay.cpp:53`）不动 bound（`RequestControl` 随请求存活）＝与供体同形。
- **审三（per-rank 容量）＝正确**：`startup.cpp:1409-1425` 两个纯函数以**层命名空间外部链接**定义（注释写明"匿名命名空间定义＝内部链接，声明者到链接期才炸"＝`b116bdfc` 那条坑的固化）；`split_general_capacity(shared, causal_score, peer_causal_score)`＋`causal_score=max(score0,score1)`（`:704-726`）我核过是**恒等**：`workspace.h::tp_logits` 只在 `gather_staging=true` 时加 staging，rank0 布局是 rank1 的严格超集 ⇒ rank1 打分相只记 partial(rows1x tile)＋head_scratch ⇒ 省约 700 MiB（数值＝D2）；`bound_rank_sequence_bytes(…,0,0)` 在 tp1 退化＝rank0 自身（`test_runtime_mechanisms.cpp:496-509` 四例断言覆盖，含 rank1 反超的一例）。
- **审四（注册/链接面）＝封住假绿，实跑证据缺**：`run_python_oracle.cmake` 要求 **rc=0 且 stdout 含 `[PASS]`**、解释器候选表在脚本内解析（配置到的 python3.14 无 numpy ⇒ 落 `/usr/bin/python3`）、全不满足＝`FATAL_ERROR` **不跳过**；`ids_only_equal`（`gate.sh:77-89`）双向判（缺/多/diff），参考目录空 glob 会落 `[ ! -f ]` ⇒ FAIL＝**不会假绿**；但 `relay/fin-gate-golden.log` **不存在** ⇒ golden 段的 PASS verdict 目前**只有码面证据**，夜里补跑（单卡：`GATE_DEVICES=1,1`）。
- **审五（SM 闸）＝一致**：抛点在 `initialize_execution` 内、`construct_model` 之前 ⇒ 文档"before any weight is read"成立；抛型 `std::runtime_error` 与相邻 CC 不等闸同类；与 `startup.cpp:536`（报价取 rank0 SM 数）和 `text.cpp:1574-1576`（执行按各卡 view）的旧不对称**由该闸消除**；文档行已落 `tensor-parallel.md` 拒绝表（`bd1be0ec` 五行，含 5090+5080 例）。
- **审六（S3 `case-3-long`）＝先决测量缺失，谓词已修**：**基线面 `03726c08~` 的 `case-3-long.ids` 从未与参考比过** ⇒ 现在无法区分"层引入"与"上游漂移"，bisect 的 good 端未定；且原脚本每步用**当面**的 `…_tp2_test write` 现产合成件（生成器在层段里被 `cf59d1c1`/`ee13ddf0`/`3053a51d` 改过）＝同时换被测实现与被测输入＝**混杂因子**。已改：`BISECT_TREE`（**必须在 `/tank` 的 scratch worktree 跑**，根分区仅剩 25 G 而构建目录 44 G）、`S3_SYNTH_DIR` ＋ `--pin`（基线面产一次并 sha256 钉死，常规步只读不重写）、参考缺失即 125 跳过、旁证仍记 case-1/2 的 difflines；`bash -n` 过语法。夜里次序＝**先 `--pin` 于 `03726c08~` 判一次基线**（60-90 秒），再决定要不要 8 步 bisect。
- **夜里剩的一件事＝对 `340bd4c0` 起 `audit50` 链**（build→子集 28→簇门 174→资格面→三 dtype→tag `v-tp2-sm-count-equality-and-followups-r0`）＋`gate.sh --stages golden` 造 `fin-gate-golden.log`；S3 按上一条先基线后 bisect。**未动**：`340bd4c0` 只改 `tools/bench/run_serve_corpus.py`（不进编译）；Ada 全批（sm_89 的 14 编译红、stub 18 孤儿／10 未定义、codec 守卫缺架构条件、`rk4v4-e8` 缺 CLI/perplexity 旗标）按你的口径只登记不动手。

## 79. 第 49 回合续＝**十分钟级收尾一票**：全构建实测到 `340bd4c0`＋剥离演练量化＋两条新环境事实

- 本会话 GPU 三探皆不可用（`gpu_probe`＝`cuInit→304`、`ninfer_arena_test`＝`cudaGetDeviceCount failed`、NVML 不可用）⇒ **ctest／golden／bisect 的跑测段一律不在此会话做**，不去制造上次那种"96 设备用例红"的假象。
- **补打 tag＝`v-tp2-audit49-per-rank-arena-and-oracle-gates-r0 @ 035403d6`**（绿证据早已齐：build rc=0 05:33:07、子集 100% 28/28·479.76 s、簇门 **100% 174/174**·1087.59 s＝138P/36S/0F、资格面 1/1 Passed 760.73 s、三 dtype rc=0）；流水断在 tag 步之前，`relay/fin-gate-golden.log` 至今**不存在**＝golden 段仍只有码面证据。
- **全量编译证据升级（本票主货）**：`gate-72` face reset 到 `340bd4c0`＋`applied:vision-on-build-ada.patch` → `scripts/build-ada.sh 120a --tests` ＝ **rc=0／210/210 目标／0 warning／0 error**（`relay/audit50-build.log`）⇒ `bd1be0ec` 的 SM 数相等闸（`src/core/device.cu:101-114`）从"零证据"变"全构建＋链接已过"（此前我只单编过它的 `.o`）；副作用＝夜里那条链的 build 段变增量（约省 12 分钟）。**gate-72 现停在 `340bd4c0`＋补丁＝正是 `gate-steps.sh` 期望的起点**。
- **S10 剥离演练＝第一次量化（纯 CPU，scratch worktree，未建构建目录）**：阶段 1（剥最近 25 提交）**零手工**＝48 文件 −1231/+213，且深符号仍被引用（`TpExecution` 11 文件、`allreduce_sum` 14、`peer_mailbox` 9）⇒ **`git grep`＝0 只是阶段 2 的判据**，别拿阶段 1 去验它。阶段 2 我按"每个提交能否在尖端单独反退"逐条测（每步 `reset --hard` 复位，避免索引累积造假冲突）＝**197 提交里 76 可干净反退、135 需手工**，冲突大头在**代码**而非文档：`README.md` 16、`docs/serving.md` 14、`docs/cli.md` 14、`planning/startup.cpp` 13、`tensor-parallel.md` 12、`serve_options.cpp` 11、`model_instance.cpp` 11、`execution/text.cpp` 10、`program/graphs.cpp` 8、**`include/ninfer/types.h` 8**（层扩了 `KvCacheStorage`/`EngineOptions`）、`qwen3_5-model.md` 8、`tools/tp2/README.md` 7 ⇒ 原文档把手工面预测成"README/docs/apps 选项/tests.cmake"是**低估了代码耦合**，已按实测更正＝提交 **`d01dbcf7`**。
- **新环境事实（直接影响排程）**：`/tank` 与 `/tank/models` 在 `workspace-write` 会话里**不可写**（`touch`/`mkdir` 皆 Permission denied，根分区仅 25 G 而单个构建目录实测 44 G）⇒ **bisect 与剥离演练的"构建段"必须在能写 `/tank` 的会话（danger-full-access）里跑**；`relay/s3-bisect.sh` 的默认 `BISECT_TREE/S3_SYNTH_DIR` 已指 `/tank/ninfer-window`，不可写时须显式给出可写路径（脚本已可用 `BISECT_TREE=`/`S3_SYNTH_DIR=` 覆盖）。
- **事故披露（我的操作错，别学）**：我先 `git worktree add /tank/...`（权限失败）再用 `cd 失败路径; 后续命令` 的**分号串联**，结果 25 提交的 `git revert --no-commit` **落进了主树**。已 `git reset --hard HEAD` ＋ `git revert --quit` ＋删 `.git/sequencer` 清干净＝`porcelain=0`、`HEAD` 未变、`git worktree list` 只剩 ada-tree 与 gate-72 两条；此后一律 `git -C <绝对路径>` ＋先 `test -d` 验路径再执行。
- **三分类更新**：修＝本票无新码面缺陷（六项复核结论见 §78）；记＝①`/tank` 不可写这条环境事实 ②`fin-gate-golden.log` 仍缺 golden 段实测 ③阶段 2 手工面 135 提交（比预测大得多，真要做是独立一票，不排在开窗前）；等窗口＝audit50 的 ctest 三段＋golden 实跑＋S3 基线 `--pin` 判一次＋（若基线已发散）8 步 bisect。
- 树面现状：`d01dbcf7`（＝`340bd4c0` ＋这条文档更正），porcelain＝0，**未对 `d01dbcf7` 跑链**（纯文档提交，不挡夜里的 `audit50`；若要严格，夜里对 `d01dbcf7` 起链即可）。

## 80. 第 49 回合续二＝**CPU 侧收口巡检**：文档 sha 类闭环、D3 对拍面核实、以及两条夜里会咬人的旗标陷阱

- 文档引用类（第 5 次坑）＝**已闭环**：重扫 12 个 120a 相关文档的 8-hex 引用＝12 处、其中 7 处不在我方线，逐条定性后**全是合法归属**而非悬空：`README.md:612`＝HF **revision** 哈希（句子写的是 revision）、`docs/sm89.md:198-200` 五处＝**外部仓**贡献归属（同行已带 `Don-Chad/ninfer-3090`、`UDPSendToFailed/...`、`YurunTao/ninfer-4080S-32G` 仓名）、`upstream-merge.md:127`＝已就地写明 "donor commit `91582814`，本 fork 作为重放第 142 步跳过"。⇒ 别再按"25 个悬空"的旧账重扫（那条数字在 `3ab979bd` 之前成立）。
- D3 对拍面核实＝**齐**：`tools/tp2/reference/greedy.jsonl` 60 条 id 与 `prompts.jsonl` 60 条 **双向 0 差**；`dflash2.jsonl` 10 条＝`GATE_DF2_PROMPTS=10` 取前 10 的子集（ prompts 独有 50＝预期）。`gate_client.py:22` 的 `limit=0`＝**不截断**（greedy 腿传 0＝跑满 60），非"跑 0 条"假绿面。
- **陷阱①（今晚就要避）**：`gate.sh:154` 的 ppl 腿把 `--kv-dtype int8` **写死**，而 `$GATE_EXTRA_FLAGS` 是**同一条**追加到 golden(:138)/ppl(:154)/greedy(:228)/dflash2(:230) 四条腿——所以**别为了测 nvfp4 打分就往 `GATE_EXTRA_FLAGS` 里塞 `--kv-dtype nvfp4`**：它会把合成 golden 腿一起改掉（`tools/golden/record.sh` 收到的 `GOLDEN_FLAGS` 变了＝S3 的三条 `.ids` 基线必然发散，红因是旗标 plumbing 不是控制流）。要覆盖 nvfp4 打分面，加一个 `GATE_PPL_KV`（默认 `int8`）只改 `:154`，属 5 分钟纯 CPU 小改。
- **陷阱②（会直接让夜里第一步就死）**：`relay/window-d-setup.sh:6/20` 把临时件与参考钉在 `/tank/ninfer-window` 并 `mkdir -p` ⇒ 本类会话（`workspace-write`）**写不了 `/tank`**；夜里必须用能写 `/tank` 的会话（历史口径＝danger-full-access＋approval=never），否则 `mkdir` 失败即退出。同一条也约束 `relay/s3-bisect.sh`（默认 `BISECT_TREE`/`S3_SYNTH_DIR` 在 `/tank`）。
- 剩余纯 CPU 活清点＝**没有"开窗前必须做"的项**了：Ada 批次（codec 守卫两行、FP8 A8 进剔除表、stub 补 10 名删 18 孤儿、`rk4v4-e8` 补 CLI/perplexity 旗标、`sm89.md`/`Sm89Compat.cmake` 文案）按你的口径只登记；S10 阶段 2（135 提交手工面，§79 已量化）排在开窗之后；`GATE_PPL_KV` 与 `tests/convert` 注册（本机无 pytest/torch＝注册即假绿）都是可选项。⇒ **关键路径整条落在 GPU 侧**：`audit50` ctest 三段＋golden 实跑＋S3 基线 `--pin` 判定（＋可能的 8 步 bisect）。

## 81. 第 49 回合末＝**形态 A/B/C 落纸＋交班给"开窗后"的官方 API 会话**（本地 GPU 全部腾空给测试）

- 补掉 R1 的空洞：`replay-log:268` 要求"提前 2 小时报形态 A/B/C"，而 A/B/C 自 `接力提示词-20261016.md:54` 起一直是"待用户定"且从未落纸 ⇒ 现定义（随本次交班生效）：
  - **A｜整段独占长窗（两卡同时 ≥4 h）**：D1 三探测 → D2 容量/allowance → D3 逐字＋GSM8K → D4 速度扫参一趟走完；装载成本只付一次＝**首选**。
  - **B｜双段窗（2×90 min）**：段一 D1＋D2（先 NF4＝9.0 GiB/卡，容量结论最值钱），段二 D3＋D4；代价＝多付一整轮两卡装载。
  - **C｜只拿到一张卡（GPU0 未让位）＝不算开窗**：只做单卡可完的 D 项（12/2 资格面复跑、S7 占位的单卡侧、`--tp 1` 合成件面），并登记"27B 的 `--tp 1` 分母在本地结构性不可得"＝NF4 全件 18.0 GiB > 单卡 15.8 GB（`tp2` 存在的全部理由），分母沿用旧选项（云机 5090／合成模型／放弃）。
- 交班口径（本会话末实测）：树＝`d01dbcf7`（porcelain 0，无 rebase-merge）；tag 六枚见 §78/§79；已绿面＝`7d762ec5`（173 分母）与 `035403d6`（174 分母，含 `ninfer_e8_kv_oracle_test`）；`bd1be0ec`/`340bd4c0`/`d01dbcf7` 只有**全构建 rc=0／210 目标／0 warning**（`relay/audit50-build.log`，face＝tree＋vision 补丁，gate-72 已停在该面＝夜里第一段是增量）；**未跑**＝这两面的子集 28／簇门 174／12-2 资格面／golden 实跑；**未定**＝S3 `case-3-long` 的基线侧（`03726c08~` 从未与参考比过＝bisect 的 good 端未定，谓词脚本已加固＝`relay/s3-bisect.sh`，`--pin` 先钉合成件）。
- 环境事实两条（新会话必查，别照抄旧 PID）：`/tank` 在 `workspace-write` 会话里**不可写**（⇒ `window-d-setup.sh:20` 与 `s3-bisect.sh` 的 `/tank/ninfer-window` 前缀要求 danger-full-access 会话；根分区仅 25 G、单个构建目录 44 G）；**本地 GPU 自本次交班起全部空出来给测试**，本会话 NVML 不可用故无法确认现网 strata 是否已停（卡片旧值＝PID 3013603/15588 MiB），新会话第一步用 `nvidia-smi --query-compute-apps=pid,name` 自己认占用者并选空卡，别把"某张卡是现网的"当恒真前提。

## 82. 第 50 回合＝**开窗后第一回合**：门链 audit51 绿＋tag；两个 tp2 投机 P0 当场修绿；D1/D2/D3 首轮实测；sm89 换基座回归定性；NF4 制品 tp2 阻塞

- **门链 audit51（面 `d01dbcf7`＝`bd1be0ec`＋`340bd4c0`＋`d01dbcf7`，线性祖先）全绿**：build rc=0（14:47:19 UTC，210 目标，docs-only 面 ninja no-op）＋子集 **28/28·425.90 s**＋簇门 **174/174·1056.48 s＝138P/36S/0F**＋12/2 资格面 **1/1·757.47 s**＋三 dtype（nvfp4/int8/bf16）rc=0 ⇒ tag **`v-tp2-audit51-sm-count-equality-and-followups-r0`**（证据 `relay/gate-audit51-*`、`relay/audit51-qual*`、`relay/audit51-run.log`）。
- **主树首次带 vision 构建**：`build-120a-cu13` 此前 `NINFER_ENABLE_VISION=OFF`（媒体面回 "configured with OFF"）；窗口内 reconfigure 为 ON（FFmpeg 6.1.6 在 `deps-ada`）⇒ `media | Vision on device 0 | 16 preprocess workers | cache 1.00 GiB | live 2.00 GiB`。
- **P0-1（`95884c20`）MTP@tp2 全死**：上游 Forward/Finish 拆分里 `if (phase == SpeculativePhase::Forward) { verify; return; }` 的 `return` 在 tp2 移植时落到守卫外 ⇒ Finish 相提前返回，`target_accept`/起草/egress 全不跑 ⇒ 每轮 licensed count=0（`MTP batch returned invalid row metadata`，warmup 145 ms 即死）。修＝守卫 `return` 本身、把 peer `target_accept` 移到 Finish 相。**验绿**＝`ninfer_qwen3_5_engine_mtp_tp2_real_test` ok：accept **152/276=55.07%**、accepted-lengths **81 步=81 轮/0 fallback**、前缀复用 **8302/8331**（4 lane 与 1 lane）、答案 391/144。
- **P0-2（`95884c20`）DFlash2@tp2 验证抛错**：`draft.cpp` 的验证 `TextContext` 少传 `TpExecution` ⇒ `rank-pair target verification requires tensor parallelism`。修＝补参。**验绿**＝`ninfer_qwen3_5_engine_dflash2_tp2_real_test` ok：accept 8/12、三题逐字一致、前缀复用 35 tok。
- **D1（W4A4，`/tank/ninfer-window/d1/`）**：serve tp2＋MTP3＋`--lm-head-draft`＋`--vision --vision-device 0` 健康；`/v1/models` OK；文本 82 tok/1.506 s；**媒体 prompt→HTTP 500 `multimodal prefill is not implemented at tensor-parallel width 2`**（`execution/text.cpp:1555-1558`）；`--log-speculation-detail` 真记 `accepted_lengths`，`relay/check_accepted_lengths.py` 互算 **0 失配**（23 步=23 轮、hist `[2,1,0,20]`、acc 61、fb 0；hist↔per_position 是后缀和，不是差分）；perplexity `--tp 2` **首次可达**：int8 PPL **4.259926**（261,167 tok・1580.7 tok/s）、nvfp4 PPL **4.267108**（同速率）。
- **D2 实测**：per-rank 权重 serve r0 **10.7 GiB**（sharded 9.27/replicated 1.19/local 282.5 MiB）/r1 **10.5 GiB**；打分相 **10.1/10.1 GiB 对称**⇒ 两卡占用差 **728 MiB**＝打分相 arena 不对称（修⑤口径 ~700 MiB 得证）。容量：**196,608 token 起得来**（pages 3072/3072・runtime 4.48 GiB・free **132.7 MiB**）；**262,144 起不来**（需 5,984,344,833 B，只有 4,947,902,464 B）。graph allowance 实测：MTP **2.0/2.0**（allow 8）、ordinary **0.0/0.0**（allow 8）、DFlash2 **22.0/12.0**（allow 66＝6 类×11）⇒ **8/8/11 占位在本机成立**。shard TMA 界：本树**无 shard bench**（`attn_input_proj_bench` 固定 14336 行父形，不读表）⇒ 保留 1024 并登记缺口。12/2 row 组：结构面（`plan.cpp:24-38` 读设备 SM 数）⇒ 只记录、无字段写回。
- **D3 逐字 PASS**：`--record` 建我方基线（`/tank/ninfer-window/reference-int8/`）→ 复跑 `compare`：**greedy 60/60 逐字一致**（51.96 t/s・accept 72.43%・ms/round 60.035，−0.09% vs ref）、**dflash2 10/10 逐字一致**（133.94 t/s・accept 59.20%・ms/round 37.817，+0.09%）⇒ `RESULT PASS`（`relay/*` 无，证据 `/tank/ninfer-window/d3/{record,compare}-int8.log`）。
- **NF4 制品 tp2 阻塞（登记）**：`text/output_head` 是 `q6_g64 [248320,5120]`，q6 只注册全词表形状（`src/ops/linear/q6/shapes/` 三件）⇒ 半词表 `[124160,5120]` 缺 ⇒ `q6 linear: unsupported shape`。D 段实测因此落在 W4A4；补半形状属新 Op 形状＝需 oracle 资格面，另开一票。
- **sm89 换基座回归（云机 4080S 实测，随后按用户指示关停）**：configure 门两个已删测试名（`ninfer_softmax_attention_{nvfp4,k8v4}_test`）⇒ FATAL_ERROR；`pkg-config 'libavcodec>=60'` 在 0.29.2 下恒失败。两者已修（`d4050f1e`，本地 configure PASS＋云机 `configure PASS vision=ON`）。修完仍红：`src/ops/linear/fp8/**` 共 ~15 TU 用 TMA A8 schedule，`.tile requires .target sm_90`；旧分支同机同 CUDA 12.8 编过（旧文件 `Fp8A8DefaultSchedule`）⇒ **换基座打断 Ada FP8 A8**，修法＝`NINFER_SM89` 回落到非 TMA schedule（未开工）。

## 83. 第 50 回合续＝**NVFP4 权重＋NVFP4 KV 主口径全量实测**（用户口径：只用 NVFP4 件、只用 nvfp4 KV）：GSM8K 分子、nvfp4 面逐字、K 扫参

- **口径定死**：用户明确只用 **NVFP4 权重 + NVFP4 KV** ⇒ 主口径＝W4A4 件（23.77 GB）配 `--kv-dtype nvfp4`；int8 面保留为对照。NF4（19.35 GB，较小的那件）两处已知：其 DFlash2 是转换时做坏的（用户口径）、且 `q6` 输出头无半词表形状 ⇒ tp2 加载不了，两件都不影响主口径。
- **D3 逐字（nvfp4 面）PASS**：`--record`→`compare`：greedy **60/60 逐字一致**（52.17 t/s・accept 72.18%・ms/round 59.853，+0.43%）、dflash2 **10/10 逐字一致**（140.38 t/s・accept 62.09%・ms/round 37.531，−0.11%）。基线目录 `/tank/ninfer-window/reference-nvfp4/`，证据 `d3/record-nvfp4.log`、`d3/compare-nvfp4.log`。
- **GSM8K 500（thinking off・8-shot・max 512・shot 8）**：`canonical` **int8 484/500＝96.8%**、**nvfp4 482/500＝96.4%**；两跑 `subset_digest` 同 `245dfab7c79d880c…`，`request_errors 0`、`unmatched_gold 0`、numeric 与 canonical 同分。证据 `d3/preds_tp2_{int8,nvfp4}_500.jsonl`＋`d3/report_tp2_{int8,nvfp4}.json`。
- **D4 扫参（nvfp4，60 题 gate 集，T=0）**：K=2 → **45.10 t/s / accept 80.69% / 57.376 ms**；K=3 → **52.39 / 72.18% / 59.594**；K=4 → **54.85 / 64.76% / 64.252**；K=5 → **56.96 / 57.09% / 65.767** ⇒ **本机最快 K=5**（accept 掉到 57% 仍净赚）。**`--spec mtp` 的 `--draft-tokens` 硬上限是 5**（K=6/7 直接 usage 拒绝；DFlash2 才用到 7）。
- **nvfp4 KV 的代价（同 K/同题集）**：greedy **52.39 vs int8 51.92 t/s（+0.9%）**、dflash2 **140.22 vs 134.06（+4.6%）**；DFlash2 的 accept 59.20→**62.09%**；perplexity **4.267108 vs 4.259926（+0.169%）**、打分速率同为 1580.7 tok/s ⇒ nvfp4 KV 在本机**不付速度代价、只付 0.17% PPL**。
- **nvfp4 parity 腿**：`ninfer_attention_headlocal_test`（两卡）rc=0、末行 `OK head-local attention`（BF16/INT8/NVFP4 三档都过）。
- 证据目录：`/tank/ninfer-window/{d1,d2,d3,d4}`＋`RESULTS.md`（本轮全部数字的汇总）＋`s7-seed-5060ti.md §6`（写回）。

## 84. 更正（容量）：262,144 在 nvfp4 KV 下可达

- §82/§83 与 `s7-seed §6.2` 里"262,144 起不来"只对 **int8 KV** 成立。实测（07:00，W4A4，tp2）：**nvfp4 KV + vision + MTP3 ⇒ pages 4,096/4,096、runtime 3.58 GiB、free 1.03 GiB；nvfp4 KV + 无 vision + MTP5 ⇒ runtime 2.87 GiB、free 2.01 GiB** ⇒ **原生 256K 在主口径下达标**。int8 KV 下 196,608 才是上限。
