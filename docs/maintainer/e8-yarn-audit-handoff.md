# E8 4-bit KV + YaRN 移植 —— 审计复核阶段接力提示词

> 给接力者（agent 或人）的接手说明。前序开发阶段已完成并全部提交；你的任务是**代码审计复核 + 尽量在零 GPU 环境下消解数值风险**。
>
> **阶段划分**：你负责第一、二阶段——§3 代码审计 + §4.1/§4.2 零 GPU 编译与 CPU 数值验证（发现即修，一层一提交）。第三阶段——内核数值 vs oracle、长上下文生成质量（§4.3 全部）——由用户在 GPU 环境独立验收，**不是你的交付条件**；你的报告只需把 §4.3 清单原样移交并说明哪些风险已在 CPU 侧消解。若运行环境没有 GPU，凡触到 `cudaInitialize` 的测试一律记"未执行（无 GPU）"，不得视为失败或试图绕过。

## 0. 项目与机器现状

- 仓库：`/home/ai-agent/DSHW/ninfer-blackwell-ada`（master，工作树干净，HEAD=`b4a0c0ee`）。
- 上游：Neroued/ninfer（`sm_120a`）；本仓新增 Ada `sm_89` 构建（`docs/sm89.md` 为架构对照权威）。
- **本机零 GPU**。CUDA 13.2 在 `/usr/local/cuda`。构建配方（已验证可用）：
  ```bash
  export PATH=/home/ai-agent/.local/bin:/usr/local/cuda/bin:$PATH   # nvcc wrapper 是坏的，必须用 /usr/local/cuda/bin/nvcc
  cmake -B build-e8-sm89   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89  -DNINFER_ENABLE_VISION=OFF -DBUILD_TESTING=OFF -DNINFER_BUILD_APPS=OFF -G Ninja
  cmake -B build-e8-sm120a -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_ENABLE_VISION=OFF -DBUILD_TESTING=OFF -DNINFER_BUILD_APPS=OFF -G Ninja
  cmake --build build-e8-sm89 -j$(nproc)      # 两目录均已存在且最近一次全绿（87/87）
  ```
  注意：cmake 是 pip 装的 4.3.2；**重新 configure 会清掉缓存变量**，必须重传上面全部 `-D`。
- 测试二进制在无 GPU 机器上会 abort（rc 134，`cudaInitialize` 失败）——这是基础设施限制，不是缺陷。`BUILD_TESTING=OFF` 下测试不进 cmake；独立编译检查用：
  `g++ -std=c++20 -fsyntax-only -Itests -Iinclude -Isrc -Ithird_party -Ithird_party/spdlog/include -I. -I/usr/local/cuda/include <file>`

## 1. 提交链（审计对象）

| 提交 | 层 |
|---|---|
| `3120a690` | P1：E8 移植冲突面盘点（`docs/maintainer/e8-port-surface.md`） |
| `52caf5cb` | P1：E8 编解码落树未接线（`e8_lattice.cuh`、`e8_root_codec.cuh` 后者属 RK2V4E8，本系列未使用，保留） |
| `e48c5845` (A) | 存储/追加/容量/CLI/门控：`KvCacheStorage::RK4V4E8`、layout、append 行函数、sm_120a stub、`--kv-dtype rk4v4-e8` |
| `b5b45039` (B) | 注意力读侧：`small_t_e8.*` + `prompt_e8.*`（i8 机制镜像），四入口点短路，E8 测试 oracle |
| `b4a0c0ee` (C) | YaRN 双架构：`TextRopeScaling` 系数表、`YarnOptions` 流经 `EngineOptions→Parameters→text/MTP/MRoPE`、CLI/serve 旗标、HF fixture 测试 |

## 2. 硬约束（审计时不得破坏）

- 不 push origin；一层一提交；BFloat16/Int8Group64 既有路径零 diff（E8/YaRN 只加分支）。
- E8 = sm_89-only 设备路线（sm_120a 走显式错误 stub，抛 `sm120a_route_unsupported`）；YaRN = 唯一双架构流。
- 除上游 codec 语义（半余弦 rint 是该模式的定义存储）外不引入数值近似；逻辑保持简单。
- 标度契约（A 提交内修过 bug，重点复核）：**K、V 各自从自己旋转后行的 absmax/7 定独立 FP16 标度**（`kv_cache_append_full_e8_row` 内 V 累加前 `local_absmax = 0.0F;` 复位）；K code clamp [-8,7]（i8 域），V code clamp [-7,7]（i4 域）；decode = code × scale。

## 3. 审计优先级清单（按风险排序）

1. **E8 NLP warp 版 vs 标量版等价性（最大未验证项）**。设备走 `e8_project_8d_warp_single`（[src/ops/kv_cache/e8/e8_lattice.cuh](../../src/ops/kv_cache/e8/e8_lattice.cuh) L109+），host oracle 转录的是**标量** `e8_project_8d_fast`（L37-106）。半余弦 tie 的舍入路径若不一致，E8 全部数值测试会系统性失败。→ 零 GPU 可消解大半，见 §4.2。
2. **`small_t_e8.cuh` / `prompt_e8.cuh` 与 i8 镜像的逐项对照**（[src/ops/softmax_attention/dense/causal_cache/](../../src/ops/softmax_attention/dense/causal_cache/)）：arena 四区别名（k_code/v_code/k_i8/v_f16 = 4×Bc×128 B，必须与 i8 arena 逐字节一致）；Q 预处理 = D256 旋转 + warp_max amax + `kv_cache_int8_quant_code`；K tile 解码 `16B→32 codes→causal_small_t_store_byte_swizzled`；V loader `4B→8 codes×vscale→f16`，vscale 广播 `__shfl_sync(FullMask, vs, (lane&7)*8)`；流水线 `issue_kv_tile(0); cp_wait<0>(); decode_k_tile();` 的相位。tier 表必须与 i8 完全一致（`launch_tc_partial_e8`）。
3. **prompt_e8 资源**：smem 91 136 B（static_assert 在文件内），`__maxnreg__(120)` 是沿用 i8 的值——Ada 实机 occupancy 未验证，若 prompt 路由慢，先查这里。
4. **入口分派**：`causal_softmax_attention.cpp` 的 `resolve_route` E8 分支（`width<=8 ? 0 : 256`，继承 int8 prompt_limit）+ `small_t.cu`/`prompt.cu` 四个入口点在 Nvfp4Group16 块之后的 `RK4V4E8` 短路。
5. **sm_120a 门控**：[cmake/Sm120Compat.cmake](../../cmake/Sm120Compat.cmake) 过滤 `/kv_cache/append/e8_launch\.cu$` 与 `/causal_cache/(small_t|prompt)_e8\.cu$`，链入 `src/ops/sm120a/e8_sm120a_stubs.cpp`（append + attention 全部 stub 抛 `sm120a_route_unsupported`）；`NINFER_SM89_BUILD` 下该机制 no-op。核对：120a 构建里没有任何 E8 设备对象残留。
6. **YaRN 分派完整性**：`src/ops/wrapper/rope.cpp` 的三个 `rope()` 入口都必须先过 `require_scaling`（D256/R64 或三轴 MRoPE 之外一律拒绝）再路由 `launch_generic`；fixed 内核（含 DFlashText1D）仅在 scaling 为空/禁用时可达。`Parameters` 构造（[src/models/qwen3_5/execution/parameters.cpp](../../src/models/qwen3_5/execution/parameters.cpp)）只在 `factor>1 && rope_parameters` 时建表，factor=1 必须逐位走原路径。
7. **YaRN 校验位置**：`validate_yarn` 在 `validate_options`（`model_instance.cpp:19`，由 `construct_model:159` 调用）——覆盖 CLI/serve/公共 Engine API 全部入口；CLI 与 serve 的 parse 层另有独立校验（`apps/cli/options.cpp`、`src/serve/serve_options.cpp`）。注意 `CausalScoring` purpose 只重置 speculative/vision/context_cache，**不重置 yarn**——确认这是否符合产品意图（打分窗口用 YaRN 旋转是自洽的，但需明确）。
8. **容量算术单一来源**：`paged_kv_storage_layout(RK4V4E8,256)` = 对称 {U8,128,FP16,1} = 260 B/tok/head（int8_g64 528、bf16 1024）。所有 `kv-capacity` 解析/压力规划/测试都应经它，无旁路常数。
9. **文档一致性**：`docs/cli.md`、`docs/serving.md` 的 `#yarn-context-extension` 小节、README 支持矩阵行、`docs/sm89.md` 的 "Everything else" 行、AGENTS.md 段落——五处都已加，核对措辞与代码一致（尤其"参考窗固定 262144"这一 Qwen3.8-27B 模型属性）。

## 4. 云机（无 GPU）可验证的部分 —— 回答"能不能不插 GPU 验证"

**能，且覆盖面不小。分三档：**

### 4.1 直接可跑（纯 CPU，无需改码）
- **双架构全量编译**：nvcc 编译/链接不需要 GPU（只是无法执行）。两条 `cmake --build` 就是 CI 级门。
- `test_yarn_parameters`：HF transformers 5.15.1 fixture 交叉验证 `make_text_yarn_scaling`（相对 3e-7）+ `validate_yarn` 边界 + `yarn_cache_binding` 契约。**main() 不调任何 CUDA API，零 GPU 机器可直接执行**（首跑确认；若静态链入的 CUDA 代码在 load 期触发驱动探测再议）。
- `test_rope_scaling_options`：CLI/serve 两侧 `--rope-yarn-factor / --rope-original-max-position` 的解析、拒绝与默认值。纯 CPU。
- 套件中其余纯 CPU 测试（serve 选项、schema、request log 等）照常 `ctest`。

### 4.2 值得新增的零 GPU 等价性检查（消解 §3.1 的大半风险）
写一个 host-only 小工具（不进产品树，放 `/tmp` 或 `tools/` 一次性）：
1. 把 `e8_lattice.cuh` 的**标量** `e8_project_8d_fast` 以 `#define __device__ / __forceinline__` 为空宏的方式 #include 进 host TU（它不依赖 CUDA intrinsic，可行）；
2. 与测试树里的 host 转录 `e8_project_8d_host`（`tests/ops/softmax_attention/causal_cache.cpp` 内）做 **10^6 量级随机向量对拍**（D256 旋转后的分布形态）；
3. 对全部 240 个 E8 单向量（及若干 2 倍向量）验证投影恒等 + 最近点性质；
4. 再对 8 维纯轴向量、半余弦边界构造向量（两候选等距）定向打 tie-break 路径。
做完后"warp==scalar"虽仍需 GPU 收尾，但"标量语义本身正确 + host oracle 是忠实转录"这两半在云机就锁死了。

### 4.3 必须真 GPU 的部分（如实报告给用户，勿假装验证）
- E8 append/attention 内核 vs oracle 的全部数值测试（`test_kv_cache`、`test_kv_capacity`、`causal_cache` E8 profile）——`test_rope*` 同样（有 SKIP rc 77 门）。
- YaRN 端到端长上下文质量（`tools/bench/verify_yarn_context.py`，需运行中服务 + 本地 tokenizer）；E8 判据 `kAttentionE8Criterion` 目前沿用 NVFP4 宽值（1.5e-2/5.0e-3/1.1e-2），**实机数值出来后应评估收紧**。
- sm_120a stub 在 5090 上的实际抛错行为；prompt_e8 occupancy（91 136 B smem / 120 寄存器）。

## 5. 前序遗留的已知未验证清单（原样带入你的报告）

E8 数值未过 GPU；E8 判据偏宽；warp-vs-scalar 等价性（§3.1）；sm120a stub 未上 5090；YaRN 端到端未验证；`e8_root_codec.cuh`（RK2V4E8）在树内未使用属预期；27B 实际层数/head 数请以用户机 artifact config 为准（本仓无 27B artifact，只有合成测试形状）。

## 6. 审计产出格式

每个发现：`结论 + file:line 证据`；修 BUG 时一层一提交（Conventional Commit 小写主题，如 `fix(kv): ...` / `fix(rope): ...`）；不 push；改完保持双架构 `cmake --build` 全绿再交付。
