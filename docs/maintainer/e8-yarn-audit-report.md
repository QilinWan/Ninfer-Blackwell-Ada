# E8 / YaRN 审计报告（阶段 1–2：§3 代码审计 + §4.1/§4.2 零 GPU 验证）

审计基线：`b4a0c0ee`（含 A/B/C 三层：`e48c5845` 存储/append/CLI、`b5b45039` attention 读路径、`b4a0c0ee` YaRN；P1 codec `52caf5cb`）。
审计提交：`be84d07f`、`61730546`、`a2851c04`、`b894e7ef`、`5c845951`（每层一提交，未 push，`master` ahead 10）。
机器：无 GPU（nvcc 可编译/链接 = CI 级门禁；`cudaInitialize` 触发的二进制 rc 134 记「未执行（无 GPU）」）。

## 0. 结论摘要

- 修复 5 项（2 个真实数值 bug + 1 个链接期隐患 + 1 个文档缺口 + 1 个测试接线），全部已提交（`master` ahead 10，未 push）。
- 前序上游遗留 1 项（i8 两条读路径的 V 解码 4 字节欠载）——按「既有路径零 diff」约束不动，单列并给出复现证据。
- 零 GPU 数值面全部锁死：E8 标量投影 = 真最近格点算法（20000 个生产形状块，独立穷举 oracle）；测试树 host oracle = 设备头文件的逐位忠实转录（1e6 块零分歧）；240 个 E8 根向量 + 2× 全部投影不动；边界平局行为与文档一致。YaRN：`test_yarn_parameters`、`test_rope_scaling_options` 独立编译运行均 OK（两台机器各跑一遍全绿，见 §4.1）。
- 两台机器的编译/链接门禁全关：审计机 sm_89+sm_120a 对象/库与 `build-sm120a-full`（全测试树）rc=0；云机 sm_89 `build-sm89-full`（关键 15 目标，CUDA 12.8）rc=0。
- **§4.3 GPU 数值验收**：E8 路线首次在真 Ada GPU（云机 4080S）上执行 → **失败**（完整证据 §5.3）；GPU 调试已按 `e8-yarn-audit-relay.md` §4.3 的探针程序移交下一位调试者（本审计者的沙箱不可安全使用云机 GPU）。

## 1. 代码修复（结论 + 证据）

### 1.1 prompt-E8 V 解码：半字节索引错 + 栈越界 —— 已修（`be84d07f`）
**结论**：`causal_prompt_e8_dequant_f16x8` 把 Int8Group64 的「每维一字节」循环原样搬到了 4-bit 打包平面上：`load_vec<int2>`（4 字节 = 8 个 code，宽度本身对）之后按 `p[2*i]`/`p[2*i+1]` 取 code，i<4 时读 `p[4..7]` = 栈上 4 字节越界；且两个操作数都取 `high=0`（低半字节），奇维 code（高半字节）整维丢失 → 整条 prompt-E8 路线的 V 全错。
**证据**：`src/ops/softmax_attention/dense/causal_cache/prompt_e8.cuh` 修复前 L67-68（`unpack_i4(p[2*i], 0)` / `unpack_i4(p[2*i+1], 0)`）；正确参照：同文件 K 解码与 `src/ops/kv_cache/e8/e8_i4_code.cuh:101-102`（`unpack(src8[i],0)` / `unpack(src8[i],1)`，每字节两半字节全取）；半字节语义 `e8_i4_code.cuh:91-94`（`high=0` 低半字节=偶维）。
**修法**：`p[i]` 的低、高半字节各解一维（`unpack_i4(p[i],0)` / `unpack_i4(p[i],1)`），与 small-T E8 的 V 解码一致。

### 1.2 small-T-E8 scale 分档漏 tile 基址 —— 已修（`61730546`）
**结论**：`issue_kv_tile` 给 per-key scale 平面分档时用 `(4*q4) & kPagedKVPageMask` 作页内偏移，漏掉 tile 基址；K/V code 平面（同函数 L310-311）用 `key & kPagedKVPageMask`（`key = tile_k0 + 4*q4`）是对的。Bc=32 tile 且 `tile_k0 % 64 == 32`（如 split_start ∈ [33,63]）时，scale 从同页早 32 个 key 处读 → 该 tile 内 QK 重缩放与 V 解码全部乘错 scale。Bc=64 tile 页对齐，不受影响。
**证据**：`src/ops/softmax_attention/dense/causal_cache/small_t_e8.cuh` 修复前 L294-296；i8 原型同款分档用 `key & kPagedKVPageMask`（`small_t_i8.cuh:351-352`）；4-key cp_async 组永不过页（Bc≤64 tile 落在单个 64-token 页内，已核对）。
**修法**：与 code 平面一致改 `key & kPagedKVPageMask`。

### 1.3 `append/e8_launch.cu` 未入源列表 —— 已修（`a2851c04`）
**结论**：`src/ops/kv_cache/sources.cmake` 只列了 `kv_cache_append.cpp / launch.cu / nvfp4_launch.cu / k8v4_launch.cu`，`append/e8_launch.cu`（定义 `kv_cache_append_e8_launch` / `_batch_launch`）不在任何列表 → sm_89 静态库中两符号未定义。因 `BUILD_APPS=OFF`+`BUILD_TESTING=OFF` 不做最终链接，隐患被掩盖；任何链接产品二进制的配置都会炸。
**证据**：`src/ops/kv_cache/sources.cmake` 修复前 6 行无 e8；`build-e8-sm89/build.ninja` 有 `small_t_e8.cu.o`/`prompt_e8.cu.o` 而无 `e8_launch.cu.o`；sm_120a 侧 `cmake/Sm120Compat.cmake:14` 的正则 `/kv_cache/append/e8_launch\.cu$` 会再把它从 120a 构建树剔除（由 `src/ops/sm120a/e8_sm120a_stubs.cpp` 的显式 `sm120a_route_unsupported` 桩顶替），故本修复只影响 sm_89，120a 行为不变。
**修法**：源列表补 `"${CMAKE_CURRENT_LIST_DIR}/append/e8_launch.cu"`。

### 1.4 README 支持矩阵缺 RK4V4E8 —— 已修（`b894e7ef`）
**结论**：矩阵行只列 `INT8, FP8, BF16`（Ada 列），与 `docs/sm89.md`（L37、L42-44）和 `docs/cli.md`（L214、L259）已文档化的 Ada-only `rk4v4-e8` 路线不符。
**修法**：Ada 列补 `RK4V4E8 (Ada only)`。

### 1.5 `ninfer_yarn_parameters_test` 漏链 `ninfer::json` —— 已修（`5c845951`）
**结论**：该测试直接 `#include <nlohmann/json.hpp>` 却只在 LIBRARIES 里链 `ninfer_ops`，nlohmann 头由 `ninfer::json` 目标提供。任何开启 `BUILD_TESTING=ON` 的完整构建都会在其编译期炸（`nlohmann/json.hpp: No such file or directory`）。前 4 个修复都在 `BUILD_TESTING=OFF` 的对象/库阶段，故这个测试接线 bug 只有跑**完整测试构建**时才暴露——正是 §4.1 全量本地构建（`build-sm120a-full`）逮到的。
**证据**：`tests/test_yarn_parameters.cpp:4`（直接 include）；`tests/cmake/NinferTests.cmake` 修复前该 target 的 LIBRARIES 仅 `ninfer_ops`；同文件里其它消费 json 的 target 均显式链 `ninfer::json`。
**修法**：LIBRARIES 补 `ninfer::json`（对齐同文件其它 json 测试 target 的既有写法）。

## 2. 报告项（未修，附证据）

### 2.1 【前序上游遗留】i8 两条读路径 V 解码 4 字节欠载
**结论**：Int8Group64 的 small-T 与 prompt 两条读路径，V 解码 helper 用 `load_vec<int2>`（4 字节）却按 8 个 code 索引 `c[0..7]`：后 4 个 code 读的是栈上其它局部数据（编译期确定但非输入）。这是本系列之前（`ff8f6034` 及更早的上游代码）就存在的模式，本系列对 i8 文件零 diff（已核对 `git diff 76e7f2ed..HEAD` 空）。按「既有路径零 diff」约束本次不动；建议单独走一轮 GPU oracle 验证后再修（修复会把 i8 数值行为改变，必须 GPU 复验）。
**证据**：`prompt_i8.cuh:51-66`（`load_vec<int2>(codes8)` 后 `c[2*i]`/`c[2*i+1]`，i<4 → `c[4..7]` 越界）、`small_t_i8.cuh:541-549`（同模式）；host 复现（`/tmp/e8audit/loadvec_host.c`，g++ -O3）：期望 1520，实测 1620 → 后 4 字节确为栈数据。

### 2.2 E8 small-T 复用 BF16 split 策略（性能注记）
`active_splits<Geometry,false>`（bf16 split policy）对 RK4V4E8 生效；host launcher 与 kernel 两侧策略一致（RK4V4E8 ≠ Int8Group64 → 两侧都落到默认策略），语义自洽；E8 tile 解码成本高于 int8，split 数的最优值属 §4.3 GPU 性能项。

### 2.3 `yarn_cache_binding` 无产品消费方
`src/runtime/contract/yarn.h:28` 的 binding 串目前只有测试引用；各 Engine 的 yarn 状态不跨 Engine 边界，产品行为一致，属防御性契约，保留。

### 2.4 prompt-E8 占用率参数
`__maxnreg__(120)`（自 i8 继承，`prompt_e8.cuh:76`）与 smem 91136 B（L53 static_assert）在 Ada 上的实际占用率未测 —— §4.3 GPU 项。

### 2.5 `kAttentionE8Criterion` 沿用 NVFP4 宽阈值
`tests/ops/softmax_attention/causal_cache.cpp:86`（1.5e-2 / 5.0e-3 / 1.1e-2）；是否收紧等 §4.3 真实 GPU 数值后再评。

### 2.6 `e8_root_codec.cuh` 未被引用
预期状态（标量/warp 投影在 `e8_lattice.cuh`），无需处理。

## 3. §3 逐项审计结论（1–9）

- **3.1 E8 投影（scalar + warp）**：`e8_lattice.cuh` L37-106（标量）与 L109-182（warp）结构一致：同一「rint → 奇偶性在最坏维（误差最大，平局取最小下标；方向 `x>=f ? +1 : -1`）→ D8 vs D8+0.5 双候选、`dist_d8 <= dist_coset1 ? d8 : coset1`」；warp 版 3 步 `shfl_xor` 蝶式求和/最坏 lane 归约的合并规则（取大、平局取小 lane）可交换可结合，4 个 8-lane 子组（`0xFFu << (lane&24)`）各自独立。唯一差异是 FP 加法顺序（顺序 vs 蝶式）——精确等距时两边加和结构相同仍精确相等 → 平局一致；warp==scalar 的最终锁死在 §4.3（GPU）。**标量语义已由 §4.2 独立穷举 oracle 锁死**。
- **3.2 codec 契约（`e8_i4_code.cuh`）**：per-token FP16 scale = `absmax/7`（`min(65504, max(0x1p-24, ·))` 有界，L63-70）；K 投影后 `rintf` 夹 [-8,7]（半余弦坍缩 = 文档化存储语义）；V `__float2int_rn` 夹 [-7,7]；V 累积前 `local_absmax = 0.0F` 复位（K/V 独立 scale ✓）；偶 lane `pack_i4(code, shfl_down(code,1))` 写 `code_index(d/2)` ✓（两 code/字节，低半字节=偶维）。
- **3.3 append（`append/e8_kernel.cuh`、`e8_launch.cu`）**：warp-per-token（lane l 持维 l+32r），(r, lane&7 组) 恰好覆盖互不重叠的 8 维连续块，与测试 oracle 的分块一致；偶 lane 打包对 `(2i,2i+1)` 维正确；token≥128 && KVHeads==2 走页 kernel。masked/batch 变体与裸变体共用行逻辑。
- **3.4 small-T-E8 流水线**：arena 4 槽别名与 i8 同构；`issue_kv_tile(first_tile); cp_wait<0>(); decode_k_tile()` + 循环内 `__syncthreads()`（L522）+ `if (has_next){ cp_wait<0>(); decode_k_tile(); }` —— 末迭代省 sync 无竞态（smem 槽在本 warp 组内顺序复用）；4-key 组不过页（已核对）。除 1.2 的 scale 偏移外无其它缺陷。
- **3.5 prompt-E8**：smem 布局 `kCausalPromptE8SmemBytes == 91136`（L53，i8 为 92672）；vscale 广播 `__shfl_sync(FullMask, ..., (lane&7)*8)` ✓；prompt tile 恒 Bc=64 页对齐，`(4*q4)` 未掩码在此正确（页内偏移==全局偏移）。V 解码缺陷见 1.1。
- **3.6/3.7 YaRN**：`yarn.h` `validate_yarn`（factor 有限 ∈[1,4]、original_context 恰 262144、max_context ≤ 262144×factor）；`rope.h:12-16` `TextRopeScaling` NSDMI 使 factor==1 早退路径零化/禁用（`enabled=false`）；`rope.cpp` 两重载先 `require_scaling`（仅收 D256/R64 与 3 轴 MRoPE）再派发；`rope.cu` 启用时仅走 `launch_generic`；`rope.cuh:181-243` 启用分支 = 表驱动逆频率 + FP64 角 + 2π `nearbyint` 约简 + ×attention_factor，禁用分支逐位等同改动前（`sincosf(position * powf(theta, exponent))`）；DFlash 固定 sincos 快路径在启用时被 `require_scaling` 拒达不成；`draft.cpp:211,457`、`vision.cpp:371` 传默认 `nullptr` ✓；`parameters.cpp:260-263` 仅 factor>1 建表；`model_instance.cpp` `validate_options`（L19，`validate_yarn` L51）经 `construct_model` L159 覆盖 CLI/serve/public Engine；CausalScoring 的 `normalize_engine_options` 重置除 yarn 外全部项 —— 产品一致（打分窗同旋转；scoring 下 context_cache 关闭 → 无跨 factor KV 恢复）。
- **3.8 存储布局**：`paged_kv_storage.h:58-97` RK4V4E8 对称 `{U8,128,FP16,1}` = 130 B/向量 = 260 B/tok/head（对照 Int8Group64 528、BF16 1024）；全部消费者（startup、decoder_state、small_t、causal_softmax_attention、kv_cache_append、test_kv_capacity）走 `paged_kv_storage_layout`，无旁路常量。
- **3.9 文档**：YaRN 五处（cli.md L283-309、serving.md L1002-1027、README L30-37、sm89.md L37、AGENTS.md L33）齐全一致；README 矩阵缺 E8 行已修（1.4）。
- **路由接线**：`causal_softmax_attention.cpp:367` RK4V4E8 → `prompt_limit = width<=8 ? 0 : 256`（与 Int8Group64 同）；四个入口短路（small_t.cu:394/441、prompt.cu:79/112）+ append 派发（kv_cache_append.cpp:205）✓。sm_120a：构建树仅含 `e8_sm120a_stubs.cpp.o`、零 E8 设备目标，6 个入口桩抛 `std::invalid_argument("NInfer sm_120a build: the <route> route is an Ada (sm_89) capability…")`，桩签名链接干净。
- **测试 oracle 忠实度**：`tests/ops/softmax_attention/causal_cache.cpp` `e8_project_8d_host`（L671-725）= 标量核逐语句转录（nearbyint==rintf、同最坏维/方向/平局规则、`dist_d8 <= dist_coset1`）；host `normalized_hadamard_d256`（L548-560 浮点蝶式 ×0x1p-4f，另有 popcount 矩阵变体 L562）；`encode_e8_rotated_row`（L737+）：旋转后 absmax → 有界 scale → f16 位 → 反演，K 投影后夹 [-8,7]、V 夹 [-7,7]，nibble 打包（偶 i→低半字节）与设备一致。**逐位转录性已由 §4.2 B 项 1e6 块锁死**。

## 4. §4.1 / §4.2 零 GPU 验证记录

### 4.1 CPU 测试（BUILD_TESTING=OFF 下不在 ctest，独立编译运行）
| 测试 | 编译配方 | 结果 |
|---|---|---|
| `test_yarn_parameters` | g++ -O2 -std=c++20，`tests/test_yarn_parameters.cpp` + `src/ops/wrapper/rope.cpp` + `libninfer_ops.a`/`libninfer_core.a`/`libninfer_runtime_support.a` + `-lcudart` + `-DNINFER_SOURCE_DIR` | **OK**（`OK YaRN CPU reference and cache contract`：YaRN 系数表 == `tests/data/qwen3_8_yarn_hf.json` HuggingFace 参考；`validate_yarn` 契约）——审计机独立编译运行 OK；**云机** `build-sm89-full/tests/ninfer_yarn_parameters_test` 亦 OK（CUDA 12.8 构建产物） |
| `test_rope_scaling_options` | 同上 + `apps/cli/options.cpp` + `src/serve/serve_options.cpp` + `src/product/logging/logging.cpp` | **OK**（`OK CLI/server YaRN options`：CLI 与 serve 双侧解析、非法 factor 拒绝） |
| `e8_root_codec`（`ninfer_e8_root_codec_test`） | 云机 `build-sm89-full` 目标 | **OK**（`OK e8_root_codec`）——E8 编解码 CPU 侧契约 |
| 系列相关 GPU 二进制（`test_kv_cache`、`test_kv_capacity`、`tests/ops/softmax_attention`（含 E8 profile）、`tests/ops/test_rope_yarn`） | 审计机未编译（BUILD_TESTING=OFF）；云机已编译（关键 15 目标集） | 审计机 **未执行（无 GPU）**——本机 `cudaInitialize` 会 abort（rc 134），属基础设施限制非失败；云机执行见 §5.3 |

### 4.2 E8 零 GPU 数值等价性（独立工具，`/tmp/e8audit/e8_host_check.cpp`）
工具把 `e8_lattice.cuh` 的标量段（`__device__`/`__forceinline__` 置空）在 host 编译，另配一个**独立**的最近格点穷举 oracle（对 D8 与 D8+0.5 各自枚举「逐维取整 ± 单坐标 ±1 翻转」的全部候选——最近格点与逐维取整至多差一维，盒子穷尽且不依赖被审代码的最坏维规则）：

| 项 | 规模 | 结果 |
|---|---|---|
| A 标量语义 | 20000 个 8 维 Sylvester 旋转高斯块（缩放进 ±7 code 域，生产分布形状） | 格点违例 0、非最近点 0、与独立穷举逐位一致 20000/20000 → `e8_project_8d_fast` 是真最近格点算法 |
| B host oracle 忠实度 | 1e6 同分布块 | 设备头文件 vs 测试树 `e8_project_8d_host`：0 个逐位分歧 |
| C 根向量 | 240 个 E8 根（112 整型 ±1,±1 + 128 半整型（负半维数偶））+ 240 个 2× 根 | 全部投影不动 |
| D 边界 | 8×5 轴向量、7 个均匀向量、(1/4)^8 与 (3/4)^8 精确 D8/coset 平分线、e1 三方平局 | 全过：平分线按文档 `<=` 规则判 D8（(1/4)^8→0^8、(3/4)^8→1^8）；e1 落在最近 D8 点 |

**含义**：标量 codec 语义与测试 oracle 忠实度已在 CPU 锁死。GPU 侧（§4.3）任何 kernel-vs-oracle 失败可定位在 warp 蝶式 / 流水线（旋转→scale→打包→分档→解码→MMA）/ 占用率，而不再是「codec 数学本身」。

## 5. §4.3 GPU 验收移交（按交接原文，用户独立执行）

### 5.1 移交清单（原文保留）
1. **kernel vs oracle 数值测试**：`tests/ops/softmax_attention`（E8 profile：fused/decode/prompt/batch 扫描）、`test_kv_cache`、`test_kv_capacity` 在 GPU 上跑；E8 判据 `kAttentionE8Criterion`（`causal_cache.cpp:86`，现沿用 NVFP4 宽阈值 1.5e-2 / 5.0e-3 / 1.1e-2）拿到真实数值后再评是否收紧。
2. **长上下文质量**：YaRN 端到端经 `tools/bench/verify_yarn_context.py`；27B 的真实层数/头数取你机器上的 artifact config（本仓库只有合成测试形状）。
3. **sm_120a 桩行为**：5090 上 6 个 E8 入口桩应抛 `std::invalid_argument`（route 不支持）而非段错误。
4. **prompt-E8 占用率**：`__maxnreg__(120)` + smem 91136 B 在 Ada 上的实测占用率（本报告 2.4）。
5. （建议）2.1 的 i8 V 解码欠载若 GPU oracle 复现异常，走独立修复 + 复验。

### 5.2 CPU 侧已消除 / 残余风险
**已消除**：
- E8 标量投影的最近性语义（独立穷举 oracle，A 项）；
- 测试 host oracle 的逐位忠实度（B 项）——GPU 测试的 oracle 一侧不再可能是误差源；
- 两处真实数值 bug（1.1 prompt-E8 V 解码、1.2 small-T-E8 scale 偏移）——修复后这两条路线在 GPU 上的数值测试才有通过可能；
- sm_89 链接期未定义符号（1.3）——产品链接配置下 E8 append 入口可解析；
- YaRN 系数表对 HuggingFace 参考的逐值一致与 CLI/serve 双侧解析契约（4.1）。

**残余（必须 GPU）**：
- warp 蝶式投影（`e8_project_8d_warp_single`）== 标量投影 —— **从未执行过**，零 GPU 工具只覆盖了标量版；
- 完整流水线（D256 旋转、per-token scale、nibble 打包/解包、cp_async 分档、QK int8 MMA + PV f16 MMA）与 oracle 的端到端数值；
- 长上下文真实质量（YaRN factor 2–4）与 E8 判据收紧决策；
- sm_120a 桩的运行时行为与 prompt-E8 占用率。

### 5.3 首次 GPU 执行记录（云机 4080S，2026-09-25，CUDA 12.8 构建）

`./tests/ninfer_softmax_attention_test`（模型共存下 <1GB 单条执行）：**仅 `rk4v4-e8` 路线失败**，同机 k8v4/nvfp4 oracle 质量行通过（环境/工具链健全）。症状：

1. `cache-v-code` / `cache-v-scale: exact mismatch at index 0`（fused 用例，多 mapping/T 组合）——内联 fill 写入与 host oracle 从首字节起即不一致；
2. 注意力输出整体垃圾（`actual=1.83594 reference=-0.949615`，远超 E8 判据 1.5e-2 量级）；
3. `q unchanged: exact mismatch at index 5632`——**设备端越界写落入 Q 缓冲**；
4. `cudaDeviceSynchronize: illegal memory access` → abort（rc 134，此处是真 OOB，非「无 GPU」语义）；
5. **cached（只读）变体同样失败**：设备缓存由测试用 host oracle 正确内容填充后 E8 解码仍出垃圾 → 解码路径存在独立于 append 的缺陷。

**逻辑结论**：解码路径必坏 + fused 内联 fill 也坏（或同源）；E8 设备路径此前从未在任何真机端到端跑过（上游在 5090/sm_120a 调优，E8 在该架构是抛异常桩）。

**静态排除（前审计者已做，勿重查）**：设备 warp Sylvester 旋转与 host 测试矩阵逐元素同矩阵（比特序无关性）；4 键 scale 组与 32/64 tile 均不跨 64 页界；V 码 16B cp_async 对齐；V loader warp 内 key_l 均匀（VLoaderThreads 为 32 倍数）；E8 的 QK 片段机制（`k_b16=(bf16*)k_i8` + `ldmatrix_x2` + `mma_s8`）与本机实测通过的 i8 路线逐指令同构；append 行函数打包/寻址/scale 语义与 oracle 一致。

**GPU 调试移交**：本审计者沙箱无法安全占用云机 GPU（会危及主人驻留的 27B 模型进程），按主人指示停止在首次执行取证处。隔离程序（探针 1 fill → 探针 2 decode → 探针 3 memcheck）+ 决策树 + 嫌疑排序已完整写入 `e8-yarn-audit-relay.md` §4.3，探针源码在 `docs/maintainer/e8-gpu-debug-probe.cpp`（云机副本 `/root/probe_e8_fill.cpp`），完整失败日志在云机 `/root/e8_test_full.log`。

## 6. 构建状态（交付门禁）

审计机（无 GPU，CUDA 13.2，pip cmake 4.3.2 + ninja；libcurl 8.19.0 自建于 `/tmp/e8audit/curl-install`，`NINFER_ENABLE_VISION=OFF`）：
- `build-e8-sm89` / `build-e8-sm120a`（TESTING=OFF）：rc=0（对象+库；含 `e8_launch.cu.o` 与重编译的 `small_t_e8.cu`/`prompt_e8.cu`；120a 侧 Sm120Compat 剔除 E8 attention .cu 仅留 stub）。
- `build-sm120a-full`（TESTING=ON, APPS=OFF, LINK_SMOKE=ON, VISION=OFF）：**rc=0，262 步**——全部测试二进制 + `ninfer-link-smoke` 链接成功；`ninfer_yarn_parameters_test` 在该树独立运行 OK（首次全量测试构建暴露并修复了 §1.5 的 json 接线）。

云机（4080S，CUDA 12.8，miniconda pip cmake 4.4.3 + ninja 1.13.2，`/root/curl-install`）：
- `build-sm89-full`（TESTING=ON, APPS=ON, LINK_SMOKE=ON, VISION=OFF，30GB 根盘墙下的 15 关键目标集）：**rc=0，449/449**；675 个 CUDA 编译单元在 CUDA 12.8 下全绿（无工具链代差问题）。
- 三个免 GPU 测试（§4.1 表）在该树运行全 OK。

未 push origin（`master` ahead 10，转移可 `git bundle`）；交接文档 `e8-yarn-audit-handoff.md`、本报告、`e8-yarn-audit-relay.md`、`e8-gpu-debug-probe.cpp` 均保持 untracked。
