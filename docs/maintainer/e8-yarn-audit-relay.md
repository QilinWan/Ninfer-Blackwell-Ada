# E8/YaRN 审计接力文档（给下一个接手的调试者——可能是另一个 AI 模型）

> 前一位审计者（agent）已完成：代码审计 + 零 GPU 数值验证 + 两台机器上"编译与链接"全部关门 + 5 个缺陷修复，
> **并且首次在 Ada GPU（云机 4080S）上运行了 E8 测试——它失败了**（完整证据见 §4.3）。
> **你的任务 = 按 §4.3 的探针程序把 E8 设备路径的 bug 隔离出来并修复**，然后全量 ctest，最后主人做引擎验证 + 生产部署 + 性能测试。
> 深度证据（每个结论的 file:line）在同目录 `e8-yarn-audit-report.md`；任务书在 `e8-yarn-audit-handoff.md`。本文档自包含，不需要先读它们也能开工。

## 0. TL;DR —— 你先做什么

1. **先与主人确认 GPU 可用性**：云机 4080S 上跑着主人的 27B 模型（~29.5/32GB）。单条测试/探针 <1GB 可共存；`compute-sanitizer` 需要独占 GPU，必须先让主人释放。
2. 云机（`ssh -p 50730 root@connect.westc.seetacloud.com`，见 §3 凭据）上 sm_89 构建树 `/root/ninfer-e8/build-sm89-full` **已构建且绿**（15 个关键目标，rc=0），三个免 GPU 测试**已验证通过**（e8_root_codec / yarn_parameters / rope_scaling_options，均 "OK"）。
3. **E8 GPU 数值测试已确认失败**（仅 `rk4v4-e8` 路线；k8v4/nvfp4 同机通过 = 环境健全）。前一位审计者做了大量静态排除（§4.3.3），你从这里接手：跑 §4.3.4 的**探针 1（fill 隔离）**，按 §4.3.5 的决策树走，修复后重跑 §4.2/§4.3 全套验证。
4. E8 路线绿了 → 主人释放显存后跑全量 ctest（§4.4）；5060 Ti 侧的 sm_120a 架构级测试由主人在本地机器跑（§4.5，命令已备好）。
5. 拿不准就停下、带证据回来（§5 分诊树），不要猜测式改代码。

## 1. 项目与审计背景（30 秒版）

NInfer 是 from-scratch 的 C++/CUDA 单 GPU 推理引擎（Qwen3_5 稠密/MoE），本 fork 在上游 `sm_120a`（Blackwell，上游在 5090 调优）之外增加了 **Ada `sm_89`** 构建。本次审计覆盖两个功能层：

- **RK4V4E8**：E8（Conway–Sloane 格）4-bit K/V 量化缓存路线。**仅 Ada（sm_89）有设备实现**；sm_120a 构建里这 6 个入口是显式抛 `std::invalid_argument` 的桩（`src/ops/sm120a/e8_sm120a_stubs.cpp`）。
  编码契约：D256 Hadamard 旋转（×1/16）→ 逐 token FP16 scale = absmax/7 → K 走 E8 投影 + `rintf` 截断 [-8,7]（半余格坍缩是**文档化的存储语义**，不是 bug）→ V 走 `__float2int_rn` 截断 [-7,7]；每字节两个 4-bit 码，低半字节 = 偶数维（`src/ops/kv_cache/e8/e8_i4_code.cuh:91-94`）。
- **静态 YaRN**：上下文外推因子 1..4，参考窗口 262144，双架构同实现。因子 1 必须与原生 RoPE 逐位一致。

关键判定：**CPU 侧已全部锁死**（E8 标量投影被独立穷举 oracle 证明是真最近格点算法、host 测试 oracle 逐位忠实、YaRN 系数表 == HF 参考、rope 派发/约简/禁用分支逐位不变）。因此 GPU 测试若失败，嫌疑只剩：warp 蝶式归约 vs 标量的 FP 加序差、cp_async 流水线、占用率（`__maxnreg__(120)` + 91136 B smem）。

## 2. 前一位审计者修复的 5 个缺陷（`master` 领先 origin 10 个提交，**勿 push**）

| 提交 | 缺陷 | 位置 | 后果（若未修） |
|---|---|---|---|
| `be84d07f` | prompt-E8 的 V 解码只取了每字节低半字节（`p[2*i]`/`p[2*i+1]`），丢掉全部奇数维 V + 4 字节栈越界 | `src/ops/softmax_attention/dense/causal_cache/prompt_e8.cuh`（`causal_prompt_e8_dequant_f16x8`） | 整条 prompt-E8 路线 V 全乱 |
| `61730546` | small-T-E8 的 scale 暂存用 `(4*q4)&page_mask` 丢了 tile 基址 | `src/ops/softmax_attention/dense/causal_cache/small_t_e8.cuh`（`issue_kv_tile`） | Bc=32 tile 在页内偏移 32 处取错 per-key scale |
| `a2851c04` | `e8_launch.cu` 未入 sm_89 源列表 | `src/ops/kv_cache/sources.cmake` | sm_89 库缺 `kv_cache_append_e8_launch*` 符号（链接期才炸） |
| `b894e7ef` | README 支持矩阵漏列 RK4V4E8 | `README.md:28` | 文档 |
| `5c845951` | `ninfer_yarn_parameters_test` 漏链 `ninfer::json`（直接 include 了 `<nlohmann/json.hpp>`） | `tests/cmake/NinferTests.cmake:27` | 首个全量测试构建必炸（`nlohmann/json.hpp: No such file or directory`） |

**报告级发现（未修，待主人决策，见报告 §2）**：i8（INT8 路线，前序上游代码，本次未动）small-T 与 prompt 的 V 解码 4 字节欠载（`small_t_i8.cuh:541-549`、`prompt_i8.cuh:51-66`，`load_vec<int2>` 只读 4B 却解 8 码，`c[4..7]` 读的是栈）——会污染 INT8 prompt 路线数值。**INT8 测试若出超差先怀疑这里**；修它要独立走一轮 GPU 复验。

## 3. 机器地图与通道

| 机器 | 硬件/工具链 | 角色 |
|---|---|---|
| **审计机**（前 agent 所在，无 GPU） | 128C/503GB，CUDA 13.2，pip cmake 4.3.2，g++ 11.4，ninja | 双架构 CI 门禁（对象+库全绿）+ 零 GPU 数值工具 + 全量 sm_120a 测试树编译/链接（`build-sm120a-full`，rc=0）。**注意：它没有 GPU，测试二进制在它那跑必 rc=134（cudaInitialize 失败）——记"未执行（无 GPU）"，不是失败** |
| **云机**（autodl 容器，**RTX 4080 SUPER 32GB = Ada sm_89**） | CUDA 12.8，驱动 595.71（支持 CUDA 13 工具链），miniconda py3.12（pip 已装 cmake 4.4.3 + ninja 1.13.2），FFmpeg 5（vision 需 ≥60 → 一律 `NINFER_ENABLE_VISION=OFF`），**根盘仅 30GB** | sm_89 全量编译 + **E8 原生数值验证主场**（4080S 就是 Ada）。源码在 `/root/ninfer-e8/`，libcurl 在 `/root/curl-install/` |
| **主人本地** | RTX 5060 Ti 16GB（sm_120a），工具链自备 | sm_120a 架构级测试（120a 桩抛错、NVFP4/K8V4 原生路线）。16GB 够全部合成测试；27B 模型级受量化/上下文预算限制 |

**通道**：审计机 → 云机走 SSH 密钥（审计机 `/tmp/e8audit/ssh/id_ed25519`，公钥已装云机 `/root/.ssh/authorized_keys`）：
`rsync -a -e "ssh -i /tmp/e8audit/ssh/id_ed25519 -p 50730" ./ root@connect.westc.seetacloud.com:/root/ninfer-e8/`
云机密码（备用）：`6d+6VffmoXT1`。⚠ 某些沙箱会拦 pty → sshpass 失效；paramiko 方案见审计机 `/tmp/e8audit/cssh.py`。

**两条铁律**：
1. **云机 GPU 上跑着主人的 27B 模型（~29.5/32GB 显存，pid 见 `nvidia-smi`）——永不 kill、永不干扰**。只读 `nvidia-smi` 可以；GPU 测试单条 <1GB 可与其共存；全量 ctest 等主人释放显存。
2. **两台机器都不 push origin**。`master` 领先 origin 10 个提交，推送时机主人定（可 `git bundle` 转移）。

## 4. 分步操作（含已验证的命令）

### 4.1 云机：sm_89 构建（关键目标集）

⚠ **磁盘坑（实测）**：完整构建树 ≈27GB 会把 30GB 根盘写满（`ld: No space left on device`，编译单元本身不受影响）；`/root/autodl-fs` 网络卷**主人规定禁用**。解法 = 只构建验证链需要的 15 个目标（对象+库 ~11GB + 关键二进制 ~5GB，峰值 ~16GB，放得下）：

```bash
ssh -p 50730 root@connect.westc.seetacloud.com
cd /root/ninfer-e8
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH \
       PKG_CONFIG_PATH=/root/curl-install/lib/pkgconfig CMAKE_PREFIX_PATH=/root/curl-install
cmake -S . -B build-sm89-full -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_MAKE_PROGRAM=/root/miniconda3/bin/ninja \
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_LINK_SMOKE=ON -DNINFER_ENABLE_VISION=OFF
cmake --build build-sm89-full -j64 --target \
  ninfer ninfer-serve ninfer-link-smoke \
  ninfer_e8_root_codec_test ninfer_softmax_attention_test \
  ninfer_rope_test ninfer_rope_yarn_test ninfer_rope_scaling_options_test \
  ninfer_rmsnorm_rope_test ninfer_kv_cache_test ninfer_kv_cache_append_test \
  ninfer_kv_capacity_test ninfer_context_kv_materialize_test \
  ninfer_yarn_parameters_test ninfer_sliding_window_attention_test
# 预期：rc=0（前审计者已实测编译 675 单元全绿；链接受磁盘限制按上述目标集进行）
```

坑位备忘：远程 shell PATH 不含 miniconda bin 与 cuda（故全显式）；cmake 自动探测不到编译器（故显式 cc/c++）；`ninfer-link-smoke` 目标只在 `NINFER_BUILD_LINK_SMOKE=ON` 时存在；libcurl 依赖若 `/root/curl-install` 丢失，源码构建配方在审计机 `/tmp/e8audit/curl-8.19.0`（`./configure --prefix=... --without-ssl --without-zlib --without-libpsl --disable-manual --disable-docs`）。

### 4.2 云机：免 GPU 的三个测试（✅ 前审计者已在云机跑过，全 "OK"，rc=0）

```bash
cd /root/ninfer-e8/build-sm89-full/tests
./ninfer_e8_root_codec_test      # "OK e8_root_codec"
./ninfer_yarn_parameters_test    # "OK YaRN CPU reference and cache contract"
./ninfer_rope_scaling_options_test  # "OK CLI/server YaRN options"
```

### 4.3 云机 4080S：E8 GPU 测试**已失败** —— 你从这里接手调试

#### 4.3.1 前审计者实测到的故障现象（2026-09-25，4080S，CUDA 12.8 构建）

`./tests/ninfer_softmax_attention_test` 中**仅 `rk4v4-e8` 路线失败**（同机 k8v4/nvfp4 的 oracle 质量行通过 → 环境/工具链健全）。症状链（完整日志在云机 `/root/e8_test_full.log`）：

| 症状 | 含义 |
|---|---|
| `cache-v-code: exact mismatch at index 0` / `cache-v-scale: exact mismatch at index 0`（多个 mapping/T 组合） | fused 内核内联 fill 写入的 V 码/标度与 host oracle（测试自洽编码）从第 0 个字节起就不一致 |
| `reduction criterion failed ... actual=1.83 reference=-0.95`（差值巨大，不是 epsilon 级） | 注意力输出整体垃圾，不是量化误差量级 |
| `q unchanged: exact mismatch at index 5632`（d256-h24：= head 22, dim 0） | **设备端有越界写入**，落在了 Q 缓冲区 |
| `cudaDeviceSynchronize: illegal memory access` → 进程 abort（rc=134） | E8 路径存在真实的 OOB 访问 |
| **cached（只读）变体也失败**：设备缓存由测试用 host oracle 正确内容填充后，E8 解码仍出垃圾输出 | **small_t_e8 的解码路径有独立于 append 的 bug**；fused 内联 fill 另有其错（或同源） |

#### 4.3.2 关键逻辑结论（前审计者已推完，直接用）

1. 测试 a2（fused）= 内联 fill + 解码 + 输出；a3（cached）= 正确缓存 + 纯解码。a3 失败 ⇒ **解码路径必坏**；a2 的码/标度错位 ⇒ **fused 内联 fill 也坏**（或两者共享同一处暂存/寻址代码）。
2. 前审计者已修的 5 处都在 CPU 可锁的层面；E8 设备路径**从未在任何真机上端到端跑过**（上游在 5090/sm_120a 调优，E8 在那里是抛异常桩）——本次是 E8 首次上 GPU。

#### 4.3.3 前审计者已静态排除的嫌疑（不必重查）

- **旋转矩阵**：设备 warp Sylvester 蝶式（`hadamard_d256.cuh:46-65`，lane l 持 dim l+32r）与测试 host 矩阵（`causal_cache.cpp:548-560` 及 `:562-577` 的 `(-1)^{popcount(row&col)}·2⁻⁴`）**逐元素同矩阵**（比特序无关性），排除"不同合法 Hadamard"。
- **scale 4 键组跨页**：key ≡ 0 (mod 4) 且 tile 32 对齐 ⇒ 4 键组永不跨 64 页界（`small_t_e8.cuh` issue_kv_tile，前修复 `61730546` 已修 tile 基址）。
- **tile 跨页**：Bc=32/64 的 tile 起点 32 对齐 ⇒ 单 tile 恒在单物理页内。
- **V 码 16B cp_async 对齐**、**V loader warp 内 key_l 均匀性**（VLoaderThreads 是 32 的倍数）：均成立。
- **QK 片段机制**：E8 与 i8（本硬件实测通过的路线）**逐指令同构**（`k_b16=(bf16*)k_i8` + `ldmatrix_x2` + `mma_s8`；`small_t_i8.cuh:93,426-429` vs `small_t_e8.cuh` 对应段）。
- **append 行函数**（`e8_i4_code.cuh:106-176`）：打包（低半字节=偶维）、寻址（`code_index(d/2, page_offset)`）、scale 写入与测试 oracle 语义一致。
- **零 GPU 数值工具**（审计机 `/tmp/e8audit/e8_host_check.cpp`）：E8 **标量**投影 = 真最近格点（20000 块独立穷举）、host oracle 逐位忠实（1e6 块）、240 根向量 + 边界平局全过。

**仍未被任何执行证据覆盖的三块**（你的探针要打这里）：
① `e8_project_8d_warp_single`（**warp 蝶式版**投影，`e8_lattice.cuh`）——从未执行过，工具只验证过标量版；
② `small_t_e8` 解码暂存/指针计算（V 码 `key_l*CodeExtent + (d>>1)`、k_scale_s 广播）在真机上的行为；
③ fused 内联 fill 的 tile 参数传递（`small_t_e8.cuh:207` 一带）+ Q 越界写入源。

#### 4.3.4 探针程序（按序执行；每条都只需 <1GB 显存，与模型可共存）

**探针 1 —— fill 隔离**（源码已就位：仓库 `docs/maintainer/e8-gpu-debug-probe.cpp`，云机 `/root/probe_e8_fill.cpp`）：
单独走公开 API `kv_cache_append` 填 1 个 key（kv4），解码设备平面，与 host oracle（同矩阵旋转 + absmax/7 + E8 投影/rint + 打包）逐维对比。

```bash
cd /root/ninfer-e8 && B=build-sm89-full
g++ -O2 -std=c++20 /root/probe_e8_fill.cpp -o /root/probe_e8_fill \
  -I include -I src -I /usr/local/cuda/include \
  $B/src/ops/libninfer_ops.a $B/src/core/libninfer_core.a \
  $B/src/runtime/libninfer_runtime_support.a \
  -L/usr/local/cuda/lib64 -lcudart -lpthread
/root/probe_e8_fill
```

**探针 2 —— decode 隔离**（探针 1 PASS 时做）：用 standalone append 把缓存填成**正确内容**，然后跑 cached（只读）E8 注意力 T=1 vs 理想参考；或直接跑测试二进制看 a3 用例（`causal_softmax_attention_cached ... rk4v4-e8`）。若解码仍垃圾 ⇒ bug 在 `small_t_e8.cuh` 的 V 解码暂存 / k_scale 广播 / PV MMA 片段，逐段对照 i8 的对应实现（i8 在本硬件通过 = 结构模板）。

**探针 3 —— 越界定位**（需要主人先释放 GPU 独占）：
```bash
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck \
  /root/ninfer-e8/build-sm89-full/tests/ninfer_softmax_attention_test
```
memcheck 会直接指出 OOB 访问的 kernel/行号（预计指向 fused E8 路径的某处写）。

#### 4.3.5 决策树

```
探针1 FAIL（fill 就错） → bug 在 kv_cache_append_full_e8_row 链：
    优先查 e8_project_8d_warp_single（warp 蝶式投影，唯一未执行过的 E8 数学）
    → 修复 → 探针1 PASS → 回到探针2
探针1 PASS，探针2 FAIL → bug 在 small_t_e8 解码（对照 i8 逐段 diff）
两者 PASS 但测试仍红 → fused 内联 fill 专属 bug（small_t_e8.cuh:207 内联 fill 调用参数）
+ Q 越界：无论哪条路径，最后都用探针3（memcheck）钉死 OOB 写源
```

每修一处 = 一个 Conventional Commit（`fix(e8): ...`），重跑 §4.2 + §4.3.1 的测试二进制确认。

**修复全绿后**：跑 §4.2 的 YaRN/rope/KV 测试集（`ninfer_rope_yarn_test`、`ninfer_rope_test`、`ninfer_kv_cache_test` 等，命令同 §4.3 原表）；E8 判据现用 NVFP4 宽阈值（1.5e-2 / 5.0e-3 / 1.1e-2，`causal_cache.cpp:86`）——通过标准 = 该判据下全绿，阈值收紧是主人的后续决策。

### 4.4 主人释放显存后：全量 ctest（磁盘管理配方）

```bash
# 剩余 ~55 个未构建的测试二进制：小批量 --target 分批链接，每批后清对象腾空间：
cmake --build build-sm89-full -j64 --target <下一批目标>
find build-sm89-full -name "*.o" -delete        # 可执行文件不受影响；对象删掉后该批目标不可再增量重链
ctest --test-dir build-sm89-full --output-on-failure
```

### 4.5 主人本地 5060 Ti：sm_120a 架构级（主人自跑，命令备查）

```bash
cmake -S <源码> -B build-sm120a-full -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DNINFER_ENABLE_VISION=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc   # 本机实际路径；libcurl≥7.85 缺失时同 §4.1 配方
cmake --build build-sm120a-full -j
ctest --test-dir build-sm120a-full --output-on-failure
# 预期：NVFP4/K8V4/INT8/FP8/BF16+YaRN 全绿；E8 六入口报 std::invalid_argument（"sm_120a build: ... Ada (sm_89) capability"）——这是桩的正确行为
```

### 4.6 主人收尾（主人自做）

引擎验证 → 生产部署一次（功能可用确认）→ 性能测试。性能数字架构归属：sm_89 出自 4080S（云机），sm_120a 出自 5060 Ti，**均不可与上游 5090 调优数字直接对比**（SM 数/带宽差异），跨卡只比相对比例。

## 5. 故障分诊树

| 症状 | 第一嫌疑 | 动作 |
|---|---|---|
| E8 注意力测试超差（阈值内失败） | warp 蝶式 vs 标量 FP 加序差 / cp_async 流水线 / 占用率（CPU 侧已锁死，见 §1） | 收集：失败 shape、超差维分布（是否集中在奇数维 → 疑 V 解码回归）、`tests/` 里 i8/e8 对照（审计机留过 diff）；**不要先改代码** |
| INT8 路线 prompt 测试超差 | 前序 i8 V 解码 4B 欠载（§2 末段） | 先确认是否 i8 既有问题再谈新代码 |
| E8 测试数值对但 12.0 卡上跑不了 | 正常——E8 是 Ada 专属，120a 上应抛桩异常 | 若 12.0 卡上要跑 E8 数值：sm_89 构建加 `-gencode arch=compute_89,code=compute_89` 内嵌 PTX，驱动前向 JIT（指令集 `ldmatrix.b16`/`mma.m16n8k16.f16`/`mma.m16n8k32.s8`/cp.async/shfl 在 12.0 全可得，`src/ops/common/mma.cuh:8-53`；数值可转移，性能仅供参考） |
| 链接期 undefined reference | 源列表遗漏（`a2851c04` 同类） | `nm -C --defined-only 库.a \| grep 符号名` 定位，查 `sources.cmake` |
| `ld: No space left on device` | 根盘 30GB 墙 | 按 §4.1 目标集；或 `find build-sm89-full -name '*.o' -delete` 腾空间 |
| 测试 rc=134 | 该机器无 GPU（cudaInitialize 失败） | 记"未执行（无 GPU）"，不是失败 |
| 云机 OOM/测试挂死 | 与主人模型抢显存 | `nvidia-smi` 确认余量；等主人释放 |
| CUDA 12.8 报 13.x API 错误 | 工具链代差（审计机 13.2 编过） | 驱动 595.71 支持 CUDA 13 工具链，可换 13.x 工具链重试 |

## 6. 不变量（任何修复都必须保持）

- 每个修复一层一个 Conventional Commit；不 push origin。
- BFloat16 / Int8Group64 既有路径零 diff（YaRN 因子 1 与原生 RoPE 逐位一致）。
- E8 = sm_89 专属设备路线 + sm_120a 显式桩；YaRN = 双架构同流。
- `docs/maintainer/e8-yarn-audit-handoff.md`、`e8-yarn-audit-report.md` 与本文件均为**未跟踪文件，保持原样**。
- 遗留决策项（不阻塞你，记录给主人）：i8 欠载修不修；E8 判据收紧与否（待真实数值）；`yarn_cache_binding` 无消费方（防御性保留）；`e8_root_codec.cuh` 未被引用（预期，E8 走 `e8_lattice.cuh` 投影路径）。

## 7. 工件位置速查

| 工件 | 位置 |
|---|---|
| 源码（含前序 + 5 个审计修复，master 领先 origin 10 提交） | 审计机 `ninfer-blackwell-ada/`；云机 `/root/ninfer-e8/`（与审计机逐字节同步过） |
| **探针 1（fill 隔离）源码** | 仓库 `docs/maintainer/e8-gpu-debug-probe.cpp`（未跟踪）；云机副本 `/root/probe_e8_fill.cpp`（编译配方见 §4.3.4） |
| **E8 GPU 失败完整日志** | 云机 `/root/e8_test_full.log`（4080S 实测，含全部症状行） |
| 零 GPU E8 数值工具（20000 块穷举 + 1e6 块逐位对照 + 240 根向量 + 边界平局，ALL PASS） | 审计机 `/tmp/e8audit/e8_host_check.cpp`（配方：g++ -O2 -I/usr/local/cuda/include + 截断的 `e8_lattice.cuh` 标量段） |
| i8 欠载 host 复现（期望 1520 实得 1620 → 读栈） | 审计机 `/tmp/e8audit/loadvec_host.c` |
| i8/e8 对照 diff | 审计机 `/tmp/diff_small_t_i8_e8.txt`、`/tmp/diff_prompt_i8_e8.txt` |
| 构建日志 | 审计机 `/tmp/build-*.log`、`/tmp/cfg-*.log`（sm120a-full rc=0 在 `/tmp/build-sm120a-full2.log`）；云机 `/tmp/build-sm89d.log`（rc=0）、`/tmp/cfg-sm89d.log` |
| 审计结论全文（含 §3 逐条审计 + §4 验证记录 + §5 GPU 移交） | `docs/maintainer/e8-yarn-audit-report.md`（未跟踪） |
