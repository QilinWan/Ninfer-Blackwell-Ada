# S11 开发计划书 — tp 2 多模态（merged-embedding handoff）

> 读者：在 `/data/s11` 工作区里接手本任务的模型/人。
> 基线：本仓库 `blackwell-only` 分支（`b55da868`，或更新的尖端）。任务 = 让 `--tp 2` 的
> **图像/视频 prompt** 真正跑起来，而不是被 `require_single_device("multimodal prefill")` 拒绝。
> 先行阅读（都在克隆树里）：`research/README.md`（项目状态与教训）、`research/s7-seed-5060ti.md`
> （实测口径）、`research/replay-log-20261016.md` §82–§84（本轮数据）。

---

## 0. 一句话目标

`ninfer-serve --tp 2 --vision --vision-device 0` 收到带图/带视频的 chat 请求时返回 **200**，
输出与同一制品的单卡（tp 1）参考一致；纯文本 tp 2 行为与性能**零回归**。

**不要**顺手做别的事：不改 Ada（已删除）、不换量化、不改 CLI 契约、不动 KV 布局。

---

## 1. 现状：为什么现在会失效

### 1.1 失败点（代码锚点，行号以 `95884c20` 为参考，克隆后以符号 grep 为准）

| 位置 | 内容 |
|---|---|
| `src/models/qwen3_5/execution/text.cpp` `require_single_device(...)`（≈:1555-1560） | 抛 `invalid_argument("<feature> is not implemented at tensor-parallel width 2")` |
| 同文件 prefill 的调用点（≈:3073 / :3089） | 图像 / 视频 prefill 入口 |
| `src/models/qwen3_5/program/speculative/mtp.cpp` `mtp_bridge_and_propose` 的 tp2 分支 | 抛 `the MTP bridge takes no composed embedding at tensor-parallel width 2` |
| `docs/cli.md` "Two GPUs" 段、`docs/maintainer/tensor-parallel.md` §8 | 契约里已写明"media 在 tp 2 被拒" |

复核命令：

```bash
cd /data/s11/tree
grep -rn 'require_single_device' src/ | head
grep -rn 'multimodal prefill' src/ tests/ docs/ | head
grep -rn 'composed embedding' src/ | head
grep -rn 'Vision at tp 2' docs/maintainer/tensor-parallel.md
```

### 1.2 根因（不是硬件，是缺一条数据通路）

- tp 2 的每一层，**两个 rank 的输入激活必须逐位一致**：两卡算同一批行，只把 attention 头 /
  GDN 头 / MLP 列切开，残差流是复制的。
- 纯文本：token embedding 表每卡一份（`--embedding-host` 只是换成 pinned host 单份），
  两卡各自 embed 同一批 id，天然一致。
- 图像：图像 token 的输入向量来自 **rank 0 的视觉塔**（`--vision-device 0`），rank 1
  **没有塔权重、也没有接收这些行的通道**；rank 1 只能拿占位符的文本 embedding 去算 →
  残差流分叉。所以引擎选择响亮拒绝，而不是静默算错。
- 同理，MTP 桥在 tp 2 也拒绝"composed embedding"（投机草稿阶段需要同一份输入）。

**结论**：S11 = 补一条 "rank 0 编码 → 两卡拿到同一份 merged embedding" 的通路，
外加位置（mrope）与 KV/state 的对齐。**不是重写引擎**。

---

## 2. 设计选项（先读再选）

### 方案 A（推荐）：rank 0 编码 + 广播图像行

- rank 0 跑视觉塔 + projector，得到 `[N_img, 5120]` 的图像 embedding（bf16）。
- 把这几行从 rank 0 送到 rank 1：无 P2P ⇒ 走 pinned host 中转（现有 peer mailbox /
  `peer_exchange` 机制可复用），或在 prefill 前做一次 `cudaMemcpyPeer` 风格的分段拷贝。
- 两卡各自把图像行合并进本 chunk 的输入 embedding（`scatter` 到占位符位置）。
- 位置：mrope 的 3 轴位置必须两卡一致；单卡路径已有计算，**照抄**（或 rank 0 算完广播）。
- KV / state：图像 token 只是 prefill 的额外行，按现有位置推进即可；关键是两卡位置一致。

代价：每请求一次 `N_img × 5120 × 2B` 的传输。`--max-vision-tokens 4096` 时约 40 MiB，
一次性、可接受；`--max-vision-tokens 16384` 时约 160 MiB，需实测。

### 方案 B（兜底）：两卡各复制一份视觉塔

- 两卡都加载塔（每卡 +约 1 GiB），各自编码 → 零传输、正确性最直接。
- 代价：显存（在 16 GB 卡上很贵）+ 编码 2×。**只有在 A 的传输/同步做不顺时才用。**

### 方案 C（不选）：只在 rank 0 算图像、rank 1 用零填充

会改变输出语义，不可接受。

---

## 3. 分阶段实施

每个阶段都必须：**先跑基线 → 改 → 编译 → 定向测试 → 记录数字**。红门不提交"完成"。

### P0 — 环境与复现（0.5 天）

1. `git clone -b blackwell-only <repo> /data/s11/tree`（若已克隆则跳过）。
2. 建 deps：`/data/s11/deps-ada`（libcurl ≥7.85 + FFmpeg ≥6；命令见 §5）。
3. 配置 + 全量构建到 `/data/s11/build-120a-cu13`（见 §5）。
4. 复现失败：`ninfer-serve --tp 2 --vision --vision-device 0` 起服务，发一张图 →
   期望 **HTTP 500**、message 含 `multimodal prefill is not implemented at tensor-parallel width 2`。
5. 记录：构建 rc、耗时、显存占用、`/v1/models`、纯文本请求的 decode tok/s（基线）。

**验收**：能稳定复现 500；纯文本 tp 2 正常。

### P1 — 给 tp 2 prefill 加"预计算 embedding"入口（1–2 天）

- 目标：机制通路先通，**无图像时不改变任何行为**（传 null 走原路）。
- 改动面：
  - `execution/text.cpp`：tp 2 的 prefill 入口增加可选的 `const Tensor* hidden`
    （形状 `[hidden, tokens]`，bf16），非空时跳过 `embed(ids)`；
  - `execution/tp.*`：若需要，给 rank 1 的输入留出同样的 hidden 缓冲；
  - `program/planning/startup.cpp`：为这块 hidden 记 workspace（按最大 chunk × hidden 估算）。
- 验收：传 null 时，`ninfer_qwen3_5_engine_tp2_real_test` 与纯文本 serve 的数字与 P0 基线一致
  （decode 波动 <5%、输出逐字一致）。

### P2 — 图像行：rank 0 编码 + 送到 rank 1（2–4 天）

- rank 0：调用现有视觉塔/预处理（单卡路径已有），得到图像 embedding 与 mrope 位置。
- 传输：复用 `peer_exchange` / pinned host 中转；在 prefill 前完成，两卡同步点明确。
- 合并：两卡各自把图像行 scatter 到本 chunk 的 hidden 对应位置（图像 token 是连续段）。
- 解除 `require_single_device("multimodal prefill")` 的拦截（只对"有 handoff"的路径）。
- 验收（机制级）：两卡 fed 的 hidden 逐位一致（加一个调试断言/哈希）；
  图像请求不再 500；答案非空。

### P3 — 位置、KV、state 对齐（2–3 天）

- mrope 3 轴位置（文本/图像不同）两卡必须完全一致；
- 图像 token 的 KV 行在镜像缓存里按同一位置写入；
- GDN 线性状态在图像 token 上的推进与单卡一致；
- `--prefill-chunk` 分块边界不得把图像段切出不一致（两卡同 chunk 计划）。
- 验收：同一请求两次运行贪心输出逐字一致；长图/多图不崩。

### P4 — MTP 桥解除拒绝（1–2 天）

- `mtp_bridge_and_propose` 的 tp 2 分支不再拒绝 composed embedding：
  把图像行同样送到 rank 1（或保证桥阶段用的是已合并的 hidden）。
- 验收：`--spec mtp` + 图像请求正常；`ninfer_qwen3_5_engine_mtp_tp2_real_test` 仍 ok。

### P5 — 测试与契约（2–3 天）

- 把 `tests/models/qwen3_5/test_engine_vision_tp2_real_test.cpp` 从"跳过/拒绝"改成**真跑**：
  - 一张带可读数字的图 + 一张多物体图，断言答案含正确内容；
  - 与单卡路径的 **视觉特征** 对拍（同一塔、同一预处理 → 逐位一致）；
  - 贪心确定性（同配置两跑逐字一致）；
  - 文本-only 的 tp 2 套件不回归。
- 文档：`docs/cli.md` Two GPUs 段、`docs/maintainer/tensor-parallel.md` §8 改成"已支持（机制：handoff）"。

### P6 — 门链与交付（1 天）

- 家族子集（`linear|fp8|nvfp4|bf16`）＋ 簇门（`-E softmax_attention`，分母取当次）
  ＋ 12/2 资格面（`-R softmax_attention` ＋三 dtype）全绿；
- 打 tag（红门不打）；分支推 GitHub：`s11-multimodal-tp2`；
- 写一页 `docs/maintainer/s11-notes.md`（机制、限制、实测数字）。

---

## 4. 测试与判据（照本机口径）

| 项 | 判据 |
|---|---|
| 构建 | 全量 rc=0，0 warning（`ninja` 日志） |
| 纯文本 tp 2 | decode 与 P0 基线差 <5%；60 题 greedy **逐字一致** |
| 图像请求 | HTTP 200；答案含图中可读内容；同配置两跑逐字一致 |
| 视觉特征 | 与单卡路径逐位一致（同一塔） |
| 二卡套件 | `vision_tp2_real` / `mtp_tp2_real` / `dflash2_tp2_real` 全 ok |
| 门链 | 子集 100% + 簇门 100%（当次分母）+ 资格面 1/1 |
| 显存 | 196,608 上下文（int8 KV）下仍可起；256K（nvfp4 KV）不回归 |

注意：单卡上二卡套件会 **skip 77**，绿不代表两卡证据；必须在 `CUDA_VISIBLE_DEVICES=0,1` 上跑。

---

## 5. 环境与构建（全部落在 `/data/s11`，根分区零增长）

```bash
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/data/s11/deps-ada/lib
cd /data/s11/tree

# deps（一次性；libcurl 8.11 + FFmpeg 6.1.6，装到 /data/s11/deps-ada）
#   curl: 见 research/README.md §5 的命令，PREFIX=/data/s11/deps-ada
#   ffmpeg: 6.x --prefix=/data/s11/deps-ada --enable-shared

cmake -S . -B /data/s11/build-120a-cu13 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_ENABLE_VISION=ON -DBUILD_TESTING=ON \
  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON -DCMAKE_PREFIX_PATH=/data/s11/deps-ada
JOBS=14 nice -n 12 cmake --build /data/s11/build-120a-cu13 -j14
```

只编需要的目标可以省时间/磁盘：

```bash
cmake --build /data/s11/build-120a-cu13 -j14 --target \
  apps/ninfer-serve tests/ninfer_qwen3_5_engine_vision_tp2_real_test \
  tests/ninfer_qwen3_5_engine_mtp_tp2_real_test tests/ninfer_qwen3_5_engine_tp2_real_test
```

运行测试：

```bash
cd /data/s11/build-120a-cu13
export LD_LIBRARY_PATH=/data/s11/deps-ada/lib
CUDA_VISIBLE_DEVICES=0,1 ./tests/ninfer_qwen3_5_engine_vision_tp2_real_test
```

---


### 5.1 deps 精确命令（libcurl 8.11 + FFmpeg 6.1.6 → /data/s11/deps-ada）

```bash
PREFIX=/data/s11/deps-ada
mkdir -p "$PREFIX/src" && cd "$PREFIX/src"

# libcurl 8.11（CMake 要求的 >= 7.85）
curl -LJO https://github.com/curl/curl/releases/download/curl-8_11_0/curl-8.11.0.tar.xz
tar xf curl-8.11.0.tar.xz && cd curl-8.11.0 && mkdir -p b && cd b
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_CURL_EXE=OFF -DCURL_USE_OPENSSL=OFF \
  -DCURL_USE_LIBPSL=OFF -DBUILD_TESTING=OFF
cmake --build . -j14 && cmake --install .

# FFmpeg 6.1.6（vision 需要 >= 60）
cd "$PREFIX/src"
curl -LJO https://ffmpeg.org/releases/ffmpeg-6.1.6.tar.xz
tar xf ffmpeg-6.1.6.tar.xz && cd ffmpeg-6.1.6
./configure --prefix="$PREFIX" --enable-shared --disable-static --disable-doc --disable-programs
make -j14 && make install
```

- 没有 `nasm` 时给 `./configure` 加 `--disable-x86asm`（功能不变、略慢）。
- 校验：`ls "$PREFIX"/lib/libcurl.so* "$PREFIX"/lib/libavcodec.so*`；
  `PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --modversion libavcodec` 应 ≥ 60。
- CMake 配置时 `-DCMAKE_PREFIX_PATH=/data/s11/deps-ada` + `PKG_CONFIG_PATH` 两者都给，
  可避免 FFmpeg 被 "not found" 静默关掉 vision。

## 6. 制品（本机已删除，需重新获取）

| 件 | 大小 / sha256 | 用途 |
|---|---|---|
| `Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer` | 23,773,675,264 B；sha256 `3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808` | S11 的主制品（含 vision / MTP / DFlash2） |
| `qwen3.8-27b-lynn-nf4.ninfer` | 19,345,900,804 B；sha256 `faf551e2c782022606e0ffd382e8d015cec574914a9476ca6007f346c4c53268` | **不可用于 tp 2**：输出头是 `q6_g64`，无半词表形状 |

重量级注意：W4A4 下载约 24 GB（家宽 1–1.5 h）。若只想先验机制，可用任何一个 **vision+proposal**
的 27B 制品；但最终验收要用上表两件之一。

---

## 7. 风险与未知（先写在这里，别在实现时才发现）

1. **视觉塔内存**：方案 A 下 rank 0 多一份塔（约 1 GiB）+ 图像行缓冲；16 GB 卡上 256K 上下文可能压不下。
   先按 196,608（int8 KV）验收，256K（nvfp4 KV）作为"不回归"项。
2. **传输延迟**：无 P2P，图像行经 host；`--max-vision-tokens` 越大越贵，先按 4096 验收。
3. **mrope 位置**：3 轴（t/h/w）在视觉 token 上与文本不同；错一位就整段发散，必须单元级对拍。
4. **prefill 分块**：图像段跨 chunk 边界时两卡的计划必须一致；必要时把图像段对齐到 chunk。
5. **MTP/DFlash2 与视觉共存**：投机后端在图像请求上是否启用要单独测（MTP 桥的拒绝在 P4 解除）。
6. **视频**：任务书只要求图像；视频走同一 handoff，但帧数/位置更复杂，放到最后，不阻塞。

---

## 8. 工程纪律（本项目血泪规则，务必遵守）

1. 写文本用 heredoc；python 字符串里不要嵌套 ASCII 双引号。
2. **新跨头函数至少定向链接一次**（单 `.o` rc=0 在跨 TU 符号上零证据力）。
3. 注册进 ctest ≠ 跑得动：python oracle 必须判 rc=0 **且** stdout 含 `[PASS]`。
4. 红门/缺段的门**不打 tag**；ctest 分母取当次计数。
5. `kill` 用 PID；**`pkill -f` 会匹配到自己的命令行**。
6. 门链 ctest 期间主树不重建不链接；临时件/日志/参考一律落 `/data/s11` 或 `/tank`。
7. 改 C++ 里 `"...\n"` 行时，edit 的 `old_string` 要把反斜杠写全。
8. 文档段落逐句核符号是否存在（同类坑已五次）。
9. 两卡测试必须 `CUDA_VISIBLE_DEVICES=0,1`；单卡 skip 77 不算证据。
10. 每阶段结束写 ≤25 行记录（做了什么、数字、下一步），放 `notes/` 下。

---

## 9. 时间与交付

| 阶段 | 估计 |
|---|---|
| P0 环境/复现 | 0.5 天（含下载权重的等待可并行） |
| P1 通道 | 1–2 天 |
| P2 图像 handoff | 2–4 天 |
| P3 位置/KV/state | 2–3 天 |
| P4 MTP 桥 | 1–2 天 |
| P5 测试/文档 | 2–3 天 |
| P6 门链/交付 | 1 天 |
| **合计** | **约 1.5–2 周**（单人/单模型专注） |

交付物：`s11-multimodal-tp2` 分支、绿门链、`vision_tp2_real` 真跑、`docs/maintainer/s11-notes.md`。

---

## 10. 参考

- `research/README.md` §3（为什么这项在硬件上曾判死）、§7（踩坑清单）
- `research/s7-seed-5060ti.md` §6（本机实测：容量/并发/速度）
- `docs/maintainer/tensor-parallel.md` §1/§6/§8/§9（拒绝表、状态与 KV、Vision@tp 2、tp 1 恒等）
- `docs/cli.md` Two GPUs、`docs/serving.md` 多模态输入
- 单卡视觉路径：`src/models/qwen3_5/execution/text.cpp` 的 prefill/multimodal 段（S11 的模板）

---

## 11. 原子化任务包与 GPU 需求矩阵（排期用）

**先说结论**：
- **编译永远不需要 GPU**（nvcc 交叉编译，无设备也能出 .o / 链接）；
- **任何 tp 2 行为的验证都必须双卡**（单卡上二卡套件会 skip 77，绿是假绿）；
- 单卡只用于三件事：tp1 视觉回归、门链（设计上单卡跑、tp2 套件被 skip）、以及纯文本单卡对照。

共 **14 个工作包**：**6 个纯 CPU、8 个需要 GPU**；其中 **6 个必须双卡**、**2 个单卡即可**。

| # | 工作包 | GPU | 估时 | 依赖 | 验收 |
|---|---|---|---|---|---|
| W0a | deps（libcurl 8.11 + FFmpeg 6.1.6 → `/data/s11/deps-ada`） | **CPU** | 0.5 天 | — | `pkg-config --modversion libavcodec` ≥60 |
| W0b | 下载 QUASAR 17.4 GB（可与 W0a 并行） | **CPU/网络** | 1–2 h | — | `sha256sum -c` 通过 |
| W1 | 复现：tp2 + vision + 图像 → 500；纯文本 tp2 基线 | **2G** | 0.5 天 | W0a/W0b | 500 的 message 逐字；基线 tok/s 记录 |
| W2 | P1 代码：tp2 prefill 的 `hidden` 入口 + workspace 记账 | **CPU** | 1–2 天 | W0a | 编译 + 定向链接；null 路径不改变逻辑 |
| W3 | P1 回归：纯文本 tp 2 与 W1 基线一致 | **2G** | 0.5 天 | W1,W2 | 60 题逐字一致；decode <±5% |
| W4 | P2 代码：rank0 编码 + 传输 + 合并（handoff） | **CPU** | 2–4 天 | W2 | 编译 + 单元级形状断言 |
| W5 | P2 机制验证：两卡 hidden 逐位一致、图像请求 200 | **2G** | 1 天 | W4 | 断言通过；答案非空 |
| W6 | P3 代码：mrope 位置 / KV 行 / state 对齐 | **CPU** | 2–3 天 | W4 | 编译；位置单测 |
| W7 | P3 数值验证：多图/长图、两跑确定性 | **2G** | 1 天 | W5,W6 | 贪心逐字一致；无 NaN/CUDA 错 |
| W8 | P4 代码：MTP 桥解除拒绝（composed embedding） | **CPU** | 1–2 天 | W6 | 编译；桥路径单测 |
| W9 | P4 验证：`--spec mtp` + 图像；mtp/dflash2 tp2 real | **2G** | 0.5–1 天 | W7,W8 | 三个二卡套件 ok |
| W10 | P5 单卡视觉回归（tp1 视觉不变） | **1G** | 0.5 天 | W7 | tp1 图像答案与改造前一致 |
| W11 | P5 二卡套件全绿 + vision_tp2_real 真跑 | **2G** | 0.5 天 | W9,W10 | 套件 ok；测试从 skip 变真跑 |
| W12 | P6 门链（子集 28 + 簇门 174 + 12/2 资格面） | **1G** | 1 天 | W11 | 全绿（分母取当次） |
| W13 | P6 文档/契约/tag/推送 | **CPU** | 0.5 天 | W12 | 分支推送；文档更新 |

### 排期建议

- **CPU 段可连续做**：W0a/W0b → W2 → W4 → W6 → W8 → W13（其中 W13 在门链后）。
- **双卡窗口共 6 次**（W1、W3、W5、W7、W9、W11），每次 0.5–1 天；可合并成 **3 个半天窗口**：
  窗口 1 = W1+W3；窗口 2 = W5+W7；窗口 3 = W9+W10+W11。
- **单卡窗口 2 次**（W10、W12），各半天。
- 每个窗口前把代码与测试写好、编译过，窗口只做"跑 + 记录"，避免占着卡调代码。

---

## 12. 权重推荐（别再下 23.77 GB 那件）

**推荐：`Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer`**（HF），`qwen3_8_27b_quasar_nvfp4.ninfer`

| 项 | 值 |
|---|---|
| 大小 | **17,397,699,076 B（约 16.2 GiB / 17.4 GB）** |
| SHA-256 | `808a0fa3bb5b3256aadef0d5c8264c106122d1e8483bd2cf74db4e1183cbb570` |
| 组件 | **text + vision + MTP**；**不含 DFlash2** |
| tp 2 实测（供体） | 2× RTX 5070 Ti 16 GB：**196,608 ctx + Vision + MTP，13,083 / 12,743 MiB per board；262,144 也放得下** |
| 为什么合适 | 全 NVFP4（QAT，512 个投影），`lm_head`/embedding 是 FP8 行缩放（**有半词表形状，tp 2 能加载**）；比官方 22.09 GiB 件轻 5 GiB；S11 不需要 DFlash2 |

**为什么不用之前那两件**：
- `…W4A4…`（23.77 GB）：多出来的主要是 DFlash2 等组件；S11 用不到，白占 6 GB 下载与每卡显存。
- `qwen3.8-27b-lynn-nf4`（19.35 GB）：输出头是 `q6_g64`，**没有半词表形状，tp 2 直接加载失败**；
  且其 DFlash2 在转换时被做坏——与 S11 无关，但别拿它做基线。

**注意**：QUASAR 卡自己写着 "this tree rejects `--vision` at `--tp 2`"——那正是 S11 要修的东西；
修好之后，这件 17.4 GB 的权重就是最合适的验收制品（196K + 视觉正好落在 2×16 GB 的预算内）。

---

## 13. 执行分工（2026-10-18 用户裁定）

**双卡阶段全部由 DeepSeek-V4.1-Flash（API 会话）接手开发与验证；其余阶段酌情由本地 Qwen3.8 负责。**

| 工作包 | 是否双卡 | 负责人 |
|---|---|---|
| W0a deps 构建 | 否 | 本地 Qwen3.8 |
| W0b 下载 17.4 GB + 校验 | 否 | 本地 Qwen3.8 |
| W1 复现 500 + 基线 | **是** | **DeepSeek** |
| W2 P1 通道代码 | 否 | 本地 Qwen3.8 |
| W3 P1 回归 | **是** | **DeepSeek** |
| W4 P2 handoff 代码 | 否 | 本地 Qwen3.8 |
| W5 P2 机制验证 | **是** | **DeepSeek** |
| W6 P3 位置/KV 代码 | 否 | 本地 Qwen3.8 |
| W7 P3 数值验证 | **是** | **DeepSeek** |
| W8 P4 MTP 桥代码 | 否 | 本地 Qwen3.8 |
| W9 P4 验证 | **是** | **DeepSeek** |
| W10 tp1 视觉回归 | 否（单卡） | 本地 Qwen3.8 |
| W11 二卡套件全绿 | **是** | **DeepSeek** |
| W12 门链 | 否（单卡） | 本地 Qwen3.8 |
| W13 文档/推送 | 否 | 本地 Qwen3.8 |

**交接约定**：
1. 每个工作包完成时，在本目录 `notes/W<n>-<状态>.md` 写 ≤25 行：改了什么、跑出的数字、失败点、下一步。
2. 本地模型在"非双卡"包里只做三件事：写码/编译、单卡跑、记录；**不要尝试在单卡上验证 tp 2 行为**（会 skip 77，假绿）。
3. DeepSeek 双卡窗口开工前，先读 `notes/` 最新记录与 `git log`，不要凭上一段叙述假设状态。
4. 双卡窗口结束后必须留下：命令、日志路径、tok/s、显存、套件结果、下一步。
5. 任何人不得把 `--vision` 的拒绝当成"已支持"——验收以 `vision_tp2_real` 真跑为准。

