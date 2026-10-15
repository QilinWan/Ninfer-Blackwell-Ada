# NInfer

> 精选检查点，单卡推理性能拉满。

**其他语言：[English](README.md)**

NInfer 是从零实现的 C++/CUDA 推理引擎，面向单张 NVIDIA GeForce 显卡运行 Qwen3.5 Dense 与
MoE 架构。支持文本、图像、视频输入，提供本地 CLI 以及 OpenAI / Anthropic 兼容的 HTTP 接口。
运行时是刻意专精化的：**一块 GPU、一个常驻模型、启动时固定 1~8 路并发**。

---

## 本仓库：NInfer 的 Ada（sm_89）支持

上游 [Neroued/ninfer](https://github.com/Neroued/ninfer) 只编译一种架构
`sm_120a`（RTX 5090）。本仓库完整保留该构建，并**新增 Ada（`sm_89`）构建**：
RTX 4090 / 4090D / 4080 (SUPER) / 4070 (Ti) / 4060。两者消费同一套 **v3 `.ninfer` 产物**——
v3 只是容器与清单变更，不是重新量化，因此**无需重新下载或转换**。

Ada 支持什么、不支持什么、为什么、每一条如何验证：
**[docs/sm89.zh-CN.md](docs/sm89.zh-CN.md)**。

---

## 本构建的两项核心能力

### 一、在 Ada 上实现 4-bit KV（`rk4v4-e8`）

**Ada 此前没有可用的 4-bit KV 路线。** 本仓库新增基于 **E8 Conway–Sloane 格点**的编解码器，
把 **K 与 V 都压成 4-bit 码**：

* **K**：D256 Hadamard 旋转（×1/16）→ 每 8 维子空间做 E8 格点投影 → `rintf` 截断到 [-8,7]
* **V**：同一旋转 → `__float2int_rn` 截断到 [-7,7]
* **每 token 每 kv head 仅 130 字节**，比 INT8 KV **小约 2.5 倍**
* 解码时把 4-bit 码解包为 int8，**直接喂给已有的 s8 张量核 QK 与 FP16 PV 通路**，
  没有额外的反量化开销

**实测（32 GB Ada 卡 + 27B 产物）**：文本生成正确、多模态读图正确，
DFlash2 接受率 **36.1% / 41.8%**、解码 **83.4 / 91.5 tok/s**；
MTP 接受率 **63.2% / 66.7%**、解码 81.8 / 83.0 tok/s。

> 同一张 Ada 卡上 `nvfp4` / `k8v4` 是**刻意不可用**的：它们需要 Blackwell 的 FP4 转换、
> 块缩放 MMA 与 TMA，相关代码是**显式报错的桩**，宁可拒绝也不算出错误结果。

### 二、全系上下文外推（YaRN）——`sm_89` 与 `sm_120a` 双架构

静态 YaRN 在本构建中**同时覆盖两代架构**：

```bash
--rope-yarn-factor 4 --rope-original-max-position 262144 --max-context 580000
```

**工程要点**：位置上限、因果注意力包络、草稿窗口**三者全部跟随因子缩放**，
而切分几何本就随 `max_visible_keys` 增长——因此**不需要改动任何内核即可完成外推**。

| 实测上下文上限（32 GB 卡）| Ada `sm_89`（`rk4v4-e8`）| Blackwell `sm_120a`（`nvfp4`）|
|---|---|---|
| 裸跑（无多模态、无投机）| **约 820,000 token** | **约 790,000 token** |
| + 多模态 | 约 720,000 token | 约 730,000 token |
| + 多模态 + 投机 + 2 并发 + 全调优 | **512,000 已验证**（上限约 544K）| **580,000 已验证** |
| 1,048,576（1M）| ❌ 需约 34.7 GiB | ❌ 需约 34.7 GiB |

**1M 不可行是显存预算判定（引擎会明确报 `minimum Engine runtime reservation requires …`），
不是软件限制**；需要 48 GB 级显卡。

完整细节（r2 修复清单：r1 的 E8 因缺输出逆旋转而输出乱码；以及 `nf4` 产物上 DFlash2
不可用的选择器格式问题）见 **[docs/FEATURES-4bit-kv-and-yarn.zh-CN.md](docs/FEATURES-4bit-kv-and-yarn.zh-CN.md)**。

---

## 双架构支持矩阵

| | Ada（`-DCMAKE_CUDA_ARCHITECTURES=89`）| Blackwell（`120a`，上游默认）|
|---|---|---|
| 产物格式 | 仅 v3 | 仅 v3 |
| 权重配方 | `groupwise-int`（Q4/Q5/Q6/Q8 + Q8 词表）| groupwise-int、`nvfp4` |
| 投机解码 | MTP、DFlash、DFlash2 | MTP、DFlash、DFlash2 |
| **KV 缓存** | INT8、FP8、BF16、**RK4V4E8（Ada 专属）** | 另加 **NVFP4、K8V4** |
| 多模态、CLI、HTTP 服务 | 是 | 是 |
| **静态 YaRN 上下文外推** | **是** | **是** |

本构建有一处已知限制：离线 `ninfer` 与 `ninfer-perplexity` 的参数解析器只接受
`bf16|int8|fp8|nvfp4|k8v4`，因此 `rk4v4-e8` 只能通过服务端使用
（`ninfer-serve --kv-dtype rk4v4-e8`），两个 CLI 用不了。发布的 Ada 包与 r1 一致，只含
`bin/ninfer-serve`。

一个 cubin 覆盖整个 Ada 家族：SM 数量、L2 大小与驻留策略都在运行时从设备读取，
因此 4090（128 SM）、4090D（114 SM）与 4080 SUPER（80 SM）**共用一个二进制**。

---

## 官方产物

官方提供多份产物，快速开始命令默认使用 Qwen3.8-27B NVFP4（需要 Blackwell）。
Ada 用户请选择 `groupwise-int` 配方，并使用本仓库的 Ada 构建。详见
[docs/sm89.zh-CN.md](docs/sm89.zh-CN.md) 的产物章节。

---

## 预编译引擎

* **[engine-sm120a-cu131-r2](https://github.com/QilinWan/Ninfer-Blackwell-Ada/releases/tag/engine-sm120a-cu131-r2)**
  —— RTX 50 系（sm_120a）/ CUDA 13.1，已附带运行所需依赖，无需安装 CUDA 工具链。
* **[engine-sm89-cu128-r2](https://github.com/QilinWan/Ninfer-Blackwell-Ada/releases/tag/engine-sm89-cu128-r2)**
  —— RTX 40 系（sm_89）/ CUDA 12.8，带多模态，含 `rk4v4-e8` 与 YaRN 全部修复。

已下架的 `engine-sm89-cu128-r1` 请勿使用：它的 E8 路线缺少输出逆旋转，会输出乱码，由 r2 取代。

两个发行版都附带 `.sha256` 校验文件；构建溯源与实测上限见各自的发行说明。

---

## 更多文档

| 文档 | 内容 |
|---|---|
| [docs/FEATURES-4bit-kv-and-yarn.zh-CN.md](docs/FEATURES-4bit-kv-and-yarn.zh-CN.md) | 两项核心能力详述、r2 修复清单、已知限制 |
| [docs/sm89.zh-CN.md](docs/sm89.zh-CN.md) | Ada 构建指南、支持矩阵、踩坑清单、实测数据 |
| [docs/serving.md](docs/serving.md) | 服务参数、YaRN 用法与缓存兼容性 |
| [docs/maintainer/rtx4080s-32g/README.md](docs/maintainer/rtx4080s-32g/README.md) | RTX 4080 SUPER 32G 双权重调优实测报告 |
| [docs/maintainer/sm89-e8-verification/](docs/maintainer/sm89-e8-verification/) | 跨机复验包（启动脚本、上限探测、needle 外推、编码器探针）|
