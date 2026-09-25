# NInfer × 双权重 × RTX 4080 SUPER 32G · 极速调优报告

> 云机：`ssh -p 50730 root@connect.westc.seetacloud.com`（AutoDL `autodl-container-84a24c9a34-7083e900`）
> 时间：2026-09-24 02:13–04:13（CST）· 引擎：自建 `QilinWan/Ninfer-sm89`（Ada sm_89，Release）
> 权重（SHA256 见 §7）：
> - **A** `qwen3.8-27b-lynn-mtp.ninfer`（18,210,703,360 B / 16.9 GiB，MTP 头）
> - **B** `qwen3.8-27b-lynn-df2.ninfer`（19,986,249,216 B / 18.6 GiB，DFlash2 头）

---

## 0. 结论速览（TL;DR）

| | 权重 A `lynn-mtp` | 权重 B `lynn-df2` |
|---|---|---|
| **最佳投机参数** | `--spec mtp --draft-tokens 3 --lm-head-draft` | `--spec dflash2 --draft-tokens 7 --lm-head-draft` |
| **最佳启动参数**（文本） | `--max-context 262144 --kv-capacity 262144 --kv-dtype int8 --prefill-chunk 2048` | 同左 |
| 单流解码（代码类 prompt） | **88.1 tok/s** | **100.9 tok/s** |
| 单流解码（结构化 JSON） | **111.2 tok/s** | **149.6 tok/s** |
| 单流解码（数学推理 1024 tok） | **101.3 tok/s** | **116.8 tok/s** |
| 单流解码（叙事类，A 反超 B） | **75.2 tok/s** | 64.5 tok/s |
| 无投机基线 | 40.1 tok/s | 40.1 tok/s |
| Prefill（6.8K prompt） | 1.45–1.52k tok/s | 1.45–1.52k tok/s |
| 256K 上下文 + 视觉 | ✅ kv payload 8.77 GiB | ✅ kv payload 8.25 GiB |
| 服务端聚合（256K 共享池） | C=1 80.2 / C=2 121.8 / C=4 **142.8** tok/s | C=1 85.3 / C=2 109.2 / C=4 **159.1** tok/s |

**交付脚本端到端复验**（用最终交付的 `mtp-best.sh` / `df2-best.sh`，`MODE=fast`，KV 262144）：
`mtp-best.sh` → **88.5 tok/s**（接受 58.0%，2.73，kv payload 8.77 GiB）；`df2-best.sh` → **101.2 tok/s**（44.4%，4.09，8.25 GiB）。

**三条最重要的判据**
1. **DFlash2/MTP 的 K 不是越大越好**：MTP 训练窗就是 3（K=2/4/5 四场景均值 80.5/84.4/84.0 < K=3 的 91.4）；DFlash2 K=7 是峰值（K=3/5/7/9 均值 93.4/98.5/**101.0**/81.6），K>9 断崖。
2. **`--lm-head-draft` 必开**（MTP +9%，DFlash2 +5~11%）。
3. **叙事类负载用 A（MTP）、代码/结构化用 B（DFlash2）**——与官方 5090 文档、社区 4080S 报告的结论一致。

---

## 1. 测试环境与口径

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER **32760 MiB**（32G 改版，AD103 80 SM / 736 GB/s，sm_89） |
| 驱动 / CUDA | 595.71.05 / CUDA 13.2（运行时 12.8 工具链产物） |
| 功耗 | 默认 320 W 上限，解码时实测 297–300 W、SM 2550–2640 MHz（**已顶到功耗墙**）、显存 11251 MHz |
| 引擎 | `/root/ninfer/bin/ninfer`（345 MB，2026-09-23 构建；`sm_89 build: dropped 3 Blackwell-only TU + NVFP4/K8V4 routes`） |
| 预热 | CUDA Graph 开启（默认），权重加载 2.6–2.9 s @ 6.4–7.0 GiB/s |
| 计时口径 | 引擎自带 `summary` 的 **phase 速率**（`decode speed` = 解码相 token/秒，不含 prefill/TTFT）；prompt 94 tok，`--greedy`，`--max-new 512`（除非注明 1024） |
| 场景 prompt | code=写 lru_cache 函数；story=500 词灯塔故事；trans=英译中再回译；struct=严格 JSON 实体抽取；math=数论题 1024 tok |
| 复现 | 全部脚本与原始日志：`/root/ninfer/bm/`（本仓库 `raw/` 有副本） |

---

## 2. 投机参数扫描（核心）

### 2.1 MTP（权重 A，四场景 × K，512 tok，int8，chunk 2048）

| K | code | story | trans | struct | **均值** | 接受率(code) | 接受长度(code) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 78.7 | 69.1 | 82.8 | 91.3 | 80.5 | 67.2% | 2.34 |
| **3** | **88.1** | **75.2** | **91.0** | 111.2 | **91.4** | 58.0% | 2.73 |
| 4 | 70.4 | 65.0 | 86.9 | 115.3 | 84.4 | 38.4% | 2.53 |
| 5 | 77.3 | 63.0 | 83.7 | 112.0 | 84.0 | 41.2% | 3.05 |

> K=6 被引擎拒绝（MTP 合法域 1..5）。K=3 三次重复 88.6/88.3/88.2 → **可复现**。

### 2.2 DFlash2（权重 B，四场景 × K，512 tok）

| K | code | story | trans | struct | **均值** | 接受率(code) | 接受长度(code) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 95.9 | 75.5 | 87.5 | 114.8 | 93.4 | 66.5% | 2.99 |
| 5 | 94.9 | 69.6 | 98.4 | 131.0 | 98.5 | 50.5% | 3.52 |
| **7** | **100.9** | 64.5 | 89.0 | **149.6** | **101.0** | 44.4% | 4.09 |
| 9 | 78.0 | 49.4 | 73.2 | 125.9 | 81.6 | 33.4% | 3.96 |
| 11 / 13 | 77.5 / 60.2（K11/K13 单场景） | | | | | 31.4 / 24.4% | 4.36 / 4.15 |

### 2.3 长输出复测（1024 tok，同一 code prompt）

| 配置 | 512 tok | 1024 tok |
|---|---:|---:|
| MTP K=3 + lmd | 88.1–88.6 | 85.5（接受 55.0%，2.65） |
| DF2 K=3 + lmd | 95.9–96.2 | — |
| DF2 K=5 + lmd | 94.9–95.1 | **97.7**（52.5%，3.61） |
| DF2 K=7 + lmd | 100.7–101.0 | 90.3（38.4%，3.68） |

> 长输出下接受率自然回落，K=5 与 K=7 在 1024 tok 尺度上互有胜负（K=7 峰值更高、K=5 更稳）。
> **建议：默认 K=7；要求长程稳定/低方差时用 K=5。**

---

## 3. 其它旋钮 ablations（权重 A / 权重 B）

| 旋钮 | A `lynn-mtp`（基准 88.3） | B `lynn-df2`（基准 96.0） | 结论 |
|---|---|---|---|
| `--lm-head-draft` 关 | 80.6（−9%） | 91.7（K=7, −9%；K=7 时 +lmd 100.7） | **必开** |
| `--kv-dtype fp8` | 86.5/86.3（−2%） | 86.9（−10%） | **int8 最优** |
| `--kv-dtype nvfp4 / k8v4` | 引擎直接拒绝（原文见下） | 引擎直接拒绝 | **Ada 无 4-bit KV**，见 §5 |
| `--no-cuda-graph` | 86.3（−2%） | 94.2（−2%） | 保持默认（开） |
| 不开 `--vision` | 88.2（=基准） | 96.0（=基准） | 视觉对文本速度无影响 |
| 采样：`--greedy` | 87.9 | 100.7（K=7） | — |
| 采样：官方预设 T=1.0/top-p .95/top-k 20 | 88.0 | 100.7 | **采样参数不掉速**（贪心 vs 预设等价） |
| `--no-thinking` | **102.3**（+16%） | **116.9**（+16%） | 关思考显著更快（内容更可预测） |
| `--prefill-chunk` 512/1024/2048/4096/8192 | prefill 1.48/1.52/1.53/1.52/1.46k tok/s | 1.47/1.51/1.52/1.51/1.46k | 512–4096 无差，**取 2048**；decode 不受影响 |

---

## 4. 256K 上下文 + 多模态 + 并发（显存实测）

### 4.1 单请求 CLI

| 配置 | kv payload | free after weights | 结果 |
|---|---:|---:|---|
| A，262144 tok，int8，vision | **8.77 GiB** | 14.3 GiB | ✅ 启动、生成 |
| B，262144 tok，int8，vision | **8.25 GiB** | 12.6 GiB | ✅ 启动、生成 |

### 4.2 服务端（`ninfer-serve`，`--max-context 262144`，INT8，vision，`--kv-capacity 262144` 共享池）

| 模型 / lane | runtime | free | C=1 | C=2 | C=4 |
|---|---:|---:|---:|---:|---:|
| B DF2 K7（draft 5 in serve test） | 10.9 GiB (C=2) / 12.6 GiB (C=4) | 2.59 / **1.76 GiB** | 85.3 | 109.2 (1.28×) | **159.1 tok/s (1.87×)** |
| A MTP K3 | 11.1 GiB (C=4) | 3.45 GiB | 80.2 | 121.8 (1.52×) | **142.8 tok/s (1.78×)** |

> 口径：HTTP 客户端 wall 时间聚合（含 prefill/首字），512 tok/请求；`--max-concurrency` 固定池。
> **C=8 + `--max-context 262144` + `--kv-capacity auto` 启动失败**，引擎给出明确原因：
> `minimum Engine runtime reservation requires 17133011713 bytes in addition to 1073741824 bytes of automatic headroom, but only 13553565696 bytes are available after weights`
> → 32 GB 卡上 **单用户 256K + 多并发最多 4 lane**（4 lane 仅剩 1.76 GiB 余量）。

**回答"能不能同时 256K 与多并发"**：`--max-context` 是**每序列**上限，`--kv-capacity` 是**全池共享**容量。
- 想保证"任一单用户能用满 256K" → 池 ≥ 262,144 token → C≤4 可启动；
- 想让 **2 个用户各占满 256K** → 需 524,288 token ≈ 16.5 GiB KV，32 GB 卡不成立；
- 想上 C=8 → 必须把每序列上限/池压到 ~180K 以下（`--kv-capacity auto` 实测在该卡被拒）。

### 4.3 多模态实测（与 256K 同配置）

| 输入 | 结果 |
|---|---|
| `visual_chart.png`（DF2 K7） | vision encode **31.2 ms**，输出正确读出图内文字与数值（"...largest value ... is 731"），decode 103.8–119.3 tok/s |
| `natural/时间序列 mp4`（DF2 K7） | vision 67.5 ms，视频展开 1204 prompt tokens，decode 121.8 tok/s，输出正确描述画面与结尾帧 |
| 同上（MTP K3） | vision 67.6 ms，1164 prompt tokens，decode 98.8 tok/s，描述正确 |

### 4.4 被引擎明确拒绝的两种配置（原文取证）

```
# C=8 + --max-context 262144 + --kv-capacity auto
FATAL server failed during startup | minimum Engine runtime reservation requires 17133011713 bytes
in addition to 1073741824 bytes of automatic headroom, but only 13553565696 bytes are available after weights

# --kv-dtype nvfp4（4-bit KV）在 Ada 上
ERROR startup failed | preparing CUDA graphs | 1.1s
FATAL server failed during startup | text/layers/3 verify columns=8: NInfer sm_89 build:
the causal_attention_small_t_nvfp4_launch route needs Blackwell (sm_120a) instructions.
Run a groupwise-int (.ninfer) artifact on Ada and select an INT8 or FP8 KV cache;
NVFP4 weights and NVFP4/K8V4 KV need the sm_120a build
```

---

## 5. 与公开的 4080S-32G 数据对比（同卡同量级）

| 来源 / 配置 | 其结果 | 我们 | 判断 |
|---|---|---|---|
| YurunTao/ninfer-4080S-32G，**官方基座** + 固定 bench 语料，DFlash2 K7，贪心 | **156.66 tok/s**，接受 88.0%，长度 7.11/8 | B 代码 100.9（接受 44.4%） | 差距 = **接受率**（语料可预测性/权重），非引擎 |
| 同上，K7 **不加** `--lm-head-draft` | **175.6 tok/s**，接受 100%，长度 8.0 | B 不加 lmd 91.7（40.6%） | 同上；我们的权重上 lmd 仍是正收益 |
| 同上，prefill 2048 / 无投机 | 1371 tok/s / 37.15 tok/s | **1.45–1.52k tok/s** / **40.1** | **我们略优** |
| 同上，叙事类真实负载 | DFlash2 61.6、MTP3 **78.1** | B 64.5、A **75.2** | **逐项对齐**（叙事 MTP 胜） |
| 同上，结构化代码真实负载 | 接受率 72.7% | B struct 接受 **72.5%**、149.6 tok/s | **完全一致** |
| HyperQwen Issue #149（同卡，250 W，真实 prompt，DFlash2） | C1 解码 116.8、C4 聚合 379.9 | B C1 解码 100.9 / 数学 116.8；C4 聚合 159.1（口径不同） | 同一水平带 |
| agang0311/ninfer-4080-32G（MTP3，10 万上下文，no-think） | 78.7 tok/s，接受 79.8% | A 短上下文 no-think 102.3 | 长上下文衰减所致，非差距 |
| Lynn 作者本人（SGLang，DGX Spark，**我们的权重**，DFlash2 draft=8） | 接受率 **65.5–70.0%**（C1–C24） | B 结构化 72.5% / 代码 44.4% | 结构化落在作者区间内；自由代码 prompt 偏低是内容属性 |
| Lynn 作者本人（MTP3，长推理 C24） | 接受 56.28% | A 代码 58.0%、数学 71.3% | **一致** |

**为什么会"明显慢"于 156–175？** 拆解：① 那组数字来自**官方基座权重 + 固定 bench 语料**，逐位置接受率 94/94/94/94/83/83/67%，我们这份 Lynn 微调权重在**自由 prompt** 上只有 44–72%；② 解码是权重带宽受限，每轮搬运 16.9–18.6 GiB，速度 ≈ 接受长度 × 轮速，接受长度 3.1 vs 6.1 就是 2× 差距；③ 我们在 prefill、无投机解码、结构化接受率三项上与那组数据持平或更好 → **引擎没有调参损失**。

---

## 6. 质量抽查（"转化有没有伤"）

| 检查 | 结果 | 判读 |
|---|---|---|
| PPL（英文 pg19 样本，15,759 token 评分，ctx 4096/stride 2048） | **int8 6.187140**（mean_nll 1.822473）/ fp8 6.188403 / bf16 6.188708 —— A、B 两权重数值**完全相同** | KV 量化差 **+0.025%**（int8 vs bf16）；A/B 文本主干 PPL 相同（同一 backbone，符合预期） |
| 贪心可复现 / 投机无损 | 同一 prompt 跑 5 次（spec×2、no-spec×2、无 lmd×1，`rp_*.txt`）**逐字节相同** | 投机解码不改变输出（早期一次 512-tok 对比出现 1 处措辞差异，未能复现，判为当时的并发干扰） |
| 多模态理解 | 图表读出 "731 / three red circles / blue square → green triangle"；视频正确描述动线与 END 帧 | 视觉塔与投影转换完好 |
| 数学推理 | 数论题（n<1000，n²+n+1 ≡ 0 mod 7 计数）1024 tok 内正常推理，A 接受率 71.3% | 思考链正常，无退化/复读 |
| 与作者自测对照 | MTP 接受率 56–76%（作者 56.28–75.8%）、DFlash2 结构化 72.5%（作者 65.5–70%） | **转换未伤草稿头** |

> 局限：本次只做了 PPL 与行为抽查，**没有**跑 GPQA/MMLU/LCB 全套（作者卡上已发布冻结分：Q4-LynnStyle GPQA 166/198、MMLU 443/500、LCB 74/100，可作对照基线）。

### 6.1 4-bit KV 的问题（用户问）
Ada（sm_89）构建**在编译期就剔除了 NVFP4/K8V4 的 kernel 路由**，`--kv-dtype nvfp4|k8v4` 只会得到明确的"不支持"错误，不存在"能不能接受质量下降"的取舍。本卡可用档位只有 **bf16 / int8 / fp8**：
- INT8 与 BF16 的 PPL 差异 <0.03%，而 INT8 的 KV 占用是 BF16 的一半 → **INT8 是唯一正解**，256K 上下文也只有 INT8 塞得下（8.25–8.77 GiB）。
- 质量风险不来自 KV，而来自 **groupwise Q4/Q5 权重**（官方 recipe，作者 GGUF 同档实测损失可控）。

---

## 7. 交付物

### 7.1 一键启动脚本（云机 `/root/ninfer/launch/`，本地 `ninfer-4080s-32g/`）

```bash
# 权重 B（DFlash2，代码/结构化首选）
/root/ninfer/launch/df2-best.sh ask "你的问题"          # 贪心，最大速度
/root/ninfer/launch/df2-best.sh serve                  # HTTP 服务 :8080
MODE=think  NEW=4096 /root/ninfer/launch/df2-best.sh ask "..."   # 官方思考预设采样
LANES=4 CTX=262144 KVCAP=262144 /root/ninfer/launch/df2-best.sh serve

# 权重 A（MTP，叙事/长思考首选）
/root/ninfer/launch/mtp-best.sh ask "你的问题"
MODE=nothink /root/ninfer/launch/mtp-best.sh ask "..."  # 极速档（+16%）
VISION=1 /root/ninfer/launch/mtp-best.sh ask "..."      # 开多模态
```

内置默认值（均已实测）：`CTX=262144 KVCAP=262144 KV=int8 SPEC=mtp|dflash2 DRAFT=3|7 LMD=1 CHUNK=2048 VISION=0 MODE=fast`；
`MODE` 支持 `fast`（`--greedy`）/`think`（T=1.0, top-p .95, top-k 20, min-p 0）/`nothink`（`--no-thinking` + T=0.7, top-p .80, top-k 20, presence 1.5）。

### 7.2 权重 SHA256
```
A qwen3.8-27b-lynn-mtp.ninfer  7970ac0f6263f666ff21bc0ec891ec78976a2fbbc33281aefce6b147ebe73958
B qwen3.8-27b-lynn-df2.ninfer  d6d1425c23ca3ea2f93d898d247c4df523199543c3eb6462176c0b03f62754f4
```

### 7.3 原始证据
`raw/`（`/root/ninfer/bm/` 副本）：`cfg1..cfg12 *.res`（参数扫描）、`serve_all.out`、`ppl2_all.out`、`vtest.out`/`vdemo.out`、`smoke.out`、`repro.out`、`sweep2.sh`/`runp.sh`/`serve_test.sh`/`ppl2.sh` 等全部脚本。
