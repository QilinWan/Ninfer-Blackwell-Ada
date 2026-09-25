# NInfer — two-GPU tensor-parallel fork

> **This is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer)** that adds an
> experimental two-GPU tensor-parallel mode (`--tp 2 --devices A,B`), so that Qwen3.8-27B NVFP4,
> which does not fit one 16 GB board, runs split across two. It builds on earlier TP2 forks by
> Wael Mansour, natpate, ivanov84 and parallelno; see [NOTICE](NOTICE) for attribution.
>
> - **Covered:** ordinary decoding, `--spec mtp`, `--spec dflash2 --lm-head-draft`, prefix
>   reuse, concurrent requests, CUDA Graph decode and `--vision`, with `bf16` or `int8` KV. See
>   [Two GPUs](docs/cli.md#two-gpus) and [serving](docs/serving.md).
> - **Verified on:** two RTX 5070 Ti 16 GB without peer access, Linux, CUDA 13.1, core clocks
>   capped at about 2.1 GHz. Other GPUs and P2P-capable pairs are untested.
> - **Measured** with upstream's own benchmark suite against the published RTX 5090 runs, same
>   weights on both, uncapped clocks: plain decode 94-96% of the 5090 (68.7 vs 71.2 tok/s at 7.7k),
>   MTP3 79-87% (corpus of 75 requests 139.7 vs 161.1), prefill 58-86%; with the QUASAR-QAT
>   artifact 90-97% MTP3 and above the 5090 in plain decode. Full tables, per category and
>   concurrency 1-8, in [Two-GPU performance](docs/performance/two-gpu.md). At clocks capped to
>   2.1 GHz, one request, decode at 0 / 16K context: plain 58 / 57, MTP3 110 / 117, DFlash2 118 / 120.
>   GSM8K 0.975-0.985, the same as vLLM on the same weights.
> - **Related forks and prior art.** The two-GPU design originates in Wael Mansour's
>   [ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) (August 2026, 2× RTX 5090, YaRN 1M),
>   continued by giocom, ivanov84 (pinned-host mailbox) and parallelno (DFlash2 at tp 2), all
>   credited in [NOTICE](NOTICE). A sibling continuation of the same line,
>   [lynx-gt/ninfer-tp2-5060ti](https://github.com/lynx-gt/ninfer-tp2-5060ti) (upstream base of
>   2026-08-20, qualified on 2× RTX 5060 Ti, KV tiers, `/health` self-heal, no Vision or DFlash2 at
>   tp 2), decodes at the same rate as this fork on the same pair with the official artifact
>   (server-side, MTP3, C=1: 128-144 vs 130-143 tok/s from 0 to 184K context, measured 2026-09-25).
>   What this fork adds on top of that shared core is the upstream v3 base, Vision, DFlash2 and
>   concurrent requests at tp 2, the NVFP4 split projections, and the benchmark suite comparison.
>   A QUASAR-QAT NInfer artifact (with DFlash2) was published first by
>   [MirkoCovizzi](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer) (2026-08-26);
>   the one here differs in recipe and components, see its card. Single-GPU ports to other
>   architectures exist too: [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) (sm_86).
> - `--tp 1` is meant to behave exactly as upstream. The commits on top of upstream are grouped
>   so that they can be proposed upstream in pieces.
>
> **Known limitations of the two-GPU mode** (read these before trying it on other hardware):
>
> - **Hardware.** Verified only on two RTX 5070 Ti 16 GB (sm_120) without peer-to-peer access,
>   on Linux with CUDA 13.1. Other Blackwell GeForce pairs should work but are untested. A pair
>   with peer access keeps its all-reduces on direct copies (the host mailbox is only used without
>   P2P); that path is untested too. Exactly two GPUs: `--tp` accepts 1 or 2.
> - **Windows via WSL2.** Reported working by [@Zeppe79](https://github.com/ValerioDolci/ninfer-tp2/issues/1)
>   on two RTX 5070 Ti (Windows 11, WSL2 Ubuntu 24.04, CUDA 13.1 from the `wsl-ubuntu` repo): build as
>   on Linux, keep the weights on the Linux filesystem, and start with **`--no-tp-mailbox`**, because
>   the pinned-host mailbox exchange times out under WSL2's GPU virtualization (the spin limit is
>   about 0.8 s, see below). The copy path works but is slower: about 50-65 tok/s reported with
>   MTP3 against 110-120 on native Linux with the same cards. Reaching the server from Windows needs a
>   `netsh interface portproxy` rule to the WSL2 address, which changes at every restart.
> - **Weights.** Verified with the Qwen3.8-27B NVFP4 artifact (`qwen3_8_27b_nvfp4.ninfer`: NVFP4
>   MLP in layers 0-55, FP8 elsewhere) and with an all-NVFP4 conversion of the QUASAR-QAT
>   checkpoint (every large layer projection NVFP4; GDN `a`/`b` decoded to BF16, head and embedding
>   FP8). The split projections take FP8 or NVFP4; the MTP head splits only in Q8. The MoE model
>   and the groupwise-int artifacts are rejected at startup (paired Q4/Q5 input projections have no
>   split route).
> - **Memory per board.** Each board holds half the weights plus its half of the KV cache, so the
>   context and the number of retained conversations trade against each other:
>   - MTP3 runs at 262,144 tokens with 4 device state slots (the default 8 slots do not fit);
>   - the DFlash2 drafter lives whole on the first GPU, so with the official NVFP4 artifact DFlash2
>     stops at about 150,000 tokens; at 131,072 it starts with 2 device state slots. The
>     all-NVFP4 QUASAR-QAT artifact has no DFlash2 component.
> - **No host tier.** `--host-state-slots` and `--host-kv-mib` must be 0 at `--tp 2`: retained
>   conversations live only in the device state slots. With many conversations in parallel the
>   oldest idle ones are evicted, and their next turn is prefilled again.
> - **Vision.** `--vision` is rejected at `--tp 2`: the tower and its encode workspace have no
>   place in the split layout. Run `--tp 1` for image and video input.
> - **Speculative decoding.** DFlash2 needs `--lm-head-draft` and a drafter without
>   full-attention layers (the published drafter has none). `--spec dflash` is not supported.
> - **KV cache types.** `bf16` and `int8` only; `fp8`, `nvfp4` and `k8v4` are rejected.
> - **Where it pays off.** The gain grows with context: prompt processing is 1.5-2.1x and decode
>   1.1x at short context to 1.65x at 184K (1.75x with uncapped clocks) against llama.cpp on the same two boards. On short
>   prose prompts llama.cpp with MTP was about 8% faster. Measured with one request at a time;
>   concurrency is measured up to 8 requests (see the two-GPU performance page).
> - **Numerics.** The split matmuls sum their two halves in a different order than one GPU does,
>   so long greedy generations can drift from a single-GPU run of the same weights. Measured
>   quality matches: GSM8K 0.975-0.98 at `--tp 2`, vLLM on the same weights 0.98.
> - **Profilers.** Inside captured decode rounds the two GPUs wait for each other for at most
>   about 0.8 s; if one stalls longer (a profiler that serializes kernels can do that) the engine
>   stops serving until it is restarted. Use `--no-tp-mailbox` when profiling.
> - **Upstream.** Based on upstream `bace20dc` (24 September 2026); later upstream changes are
>   merged by hand.
>
> Below is the upstream README, with the fork's additions marked **Fork note**: its single-RTX-5090
> statements describe upstream's product, not this fork's tested configuration.

> Selected checkpoints. Maximum single-GPU inference performance.

**Other languages: [中文](README.zh-CN.md)**

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce GPU. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and one to eight execution lanes fixed at startup.

## This repository: NInfer for Ada (sm_89)

Upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) compiles for exactly one architecture,
`sm_120a` (RTX 5090). This repository keeps that build intact and adds a second one for **Ada
(`sm_89`)**: RTX 4090 / 4090D / 4080 (SUPER) / 4070 (Ti) / 4060. Both builds consume the same
**v3 `.ninfer` artifacts** — the v3 switch is a container and manifest change, not a requantization,
so no re-download or re-conversion is needed.

What Ada does and does not support, why, and how each claim is checked:
**[docs/sm89.md](docs/sm89.md)**. Everything else on this page is upstream's product documentation.

> 中文构建指南、支持矩阵与踩坑清单见 **[docs/sm89.zh-CN.md](docs/sm89.zh-CN.md)**。

**中文版首页：[README.zh-CN.md](README.zh-CN.md)**

### Two headline capabilities in this build

**1. 4-bit KV on Ada (`rk4v4-e8`).** Ada had no usable 4-bit KV route. This repository adds an
E8 Conway–Sloane lattice codec that stores **both K and V as 4-bit codes** — a D256 Hadamard
rotation (×1/16) followed by a per-8-dimension E8 lattice projection for K (`rintf`, clamped to
[-8,7]) and `__float2int_rn` (clamped to [-7,7]) for V. **130 bytes per token per kv head**,
about **2.5× smaller than INT8 KV**, decoded straight into the existing s8 tensor-core QK and
FP16 PV paths with no extra dequantization step.

Measured on a 32 GB Ada card with the 27B artifacts: correct text, correct image caption,
DFlash2 acceptance **36.1% / 41.8%** at **83.4 / 91.5 tok/s**, MTP acceptance
**63.2% / 66.7%** at 81.8 / 83.0 tok/s.

**2. Context extrapolation across the whole family.** Static YaRN
(`--rope-yarn-factor 1..4` with `--rope-original-max-position 262144`) works on **both**
`sm_89` and `sm_120a`. The positional cap, the causal attention envelope and the draft window
all follow the factor, and the split geometry already scales with `max_visible_keys`, so no
kernel had to change.

| Measured context ceiling (32 GB card) | Ada `sm_89` (`rk4v4-e8`) | Blackwell `sm_120a` (`nvfp4`) |
|---|---|---|
| bare | **~820,000 tokens** | **~790,000 tokens** |
| with vision | ~720,000 tokens | ~730,000 tokens |
| vision + speculation + 2 lanes + tuning | **512,000 verified** (ceiling ~544K) | **580,000 verified** |
| 1,048,576 | does not fit (~34.7 GiB needed) | does not fit |

The 1M rejection is the VRAM budget check, not a software limit; it needs a 48 GB class GPU.

Full detail, including the r2 fix list (r1's E8 emitted garbage for want of an output
un-rotation) and the `nf4`/DFlash2 selector-format limitation:
**[docs/FEATURES-4bit-kv-and-yarn.md](docs/FEATURES-4bit-kv-and-yarn.md)**.

| | Ada (`-DCMAKE_CUDA_ARCHITECTURES=89`) | Blackwell (`120a`, upstream default) |
|---|---|---|
| Artifact format | v3 only | v3 only |
| Weight recipes | `groupwise-int` (Q4/Q5/Q6/Q8 + Q8 vocab) | groupwise-int, `nvfp4` |
| Speculation | MTP, DFlash, DFlash2 | MTP, DFlash, DFlash2 |
| KV cache | INT8, FP8, BF16, RK4V4E8 (Ada only) | plus NVFP4, K8V4 |
| Vision, CLI, HTTP serving | yes | yes |
| Static YaRN context extension | yes | yes |

One caveat in this build: the offline `ninfer` and `ninfer-perplexity` argument parsers accept only
`bf16|int8|fp8|nvfp4|k8v4` for `--kv-dtype`, so `rk4v4-e8` is reachable through serving
(`ninfer-serve --kv-dtype rk4v4-e8`) but not through those two CLIs. The released Ada archive ships
`bin/ninfer-serve` only, as r1 did.

## Optional YaRN for Qwen3.8-27B (both builds)

`--rope-yarn-factor 1.5 --rope-original-max-position 262144 --max-context 393216`
extends the per-request window using Qwen's published static YaRN recipe. Native behavior is
the default (`--rope-yarn-factor 1`). Text, MTP and multimodal MRoPE share per-Engine
coefficients; the Vision tower stays native. See [serving](docs/serving.md#yarn-context-extension)
for cache compatibility and validation scope.

One cubin covers the whole Ada family: SM count, L2 size and residency come from the device at
runtime, so a 4090 (128 SM), a 4090D (114 SM) and a 4080 SUPER (80 SM) share one binary.

## Official artifacts

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4, which needs
the Blackwell build; on Ada use the `groupwise-int` artifact.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

**Fork note.** This fork adds an unofficial artifact (published by the fork author, not by upstream)
for the two-GPU mode: an all-NVFP4 conversion of the QUASAR-QAT checkpoint (17.0 GiB of weights
instead of 20.9), verified at `--tp 2` on two 16 GB boards. It has no DFlash2 component.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.8-27B QUASAR-QAT | `nvfp4` (all large projections), no DFlash2 | `qwen3_8_27b_quasar_nvfp4.ninfer` | [Qwen3.8-27B QUASAR-QAT NVFP4](https://huggingface.co/Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer) · [card and recipe](model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Prebuilt engines

Both builds ship as relocatable archives — no compiler and no CUDA Toolkit on the target host, the
archive carries its own FFmpeg 6 and CUDA runtime libraries.

* **[engine-sm120a-cu131-r2](https://github.com/QilinWan/Ninfer-Blackwell-Ada/releases/tag/engine-sm120a-cu131-r2)**
  — RTX 50 series (`sm_120a`) / CUDA 13.1.
* **[engine-sm89-cu128-r2](https://github.com/QilinWan/Ninfer-Blackwell-Ada/releases/tag/engine-sm89-cu128-r2)**
  — RTX 40 series (`sm_89`) / CUDA 12.8, vision on, with `rk4v4-e8` and YaRN.

The withdrawn `engine-sm89-cu128-r1` must not be used: its E8 route predates the output
un-rotation and emits garbage. r2 replaces it.

Each release carries a `.sha256` companion file; the build provenance and the measured ceilings are
in the release notes.

## Quick start

NInfer requires 64-bit Linux, a C++20 host compiler, CMake 3.28 or newer, Ninja and `pkg-config`,
plus one of:

| Target | GPU | CUDA toolkit | Extra dependencies | Build flag |
|---|---|---|---|---|
| `sm_120a` (upstream) | RTX 5090; two 16 GB `sm_120` boards for this fork's `--tp 2` | **>= 13.0** (13.1 built and run here; every 12.x fails) | FFmpeg dev (`libavformat`, `libavcodec`, `libavutil`, `libswscale`), `libcurl >= 7.85` | default |
| `sm_89` (this fork) | RTX 40 series | **>= 12.8** (12.8 + 13.2 validated) | same; FFmpeg optional with `-DNINFER_ENABLE_VISION=OFF` | `-DCMAKE_CUDA_ARCHITECTURES=89` |

The Blackwell floor is not cosmetic. Outside the tiled wide-column schedules the Q8 k-split and NVFP4
routes keep their staging union in a plain `__shared__` declaration
(`src/ops/linear/q8/q8_ksplit_mma.cuh`), and CUDA 12.x ptxas/nvlink still applies the 48 KB static
shared-memory limit to `sm_120a`. A 12.x build therefore compiles every translation unit and then
dies at the Ops device link with `uses too much shared data (... 0xc000 max)`; CUDA 13.0 raised that
limit. The Ada build sidesteps the limit because `NINFER_SM89` routes every union larger than 48 KB
through the dynamic window instead. Configure refuses a 12.x toolkit for `sm_120a` up front.

CMake accepts exactly those two architectures and rejects anything else, so a mis-set target fails at
configure time rather than producing a binary that cannot launch.

Build the product binaries:

```bash
git clone https://github.com/QilinWan/Ninfer-Blackwell-Ada.git
cd Ninfer-Blackwell-Ada

# Ada (RTX 4090 / 4090D / 4080 SUPER / 4070 / 4060)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# Blackwell (RTX 5090): identical to upstream
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
```

**Fork note — two 16 GB boards.** The single-GPU examples below need a 5090; on two 16 GB boards
use the two-GPU mode instead (this is the configuration verified in this fork; `fp8` KV, `--vision`
and the host tiers are rejected at `--tp 2`):

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_quasar_nvfp4.ninfer \
  --tp 2 --devices 0,1 --kv-dtype int8 \
  --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 \
  --spec mtp --draft-tokens 3
```

The official `qwen3_8_27b_nvfp4.ninfer` runs with the same flags (about 2 GiB more per board);
see [Two GPUs](docs/cli.md#two-gpus) for the option reference.

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two execution lanes and prefix caching:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
resident requests and retained prefixes. Requests acquire KV pages as execution advances; under
pressure, the scheduler can pause a request and resume it later. The profile provides two extra
Device StateImages and the default shared pinned Host budget: 8 GiB plus eight model StateImages,
used for retained state, KV and pause snapshots.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable checkpoint combines KV with the complete continuation state at an exact token frontier.
The engine retains completed conversation endpoints and stable input boundaries for multi-turn and
agent reuse. Inactive checkpoints share Device and pinned Host capacity; pressure reclaims retained
resources before pausing resident requests. Paused requests resume from a snapshot or rebuild their
state by replaying already committed tokens.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Blackwell measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results. Qwen3.8 uses FP8 E4M3 row-256 KV;
Qwen3.6 uses INT8 group-64 KV. Tuned Ada measurements (RTX 4080 SUPER 32 GB) are at the end of
this section; see [NInfer on Ada](docs/sm89.md#measured-throughput) for the full sweep records.

### Concurrent MTP3 decode

Saturated decode used CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### Single-request serving

The serial serving corpus used CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

### Ada field measurements (RTX 4080 SUPER 32 GB, sm_89)

Tuned on a 32 GB modded 4080 SUPER (80 SM, power-limited at ~300 W) with two community Qwen3.8-27B
`groupwise-int` artifacts carrying an MTP head and a DFlash2 head. Single lane, greedy, INT8 KV,
`--prefill-chunk 2048`; decode rates are phase rates excluding prefill.

| Profile | Best spec setting | Code | Structured | Math | Narrative | No-spec baseline |
|---|---|---:|---:|---:|---:|---:|
| MTP head | `--spec mtp --draft-tokens 3 --lm-head-draft` | 88.1 | 111.2 | 101.3 | **75.2** | 40.1 |
| DFlash2 head | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | **100.9** | **149.6** | **116.8** | 64.5 | 40.1 |

Rates are tok/s. Draft-token count is not monotonic: the MTP head peaks at K=3 (four-scenario mean
91.4 vs 80.5/84.4/84.0 at K=2/4/5), DFlash2 peaks at K=7 (101.0, collapsing to 81.6 at K=9).
`--lm-head-draft` is a net win on both heads on this card (+9% MTP, +5-11% DFlash2). INT8 is the
optimal KV dtype (fp8 -2~-10%, NVFP4/K8V4 refused at build time on Ada). `--no-thinking` adds ~16%.
At 262,144 tokens per sequence the INT8 KV pool costs 8.77 GiB (MTP) / 8.25 GiB (DFlash2) and the
server runs up to 4 lanes: aggregate 142.8 (MTP) / 159.1 (DFlash2) tok/s at C=4; C=8 at full
context exceeds the runtime reservation.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

**Fork note.** The two-GPU mode and the QUASAR-QAT artifact were checked separately, with lm-evaluation-harness through
the OpenAI route at `--tp 2` on two RTX 5070 Ti (thinking on, MTP3, one sample per problem) and compared
per item against the same weights on vLLM 0.30: GSM8K 0.985 vs 0.975, MMLU-Pro (308) 0.789 vs 0.802,
IFEval (200) 0.870 vs 0.880 — no paired difference is significant (exact McNemar p ≥ 0.48). COMET on
FLORES-200 it↔en equals the official NVFP4 weights served by vLLM (0.885 / 0.891 vs 0.885 / 0.892), and
a synthetic long-context suite at 8k/126k matches the official NVFP4 artifact on this runtime. Details,
sources and limits are in the
[QUASAR-QAT model card](model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md).

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile (**Fork note:** Docker is untested
at `--tp 2`; for two boards pass `--gpus '"device=0,1"'` and the two-GPU flags of the quick start
instead of the single-GPU profile below):

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 (or two 16 GB boards at `--tp 2`, see the note below) and one resident model per Engine;
- one to eight resident execution lanes with bounded FIFO ingress;
- resource-pressure preemption with snapshot or token-replay recovery;
- no priority/QoS, weight offload, multi-GPU beyond the experimental two-GPU mode described
  below, or distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

Note: an experimental two-GPU tensor-parallel mode (`--tp 2 --devices A,B`, see
[Two GPUs](docs/cli.md#two-gpus)) splits a dense model across two 16 GB boards that cannot hold it
alone. It is verified only on two RTX 5070 Ti without peer access and leaves every other point of
the product boundary above unchanged.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). The fork's
Qwen3.8-27B QUASAR-QAT NVFP4 artifact imports the weights of
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
(revision `15d2e47b`). These source repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
