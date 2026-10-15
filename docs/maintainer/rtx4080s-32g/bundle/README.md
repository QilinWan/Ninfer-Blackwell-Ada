# NInfer engine — Ada (sm_89), CUDA 12.8, vision ON (r2)

Prebuilt, relocatable NInfer engine for NVIDIA Ada GPUs. The host needs **no compiler and no
CUDA Toolkit**: the archive carries its own FFmpeg 6 and CUDA 12.8 runtime libraries.

| | |
|---|---|
| GPU | NVIDIA Ada, compute capability **sm_89** — RTX 4090 / 4090D / 4080 (SUPER) / 4070 (Ti) / 4060 |
| Driver | NVIDIA driver with CUDA 12.8 runtime support (validated on 595.71.05) |
| OS | Linux x86_64, glibc ≥ 2.35 (Ubuntu 22.04 or newer) |
| Model | a v3 `.ninfer` artifact — **not included**, weights are decoupled from the engine |

> Not for Blackwell (sm_120a). Ada and Blackwell are separate builds.

## Quick start

```sh
tar -xf ninfer-sm89-cu128-r2-x86_64.tar.xz
cd ninfer-sm89-cu128

# 512K context, Ada 4-bit KV, MTP speculation, vision, two concurrent lanes
./start.sh /path/to/qwen3.8-27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 512000 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --rope-yarn-factor 4 --rope-original-max-position 262144 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --media-preprocess-threads 16 \
  --max-concurrency 2 --max-pending-requests 8 --pending-timeout-ms 30000 \
  --prefill-chunk 8192 --max-request-mib 64
```

Then talk to the OpenAI-compatible endpoint:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"12+7 等于几？"}],"max_tokens":64}'
```

`./start.sh --help` prints the full option list.

## The two capabilities of this build

**4-bit KV on Ada (`rk4v4-e8`).** D256 Hadamard rotation plus an E8 Conway–Sloane lattice
projection, packed two nibbles per byte: **130 B per token per KV head**, about 2.5× smaller than
INT8. Also available: `bf16`, `int8`, `fp8`. `nvfp4` and `k8v4` are Blackwell-only and throw on
Ada — they need block-scaled MMA, `cvt.rn.satfinite.e2m1x2.f32` and TMA.

**Static YaRN context extension.** `--rope-yarn-factor 1..4` with
`--rope-original-max-position 262144` extends the per-request window using Qwen's published static
YaRN recipe. The positional cap, the causal attention envelope and the speculation draft window
all follow the factor. Native behaviour is the default (`--rope-yarn-factor 1`). Text, MTP and
multimodal MRoPE share per-Engine coefficients; the Vision tower stays native.

### Measured context ceilings (32 GB Ada card)

| Configuration | Ceiling |
|---|---|
| bare engine, `rk4v4-e8` | ~820,000 tokens |
| with `--vision` | ~720,000 tokens |
| vision + speculation + 2 lanes + tuning | **512,000 verified** (ceiling ~544K) |
| 1,048,576 | does not fit (~34.7 GiB KV needed) — needs a 48 GB class GPU |

The 1M rejection is the VRAM budget check, not a software limit.

### Measured throughput (512K, 2 lanes, `rk4v4-e8`)

| Speculation | Acceptance per lane | Decode |
|---|---|---|
| DFlash2 (`--draft-tokens 7`) | 36.1% / 41.8% | 83.4 / 91.5 tok/s |
| MTP (`--draft-tokens 3`) | 63.2% / 66.7% | 81.8 / 83.0 tok/s |

## Contents

```
bin/ninfer-serve        engine (NInfer code + sm_89 CUDA kernels)
lib/libav*.so.*         FFmpeg 6 (LGPL-2.1+), dynamically linked
lib/libcudart.so.12     CUDA 12.8 runtime
lib/libcurl.so.4        curl (HTTPS model fetching)
lib/libstdc++.so.6      GCC 12 C++ runtime (GLIBCXX_3.4.30)
lib/libgcc_s.so.1       GCC runtime support
start.sh                launcher
LICENSE                 Apache License 2.0
THIRD_PARTY_NOTICES.md
LGPL-2.1.txt
```

## Known limitation

The DFlash2 selector head must be left in a full-precision or `Q8`/`FP8` row format when the
artifact is converted. An artifact whose selector head was converted to NVFP4 is rejected at
startup with `linear_topk: unsupported head profile`. MTP is unaffected.
