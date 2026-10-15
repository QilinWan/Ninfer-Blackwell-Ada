# This release in two lines: 4-bit KV on sm_89, and context extrapolation across both architectures

> For RTX 40-series (Ada / sm_89) and RTX 50-series (Blackwell / sm_120a).  Every number below
> is measured, not estimated.

## 1. 4-bit KV on sm_89 (Ada / RTX 4080S) — new in this release

Ada had no usable 4-bit KV route.  This release adds **`rk4v4-e8`**, an E8 Conway-Sloane
lattice codec that stores both K and V as 4-bit codes:

| | |
| --- | --- |
| K | D256 Hadamard rotation (x1/16), then per-8-dim E8 lattice projection, `rintf` clamped to [-8,7] |
| V | the same rotation, then `__float2int_rn` clamped to [-7,7] |
| Cost | **130 bytes per token per kv head** (128 code bytes plus one FP16 scale each) |
| Versus int8 KV | about **2.5x smaller** |
| Decode | nibbles unpack to int8, fed to the s8 tensor-core QK and the FP16 PV MMA |

Measured on a 32 GB Ada card with the 27B artifact: correct text, correct image caption,
DFlash2 acceptance 36.1% / 41.8% at 83.4 / 91.5 tok/s and MTP acceptance 63.2% / 66.7% at
81.8 / 83.0 tok/s.

`nvfp4` and `k8v4` are deliberately unavailable on the same card: they need Blackwell FP4
conversion, block-scaled MMA and TMA, and `src/ops/sm89/nvfp4_sm89_stubs.cpp` raises instead
of computing something wrong.  On Ada the 4-bit route is `rk4v4-e8`.

## 2. Context extrapolation (YaRN) on both architectures

Static YaRN covers both generations: `--rope-yarn-factor F` (1 to 4) with
`--rope-original-max-position 262144` widens the usable window to F times the native one.
The positional cap, the causal attention envelope and the draft window all follow the factor,
and the split geometry already scales with `max_visible_keys`, so no kernel change was needed.

| Configuration (32 GB card) | sm_89 / Ada (`rk4v4-e8`) | sm_120a / 5090 (`nvfp4`) |
| --- | --- | --- |
| bare | **~820,000 tokens** | **~790,000 tokens** |
| with vision | ~720,000 tokens | ~730,000 tokens |
| with vision + speculation + 2 lanes + tuning | **512,000 verified** (ceiling ~544K) | **580,000 verified** |
| 1,048,576 | does not fit (~34.7 GiB needed) | does not fit |

The 1M limit is a VRAM budget check, not a software limit; it needs a 48 GB class GPU.

## 3. What r2 fixes over r1

r1 emitted fluent-looking noise on `rk4v4-e8`.  The encoding was correct; the **attention
output was missing its un-rotation** (`V_stored = H*V` implies `O = H*O_true`, so H must be
applied once more).  r2 also fixes three further defects: the page table is published before
the fused append reads it, the batched append dispatches RK4V4E8 instead of falling through to
the BF16 kernel, and the small-T arena no longer overlaps.  Use r2 or newer for E8.

## 4. Known limitation: DFlash2 on the `nf4` artifact

`linear_topk` (the DFash2 draft selector) accepts only `Q8_G32_FP16`,
`FP8_E4M3FN_ROW_BF16` or `Q4_G64_FP16`; the `nf4` conversion left that head in NVFP4, so the
server refuses to start.  That is a weight-conversion oversight, not an engine defect - the
target weights load as NVFP4 normally.  MTP is unaffected.
