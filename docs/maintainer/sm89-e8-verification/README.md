# Ada (sm_89) 4-bit KV verification kit — and what to re-run on a 5090

Everything here is machine-independent: pass the engine binary and artifact as arguments.
The numbers below are the **measured Ada baseline**; re-run the same commands on the 5090
and compare instead of re-deriving expectations.

## Measured baseline (RTX 4090-class 32 GB, 2026-09-25, 27B `qwen3.8-27b-lynn-*`)

| Configuration | Max context | VRAM | Speculative acceptance | decode |
| --- | --- | --- | --- | --- |
| bare, `--kv-dtype rk4v4-e8` | ~820,000 tok | 31.2 GiB | — | — |
| `--vision --spec dflash2` + `--max-concurrency 2` | ~544,000 tok | 30.3 GiB | 36.1% / 41.8% | 83.4 / 91.5 tok/s |
| `--spec mtp` (same artifact pair) | 693,888 tok KV | 31.1 GiB | **63.2% / 66.7%** | 81.8 / 83.0 tok/s |
| 1,048,576 tok (1M) | does not fit | needs ~34.7 GiB | — | — |

KV cost on this architecture: **18.28 KiB per token**
(16 full-attention layers x 4 kv heads x 260 B per token per head).

## Scripts

| Script | Purpose |
| --- | --- |
| `launch-512k.sh <engine> <artifact> [dflash2\|mtp]` | the validated 512K / 2-concurrent / vision / speculative launcher |
| `probe-max-context.sh <engine> <artifact> "<extra flags>" ctx...` | largest `--max-context` that starts; OOM- and death-aware, exits early |
| `needle-extrapolation.sh <endpoint> <model-id> [tokens]` | needle-in-haystack past the native window (YaRN) |
| `e8-encoder-probe.cpp` | standalone bit-exactness probe for the E8 K/V encoder (T>1, codec-edge data) |
| `ada-fix-patches.tgz` | the 18 commits that took the Ada port from "crashes / garbage" to working, as patches |

## What to re-verify on the 5090

1. **NVFP4 / K8V4 KV become available** — on Ada they are deliberate stubs
   (`src/ops/sm89/nvfp4_sm89_stubs.cpp`); a Blackwell build drops those and links the real
   routes.  `--kv-dtype nvfp4` and `--kv-dtype k8v4` should start where Ada throws.
   Compare their KV footprint against `rk4v4-e8` (18.28 KiB/token) at the same context.
2. **Context ceiling** — run `probe-max-context.sh` with the same flag sets.  The Ada ceiling
   is a pure VRAM budget check (`minimum Engine runtime reservation requires ...`), so expect
   it to move with the card, not with the architecture.
3. **YaRN extrapolation** — `needle-extrapolation.sh` at ~280K tokens (past the 262144 native
   window) must return `ZX-8147-QQ`.  On Ada it did, in the reasoning channel.
4. **Speculative acceptance** — MTP beat DFlash2 on Ada at equal settings.  Re-measure; the
   ranking may differ on Blackwell.
5. **Multimodal** — a text-in-image request must return the exact caption.

## Build prerequisites

* CUDA 12.8 toolchain and **CMake >= 3.28** — a 3.22 system cmake fails the configure in a way
  that is easy to miss, so use a pinned cmake (the Ada work used `/root/ninfer/tools/cmake`).
* Vision needs FFmpeg 6 via `PKG_CONFIG_PATH` and a matching `LD_LIBRARY_PATH` at runtime;
  `NINFER_ENABLE_VISION=ON` is required, otherwise image requests return `invalid_media`.

## Known open item

The synthetic suite's E8 expectations are still red: the test's own `ideal_attention` oracle
does not apply the output un-rotation that the engine now performs, so it compares a
rotated-domain reference against an un-rotated result.  The engine itself is verified
end-to-end by the scripts above; fixing the oracle is what would let CI guard the fix.
