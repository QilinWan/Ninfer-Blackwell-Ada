Prebuilt, **relocatable** NInfer engine bundle for Ada, with vision (image/video input) enabled.

The target host needs **no compiler and no CUDA Toolkit** — the archive carries its own
FFmpeg 6 and CUDA 12.8 runtime libraries.

## Requirements

| | |
|---|---|
| GPU | NVIDIA Ada, compute capability **sm_89** — RTX 4090 / 4090D / 4080 SUPER / 4070 family |
| Driver | NVIDIA driver with CUDA 12.8 runtime support (validated on 595.71.05) |
| OS | Linux x86_64, glibc ≥ 2.35 (Ubuntu 22.04 or newer) |
| Model | a v3 `.ninfer` artifact — **not included**, weights are decoupled from the engine |

> **Not for Blackwell (sm_120a).** The Ada and Blackwell targets are separate builds.

## What changed in r2

r1 is withdrawn: its `rk4v4-e8` route emitted garbage. The E8 codec rotates Q, K **and** V, so the
attention output carries the rotation and has to be rotated back before it leaves the fused kernel.
r1 was missing that step, and every downstream number it produced was noise. r2 carries all eight
fixes found by the audit:

| Fix | Symptom it removes |
|---|---|
| Inverse E8 output rotation in the decode (small-T) and prefill (prompt) attention paths | garbage text on `rk4v4-e8`; the answer went from empty to correct |
| E8 page-table publication barrier before the fused append reads it | deterministic `cudaErrorIllegalAddress` at `small_t_e8.cu:98` |
| `RK4V4E8` dispatch in the batched KV append launcher | 32 out-of-bounds writes reached from E8 prompt attention |
| Four distinct arena planes in the E8 small-T kernel | FP16 V staging overlapping the packed K plane |
| YaRN factor in the startup position cap | `max_context exceeds the configured position capacity` for any context above 262,144 |
| YaRN factor in the speculation draft window cap | `max_context exceeds the selected draft position capacity` |
| Causal attention envelope raised from 262,144 to 1,048,576 keys | `causal_softmax_attention workspace: invalid profile or interval` |

With those in place both headline capabilities are usable on Ada:

* **4-bit KV — `--kv-dtype rk4v4-e8`.** A D256 Hadamard rotation plus a per-8-dimension E8
  Conway–Sloane lattice projection, two nibbles per byte: **130 B per token per KV head**, about
  2.5× smaller than INT8. The K and V planes decode to INT8 and feed the same s8 tensor-core QK /
  FP16 PV route as `int8`. Also available: `bf16`, `int8`, `fp8`. `nvfp4` and `k8v4` are
  Blackwell-only and refuse on Ada by construction — they need block-scaled MMA,
  `cvt.rn.satfinite.e2m1x2.f32` and TMA.
* **Static YaRN context extension — `--rope-yarn-factor 1..4` with
  `--rope-original-max-position 262144`.** Qwen's published static recipe, applied to the
  positional cap, the causal attention envelope and the speculation draft window alike; the split
  geometry already scales with `max_visible_keys`, so no kernel changed. Text, MTP and multimodal
  MRoPE share per-Engine coefficients; the Vision tower stays native. The default
  (`--rope-yarn-factor 1`) is unchanged native execution.

### Measured context ceilings (32 GB Ada card, all at factor 4)

| Configuration | Ceiling |
|---|---|
| bare engine, `rk4v4-e8` | **~820,000 tokens** |
| with `--vision` | ~720,000 tokens |
| vision + speculation + 2 lanes + tuning | **512,000 verified** (ceiling ~544K, 30.3 GiB) |
| 1,048,576 | does not fit — ~34.7 GiB of KV needed; the engine refuses at the VRAM budget check |

### Measured throughput (512,000 tokens, 2 lanes, `rk4v4-e8`, vision on)

| Speculation | Acceptance per lane | Decode |
|---|---|---|
| DFlash2, `--draft-tokens 7` | 36.1% / 41.8% | 83.4 / 91.5 tok/s |
| MTP, `--draft-tokens 3` | 63.2% / 66.7% | 81.8 / 83.0 tok/s |

## Usage

```sh
tar -xf ninfer-sm89-cu128-r2-x86_64.tar.xz
cd ninfer-sm89-cu128

./start.sh /path/to/qwen3.8-27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 512000 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --rope-yarn-factor 4 --rope-original-max-position 262144 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --media-preprocess-threads 16 \
  --max-concurrency 2 --max-pending-requests 8 --pending-timeout-ms 30000 \
  --prefill-chunk 8192 --max-request-mib 64
```

`start.sh` resolves its own directory and points `LD_LIBRARY_PATH` at the bundled `lib/`, so the
bundle can be extracted to any path. It carries no absolute path from the build host. `README.md`
inside the archive repeats this example; `./start.sh --help` lists every flag.

## Assets

| File | Size | SHA-256 |
|---|---|---|
| `ninfer-sm89-cu128-r2-x86_64.tar.xz` | 151.0 MiB (158,364,508 B) | `fc45942ac608d72cbf88326c1aaf8120facffd86f3e1b1aba0ba51c8bbed7fa6` |
| `ninfer-sm89-cu128-r2-x86_64.tar.xz.sha256` | — | checksum file |

## Bundle contents

```
bin/ninfer-serve        engine (NInfer code + sm_89 CUDA kernels)
lib/libav*.so.*         FFmpeg 6 (LGPL-2.1+), dynamically linked
lib/libcudart.so.12     CUDA 12.8 runtime
lib/libcurl.so.4        curl (HTTPS model fetching)
lib/libstdc++.so.6      GCC 12 C++ runtime (GLIBCXX_3.4.30)
lib/libgcc_s.so.1       GCC runtime support
start.sh                launcher
README.md               quick start and measured numbers
LICENSE                 Apache License 2.0
THIRD_PARTY_NOTICES.md
LGPL-2.1.txt
```

`libstdc++` and `libgcc_s` are bundled so the engine's `GLIBCXX_3.4.30` requirement is
met from inside the bundle. Without them, activating a conda environment whose
`$CONDA_PREFIX/lib` lands on `LD_LIBRARY_PATH` would shadow the host copy with an older
`libstdc++` (conda ships 3.4.29) and the engine would fail to load.

The bundle ships the server engine, as r1 did. The offline `ninfer` CLI is not included — see
*Known limitations*.

## Build provenance

- Source: repository tree at `381f43d1` (tree `1589d092f74db6d65a7eef95222d101cd57941a5`)
- Configure: `-DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_BUILD_APPS=ON -DNINFER_ENABLE_VISION=ON`
  (`-O3 -DNDEBUG -DNINFER_SM89=1`)
- Toolchain: CUDA 12.8.93, GCC, Ubuntu 22.04.5 LTS, FFmpeg 6.1 (`libavcodec` 60.31.102)
- Built **on the sm_89 target machine** (RTX 4080 SUPER 32 GB) — not cross-compiled

## Verification

- `rk4v4-e8` end to end at 512,000 tokens with vision on, two concurrent lanes: MTP accepted
  63.2% / 66.7% (decode 81.8 / 83.0 tok/s), DFlash2 accepted 36.1% / 41.8%
  (83.4 / 91.5 tok/s); text answers correct and the image question returned the expected code.
- A 512K needle-in-a-haystack run at factor 4 retrieved the planted value.
- One cubin covers the whole Ada family: SM count, L2 size and residency come from the device at
  runtime, so a 4090, a 4090D and a 4080 SUPER share this binary.
- The archive extracts and launches from an arbitrary directory; the dependency closure resolves
  entirely from the bundled `lib/` (`ldd` reports zero `not found`, and `libstdc++`, `libgcc_s`,
  `libcudart`, `libav*` and `libcurl` all resolve inside the bundle).
- `bin/ninfer-serve` inside the archive is byte-identical to the binary that was smoke-tested on the
  RTX 4080 SUPER — SHA-256
  `92e1a92093afde1ba7f5d7b9737d91582ab374661450036f2b078f7a95c6e905`.

## Known limitations

* **Offline CLI.** The `ninfer` and `ninfer-perplexity` argument parsers in this tree accept only
  `bf16|int8|fp8|nvfp4|k8v4`; they cannot request the Ada-only `rk4v4-e8` route. The serving
  surface (`bin/ninfer-serve`) is unaffected and is what this bundle ships.
* **DFlash2 selector format.** The DFlash2 selector head must stay in a full-precision, `Q8` or
  `FP8` row format at conversion time. An artifact whose selector head was converted to NVFP4 is
  rejected at startup with `linear_topk: unsupported head profile`. MTP is unaffected.

## Licensing

NInfer is Apache-2.0. The bundle redistributes FFmpeg 6 shared libraries (LGPL-2.1+) and the
NVIDIA CUDA runtime. The FFmpeg libraries are dynamically linked and ship as replaceable
shared objects under `lib/`, so they can be substituted without relinking the engine —
see `THIRD_PARTY_NOTICES.md` inside the archive.
