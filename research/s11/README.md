# S11 — multimodal (image/video) prefill at tensor-parallel width 2

This directory archives the S11 project that ran in `/data/s11` (workspace now retired):
the plan, the per-package notes, the harness used for the measurements, and the serving
recipe for the local `:8081` llama UI front end.

The engine code is on this branch (`blackwell-tp2`, formerly `s11-multimodal-tp2`):
`e6977697` (handoff) + `52ea073b` (tp 2 vision-receiver plan fix).

## Layout

| Path | Content |
|---|---|
| `plan.md` | the 383-line development plan (mechanism, P0-P6, work packages, gates) |
| `notes/` | one note per package: build, W1 repro, text parity, handoff, multi-image, MTP, tp 2 suites, gate, delivery, performance, 512K, 8081 serving, DFlash2 |
| `harness/` | the scripts used to measure: `serve-start.sh`, `chat-request.py`, `bench.py`, `run-60.py`, `think-test.py`, `text-prompts-60.json`, the bench wrappers |
| `serve/` | `ninfer-ui-proxy.py` (llama.cpp-shaped front end for ninfer-serve) and `model-8081.md` (the model.sh slot and launch recipe) |

## Reproduce the build

```bash
# deps prefix (not in git): libcurl 8.11 + FFmpeg 6.1.1, see notes/W0a-done.md
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
cmake -S . -B build-120a-cu13 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_ENABLE_VISION=ON -DBUILD_TESTING=ON \
  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON -DCMAKE_PREFIX_PATH=<deps prefix>
cmake --build build-120a-cu13 -j14
```

The harness scripts carry the `/data/s11/...` paths they were written with; adjust
`--log`/`--image`/`MODEL` arguments or the `S11_*` environment overrides when reusing them.

## Serving recipe (:8081)

See `serve/model-8081.md`. The engine binary and its runtime libraries live in
`/tank/Ninfer/bin/{apps,lib}`; the artifact is
`/tank/models/kybrcore/Qwen3.8-27B-QUASAR-NVFP4-NInfer/qwen3_8_27b_quasar.ninfer`
(W4A4 + vision + MTP + DFlash2), and the UI dist is `/tank/Strata/web-llama`.
