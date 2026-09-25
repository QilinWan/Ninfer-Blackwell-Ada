# ninfer-tp2 — prebuilt binaries

Two-GPU fork of [Neroued/ninfer](https://github.com/Neroued/ninfer) (Apache-2.0), source and
documentation at https://github.com/ValerioDolci/ninfer-tp2. `COMMIT` records the exact commit.

## Contents

| Path | What |
|---|---|
| `bin/ninfer-serve` | OpenAI- and Anthropic-compatible server |
| `bin/ninfer` | command-line generation |
| `bin/ninfer-perplexity` | perplexity evaluator |
| `lib/libcudart.so.13` | the CUDA 13.1 runtime the binaries link (NVIDIA redistributable) |
| `serve-tp2.sh` | launcher with the two-GPU production profile |
| `SHA256SUMS` | digests of every file above; check with `sha256sum -c SHA256SUMS` |

## Requirements

- Linux x86_64 with the distribution named in the tarball (the binaries link its FFmpeg and libcurl:
  `libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libcurl4`). On another distribution
  build from source or use the Dockerfile.
- An NVIDIA driver that supports CUDA 13.1 (R580 or newer). No CUDA toolkit is needed.
- Two `sm_120` boards for `--tp 2` (validated on 2× RTX 5070 Ti 16 GB without P2P), or one RTX 5090
  for the single-GPU modes.

## Run

```bash
tar -xzf ninfer-tp2-*.tar.gz && cd ninfer-tp2-*/
sha256sum -c SHA256SUMS
./serve-tp2.sh /path/to/qwen3_8_27b_quasar_nvfp4.ninfer          # two boards, production profile
NINFER_PORT=8099 ./serve-tp2.sh /path/to/model.ninfer --api-key secret   # extra flags are appended
```

Artifacts: `Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer` (17.4 GB, verified with this fork, SHA-256 in
its `SHA256SUMS`) or the official `Qwen3.8-27B-nvfp4-NInfer`; both run with the same flags.

`GET /health` answers 200 while the Engine accepts work and 503 after an Engine-wide failure; the
server then exits with status 2, so run it under a supervisor that restarts on a non-zero exit
(`Restart=on-failure`, or llama-swap, which reloads the model on the next request).
