# Qwen3.5/3.8-9B on the Ada (sm_89) engine — branch `qwen3.5-9b-ada`

Branch `qwen3.5-9b-ada` makes the Ada port of NInfer run the Qwen3.5/3.8-9B
geometry — the exact geometry of the published `qwen3.5-9b` / `groupwise-int`
`.ninfer` artifacts (32 layers, hidden 4096, 16 Q heads × 256, 4 KV heads × 256,
24 linear-attention layers with 32 GDN value heads, FFN 12288).

Until now the Ada port implemented exactly two densities (the 27B and the 35B-A3B
MoE geometry). The fused Op catalogs (`attn_input_proj`, `gdn_input_proj`,
`linear_swiglu`, `linear_add`, `gdn_norm_gating_proj`, the per-shape `linear`
tables) are closed sets of those shapes and reject anything else, and the shape
gates are spread over ~43 files. Extending all of them is a multi-week port.

This branch takes the other route: it transplants the **composed projection
routes** first written for the Blackwell engine on the `qwen3_5_9b-v3` branch of
`BenJGadd/ninfer` (a mirror of `Neroued/ninfer`, Apache-2.0). See NOTICE.

## What the composed route does

`src/models/qwen3_5/execution/composed.{h,cpp}` runs the same mathematics through
shape-generic Ops (`linear`, `rmsnorm`, `silu_mul`, `residual_add`,
`gdn_gating`, the family's projected causal-convolution launcher) whenever the
plan catalogs report that a geometry is *not* registered:

| leaf | fused (registered) | composed (unregistered) |
|---|---|---|
| attention input | `attn_input_proj(pair)` | four `linear` calls on the logical query/key/gate/value |
| GDN input (prefill) | `gdn_input_proj(pair)` | `linear(query|key|value)` + `linear(z)` |
| GDN snapshot / record | `gdn_input_proj_conv_{snapshot,record}` | the two linears + `gdn_projected_conv_*_launch` |
| GDN control | `gdn_norm_gating_proj` | `rmsnorm` + `linear(a)` + `linear(b)` + `gdn_gating` |
| output projection | `linear_add` | `linear` + `residual_add` |
| dense FFN | `linear_swiglu` + `linear_add` | the leaf's existing MTP composition |

The choice is made once per block in `parameters.cpp` by asking the plans'
`*_admits` predicates with the artifact geometry, so the 27B/35B paths are
untouched and a future fused catalog entry for a 9B shape flips that block back
with no further change.

## Ada-specific work on top of the transplant

1. **NVFP4 shape stubs** — `cmake/Sm89Compat.cmake` drops the Blackwell-only
   NVFP4 translation units, so `nvfp4_dispatch.cpp` needs stub definitions for
   the seven new 9B shapes (`src/ops/sm89/nvfp4_sm89_stubs.cpp`). A groupwise-int
   artifact never selects them.
2. **BF16 `W[32,4096]`** — the 9B GDN control projection is BF16; the BF16
   catalog registers three exact 27B shapes and the transplant only added generic
   fallbacks to the q4/q5/q6/q8 dispatchers. `src/ops/linear/bf16/shapes/n32_k4096.cu`
   adds a 32-row-CTA / 32-row-MMA geometry (untuned but correct).
3. **Split parent layout** — the transplant assumes `query|key|value` live in one
   contiguous artifact parent (the maintained recipes group them). Artifacts from
   the `groupwise-int` converter split them into `query_key` (Q4) and `value_z`
   (Q5) — different formats, so they cannot be merged into one weight. The
   composed GDN projection therefore stages the two halves and copies them into
   the two channel bands of the convolution input with `cudaMemcpy2DAsync`
   (`composed.cpp`, plus matching `composed_gdn_*_workspace_bytes`).
4. **Server-side content log** (`--request-log-content FILE`) — appends a JSONL
   record per request with the raw client body and the generated text/reasoning
   (rotated at 64 MiB). Independent of `--request-log-jsonl`, which carries
   metadata only. Adds the raw body to `RequestLogContext`.
5. **`SO_REUSEADDR`** — httplib's `default_socket_options` only sets
   `SO_REUSEPORT`, so a restart while previous connections sit in TIME_WAIT
   fails with `cannot bind <host>:<port>`. `configure_http_server_socket` now
   sets `SO_REUSEADDR` too.

## Build and run

Unchanged from the rest of this repository (`cmake -S . -B build -G Ninja
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`, then `ninja -C build`).
CUDA 13.3 was used here; the Ada build needs CUDA ≥ 12.8 (`<cuda_fp4.h>`).

```bash
./build/apps/ninfer-serve models/qwen3_5_9b.ninfer \
  --host 0.0.0.0 --port 8090 --max-context 262144 --kv-capacity auto \
  --kv-dtype int8 --spec mtp --draft-tokens 3 \
  --chat-template tools/chat_templates/qwen3_8.jinja
```

## Measured on an RTX 4070 (12 GB, Ada)

| | |
|---|---|
| weights | 5.28 GiB (5.52 GiB with the MTP head resident) |
| prefill | ~2.8k tok/s on a long prompt, ~280 tok/s on a one-line prompt |
| decode | 130–160 tok/s with MTP draft 3 (24 layers are linear-attention, so only 8 layers attention a real KV cache) |
| server max context | bf16 165,376 tokens · **int8 262,144 (the model's native window)** · fp8 262,144 |
| perplexity | 1.17 on a short English corpus (2048/1024, int8 and bf16 within 0.0005) |
| thinking | off by default in this deployment; on request it reasons and answers (`reasoning_content`), effort none/minimal/low/medium/high/xhigh/max |

## Known defect

The ordinary (non-speculative) decode step is broken: it emits the first sampled
token forever. Prefill is validated by perplexity and the speculative path
(`--spec mtp`) produces correct text, so the deployment runs with speculation on.
The defect is specific to T=1: T≥2 verification blocks are correct, KV dtype does
not matter, and `--no-cuda-graph` does not change it. Diagnosis so far points at
state advance in the single-token step (KV append / conv state / recurrent
state); it is **not** in the composed routes, which prefill exercises for every
token of a request.
