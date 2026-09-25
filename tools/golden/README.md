# tp 1 golden gate: this fork against upstream, token for token

`--tp 1` is meant to be upstream's single-GPU engine, untouched. This directory turns that
intention into a check: the same runner source, built against this fork and against the upstream
commit the fork is rebuilt on, must generate identical token ids from identical prompts.

## What runs

- `tp1_golden.cpp` — loads an artifact on one device through the public `ninfer::Engine` API,
  prepares a deterministic byte-id prompt (`prepare_tokens`, no chat template, no prefix reuse),
  decodes it greedily (temperature 0) and prints the generated ids. Public headers only, so it
  builds unchanged in both trees.
- `record.sh` — the three cases, chosen for their prefill regimes rather than their content:

  | case | prompt tokens | prefill chunks (1024) | extra flags |
  |---|---:|---:|---|
  | 1 short | 32 | 1 | `--max-context 4096` |
  | 2 medium | 2,048 | 2 | `--max-context 4096` |
  | 3 long | 6,144 | 6 | `--max-context 8192 --kv-dtype int8` |

  Common set: `--seed 7 --max-new 128`. `diff -r` between two recorded directories is the gate.

## The artifact

No Qwen3.8-27B artifact fits one 16 GB board (the smallest, QUASAR NVFP4, is 17.4 GB of weights),
so on the hardware this fork is developed on the gate runs the synthetic two-layer Qwen3.5 model
of `tests/models/qwen3_5/test_text_context_tp2.cpp` (one Gated DeltaNet block, one full-attention
block, FP8 projections with BF16 norms, a byte-level tokenizer). That executable writes it:

```bash
./build/tests/ninfer_qwen3_5_text_context_tp2_test write /tmp/synthetic   # -> /tmp/synthetic/model.ninfer
```

Scope: the Engine, scheduler, prefill/decode programs, attention, GDN, FP8 and BF16 Ops on one
device. Not covered: the NVFP4 and INT-quantised GEMMs (no such artifact fits at tp 1 here). With
a board that holds a 27B artifact, `record.sh` takes it as-is and covers them too.

## Building the runner in both trees

This fork: `ninja -C build ninfer-tp1-golden` (registered in `apps/CMakeLists.txt`).

Upstream, in a detached worktree at the fork's base commit (the runner is not part of upstream):

```bash
git -C /path/to/upstream worktree add --detach /path/to/upstream-base <base-commit>
cp tools/golden/tp1_golden.cpp /path/to/upstream-base/apps/
cat >> /path/to/upstream-base/apps/CMakeLists.txt <<'EOF'
add_executable(ninfer-tp1-golden tp1_golden.cpp)
target_link_libraries(ninfer-tp1-golden PRIVATE ninfer_engine)
EOF
cmake -S /path/to/upstream-base -B /path/to/upstream-base/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=OFF
ninja -C /path/to/upstream-base/build ninfer-tp1-golden
```

Then, with the artifact written above:

```bash
tools/golden/record.sh /path/to/upstream-base/build/apps/ninfer-tp1-golden /tmp/synthetic/model.ninfer out/upstream
tools/golden/record.sh build/apps/ninfer-tp1-golden                         /tmp/synthetic/model.ninfer out/fork
diff -r out/upstream out/fork && echo IDENTICAL
```

## Recorded runs

See `recorded/`: one directory per run with both sides' `case-*.ids` and the commits compared.
