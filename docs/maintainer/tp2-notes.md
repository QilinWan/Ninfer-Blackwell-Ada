# TP2 strip drill, the placeholder register and the Ada follow-up

This file is the maintenance companion of [Merging upstream](upstream-merge.md), which owns where
the two-GPU code lives, how it attaches to upstream's files and the mandatory gate
([§1](upstream-merge.md#1-where-our-code-lives), [§4](upstream-merge.md#4-the-gate-mandatory-for-every-merge-and-every-tp2-change)),
and of [Two-GPU tensor parallelism](tensor-parallel.md), which owns the product contract. What is
recorded here is only what those documents do not cover: how to peel the layer off this fork, which
numbers in the layer are placeholders, which batch is deferred, and what Ada still owes.

## 1. The layer in this fork, in one list

This fork carries the layer **inline** in each layer's own sources; the upstream layout that moves
it into `execution/tp2/`, `program/tp2/`, `ops/*/tp2/`, `include/ninfer/ops/tp2/` and
`runtime/engine/tp2/` is deferred (§4). The faces, newest first:

- `src/core/tp2/`: `decode_graph_peer.h` + `decode_graph_tp2.inc` (the dual-device capture),
  `paged_kv_cache_tp2.inc` (the KV mirrors), `device_tuning.h` (the tuning table, §3).
- `src/ops/common/`: `allreduce.cu`, `peer_mailbox.cu`, `peer_exchange.cuh` (transports, the
  pipelined kernel with `NINFER_TP_MAILBOX_LEGACY=1`, the degrade path), `split_launch.h` (validate
  both ranks before either issues work); the per-format split entries (`linear` row/column-parallel,
  `linear_add` row-parallel, `linear_swiglu`/`attn_input_proj`/`gdn_input_proj` column-parallel) and
  the head-local `[256,12,2]` causal geometry for BF16, INT8-G64 and NVFP4-G16.
- `src/models/qwen3_5/`: `execution/tp.{h,cpp}` (the rank array and per-rank views),
  `execution/text.cpp` (the split schedules; `require_single_device` guards every entry with no
  split form), `execution/draft.cpp` (the drafter, whole on rank 0),
  `program/program_impl.{h,cpp}` (`PeerRuntime`: rank 1's persistent layout and arena),
  `program/planning/startup.cpp` (the width-2 workspace plan, per-rank arena shares, the causal
  score-tile phase, the Vision phase on the encoding rank, the per-rank memory plan).
- `src/runtime/engine/`: `engine.cpp` (`initialize_execution`, peer access, the S6 capability
  verdict per rank), `model_instance.cpp` (two-device construction and rank budgets, the S6 KV
  verdict), `kv_capacity.cpp` (`resolve_kv_capacity_symmetric`: one page count, the tightest rank).
- `src/product/`: `tensor_parallel_options.h` (option parsing and the two pure verdicts),
  `vision_options.h` (`--vision-device` must name one of `--devices`).
- Tests: five `test_*_tp2_*.cpp` suites in `tests/models/qwen3_5/`, the `test_*split*.cpp` Op
  halves and `test_attention_headlocal.cpp`; all return the named skip code 77 below two devices,
  which is why the cluster gate is green on one board.
- Tools: `tools/tp2/` (`gate.sh`, `check_shard_sources.py`, `mailbox_probe.cu`, `ppl_paired.py`,
  `surface.sh`, `reference/`).

## 2. Peel drill: strip TP2 and keep building

The layer is additive over the width-1 product, and `--tp 1` is meant to stay upstream's engine
([tensor-parallel.md §9](tensor-parallel.md#9-tp-1-identity)): a strip that changes a byte of the
single-device path is wrong, not partial. Run the drill in a scratch worktree with its own build
directory, never in a tree that carries a gate run.

The layer's boundary in this fork's own history, measured (not inherited from a note): the
oldest face is `03726c08 feat(ops): register tensor-parallel shard shapes for fp8, nvfp4 and bf16
linear` (the half-shape registration), and everything from its parent to `HEAD` is the layer and
its follow-up = **197 commits**. The `v-tp2-skeleton` tag is **not an ancestor of `HEAD`** (the
layer arrived through the replayed donor steps, whose history is not linear), so the peel base is
that parent, not the tag.

```bash
git worktree add ../peel-drill HEAD              # isolated tree, own build dir
cd ../peel-drill
# Stage 1: peel a bounded newest segment (25 commits) to measure the method and the conflicts.
git revert --no-commit HEAD~25..HEAD
git commit -m "drill: strip a bounded tp 2 segment"
JOBS=14 nice -n 12 bash scripts/build-ada.sh 120a --tests      # criterion: rc=0
# Stage 2: peel the whole layer.
git revert --no-commit 03726c08~..HEAD
git commit -m "drill: strip the tensor-parallel layer"
bash scripts/build-ada.sh 120a --tests            # criterion: rc=0
git grep -n "tensor_parallel\|TpExecution\|PeerRuntime\|peer_mailbox"   # criterion: 0 hits
```

Judged by the build and by the grep, not by the tests. Expect the full-range peel to stop for hand
work, and expect the hand work mostly **in code**, not in the product prose this fork also edits:
measured on `340bd4c0` by reverting each of the 197 layer commits one at a time on a clean tip in a
scratch worktree (every step reset hard, so each verdict states whether *that* commit peels alone) -
**76 peel cleanly, 135 need hand work**, and the files that collide are, in order, `README.md` 16,
`docs/serving.md` 14, `docs/cli.md` 14, `planning/startup.cpp` 13, `tensor-parallel.md` 12,
`serve_options.cpp` 11, `model_instance.cpp` 11, `execution/text.cpp` 10, `program/graphs.cpp` 8,
**`include/ninfer/types.h` 8** (`KvCacheStorage`/`EngineOptions` carry the layer's enums),
`qwen3_5-model.md` 8, `tools/tp2/README.md` 7. The bounded stage is cheap and clean: peeling the
newest 25 commits takes **no** manual edit at all (48 files, -1231/+213), and it leaves the deep
symbols (`TpExecution` 11 files, `allreduce_sum` 14, `peer_mailbox` 9) still referenced exactly as
expected - so `git grep` = 0 is a stage-2 criterion only. Record every hand edit in the ledger; the
drill is the measurement, not a cleanup. Two traps this fork measured, which a peel hits first:

- **A removed definition still referenced by kept code.** Retire symbol by symbol, not file by
  file: `git grep <type>` for the consumers before deleting, because whole-file reversion strips
  fields other faces compile against (this fork shipped `PrefillMissReport does not name a type`
  that way).
- **Moved and orphaned files.** A move-only step leaves an orphan when its base is skipped, and a
  dangling include surfaces only in a full build (this fork shipped
  `ninfer/ops/tp2/argmax.h` that way). After every removal check both sides:
  `git ls-tree -r HEAD --name-only | grep <path>` and `git grep -l <header>`.

Record the outcome, including every file the strip had to touch by hand, in the maintenance ledger.

## 3. Placeholder register (measure on this board, then write back)

`src/core/tp2/device_tuning.h` keys by compute capability and SM count; the 5060 Ti row
(CC 12.0, **36 SM**, measured here by `cudaDeviceGetAttribute`) carries numbers inherited from the
70-SM RTX 5070 Ti device and **not measured here**:

| Field | Current value | Source | Measured by |
|---|---|---|---|
| the three graph-allowance classes | 8 / 8 / 11 MiB | 70-SM notes | D2 capture accounting per class, per rank |
| shard TMA boundary | 1024 tokens | 70-SM retune (the 128-token tile rule) | D4 shard bench sweep at T = 1, 4, 8, 16, 32, 512, 1024 |
| NVFP4 12/2 `row_tiles` group and key shift | group of six, shift 7/8 | inherited with the 12/2 instantiation | D4 causal-attention bench at 36 SM; the plan's SM bounds read the device count, not the table |
| causal head counts per device | 12 query / 2 KV | structural (half of 24/4) | already fixed by the shard geometry |

Until those are measured, the plan's conservative values stand and the startup log's launch plan
must report 36 SM (the S7 acceptance point).

## 4. Deferred batch: the upstream layout move

Upstream moves the code of §1 into `tp2/` files and `.inc` fragments, and the two-device Vision
receiver (`WorkspacePlan::vision_receiver` plus the event-synchronized copy of the merged
embeddings), the split DFlash2 drafter (its top-k proposal split, candidate selector on rank 1) and
the wide mailbox slots land on that base. This fork skipped the moves as a block, so those
capabilities are not reachable piecemeal: adopting them is one batched change with a full build and
the cluster gate as its escort. The user-visible consequence today: `--vision` at `--tp 2` keeps the
tower resident on rank 0, and a prompt that carries media is rejected on the two-device schedule
([tensor-parallel.md §8](tensor-parallel.md#8-vision-at-tp-2)).

