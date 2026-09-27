# Two-GPU tensor parallelism

This fork adds a two-device tensor-parallel mode, `--tp 2 --devices A,B`, so that a dense Qwen3.5
artifact too large for one 16 GB board (Qwen3.8-27B NVFP4, 20.9 GiB of weights) runs split across
two. This reference records the design of that mode from end to end. Narrower contracts stay with
their owners: the per-layer mathematics of the split schedule is in
[Qwen3.5 model](qwen3_5-model.md#tensor-parallel-execution), the column- and row-parallel Op forms
in [Op development](op-development.md#26-tensor-parallel-forms), the KV mirror in
[Paged KV context store](paged-kv-cache.md#56-tensor-parallel-mirror), parent slicing in
[Artifact container](artifact-container.md), and DFlash2 at width 2 in
[DFlash and DFlash2](dflash.md#tensor-parallel-execution). This document links them and owns the
contracts that cut across them. It was checked against the tree at `a0369376`. Points where the
code, its comments and the development history disagree are collected in
[Open questions](#open-questions).

## 1. Purpose and constraints

The width is exactly 1 or 2 at every layer that accepts it: `product::parse_tp`
([`tensor_parallel_options.h`](../../src/product/tensor_parallel_options.h)),
`runtime::normalize_engine_options` ([`model_instance.cpp`](../../src/runtime/engine/model_instance.cpp)),
`execution::kTensorParallelWidth` and `artifact::kMaximumDevices`. `--devices A,B` names one CUDA
device per rank, rank 0 first; `resolve_tensor_parallel_devices` makes `A` the Engine's `device` and
rejects an explicit `--device` that differs. `ExecutionContext` ([`device.h`](../../src/core/device.h))
rejects two devices of different compute capability, and the Qwen3.5 planner requires compute
capability 12.0.

Rank 0 is the Engine's device. Scheduler, ResourceManager, admission, sampling, request egress and
all page, execution-row and StateImage-slot bookkeeping run there. Rank 1 holds its shards of the
weights, its half of the KV planes and GDN state, its own workspace and a copy of every per-round
control tensor. It decides nothing; it replays at the same indices the physical mutations rank 0
issues (§6). The Engine and the context cache see one model, and the Program has one logical store.

**Peer access is not assumed.** `initialize_execution` ([`engine.cpp`](../../src/runtime/engine/engine.cpp))
calls `ops::enable_peer_access`, which enables P2P only when `cudaDeviceCanAccessPeer` holds in both
directions and otherwise returns false without failing (`LoadSummary::peer_access`). Every
cross-device transfer is a stream-ordered `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` over
unified virtual addresses: a direct PCIe copy with peer access, staged by the driver through host
memory without it, as on the GeForce boards the mode was verified on. `cudaMemcpyPeerAsync` is not
used because stream capture rejects it (`pull_peer` in
[`allreduce.cu`](../../src/ops/common/allreduce.cu)). When the driver grants peer access,
`Engine::Impl` turns the pinned-host mailbox off (§4): it was qualified only without P2P.

What the mode rejects, and where the first check sits:

| Rejected at tp 2 | First check |
|---|---|
| `--spec dflash` | `validate_options` in `model_instance.cpp`, before the artifact is read; again in the planner and `ProgramImpl` |
| `--spec dflash2` without `--lm-head-draft` | `validate_options`; `validate_tensor_parallel` in [`load.cpp`](../../src/models/qwen3_5/load.cpp); planner |
| DFlash2 drafter with full-attention layers | planner (`startup.cpp`): that drafter KV pool would have no rank 1 mirror |
| KV storage other than `bf16` and `int8` | `validate_options`; `TextContext`: the head-local `[256,12,2]` attention geometry is registered for BF16 and INT8-G64 only |
| nonzero Host State slots or Host KV bytes | `validate_options`; `ninfer-serve` turns omitted values into 0 and rejects explicit nonzero ones ([`serve_options.cpp`](../../src/serve/serve_options.cpp)) |
| MoE architecture | `validate_tensor_parallel(config)` in `load.cpp`; also `shard_text_config` and the shard rules |
| paired (two-parent) Q4/Q5 input projections | `TextContext` constructor ([`text.cpp`](../../src/models/qwen3_5/execution/text.cpp)); split routes in `attention.cpp` and `gdn.cpp` |
| a shard shape without a registered problem | the Op's shape registry, as for any unregistered shape |

`--tp 1` is meant to be upstream's engine. The mechanism is structural: the width-1 path passes
no `ExecutionContext` into loading and Program construction, and every two-device branch keys off
its presence. §9 describes this path and its gate.

## 2. Weight partitioning

The placement rules live in `shard_rule` ([`sharding.cpp`](../../src/models/qwen3_5/load/sharding.cpp),
documented in [`sharding.h`](../../src/models/qwen3_5/load/sharding.h)), keyed by logical parameter
name. Heads are split into contiguous per-rank blocks, which keeps every query head with its KV head
and every GDN value head with its key head, so no mixer core needs data from the peer. The
hidden/residual axis is never split.

| Logical parameter | Placement | 27B shape → per-rank shard |
|---|---|---|
| `attention/{query,gate}`, `attention/{key,value}` | Rows, by query heads / KV heads | packed Q\|K\|gate\|V parent `[14336,5120]` → `[7168,5120]` (12 Q, 2 KV heads of 256) |
| `attention/output` | Columns, by query heads | `[5120,6144]` → `[5120,3072]` |
| `gdn/{query,key}`, `gdn/{value,z}` | Rows, by key heads / value heads | packed Q\|K\|V\|Z `[16384,5120]` → `[8192,5120]` (8 key, 24 value heads) |
| `gdn/{a_projection,b_projection,a_log,dt_bias}` | Rows, by value heads | `a`, `b` `[48,5120]` → `[24,5120]` (BF16 for `gdn_gating_proj`); `a_log`, `dt_bias` 48 → 24 |
| `gdn/convolution` | Columns, three ranges (Q, K, V channel sections) | 10240 → 1024 + 1024 + 3072 channels |
| `gdn/output` | Columns, by value heads | `[5120,6144]` → `[5120,3072]` |
| `mlp/{gate,up}` | Rows | gate\|up `[34816,5120]` → `[17408,5120]` |
| `mlp/down` | Columns | `[5120,17408]` → `[5120,8704]` |
| `text/output_head` | Rows, by vocabulary | `[248320,5120]` → `[124160,5120]` |
| `mtp/input_projection` | Columns: rank 0 the embedding half, rank 1 the hidden half | `[5120,10240]` → `[5120,5120]` |
| `mtp/layers/*` | as the Text block rules | packed `[7168,5120]`, `[5120,3072]`, `[17408,5120]`, `[5120,8704]` |
| norms, `text/token_embedding` | Replicated | whole on both ranks |
| `vision/*` | SingleDevice(`vision_rank`) | whole on the Vision rank (§8) |
| `dflash/*`, `dflash2/*`, `proposal/*` | PrimaryOnly | whole on rank 0 |

A shard is a standalone weight of the parent's format and layout with one axis narrowed, never a
view into the parent's payload, so each rank runs the ordinary single-device dispatch at the shard's
own registered problem. A format has a two-device route only where its shard shapes are registered:

- **FP8** (`FP8_E4M3FN_ROW_BF16`): the attention and GDN input shards (`attn_input_proj`,
  `gdn_input_proj` column-parallel), `[5120,3072]` and `[5120,8704]` for `linear_add`,
  `[17408,5120]` for `linear_swiglu`, and the vocabulary half `[124160,5120]`
  ([`linear.h`](../../include/ninfer/ops/linear.h), [`linear_add.h`](../../include/ninfer/ops/linear_add.h)).
- **NVFP4**: the same projection shards (the attention and GDN input ones since `b14be3d5` and
  `7a740326`), but no vocabulary half.
- **Q8**: the MTP halves `[5120,5120]`, `[7168,5120]`, `[17408,5120]`, `[5120,3072]`, `[5120,8704]`;
  the official recipes store the MTP head in Q8, and it splits only there.
- **BF16**: GDN gating (`gdn_gating_proj_column_parallel`, 24 heads per rank), plus `[7168,5120]`
  and `[5120,3072]` for `linear`.

The groupwise-int artifacts are rejected because their attention and GDN input projections are
paired Q4/Q5 parents, which have no split route; any unregistered shard shape is refused like any
unregistered problem. The user docs ([README](../../README.md), [CLI](../cli.md#two-gpus)) summarize
this as "FP8 or NVFP4 split projections, MTP in Q8".

**Loading slices at load time.** `plan_load` ([`load.cpp`](../../src/models/qwen3_5/load.cpp))
builds an `artifact::Binder` for `options.tp` devices and calls `loading::install_shard_resolver`.
For every parent, `parent_placements` combines the placements of all parameters bound to it: row
ranges map through the Binding parts and must partition the parent, and column ranges must agree
across its parameters. `artifact::tensor_slice` ([`slices.h`](../../src/artifact/slices.h)) turns a
Rows or Columns placement into a shard geometry plus `PlaneCopy` byte ranges of the encoded parent;
nothing is repacked. Layouts bound the cuts (`BlockScaleK16M128x4` rows in 128-row tiles and columns
in multiples of 64, `RowSplit` columns in multiples of 128, several column ranges only for
`Contiguous`, as in the GDN convolution), and a column slice is one copy per row per range.
`artifact::materialize(reader, plan, ExecutionContext&)`
([`materializer.cpp`](../../src/artifact/materializer.cpp)) reads each file chunk once through the
pinned staging slots and cuts every device's copies from it. A shard-holding arena is created
zero-filled, since no copy writes the shard's plane-alignment gaps; other arenas, including every
arena at tp 1, are not zeroed, as upstream. Rank `r` then builds `execution::Parameters(model, r)`,
and `LoadSummary::devices` reports its sharded, replicated and rank-local bytes.

## 3. The forward pass at tp 2

`TextContext` runs the split schedule when it is given an `execution::TpExecution`
([`tp.h`](../../src/models/qwen3_5/execution/tp.h)) naming rank 1's Parameters, arena, GDN state
pool, KV caches, prefill KV row, ordinary-decode frame, ReplaySSM records and MTP storage; without
one it runs the single-device schedule unchanged. The Program owns one `TpExecution`, and
`ProgramImpl::prefill_tp_binding` copies it per prefill call to name the sequence's rank 1 MTP row.
Each Text layer (`TextContext::run_layers_tp2`) issues each rank's work on that device's stream:

```text
attention:  q|k|gate|v_r = attn_input_proj shard(rmsnorm(x))   column-parallel, no exchange
            a_r = gated head-local attention, rank r's 12 Q / 2 KV heads and its KV planes
            x   = x + all_reduce(o_r a_r)                       linear_add_row_parallel
GDN:        g|beta_r, q|k|v|z_r = column-parallel projections, then conv and recurrence
            x   = x + all_reduce(out_r a_r)                     rank r's 8 key / 24 value heads
MLP:        x   = x + all_reduce(down_r swiglu(gate|up_r rmsnorm(x)))
```

Each layer therefore contains exactly two all-reduces, 128 per decode token for the 64-layer 27B.
`ops::linear_add_row_parallel` ([`linear_add.cpp`](../../src/ops/wrapper/linear_add.cpp)) adds the
residual once: rank 0 runs `linear_add` into its residual copy, rank 1 overwrites its copy with its
plain partial, and `allreduce_sum` leaves `residual + partial_0 + partial_1` on both. Both ranks then
hold the identical BF16 residual, so every later per-rank input (norms, KV pages, GDN state) agrees
without further exchange. Prefill has its own chunk loop, `TextContext::prefill_impl_tp2`, separate
from the single-device `prefill_impl`.

**Logits are assembled on rank 0 only.** The final norm is replicated and each rank projects its
vocabulary half. In `execution::output_logits_split_rank0` ([`tp.cpp`](../../src/models/qwen3_5/execution/tp.cpp))
rank 1 records `inputs_ready(1)`; rank 0 waits on it, pulls rank 1's contiguous `[V/2, C]` partial
into rank-0 staging with one D2D copy, and `ops::detail::concat_rows_bf16_launch`
([`concat_rows.cu`](../../src/ops/launcher/concat_rows.cu)) interleaves both halves column by column
into the `[V, C]` logits; rank 1's stream waits on `pull_done(0)` before it may overwrite its
partial. The interleave used to be two `cudaMemcpy2DAsync` copies. `07f76beb` replaced them with the
kernel because a captured 2D memcpy node cannot be updated in place when its column count or buffers
change between CUDA Graph profiles of one class (§5). The same gather serves prefill, ordinary
decode, verification, the MTP proposals, the zero-suffix head and the score tiles of causal
scoring (`ProgramImpl::project_score_tile_split`); rank 1 keeps no logits.
`ops::allgather_rows` ([`allreduce.h`](../../include/ninfer/ops/allreduce.h)) is used only by tests.

**After the gather.** In an ordinary round ([`decode.cpp`](../../src/models/qwen3_5/program/decode.cpp))
`ops::sample`, the scatter of the final hidden into the continuation store and the egress copy all
run on rank 0's stream; rank 1's last work is its vocabulary half, and it idles until the next round
uploads its ingress. Verification computes the target argmax and the acceptance on rank 0; rank 1
receives the results by copy (§7) and selects and retains its own accepted hidden. MTP proposals
take their argmax on rank 0 over the gathered logits (`TextContext::proposal_argmax_tp2`), or, with
`--lm-head-draft`, rank 0 alone projects through the optimized proposal head.

Rank 1 takes its control tensors from its own upload of the host ingress record rank 0 receives
(`OrdinaryPeerFrame`; the MTP and DFlash2 frames likewise), not from copies of rank 0's device
frames, so tokens, positions, KV rows and state slots agree by construction. It never reads the
record's sampling configs.

## 4. Transport between the two devices

### 4.1 Staged copies

`ops::allreduce_sum` ([`allreduce.cu`](../../src/ops/common/allreduce.cu)) implements the transport
that works everywhere. Every transfer is a pull: rank `r` reads the peer's operand into storage only
it owns, on its own stream. Each rank records `inputs_ready(r)` and waits on the peer's before
pulling, then records `pull_done(r)` and waits on the peer's before its in-place combine; that
second wait is the write-after-read barrier that makes back-to-back calls on the same buffers safe
without host synchronization. The combine is the qualified `residual_add` body (FP32 accumulation of
the two BF16 operands, one round-to-nearest-even store). The events live in `ops::PeerEvents`, one
instance per stream pair, owned by the Program; inside a capture that enrolls both streams they
become graph edges. Without peer access the driver stages each pull through host memory, and the
event chain costs more than the payload.

### 4.2 The pinned-host mailbox

`ops::PeerMailbox` ([`peer_mailbox.h`](../../include/ninfer/ops/peer_mailbox.h),
[`peer_mailbox.cu`](../../src/ops/common/peer_mailbox.cu)) owns a pinned, UVA-mapped host slab (per
rank and slot one payload and one release word, plus a hang word) and per-device arrival and epoch
words. `captured_mailbox` in `allreduce.cu` routes an `allreduce_sum` through it when a mailbox is
attached to the `PeerEvents` and serves this pair, no `PeerEvents::StagedScope` is open, the payload
is whole aligned 16-byte vectors within one slot, and **both** ranks' streams are capturing into the
same capture. The call then becomes one exchange kernel per device
([`peer_exchange.cuh`](../../src/ops/kernel/peer_exchange.cuh)): each writes its partial to its own
host slot, fences, releases a per-slot epoch flag, spins on the peer's flag, reads the peer's slot
and combines with the staged path's arithmetic, so the two transports are bit-identical. Flags carry
epochs rather than 0/1, so replays need no host reset, and two alternating slots suffice for any
sequence of captured exchanges (the "slot reuse" argument in the kernel header).

The `ProgramImpl` constructor ([`program_impl.cpp`](../../src/models/qwen3_5/program/program_impl.cpp))
sizes one slot for the widest single-request exchange, `hidden × (K+1)` BF16: 10 KiB for ordinary
decode, 40 KiB for MTP3 at hidden 5120. Every all-reduce of a single-request captured round fits;
batched rounds whose payload exceeds the slot keep the staged path inside the same graph, and eager
calls (every prefill and warmup) always do.

A poller gives up after `kPeerSpinLimit` probes, about 0.8 s (the header records 4M probes measured
at 3.25 s on two RTX 5070 Ti; `run_mailbox_hang_case` in
[`test_allreduce.cpp`](../../tests/ops/test_allreduce.cpp) keeps it under a 2 s display watchdog),
sets the sticky hang word and skips its combine. `ProgramImpl::synchronize_devices`, called after
every round, checks the word after syncing both devices and throws `std::runtime_error`, because
the ranks' results have diverged. After startup that is an Engine-wide failure: every later round
fails, and `ninfer-serve` exits with status 2 so a supervisor restarts it
([`apps/serve/main.cpp`](../../apps/serve/main.cpp)).

### 4.3 Why both exist

Without P2P, a staged all-reduce is two host-staged copies and an event chain, 128 times per token;
the mailbox replaces that with one kernel per device. It cannot replace the copies everywhere: its
exchange has no host reset point and no event ordering, so both halves must be nodes of one graph
that every launch runs on both devices (an eager call would spin without a partner). Wider payloads
and the one-sided pulls (logits, acceptance results, draft tokens, Vision embeddings) are copies by
nature. With direct P2P the staged path is a direct copy, so the Engine keeps it there.

### 4.4 Startup probe

Right after creating the mailbox, before any decode graph is captured,
`ProgramImpl::probe_peer_mailbox` captures a two-device graph holding one 4 KiB `allreduce_sum` on
scratch buffers, checks that the capture selected the mailbox (`node_count() == 2`), launches it
once and times the round trip. If the hang word is set, the mailbox was not selected, or the round
trip exceeds 50 ms (`kMaxRoundTripMs`), the probe detaches and destroys the mailbox, records the
reason, and the transport becomes `copies`. `NINFER_TP_MAILBOX_PROBE=off` skips the probe; `=fail`
makes only rank 0 enqueue its half, so the poller must give up, which exercises the real fallback.
The probe proves one exchange shape only; a mailbox that passes it and hangs later is handled by
§4.5 during startup and by the Engine-wide failure above after it.

### 4.5 Step-down in `prepare_graphs`

`ProgramImpl::prepare_graphs` ([`graphs.cpp`](../../src/models/qwen3_5/program/graphs.cpp)) runs
warmup, capture and instantiation (`capture_and_instantiate`) in a retry loop. The first launch of
each topology class is followed by `synchronize_devices()`. If that throws and the hang word is set,
`degrade_peer_mailbox` discards every graph family first, since the executables bake in the hung
mailbox's addresses, and then takes one step down. The loop then captures everything again:

| Step | Transport string (`LoadSummary::tp_transport`) | Entered when |
|---|---|---|
| 1 | `mailbox` | the probe passed (or was skipped) |
| 2 | `mailbox, MTP draft on copies` | `--spec mtp` only: a first launch hung at step 1, or `NINFER_TP_MAILBOX_DRAFT=copies` at startup |
| 3 | `copies` | a first launch hung at step 2, or at step 1 without MTP; also `--no-tp-mailbox`, direct P2P, `--no-cuda-graph`, or a failed probe |

Step 2 creates a **new** mailbox and sets `TpExecution::staged_draft_collectives`. The MTP round then
opens a `PeerEvents::StagedScope` around its draft phase, so the verification's all-reduces keep the
mailbox while the MTP head's are captured on the copies. `NINFER_TP_MAILBOX_FAULT=draft` sets the
hang word at `prepare_graphs`' synchronizations while the MTP draft phase is on the mailbox, and
`=any` whenever a mailbox is attached. Both are test aids for hardware where no exchange hangs.

### 4.6 Reporting and the standalone probe

`Program::tp_transport()` returns a `TpTransportStatus`, which `construct_model_on` copies into
`LoadSummary::tp_transport`, `tp_mailbox_probe_ms` and `tp_mailbox_fallback` (reasons joined by
`"; then "`; [`types.h`](../../include/ninfer/types.h)). `ninfer-serve` logs a `tensor parallel`
line with `p2p on` or `p2p off (host-staged copies)`, a `captured all-reduces: … | mailbox probe X ms`
line, and a warning whenever the mailbox was dropped or narrowed
([`operational_log.cpp`](../../src/serve/operational_log.cpp)); the timeout error names the
transport and the probe time. [`tools/tp2/mailbox_probe.cu`](../../tools/tp2/mailbox_probe.cu) runs
the same check without the engine or a model (nvcc and the CUDA runtime only, the production
exchange kernel inlined, the spin limit as a parameter) and prints driver, devices, P2P, the startup
exchange, both transports' per-exchange cost and a verdict: exit 0 mailbox usable, 2 copies, 1 CUDA
error or wrong sums, 77 fewer than two devices ([Tools](../../tools/README.md#standalone-tp2-mailbox-probe)).

### 4.7 Measured costs

Only figures already recorded, with their scope:

| Measurement | Scope | Source |
|---|---|---|
| mailbox 8.7 µs vs copies 17.2 µs per 10 KiB exchange in a graph; startup exchange 0.03 ms; missing peer reported after ~0.8 s | 2× RTX 5070 Ti, no P2P, driver 595.91, CUDA 13.1, standalone probe | [Tools](../../tools/README.md#standalone-tp2-mailbox-probe) |
| mailbox ~41 µs vs staged ~277 µs per 10 KiB reduction, graph replay | 2× RTX 5060 Ti, Windows 11 WDDM, no P2P | `peer_mailbox.h` header |
| copies cost ~220 µs per hop instead of ~17; decode without `--spec` 60 tok/s on the mailbox vs 22.5 on copies; MTP3 49 tok/s on copies everywhere | WSL2, 2× RTX 5070 Ti, reported in issue #1 | [README](../../README.md) |
| MTP3 decode 102.3 tok/s at step 1, 101.1 at step 2, 97.1 with `--no-tp-mailbox`; plain decode 69.1 vs 59.8 on copies; identical output text on every transport | 2× RTX 5070 Ti, native Linux, QUASAR-QAT artifact, 400-token greedy generation, 3 runs, 2026-09-26 | development measurements, not otherwise published |

Whole-model throughput at tp 2, with its own scope (clocks, weights, benchmark suite), is in
[Two-GPU performance](../performance/two-gpu.md).

## 5. CUDA Graphs across two devices

A tensor-parallel decode program is **one** graph holding both devices' nodes: two live captures
cannot be linked (a `cudaStreamWaitEvent` across them fails with `cudaErrorStreamCaptureMerge`).
`capture_graph` ([`graph_execution.h`](../../src/models/qwen3_5/program/graph_execution.h)) forks
rank 1's stream into rank 0's capture and joins it back before the capture ends, through
`DecodeGraphPeerBridge` ([`decode_graph.h`](../../src/core/decode_graph.h)). The constraints
qualified on CUDA 13.1 head [`decode_graph.cpp`](../../src/core/decode_graph.cpp): memcpy nodes name
memory by UVA pointer, `cudaGraphExecUpdate` requires the same topology including node device
residency, instantiation keeps flags 0 (device-launchable graphs must be single-device), and event
records and waits become edges, not nodes.

Before any capture at tp 2 every batch size is warmed eagerly on both devices, since a module first
touched inside a capture cannot be loaded there and batch shape selects kernels; tp 1 warms batch 1
only, as upstream. `run_prepared` launches on rank 0's stream after
`DecodeGraphPeerBridge::gate_launch`, which orders the launch after work already issued on rank 1's
own stream (the round's mirrored page and table updates); the graph's own edges order rank 1's nodes
only after the graph root. Right after the launch, `gate_peer_after_launch` orders rank 1's stream
after the whole graph, so rank 1 work issued before the round's `synchronize_devices()` (none today)
could not overtake the graph's rank 1 nodes either.

**Topology classes and profile swaps.** As on one device, a graph family keeps one executable per
topology class and installs a class's other profiles with `DecodeGraphExecutable::update`
(`cudaGraphExecUpdate`). At tp 2 the captured cross-device nodes are the 1D memcpy nodes of the
staged pulls, whose parameters update freely, and the mailbox kernel nodes. The one node that could
not be updated was the 2D memcpy of the old logits interleave. Its column count and buffers differ
between profiles of one class, and the update failed with `ParametersChanged` only when a swap
crossed such profiles, hence late and intermittently. The fork first absorbed the rejection with a
re-instantiate fallback (`440481b2`, `f3846fcb`, narrowed to tp 2 in `f695d347`). Once
`concat_rows` removed the cause, `fa8c9e39` dropped the fallback again. A rejected update now throws
at both widths, as upstream, and `update()` keeps a diagnostic that names the update result and the
rejected node.

**Graph allowance.** At tp 1 the planner budgets graph classes from upstream constants. At tp 2 it
uses measured constants in `startup.cpp`: `kTp2OrdinaryGraphAllowance` (8 MiB per batch size),
`kTp2MtpGraphClassAllowance` (8 MiB per topology class and batch size) and
`kTp2DFlash2GraphClassAllowance` (11 MiB per class). Each is max(3 × observed, 8 MiB) per device,
observed being the free memory `prepare_graphs()` consumed per rank on two RTX 5070 Ti at 32K
context, concurrency 1 and INT8 KV: 2.0/2.0 MiB for ordinary and for MTP3 (one class each), and
18.0/12.0 MiB on rank 0/1 for DFlash2 K=4 over five classes. Concurrency above 1 was not measured;
the allowances scale linearly with it. The per-rank observation
(`MemorySummary::cuda_graph_observed_bytes`) is logged against the allowance, and an overrun is
warned about, not enforced.

**The WSL2 case (issue #1).** Under WSL2 on two RTX 5070 Ti the startup probe passes and plain
captured decode runs on the mailbox, but a full `--spec mtp` round hangs in its first launch during
`prepare_graphs` (reported even with K=1). The code responds with the step-down of §4.5, putting the
MTP draft phase on the copies first. The reason for that order is a **hypothesis**, not reproducible
on native Linux, although the comments on `PeerEvents::StagedScope` and in `graphs.cpp` state it as
fact: under WSL2 a host-staged cross-device copy stalls while an exchange kernel spins on its source
device, and the two ranks then wait on each other until the spin limit. The round's structure fits
it. In plain decode the only cross-device copy is the logits pull, after the last exchange. In an
MTP round, verification's 128 exchanges all precede the first copy (the pull of the accepted
counts), while the draft phase interleaves copies with the MTP head's all-reduces: rank 1 pulls
anchors, frontiers and licensed counts after `inputs_ready(0)`, and rank 0 pulls each full-head
proposal's logits. Step 2 captures exactly those all-reduces on copies, at about 1% cost on native
Linux (§4.7).

## 6. State and KV at tp 2

**Mirrors.** Before any mutation, `ProgramImpl::attach_tensor_parallel_mirrors` attaches rank 1's
Text KV page pool and execution tables, its MTP ones and its `StateImageDevicePool` as mirrors of
rank 0's. Every physical mutation rank 0 issues (page zero or copy, execution-row acquire, release
and publication, StateImage slot zero or copy) is replayed at the same index on rank 1's stream
([`paged_kv_cache.cpp`](../../src/core/paged_kv_cache.cpp),
[`state_image.h`](../../src/models/qwen3_5/state/state_image.h)); Host transfers are rejected once a
mirror is attached. Both ranks' persistent and workspace arenas start zero-filled at tp 2
(`tensor_parallel_zero_fill`). Rank 1's persistent layout (`SequencePlanImpl::peer_persistent`) is
rank 0's without the masked drafter's state, with KV heads, GDN channels and value heads halved by
`shard_text_config`.

**Bound KV rows.** Prefill and forced-token continuation read the sequence's execution row from
Program-wide scalars (`io.text_kv_table_row`, `io.backend_kv_table_row`, and rank 1's
`TpExecution::text_kv_table_row`). `ProgramImpl::publish_kv_rows`
([`context.cpp`](../../src/models/qwen3_5/program/storage/context.cpp)) writes both ranks'
scalars, checking that rank 1's mirrored row index equals rank 0's, at every `bind_sequence_kv`,
every staged prefill step and every forced-token prefill. Writing only at bind let another lane's
bind repoint the scalar, so forced tokens prefilled into the wrong lane's pages; the republication
(`c0f88d56`) also applies at tp 1 (§9). Decode rounds read their rows from the uploaded ingress.

**Rank 1's execution-row leases.** Rank 1 never calls `acquire`. `KVExecutionTablePool::acquire`
takes the mirror's lease on the same row first (so a failure changes neither pool) and keeps it in
`mirror_rows_`; `release_row` drops it with rank 0's lease. Every path that ends rank 0's lease
(unbind, deactivation, strict release) therefore ends rank 1's, and there is no separate release
path. Rank 1 reads its row through `mirror_row()`, as `prefill_tp_binding` does for the MTP row. The
MTP Engine test's "repeated requests" leg catches a leaked rank 1 row at a later bind.

**State slots and context-cache defaults.** Rank 1 has no Host copy of KV or state, so both Host
tiers are 0 and every checkpoint must fit a Device StateImage. `normalize_engine_options` raises the
tp 2 defaults to `device_state_slots = max(2C, 8)` extra slots (C = `--max-concurrency`; tp 1 uses
`C`) and a private catalog of `max(2C, 8)` (tp 1: `2C`). Each slot costs one StateImage per rank,
about 73 MiB for Qwen3.8-27B. `--device-state-slots` overrides the default.

**Prefix reuse.** Without a Host tier, a private in-prefill capture (TurnClosure rewrite, long
anchor) that finds the Device pool full cannot snapshot to Host. `EngineCore` therefore enables
`ResourceManager::reclaim_private_owners_for_capture` only when `options.tp > 1`
([`engine_core.h`](../../src/runtime/engine/engine_core.h)): it releases idle private continuations
without a live session or active edge, lowest retention weight and oldest first, until the capture
fits or no candidate is left, and does not restore them if the capture is still skipped
([Resource scheduling](resource-scheduling-and-context-cache.md#101-retention-policy)). A retained
prefix at tp 2 carries:

- KV pages and StateImages on both ranks, through the mirrors;
- under MTP, rank 1's own copy of every target hidden the Program retains (`peer_retains_hidden`):
  prompt and forced-token tails (`copy_tail`), capture frontiers inside a chunk and accepted
  verification columns, each in the same StateImage slot of rank 1's pool, so Forks, Moves and
  copies carry it with rank 0's; the MTP bridge of a resumed prefix runs the split head from both;
- for a zero-suffix reuse, nothing on rank 1: the first token comes through the vocabulary-split
  head from rank 0's retained hidden, which rank 1 pulls once (`project_split_output_head` in
  [`prefill.cpp`](../../src/models/qwen3_5/program/prefill.cpp)), for every backend;
- under DFlash2, the drafter's rings in rank 0's StateImages only.

The current code declines no reuse plan at tp 2. `815615e6` briefly made admission decline
zero-suffix and speculative reuse at tp 2 and turned unmaterializable checkpoints into misses; once
MTP and DFlash2 resume existed (`c29ccc46` and following), `4534b7bf` restored the upstream
`logic_error` in `ProgramImpl::inspect_lane`
([`request_plan.cpp`](../../src/models/qwen3_5/program/planning/request_plan.cpp)). Only reuse that
needs a Host replica is unavailable.

**`--kv-capacity auto`.** Both ranks address the same pages, so they must agree on one page count.
`construct_model_on` takes each rank's free memory (`rank_free_bytes`), credits a rank with the
reservation part it does not allocate (`unallocated_reservation_bytes`: the Vision encode workspace
of the rank without the tower), and `runtime::resolve_kv_capacity_symmetric`
([`kv_capacity.cpp`](../../src/runtime/engine/kv_capacity.cpp)) resolves against the minimum. The
automatic headroom is the fixed `kDefaultKvCapacityHeadroomBytes` (1 GiB), subtracted once from
that bottleneck budget as at tp 1; an explicit capacity carries none. Under DFlash2 rank 1 is still
budgeted for rank 0's drafter state; rank 0, which also holds the drafter weights, is normally the
bottleneck anyway.

## 7. Speculative decoding at tp 2

**MTP.** The MTP head is split like a Text layer. Its `fc` input projection is split by input
columns, so rank 0 contracts the normalized token embedding, rank 1 the normalized target hidden,
and only rank 0 embeds tokens; each MTP forward contains three all-reduces. `--draft-tokens` keeps
its range 1..5, and K sets the mailbox slot size and the MTP graph profiles. With `--lm-head-draft`
the optimized proposal head is PrimaryOnly and proposes on rank 0 alone; otherwise the full head is
the vocabulary-split `text/output_head`, gathered on rank 0 before the argmax. One captured round
(`mtp_decode_batch_body` in [`speculative/mtp.cpp`](../../src/models/qwen3_5/program/speculative/mtp.cpp)):

1. Upload the ingress record to both ranks' frames; each prepares its verify ids and positions.
2. Verify K+1 columns on both ranks (`target_verify_accept`,
   [`target_verification.cpp`](../../src/models/qwen3_5/program/speculative/target_verification.cpp)),
   each recording ReplaySSM inputs for its own GDN heads; rank 0 gathers the logits and accepts
   (greedy or sparse).
3. Rank 0 records `inputs_ready(0)`; rank 1 waits, pulls the accepted counts, and selects and
   scatters its own accepted hidden into its continuation store.
4. **Draft phase**, where the cross-device dependency sits in the middle of the graph and where
   step 2 of §4.5 opens its `StagedScope`: rank 0 prepares the next round; rank 1 waits on
   `inputs_ready(0)`, pulls anchors, frontiers and licensed counts, and prepares its copy.
5. Both ranks run the split MTP head over the alignment columns and select the accepted hidden;
   each proposal step then gathers logits on rank 0 (full head) or projects on rank 0 alone.

The commit folds each rank's ReplaySSM records into its own GDN state with the same rows
(`replay_fold` in `prefill.cpp`). Forced tokens prefill through `prefill_text_chunk` with
`prefill_tp_binding`, so they run on both ranks and update both ranks' MTP KV and retained hidden.
The prompt MTP alignment runs on both ranks, rank 1 keeping its final-normed chunk in its own
`prefill_hidden`, and rank 1 keeps its own RoPE delta (`TpExecution::rope_delta`).

**DFlash2.** Only the target is split; the drafter, its context features and the optimized proposal
head stay whole on rank 0. The feature tap reads rank 0's residual after each captured layer, which
the all-reduce has already completed, so features need no exchange. In a round
(`dflash_decode_batch_body` in [`draft.cpp`](../../src/models/qwen3_5/execution/draft.cpp)) rank 0
appends the context and proposes, rank 1 pulls the draft tokens after `inputs_ready(0)`, both
verify, rank 0 accepts and rank 1 pulls the accepted counts. `--lm-head-draft` is mandatory at
tp 2: candidate ranking (`linear_topk`) is registered only for one complete head, the full output
head is split by vocabulary, and only the optimized head, whole on rank 0, can serve the rank-0
drafter. The Engine, `plan_load` and the planner each reject the other case. A drafter with
full-attention layers is also rejected, since its paged KV would have no rank 1 mirror while
`publish_kv_rows` requires one; the published drafter has none. The full contract is in
[DFlash and DFlash2](dflash.md#tensor-parallel-execution).

## 8. Vision at tp 2

The tower is not split. `vision/*` is placed SingleDevice on `LoadOptions::vision_rank`, the index
of `--vision-device` in `--devices` (default rank 0; [`load_options.h`](../../src/models/load_options.h)),
and `--vision-device` must be one of `--devices`. The encoding rank's workspace follows the
single-device Vision plan; the other rank's (`WorkspacePlan::vision_receiver`) is its general prefix
plus a handoff of the same extent, with no encode region. `VisionPrefillSession`
([`vision.cpp`](../../src/models/qwen3_5/execution/vision.cpp)) encodes each item once on the
encoding rank; the other rank's stream waits on an event, copies the `[H, V]` merged embeddings into
its own handoff (an eager, host-staged copy without P2P), and records a second event before the
encoder may reuse its handoff. Each rank scatters its copy into its own residual.

Memory cost: only the tower's rank holds the Vision weights and the encode workspace; the receiver
holds a handoff of `hidden × 2` bytes per token of the largest item (10 KiB per token at hidden
5120). `--max-vision-tokens` shrinks both. `--kv-capacity auto` credits the receiver with the encode
workspace it does not allocate (§6), so the tower's rank pays for it alone, which is why the user
docs recommend the GPU with more free memory as `--vision-device`.

## 9. tp 1 identity

`--tp 1` keeps upstream's single-device path because nothing two-device is constructed there.
`construct_model_on` passes a null `ExecutionContext`, so loading, `materialize`, the sequence planner
and `create_program` take their single-device overloads; the Binder has no shard resolver, so every
parent is Replicated and the layout is byte-identical to upstream's. `TextContext` without a
`TpExecution` runs the single-device schedule, and no `PeerEvents`, `PeerMailbox` or
`DecodeGraphPeerBridge` exists. Arenas are not zero-filled, graph allowances are upstream's
constants, the private-capture reclaim is off, only batch 1 is warmed before capture, and a
rejected `cudaGraphExecUpdate` throws (§5).

Some deliberate differences remain at tp 1: the per-prefill-step KV row publication (`c0f88d56`, a
fix for a multi-lane forced-token bug that upstream `bace20dc` also has); `FuncAttrPerDevice`
([`kernel_attr_once.h`](../../src/ops/launcher/kernel_attr_once.h)), which costs one `cudaGetDevice`
per eager launch of an attribute-carrying kernel; a device synchronization and free-memory queries
around `prepare_graphs` for the graph observation; and new fields in the public `EngineOptions`,
`MemorySummary` and `LoadSummary`.

**Golden gate.** [`tools/golden/`](../../tools/golden/README.md) turns the intention into a
token-identity check. `tp1_golden.cpp` (target `ninfer-tp1-golden` in `apps/CMakeLists.txt`, public
Engine API only) builds unchanged against this fork and against upstream at the fork's base, and
decodes a deterministic byte-id prompt greedily, without chat template or prefix reuse. `record.sh`
runs three cases chosen for their prefill regimes (32, 2,048 and 6,144 prompt tokens over 1, 2 and 6
chunks, the last with INT8 KV); `diff -r` of the two output directories is the gate. No 27B artifact
fits one 16 GB board, so the model is the synthetic two-layer Qwen3.5 of
[`test_text_context_tp2.cpp`](../../tests/models/qwen3_5/test_text_context_tp2.cpp) (one GDN and one
full-attention block, FP8 projections, BF16 norms, a byte-level tokenizer), written by the `write`
mode of the `ninfer_qwen3_5_text_context_tp2_test` executable that
[`tests.cmake`](../../tests/models/qwen3_5/tests.cmake) registers. The recorded runs found fork
`2e7f7d3a` and upstream `bace20dc` (`recorded/2026-09-25`), and fork `b3f93dd6`, the merge of
upstream `e31bc99b`, and upstream `e31bc99b` (`recorded/2026-09-27`), identical on all three cases.

The gate covers the Engine, scheduler, prefill and decode programs, attention, GDN, and the FP8 and
BF16 Ops on one device. It does not cover the NVFP4 and INT-quantized GEMMs, MTP, DFlash/DFlash2,
Vision, prefix reuse or concurrent requests; a board that holds a 27B artifact would let
`record.sh` cover the GEMMs as-is.

**Perplexity at tp 2.** `ninfer-perplexity --tp 2` ([Perplexity](../perplexity.md#two-gpus)) scores
through `ProgramImpl::causal_score`: its chunks run the split prefill, and each score tile of up to
1,024 columns goes through the vocabulary-split head. It is the numerical check a 27B artifact has
at tp 2: two builds compared on the same corpus at tp 2, and tp 1 against tp 2 on the synthetic
model above, whose FP8 head halves (`n124160_k5120`) it drives at up to 1,024 columns.

## 10. Tests

Every two-device test uses devices 0 and 1 and returns 77 (`SKIP_RETURN_CODE 77`, reported as
skipped) below two devices; the `*_real` tests also need `NINFER_TEST_ARTIFACT`. They are registered
in [`tests/ops/tests.cmake`](../../tests/ops/tests.cmake), `tests/artifact/tests.cmake`,
[`tests/models/qwen3_5/tests.cmake`](../../tests/models/qwen3_5/tests.cmake) and `tests/cmake/*.cmake`;
[Tests](../../tests/README.md) has the run commands.

| Test | Checks |
|---|---|
| `ninfer_{allreduce,linear_split,output_head_split,attention_headlocal,attn_input_proj_split,gdn_projections_split,gdn_headsplit,linear_swiglu_split,linear_add_split}_test` | each split form at its shard shapes against the single-device Op; the all-reduce and row gather against FP64 and exact oracles; the mailbox inside a replayed graph, its selection by node count, and the hang report within the watchdog bound |
| `ninfer_artifact_sharded_materialization_tp2_test` (+ one-device plan tests, `ninfer_qwen3_5_shard_map_test`) | Replicated, Rows, Columns, PrimaryOnly and SingleDevice parents on both devices, byte-compared with host-applied slices |
| `ninfer_decode_graph_test`, `ninfer_kv_cache_test` | two-device capture and the update diagnostic; mirror replay of page and row mutations (on one device) |
| `ninfer_qwen3_5_sharded_load_real_test` (+ `_mtp_real`) | per-device placement and bytes of the loaded artifact |
| `ninfer_qwen3_5_text_context_tp2_test` | synthetic model: tp 2 prefill and decode logits within two BF16 ulps of tp 1; the rank-0 logits gather byte for byte |
| `ninfer_qwen3_5_text_context_tp2_real_test` | real artifact at tp 2: finite logits, "What is 17*23?" answers 391 |
| `ninfer_qwen3_5_engine_tp2_real_test` | Engine with graphs, context cache and two lanes: single and concurrent answers (rank 1 must prefill through each lane's own row), prefix reuse, ≥ 80% reuse of a ~6k-token prefix with 4 and 1 lanes, reuse after a one-shot flood |
| `ninfer_qwen3_5_engine_mtp_tp2_real_test` (+ `_optimized_real`) | MTP K=3 against the same model without speculation (answers right, common prefixes, half identical), acceptance floor, two-lane rounds, repeated binds (leaked rank 1 row), ~9k-token prefix resume through the bridge, zero-suffix repeat |
| `ninfer_qwen3_5_engine_dflash2_tp2_real_test` | DFlash2 K=4 `--lm-head-draft`: token ids identical to plain tp 2 on three prompts, ≥ 30% acceptance, two-lane rounds, prefix reuse |
| `ninfer_qwen3_5_engine_vision_tp2_real_test` (+ `_mtp_real`, `_dflash2_real`) | `vision_device` outside `devices` rejected, text identical with and without Vision, colors of synthetic images with the tower on device 0, same token ids with it on device 1, second turn resumed |
| `ninfer_serve_engine_failure_real_test` | a thrown decode-round failure (injected by wrapping `cudaStreamSynchronize`) stops the server with `engine_failed()` set, so it exits 2 |

A real artifact does not fit one board, so the real tests are semantic and do not compare with
tp 1. Beyond the per-Op suites, the numerical tp 2 versus tp 1 comparisons are the synthetic
`text_context_tp2` parity (FP8 projections, ordinary prefill and decode) and, by hand, the
perplexity of the same synthetic model at both widths (§9). The golden gate is a tool
run by hand, not a CTest.

## 11. Known limits and not-done items

- **Two ranks, one verified pair.** Verified only on two RTX 5070 Ti 16 GB without P2P, Linux,
  CUDA 13.1. P2P-capable pairs (which keep the copies as captured transport) and other boards are
  untested.
- **Memory report.** `ProgramImpl::memory_summary` describes rank 0's device (weights, persistent,
  workspace), and `available_after_startup_bytes` is the bottleneck rank's free memory; rank 1 shows
  only in `LoadSummary::devices` and the per-rank graph observation.
- **Default state slots.** `max(2C, 8)` does not adapt to the memory left; at long contexts with
  Vision it must be lowered by hand (the README reports that only 4 fit at 196,608 tokens with Vision
  and the official NVFP4 artifact).
- **`--kv-capacity auto` headroom.** Fixed at 1 GiB on the bottleneck rank. The tp 2 graph
  allowances come from one measurement (32K, C=1, INT8 KV); an overrun is absorbed by that headroom,
  and an explicit capacity has none.
- **Unused rank 1 memory.** Rank 1's round state keeps the full-vocabulary logits frames of rank 0's
  layout, which it never writes (`persistent_layout` in `startup.cpp`); under DFlash2 it is also
  budgeted for rank 0's drafter state.
- **Transport after startup.** A mailbox hang after `prepare_graphs` stops the Engine; there is no
  runtime step-down. Kernel-serializing profilers can trigger it (`--no-tp-mailbox` avoids it). The
  probe exercises one 4 KiB exchange. DFlash2 steps straight to copies and has not been run under
  WSL2.
- **Numerics.** Split projections sum their halves in a different order and round each partial to
  BF16 before the all-reduce, so long greedy generations can drift from a single-GPU run.
- **Host-less reuse.** Many parallel conversations evict the oldest idle ones. The reclaim does not
  check beforehand whether its releases can make the capture fit, so it can release continuations
  for a capture that is still skipped.
- **Duplicated code.** `TextContext::prefill_impl_tp2` duplicates `prefill_impl`; a change to one
  must be carried to the other.
- **Not ported.** YaRN long-context RoPE from the pre-v3 fork line is not in this tree; RoPE has no
  per-rank override.

## 12. File map

| Path | Role in the two-GPU mode |
|---|---|
| `include/ninfer/types.h` | `EngineOptions::{tp,devices,vision_device,tp_mailbox}`, `LoadSummary::{devices,peer_access,tp_transport,…}`, graph observation |
| `src/product/tensor_parallel_options.h`, `apps/cli/options.cpp`, `apps/perplexity/options.cpp`, `src/serve/serve_options.cpp` | `--tp`, `--devices`, `--no-tp-mailbox`, Host-tier defaults at tp 2 |
| `src/runtime/engine/engine.cpp`, `model_instance.cpp`, `kv_capacity.*` | `ExecutionContext` and peer access, tp 2 checks and defaults, per-rank budgets, symmetric KV resolution, `LoadSummary` |
| `src/runtime/engine/engine_core.h`, `context_cache/resource_manager.h` | Host-less private-capture reclaim |
| `src/core/device.h`, `decode_graph.*`, `paged_kv_cache.*` | `ExecutionContext`; two-device capture bridge, launch gate, update diagnostic; KV mirrors and mirror leases |
| `src/artifact/slices.*`, `binder.*`, `materializer.*`, `views.cpp` | shard geometry and plane copies, per-device placement, upload and views |
| `src/models/load_options.h`, `src/models/qwen3_5/load/sharding.*`, `load.cpp` | `LoadOptions::{tp,vision_rank}`, placement rules, tp load checks |
| `src/models/qwen3_5/execution/tp.*` | `TpExecution`, `shard_text_config`, rank-0 logits gather |
| `src/models/qwen3_5/execution/{text,attention,gdn,ffn,mtp,draft,vision}.*`, `linear.h` | split Text, MTP, verification, DFlash2 and Vision schedules |
| `src/models/qwen3_5/state/state_image.h` | StateImage mirror |
| `src/models/qwen3_5/program/program_impl.*`, `graphs.cpp`, `graph_execution.h` | `PeerRuntime`, mirrors, mailbox, probe, step-down, capture and launch; causal scoring's split head |
| `src/models/qwen3_5/program/planning/startup.*` | per-rank layouts, tp 2 graph allowances, tp 2 rejections |
| `src/models/qwen3_5/program/{storage/context,prefill,decode,transactions/commit}.cpp`, `speculative/*` | row publication, rank 1 retained hidden, ingress upload, forced tokens, MTP round |
| `include/ninfer/ops/allreduce.h`, `peer_mailbox.h`, `src/ops/common/{allreduce,peer_mailbox}.cu`, `src/ops/kernel/peer_exchange.cuh` | staged collectives, `PeerEvents`, mailbox and exchange kernel |
| `include/ninfer/ops/{linear,linear_add,linear_swiglu,attn_input_proj,gdn_input_proj,gdn_gating_proj}.h`, `src/ops/common/split_launch.h`, `src/ops/launcher/concat_rows.*` | split forms, registered shard shapes, pair validation, logits interleave |
| `src/serve/operational_log.cpp` | tensor-parallel startup lines and warnings |
| `tools/tp2/mailbox_probe.cu`, `tools/golden/` | standalone mailbox check; tp 1 token-identity gate |

## Open questions

1. ~~Stale descriptions of changed code.~~ Aligned on 2026-09-27: the comment on
   `output_logits_split_rank0` in `execution/tp.h` and the [Qwen3.5 model](qwen3_5-model.md#tensor-parallel-execution)
   note describe the `concat_rows` kernel, the `TpTransportStatus` comment in `program/program.h` lists the
   intermediate transport, and the serve `tensor parallel` line describes the reclaim before a skip.
2. **The WSL2 mechanism is a hypothesis.** The comments on `PeerEvents::StagedScope` and in
   `graphs.cpp` state that a staged copy stalls behind a spinning exchange kernel; the development
   notes call this a hypothesis not reproducible on native Linux. When they were written it was
   also unconfirmed whether step 2 fixes the reported hang or the engine ends on step 3.
3. **DFlash2 under that hypothesis.** The notes say the DFlash2 round has no exchange after a
   cross-device copy. The code (`dflash_decode_batch_body`) pulls the draft tokens to rank 1 before
   verification's exchanges, the pattern the hypothesis blames. DFlash2 has no intermediate step and
   has not been run under WSL2.
4. **Placement and test lists in the porting notes.** A porting note counts the GDN convolution
   among the replicated weights; the current rules split it by Q/K/V channel sections and split
   `gdn/{a_projection,b_projection,a_log,dt_bias}` by value heads, and only norms and
   `text/token_embedding` are replicated. The same notes list a `test_engine_yarn_real`: no YaRN code
   or test exists in this tree (both belong to the pre-v3 line, `cfc8f749` and `b40d17ac`, which are
   not ancestors of `main`). This document follows the code.
5. **Notes that describe earlier code.** Declined reuse plans at tp 2, the `cudaGraphExecUpdate`
   re-instantiate fallback and explicit peer-lease releases in `release_*_strict` appear in the
   development notes. The first two were removed (`4534b7bf`, `fa8c9e39`); the last describes the pre-v3
   fork, and the current lease rides inside `KVExecutionTablePool::mirror_rows_`.
6. **Golden gate re-recorded after the upstream merge.** `839fdea0` and `a0369376` changed
   `ProgramImpl` construction and turned `prepare_graphs`, which tp 1 also runs, into the retry
   loop after the first record (`2e7f7d3a`); the second record (`b3f93dd6`, after merging upstream
   `e31bc99b`) covers them. The gate still does not cover NVFP4, the format upstream rewrote most.
7. ~~One-way launch gate.~~ Audited on 2026-09-27: at each of the three gated launch sites
   (ordinary, MTP, DFlash2 rounds in `decode.cpp`) the host issues nothing on rank 1's stream
   between the launch and the round's `synchronize_devices()`, on the success and the failure path;
   the startup launches (`probe_peer_mailbox`, `instantiate_graph_family`) are bracketed by
   synchronizations. `gate_peer_after_launch` now also orders rank 1's stream after each launch,
   so the property no longer rests on that call-site discipline.
8. **Unpublished figures.** The step-by-step MTP3 and plain-decode figures in §4.7 come from
   development measurements published nowhere else in the repository; they are quoted with their
   scope but cannot be re-derived from repository files.
