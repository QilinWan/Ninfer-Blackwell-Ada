# Token embedding in host memory (`--embedding-host`)

`--embedding-host` (`EngineOptions::embedding_host`) keeps the `text/token_embedding` table in one
copy in mapped page-locked host memory, which the gather kernels of every rank read in place over
PCIe, instead of one copy in each rank's device memory. The bytes are the artifact's own, so every
result is identical to the default placement; only the address the gather reads changes.

Qwen3.8-27B stores the table as `fp8_e4m3fn_row_bf16`: 248,320 rows of 5,120 FP8 codes plus one
BF16 scale per row, 1,271,895,040 B (1.18 GiB). At `--tp 2` the default placement replicates it,
so the option frees 1.18 GiB on **each** GPU (2.37 GiB in total) for the KV pool and costs 1.18 GiB
of pinned host RAM once. At `--tp 1` it frees 1.18 GiB on the one GPU.

## Placement

The table gets its own placement kind, `artifact::ShardAxis::HostMapped`
([`slices.h`](../../src/artifact/slices.h)). `loading::shard_rule`
([`sharding.cpp`](../../src/models/qwen3_5/load/sharding.cpp)) returns it for
`text/token_embedding` when `LoadOptions::embedding_host` is set, at tp 1 as well as tp 2, and
`install_shard_resolver` installs the resolver at tp 1 for that case only. Every other parameter
keeps its placement. MTP and DFlash/DFlash2 bind the embedding by WeightId
(`load/mtp.cpp`, `load/dflash.cpp`), so the MTP stem, the target verification, the ordinary decode,
prefill and the split DFlash2 drafter all read the same host copy.

`artifact::Binder::place_device` ([`binder.cpp`](../../src/artifact/binder.cpp)) plans a
HostMapped parent as one `HostMappedPlacement` at the next aligned offset of the plan's host-mapped
allocation (`MaterializationPlan::host_mapped_bytes`); it adds nothing to any
`per_device_capacity_bytes`, so the weight arenas shrink and the KV sizing that follows sees the
freed memory.

## Materialization

`artifact::materialize` ([`materializer.cpp`](../../src/artifact/materializer.cpp)) allocates one
`PinnedHostBuffer(bytes, HostMapping::Yes)` ([`arena.h`](../../src/core/arena.h)), that is
`cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable)`, before any weight upload, reads each
HostMapped parent into it once (no staging slot, no H2D copy) and gives every device a
`WeightParent` whose `data` is that device's address of the bytes (`cudaHostGetDevicePointer` with
the device selected; with unified addressing all addresses coincide with the host pointer). The
`DeviceShard` of such a parent records `ShardAxis::HostMapped` with no ranges, so views treat it as a
complete parent. `MaterializedArtifact` owns the buffer: it lives as long as the Model, hence
longer than every Program and every captured graph.

## CUDA Graphs

The gather (`ops::embedding` → `embed_gather_fp8_kernel`) is unchanged and runs inside the
captured decode, MTP and DFlash2 graphs exactly as before: a kernel argument that points into mapped
host memory is as valid in a graph as one that points into device memory. The rules are those of
the tp 2 peer mailbox (`include/ninfer/ops/peer_mailbox.h`), which already reads mapped host memory
from captured kernels: allocate before the capture, keep the allocation alive as long as the graph
executables. Both hold by construction (allocation at load, ownership by the Model).

## Cost

A gather reads one 5 KiB row per token over PCIe instead of from device memory. Decode reads a few
rows per round (1 per rank in ordinary decode, 14 for an MTP3 round, about 32 for a DFlash2 K=7
round), so the PCIe latency adds microseconds to a round of milliseconds. Prefill reads 5 KiB per
prompt token on each rank, about 5 MiB per 1,024-token chunk. The tp 2 measurements on two
RTX 5070 Ti are in the fork's validation notes; the option is off by default.

## Reporting

`MaterializationStats::host_mapped_bytes` (the allocation) and `per_device_host_mapped_bytes` (the
parents each device reads) reach `LoadSummary::host_mapped_bytes` and
`LoadDeviceSummary::host_mapped_bytes`. The startup log adds ` | host-mapped 1.18 GiB` to each tp 2
rank line, whose `weights` and `replicated` then exclude the table, and one
`host-mapped weights | 1.18 GiB of pinned host memory | read in place by N devices over PCIe` line.
The request log records `engine.embedding_host` and `artifact.host_mapped_bytes` in `server_start`.

## Tests

- `tests/artifact/test_sharded_materialization.cpp` (`host_mapped_upload`): plan layout, no device
  capacity, statistics, one shared host address for both devices, page-locked residency, the bytes
  each device reads through its own parent, views; on one device and on two.
- `tests/models/qwen3_5/test_shard_map.cpp`: the HostMapped rule at tp 1 and 2, and the planned
  fixture model with the embedding out of every device arena.
- The tp 2 gate (`tools/tp2/gate.sh`) with `GATE_EXTRA_FLAGS=--embedding-host` checks golden (tp 1),
  perplexity, greedy and DFlash2 against the same reference as the default placement.
