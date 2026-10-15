#pragma once

// Two-device (tp 2) form of context_kv_materialize. Our file: ninfer/ops/context_kv_materialize.h
// is upstream's and carries no hook.

#include "ninfer/ops/context_kv_materialize.h"

namespace ninfer::ops {

// KV heads of one tensor-parallel rank's block.
inline constexpr std::int32_t kContextKVMaterializeBlockKVHeads = 4;

/**
 * context_kv_materialize over one tensor-parallel rank's contiguous block of four KV heads: each
 * layer's key/value weights are the rank's RowSplit Q8_G32_FP16 [512,5120] blocks of the
 * [1024,5120] parents, and each cache is a capacity-2048 D128/H4 cyclic cache. Every other
 * contract is context_kv_materialize's. Rank r's rings hold exactly the complete Op's heads
 * [4 r, 4 r + 4) (upstream's kernels, compiled for four heads).
 */
void context_kv_materialize_head_block(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t context_kv_materialize_head_block_workspace_capacity_bytes(
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

} // namespace ninfer::ops
