#pragma once

// Two-device (tp 2) forms of attn_input_proj. Our file: ninfer/ops/attn_input_proj.h is upstream's and carries no hook.

#include "core/device.h"
#include "ninfer/ops/attn_input_proj.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Tensor-parallel form over two devices.
//
// The single-parent projection splits by heads. Rank r's weight is a standalone shard whose four
// sections are rank r's halves of the parent's query, key, gate and value sections, stacked in the
// parent's query, key, gate, value row order, so each rank computes its own heads' outputs with no
// communication. For the registered FP8 and NVFP4 parents `[14336,5120]` the shard is
// `[7168,5120]` with row counts `[3072,512,3072,512]`: 12 of the 24 query/gate heads and 2 of the 4
// key/value heads. The shard is never a view into the parent's payload.
//
// Every requirement of the single-parent attn_input_proj() applies per rank at the shard shapes.
// `x[r]`, `w[r]`, the four outputs and `workspace[r]` must be resident on `ec.dev[r]`, and rank r's
// work is enqueued on `ec.dev[r]->stream`. The caller obligation of ninfer/ops/allreduce.h applies:
// inputs staged on a device's legacy default stream must be retired before the call. The form does
// not synchronize, and the caller's current CUDA device is preserved.

/**
 * Returns the per-rank transient capacity attn_input_proj_column_parallel() requires for every T
 * in `[min_tokens,max_tokens]`. FP8_E4M3FN_ROW_BF16 and NVFP4 shards are registered; the shard
 * keeps the parent's input rows, so this equals the parent's capacity.
 */
[[nodiscard]] std::size_t attn_input_proj_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * @brief Column-parallel (head-split) single-parent projection across two devices.
 *
 * @details Rank r computes `q[r]`, `gate[r]`, `k[r]` and `v[r]` from `x[r]` and its shard
 * `query_key_gate_value_weight[r]`. Concatenating the ranks' outputs along `ne[0]` gives the
 * single-device outputs of the parent, and each rank's numerical contract is that of
 * attn_input_proj() under the same policy. Both ranks must agree on the weight format, on `K` and
 * on the token count `T`. FP8_E4M3FN_ROW_BF16 and NVFP4 `[7168,5120]` shards are registered.
 *
 * @param[in] x Per-rank BF16 activation `[5120,T]`, identical on both ranks.
 * @param[in] query_key_gate_value_weight Per-rank FP8 or NVFP4 shard `[7168,5120]`.
 * @param[out] q,gate Per-rank BF16 `[3072,T]`.
 * @param[out] k,v Per-rank BF16 `[512,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized by
 * attn_input_proj_column_parallel_workspace_capacity_bytes().
 * @param[in] ec Execution context holding two distinct devices.
 */
void attn_input_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_gate_value_weight,
    const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
    const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

/// A16-only column-parallel form; it requires no transient workspace.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_gate_value_weight,
                                     const std::array<Tensor, 2>& q,
                                     const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     const ExecutionContext& ec);

/**
 * The DFlash2 drafter's Q8 query|key|value projection over one tensor-parallel rank's heads: the
 * Q8_G32_FP16 RowSplit shard `[3072,5120]` stacks the rank's rows of the `[6144,5120]` parent's
 * query (2048 of 4096: 16 of 32 D128 heads), key and value (512 of 1024: 4 of 8 KV heads) sections.
 * `x` BF16 `[5120,T]`, `q` BF16 `[2048,T]`, `k` and `v` BF16 `[512,T]`, all contiguous; every
 * other contract is attn_input_proj()'s three-output DFlash2 profile. The launchers are upstream's
 * DFlash2 kernels with the shard's section output, so each output row equals the parent's row bit
 * for bit. No workspace.
 */
void attn_input_proj_head_block(const Tensor& x, const Weight& query_key_value_weight, Tensor& q,
                                Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops
