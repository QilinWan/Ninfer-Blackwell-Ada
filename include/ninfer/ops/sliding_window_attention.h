#pragma once

#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SlidingWindowAttentionExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Symmetric non-causal sliding-window grouped-query attention.
 *
 * The registered profiles are D=128, Hq=32, Hkv=8 (group 4), window/cyclic capacity 2048 or
 * 4096, scale=1/sqrt(128), T=1..16, and B=1..8. q/out are contiguous BF16 [128,32,T,B],
 * query_k/query_v are contiguous BF16 [128,8,T,B], positions is contiguous device I32 [T,B], and
 * valid_columns and lanes are contiguous device I32 [B]. Row b has V=valid_columns[b] live query
 * columns, positions[i,b]=L+i for i<V, and lanes[b] selects its cyclic-cache lane. Columns i>=V
 * are an inert physical tail and produce exact BF16 zero.
 *
 * The read-only cyclic BFloat16 context stores BF16 K and FP16 V for committed absolute positions
 * [max(0,L-window),L), with absolute position p stored at slot p mod window. Temporary query K/V is
 * a separate BF16 segment at [L,L+V). For live query position p_i, a populated context or
 * temporary key at p_j is visible
 * exactly when abs(p_j-p_i)<window. Thus distance window-1 is included and distance window is
 * excluded; every live query also sees every live temporary query row in its batch row. Query
 * head h reads KV head floor(h/4).
 *
 * The independent oracle evaluates the complete formula naively in FP64 from represented BF16
 * query/context K and BF16 query V / FP16 cached V, using a stable Softmax over exactly that
 * visible set. BF16 out is promoted for comparison; storage rounding belongs to the Op criterion.
 * q/query_k/query_v/positions/valid_columns/lanes, every cyclic-cache byte, out, and live workspace
 * allocations are pairwise non-overlapping. Inputs and context remain unchanged; out is the only
 * observable mutation and is completely overwritten.
 *
 * The caller guarantees envelope.min_context <= L <= envelope.max_context, sequential nonnegative
 * live positions, and that the selected lane contains the declared committed interval. The
 * envelope is a host launch/workspace resource promise; it never changes the admitted key set or
 * numerical result.
 */
void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& lanes, AttentionHeadGeometry geometry,
                              std::uint32_t window, float scale,
                              const CyclicKVCacheLayerView& context,
                              SlidingWindowAttentionExecutionEnvelope envelope,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Return transient capacity for every T in the inclusive optimized interval at the exact batch
 * size. Geometry, window, and execution envelope are fixed profile inputs; invalid inputs throw.
 */
[[nodiscard]] std::size_t sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size);

/**
 * @brief sliding_window_attention over one tensor-parallel rank's contiguous block of heads of
 * the registered D128 32/8 profile (tp 2, the split DFlash2 drafter): geometry {128,16,4} (the
 * same group of four query heads per KV head), q/out BF16 [128,16,T,B], query_k/query_v BF16
 * [128,4,T,B] and a D128/H4 cyclic context. Every other contract is sliding_window_attention's.
 * Each KV head's CTA computes exactly what the 32/8 profile computes for that head, so rank r's
 * output equals heads [16 r, 16 r + 16) of the complete profile's output bit for bit.
 */
void sliding_window_attention_head_block(const Tensor& q, const Tensor& query_k,
                                         const Tensor& query_v, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& lanes,
                                         AttentionHeadGeometry geometry, std::uint32_t window,
                                         float scale, const CyclicKVCacheLayerView& context,
                                         SlidingWindowAttentionExecutionEnvelope envelope,
                                         WorkspaceArena& workspace, Tensor& out,
                                         cudaStream_t stream);

/**
 * Return transient capacity for every T in the inclusive interval at the exact batch size for
 * sliding_window_attention_head_block. Geometry, window, and execution envelope are fixed profile
 * inputs; invalid inputs throw.
 */
[[nodiscard]] std::size_t sliding_window_attention_head_block_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size);

} // namespace ninfer::ops
