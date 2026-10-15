#pragma once

// Two-device (tp 2) form of sliding_window_attention. Our file: ninfer/ops/sliding_window_attention.h
// is upstream's and carries no hook.

#include "ninfer/ops/sliding_window_attention.h"

namespace ninfer::ops {

/**
 * sliding_window_attention over one tensor-parallel rank's contiguous block of heads of the
 * registered D128 32/8 profile: geometry {128,16,4} (the same group of four query heads per KV
 * head), q/out BF16 [128,16,T,B], query_k/query_v BF16 [128,4,T,B] and a D128/H4 cyclic context.
 * Every other contract is sliding_window_attention's. Each KV head's CTA computes exactly what
 * the 32/8 profile computes for that head, so rank r's output equals heads [16 r, 16 r + 16) of
 * the complete profile's output bit for bit.
 */
void sliding_window_attention_head_block(const Tensor& q, const Tensor& query_k,
                                         const Tensor& query_v, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& lanes,
                                         AttentionHeadGeometry geometry, std::uint32_t window,
                                         float scale, const CyclicKVCacheLayerView& context,
                                         SlidingWindowAttentionExecutionEnvelope envelope,
                                         WorkspaceArena& workspace, Tensor& out,
                                         cudaStream_t stream);

[[nodiscard]] std::size_t sliding_window_attention_head_block_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size);

} // namespace ninfer::ops
