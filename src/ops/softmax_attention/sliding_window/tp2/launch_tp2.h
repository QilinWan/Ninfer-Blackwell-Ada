#pragma once

// Launch prototype of the two-device head block of sliding_window_attention
// (ninfer/ops/tp2/sliding_window_attention.h). Our file.

#include "ops/softmax_attention/sliding_window/launch.h"

namespace ninfer::ops::detail {

inline constexpr int kSlidingWindowHeadBlockQHeads  = 16;
inline constexpr int kSlidingWindowHeadBlockKVHeads = 4;

void sliding_window_attention_head_block_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& lanes, float scale,
    const CyclicKVCacheLayerView& context, const SlidingWindowAttentionPlan& plan,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
