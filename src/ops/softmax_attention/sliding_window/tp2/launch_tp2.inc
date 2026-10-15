// Translation-unit fragment of src/ops/softmax_attention/sliding_window/launch.cu, included once at
// its end (after its closing namespace): the two-device 16/4 head block launch, sharing that file's
// private token dispatch. The body is sliding_window_attention_launch's with the head counts as
// template arguments of upstream's kernels (their defaults are the 32/8 profile). Not a header.
// Our file; see docs/maintainer/upstream-merge.md.

#include "ops/softmax_attention/sliding_window/tp2/launch_tp2.h"

namespace ninfer::ops::detail {

void sliding_window_attention_head_block_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& lanes, float scale,
    const CyclicKVCacheLayerView& context, const SlidingWindowAttentionPlan& plan,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    constexpr int QHeads  = kSlidingWindowHeadBlockQHeads;
    constexpr int KVHeads = kSlidingWindowHeadBlockKVHeads;
    const bool direct     = plan.route == SlidingWindowAttentionRoute::Direct;
    if ((plan.window != 2048 && plan.window != 4096) ||
        context.capacity != static_cast<std::uint32_t>(plan.window) ||
        plan.key_block != 32 || plan.reduce_warps != (plan.window == 2048 ? 4 : 1) ||
        plan.split_capacity < 1 || plan.split_capacity > kSlidingWindowMaxSplits ||
        (direct && plan.split_capacity != 1))
        throw std::invalid_argument("sliding_window_attention head block: inconsistent plan");
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        constexpr int KeyBlock = 32;
        constexpr std::size_t SmemBytes =
            2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
        if (plan.warps != Warps)
            throw std::invalid_argument(
                "sliding_window_attention head block: inconsistent query layout");
        const auto partial = [&]<bool Direct>() {
            const dim3 grid(KVHeads, Direct ? 1 : plan.split_capacity, q.ne[3]);
            sliding_window_attention_split_partial_kernel<Tokens, Warps, KeyBlock, Direct, QHeads,
                                                          KVHeads>
                <<<grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __half*>(context.v.data),
                    static_cast<int>(context.padded_capacity), plan.window - 1, plan.max_context,
                    Direct ? 1 : plan.split_capacity, scale,
                    static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
                    static_cast<float*>(partial_l.data), static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (direct) {
            partial.template operator()<true>();
            return;
        }
        partial.template operator()<false>();
        const auto reduce = [&]<int ReduceWarps>() {
            constexpr int Rows = QHeads * Tokens;
            const dim3 grid((Rows + ReduceWarps - 1) / ReduceWarps, 1, q.ne[3]);
            sliding_window_attention_reduce_kernel<Tokens, KeyBlock, ReduceWarps, QHeads>
                <<<grid, ReduceWarps * 32, 0, stream>>>(
                    static_cast<const float*>(partial_acc.data),
                    static_cast<const float*>(partial_m.data),
                    static_cast<const float*>(partial_l.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data), plan.window - 1,
                    plan.max_context, plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (plan.reduce_warps == 4)
            reduce.template operator()<4>();
        else
            reduce.template operator()<1>();
    });
}

} // namespace ninfer::ops::detail
