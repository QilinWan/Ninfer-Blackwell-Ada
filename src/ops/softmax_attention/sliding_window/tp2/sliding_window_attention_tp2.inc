// Translation-unit fragment of src/ops/softmax_attention/sliding_window/sliding_window_attention.cpp,
// included once at its end (after its closing namespace): the two-device 16/4 head block of the
// Op, declared in ninfer/ops/tp2/sliding_window_attention.h, sharing the file's private checks.
// Not a header. Our file; see docs/maintainer/upstream-merge.md.

#include "ninfer/ops/tp2/sliding_window_attention.h"
#include "ops/softmax_attention/sliding_window/tp2/launch_tp2.h"

namespace ninfer::ops {
namespace {

constexpr std::int32_t kBlockQHeads  = detail::kSlidingWindowHeadBlockQHeads;
constexpr std::int32_t kBlockKVHeads = detail::kSlidingWindowHeadBlockKVHeads;

void require_block_profile(AttentionHeadGeometry geometry, std::uint32_t window, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kBlockQHeads || geometry.kv_heads != kBlockKVHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    if (!registered_window(window)) {
        throw std::invalid_argument(std::string(op) + ": supported windows are 2048 and 4096");
    }
}

void validate_block_context(const CyclicKVCacheLayerView& context, std::uint32_t window,
                            const char* op) {
    if (context.num_kv_heads != kBlockKVHeads || context.head_dim != kHeadDim ||
        context.capacity != window || context.padded_capacity < context.capacity ||
        context.lane_capacity <= 0) {
        throw std::invalid_argument(std::string(op) + ": invalid cyclic context");
    }
    if (context.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": padded capacity exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    if (context.k.dtype != DType::BF16 || context.v.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": context K/V must be BF16/FP16");
    }
    require_shape(context.k, kHeadDim, padded, kBlockKVHeads, context.lane_capacity, op,
                  "context k");
    require_shape(context.v, kHeadDim, padded, kBlockKVHeads, context.lane_capacity, op,
                  "context v");
    require_contiguous_nonnull(context.k, op, "context k");
    require_contiguous_nonnull(context.v, op, "context v");
}

template <class Allocator>
PartialWorkspace allocate_block_workspace(Allocator& workspace, std::int32_t tokens,
                                          std::int32_t splits, std::int32_t batch_size) {
    return {
        workspace.alloc(DType::FP32, {kHeadDim, kBlockQHeads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {kBlockQHeads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {kBlockQHeads, tokens, splits * batch_size}),
    };
}

} // namespace

std::size_t sliding_window_attention_head_block_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size) {
    require_block_profile(geometry, window, "sliding_window_attention head block workspace");
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > 8 || envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "sliding_window_attention head block workspace: invalid envelope or token interval");
    }
    std::size_t capacity = 0;
    for (int tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        const auto plan =
            detail::sliding_window_attention_resolve_plan(window, tokens, batch_size, envelope);
        if (plan.route == detail::SlidingWindowAttentionRoute::Direct) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_block_workspace(layout, tokens, plan.split_capacity, batch_size);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return capacity;
}

void sliding_window_attention_head_block(const Tensor& q, const Tensor& query_k,
                                         const Tensor& query_v, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& lanes,
                                         AttentionHeadGeometry geometry, std::uint32_t window,
                                         float scale, const CyclicKVCacheLayerView& context,
                                         SlidingWindowAttentionExecutionEnvelope envelope,
                                         WorkspaceArena& workspace, Tensor& out,
                                         cudaStream_t stream) {
    constexpr const char* op = "sliding_window_attention head block";
    require_block_profile(geometry, window, op);
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || positions.dtype != DType::I32 ||
        valid_columns.dtype != DType::I32 || lanes.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": invalid dtypes");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    if (tokens < 1 || tokens > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument(std::string(op) + ": T must be 1..16 and B 1..8");
    }
    require_shape(q, kHeadDim, kBlockQHeads, tokens, batch, op, "q");
    require_shape(query_k, kHeadDim, kBlockKVHeads, tokens, batch, op, "query k");
    require_shape(query_v, kHeadDim, kBlockKVHeads, tokens, batch, op, "query v");
    require_shape(positions, tokens, batch, 1, 1, op, "positions");
    require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns");
    require_shape(lanes, batch, 1, 1, 1, op, "lanes");
    require_shape(out, kHeadDim, kBlockQHeads, tokens, batch, op, "out");
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &q, &query_k, &query_v, &positions, &valid_columns, &lanes, &out}) {
        require_contiguous_nonnull(*t, op, "operand");
    }
    validate_block_context(context, window, op);
    if (envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(128)");
    }
    auto scope = workspace.scope();
    const auto plan =
        detail::sliding_window_attention_resolve_plan(window, tokens, batch, envelope);
    PartialWorkspace partial{};
    if (plan.route == detail::SlidingWindowAttentionRoute::SplitKv) {
        partial = allocate_block_workspace(workspace, tokens, plan.split_capacity, batch);
    }
    detail::sliding_window_attention_head_block_launch(q, query_k, query_v, positions,
                                                       valid_columns, lanes, scale, context, plan,
                                                       partial.acc, partial.m, partial.l, out,
                                                       stream);
}

} // namespace ninfer::ops
