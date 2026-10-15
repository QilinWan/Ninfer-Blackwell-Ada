#include "ops/softmax_attention/dense/causal_cache/e8/launch.h"

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// One small-T invocation over the whole request: the partial lanes come from the arena, and the
// fused append publishes the page table in the order the producer expects.
void small_t(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
             const Tensor& valid, const Tensor& rows, float scale, PagedKVBatchLayerView cache,
             CausalAttentionExecutionEnvelope envelope, std::int32_t splits,
             WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const auto partial        = allocate_causal_partials(workspace, q.ne[1], q.ne[2], splits,
                                                    q.ne[3]);
    causal_attention_small_t_e8_launch(q, k, v, positions, valid, rows, scale, cache, envelope, 0,
                                       q.ne[2], partial.acc, partial.m, partial.l, out, stream);
}

// Wide requests run as consecutive query chunks; each chunk opens its own partial scope, so the
// arena peak stays at one chunk and the reduction order stays the one the split count selects.
void chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                     const Tensor& valid, const Tensor& rows, float scale,
                     PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                     const E8KvCausalPlan& plan, WorkspaceArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += plan.chunk_tokens) {
        const std::int32_t count  = std::min(plan.chunk_tokens, q.ne[2] - begin);
        auto chunk_scope          = workspace.scope();
        const std::int32_t splits = e8_kv_split_capacity(q.ne[1], count, envelope, q.ne[3]);
        const auto partial = allocate_causal_partials(workspace, q.ne[1], count, splits, q.ne[3]);
        causal_attention_small_t_e8_launch(q, k, v, positions, valid, rows, scale, cache, envelope,
                                          begin, count, partial.acc, partial.m, partial.l, out,
                                          stream);
    }
}

void cached(const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
            CausalAttentionExecutionEnvelope envelope, std::int32_t splits,
            WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, q.ne[1], q.ne[2], splits, 1);
    causal_attention_cached_small_t_e8_launch(q, positions, scale, cache, envelope, partial.acc,
                                             partial.m, partial.l, out, stream);
}

void cached_chunked(const Tensor& q, const Tensor& positions, float scale,
                    const PagedKVLayerView& cache, CausalAttentionExecutionEnvelope envelope,
                    const E8KvCausalPlan& plan, WorkspaceArena& workspace, Tensor& out,
                    cudaStream_t stream) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += plan.chunk_tokens) {
        const std::int32_t count  = std::min(plan.chunk_tokens, q.ne[2] - begin);
        auto chunk_scope          = workspace.scope();
        const std::int32_t splits = e8_kv_split_capacity(q.ne[1], count, envelope, 1);
        const auto partial        = allocate_causal_partials(workspace, q.ne[1], count, splits, 1);
        const Tensor q_chunk      = q.slice(2, begin, count);
        const Tensor position_chunk = positions.slice(0, begin, count);
        Tensor out_chunk            = out.slice(2, begin, count);
        causal_attention_cached_small_t_e8_launch(q_chunk, position_chunk, scale, cache, envelope,
                                                 partial.acc, partial.m, partial.l, out_chunk,
                                                 stream);
    }
}

} // namespace

void e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_e8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope, execution.multiprocessor_count);
    switch (plan.family) {
    case E8KvFamily::Prompt:
        // The prompt form appends through the batched appender, then reads the pages it wrote.
        causal_attention_prompt_e8_launch(q, k, v, positions, valid, rows, scale, cache, out,
                                          stream);
        return;
    case E8KvFamily::ChunkedSmallT:
        chunked_small_t(q, k, v, positions, valid, rows, scale, cache, envelope, plan, workspace,
                        out, stream);
        return;
    case E8KvFamily::SmallT:
        small_t(q, k, v, positions, valid, rows, scale, cache, envelope, plan.splits, workspace,
                out, stream);
        return;
    }
}

void e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_e8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    switch (plan.family) {
    case E8KvFamily::Prompt:
        causal_attention_prompt_e8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    case E8KvFamily::ChunkedSmallT:
        cached_chunked(q, positions, scale, cache, envelope, plan, workspace, out, stream);
        return;
    case E8KvFamily::SmallT:
        cached(q, positions, scale, cache, envelope, plan.splits, workspace, out, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
