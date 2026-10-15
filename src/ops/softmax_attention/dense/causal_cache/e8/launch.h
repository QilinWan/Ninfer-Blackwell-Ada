#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_geometry.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The rk4v4-e8 route follows the same per-storage contract as the five reorganized routes: the
// public op selects it by KV storage and hands it the execution envelope, the arena and the device
// view, and the route's own plan chooses between small-T, chunked small-T and prompt.
void e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution);

void e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution);

// A small-T invocation describes one query window over the paged KV: which table rows to read, how
// wide the full request is, and which column range of it this invocation covers.
struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

// Kernel entries the route drives. The small-T form fuses the KV append into the partial producer,
// so the order in which it publishes the page table is part of the kernel contract; the prompt form
// appends through the batched appender before reading.
void causal_attention_small_t_e8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_e8_launch(const Tensor& q, const Tensor& pos, float scale,
                                               const PagedKVLayerView& cache,
                                               CausalAttentionExecutionEnvelope envelope,
                                               Tensor& partial_acc, Tensor& partial_m,
                                               Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_prompt_e8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& positions, const Tensor& valid_columns,
                                       const Tensor& table_rows, float scale,
                                       PagedKVBatchLayerView cache, Tensor& out,
                                       cudaStream_t stream);

void causal_attention_prompt_e8_attention_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
