#pragma once

// The FP8 [7168,5120] two-device shard of attn_input_proj: one device's head-local half of the
// [14336,5120] parent (12 of 24 query/gate heads, 2 of 4 key/value heads), rows
// query|key|gate|value = 3072|512|3072|512. Our file.
//
// The shard runs upstream's own FP8 attn_input_proj sources, compiled a second time by the
// fp8_attn_input_shard_*.cu/.cpp translation units of this directory with the section output
// and the entry points renamed (fp8_attn_input_shard_names.h). Both share K, so every route,
// schedule and token cutoff is the parent's, by construction and after every upstream change;
// only the split-K partial reservation is the shard's own (fewer row tiles split earlier).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_shard_partial_capacity_bytes(std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_attn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

void fp8_attn_input_shard_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& gate, Tensor& k, Tensor& v,
                                               cudaStream_t stream);
void fp8_attn_input_shard_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_shard_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                    cudaStream_t stream);

// fp8_attn_input_dispatch() at the shard shape.
void fp8_attn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, LinearPolicy policy,
                                   WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
