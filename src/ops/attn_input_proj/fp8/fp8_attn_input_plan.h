#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens);
// The same for the problem with `rows` output rows: the [14336,5120] parent above, or the
// [7168,5120] two-device shard, whose half as many row tiles leave a split-K tail wave from the
// first bulk width on.
[[nodiscard]] std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t rows,
                                                                std::int32_t max_tokens);

inline constexpr int kFp8AttnInputLastSmallMmaT = 32;

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);
// The [7168,5120] two-device shard's capacity: the parent's routes and its own split-K partials.
[[nodiscard]] std::size_t fp8_attn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

// Every launcher and fp8_attn_input_dispatch() serve the whole [14336,5120] parent and one
// device's [7168,5120] two-device shard (fp8_attn_input_output.cuh), selected by `weight.n`.
// Both share K, so the routes and their token cutoffs are the same; only the split-K partials
// differ.
void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
