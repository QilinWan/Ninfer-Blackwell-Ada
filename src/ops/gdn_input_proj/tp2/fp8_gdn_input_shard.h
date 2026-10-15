#pragma once

// The FP8 [8192,5120] two-device shard of gdn_input_proj: one device's 8 of 16 key heads and 24 of
// 48 value heads of the [16384,5120] parent, in the parent's Q|K|V|Z order, so qkv takes
// 1024 + 1024 + 3072 rows and z 3072. Our file.
//
// The shard runs upstream's own FP8 gdn_input_proj sources, compiled a second time by the
// fp8_gdn_input_shard_*.cu/.cpp translation units of this directory with the section output and
// the entry points renamed (fp8_gdn_input_shard_names.h). Both share K, so every route, schedule
// and token cutoff is the parent's, by construction and after every upstream change; only the
// split-K partial reservation is the shard's own (fewer row tiles split earlier).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_input_shard_partial_capacity_bytes(std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_gdn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void fp8_gdn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, cudaStream_t stream);
void fp8_gdn_input_shard_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, cudaStream_t stream);
void fp8_gdn_input_shard_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   Fp8A8Workspace workspace, cudaStream_t stream);

// fp8_gdn_input_{a16,a8}_dispatch() and fp8_gdn_input_dispatch() at the shard shape.
void fp8_gdn_input_shard_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);
void fp8_gdn_input_shard_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     WorkspaceArena& workspace, cudaStream_t stream);
void fp8_gdn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  LinearPolicy policy, WorkspaceArena* workspace,
                                  cudaStream_t stream);

// Whether the snapshot and record forms quantize the activation for this policy and B/W block.
// Their A8 frontier differs from the bare projection's; the two-device shard follows it too
// (tp2/fp8_gdn_conv_plan_tp2.inc).
[[nodiscard]] bool fp8_gdn_conv_uses_a8(LinearPolicy policy, std::int32_t width,
                                        std::int32_t batch_size);

} // namespace ninfer::ops::detail
