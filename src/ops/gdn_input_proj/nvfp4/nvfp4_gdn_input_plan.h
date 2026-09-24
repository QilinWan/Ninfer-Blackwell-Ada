#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

// Every launcher and nvfp4_gdn_input_dispatch() serves the whole [16384,5120] parent and one
// device's [8192,5120] two-device shard (nvfp4_gdn_input_output.cuh), selected by `weight.n`. Both
// keep the parent's K, so their routes, route cutoffs and workspace capacity are the same.
void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
