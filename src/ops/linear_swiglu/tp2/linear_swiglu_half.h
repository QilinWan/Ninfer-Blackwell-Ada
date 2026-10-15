#pragma once

// The two-device output-row half [17408,5120] of the FP8 and NVFP4 [34816,5120] gate/up problem:
// one device's block of the gate rows over its matching block of the up rows, out [8704,T]. Our
// file.
//
// The half runs upstream's own linear_swiglu launchers and plans, compiled a second time by the
// *_linear_swiglu_half_*.cu/.cpp translation units of this directory with the problem geometry
// and the entry points renamed (*_linear_swiglu_half_names.h). Both share K, so every route,
// schedule and token cutoff is the parent's, by construction and after every upstream change.
// Upstream's A8/A4 GEMM, TMA and decode launchers take the row count from the weight and serve the
// half unchanged; the wrapper (src/ops/wrapper/linear_swiglu.cpp) sends the half to the dispatchers
// below.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLinearSwiGluHalfRows = 17408;

void fp8_linear_swiglu_half_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void fp8_linear_swiglu_half_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream);
void fp8_linear_swiglu_half_matrix_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] std::size_t fp8_linear_swiglu_half_workspace_capacity_bytes(LinearPolicy policy,
                                                                          std::int32_t min_tokens,
                                                                          std::int32_t max_tokens);
// fp8_linear_swiglu_dispatch() at the half shape.
void fp8_linear_swiglu_half_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                     LinearPolicy policy, WorkspaceArena* workspace,
                                     cudaStream_t stream);

void nvfp4_linear_swiglu_half_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t
nvfp4_linear_swiglu_half_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens);
// nvfp4_linear_swiglu_dispatch() at the half shape.
void nvfp4_linear_swiglu_half_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                       LinearPolicy policy, WorkspaceArena* workspace,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
