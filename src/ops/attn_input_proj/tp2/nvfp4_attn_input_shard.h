#pragma once

// The NVFP4 [7168,5120] two-device shard of attn_input_proj: one device's head-local half of the
// [14336,5120] parent (12 of 24 query/gate heads, 2 of 4 key/value heads), rows
// query|key|gate|value = 3072|512|3072|512. Our file.
//
// The shard runs upstream's own NVFP4 attn_input_proj sources, compiled a second time by the
// nvfp4_attn_input_shard_*.cu/.cpp translation units of this directory with the section output
// and the entry points renamed (nvfp4_attn_input_shard_names.h). Both share K, so every route,
// schedule, token cutoff and the workspace capacity are the parent's, by construction and after
// every upstream change. One measured exception (tune 2a596191): at T=1024 the shard's A4 TMA
// route keeps the 128-token scale tiles (nvfp4_attn_input_shard_a4.cu).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_attn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                          std::int32_t min_tokens,
                                                                          std::int32_t max_tokens);

void nvfp4_attn_input_shard_a16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void nvfp4_attn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v,
                                          cudaStream_t stream);
void nvfp4_attn_input_shard_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                           Tensor& gate, Tensor& k, Tensor& v,
                                           cudaStream_t stream);
void nvfp4_attn_input_shard_a4_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v,
                                      Nvfp4A4Workspace workspace, cudaStream_t stream);
// Upstream's A4 launcher at the shard shape, which nvfp4_attn_input_shard_a4_launch() wraps.
void nvfp4_attn_input_shard_a4_launch_upstream(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& gate, Tensor& k, Tensor& v,
                                               Nvfp4A4Workspace workspace, cudaStream_t stream);
// In the non-RDC archive, like upstream's launch_nvfp4_a4_tma_attention().
void launch_nvfp4_a4_tma_attention_shard(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                         __nv_bfloat16* gate, __nv_bfloat16* key,
                                         __nv_bfloat16* value, cudaStream_t stream);

// nvfp4_attn_input_dispatch() at the shard shape.
void nvfp4_attn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& q,
                                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
