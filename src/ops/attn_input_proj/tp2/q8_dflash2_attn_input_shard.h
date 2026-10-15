#pragma once

// The two-device [3072,5120] shard of the DFlash2 drafter's Q8 query|key|value projection
// [6144,5120]: a rank's 16 query and 4 KV heads of D128. Its launchers are upstream's
// q8/q8_dflash2_attn_input.cu compiled with the shard's section output
// (q8_dflash2_attn_input_shard.cu); the dispatch takes the parent's DFlash2 routes. Our file.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kQ8DFlash2ShardQueryRows = 2048;
inline constexpr std::int32_t kQ8DFlash2ShardKvRows    = 512;
inline constexpr std::int32_t kQ8DFlash2ShardRows =
    kQ8DFlash2ShardQueryRows + 2 * kQ8DFlash2ShardKvRows;

void q8_dflash2_attn_input_shard_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void q8_dflash2_attn_input_shard_mma_r32_c64_launch(const Tensor& x, const Weight& weight,
                                                    Tensor& q, Tensor& k, Tensor& v,
                                                    cudaStream_t stream);
void q8_dflash2_attn_input_shard_mma_r64_c128_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& q, Tensor& k, Tensor& v,
                                                     cudaStream_t stream);
void q8_dflash2_attn_input_shard_mma_r16_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                         Tensor&, Tensor&, cudaStream_t);
void q8_dflash2_attn_input_shard_mma_r32_c32_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                         Tensor&, Tensor&, cudaStream_t);
void q8_dflash2_attn_input_shard_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                         Tensor&, Tensor&, cudaStream_t);

// The parent's DFlash2 route table (q8/q8_attn_input_plan.cpp, kDFlash2Routes) over the shard's
// launchers: small-T sliced-K up to 48 columns, then the row-tiled MMA schedules.
void q8_dflash2_attn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
